"""Task B deploy: trained student nav (RNN+CNN) + frozen squat LL policy.

Not the four-phase SEARCH/LOCK/DETECT/GRASP stack in demo/solution.py — this mirrors
``scripts/train_nav_taskb_student.py`` (head+ee rgb+depth, detect arm hold, vel_cmd nav).

Platform bundle (solution/ directory):
  solution.py              <- copy this file
  agent.yaml               <- copy agent_taskb_student.yaml
  model_taskb_student.pt   <- your rsl_rl checkpoint (model_*.pt)
  policy.pt                <- squat LL JIT (e.g. down_policy.pt)
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_demo_dir = os.path.dirname(os.path.abspath(__file__))
if _demo_dir not in sys.path:
    sys.path.insert(0, _demo_dir)
from depth_preprocess import prep_depth  # noqa: E402

# Same detect-hold arm env actions as taskb_student_env / squat_env_cfg training.
_TASK_B_LOW_CLEAR_ARM_ACTION = (0.0, 5.4, -3.6, 0.0, -1.2, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Inlined TaskDStudentActorCritic (deploy has no rsl_rl / repo deps)
# ---------------------------------------------------------------------------


class EmpiricalNormalization(nn.Module):
    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class Memory(nn.Module):
    def __init__(self, input_size, type="gru", num_layers=1, hidden_size=256):
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        if masks is not None:
            if hidden_states is None:
                raise ValueError("Hidden states required in batch mode")
            out, _ = self.rnn(input, hidden_states)
            return out
        out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None, hidden_states=None):
        if dones is None:
            self.hidden_states = hidden_states
        elif self.hidden_states is not None:
            if isinstance(self.hidden_states, tuple):
                for hs in self.hidden_states:
                    hs[..., dones == 1, :] = 0.0
            else:
                self.hidden_states[..., dones == 1, :] = 0.0


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str = "elu"):
        super().__init__()
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), act]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


# Task B critic priv: pose(4)+ee(3)+r_vel(2)+contact(1)+min_xy(1)+min_3d(1)+scored(18)+trash(54)
_TASK_B_CRITIC_EXTRA_DIM = 84


class TaskBStudentActorCritic(nn.Module):
    """Matches atec_rl_lab TaskDStudentActorCritic / TaskBStudentActorCritic (nav head only at deploy)."""

    is_recurrent = True

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        *,
        img_h: int = 48,
        img_w: int = 64,
        img_hw: int | None = None,
        img_channels: int = 4,
        proprio_dim: int = 9,
        enc_dim: int = 128,
        fuse_dim: int = 256,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dims: list | None = None,
        critic_hidden_dims: list | None = None,
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        nav_action_dim: int = 2,
        leg_action_dim: int = 12,
        **_kwargs,
    ):
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128]
        if img_hw is not None:
            img_h = img_w = int(img_hw)

        self.obs_groups = obs_groups
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.img_channels = int(img_channels)
        self.head_flat = self.img_channels * self.img_h * self.img_w
        self.ee_flat = self.img_channels * self.img_h * self.img_w
        self.proprio_dim = int(proprio_dim)
        self.nav_action_dim = int(nav_action_dim)
        self.leg_action_dim = int(leg_action_dim)

        self.head_encoder = ConvEncoder(in_ch=self.img_channels, out_dim=enc_dim)
        self.ee_encoder = ConvEncoder(in_ch=self.img_channels, out_dim=enc_dim)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(self.proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(enc_dim + enc_dim + 64, fuse_dim),
            nn.ReLU(inplace=True),
        )
        self._base_obs_dim = self.head_flat + self.ee_flat + self.proprio_dim
        raw_critic_dim = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self._critic_priv_dim = max(0, int(raw_critic_dim - self._base_obs_dim))
        self.critic_priv_mlp = (
            nn.Sequential(
                nn.Linear(self._critic_priv_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
            )
            if self._critic_priv_dim > 0
            else nn.Identity()
        )

        self.memory_a = Memory(fuse_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        critic_in = fuse_dim + (64 if self._critic_priv_dim > 0 else 0)
        self.memory_c = Memory(critic_in, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.nav_actor = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.nav_action_dim),
        )
        self.pit_actor = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.leg_action_dim),
        )
        self.actor = self.nav_actor
        self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, "elu")
        self.actor_obs_normalizer = (
            EmpiricalNormalization(fuse_dim) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_in) if critic_obs_normalization else nn.Identity()
        )
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def _get_flat_obs(self, obs: dict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[g] for g in groups], dim=-1)

    def _encode_base(self, flat_obs: torch.Tensor) -> torch.Tensor:
        lead_shape = flat_obs.shape[:-1]
        x = flat_obs.reshape(-1, flat_obs.shape[-1])
        head = x[:, : self.head_flat].view(-1, self.img_channels, self.img_h, self.img_w)
        ee = x[:, self.head_flat : self.head_flat + self.ee_flat].view(
            -1, self.img_channels, self.img_h, self.img_w
        )
        proprio = x[:, self.head_flat + self.ee_flat : self.head_flat + self.ee_flat + self.proprio_dim]
        out = self.fuse(torch.cat([self.head_encoder(head), self.ee_encoder(ee), self.proprio_mlp(proprio)], dim=-1))
        return out.view(*lead_shape, -1)

    def act_inference(self, obs: dict) -> torch.Tensor:
        encoded = self._encode_base(self._get_flat_obs(obs, self.obs_groups["policy"]))
        encoded = self.actor_obs_normalizer(encoded)
        out_mem = self.memory_a(encoded).squeeze(0)
        return self.nav_actor(out_mem)


def _load_deploy_cfg(demo_dir: str) -> dict:
    for name in ("agent_taskb_student.yaml", "agent.yaml"):
        agent_yaml = os.path.join(demo_dir, name)
        if not os.path.isfile(agent_yaml):
            continue
        try:
            import yaml

            with open(agent_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("policy", {})
        except Exception:
            continue
    return {}


def _resolve_path(demo_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    cand = os.path.join(demo_dir, path)
    if os.path.isfile(cand):
        return cand
    return path


class AlgSolution:
    """Task B student nav: head+ee rgb+depth + proprio -> vel_cmd @10Hz, squat LL @50Hz."""

    def __init__(self):
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        policy_cfg = _load_deploy_cfg(demo_dir)

        legacy_hw = policy_cfg.get("img_hw")
        if legacy_hw is not None and "img_h" not in policy_cfg:
            self.image_h = int(legacy_hw)
            self.image_w = int(legacy_hw)
        else:
            self.image_h = int(policy_cfg.get("img_h", 48))
            self.image_w = int(policy_cfg.get("img_w", 64))
        self.img_channels = int(policy_cfg.get("img_channels", 4))
        self.depth_only = self.img_channels == 1
        self.depth_max = float(policy_cfg.get("depth_max", os.environ.get("NAV_DEPTH_MAX", "5.0")))

        self.vx_min = float(os.environ.get("NAV_VX_MIN", policy_cfg.get("vx_min", -1.0)))
        self.vx_max = float(os.environ.get("NAV_VX_MAX", policy_cfg.get("vx_max", 1.0)))
        self.vy_max = float(os.environ.get("NAV_VY_MAX", policy_cfg.get("vy_max", 0.0)))
        self.wz_max = float(os.environ.get("NAV_WZ_MAX", policy_cfg.get("wz_max", 1.0)))

        inner_steps = int(os.environ.get("NAV_INNER_STEPS", policy_cfg.get("inner_steps", 5)))
        self.nav_hold_steps = max(1, inner_steps)

        student_ckpt = os.environ.get(
            "ATEC_TASKB_STUDENT_CKPT",
            policy_cfg.get("student_ckpt", "model_taskb_student.pt"),
        )
        student_ckpt_path = _resolve_path(demo_dir, student_ckpt)
        ll_name = os.environ.get("ATEC_LL_POLICY", policy_cfg.get("ll_policy", "policy.pt"))
        ll_policy_path = _resolve_path(demo_dir, ll_name)

        actor_dim = 2 * self.img_channels * self.image_h * self.image_w + 9
        critic_dim = actor_dim + _TASK_B_CRITIC_EXTRA_DIM
        obs = {"policy": torch.zeros(1, actor_dim), "critic": torch.zeros(1, critic_dim)}
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        ac_kwargs = {
            "img_h": self.image_h,
            "img_w": self.image_w,
            "img_channels": self.img_channels,
            "proprio_dim": int(policy_cfg.get("proprio_dim", 9)),
            "enc_dim": int(policy_cfg.get("enc_dim", 128)),
            "fuse_dim": int(policy_cfg.get("fuse_dim", 256)),
            "rnn_type": policy_cfg.get("rnn_type", "gru"),
            "rnn_hidden_dim": int(policy_cfg.get("rnn_hidden_dim", 256)),
            "rnn_num_layers": int(policy_cfg.get("rnn_num_layers", 1)),
            "actor_obs_normalization": bool(policy_cfg.get("actor_obs_normalization", True)),
            "critic_obs_normalization": bool(policy_cfg.get("critic_obs_normalization", True)),
            "actor_hidden_dims": policy_cfg.get("actor_hidden_dims", [256]),
            "critic_hidden_dims": policy_cfg.get("critic_hidden_dims", [256, 128]),
            "init_noise_std": float(policy_cfg.get("init_noise_std", 0.6)),
            "noise_std_type": policy_cfg.get("noise_std_type", "scalar"),
        }
        self.nav_policy = TaskBStudentActorCritic(obs, obs_groups, num_actions=2, **ac_kwargs).to(self.device)
        loaded = torch.load(student_ckpt_path, map_location=self.device, weights_only=False)
        state = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        self.nav_policy.load_state_dict(state, strict=True)
        self.nav_policy.eval()
        print(
            f"[TaskBStudentNav] img={self.img_channels}ch@{self.image_h}x{self.image_w} "
            f"depth_only={self.depth_only} nav_hold={self.nav_hold_steps} "
            f"vx=[{self.vx_min:.1f},{self.vx_max:.1f}] vy=0 wz=±{self.wz_max:.1f} "
            f"ckpt={os.path.basename(student_ckpt_path)} ll={os.path.basename(ll_policy_path)}",
            flush=True,
        )

        self.ll_policy = torch.jit.load(ll_policy_path, map_location=self.device)
        self.ll_policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.train_to_env_action_scale = torch.tensor([0.25, 0.5, 0.5] * 4, device=self.device).view(1, -1)
        self.env_to_train_action_scale = torch.tensor([4.0, 2.0, 2.0] * 4, device=self.device).view(1, -1)
        self.arm_hold_action = torch.tensor(
            _TASK_B_LOW_CLEAR_ARM_ACTION, device=self.device, dtype=torch.float32
        ).view(1, -1)

        self._nav_step_counter = 0
        self._cached_vel_cmd: torch.Tensor | None = None
        self._last_vel_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

    def set_device(self, device: str) -> None:
        self.device = device
        self.nav_policy = self.nav_policy.to(device)
        self.ll_policy = self.ll_policy.to(device)
        self.train_to_env_action_scale = self.train_to_env_action_scale.to(device)
        self.env_to_train_action_scale = self.env_to_train_action_scale.to(device)
        self.arm_hold_action = self.arm_hold_action.to(device)
        self.reset()

    def bind_env(self, env) -> None:
        del env

    def reset(self, **kwargs):
        del kwargs
        self.nav_policy.reset()
        self._nav_step_counter = 0
        self._cached_vel_cmd = None
        self._last_vel_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

    @staticmethod
    def _to_batch_tensor(x, device: torch.device) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x.to(device=device, dtype=torch.float32 if x.is_floating_point() else x.dtype)
        else:
            import numpy as np

            arr = np.asarray(x)
            dtype = torch.float32 if np.issubdtype(arr.dtype, np.floating) else None
            t = torch.as_tensor(arr, device=device, dtype=dtype)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.ndim == 4 and t.shape[0] != 1:
            t = t[:1]
        return t.float() if t.is_floating_point() else t

    def _prep_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.ndim == 4 and x.shape[-1] in (3, 4):
            x = x[..., :3].permute(0, 3, 1, 2).contiguous()
        elif x.ndim == 3 and x.shape[-1] in (3, 4):
            x = x[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()
        if x.shape[-2] != self.image_h or x.shape[-1] != self.image_w:
            x = F.interpolate(
                x, size=(self.image_h, self.image_w), mode="bilinear", align_corners=False
            )
        return x

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        return prep_depth(
            x.to(self.device),
            image_h=self.image_h,
            image_w=self.image_w,
            depth_max=self.depth_max,
        )

    def _cam_flat(self, image_obs: dict, rgb_key: str, depth_key: str, alt_rgb: str | None, batch: int) -> torch.Tensor:
        depth = image_obs.get(depth_key)
        if depth is None and depth_key == "head_depth":
            depth = image_obs.get("video_depth")
        if depth is None:
            keys = sorted(image_obs.keys()) if isinstance(image_obs, dict) else []
            raise RuntimeError(
                f"Missing depth key '{depth_key}' in obs['image'] (have {keys}). "
                "Task B student nav play needs head/ee cameras: "
                "python scripts/play_atec_task.py ... --cameras-only (not --fast)."
            )

        if self.depth_only:
            flat = self._prep_depth(self._to_batch_tensor(depth, self.device)).reshape(batch, -1)
            return flat

        rgb = image_obs.get(rgb_key)
        if rgb is None and alt_rgb is not None:
            rgb = image_obs.get(alt_rgb)
        if rgb is None:
            raise RuntimeError(
                f"rgb+depth student requires '{rgb_key}' (or fallback) and '{depth_key}' in obs['image']."
            )
        rgb_t = self._prep_rgb(self._to_batch_tensor(rgb, self.device))
        depth_t = self._prep_depth(self._to_batch_tensor(depth, self.device))
        return torch.cat([rgb_t, depth_t], dim=1).reshape(batch, -1)

    def _build_actor_obs(self, obs: dict) -> torch.Tensor:
        proprio = self._to_batch_tensor(obs["proprio"], self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        batch = proprio.shape[0]
        proprio_feat = torch.cat([proprio[:, 0:3], proprio[:, 3:6], proprio[:, 9:12]], dim=-1)

        image_obs = obs.get("image", {})
        if not isinstance(image_obs, dict):
            raise RuntimeError("obs['image'] must be a dict with head/ee rgb+depth.")

        head = self._cam_flat(image_obs, "head_rgb", "head_depth", "video_rgb", batch)
        ee = self._cam_flat(image_obs, "ee_rgb", "ee_depth", None, batch)
        return torch.cat([head, ee, proprio_feat], dim=-1)

    def _nav_action_to_vel_cmd(self, nav_action: torch.Tensor) -> torch.Tensor:
        a = nav_action.clamp(-1.0, 1.0)
        vx = (a[:, 0] + 1.0) * 0.5 * (self.vx_max - self.vx_min) + self.vx_min
        vy = torch.zeros_like(vx)
        wz = a[:, 1] * self.wz_max
        return torch.stack([vx, vy, wz], dim=-1)

    def _predict_velocity_command(self, obs: dict) -> torch.Tensor:
        refresh = self._cached_vel_cmd is None or (self._nav_step_counter % self.nav_hold_steps == 0)
        if refresh:
            actor_obs = self._build_actor_obs(obs)
            with torch.inference_mode():
                nav_action = self.nav_policy.act_inference({"policy": actor_obs})
            self._cached_vel_cmd = self._nav_action_to_vel_cmd(nav_action)
        self._nav_step_counter += 1
        cmd = self._cached_vel_cmd.to(self.device)
        self._last_vel_cmd = cmd.detach().clone()
        return cmd

    def _extract_ll_obs(self, obs: dict, action_dim: int, velocity_commands: torch.Tensor) -> torch.Tensor:
        proprio = self._to_batch_tensor(obs["proprio"], self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        idx = 3
        base_ang_vel = proprio[:, idx : idx + 3]
        idx += 6
        projected_gravity = proprio[:, idx : idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx : idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_train_leg = actions_all[:, self.leg_joint_indices] * self.env_to_train_action_scale.to(
            dtype=proprio.dtype
        )

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )
        num_envs = action_train.shape[0]
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = action_train * self.train_to_env_action_scale
        action_env[:, self.arm_joint_indices] = self.arm_hold_action.expand(num_envs, -1)
        return action_env

    def predicts(self, obs, current_score):
        del current_score
        proprio = self._to_batch_tensor(obs["proprio"], self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        velocity_commands = self._predict_velocity_command(obs)
        ll_obs = self._extract_ll_obs(obs, action_dim, velocity_commands)
        with torch.inference_mode():
            action_train = self.ll_policy(ll_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train.to(self.device, dtype=torch.float32), action_dim)
        return {"action": action_env.detach().cpu().numpy().tolist(), "giveup": False}

    def get_video_overlay_lines(self) -> list[str]:
        cmd = self._last_vel_cmd[0]
        return [
            "taskb_student_nav",
            f"hl_cmd=({float(cmd[0]):+.2f},{float(cmd[1]):+.2f},{float(cmd[2]):+.2f})",
            f"img={self.img_channels}ch@{self.image_h}x{self.image_w}",
            f"nav_hold={self.nav_hold_steps}",
        ]
