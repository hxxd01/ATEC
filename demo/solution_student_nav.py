"""Task D deploy: self-contained student PPO + JIT locomotion (upload solution/ only)."""

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


# ---------------------------------------------------------------------------
# Inlined from taskd_student_actor_critic.py + rsl_rl (no repo / rsl_rl deps)
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


class TaskDStudentActorCritic(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        *,
        img_hw: int = 64,
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
        **_kwargs,
    ):
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128]

        self.obs_groups = obs_groups
        self.img_hw = int(img_hw)
        self.img_channels = int(img_channels)
        self.head_flat = self.img_channels * self.img_hw * self.img_hw
        self.ee_flat = self.img_channels * self.img_hw * self.img_hw
        self.proprio_dim = int(proprio_dim)

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
        self.actor = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )
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
        head = x[:, : self.head_flat].view(-1, self.img_channels, self.img_hw, self.img_hw)
        ee = x[:, self.head_flat : self.head_flat + self.ee_flat].view(
            -1, self.img_channels, self.img_hw, self.img_hw
        )
        proprio = x[:, self.head_flat + self.ee_flat : self.head_flat + self.ee_flat + self.proprio_dim]
        out = self.fuse(torch.cat([self.head_encoder(head), self.ee_encoder(ee), self.proprio_mlp(proprio)], dim=-1))
        return out.view(*lead_shape, -1)

    def _encode_actor(self, flat_obs: torch.Tensor) -> torch.Tensor:
        return self._encode_base(flat_obs)

    def act_inference(self, obs: dict) -> torch.Tensor:
        encoded = self._encode_actor(self._get_flat_obs(obs, self.obs_groups["policy"]))
        encoded = self.actor_obs_normalizer(encoded)
        out_mem = self.memory_a(encoded).squeeze(0)
        return self.actor(out_mem)


def _load_deploy_cfg(demo_dir: str) -> dict:
    agent_yaml = os.path.join(demo_dir, "agent.yaml")
    if not os.path.isfile(agent_yaml):
        return {}
    try:
        import yaml

        with open(agent_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("policy", {})
    except Exception:
        return {}


class AlgSolution:
    """Task D student deploy: depth cameras + proprio -> nav cmd @10Hz, JIT legs @50Hz."""

    NUM_STAGES = 4
    CRITIC_EXTRA_DIM = 17 + NUM_STAGES

    def __init__(self):
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        student_ckpt_path = demo_dir + "/model_1100.pt"
        policy_cfg = _load_deploy_cfg(demo_dir)
        self.image_hw = int(policy_cfg.get("img_hw", 24))
        self.depth_render_h = int(policy_cfg.get("depth_render_h", self.image_hw))
        self.depth_render_w = int(policy_cfg.get("depth_render_w", self.image_hw))
        self.img_channels = int(policy_cfg.get("img_channels", 1))
        self.depth_only = self.img_channels == 1
        self.depth_max = float(policy_cfg.get("depth_max", os.environ.get("NAV_DEPTH_MAX", "5.0")))

        self.vx_min = float(os.environ.get("NAV_VX_MIN", "-2.0"))
        self.vx_max = float(os.environ.get("NAV_VX_MAX", "2.0"))
        self.vy_max = float(os.environ.get("NAV_VY_MAX", "1.2"))
        self.wz_max = float(os.environ.get("NAV_WZ_MAX", "0.6"))

        inner_steps = int(os.environ.get("NAV_INNER_STEPS", "5"))
        self.nav_hold_steps = max(1, inner_steps)

        actor_dim = 2 * self.img_channels * self.image_hw * self.image_hw + 9
        critic_dim = actor_dim + self.CRITIC_EXTRA_DIM
        obs = {
            "policy": torch.zeros(1, actor_dim),
            "critic": torch.zeros(1, critic_dim),
        }
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        ac_kwargs = {
            "img_hw": self.image_hw,
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
        self.nav_policy = TaskDStudentActorCritic(obs, obs_groups, num_actions=3, **ac_kwargs).to(self.device)
        loaded = torch.load(student_ckpt_path, map_location=self.device, weights_only=False)
        state = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        self.nav_policy.load_state_dict(state, strict=True)
        self.nav_policy.eval()
        print(
            f"[AlgSolution] depth: platform 480x640 -> bilinear {self.depth_render_h}x{self.depth_render_w} "
            f"-> log1p -> {self.image_hw}x{self.image_hw} (train=deploy), "
            f"ckpt={os.path.basename(student_ckpt_path)}",
            flush=True,
        )

        policy_path = demo_dir + "/policy.pt"
        self.ll_policy = torch.jit.load(policy_path, map_location=self.device)
        self.ll_policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.train_to_env_action_scale = torch.tensor([0.25, 0.5, 0.5] * 4, device=self.device).view(1, -1)
        self.env_to_train_action_scale = torch.tensor([4.0, 2.0, 2.0] * 4, device=self.device).view(1, -1)
        self.arm_default_action = torch.zeros((1, self.arm_action_dim), device=self.device, dtype=torch.float32)

        self._nav_step_counter = 0
        self._cached_vel_cmd: torch.Tensor | None = None
        self._last_vel_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

    def set_device(self, device: str) -> None:
        self.device = device
        self.nav_policy = self.nav_policy.to(device)
        self.ll_policy = self.ll_policy.to(device)
        self.train_to_env_action_scale = self.train_to_env_action_scale.to(device)
        self.env_to_train_action_scale = self.env_to_train_action_scale.to(device)
        self.arm_default_action = self.arm_default_action.to(device)
        self.reset()

    def bind_env(self, env) -> None:
        del env

    def reset(self, **kwargs):
        del kwargs
        self.nav_policy.reset()
        self._nav_step_counter = 0
        self._cached_vel_cmd = None
        self._last_vel_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        return prep_depth(
            x.to(self.device),
            image_hw=self.image_hw,
            depth_render_h=self.depth_render_h,
            depth_render_w=self.depth_render_w,
            depth_max=self.depth_max,
        )

    def _build_actor_obs(self, obs: dict) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio_feat = torch.cat([proprio[:, 0:3], proprio[:, 3:6], proprio[:, 9:12]], dim=-1)

        image_obs = obs.get("image", {})
        if not isinstance(image_obs, dict):
            raise RuntimeError("obs['image'] is missing or not a dict; head/ee cameras are required.")
        head_depth = image_obs.get("head_depth")
        if head_depth is None:
            head_depth = image_obs.get("video_depth")
        ee_depth = image_obs.get("ee_depth")
        if head_depth is None or ee_depth is None:
            raise RuntimeError("depth_only student requires head_depth and ee_depth in obs['image'].")

        head = self._prep_depth(head_depth).reshape(proprio.shape[0], -1)
        ee = self._prep_depth(ee_depth).reshape(proprio.shape[0], -1)
        return torch.cat([head, ee, proprio_feat], dim=-1)

    def _nav_action_to_vel_cmd(self, nav_action: torch.Tensor) -> torch.Tensor:
        a = nav_action.clamp(-1.0, 1.0)
        vx = (a[:, 0] + 1.0) * 0.5 * (self.vx_max - self.vx_min) + self.vx_min
        return torch.stack([vx, a[:, 1] * self.vy_max, a[:, 2] * self.wz_max], dim=-1)

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
        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
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
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def predicts(self, obs, current_score):
        del current_score
        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
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
            "student_ckpt=model_1100.pt",
            f"hl_cmd=({float(cmd[0]):+.2f},{float(cmd[1]):+.2f},{float(cmd[2]):+.2f})",
            f"img={self.img_channels}ch@{self.image_hw} depth_only={self.depth_only}",
            f"nav_hold={self.nav_hold_steps} steps",
        ]
