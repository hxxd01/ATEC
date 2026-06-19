"""Task D MARG depth pit student deploy: depth + MARG proprio/history -> 12 leg actions."""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn

_demo_dir = os.path.dirname(os.path.abspath(__file__))
if _demo_dir not in sys.path:
    sys.path.insert(0, _demo_dir)
from depth_preprocess import prep_depth  # noqa: E402
from teleop_controller import TaskDTeleopController  # noqa: E402
from teleop_traj import TrajectoryReplayer, load_teleop_trajectory  # noqa: E402

MARG_PROPRIO_DIM = 43
MARG_HISTORY_LEN = 6
MARG_HISTORY_DIM = MARG_PROPRIO_DIM * MARG_HISTORY_LEN
MARG_ELEVATION_OUT_DIM = 16
MARG_ESTIMATOR_OUT_DIM = 7
MARG_CRITIC_PRIV_DIM = 42


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


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str = "relu"):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.ELU()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), act]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, out_dim: int = 128):
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


class MargDepthPitActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        *,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list | None = None,
        critic_hidden_dims: list | None = None,
        activation: str = "relu",
        init_noise_std: float = 0.6,
        max_noise_std: float = 2.0,
        noise_std_type: str = "scalar",
        img_h: int = 24,
        img_w: int = 32,
        depth_channels: int = 1,
        enc_dim: int = 128,
        estimator_hidden_dims: list | None = None,
        depth_hidden_dims: list | None = None,
        elevation_out_dim: int = MARG_ELEVATION_OUT_DIM,
        **_kwargs,
    ):
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        if estimator_hidden_dims is None:
            estimator_hidden_dims = [128]
        if depth_hidden_dims is None:
            depth_hidden_dims = [128, 64]

        self.obs_groups = obs_groups
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.depth_channels = int(depth_channels)
        self.depth_flat_per_cam = self.depth_channels * self.img_h * self.img_w

        self.estimator = MLP(MARG_HISTORY_DIM, MARG_ESTIMATOR_OUT_DIM, estimator_hidden_dims, activation)
        self.head_depth_encoder = ConvEncoder(in_ch=self.depth_channels, out_dim=enc_dim)
        self.ee_depth_encoder = ConvEncoder(in_ch=self.depth_channels, out_dim=enc_dim)
        self.depth_fusion = MLP(enc_dim + enc_dim, elevation_out_dim, depth_hidden_dims, activation)

        actor_in = MARG_PROPRIO_DIM + MARG_ESTIMATOR_OUT_DIM + elevation_out_dim
        critic_in = MARG_PROPRIO_DIM + MARG_CRITIC_PRIV_DIM + elevation_out_dim
        self.actor = MLP(actor_in, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_in, 1, critic_hidden_dims, activation)
        self.actor_obs_normalizer = (
            EmpiricalNormalization(actor_in) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_in) if critic_obs_normalization else nn.Identity()
        )
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.max_noise_std = float(max_noise_std)

    def _split_depth(self, depth_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.depth_flat_per_cam
        batch = depth_flat.shape[0]
        head = depth_flat[:, :half].reshape(batch, self.depth_channels, self.img_h, self.img_w)
        ee = depth_flat[:, half:].reshape(batch, self.depth_channels, self.img_h, self.img_w)
        return head, ee

    def _encode_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        head, ee = self._split_depth(depth_flat)
        feat = torch.cat([self.head_depth_encoder(head), self.ee_depth_encoder(ee)], dim=-1)
        return self.depth_fusion(feat)

    def _encode_actor(self, obs: dict) -> torch.Tensor:
        est_out = self.estimator(obs["proprio_history"])
        depth_out = self._encode_depth(obs["depth"])
        return torch.cat([obs["proprio"], est_out, depth_out], dim=-1)

    def act_inference(self, obs: dict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self._encode_actor(obs))
        return self.actor(actor_obs)


def _load_policy_cfg(demo_dir: str) -> dict:
    agent_yaml = os.path.join(demo_dir, "agent_marg_depth_pit.yaml")
    if not os.path.isfile(agent_yaml):
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
    """Task D: optional teleop traj replay (policy.pt) → MARG depth pit student."""

    LEG_ACTION_DIM = 12
    ARM_ACTION_DIM = 8
    SIM_DT = 0.02

    def __init__(self):
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = self.SIM_DT
        self._phase = "pit"
        self._sim_step = 0
        self._teleop_done = True
        self._replayer: TrajectoryReplayer | None = None
        self._teleop: TaskDTeleopController | None = None
        self._last_teleop_cmd = (0.0, 0.0, 0.0)

        ckpt_env = os.environ.get("PIT_STUDENT_CKPT", "").strip()
        default_ckpt = os.path.join(demo_dir, "model_800.pt")
        if ckpt_env:
            student_ckpt_path = ckpt_env
        elif os.path.isfile(default_ckpt):
            student_ckpt_path = default_ckpt
        else:
            raise FileNotFoundError(
                f"PIT student checkpoint not found. Place model_800.pt next to solution.py "
                f"(looked in {demo_dir}) or set PIT_STUDENT_CKPT."
            )
        if not os.path.isfile(student_ckpt_path):
            raise FileNotFoundError(f"PIT student checkpoint not found: {student_ckpt_path}")

        policy_cfg = _load_policy_cfg(demo_dir)
        self.image_h = int(policy_cfg.get("img_h", 24))
        self.image_w = int(policy_cfg.get("img_w", 32))
        self.img_channels = int(policy_cfg.get("depth_channels", 1))
        self.depth_only = self.img_channels == 1
        self.depth_max = float(policy_cfg.get("depth_max", os.environ.get("PIT_DEPTH_MAX", "5.0")))
        self.depth_render_h = int(policy_cfg.get("depth_render_h", self.image_h))
        self.depth_render_w = int(policy_cfg.get("depth_render_w", self.image_w))
        self.command_vx = float(os.environ.get("PIT_COMMAND_VX", "0.6"))

        depth_dim = 2 * self.img_channels * self.image_h * self.image_w
        obs = {
            "proprio": torch.zeros(1, MARG_PROPRIO_DIM),
            "proprio_history": torch.zeros(1, MARG_HISTORY_DIM),
            "depth": torch.zeros(1, depth_dim),
        }
        obs_groups = {
            "policy": ["proprio", "proprio_history", "depth"],
            "critic": ["proprio", "depth", "critic_priv"],
        }
        ac_kwargs = {
            "img_h": self.image_h,
            "img_w": self.image_w,
            "depth_channels": self.img_channels,
            "enc_dim": int(policy_cfg.get("enc_dim", 128)),
            "elevation_out_dim": int(policy_cfg.get("elevation_out_dim", MARG_ELEVATION_OUT_DIM)),
            "estimator_hidden_dims": policy_cfg.get("estimator_hidden_dims", [128]),
            "depth_hidden_dims": policy_cfg.get("depth_hidden_dims", [128, 64]),
            "actor_obs_normalization": bool(policy_cfg.get("actor_obs_normalization", True)),
            "critic_obs_normalization": bool(policy_cfg.get("critic_obs_normalization", False)),
            "actor_hidden_dims": policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
            "critic_hidden_dims": policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
            "init_noise_std": float(policy_cfg.get("init_noise_std", 0.6)),
            "max_noise_std": float(policy_cfg.get("max_noise_std", 2.0)),
            "noise_std_type": policy_cfg.get("noise_std_type", "scalar"),
        }
        self.policy = MargDepthPitActorCritic(obs, obs_groups, num_actions=self.LEG_ACTION_DIM, **ac_kwargs).to(
            self.device
        )
        loaded = torch.load(student_ckpt_path, map_location=self.device, weights_only=False)
        state = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        self.policy.load_state_dict(state, strict=True)
        self.policy.eval()
        self._student_ckpt_path = student_ckpt_path

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.arm_default_action = torch.zeros((1, self.ARM_ACTION_DIM), device=self.device, dtype=torch.float32)

        self._proprio_history = torch.zeros(
            1, MARG_HISTORY_LEN, MARG_PROPRIO_DIM, device=self.device, dtype=torch.float32
        )
        self._last_leg_action = torch.zeros((1, self.LEG_ACTION_DIM), device=self.device, dtype=torch.float32)
        self._step_count = 0
        self._init_teleop_replay(demo_dir)
        print(
            f"[AlgSolution-Pit] MARG depth pit student ckpt={os.path.basename(student_ckpt_path)}, "
            f"depth={self.image_h}x{self.image_w}, cmd_vx={self.command_vx}, phase={self._phase}",
            flush=True,
        )

    def _init_teleop_replay(self, demo_dir: str) -> None:
        traj_path = os.environ.get("TELEOP_TRAJ_JSON", "").strip()
        if not traj_path:
            return
        traj_path = os.path.abspath(traj_path)
        if not os.path.isfile(traj_path):
            raise FileNotFoundError(f"TELEOP_TRAJ_JSON not found: {traj_path}")

        traj_data = load_teleop_trajectory(traj_path)
        delay_env = os.environ.get("TELEOP_REPLAY_DELAY", "").strip()
        replay_delay = float(delay_env) if delay_env else float(traj_data.get("replay_delay_s", 2.0))
        self._replayer = TrajectoryReplayer(traj_data["samples"], replay_delay_s=replay_delay)
        ll_policy = os.environ.get("TELEOP_LL_POLICY", "").strip() or traj_data.get("ll_policy")
        self._teleop = TaskDTeleopController(policy_path=ll_policy or None, device=self.device)
        self._phase = "teleop"
        self._teleop_done = False
        print(
            f"[AlgSolution-Pit] teleop replay enabled: {self._replayer.num_samples} samples, "
            f"traj={self._replayer.duration:.2f}s, delay={replay_delay:.2f}s, "
            f"total={self._replayer.total_duration:.2f}s, recorded_score={traj_data.get('final_score')}",
            flush=True,
        )

    def set_device(self, device: str) -> None:
        self.device = device
        self.policy = self.policy.to(device)
        self.arm_default_action = self.arm_default_action.to(device)
        self._proprio_history = self._proprio_history.to(device)
        self._last_leg_action = self._last_leg_action.to(device)
        if self._teleop is not None:
            self._teleop.set_device(device)
        self.reset()

    def bind_env(self, env) -> None:
        del env

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        """Platform hook: return {} to use official Task D B2Piper default action config."""
        return {}

    def reset(self, **kwargs):
        del kwargs
        self._proprio_history.zero_()
        self._last_leg_action.zero_()
        self._step_count = 0
        self._sim_step = 0
        if self._replayer is not None and self._teleop is not None:
            self._phase = "teleop"
            self._teleop_done = False
            self._teleop.reset()
            self._last_teleop_cmd = (0.0, 0.0, 0.0)
        else:
            self._phase = "pit"
            self._teleop_done = True

    def _sim_time(self) -> float:
        return float(self._sim_step) * float(self.dt)

    def _predict_teleop(self, obs, current_score):
        assert self._replayer is not None and self._teleop is not None
        sim_t = self._sim_time()
        vx, vy, wz = self._replayer.cmd_at(sim_t)
        self._last_teleop_cmd = (vx, vy, wz)
        if sim_t > self._replayer.total_duration + 0.5:
            self._teleop_done = True
            self._phase = "pit"
            self._proprio_history.zero_()
            self._step_count = 0
            print(f"[AlgSolution-Pit] teleop finished at t={sim_t:.2f}s → pit student", flush=True)
            return self._predict_pit(obs, current_score)
        self._teleop.set_velocity_command(vx, vy, wz)
        resp = self._teleop.predicts(obs, current_score)
        self._sim_step += 1
        return resp

    def _predict_pit(self, obs, current_score):
        del current_score
        policy_obs = self._build_policy_obs(obs)
        with torch.inference_mode():
            leg_action = self.policy.act_inference(policy_obs)
        self._last_leg_action = leg_action.detach().clone()
        self._step_count += 1
        self._sim_step += 1

        if "proprio_history" in obs:
            action_dim = self.LEG_ACTION_DIM
        else:
            proprio = obs["proprio"]
            if isinstance(proprio, torch.Tensor):
                pshape = proprio.shape[-1]
            else:
                pshape = len(proprio[0]) if proprio else 72
            action_dim = (int(pshape) - 12) // 3

        action_env = self._to_env_action(leg_action, action_dim)
        return {"action": action_env.detach().cpu().numpy().tolist(), "giveup": False}

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        return prep_depth(
            x.to(self.device),
            image_h=self.image_h,
            image_w=self.image_w,
            depth_max=self.depth_max,
        )

    def _build_depth_flat(self, obs: dict, batch: int) -> torch.Tensor:
        if "depth" in obs:
            depth = obs["depth"].to(self.device, dtype=torch.float32)
            if depth.ndim == 1:
                depth = depth.unsqueeze(0)
            return depth

        image_obs = obs.get("image", {})
        if not isinstance(image_obs, dict):
            raise RuntimeError("obs['image'] is missing or not a dict; head/ee depth cameras are required.")
        head_depth = image_obs.get("head_depth")
        if head_depth is None:
            head_depth = image_obs.get("video_depth")
        ee_depth = image_obs.get("ee_depth")
        if head_depth is None or ee_depth is None:
            raise RuntimeError("depth student requires head_depth and ee_depth in obs['image'].")
        head = self._prep_depth(head_depth).reshape(batch, -1)
        ee = self._prep_depth(ee_depth).reshape(batch, -1)
        return torch.cat([head, ee], dim=-1)

    def _marg_proprio_from_platform(self, proprio: torch.Tensor, action_dim: int) -> torch.Tensor:
        ang_vel = proprio[:, 3:6]
        cmd = proprio[:, 6:7]
        if float(cmd.abs().max().item()) < 1e-6:
            cmd = torch.full_like(cmd, self.command_vx)
        gravity = proprio[:, 9:12]
        idx = 12
        joint_pos_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx : idx + action_dim]

        joint_pos = joint_pos_all[:, : self.LEG_ACTION_DIM]
        joint_vel = joint_vel_all[:, : self.LEG_ACTION_DIM]
        last_action = actions_all[:, : self.LEG_ACTION_DIM]
        return torch.cat([ang_vel, gravity, cmd, joint_pos, joint_vel, last_action], dim=-1)

    def _marg_proprio_direct(self, obs: dict) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        return proprio

    def _update_history(self, current: torch.Tensor) -> torch.Tensor:
        self._proprio_history[:, 1:] = self._proprio_history[:, :-1].clone()
        self._proprio_history[:, 0] = current
        return self._proprio_history.reshape(current.shape[0], MARG_HISTORY_DIM)

    def _build_policy_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        if "proprio_history" in obs and "depth" in obs:
            proprio = self._marg_proprio_direct(obs)
            batch = proprio.shape[0]
            history = obs["proprio_history"].to(self.device, dtype=torch.float32)
            if history.ndim == 1:
                history = history.unsqueeze(0)
            depth = self._build_depth_flat(obs, batch)
            return {"proprio": proprio, "proprio_history": history, "depth": depth}

        proprio_platform = obs["proprio"].to(self.device, dtype=torch.float32)
        if proprio_platform.ndim == 1:
            proprio_platform = proprio_platform.unsqueeze(0)
        action_dim = (int(proprio_platform.shape[-1]) - 12) // 3
        current = self._marg_proprio_from_platform(proprio_platform, action_dim)
        history = self._update_history(current)
        depth = self._build_depth_flat(obs, current.shape[0])
        return {"proprio": current, "proprio_history": history, "depth": depth}

    def _to_env_action(self, leg_action: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_dim <= self.LEG_ACTION_DIM:
            return leg_action
        num_envs = leg_action.shape[0]
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action
        arm_idx = [i for i in self.arm_joint_indices if i < action_dim]
        if arm_idx:
            action_env[:, arm_idx] = self.arm_default_action[:, : len(arm_idx)].repeat(num_envs, 1)
        return action_env

    def predicts(self, obs, current_score):
        if self._replayer is not None and not self._teleop_done:
            return self._predict_teleop(obs, current_score)
        return self._predict_pit(obs, current_score)

    def get_video_overlay_lines(self) -> list[str]:
        if self._phase == "teleop" and not self._teleop_done:
            vx, vy, wz = self._last_teleop_cmd
            return [
                f"phase=teleop t={self._sim_time():.2f}/{self._replayer.total_duration:.2f}s",
                f"cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f})",
            ]
        act_norm = float(self._last_leg_action.norm().item())
        return [
            f"phase=pit ckpt={os.path.basename(self._student_ckpt_path)}",
            f"cmd_vx={self.command_vx:.2f} pit_step={self._step_count}",
            f"depth={self.image_h}x{self.image_w} act_norm={act_norm:.3f}",
        ]
