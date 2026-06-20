"""Task D MARG depth pit student deploy: depth + MARG proprio/history -> 12 leg actions."""

from __future__ import annotations

import json
import math
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
from taskd_hierarchical_nav import (  # noqa: E402
    TaskDHierarchicalNavDeploy,
    box_nominal_x_from_env,
    robot_horizontal_speed_from_env,
)

MARG_PROPRIO_DIM = 43
MARG_HISTORY_LEN = 6
MARG_HISTORY_DIM = MARG_PROPRIO_DIM * MARG_HISTORY_LEN
MARG_ELEVATION_OUT_DIM = 16
MARG_ESTIMATOR_OUT_DIM = 7
MARG_CRITIC_PRIV_DIM = 42

# Platform Task D camera resolution (server.py / play); policy sees prep_depth → 24x32.
PLATFORM_CAM_H = 480
PLATFORM_CAM_W = 640

_LEG_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
# Pit-loco policy raw output -> Task D env leg action (matches teleop_controller.train_to_env_action_scale).
# Effective joint delta: pit uses hip=0.125 / other=0.25; Task D leg scale is uniform 0.5.
_PIT_TO_TASKD_LEG_ACTION_SCALE = (0.25, 0.5, 0.5) * 4


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
        head_depth_only: bool | None = None,
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
        if head_depth_only is None:
            depth_dim = obs["depth"].shape[-1]
            head_depth_only = depth_dim <= self.depth_flat_per_cam
        self.head_depth_only = bool(head_depth_only)

        self.estimator = MLP(MARG_HISTORY_DIM, MARG_ESTIMATOR_OUT_DIM, estimator_hidden_dims, activation)
        self.head_depth_encoder = ConvEncoder(in_ch=self.depth_channels, out_dim=enc_dim)
        if self.head_depth_only:
            self.ee_depth_encoder = None
            self.depth_fusion = MLP(enc_dim, elevation_out_dim, depth_hidden_dims, activation)
        else:
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

    def _reshape_head_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        batch = depth_flat.shape[0]
        head = depth_flat[:, : self.depth_flat_per_cam]
        return head.reshape(batch, self.depth_channels, self.img_h, self.img_w)

    def _reshape_ee_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        batch = depth_flat.shape[0]
        ee = depth_flat[:, self.depth_flat_per_cam :]
        return ee.reshape(batch, self.depth_channels, self.img_h, self.img_w)

    def _encode_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        head_feat = self.head_depth_encoder(self._reshape_head_depth(depth_flat))
        if self.head_depth_only:
            return self.depth_fusion(head_feat)
        ee_feat = self.ee_depth_encoder(self._reshape_ee_depth(depth_flat))
        return self.depth_fusion(torch.cat([head_feat, ee_feat], dim=-1))

    def _encode_actor(self, obs: dict) -> torch.Tensor:
        est_out = self.estimator(obs["proprio_history"])
        depth_out = self._encode_depth(obs["depth"])
        return torch.cat([obs["proprio"], est_out, depth_out], dim=-1)

    def act_inference(self, obs: dict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self._encode_actor(obs))
        return self.actor(actor_obs)


def _load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_policy_cfg(demo_dir: str) -> dict:
    for name in ("agent_marg_depth_pit.yaml", "agent.yaml"):
        data = _load_yaml(os.path.join(demo_dir, name))
        if data:
            return data.get("policy", data) if "policy" in data else data
    return {}


def _load_deploy_cfg(demo_dir: str) -> dict:
    for name in ("deploy_marg_depth_pit.yaml", "deploy.yaml"):
        data = _load_yaml(os.path.join(demo_dir, name))
        if data:
            return data
    return {}


def _resolve_bundle_path(demo_dir: str, path: str | None) -> str | None:
    if not path:
        return None
    candidate = path if os.path.isabs(path) else os.path.join(demo_dir, str(path))
    candidate = os.path.abspath(candidate)
    return candidate if os.path.isfile(candidate) else None


def _find_student_ckpt(demo_dir: str, deploy: dict, override: str | None = None) -> str:
    for rel in (override, deploy.get("pit_student_ckpt")):
        resolved = _resolve_bundle_path(demo_dir, rel)
        if resolved:
            return resolved
    for name in ("model_200.pt", "model_800.pt", "model_400.pt"):
        resolved = _resolve_bundle_path(demo_dir, name)
        if resolved:
            return resolved
    import glob

    hits = sorted(glob.glob(os.path.join(demo_dir, "model_*.pt")))
    if hits:
        return os.path.abspath(hits[-1])
    raise FileNotFoundError(
        f"PIT student checkpoint not found in {demo_dir}. "
        f"Set pit_student_ckpt in deploy_marg_depth_pit.yaml or place model_*.pt next to solution.py."
    )


class AlgSolution:
    """Task D: hierarchical nav student or teleop traj → MARG depth pit student."""

    LEG_ACTION_DIM = 12
    ARM_ACTION_DIM = 8
    SIM_DT = 0.02
    DEFAULT_NAV_MAX_STEPS = 12000

    def __init__(
        self,
        *,
        student_ckpt: str | None = None,
        teleop_traj: str | None = None,
        nav_ckpt: str | None = None,
        command_vx: float | None = None,
        handoff_warmup: int | None = None,
        nav_only: bool = False,
    ):
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        self._demo_dir = demo_dir
        deploy_cfg = _load_deploy_cfg(demo_dir)
        policy_cfg = _load_policy_cfg(demo_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = self.SIM_DT
        self._phase = "pit"
        self._sim_step = 0
        self._teleop_done = True
        self._nav_done = True
        self._replayer: TrajectoryReplayer | None = None
        self._teleop: TaskDTeleopController | None = None
        self._nav: TaskDHierarchicalNavDeploy | None = None
        self._nav_ckpt_path: str | None = None
        self._last_teleop_cmd = (0.0, 0.0, 0.0)
        self._last_box_nominal_x = float("nan")
        self._last_robot_speed_xy = 0.0
        self._nav_phase_steps = 0
        self._env = None
        self.nav_only = bool(nav_only)
        self._phase_handoff_enabled = not self.nav_only

        self.image_h = int(policy_cfg.get("img_h", 24))
        self.image_w = int(policy_cfg.get("img_w", 32))
        self.img_channels = int(policy_cfg.get("depth_channels", 1))
        self.depth_only = self.img_channels == 1
        self.depth_max = float(
            policy_cfg.get("depth_max", deploy_cfg.get("depth_max", 5.0))
        )
        self.platform_cam_h = PLATFORM_CAM_H
        self.platform_cam_w = PLATFORM_CAM_W
        self.depth_render_h = int(policy_cfg.get("depth_render_h", PLATFORM_CAM_H))
        self.depth_render_w = int(policy_cfg.get("depth_render_w", PLATFORM_CAM_W))
        if command_vx is not None:
            self.command_vx = float(command_vx)
        else:
            self.command_vx = float(deploy_cfg.get("command_vx", 0.6))

        self.head_depth_only = bool(
            policy_cfg.get("head_depth_only", deploy_cfg.get("head_depth_only", True))
        )
        self.policy = None
        self._student_ckpt_path: str | None = None
        if self.nav_only and not student_ckpt:
            print("[AlgSolution-Pit] nav_only: skipping pit student load (no --pit_ckpt)", flush=True)
        else:
            student_ckpt_path = _find_student_ckpt(demo_dir, deploy_cfg, student_ckpt)
            if not os.path.isfile(student_ckpt_path):
                raise FileNotFoundError(f"PIT student checkpoint not found: {student_ckpt_path}")
            loaded = torch.load(student_ckpt_path, map_location=self.device, weights_only=False)
            state = (
                loaded["model_state_dict"]
                if isinstance(loaded, dict) and "model_state_dict" in loaded
                else loaded
            )
            if "head_depth_only" in policy_cfg:
                self.head_depth_only = bool(policy_cfg["head_depth_only"])
            elif "head_depth_only" in deploy_cfg:
                self.head_depth_only = bool(deploy_cfg["head_depth_only"])
            else:
                self.head_depth_only = not any(k.startswith("ee_depth_encoder.") for k in state)

            num_cams = 1 if self.head_depth_only else 2
            depth_dim = num_cams * self.img_channels * self.image_h * self.image_w
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
                "head_depth_only": self.head_depth_only,
            }
            self.policy = MargDepthPitActorCritic(
                obs, obs_groups, num_actions=self.LEG_ACTION_DIM, **ac_kwargs
            ).to(self.device)
            self.policy.load_state_dict(state, strict=True)
            self.policy.eval()
            self._student_ckpt_path = student_ckpt_path

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.arm_default_action = torch.zeros((1, self.ARM_ACTION_DIM), device=self.device, dtype=torch.float32)
        self._pit_to_taskd_leg_scale = torch.tensor(
            _PIT_TO_TASKD_LEG_ACTION_SCALE, device=self.device, dtype=torch.float32
        ).view(1, -1)

        self._proprio_history = torch.zeros(
            1, MARG_HISTORY_LEN, MARG_PROPRIO_DIM, device=self.device, dtype=torch.float32
        )
        self._last_leg_action = torch.zeros((1, self.LEG_ACTION_DIM), device=self.device, dtype=torch.float32)
        self._handoff_leg_baseline = torch.zeros((1, self.LEG_ACTION_DIM), device=self.device, dtype=torch.float32)
        self._handoff_arm_action = torch.zeros((1, self.ARM_ACTION_DIM), device=self.device, dtype=torch.float32)
        self._step_count = 0
        if handoff_warmup is not None:
            self._pit_handoff_warmup_total = max(0, int(handoff_warmup))
        else:
            self._pit_handoff_warmup_total = max(0, int(deploy_cfg.get("handoff_warmup", 0)))
        self._pit_warmup_remaining = 0
        self.pit_edge_only = False
        self.pit_warmup_steps = 0
        self._play_configured = False
        self._last_obs_mgr = False
        self._platform_deploy = False
        self._use_platform_cam = True
        self._deploy_cfg = deploy_cfg
        self._nav_max_steps = int(deploy_cfg.get("nav_max_steps", self.DEFAULT_NAV_MAX_STEPS))
        self._nav_handoff_speed = float(deploy_cfg.get("nav_handoff_speed", 0.12))
        self._nav_handoff_min_runtime_s = float(
            deploy_cfg.get("nav_handoff_min_runtime_s", deploy_cfg.get("nav_handoff_still_s", 3.0))
        )
        nav_ckpt_path = nav_ckpt or deploy_cfg.get("nav_ckpt")
        use_teleop = bool(deploy_cfg.get("enable_teleop", False)) and not nav_ckpt_path
        if nav_ckpt_path:
            self._setup_nav(nav_ckpt_path, ll_policy=deploy_cfg.get("teleop_ll_policy"))
        elif use_teleop:
            self._setup_teleop(
                teleop_traj,
                ll_policy=deploy_cfg.get("teleop_ll_policy"),
                enable=True,
                replay_delay=deploy_cfg.get("teleop_replay_delay"),
            )
        else:
            self._replayer = None
            self._teleop = None
            self._nav = None
            self._phase = "pit"
            self._teleop_done = True
            self._nav_done = True
        pit_tag = (
            os.path.basename(self._student_ckpt_path)
            if self._student_ckpt_path
            else "none(nav_only)"
        )
        handoff_tag = "off(nav_only)" if not self._phase_handoff_enabled else "on"
        nav_tag = os.path.basename(self._nav_ckpt_path) if self._nav_ckpt_path else "none"
        print(
            f"[AlgSolution-Pit] MARG depth pit student ckpt={pit_tag}, "
            f"nav_ckpt={nav_tag}, phase={self._phase}, handoff={handoff_tag}, "
            f"depth={'head' if self.head_depth_only else 'head+ee'} "
            f"platform={self.platform_cam_h}x{self.platform_cam_w}->policy={self.image_h}x{self.image_w}, "
            f"cmd_vx={self.command_vx}, device={self.device}",
            flush=True,
        )

    def _setup_nav(self, nav_ckpt_path: str | None, *, ll_policy: str | None = None) -> None:
        self._replayer = None
        self._teleop = None
        self._nav = None
        resolved = _resolve_bundle_path(self._demo_dir, nav_ckpt_path)
        if resolved is None and nav_ckpt_path:
            resolved = os.path.abspath(nav_ckpt_path) if os.path.isfile(nav_ckpt_path) else None
        if resolved is None:
            resolved = _resolve_bundle_path(self._demo_dir, self._deploy_cfg.get("nav_ckpt"))
        if resolved is None or not os.path.isfile(resolved):
            self._phase = "pit"
            self._nav_done = True
            print("[AlgSolution-Pit] nav ckpt not found; starting in pit-only mode.", flush=True)
            return

        ll_path = _resolve_bundle_path(self._demo_dir, ll_policy)
        if ll_path is None:
            ll_path = os.path.join(self._demo_dir, "policy.pt")
        self._nav = TaskDHierarchicalNavDeploy(
            demo_dir=self._demo_dir,
            student_ckpt_path=resolved,
            ll_policy_path=ll_path,
            device=self.device,
        )
        self._nav_ckpt_path = resolved
        self._phase = "nav"
        self._nav_done = False
        print(
            f"[AlgSolution-Pit] hierarchical nav enabled: ckpt={os.path.basename(resolved)}, "
            f"handoff=speed<={self._nav_handoff_speed:.2f}m/s and nav_time>"
            f"{self._nav_handoff_min_runtime_s:.1f}s (max {self._nav_max_steps} steps)",
            flush=True,
        )

    def _setup_teleop(
        self,
        traj_path: str | None,
        *,
        ll_policy: str | None = None,
        enable: bool = True,
        replay_delay: float | None = None,
    ) -> None:
        self._replayer = None
        self._teleop = None
        if not enable:
            self._phase = "pit"
            self._teleop_done = True
            return

        resolved = _resolve_bundle_path(self._demo_dir, traj_path)
        if resolved is None and traj_path:
            resolved = os.path.abspath(traj_path) if os.path.isfile(traj_path) else None
        if resolved is None:
            resolved = _resolve_bundle_path(self._demo_dir, self._deploy_cfg.get("teleop_traj"))
        if resolved is None:
            self._phase = "pit"
            self._teleop_done = True
            return

        traj_data = load_teleop_trajectory(resolved)
        if replay_delay is None:
            replay_delay = traj_data.get("replay_delay_s", 2.0)
        replay_delay = float(replay_delay)
        self._replayer = TrajectoryReplayer(traj_data["samples"], replay_delay_s=replay_delay)

        ll_path = _resolve_bundle_path(self._demo_dir, ll_policy)
        if ll_path is None:
            ll_path = traj_data.get("ll_policy")
        self._teleop = TaskDTeleopController(policy_path=ll_path or None, device=self.device)
        self._phase = "teleop"
        self._teleop_done = False
        print(
            f"[AlgSolution-Pit] teleop replay enabled: {self._replayer.num_samples} samples, "
            f"traj={os.path.basename(resolved)}, duration={self._replayer.duration:.2f}s, "
            f"delay={replay_delay:.2f}s, total={self._replayer.total_duration:.2f}s, "
            f"recorded_score={traj_data.get('final_score')}",
            flush=True,
        )

    def _init_teleop_replay(self, demo_dir: str) -> None:
        """Backward-compatible alias."""
        del demo_dir
        self._setup_teleop(
            self._deploy_cfg.get("teleop_traj"),
            ll_policy=self._deploy_cfg.get("teleop_ll_policy"),
            enable=bool(self._deploy_cfg.get("enable_teleop", True)),
            replay_delay=self._deploy_cfg.get("teleop_replay_delay"),
        )

    def set_device(self, device: str) -> None:
        self.device = device
        if self.policy is not None:
            self.policy = self.policy.to(device)
        self.arm_default_action = self.arm_default_action.to(device)
        self._pit_to_taskd_leg_scale = self._pit_to_taskd_leg_scale.to(device)
        self._proprio_history = self._proprio_history.to(device)
        self._last_leg_action = self._last_leg_action.to(device)
        self._handoff_leg_baseline = self._handoff_leg_baseline.to(device)
        self._handoff_arm_action = self._handoff_arm_action.to(device)
        if self._teleop is not None:
            self._teleop.set_device(device)
        if self._nav is not None:
            self._nav.set_device(device)
        self.reset()

    def bind_env(self, env) -> None:
        """Optional sim handle for MARG-aligned proprio (teleop→pit handoff)."""
        self._env = env

    def _uses_obs_manager(self, obs: dict) -> bool:
        """True when play attached marg_proprio (not server.py deploy path)."""
        if self._platform_deploy:
            return False
        return "marg_proprio" in obs

    @staticmethod
    def estimate_teleop_video_steps(traj_path: str, *, step_dt: float = 0.02) -> int:
        with open(traj_path, encoding="utf-8") as f:
            data = json.load(f)
        delay = float(data.get("replay_delay_s", 2.0))
        samples = data.get("samples") or []
        if not samples:
            return int(delay / step_dt) + 10
        duration = float(samples[-1].get("t", 0.0)) - float(samples[0].get("t", 0.0))
        return int((delay + max(0.0, duration)) / step_dt) + 10

    @staticmethod
    def estimate_nav_video_steps(*, max_steps: int | None = None, step_dt: float = 0.02) -> int:
        steps = int(max_steps if max_steps is not None else AlgSolution.DEFAULT_NAV_MAX_STEPS)
        return steps + 10

    def _uses_nav_phase(self) -> bool:
        return self._nav is not None

    def configure_play_args(self, args) -> None:
        """Apply play_atec_task / play_taskd_teleop_pit CLI (called once after AlgSolution load)."""
        if getattr(args, "pit_command_vx", None) is not None:
            self.command_vx = float(args.pit_command_vx)
        if bool(getattr(args, "nav_only", False)):
            self.nav_only = True
            self._phase_handoff_enabled = False
        self.pit_edge_only = bool(getattr(args, "pit_edge_only", False))
        if self.pit_edge_only:
            self._pit_handoff_warmup_total = 0
        elif getattr(args, "pit_handoff_warmup", None) is not None:
            self._pit_handoff_warmup_total = max(0, int(args.pit_handoff_warmup))
        self.pit_warmup_steps = max(
            0,
            int(getattr(args, "pit_warmup_steps", 40) if self.pit_edge_only else 0),
        )
        if bool(getattr(args, "ee_depth", False)):
            self.head_depth_only = False
        self._platform_deploy = bool(getattr(args, "platform_deploy", False))
        nav_ckpt = getattr(args, "nav_ckpt", None)
        if nav_ckpt:
            self._setup_nav(
                os.path.abspath(str(nav_ckpt)),
                ll_policy=getattr(args, "ll_policy", None) or self._deploy_cfg.get("teleop_ll_policy"),
            )
        elif getattr(args, "teleop_traj", None):
            self._setup_teleop(
                os.path.abspath(str(args.teleop_traj)),
                ll_policy=getattr(args, "ll_policy", None) or self._deploy_cfg.get("teleop_ll_policy"),
                enable=True,
                replay_delay=self._deploy_cfg.get("teleop_replay_delay"),
            )
        elif self.pit_edge_only:
            self._replayer = None
            self._teleop = None
            self._nav = None
            self._phase = "pit"
            self._teleop_done = True
            self._nav_done = True
        self._play_configured = True
        if self.nav_only and self._uses_nav_phase():
            mode = "nav-only"
        elif self._uses_nav_phase():
            mode = "nav→pit"
        elif self.pit_edge_only:
            mode = "pit-edge"
        elif self._replayer:
            mode = "teleop→pit"
        else:
            mode = "pit-only"
        handoff_tag = "disabled" if not self._phase_handoff_enabled else "enabled"
        deploy_tag = ", platform_deploy=on" if self._platform_deploy else ""
        print(
            f"[AlgSolution-Pit] play args: mode={mode}, cmd_vx={self.command_vx:.2f}, "
            f"phase_handoff={handoff_tag}, handoff_warmup={self._pit_handoff_warmup_total}, "
            f"pit_warmup={self.pit_warmup_steps}, depth={'head' if self.head_depth_only else 'head+ee'}"
            f"{deploy_tag}",
            flush=True,
        )

    def _apply_pit_edge_spawn(self, env_cfg) -> None:
        from atec_rl_lab.tasks.task_d.locomotion.mdp.events import (
            TASK_D_PIT_PUSHED_BOX_LOCAL,
            task_d_pit_marg_spawn_local,
        )

        robot_spawn = task_d_pit_marg_spawn_local()
        box_spawn = tuple(float(v) for v in TASK_D_PIT_PUSHED_BOX_LOCAL)
        if getattr(env_cfg.events, "reset_robot_root", None) is not None:
            env_cfg.events.reset_robot_root.params["local_pos"] = robot_spawn
        if getattr(env_cfg.events, "reset_box_root", None) is not None:
            env_cfg.events.reset_box_root.params["local_pos"] = box_spawn
        print(
            f"[AlgSolution-Pit] pit-edge spawn: robot_local={robot_spawn}, box_local={box_spawn}",
            flush=True,
        )

    def configure_env_cfg(self, env_cfg, args) -> None:
        """Task D B2 play: pit terrain + obs_manager MARG groups + depth (matches pit-loco train)."""
        import atec_rl_lab.tasks  # noqa: F401
        from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import (
            apply_taskd_pit_student_command,
            attach_taskd_platform_marg_student_obs,
            configure_pit_e2e_cameras,
        )
        from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import apply_play_spawn, configure_play_terrain

        num_envs = max(1, int(args.num_envs))
        env_cfg.scene.num_envs = num_envs
        env_cfg.pit_width_range = (float(args.pit_width_min), float(args.pit_width_max))
        env_cfg.pit_curriculum_levels = 11
        configure_play_terrain(env_cfg, pit_level=int(args.pit_level), use_pit_curriculum=False)
        apply_play_spawn(env_cfg, spawn_x_offset=0.0, spawn_local_x=None)
        if bool(getattr(args, "pit_edge_only", False)):
            self._apply_pit_edge_spawn(env_cfg)

        if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
            env_cfg.curriculum.command_levels_lin_vel = None
            env_cfg.curriculum.pit_width_levels = None

        decimation = int(getattr(env_cfg, "decimation", 4))
        sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
        phys_dt = decimation * sim_dt
        head_depth_only = not bool(getattr(args, "ee_depth", False))
        if self._uses_nav_phase() or getattr(args, "nav_ckpt", None):
            # Phase 1 nav: head+ee in obs['depth']; phase 2 pit slices head only at inference.
            head_depth_only = False
        env_cfg.head_depth_only = head_depth_only
        self._env_head_depth_only = head_depth_only

        policy_h = int(getattr(args, "pit_cam_h", self.image_h))
        policy_w = int(getattr(args, "pit_cam_w", self.image_w))
        # Default: sim cameras 480x640, obs_manager prep_depth → policy_h x policy_w (nav + pit).
        use_platform_cam = not bool(getattr(args, "no_platform_depth", False))
        self._use_platform_cam = use_platform_cam
        if use_platform_cam:
            cam_h, cam_w = self.platform_cam_h, self.platform_cam_w
            cam_tag = (
                f"{'head' if head_depth_only else 'head+ee'} platform {cam_h}x{cam_w} "
                f"depth-only (solution prep_depth→{policy_h}x{policy_w})"
            )
        else:
            cam_h, cam_w = policy_h, policy_w
            cam_tag = (
                f"{'head' if head_depth_only else 'head+ee'} native {cam_h}x{cam_w} "
                f"(matches DAgger sim_camera, no downsample)"
            )
        env_cfg.depth_render_h = cam_h
        env_cfg.depth_render_w = cam_w
        from atec_rl_lab.tasks.task_d.env_cfg import TASK_D_NAV_DEPTH_MAX

        configure_pit_e2e_cameras(
            env_cfg,
            camera_height=cam_h,
            camera_width=cam_w,
            depth_only=True,
            tiled=False,
            # Match play_taskd_marg_depth_pit_dagger.py (5m), not platform 50m server path.
            camera_far_clip=float(TASK_D_NAV_DEPTH_MAX),
            update_period=phys_dt,
            head_depth_only=head_depth_only,
        )
        attach_taskd_platform_marg_student_obs(
            env_cfg,
            args,
            policy_h=policy_h,
            policy_w=policy_w,
            depth_max=float(self.depth_max),
            depth_render_h=cam_h if use_platform_cam else None,
            depth_render_w=cam_w if use_platform_cam else None,
            head_depth_only=head_depth_only,
        )
        env_cfg.depth_render_h = cam_h
        env_cfg.depth_render_w = cam_w
        apply_taskd_pit_student_command(env_cfg, command_vx=float(self.command_vx))

        if hasattr(env_cfg.scene, "ee_dual_camera"):
            env_cfg.scene.ee_dual_camera = None
        if hasattr(env_cfg, "observations"):
            env_cfg.observations.extero = None
        if hasattr(env_cfg.scene, "lidar_sensor"):
            env_cfg.scene.lidar_sensor = None

        mode = "pit-edge-only" if bool(getattr(args, "pit_edge_only", False)) else (
            "nav→pit" if self._uses_nav_phase() else ("teleop→pit" if self._replayer else "pit-only")
        )
        print(
            f"[AlgSolution-Pit] env cfg: Task D {mode}, num_envs={num_envs}, "
            f"cameras={cam_tag}, pit_level={int(args.pit_level)}, "
            f"obs_depth={'head' if head_depth_only else 'head+ee'}, "
            f"pit_model={'head' if self.head_depth_only else 'head+ee'}, "
            f"obs_manager=marg_proprio+depth+local_history, "
            f"cmd_vx={self.command_vx:.2f}",
            flush=True,
        )

    def on_play_reset(self, obs: dict) -> None:
        if self.pit_edge_only:
            self.init_pit_edge_start(obs)

    def _leg_joint_ids(self, robot) -> list[int]:
        return [robot.data.joint_names.index(name) for name in _LEG_JOINT_NAMES]

    def _env_leg_to_pit_raw(self, env_leg: torch.Tensor) -> torch.Tensor:
        scale = self._pit_to_taskd_leg_scale.to(device=env_leg.device, dtype=env_leg.dtype)
        return env_leg / scale

    def _arm_hold_action_from_sim(self) -> torch.Tensor:
        """Task D env arm action that holds current joint pose (avoid snap-to-default at handoff)."""
        if self._env is None:
            return self.arm_default_action.clone()
        unwrapped = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        robot = unwrapped.scene["robot"]
        n_arm = min(self.ARM_ACTION_DIM, robot.data.joint_pos.shape[-1] - len(_LEG_JOINT_NAMES))
        if n_arm <= 0:
            return self.arm_default_action.clone()
        leg_n = len(_LEG_JOINT_NAMES)
        arm_pos = robot.data.joint_pos[:, leg_n : leg_n + n_arm]
        arm_default = robot.data.default_joint_pos[:, leg_n : leg_n + n_arm]
        # Task D joint_pos_arm scale=0.5, use_default_offset=True
        hold = (arm_pos - arm_default) / 0.5
        out = self.arm_default_action.clone()
        out[:, :n_arm] = hold.to(self.device, dtype=torch.float32)
        return out

    def _capture_handoff_baselines(self) -> None:
        """Snapshot teleop end actions / arm pose before switching to pit student."""
        if self._env is None:
            self._handoff_leg_baseline.zero_()
            self._handoff_arm_action = self.arm_default_action.clone()
            return
        unwrapped = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        act = unwrapped.action_manager.action.to(self.device, dtype=torch.float32)
        self._handoff_leg_baseline = self._env_leg_to_pit_raw(act[:, : self.LEG_ACTION_DIM])
        self._handoff_arm_action = self._arm_hold_action_from_sim()
        leg_norm = float(self._handoff_leg_baseline.norm().item())
        arm_norm = float(self._handoff_arm_action.norm().item())
        print(
            f"[AlgSolution-Pit] handoff baseline: teleop leg(pit-raw) norm={leg_norm:.3f}, "
            f"arm_hold norm={arm_norm:.3f}",
            flush=True,
        )

    def _marg_proprio_from_sim(self, *, zero_last_action: bool = False) -> torch.Tensor | None:
        """Build 43D MARG proprio from bound env (matches pit train / play_taskd_marg_depth_pit_solution)."""
        if self._env is None:
            return None
        unwrapped = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        robot = unwrapped.scene["robot"]
        leg_ids = self._leg_joint_ids(robot)
        ang_vel = robot.data.root_ang_vel_b[:, :3].to(self.device, dtype=torch.float32)
        gravity = robot.data.projected_gravity_b.to(self.device, dtype=torch.float32)
        cmd = torch.full(
            (robot.data.root_ang_vel_b.shape[0], 1),
            float(self.command_vx),
            device=self.device,
            dtype=torch.float32,
        )
        joint_pos = (
            robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids]
        ).to(self.device, dtype=torch.float32)
        joint_vel = robot.data.joint_vel[:, leg_ids].to(self.device, dtype=torch.float32)
        if (
            self._phase == "pit"
            and self._step_count > 0
            and unwrapped.action_manager.action.shape[-1] > len(leg_ids)
        ):
            # Task D stores scaled env actions; MARG proprio expects pit-loco raw policy outputs.
            last_action = self._last_leg_action.to(self.device, dtype=torch.float32)
        else:
            last_action = unwrapped.action_manager.action
            if last_action.shape[-1] != len(leg_ids):
                last_action = last_action[:, : len(leg_ids)]
            last_action = last_action.to(self.device, dtype=torch.float32)
            if unwrapped.action_manager.action.shape[-1] > len(leg_ids):
                last_action = self._env_leg_to_pit_raw(last_action)
        if zero_last_action:
            last_action = torch.zeros_like(last_action)
        return torch.cat([ang_vel, gravity, cmd, joint_pos, joint_vel, last_action], dim=-1)

    def _prefill_proprio_history(self, current: torch.Tensor) -> None:
        """Match pit-loco train reset: history=[o_t, 0, 0, 0, 0, 0], not 6× identical frames."""
        current = current.to(self.device, dtype=torch.float32)
        if current.ndim == 1:
            current = current.unsqueeze(0)
        self._proprio_history.zero_()
        self._proprio_history[:, 0] = current

    def _sync_proprio_history_from_obs(self, obs: dict) -> None:
        current = self._marg_proprio_from_sim()
        if current is None:
            proprio_platform = obs["proprio"].to(self.device, dtype=torch.float32)
            if proprio_platform.ndim == 1:
                proprio_platform = proprio_platform.unsqueeze(0)
            action_dim = (int(proprio_platform.shape[-1]) - 12) // 3
            current = self._marg_proprio_from_platform(proprio_platform, action_dim)
        self._update_history(current)

    def _tensor_obs(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x.to(self.device, dtype=torch.float32)
        else:
            t = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

    def _reset_env_marg_history(self, obs: dict, *, zero_last_action: bool = True) -> None:
        if not self._uses_obs_manager(obs) or self._env is None:
            return
        u = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        hist = getattr(u, "_marg_proprio_history", None)
        if hist is None:
            return
        current = self._tensor_obs(obs["marg_proprio"]).clone()
        if zero_last_action:
            current[:, -self.LEG_ACTION_DIM :] = 0.0
        hist[:] = current.unsqueeze(1).expand(-1, MARG_HISTORY_LEN, -1)

    def _fix_marg_proprio_last_action(self, proprio: torch.Tensor, *, zero: bool) -> torch.Tensor:
        out = proprio.clone()
        if zero:
            out[:, -self.LEG_ACTION_DIM :] = 0.0
        elif self._step_count > 0:
            out[:, -self.LEG_ACTION_DIM :] = self._last_leg_action.to(out.device, dtype=out.dtype)
        return out

    def _build_policy_obs_from_obs_manager(self, obs: dict) -> dict[str, torch.Tensor]:
        if "marg_proprio" not in obs:
            raise RuntimeError(
                "Sim env missing marg_proprio obs group; call solution.configure_env_cfg() first."
            )
        zero_last = self._step_count == 0 and self._phase == "pit"
        proprio = self._fix_marg_proprio_last_action(
            self._tensor_obs(obs["marg_proprio"]),
            zero=zero_last,
        )
        batch = proprio.shape[0]
        history = self._proprio_history.reshape(batch, MARG_HISTORY_DIM)
        if "depth" not in obs:
            raise RuntimeError(
                "obs_manager pit play requires obs['depth'] (attach_dagger_depth_obs in configure_env_cfg)."
            )
        return {
            "proprio": proprio,
            "proprio_history": history,
            "depth": self._build_depth_flat(obs, batch),
        }

    def _prefill_pit_start(self, obs: dict, *, warmup_steps: int, tag: str) -> None:
        """Shared pit-lip start: prefill history + optional teleop handoff warmup."""
        self._phase = "pit"
        self._teleop_done = True
        self._last_leg_action.zero_()
        self._step_count = 0
        self._pit_warmup_remaining = max(0, int(warmup_steps))
        if self._uses_obs_manager(obs):
            current = self._fix_marg_proprio_last_action(
                self._tensor_obs(obs["marg_proprio"]),
                zero=True,
            )
            self._prefill_proprio_history(current)
            self._reset_env_marg_history(obs, zero_last_action=True)
        else:
            current = self._marg_proprio_from_sim(zero_last_action=True)
            if current is None:
                proprio_platform = obs["proprio"].to(self.device, dtype=torch.float32)
                if proprio_platform.ndim == 1:
                    proprio_platform = proprio_platform.unsqueeze(0)
                action_dim = (int(proprio_platform.shape[-1]) - 12) // 3
                current = self._marg_proprio_from_platform(proprio_platform, action_dim)
                current[:, -self.LEG_ACTION_DIM :] = 0.0
            self._prefill_proprio_history(current)
        warm_msg = (
            f", handoff_warmup={self._pit_warmup_remaining} steps"
            if self._pit_warmup_remaining > 0
            else ", no handoff warmup"
        )
        print(
            f"[AlgSolution-Pit] {tag}: prefilled proprio_history, last_action=0, "
            f"cmd_vx={self.command_vx:.2f}{warm_msg}",
            flush=True,
        )

    def init_pit_edge_start(self, obs: dict) -> None:
        """Task D pit-lip ablation: skip teleop, run student directly (control experiment)."""
        self._handoff_leg_baseline.zero_()
        self._handoff_arm_action = self._arm_hold_action_from_sim()
        self._prefill_pit_start(obs, warmup_steps=0, tag="pit-edge direct")

    def _on_teleop_to_pit_handoff(self, obs: dict) -> None:
        self._prefill_pit_start(
            obs,
            warmup_steps=int(self._pit_handoff_warmup_total),
            tag="teleop→pit",
        )

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        """Platform hook: return {} to use official Task D B2Piper default action config."""
        return {}

    def reset(self, **kwargs):
        del kwargs
        self._proprio_history.zero_()
        self._last_leg_action.zero_()
        self._handoff_leg_baseline.zero_()
        self._handoff_arm_action = self.arm_default_action.clone()
        self._step_count = 0
        self._sim_step = 0
        self._nav_phase_steps = 0
        self._last_robot_speed_xy = 0.0
        if self._replayer is not None and self._teleop is not None:
            self._phase = "teleop"
            self._teleop_done = False
            self._nav_done = True
            self._teleop.reset()
            self._last_teleop_cmd = (0.0, 0.0, 0.0)
        elif self._nav is not None:
            self._phase = "nav"
            self._nav_done = False
            self._teleop_done = True
            self._nav.reset()
        else:
            self._phase = "pit"
            self._teleop_done = True
            self._nav_done = True

    def _sim_time(self) -> float:
        return float(self._sim_step) * float(self.dt)

    def _predict_teleop(self, obs, current_score):
        assert self._replayer is not None and self._teleop is not None
        sim_t = self._sim_time()
        vx, vy, wz = self._replayer.cmd_at(sim_t)
        self._last_teleop_cmd = (vx, vy, wz)
        if sim_t > self._replayer.total_duration + 0.5:
            self._teleop_done = True
            if not self._phase_handoff_enabled:
                print(
                    f"[AlgSolution-Pit] teleop finished at t={sim_t:.2f}s (phase handoff disabled, holding teleop idle)",
                    flush=True,
                )
                self._last_teleop_cmd = (0.0, 0.0, 0.0)
                self._teleop.set_velocity_command(0.0, 0.0, 0.0)
                resp = self._teleop.predicts(obs, current_score)
                self._sync_proprio_history_from_obs(obs)
                self._sim_step += 1
                return resp
            self._capture_handoff_baselines()
            self._on_teleop_to_pit_handoff(obs)
            self._phase = "pit"
            print(f"[AlgSolution-Pit] teleop finished at t={sim_t:.2f}s → pit student", flush=True)
            return self._predict_pit(obs, current_score)
        self._teleop.set_velocity_command(vx, vy, wz)
        resp = self._teleop.predicts(obs, current_score)
        self._sync_proprio_history_from_obs(obs)
        self._sim_step += 1
        return resp

    def _env_step_dt(self) -> float:
        if self._env is not None:
            try:
                return float(self._env.unwrapped.step_dt)
            except Exception:
                pass
        return float(self.dt)

    def _nav_elapsed_s(self) -> float:
        return float(self._nav_phase_steps) * self._env_step_dt()

    def _robot_speed_xy_from_proprio(self, obs: dict) -> float | None:
        """Platform has no sim env handle; use base lin vel from proprio (0:3)."""
        try:
            proprio = obs.get("proprio")
            if proprio is None:
                return None
            if not isinstance(proprio, torch.Tensor):
                proprio = torch.as_tensor(proprio, dtype=torch.float32)
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            vxy = proprio[0, :2].to(dtype=torch.float32)
            return float(torch.linalg.vector_norm(vxy).item())
        except Exception:
            return None

    def _update_nav_handoff_progress(self, obs: dict | None = None) -> None:
        if self._env is not None:
            try:
                self._last_robot_speed_xy = robot_horizontal_speed_from_env(self._env, env_idx=0)
            except Exception:
                self._last_robot_speed_xy = float("inf")
            try:
                self._last_box_nominal_x = box_nominal_x_from_env(self._env, env_idx=0)
            except Exception:
                pass
            return
        if obs is not None:
            speed = self._robot_speed_xy_from_proprio(obs)
            if speed is not None:
                self._last_robot_speed_xy = speed

    def _nav_handoff_ready(self) -> bool:
        if not self._phase_handoff_enabled:
            return False
        if self._sim_step >= self._nav_max_steps:
            return True
        nav_t = self._nav_elapsed_s()
        slow = self._last_robot_speed_xy <= self._nav_handoff_speed
        return slow and nav_t > self._nav_handoff_min_runtime_s

    def _predict_nav(self, obs, current_score):
        assert self._nav is not None
        self._update_nav_handoff_progress(obs)
        if self._nav_handoff_ready():
            self._nav_done = True
            self._capture_handoff_baselines()
            if self._sim_step >= self._nav_max_steps:
                reason = f"timeout @ step {self._sim_step}"
            else:
                reason = (
                    f"nav_t={self._nav_elapsed_s():.1f}s>"
                    f"{self._nav_handoff_min_runtime_s:.1f}s "
                    f"and speed_xy={self._last_robot_speed_xy:.3f}<={self._nav_handoff_speed:.2f}m/s"
                )
            speed_src = "env_w" if self._env is not None else "proprio_xy"
            box_note = ""
            if self._last_box_nominal_x is not None and not math.isnan(self._last_box_nominal_x):
                box_note = f", box_nominal_x={self._last_box_nominal_x:+.3f}"
            self._on_teleop_to_pit_handoff(obs)
            self._phase = "pit"
            print(
                f"[AlgSolution-Pit] nav finished ({reason}, src={speed_src}{box_note}) "
                f"@ sim_step={self._sim_step} → pit student",
                flush=True,
            )
            return self._predict_pit(obs, current_score)
        resp = self._nav.predicts(obs, current_score)
        self._sync_proprio_history_from_obs(obs)
        self._nav_phase_steps += 1
        self._sim_step += 1
        return resp

    def _predict_pit(self, obs, current_score):
        del current_score
        if self.policy is None:
            raise RuntimeError("pit student not loaded; omit --nav_only or pass --pit_ckpt")
        self._last_obs_mgr = self._uses_obs_manager(obs)
        policy_obs = self._build_policy_obs(obs)
        with torch.inference_mode():
            leg_action = self.policy.act_inference(policy_obs)
        if self._pit_warmup_remaining > 0:
            done = max(1, int(self._pit_handoff_warmup_total))
            alpha = 1.0 - float(self._pit_warmup_remaining - 1) / float(done)
            alpha = float(max(0.0, min(1.0, alpha)))
            baseline = self._handoff_leg_baseline.to(
                device=leg_action.device, dtype=leg_action.dtype
            )
            if baseline.shape[0] != leg_action.shape[0]:
                baseline = baseline.expand(leg_action.shape[0], -1)
            leg_action = (1.0 - alpha) * baseline + alpha * leg_action
            self._pit_warmup_remaining -= 1
            if self._pit_warmup_remaining == 0:
                print("[AlgSolution-Pit] handoff warmup done, full pit actions", flush=True)
        self._last_leg_action = leg_action.detach().clone()
        self._step_count += 1
        self._sim_step += 1
        self._update_history(policy_obs["proprio"].detach())

        if "proprio_history" in obs and "depth" in obs and "marg_proprio" not in obs:
            action_dim = self.LEG_ACTION_DIM
        else:
            proprio = obs["proprio"]
            if isinstance(proprio, torch.Tensor):
                pshape = proprio.shape[-1]
            else:
                pshape = len(proprio[0]) if proprio else 72
            action_dim = (int(pshape) - 12) // 3

        action_env = self._to_env_action(leg_action, action_dim, mgr_proprio=self._uses_obs_manager(obs))
        return {"action": action_env.detach().cpu().numpy().tolist(), "giveup": False}

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Platform 480x640 (or sim matching) -> policy 24x32; same as demo/server.py deploy."""
        return prep_depth(
            x.to(self.device),
            image_h=self.image_h,
            image_w=self.image_w,
            depth_max=self.depth_max,
        )

    def _build_depth_flat(self, obs: dict, batch: int) -> torch.Tensor:
        expected = self.img_channels * self.image_h * self.image_w
        # Sim + obs_manager: marg_depth_flat matches pit DAgger train (camera buffer → prep_depth).
        if "depth" in obs and not self._platform_deploy:
            depth = self._tensor_obs(obs["depth"])
            if self.head_depth_only and depth.shape[-1] == expected:
                return depth
            if self.head_depth_only and depth.shape[-1] == 2 * expected:
                return depth[:, :expected]
            if not self.head_depth_only and depth.shape[-1] == 2 * expected:
                return depth

        image_obs = obs.get("image", {})
        if isinstance(image_obs, dict):
            head_depth = image_obs.get("head_depth")
            if head_depth is None:
                head_depth = image_obs.get("video_depth")
            if head_depth is not None:
                head = self._prep_depth(head_depth).reshape(batch, -1)
                if self.head_depth_only:
                    return head
                ee_depth = image_obs.get("ee_depth")
                if ee_depth is None:
                    raise RuntimeError("depth student requires ee_depth in obs['image'].")
                ee = self._prep_depth(ee_depth).reshape(batch, -1)
                return torch.cat([head, ee], dim=-1)

        raise RuntimeError(
            "depth missing: need obs_manager obs['depth'] (sim) or obs['image'] head_depth "
            f"(platform {self.platform_cam_h}x{self.platform_cam_w}→{self.image_h}x{self.image_w})."
        )

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
        if self._uses_obs_manager(obs):
            return self._build_policy_obs_from_obs_manager(obs)

        if "proprio_history" in obs and "depth" in obs and "marg_proprio" not in obs:
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

        current = self._marg_proprio_from_sim(
            zero_last_action=(self._step_count == 0 and self._phase == "pit")
        )
        if current is None:
            current = self._marg_proprio_from_platform(proprio_platform, action_dim)
        history = self._proprio_history.reshape(current.shape[0], MARG_HISTORY_DIM)
        depth = self._build_depth_flat(obs, current.shape[0])
        return {"proprio": current, "proprio_history": history, "depth": depth}

    def _to_env_action(
        self, leg_action: torch.Tensor, action_dim: int, *, mgr_proprio: bool = False
    ) -> torch.Tensor:
        if action_dim <= self.LEG_ACTION_DIM:
            return leg_action
        num_envs = leg_action.shape[0]
        leg_env = leg_action * self._pit_to_taskd_leg_scale.to(device=leg_action.device, dtype=leg_action.dtype)
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_env
        arm_idx = [i for i in self.arm_joint_indices if i < action_dim]
        if arm_idx:
            if mgr_proprio and self.pit_edge_only:
                arm = self._handoff_arm_action
            elif mgr_proprio:
                arm = self.arm_default_action
            else:
                arm = self._handoff_arm_action
            arm = arm.to(device=self.device, dtype=torch.float32)
            if arm.shape[0] != num_envs:
                arm = arm.expand(num_envs, -1)
            action_env[:, arm_idx] = arm[:, : len(arm_idx)]
        return action_env

    def predicts(self, obs, current_score):
        try:
            if self._nav is not None and not self._nav_done:
                return self._predict_nav(obs, current_score)
            if self._replayer is not None and not self._teleop_done:
                return self._predict_teleop(obs, current_score)
            return self._predict_pit(obs, current_score)
        except Exception:
            import traceback

            print("[AlgSolution-Pit] predicts FAILED:\n" + traceback.format_exc(), flush=True)
            raise

    def get_video_overlay_lines(self) -> list[str]:
        if self._phase == "nav" and not self._nav_done and self._nav is not None:
            depth_tag = (
                f"480x640→{self._nav.image_h}x{self._nav.image_w}"
                if getattr(self, "_use_platform_cam", True)
                else f"{self._nav.image_h}x{self._nav.image_w}"
            )
            lines = [f"phase=nav step={self._sim_step}", f"depth={depth_tag}"]
            lines.append(
                f"speed_xy={self._last_robot_speed_xy:.3f} "
                f"nav_t={self._nav_elapsed_s():.1f}/{self._nav_handoff_min_runtime_s:.1f}s"
            )
            if not math.isnan(self._last_box_nominal_x):
                lines.append(f"box_nominal_x={self._last_box_nominal_x:+.2f}")
            lines.extend(self._nav.overlay_lines())
            return lines
        if self._phase == "teleop" and not self._teleop_done:
            vx, vy, wz = self._last_teleop_cmd
            return [
                f"phase=teleop t={self._sim_time():.2f}/{self._replayer.total_duration:.2f}s",
                f"cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f})",
            ]
        act_norm = float(self._last_leg_action.norm().item())
        warm = int(self._pit_warmup_remaining)
        warm_s = f" warmup_left={warm}" if warm > 0 else ""
        obs_tag = "mgr" if self._last_obs_mgr else "manual"
        depth_tag = (
            f"480x640→{self.image_h}x{self.image_w}"
            if getattr(self, "_use_platform_cam", True)
            else f"{self.image_h}x{self.image_w}"
        )
        return [
            f"phase=pit ckpt={os.path.basename(self._student_ckpt_path)} obs={obs_tag}",
            f"depth={depth_tag} cmd_vx={self.command_vx:.2f} pit_step={self._step_count}{warm_s}",
            f"act_norm={act_norm:.3f}",
        ]
