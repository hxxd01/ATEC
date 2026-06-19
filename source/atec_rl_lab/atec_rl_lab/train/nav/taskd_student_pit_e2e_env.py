"""Task D student pit crossing: shared depth encoder obs, direct 12-D leg actions (no nav cmd / ll_policy)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

_demo_dir = Path(__file__).resolve().parents[5] / "demo"
if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))
from depth_preprocess import prep_depth as _prep_depth_shared  # noqa: E402

from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg, apply_task_d_camera_depth_clip
from atec_rl_lab.tasks.task_d.env_cfg import (
    TASK_D_BOX_SPAWN_LOCAL,
    TASK_D_ROBOT_SPAWN_LOCAL,
    reset_root_state_at_env_origin,
)
from atec_rl_lab.tasks.task_d.locomotion.env_cfg import (
    UnitreeB2PiperTaskDPitLocomotionEnvCfg,
    UnitreeB2PiperTaskDPitLocomotionMargEnvCfg,
    TaskDPitMargObservationsCfg,
    refresh_task_d_pit_locomotion_terrain_cfg,
)
from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import _apply_sim_easy_mode


LEG_ACTION_DIM = 12
DEFAULT_PIT_SPAWN_X_JITTER = 0.0


def apply_pit_train_spawn(env_cfg, args) -> tuple[float, float, float]:
    """Set base spawn local_pos and optional uniform ±x jitter on each episode reset."""
    base = tuple(float(v) for v in TASK_D_ROBOT_SPAWN_LOCAL)
    spawn_local_x = getattr(args, "pit_spawn_local_x", None)
    spawn_x_offset = float(getattr(args, "pit_spawn_x_offset", 0.0))
    if spawn_local_x is not None:
        local_pos = (float(spawn_local_x), base[1], base[2])
    else:
        local_pos = (base[0] + spawn_x_offset, base[1], base[2])

    jitter_arg = getattr(args, "pit_spawn_x_jitter", None)
    jitter = DEFAULT_PIT_SPAWN_X_JITTER if jitter_arg is None else float(jitter_arg)
    params = env_cfg.events.reset_robot_task_d.params
    params["local_pos"] = local_pos
    params["local_pos_x_jitter"] = jitter
    if jitter > 0.0:
        print(
            f"[TaskDPitSpawn] local_x={local_pos[0]:.3f} m, uniform x jitter ±{jitter:.3f} m",
            flush=True,
        )
    else:
        print(f"[TaskDPitSpawn] local_x={local_pos[0]:.3f} m (fixed, no jitter)", flush=True)
    return local_pos


def attach_task_d_box(
    env_cfg,
    *,
    box_local_pos: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Add Task D push box (0.8x1.0x0.6) + reset on episode reset (matches platform Task D)."""
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    import isaaclab.sim as sim_utils

    local_pos = tuple(float(v) for v in (box_local_pos or TASK_D_BOX_SPAWN_LOCAL))
    env_cfg.scene.box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=local_pos),
    )
    env_cfg.events.reset_box_root = EventTerm(
        func=reset_root_state_at_env_origin,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "local_pos": local_pos,
        },
    )
    print(f"[TaskDPitPlay] Task D box enabled at local_pos={local_pos}", flush=True)
    return local_pos


def pit_env_wants_box(args) -> bool:
    """Whether pit student train/play should spawn the Task D push box (default on)."""
    if bool(getattr(args, "no_pit_box", False)) or bool(getattr(args, "no_box", False)):
        return False
    return bool(getattr(args, "with_box", getattr(args, "pit_with_box", True)))


def apply_pit_box_if_requested(env_cfg, args) -> tuple[float, float, float] | None:
    if not pit_env_wants_box(args):
        return None
    return attach_task_d_box(env_cfg)


def configure_pit_e2e_env_cfg(args) -> UnitreeB2PiperTaskDPitLocomotionMargEnvCfg:
    """Pure PPO depth student: Marg env + obs manager (same obs path as DAgger/play)."""
    sim_easy = bool(getattr(args, "pit_sim_easy", False) or getattr(args, "sim_easy", False))
    env_cfg = UnitreeB2PiperTaskDPitLocomotionMargEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.pit_width_range = (
        float(getattr(args, "pit_width_min", 0.4)),
        float(getattr(args, "pit_width_max", 1.4)),
    )
    env_cfg.pit_curriculum_levels = int(getattr(args, "pit_curriculum_levels", 11))
    env_cfg.command_lin_vel_x_min = float(getattr(args, "command_vx_min", 0.0))
    vx_max = getattr(args, "command_vx_max", None)
    if vx_max is None:
        vx_max = getattr(args, "command_vx", 4.0)
    env_cfg.command_lin_vel_x_max = float(vx_max)
    env_cfg.command_curriculum_start_fraction = float(
        getattr(args, "command_curriculum_start", 0.1)
    )
    if sim_easy:
        _apply_sim_easy_mode(env_cfg, args)
    env_cfg.apply_command_config()
    refresh_task_d_pit_locomotion_terrain_cfg(env_cfg)

    ref = TaskDEnvB2Cfg()
    env_cfg.scene.head_camera = copy.deepcopy(ref.scene.head_camera)
    env_cfg.scene.ee_camera = copy.deepcopy(ref.scene.ee_camera)
    apply_pit_train_spawn(env_cfg, args)
    apply_pit_box_if_requested(env_cfg, args)
    return env_cfg


def configure_pit_e2e_dagger_env_cfg(args) -> UnitreeB2PiperTaskDPitLocomotionMargEnvCfg:
    """Pit env with MARG height scanner (teacher) + Task-D cameras (student depth)."""
    sim_easy = bool(getattr(args, "pit_sim_easy", False) or getattr(args, "sim_easy", False))
    env_cfg = UnitreeB2PiperTaskDPitLocomotionMargEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.pit_width_range = (
        float(getattr(args, "pit_width_min", 0.4)),
        float(getattr(args, "pit_width_max", 1.4)),
    )
    env_cfg.pit_curriculum_levels = int(getattr(args, "pit_curriculum_levels", 11))
    # Frozen teacher was validated in pit_marg play at fixed vx≈0.6 m/s; match that unless overridden.
    command_vx = getattr(args, "command_vx", None)
    if command_vx is not None:
        vx = float(command_vx)
        env_cfg.command_lin_vel_x_min = vx
        env_cfg.command_lin_vel_x_max = vx
    else:
        vx_min = float(getattr(args, "command_vx_min", 0.0))
        vx_max_arg = getattr(args, "command_vx_max", None)
        vx_max = float(vx_max_arg if vx_max_arg is not None else getattr(args, "command_vx", 4.0))
        if vx_min == 0.0 and vx_max == 4.0:
            vx = 0.6
            env_cfg.command_lin_vel_x_min = vx
            env_cfg.command_lin_vel_x_max = vx
            print(
                f"[TaskDMargDepthPitDAgger] command_vx={vx:.1f} m/s (play default; "
                f"override with --command_vx or --command_vx_min/max)",
                flush=True,
            )
        else:
            env_cfg.command_lin_vel_x_min = vx_min
            env_cfg.command_lin_vel_x_max = vx_max
    env_cfg.command_curriculum_start_fraction = 1.0
    if sim_easy:
        _apply_sim_easy_mode(env_cfg, args)
    env_cfg.apply_command_config()
    refresh_task_d_pit_locomotion_terrain_cfg(env_cfg)

    ref = TaskDEnvB2Cfg()
    env_cfg.scene.head_camera = copy.deepcopy(ref.scene.head_camera)
    env_cfg.scene.ee_camera = copy.deepcopy(ref.scene.ee_camera)
    apply_pit_train_spawn(env_cfg, args)
    apply_pit_box_if_requested(env_cfg, args)
    return env_cfg


def attach_dagger_depth_obs(
    env_cfg: UnitreeB2PiperTaskDPitLocomotionMargEnvCfg,
    *,
    policy_h: int,
    policy_w: int,
    depth_max: float,
    depth_only: bool,
    depth_render_h: int | None = None,
    depth_render_w: int | None = None,
) -> None:
    """Enable depth obs group on Marg pit env (obs manager; shared by DAgger, e2e PPO, play)."""
    env_cfg.depth_policy_h = int(policy_h)
    env_cfg.depth_policy_w = int(policy_w)
    env_cfg.depth_max = float(depth_max)
    env_cfg.depth_only = bool(depth_only)
    env_cfg.depth_render_h = depth_render_h
    env_cfg.depth_render_w = depth_render_w
    env_cfg.observations.depth = copy.deepcopy(TaskDPitMargObservationsCfg.DepthCfg())
    depth_dim = 2 * (1 if depth_only else 4) * int(policy_h) * int(policy_w)
    print(
        f"[TaskDMargDepthPit] depth obs via obs_manager: "
        f"{depth_dim}D ({'depth' if depth_only else 'rgb+depth'} head+ee @ {policy_h}x{policy_w})",
        flush=True,
    )


# Alias: same obs-manager depth path for e2e PPO and DAgger.
attach_marg_depth_obs = attach_dagger_depth_obs


class TaskDStudentPitE2EEnv(gym.Wrapper):
    """Depth+proprio student obs; outputs 12 leg actions into pit locomotion env (rewards from base env)."""

    inner_steps = 1
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env: gym.Env,
        *,
        device: str = "cuda",
        image_h: int = 24,
        image_w: int = 32,
        depth_max: float = 5.0,
        depth_only: bool = True,
        depth_render_h: int | None = None,
        depth_render_w: int | None = None,
    ):
        super().__init__(env)
        self._device = device
        self._image_h = int(image_h)
        self._image_w = int(image_w)
        self._depth_render_h = int(depth_render_h) if depth_render_h is not None else self._image_h
        self._depth_render_w = int(depth_render_w) if depth_render_w is not None else self._image_w
        self._depth_max = float(depth_max)
        self._depth_only = bool(depth_only)
        self._img_channels = 1 if self._depth_only else 4
        self._student_img_flat = 2 * self._img_channels * self._image_h * self._image_w
        self._proprio_dim = 9
        self._actor_dim = self._student_img_flat + self._proprio_dim

        self.num_envs = int(self.unwrapped.num_envs)
        self.device = self._device
        self.max_episode_length = int(self.unwrapped.max_episode_length)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(LEG_ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._actor_dim,), dtype=np.float32
                ),
                "critic": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._actor_dim,), dtype=np.float32
                ),
            }
        )
        self._last_obs = None
        print(
            f"[TaskDStudentPitE2E] actor_dim={self._actor_dim} "
            f"({self._img_channels}ch head+ee @ {self._image_h}x{self._image_w}), "
            f"actions={LEG_ACTION_DIM} legs (no nav cmd / no ll_policy), "
            f"rewards/terminations=pit locomotion env",
            flush=True,
        )

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf.copy_(value)

    def _prep_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] != self._image_w or x.shape[-2] != self._image_h:
            x = F.interpolate(
                x, size=(self._image_h, self._image_w), mode="bilinear", align_corners=False
            )
        return x

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        return _prep_depth_shared(
            x,
            image_h=self._image_h,
            image_w=self._image_w,
            depth_render_h=self._depth_render_h,
            depth_render_w=self._depth_render_w,
            depth_max=self._depth_max,
        )

    def _proprio_feat(self, batch: int) -> torch.Tensor:
        robot = self.unwrapped.scene["robot"]
        lin_vel = robot.data.root_lin_vel_b[:, :3]
        ang_vel = robot.data.root_ang_vel_b[:, :3] * 0.25
        gravity = robot.data.projected_gravity_b
        return torch.cat([lin_vel, ang_vel, gravity], dim=-1)

    def _camera_tensor(self, cam_name: str, batch: int) -> torch.Tensor:
        try:
            cam = self.unwrapped.scene[cam_name]
            out = cam.data.output
            if self._depth_only:
                if "depth" not in out:
                    raise KeyError(f"{cam_name} missing depth output")
                return self._prep_depth(out["depth"].to(device=self._device))
            rgb = self._prep_rgb(out["rgb"].to(device=self._device))
            depth = self._prep_depth(out["depth"].to(device=self._device))
            return torch.cat([rgb, depth], dim=1)
        except Exception:
            return torch.zeros(
                batch,
                self._img_channels,
                self._image_h,
                self._image_w,
                device=self._device,
                dtype=torch.float32,
            )

    def _build_actor_obs(self) -> torch.Tensor:
        batch = self.num_envs
        head = self._camera_tensor("head_camera", batch).reshape(batch, -1)
        ee = self._camera_tensor("ee_camera", batch).reshape(batch, -1)
        proprio_feat = self._proprio_feat(batch)
        return torch.cat([head, ee, proprio_feat], dim=-1)

    def _obs_dict(self) -> dict[str, torch.Tensor]:
        actor = self._build_actor_obs()
        return {"policy": actor, "critic": actor}

    def get_observations(self) -> tuple[dict[str, torch.Tensor], dict]:
        return self._obs_dict(), {}

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        del obs
        self._last_obs = None
        return self._obs_dict(), info

    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = actions.to(self._device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = self.env.step(actions)
        del obs
        self._last_obs = None
        return self._obs_dict(), reward, terminated, truncated, info

    def close(self):
        return self.env.close()


def configure_pit_e2e_cameras(
    env_cfg,
    *,
    camera_height: int,
    camera_width: int,
    depth_only: bool,
    tiled: bool,
    camera_far_clip: float,
    update_period: float,
) -> None:
    """Match student nav camera pipeline on pit env."""
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    apply_task_d_camera_depth_clip(env_cfg.scene, float(camera_far_clip))
    cam_cfg_cls = TiledCameraCfg if tiled else CameraCfg
    data_types = ["depth"] if depth_only else ["rgb", "depth"]
    for cam_name in ("head_camera", "ee_camera"):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is None:
            continue
        setattr(
            env_cfg.scene,
            cam_name,
            cam_cfg_cls(
                prim_path=cam.prim_path,
                spawn=cam.spawn,
                offset=cam.offset,
                height=int(camera_height),
                width=int(camera_width),
                data_types=data_types,
                update_period=float(update_period),
            ),
        )
    if depth_only:
        align_taskd_image_obs_for_cameras(env_cfg)


def align_taskd_image_obs_for_cameras(env_cfg, *, depth_only: bool = True) -> None:
    """Remove rgb obs terms when scene cameras are depth-only (obs manager would KeyError on 'rgb')."""
    if not depth_only:
        return
    observations = getattr(env_cfg, "observations", None)
    if observations is None or getattr(observations, "image", None) is None:
        return
    image_obs = observations.image
    for term_name in ("head_rgb", "ee_rgb", "ee_dual_rgb"):
        if hasattr(image_obs, term_name):
            setattr(image_obs, term_name, None)
