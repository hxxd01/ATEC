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

from atec_rl_lab.tasks.task_d.env_cfg import (
    TASK_D_NAV_DEPTH_MAX,
    TASK_D_PLATFORM_CAMERA_FAR,
    TaskDEnvB2Cfg,
    apply_task_d_camera_depth_clip,
)
from atec_rl_lab.tasks.task_d.locomotion.mdp.events import (
    PIT_BOX_SPAWN_X_JITTER_DOWN,
    PIT_BOX_SPAWN_Y_JITTER,
    TASK_D_PIT_PUSHED_BOX_LOCAL,
    task_d_pit_marg_spawn_local,
)
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


def resolve_pit_camera_far_clip(cli_value: float | None) -> float:
    """Match ``train_nav_taskd_student._resolve_camera_far_clip`` (default platform 50 m)."""
    return float(TASK_D_PLATFORM_CAMERA_FAR if cli_value is None else cli_value)


def resolve_pit_depth_pipeline(args) -> tuple[int, int, int, int, str]:
    """Match ``train_nav_taskd_student._resolve_depth_pipeline`` (native vs platform depth)."""
    policy_h = int(getattr(args, "policy_img_h", 24))
    policy_w = int(getattr(args, "policy_img_w", 32))
    if getattr(args, "camera_hw", None) is not None:
        side = int(args.camera_hw)
        policy_h = policy_w = side
    depth_render_h = getattr(args, "depth_render_h", None)
    depth_render_w = getattr(args, "depth_render_w", None)
    if depth_render_h is not None:
        policy_h = int(depth_render_h)
    if depth_render_w is not None:
        policy_w = int(depth_render_w)

    platform = bool(
        getattr(args, "platform_depth_train", False) or getattr(args, "pit_platform_depth", False)
    )
    if platform:
        cam_h = int(getattr(args, "platform_depth_h", 480))
        cam_w = int(getattr(args, "platform_depth_w", 640))
        return cam_h, cam_w, policy_h, policy_w, "platform"

    sim_h = getattr(args, "sim_camera_h", None)
    sim_w = getattr(args, "sim_camera_w", None)
    cam_h = int(sim_h) if sim_h is not None else policy_h
    cam_w = int(sim_w) if sim_w is not None else policy_w
    return cam_h, cam_w, policy_h, policy_w, "native"


# Light reset-time DR for pit MARG / depth student (sim-only; keep disable_dr_and_obs_noise=True).
PIT_DR_LIGHT_JOINT_SCALE = (0.5, 1.5)
PIT_DR_LIGHT_SPAWN_X_JITTER = 0.5
# One-sided spawn y toward robot right (+y is left in Task D pit coords).
PIT_DR_LIGHT_SPAWN_Y_JITTER_RIGHT = 0.5
# Teleop upright base_link z≈0.529; DR center slightly above + symmetric jitter (covers stand + small drop/settle).
PIT_DR_LIGHT_SPAWN_Z = 0.545
PIT_DR_LIGHT_SPAWN_Z_JITTER = 0.025
PIT_DR_LIGHT_YAW_RANGE = (-0.15, 0.15)
PIT_DR_LIGHT_LIN_VEL_X = (0.0, 0.35)
PIT_DR_LIGHT_LIN_VEL_Y = (-0.05, 0.05)
PIT_DR_LIGHT_ANG_VEL_Z = (-0.12, 0.12)


def _pit_dr_arg(args, name: str, default):
    val = getattr(args, name, None)
    return default if val is None else val


def apply_pit_dr_light(env_cfg, args) -> None:
    """Enable mild reset DR: leg joints, spawn xy/yaw, small initial root velocity."""
    if not bool(getattr(args, "pit_dr_light", False)):
        return
    if bool(getattr(args, "pit_sim_easy", False) or getattr(args, "sim_easy", False)):
        print("[TaskDPitDR] skipped (--pit_sim_easy / --sim_easy clears DR)", flush=True)
        return

    joint_scale = _pit_dr_arg(args, "pit_dr_joint_scale", PIT_DR_LIGHT_JOINT_SCALE)
    x_jitter = float(_pit_dr_arg(args, "pit_spawn_x_jitter", PIT_DR_LIGHT_SPAWN_X_JITTER))
    y_jitter_right = float(
        _pit_dr_arg(args, "pit_dr_spawn_y_jitter", PIT_DR_LIGHT_SPAWN_Y_JITTER_RIGHT)
    )
    spawn_z = float(_pit_dr_arg(args, "pit_dr_spawn_z", PIT_DR_LIGHT_SPAWN_Z))
    z_jitter = float(_pit_dr_arg(args, "pit_dr_spawn_z_jitter", PIT_DR_LIGHT_SPAWN_Z_JITTER))
    yaw_range = _pit_dr_arg(args, "pit_dr_yaw_range", PIT_DR_LIGHT_YAW_RANGE)
    lin_vx = _pit_dr_arg(args, "pit_dr_lin_vel_x", PIT_DR_LIGHT_LIN_VEL_X)
    lin_vy = _pit_dr_arg(args, "pit_dr_lin_vel_y", PIT_DR_LIGHT_LIN_VEL_Y)
    ang_wz = _pit_dr_arg(args, "pit_dr_ang_vel_z", PIT_DR_LIGHT_ANG_VEL_Z)

    env_cfg.events.reset_joints_default.params["position_range"] = tuple(float(v) for v in joint_scale)
    env_cfg.events.reset_joints_default.params["velocity_range"] = (0.0, 0.0)

    params = env_cfg.events.reset_robot_task_d.params
    local_pos = params.get("local_pos")
    if local_pos is None:
        local_pos = task_d_pit_marg_spawn_local()
    params["local_pos"] = (float(local_pos[0]), float(local_pos[1]), spawn_z)
    params["local_pos_x_jitter"] = x_jitter
    params["local_pos_y_jitter"] = 0.0
    params["local_pos_y_jitter_right"] = y_jitter_right
    params["local_pos_z_jitter"] = z_jitter
    params["yaw_range"] = tuple(float(v) for v in yaw_range)
    params["lin_vel_x_range"] = tuple(float(v) for v in lin_vx)
    params["lin_vel_y_range"] = tuple(float(v) for v in lin_vy)
    params["ang_vel_z_range"] = tuple(float(v) for v in ang_wz)

    print(
        f"[TaskDPitDR] light DR: joint_scale={joint_scale}, "
        f"spawn jitter x±{x_jitter:.2f} y right [0,{y_jitter_right:.2f}] z={spawn_z:.3f}±{z_jitter:.3f}, "
        f"yaw={yaw_range}, init_vel vx={lin_vx} vy={lin_vy} wz={ang_wz}",
        flush=True,
    )


def pit_head_depth_only(args) -> bool:
    """Pit depth student: head camera only unless --ee_depth."""
    return not bool(getattr(args, "ee_depth", False))


def attach_pit_scene_cameras(env_cfg, args) -> None:
    """Attach head (+ optional ee) scene cameras; ee is None when head-only (no ee render)."""
    head_only = pit_head_depth_only(args)
    env_cfg.head_depth_only = head_only
    ref = TaskDEnvB2Cfg()
    env_cfg.scene.head_camera = copy.deepcopy(ref.scene.head_camera)
    if head_only:
        env_cfg.scene.ee_camera = None
    else:
        env_cfg.scene.ee_camera = copy.deepcopy(ref.scene.ee_camera)
    print(
        f"[TaskDMargDepthPit] scene cameras={'head only' if head_only else 'head+ee'} "
        f"(ee_camera={'off' if head_only else 'on'})",
        flush=True,
    )


def apply_pit_train_spawn(env_cfg, args) -> tuple[float, float, float]:
    """Set base spawn local_pos and optional uniform ±x jitter on each episode reset."""
    base = tuple(float(v) for v in TASK_D_ROBOT_SPAWN_LOCAL)
    spawn_local_x = getattr(args, "pit_spawn_local_x", None)
    spawn_x_offset = float(getattr(args, "pit_spawn_x_offset", 0.0))
    if spawn_local_x is not None:
        local_x = float(spawn_local_x)
    else:
        local_x = base[0] + spawn_x_offset
    local_pos = task_d_pit_marg_spawn_local(local_x=local_x)

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
    args=None,
    *,
    box_local_pos: tuple[float, float, float] | None = None,
    use_pushed_spawn: bool = True,
) -> tuple[float, float, float]:
    """Add Task D box (0.8x1.0x0.6) + reset on episode reset.

    Default: teleop pushed pose (``TASK_D_PIT_PUSHED_BOX_LOCAL``) with x jitter downward and y jitter.
    When present, the MARG height scanner also raycasts against ``{ENV_REGEX_NS}/Box``.
    """
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    import isaaclab.sim as sim_utils
    import atec_rl_lab.tasks.task_d.locomotion.mdp as task_d_loco_mdp

    if box_local_pos is not None:
        local_pos = tuple(float(v) for v in box_local_pos)
        use_pushed = False
    elif use_pushed_spawn:
        local_pos = tuple(float(v) for v in TASK_D_PIT_PUSHED_BOX_LOCAL)
        use_pushed = True
    else:
        local_pos = tuple(float(v) for v in TASK_D_BOX_SPAWN_LOCAL)
        use_pushed = False

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

    if use_pushed:
        x_down = float(
            PIT_BOX_SPAWN_X_JITTER_DOWN
            if args is None or getattr(args, "pit_box_x_jitter_down", None) is None
            else args.pit_box_x_jitter_down
        )
        y_jitter = float(
            PIT_BOX_SPAWN_Y_JITTER
            if args is None or getattr(args, "pit_box_y_jitter", None) is None
            else args.pit_box_y_jitter
        )
        env_cfg.events.reset_box_root = EventTerm(
            func=task_d_loco_mdp.reset_box_at_task_d_spawn,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "local_pos": local_pos,
                "local_pos_x_jitter_down": x_down,
                "local_pos_y_jitter": y_jitter,
            },
        )
        print(
            f"[TaskDPitBox] pushed box local_pos={local_pos}, "
            f"x jitter -[0,{x_down:.2f}] y±{y_jitter:.2f} (height_scanner includes box)",
            flush=True,
        )
    else:
        env_cfg.events.reset_box_root = EventTerm(
            func=reset_root_state_at_env_origin,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "local_pos": local_pos,
            },
        )
        print(
            f"[TaskDPitBox] default spawn box at local_pos={local_pos} "
            f"(height_scanner includes box)",
            flush=True,
        )

    from atec_rl_lab.tasks.task_d.locomotion.env_cfg import configure_pit_marg_height_scanner

    configure_pit_marg_height_scanner(env_cfg, include_box=True)
    return local_pos


def pit_env_wants_box(args) -> bool:
    """Whether pit student train/play should spawn the Task D push box (default on)."""
    if bool(getattr(args, "no_pit_box", False)) or bool(getattr(args, "no_box", False)):
        return False
    return bool(getattr(args, "with_box", getattr(args, "pit_with_box", True)))


def apply_pit_box_if_requested(env_cfg, args) -> tuple[float, float, float] | None:
    if not pit_env_wants_box(args):
        return None
    use_pushed = not bool(getattr(args, "pit_box_default_spawn", False))
    return attach_task_d_box(env_cfg, args, use_pushed_spawn=use_pushed)


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

    attach_pit_scene_cameras(env_cfg, args)
    apply_pit_train_spawn(env_cfg, args)
    apply_pit_box_if_requested(env_cfg, args)
    apply_pit_dr_light(env_cfg, args)
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

    attach_pit_scene_cameras(env_cfg, args)
    apply_pit_train_spawn(env_cfg, args)
    apply_pit_box_if_requested(env_cfg, args)
    apply_pit_dr_light(env_cfg, args)
    return env_cfg


def configure_pit_dagger_play_env_cfg(args) -> tuple:
    """Full PitLocomotion play cfg (train / DAgger play / platform ablation).

    Returns ``(env_cfg, spawn_local_xyz)`` with obs_manager proprio+history+depth attached.
    """
    from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import apply_play_spawn, configure_play_terrain

    env_cfg = configure_pit_e2e_dagger_env_cfg(args)
    cam_h, cam_w, policy_h, policy_w, depth_mode = resolve_pit_depth_pipeline(args)
    camera_far_clip = resolve_pit_camera_far_clip(getattr(args, "camera_far_clip", None))
    command_vx = getattr(args, "command_vx", None)
    if command_vx is not None:
        vx = float(command_vx)
        env_cfg.command_lin_vel_x_min = vx
        env_cfg.command_lin_vel_x_max = vx
    env_cfg.command_curriculum_start_fraction = 1.0
    env_cfg.apply_command_config()
    configure_play_terrain(
        env_cfg,
        pit_level=int(getattr(args, "pit_level", 10)),
        use_pit_curriculum=bool(getattr(args, "pit_curriculum", False)),
    )
    spawn_local = apply_play_spawn(
        env_cfg,
        spawn_x_offset=float(getattr(args, "spawn_x_offset", 0.0)),
        spawn_local_x=getattr(args, "spawn_local_x", None),
    )
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.pit_width_levels = None
    if hasattr(env_cfg.observations, "policy") and env_cfg.observations.policy is not None:
        env_cfg.observations.policy.enable_corruption = False

    decimation = int(getattr(env_cfg, "decimation", 4))
    sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
    phys_dt = decimation * sim_dt
    head_depth_only = not bool(getattr(args, "ee_depth", False))
    depth_only = bool(getattr(args, "depth_only", True))
    depth_max = float(getattr(args, "depth_max", TASK_D_NAV_DEPTH_MAX))
    configure_pit_e2e_cameras(
        env_cfg,
        camera_height=cam_h,
        camera_width=cam_w,
        depth_only=depth_only,
        tiled=bool(getattr(args, "tiled_cameras", True)),
        camera_far_clip=camera_far_clip,
        update_period=phys_dt,
        head_depth_only=head_depth_only,
    )
    attach_dagger_depth_obs(
        env_cfg,
        policy_h=policy_h,
        policy_w=policy_w,
        depth_max=depth_max,
        depth_only=depth_only,
        depth_render_h=cam_h if depth_mode == "platform" else None,
        depth_render_w=cam_w if depth_mode == "platform" else None,
        head_depth_only=head_depth_only,
    )
    cam_tag = "head" if head_depth_only else "head+ee"
    print(
        "[TaskDMargDepthPit] play env: PitLocomotion, obs=proprio+proprio_history+depth, "
        f"12D leg, cmd_vx=[{env_cfg.command_lin_vel_x_min}, {env_cfg.command_lin_vel_x_max}], "
        f"depth={depth_mode} sim={cam_h}x{cam_w} policy={policy_h}x{policy_w} ({cam_tag}), "
        f"camera_far_clip={camera_far_clip:.1f}m, depth_max={depth_max:.1f}m",
        flush=True,
    )
    return env_cfg, spawn_local


def attach_dagger_depth_obs(
    env_cfg: UnitreeB2PiperTaskDPitLocomotionMargEnvCfg,
    *,
    policy_h: int,
    policy_w: int,
    depth_max: float,
    depth_only: bool,
    depth_render_h: int | None = None,
    depth_render_w: int | None = None,
    head_depth_only: bool | None = None,
) -> None:
    """Enable depth obs group on Marg pit env (obs manager; shared by DAgger, e2e PPO, play)."""
    if head_depth_only is None:
        head_depth_only = bool(getattr(env_cfg, "head_depth_only", True))
    env_cfg.head_depth_only = bool(head_depth_only)
    env_cfg.depth_policy_h = int(policy_h)
    env_cfg.depth_policy_w = int(policy_w)
    env_cfg.depth_max = float(depth_max)
    env_cfg.depth_only = bool(depth_only)
    env_cfg.depth_render_h = depth_render_h
    env_cfg.depth_render_w = depth_render_w
    env_cfg.observations.depth = copy.deepcopy(TaskDPitMargObservationsCfg.DepthCfg())
    num_cams = 1 if head_depth_only else 2
    depth_dim = num_cams * (1 if depth_only else 4) * int(policy_h) * int(policy_w)
    cam_tag = "head" if head_depth_only else "head+ee"
    src_tag = ""
    if depth_render_h is not None and depth_render_w is not None:
        if int(depth_render_h) != int(policy_h) or int(depth_render_w) != int(policy_w):
            src_tag = f", prep_depth {int(depth_render_h)}x{int(depth_render_w)}→{int(policy_h)}x{int(policy_w)}"
    print(
        f"[TaskDMargDepthPit] depth obs via obs_manager: "
        f"{depth_dim}D ({'depth' if depth_only else 'rgb+depth'} {cam_tag} @ {policy_h}x{policy_w}{src_tag})",
        flush=True,
    )


# Alias: same obs-manager depth path for e2e PPO and DAgger.
attach_marg_depth_obs = attach_dagger_depth_obs


def apply_taskd_pit_student_command(env_cfg, command_vx: float = 0.6) -> None:
    """Fix Task D ``command_manager.base_velocity`` for pit student deploy (matches pit-loco play)."""
    vx = float(command_vx)
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (vx, vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.ranges.heading = (0.0, 0.0)
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (10.0, 10.0)
    print(
        f"[TaskDMargDepthPit] Task D command_manager base_velocity fixed vx={vx:.2f} m/s",
        flush=True,
    )


def attach_taskd_platform_marg_student_obs(
    env_cfg,
    args,
    *,
    policy_h: int | None = None,
    policy_w: int | None = None,
    depth_max: float = 5.0,
) -> None:
    """Task D B2 play: MARG proprio/history + depth via obs_manager (platform proprio kept for teleop)."""
    head_depth_only = not bool(getattr(args, "ee_depth", False))
    ph = int(policy_h if policy_h is not None else getattr(args, "pit_cam_h", 24))
    pw = int(policy_w if policy_w is not None else getattr(args, "pit_cam_w", 32))
    marg = TaskDPitMargObservationsCfg()
    env_cfg.observations.marg_proprio = copy.deepcopy(marg.proprio)
    env_cfg.observations.marg_proprio_history = copy.deepcopy(marg.proprio_history)
    attach_dagger_depth_obs(
        env_cfg,
        policy_h=ph,
        policy_w=pw,
        depth_max=float(depth_max),
        depth_only=True,
        head_depth_only=head_depth_only,
    )
    # Depth comes from marg_depth_flat; drop platform image obs group (teleop uses proprio only).
    env_cfg.observations.image = None
    print(
        "[TaskDMargDepthPit] Task D platform deploy: obs_manager marg_proprio + marg_proprio_history + depth",
        flush=True,
    )


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
    head_depth_only: bool | None = None,
) -> None:
    """Match student nav camera pipeline on pit env (head only by default; no ee render)."""
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    if head_depth_only is None:
        head_depth_only = bool(getattr(env_cfg, "head_depth_only", True))
    env_cfg.head_depth_only = bool(head_depth_only)
    if head_depth_only:
        env_cfg.scene.ee_camera = None

    apply_task_d_camera_depth_clip(env_cfg.scene, float(camera_far_clip))
    cam_cfg_cls = TiledCameraCfg if tiled else CameraCfg
    data_types = ["depth"] if depth_only else ["rgb", "depth"]
    cam_names = ("head_camera",) if head_depth_only else ("head_camera", "ee_camera")
    for cam_name in cam_names:
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
    cam_tag = "head" if head_depth_only else "head+ee"
    print(
        f"[TaskDMargDepthPit] cameras={cam_tag}, sim={camera_height}x{camera_width}, "
        f"depth_only={depth_only}, tiled={tiled}",
        flush=True,
    )


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
    if bool(getattr(env_cfg, "head_depth_only", True)):
        for term_name in ("ee_depth", "ee_dual_depth"):
            if hasattr(image_obs, term_name):
                setattr(image_obs, term_name, None)
