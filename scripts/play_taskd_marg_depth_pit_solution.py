"""Play Task D MARG depth pit BC student via demo/solution_marg_depth_pit.py (deploy path).

NOTE: For teleop traj → pit student, prefer the standard play path (fewer env bugs):

  python scripts/play_taskd_teleop_pit.py --video --video_length 600

  # or
  python scripts/play_atec_task.py --task ATEC-TaskD-B2Piper --teleop_traj ... --pit_ckpt ... --video
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play pit BC student through AlgSolution (solution_marg_depth_pit.py)."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rsl_rl/taskd_marg_depth_pit_dagger_b2piper/2026-06-19_04-34-32/model_800.pt",
)
parser.add_argument("--pit_width_min", type=float, default=0.4)
parser.add_argument("--pit_width_max", type=float, default=1.4)
parser.add_argument("--pit_curriculum_levels", type=int, default=11)
parser.add_argument("--pit_level", type=int, default=10)
parser.add_argument("--pit_curriculum", action="store_true")
parser.add_argument("--command_vx", type=float, default=0.6)
parser.add_argument("--spawn_x_offset", type=float, default=0.0)
parser.add_argument("--spawn_local_x", type=float, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument(
    "--video_length",
    type=int,
    default=600,
    help="Pit-student phase length in steps when --traj (full session = teleop + this). "
    "Otherwise total recorded steps after --warmup_steps.",
)
parser.add_argument("--video_dir", type=str, default=None)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=6,
    help="Steps before RecordVideo starts. Default 6 = MARG proprio history length; "
    "use 0 to record from spawn (first ~6 steps use zero history).",
)
parser.add_argument("--policy_img_h", type=int, default=24)
parser.add_argument("--policy_img_w", type=int, default=32)
parser.add_argument("--sim_camera_h", type=int, default=24)
parser.add_argument("--sim_camera_w", type=int, default=32)
parser.add_argument("--depth_only", action="store_true", default=True)
parser.add_argument("--depth_max", type=float, default=5.0)
parser.add_argument("--tiled_cameras", action="store_true", default=True)
parser.add_argument(
    "--with_box",
    action="store_true",
    default=True,
    help="Spawn Task D push box beside robot (platform layout, default on).",
)
parser.add_argument(
    "--no_box",
    action="store_true",
    help="Disable Task D box (overrides --with_box).",
)
parser.add_argument(
    "--print_score",
    action="store_true",
    default=True,
    help="Print env reward + Task D platform score during play (default on).",
)
parser.add_argument(
    "--score_interval",
    type=int,
    default=50,
    help="Print score every N steps when --print_score (default 50).",
)
parser.add_argument(
    "--traj",
    type=str,
    default=None,
    help="Teleop trajectory JSON: replay with policy.pt before pit student (full Task D env).",
)
parser.add_argument(
    "--ll_policy",
    type=str,
    default=None,
    help="JIT locomotion policy for --traj replay (default: demo/policy.pt).",
)
parser.add_argument(
    "--replay_delay",
    type=float,
    default=None,
    help="Sim seconds of zero cmd before traj playback (default: JSON replay_delay_s or 2).",
)
parser.add_argument("--teleop_vx_min", type=float, default=-4.0)
parser.add_argument("--teleop_vx_max", type=float, default=4.0)
parser.add_argument("--teleop_vy_max", type=float, default=2.0)
parser.add_argument("--teleop_wz_max", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.dict import print_dict

import atec_rl_lab.tasks  # noqa: F401
import atec_rl_lab.train  # noqa: F401
from atec_rl_lab.tasks.task_d.env_cfg import TASK_D_BOX_SPAWN_LOCAL
from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import (
    attach_dagger_depth_obs,
    configure_pit_e2e_cameras,
    configure_pit_e2e_dagger_env_cfg,
    pit_env_wants_box,
)
from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import apply_play_spawn, configure_play_terrain
from demo.solution_marg_depth_pit import AlgSolution
from demo.teleop_controller import TaskDTeleopController
from teleop_extras import TrajectoryReplayer, load_teleop_trajectory, run_teleop_replay


class _TaskDPlatformScoreTracker:
    """Mirror Task D platform scoring: RewardCrossX (-1.4->+2, 2.0->+20) + box ranges (+14 each)."""

    _CROSS_THRESHOLDS = (-1.4, 2.0)
    _CROSS_VALUES = (2.0, 20.0)
    _BOX_X_RANGES = ((-0.7, 0.7), (-1.4, -0.7))
    _BOX_VALUE = 14.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.score = 0.0
        self._cross_given = [False, False]
        self._box_given = [False, False]

    def update(self, unwrapped, *, with_box: bool) -> float:
        from atec_rl_lab.tasks.task_d.mdp.env_origin import task_d_env_origin_xy, task_d_nominal_x_to_world

        robot = unwrapped.scene["robot"]
        root_x = robot.data.root_pos_w[0, 0]
        env_origin_x, _ = task_d_env_origin_xy(unwrapped)
        ox = env_origin_x[0]

        for i, (nominal_th, value) in enumerate(zip(self._CROSS_THRESHOLDS, self._CROSS_VALUES)):
            th_world = task_d_nominal_x_to_world(
                torch.tensor(float(nominal_th), device=root_x.device),
                ox.unsqueeze(0),
            )[0]
            if root_x > th_world and not self._cross_given[i]:
                self._cross_given[i] = True
                self.score += float(value)

        if with_box:
            try:
                box = unwrapped.scene["box"]
            except KeyError:
                box = None
            if box is not None:
                box_x = box.data.root_pos_w[0, 0]
                for i, (x_min, x_max) in enumerate(self._BOX_X_RANGES):
                    mn_w = task_d_nominal_x_to_world(
                        torch.tensor(float(x_min), device=box_x.device),
                        ox.unsqueeze(0),
                    )[0]
                    mx_w = task_d_nominal_x_to_world(
                        torch.tensor(float(x_max), device=box_x.device),
                        ox.unsqueeze(0),
                    )[0]
                    lo = torch.minimum(mn_w, mx_w)
                    hi = torch.maximum(mn_w, mx_w)
                    if box_x >= lo and box_x <= hi and not self._box_given[i]:
                        self._box_given[i] = True
                        self.score += self._BOX_VALUE

        return self.score


def _robot_local_x(unwrapped) -> float:
    robot = unwrapped.scene["robot"]
    return float((robot.data.root_pos_w[0, 0] - unwrapped.scene.env_origins[0, 0]).item())


_LEG_JOINT_NAMES = [
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
]


def _leg_joint_ids(robot) -> list[int]:
    return [robot.data.joint_names.index(name) for name in _LEG_JOINT_NAMES]


def _build_platform_obs(unwrapped, device: str) -> dict:
    """Build play_atec_task-style obs: 12 leg DOF only (matches MARG pit train + action_dim=12)."""
    robot = unwrapped.scene["robot"]
    leg_ids = _leg_joint_ids(robot)
    lin_vel = robot.data.root_lin_vel_b[:, :3]
    ang_vel = robot.data.root_ang_vel_b[:, :3]
    cmd = unwrapped.command_manager.get_command("base_velocity")[:, :3]
    gravity = robot.data.projected_gravity_b

    joint_pos = robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids]
    joint_vel = robot.data.joint_vel[:, leg_ids]
    last_action = unwrapped.action_manager.action
    if last_action.shape[-1] != len(leg_ids):
        last_action = last_action[:, : len(leg_ids)]

    proprio = torch.cat([lin_vel, ang_vel, cmd, gravity, joint_pos, joint_vel, last_action], dim=-1)

    head_cam = unwrapped.scene["head_camera"]
    ee_cam = unwrapped.scene["ee_camera"]
    head_depth = head_cam.data.output["depth"].to(device)
    ee_depth = ee_cam.data.output["depth"].to(device)
    return {
        "proprio": proprio,
        "image": {"head_depth": head_depth, "ee_depth": ee_depth},
    }


def _build_taskd_play_env_cfg(args_cli):
    """Full Task D env (box + pit) for teleop replay → pit student handoff."""
    from isaaclab_tasks.utils import parse_env_cfg

    num_envs = int(args_cli.num_envs)
    env_cfg = parse_env_cfg(
        "ATEC-TaskD-B2Piper",
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not bool(getattr(args_cli, "disable_fabric", False)),
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.pit_width_range = (float(args_cli.pit_width_min), float(args_cli.pit_width_max))
    env_cfg.pit_curriculum_levels = int(args_cli.pit_curriculum_levels)
    configure_play_terrain(
        env_cfg,
        pit_level=int(args_cli.pit_level),
        use_pit_curriculum=bool(args_cli.pit_curriculum),
    )
    _trim_taskd_play_sensors(env_cfg)
    spawn_local = apply_play_spawn(
        env_cfg,
        spawn_x_offset=float(args_cli.spawn_x_offset),
        spawn_local_x=args_cli.spawn_local_x,
    )
    if hasattr(env_cfg.observations, "policy") and env_cfg.observations.policy is not None:
        env_cfg.observations.policy.enable_corruption = False
    return env_cfg, spawn_local, True


def _trim_taskd_play_sensors(env_cfg) -> None:
    """Reduce GPU/RAM: drop LiDAR + obs-manager image group; keep scene depth cameras for policy/video."""
    if hasattr(env_cfg, "scene"):
        env_cfg.scene.lidar_sensor = None
        env_cfg.scene.ee_dual_camera = None
    if hasattr(env_cfg, "observations"):
        env_cfg.observations.extero = None
        env_cfg.observations.image = None
    print(
        "[PitSolutionPlay] Task D traj play: LiDAR off, obs image group off "
        "(scene head/ee depth cameras kept for pit student + video)",
        flush=True,
    )


def _build_pit_play_env_cfg(args_cli):
    """Pit-only locomotion env (no teleop replay)."""
    env_cfg = configure_pit_e2e_dagger_env_cfg(args_cli)
    env_cfg.command_lin_vel_x_min = float(args_cli.command_vx)
    env_cfg.command_lin_vel_x_max = float(args_cli.command_vx)
    env_cfg.command_curriculum_start_fraction = 1.0
    env_cfg.apply_command_config()
    configure_play_terrain(
        env_cfg,
        pit_level=int(args_cli.pit_level),
        use_pit_curriculum=bool(args_cli.pit_curriculum),
    )
    spawn_local = apply_play_spawn(
        env_cfg,
        spawn_x_offset=float(args_cli.spawn_x_offset),
        spawn_local_x=args_cli.spawn_local_x,
    )
    with_box = pit_env_wants_box(args_cli)
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.pit_width_levels = None
    if hasattr(env_cfg.observations, "policy") and env_cfg.observations.policy is not None:
        env_cfg.observations.policy.enable_corruption = False
    return env_cfg, spawn_local, with_box


def _attach_play_cameras(env_cfg, args_cli, *, attach_depth_obs: bool) -> None:
    decimation = int(getattr(env_cfg, "decimation", 4))
    sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
    phys_dt = decimation * sim_dt
    num_envs = int(getattr(env_cfg.scene, "num_envs", 1))
    use_tiled = bool(args_cli.tiled_cameras) and num_envs > 1
    if bool(args_cli.tiled_cameras) and not use_tiled:
        print("[PitSolutionPlay] num_envs=1: using CameraCfg instead of TiledCameraCfg (lower GPU use)", flush=True)
    configure_pit_e2e_cameras(
        env_cfg,
        camera_height=int(args_cli.sim_camera_h),
        camera_width=int(args_cli.sim_camera_w),
        depth_only=bool(args_cli.depth_only),
        tiled=use_tiled,
        camera_far_clip=5.0,
        update_period=phys_dt,
    )
    if attach_depth_obs:
        attach_dagger_depth_obs(
            env_cfg,
            policy_h=int(args_cli.policy_img_h),
            policy_w=int(args_cli.policy_img_w),
            depth_max=float(args_cli.depth_max),
            depth_only=bool(args_cli.depth_only),
        )


def _env_step_dt(env_cfg) -> float:
    decimation = int(getattr(env_cfg, "decimation", 4))
    sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
    return decimation * sim_dt


def _load_traj_replayer(args_cli, traj_path: str) -> tuple[dict, TrajectoryReplayer, float]:
    traj_data = load_teleop_trajectory(traj_path)
    replay_delay = (
        float(args_cli.replay_delay)
        if args_cli.replay_delay is not None
        else float(traj_data.get("replay_delay_s", 2.0))
    )
    replayer = TrajectoryReplayer(traj_data["samples"], replay_delay_s=replay_delay)
    return traj_data, replayer, replay_delay


def _resolve_video_settings(args_cli, *, use_traj: bool, replayer: TrajectoryReplayer | None, step_dt: float):
    """Return (warmup_steps, record_length, pit_max_steps) for RecordVideo + main loop."""
    warmup = max(0, int(args_cli.warmup_steps))
    pit_steps = int(args_cli.video_length)
    if not args_cli.video:
        return warmup, None, None

    if use_traj and replayer is not None:
        if warmup == 6:
            warmup = 0
        teleop_steps = int(round(replayer.total_duration / step_dt)) + 10
        record_length = teleop_steps + pit_steps
        print(
            f"[PitSolutionPlay] full-session video: warmup={warmup}, "
            f"teleop~{teleop_steps} + pit {pit_steps} = {record_length} frames",
            flush=True,
        )
        return warmup, record_length, pit_steps

    record_length = warmup + pit_steps
    return warmup, record_length, pit_steps


def main():
    device = args_cli.device if args_cli.device else "cuda"
    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    os.environ["PIT_STUDENT_CKPT"] = ckpt
    os.environ["PIT_COMMAND_VX"] = str(float(args_cli.command_vx))

    use_traj = bool(args_cli.traj)
    if use_traj:
        traj_path = os.path.abspath(args_cli.traj)
        if not os.path.isfile(traj_path):
            raise FileNotFoundError(f"Trajectory not found: {traj_path}")
        env_cfg, spawn_local, with_box = _build_taskd_play_env_cfg(args_cli)
        gym_task = "ATEC-TaskD-B2Piper"
    else:
        env_cfg, spawn_local, with_box = _build_pit_play_env_cfg(args_cli)
        gym_task = "ATEC-TaskD-PitLocomotion-B2Piper-v0"

    box_local = tuple(float(v) for v in TASK_D_BOX_SPAWN_LOCAL) if with_box else None
    _attach_play_cameras(env_cfg, args_cli, attach_depth_obs=not use_traj)

    traj_data = None
    replayer = None
    replay_delay = 2.0
    step_dt = _env_step_dt(env_cfg)
    if use_traj:
        traj_data, replayer, replay_delay = _load_traj_replayer(args_cli, traj_path)

    record_video = bool(args_cli.video)
    video_warmup, video_record_length, pit_max_steps = _resolve_video_settings(
        args_cli,
        use_traj=use_traj,
        replayer=replayer,
        step_dt=step_dt,
    )

    render_mode = "rgb_array" if record_video else None
    env = gym.make(gym_task, cfg=env_cfg, render_mode=render_mode)

    if record_video:
        video_dir = args_cli.video_dir or os.path.join(os.path.dirname(ckpt), "videos", "solution_play")
        video_dir = os.path.abspath(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": (lambda w: lambda step: step == w)(video_warmup),
            "video_length": int(video_record_length),
            "disable_logger": True,
        }
        print("[PitSolutionPlay] Recording video:", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print(f"[PitSolutionPlay] Loading AlgSolution, ckpt={ckpt}", flush=True)
    solution = AlgSolution()
    solution.set_device(device)

    unwrapped = env.unwrapped
    obs, _ = env.reset()

    if use_traj:
        assert traj_data is not None and replayer is not None
        ll_policy = args_cli.ll_policy or traj_data.get("ll_policy")
        teleop = TaskDTeleopController(
            policy_path=ll_policy,
            device=device,
            vx_min=float(args_cli.teleop_vx_min),
            vx_max=float(args_cli.teleop_vx_max),
            vy_max=float(args_cli.teleop_vy_max),
            wz_max=float(args_cli.teleop_wz_max),
        )
        print(
            f"[PitSolutionPlay] teleop replay: {replayer.num_samples} samples, "
            f"traj={replayer.duration:.2f}s, delay={replay_delay:.2f}s, "
            f"recorded_score={traj_data.get('final_score')}",
            flush=True,
        )
        obs, teleop_time, teleop_ok = run_teleop_replay(
            env,
            teleop,
            replayer,
            simulation_app=simulation_app,
            real_time=bool(args_cli.real_time),
            headless=bool(args_cli.headless),
            device=device,
            obs=obs,
        )
        del obs
        print(
            f"[PitSolutionPlay] teleop finished (ok={teleop_ok}, t={teleop_time:.2f}s); "
            f"starting pit student ckpt={os.path.basename(ckpt)}",
            flush=True,
        )
    else:
        del obs

    solution.reset()

    dt = unwrapped.step_dt
    steps = 0
    max_steps = pit_max_steps if record_video and use_traj else (
        (video_warmup + int(args_cli.video_length)) if record_video else None
    )
    total_env_reward = 0.0
    platform_score_tracker = _TaskDPlatformScoreTracker()
    score_interval = max(1, int(args_cli.score_interval))

    if record_video and video_warmup > 0 and not use_traj:
        print(
            f"[PitSolutionPlay] Video starts after warmup step {video_warmup} "
            f"(use --warmup_steps 0 to record from spawn)",
            flush=True,
        )
    elif record_video and use_traj:
        print(
            f"[PitSolutionPlay] Video records full teleop→pit session from step {video_warmup}",
            flush=True,
        )
    print(
        f"[PitSolutionPlay] command_vx={args_cli.command_vx}, spawn_local_x={float(spawn_local[0]):.3f}, "
        f"box={'on @ ' + str(tuple(round(v, 3) for v in box_local)) if box_local else 'off'}, "
        f"pit_width={env_cfg.pit_width_range}, pit_level={args_cli.pit_level}, "
        f"depth={args_cli.policy_img_h}x{args_cli.policy_img_w}"
        f"{', teleop→pit' if use_traj else ''}",
        flush=True,
    )

    while simulation_app.is_running():
        t0 = time.time()
        platform_obs = _build_platform_obs(unwrapped, device)
        with torch.inference_mode():
            resp = solution.predicts(platform_obs, 0.0)
        actions = torch.tensor(resp["action"], dtype=torch.float32, device=unwrapped.device).view(
            unwrapped.num_envs, -1
        )
        obs, reward, terminated, truncated, info = env.step(actions)
        del obs

        sim_dt = float(info.get("Step_dt", dt)) if isinstance(info, dict) else float(dt)
        if isinstance(reward, torch.Tensor):
            total_env_reward += reward.mean().item() / sim_dt
        else:
            total_env_reward += float(reward) / sim_dt
        platform_score = platform_score_tracker.update(unwrapped, with_box=with_box)

        if args_cli.print_score and (steps % score_interval == 0 or args_cli.debug):
            robot = unwrapped.scene["robot"]
            vx = float(robot.data.root_lin_vel_w[0, 0].item())
            root_x = float(robot.data.root_pos_w[0, 0].item())
            local_x = _robot_local_x(unwrapped)
            msg = (
                f"[PitSolutionPlay] step={steps} env_reward={total_env_reward:.2f} "
                f"platform_score={platform_score:.1f} robot_x={root_x:.3f} local_x={local_x:.3f} vx={vx:.3f}"
            )
            if args_cli.debug:
                act_norm = float(actions[0].norm().item())
                msg += f" act_norm={act_norm:.3f}"
            print(msg, flush=True)

        steps += 1
        if terminated.any() or truncated.any():
            term_names = []
            tm = getattr(unwrapped, "termination_manager", None)
            if tm is not None:
                term_list = tm.active_terms if isinstance(tm.active_terms, list) else list(tm.active_terms)
                for name in term_list:
                    try:
                        if bool(tm.get_term(name)[0].item()):
                            term_names.append(name)
                    except Exception:
                        pass
            if truncated.any() and "time_out" not in term_names:
                term_names.append("time_out")
            print(
                f"[PitSolutionPlay] episode end step={steps} terms={term_names or ['unknown']} "
                f"env_reward={total_env_reward:.2f} platform_score={platform_score:.1f} "
                f"cross={platform_score_tracker._cross_given} box={platform_score_tracker._box_given}",
                flush=True,
            )
            if record_video:
                break
            obs, _ = env.reset()
            del obs
            solution.reset()
            total_env_reward = 0.0
            platform_score_tracker.reset()

        if max_steps is not None and steps >= max_steps:
            print(
                f"[PitSolutionPlay] reached pit max_steps={max_steps} "
                f"env_reward={total_env_reward:.2f} platform_score={platform_score:.1f}",
                flush=True,
            )
            break

        if args_cli.real_time:
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
