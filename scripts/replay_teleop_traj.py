# Replay a recorded Task D teleop trajectory (JSON from teleop_task_d.py).

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay Task D teleop trajectory JSON in simulation.")
parser.add_argument(
    "--traj",
    type=str,
    required=True,
    help="Path to teleop trajectory JSON (logs/teleop_traj/*.json).",
)
parser.add_argument("--task", type=str, default=None, help="Override task id (default: read from JSON).")
parser.add_argument("--ll_policy", type=str, default=None, help="Override low-level policy (default: read from JSON).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (keep 1).")
parser.add_argument("--vx_min", type=float, default=-4.0, help="Min forward velocity (m/s).")
parser.add_argument("--vx_max", type=float, default=4.0, help="Max forward velocity (m/s).")
parser.add_argument("--vy_max", type=float, default=2.0, help="Max lateral velocity (m/s).")
parser.add_argument("--wz_max", type=float, default=1.0, help="Max yaw rate (rad/s).")
parser.add_argument(
    "--real-time",
    dest="real_time",
    action="store_true",
    default=True,
    help="Pace simulation to real-time (default: on).",
)
parser.add_argument(
    "--no-real-time",
    dest="real_time",
    action="store_false",
    help="Run sim as fast as possible.",
)
parser.add_argument(
    "--compare",
    action="store_true",
    default=False,
    help="At each recorded sample time, print pose error vs original trajectory.",
)
parser.add_argument(
    "--replay-delay",
    type=float,
    default=None,
    help="Sim seconds to wait with zero cmd before playback (default: JSON replay_delay_s or 2).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from demo.teleop_controller import TaskDTeleopController  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from teleop_extras import (  # noqa: E402
    SpawnPointMarker,
    TrajectoryReplayer,
    load_teleop_trajectory,
    read_robot_spawn_w,
    run_teleop_replay,
)


def _disable_heavy_sensors(env_cfg) -> None:
    if hasattr(env_cfg, "scene"):
        env_cfg.scene.head_camera = None
        env_cfg.scene.ee_camera = None
        env_cfg.scene.ee_dual_camera = None
        env_cfg.scene.lidar_sensor = None
    if hasattr(env_cfg, "observations"):
        env_cfg.observations.extero = None
        env_cfg.observations.image = None
    print("[replay] proprio-only sensors (fast reset).", flush=True)


def main() -> None:
    traj_data = load_teleop_trajectory(args_cli.traj)
    task = args_cli.task or traj_data.get("task", "ATEC-TaskD-B2Piper")
    ll_policy = args_cli.ll_policy or traj_data.get("ll_policy")
    replay_delay = (
        float(args_cli.replay_delay)
        if args_cli.replay_delay is not None
        else float(traj_data.get("replay_delay_s", 2.0))
    )
    replayer = TrajectoryReplayer(traj_data["samples"], replay_delay_s=replay_delay)

    if "TaskD" not in task:
        raise ValueError(f"--task must be a Task D env, got {task!r}")

    print(
        f"[replay] loaded {replayer.num_samples} samples, traj={replayer.duration:.2f}s, "
        f"delay={replay_delay:.2f}s, total={replayer.total_duration:.2f}s, "
        f"recorded_score={traj_data.get('final_score')}",
        flush=True,
    )
    print(f"[replay] task={task}", flush=True)
    if ll_policy:
        print(f"[replay] ll_policy={ll_policy}", flush=True)

    controller = TaskDTeleopController(
        policy_path=ll_policy,
        device=args_cli.device,
        vx_min=args_cli.vx_min,
        vx_max=args_cli.vx_max,
        vy_max=args_cli.vy_max,
        wz_max=args_cli.wz_max,
    )
    if hasattr(controller, "set_device") and args_cli.device is not None:
        controller.set_device(args_cli.device)

    env_cfg = parse_env_cfg(
        task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _disable_heavy_sensors(env_cfg)

    env = gym.make(task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    print("[replay] env.reset() ...", flush=True)
    obs, _ = env.reset()

    spawn_w = traj_data.get("spawn_robot_w") or read_robot_spawn_w(env)
    if not args_cli.headless:
        SpawnPointMarker(device=args_cli.device or "cpu").mark(spawn_w)
    print(f"[replay] holding zero cmd for {replay_delay:.2f}s, then playing trajectory.", flush=True)

    try:
        run_teleop_replay(
            env,
            controller,
            replayer,
            simulation_app=simulation_app,
            real_time=bool(args_cli.real_time),
            compare=bool(args_cli.compare),
            headless=bool(args_cli.headless),
            device=args_cli.device,
            obs=obs,
        )
    finally:
        env.close()
        print("[replay] closed.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
