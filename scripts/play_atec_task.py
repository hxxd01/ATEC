# Created by skywoodsz on 2026/02/07.

import argparse
import json
import os
import sys
import time

# Repo root contains package `demo/`; running `python scripts/...` only puts scripts/ on sys.path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from isaaclab.app import AppLauncher

from demo.solution import AlgSolution

print("[play] loading policy.pt (AlgSolution)...", flush=True)
solution = AlgSolution()
print("[play] AlgSolution loaded.", flush=True)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play Atec Tasks (ENV only, no RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--no-video-overlay",
    action="store_true",
    default=False,
    help="Disable HUD text (vel/reward/time) on recorded videos.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug prints: reward, elapsed sim time, measured base linear velocities.",
)
parser.add_argument(
    "--fast",
    action="store_true",
    default=None,
    help="Disable cameras + lidar obs for fast reset/play (default: on for TaskB when not using --video).",
)
parser.add_argument(
    "--full-obs",
    action="store_true",
    default=False,
    help="Keep all sensors (4 cameras + lidar); reset/step will be very slow.",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

_is_task_b = isinstance(args_cli.task, str) and "TaskB" in args_cli.task
_is_task_d = isinstance(args_cli.task, str) and "TaskD" in args_cli.task
if args_cli.fast is None:
    # Task B: fast by default (proprio-only). Use --full-obs to keep 4 cameras + lidar.
    # Task D: need extero (LiDAR) for box approach — do not strip sensors unless --fast.
    args_cli.fast = _is_task_b and not args_cli.full_obs and not _is_task_d

# RecordVideo needs Kit rendering; does NOT need 4× observation cameras (those slow reset).
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch Isaac Sim / Kit
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
if hasattr(solution, "set_device"):
    solution.set_device(args_cli.device)
    print(f"[play] AlgSolution device -> {args_cli.device}", flush=True)

# -----------------------------------------------------------------------------
# Imports AFTER simulation_app is created (IsaacLab pattern)
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from rl_utils import camera_follow, RenderOverlayWrapper


def _disable_heavy_sensors(env_cfg) -> None:
    """Drop cameras + lidar so reset/step only compute proprio (much faster)."""
    if hasattr(env_cfg, "scene"):
        env_cfg.scene.head_camera = None
        env_cfg.scene.ee_camera = None
        env_cfg.scene.ee_dual_camera = None
        env_cfg.scene.lidar_sensor = None
    # Must remove whole obs groups; empty group with concatenate_terms=True crashes Isaac Lab.
    if hasattr(env_cfg, "observations"):
        env_cfg.observations.extero = None
        env_cfg.observations.image = None
    print("[play] --fast: disabled cameras + lidar (proprio-only).", flush=True)


def _disable_cameras_keep_lidar(env_cfg) -> None:
    """Task D: keep LiDAR extero but skip 4 observation cameras (faster reset, no Kit cameras)."""
    if hasattr(env_cfg, "scene"):
        env_cfg.scene.head_camera = None
        env_cfg.scene.ee_camera = None
        env_cfg.scene.ee_dual_camera = None
    if hasattr(env_cfg, "observations"):
        env_cfg.observations.image = None
    print("[play] Task D: cameras off, lidar kept (use --full-obs for all sensors).", flush=True)


def _build_video_overlay_lines(
    obs,
    env,
    solution,
    timestep: int,
    total_episode_reward: float,
    total_elapsed_time: float,
) -> list[str]:
    """Build HUD lines matching --debug terminal output."""
    lines = [
        f"step={timestep}",
        f"score={total_episode_reward:.2f}",
        f"time={total_elapsed_time:.2f}s",
    ]

    proprio = obs.get("proprio") if isinstance(obs, dict) else None
    if proprio is not None:
        pr = proprio[0] if isinstance(proprio, torch.Tensor) and proprio.ndim == 2 else proprio.flatten()
        if isinstance(pr, torch.Tensor):
            pr = pr.detach().cpu()
            bv = pr[:3].tolist()
            lines.append(f"body_v=({bv[0]:+.3f},{bv[1]:+.3f},{bv[2]:+.3f})")

    try:
        robot = env.unwrapped.scene.articulations["robot"]
        wv = robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        speed_xy = (wv[0] ** 2 + wv[1] ** 2) ** 0.5
        lines.append(f"world_v=({wv[0]:+.3f},{wv[1]:+.3f},{wv[2]:+.3f})")
        lines.append(f"speed_xy={speed_xy:.3f}")
    except (AttributeError, KeyError):
        pass

    if hasattr(solution, "get_video_overlay_lines"):
        lines.extend(solution.get_video_overlay_lines())

    return lines


def _debug_print_motion(obs, env, total_episode_reward: float, total_elapsed_time: float) -> None:
    """Print reward/time and measured velocities (after env.step)."""
    print(f"total_episode_reward:{total_episode_reward: .2f}")
    print(f"total_elapsed_time:{total_elapsed_time: .2f}")
    proprio = obs.get("proprio")
    if proprio is None:
        return
    pr = proprio[0] if isinstance(proprio, torch.Tensor) and proprio.ndim == 2 else proprio.flatten()
    if isinstance(pr, torch.Tensor):
        pr = pr.detach().cpu()
        bv = pr[:3].tolist()
        print(
            "base_lin_vel_body (vx,vy,vz m/s): "
            f"{bv[0]: .3f}, {bv[1]: .3f}, {bv[2]: .3f}"
        )
    try:
        robot = env.unwrapped.scene.articulations["robot"]
        wv = robot.data.root_lin_vel_w[0].detach().cpu().tolist()
        print(
            "root_lin_vel_world (vx,vy,vz m/s): "
            f"{wv[0]: .3f}, {wv[1]: .3f}, {wv[2]: .3f}"
        )
        speed_xy = (wv[0] ** 2 + wv[1] ** 2) ** 0.5
        print(f"horizontal_speed_xy (world): {speed_xy: .3f}")
    except (AttributeError, KeyError):
        pass


def play() -> tuple[float, float]:
    if args_cli.task is None:
        raise ValueError("Please provide --task, e.g. --task ATEC-TaskA-G1")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    if args_cli.fast:
        _disable_heavy_sensors(env_cfg)
    elif _is_task_d and not args_cli.full_obs:
        _disable_cameras_keep_lidar(env_cfg)
    else:
        print("[play] full-obs: 4 cameras + lidar enabled (reset may take several minutes).", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert MARL -> single agent if needed (kept from your original script)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    overlay_wrapper = None
    use_video_overlay = args_cli.video and not args_cli.no_video_overlay

    # -------------------------------------------------------------------------
    # Optional: video wrapper
    # -------------------------------------------------------------------------
    if args_cli.video:
        if use_video_overlay:
            overlay_wrapper = RenderOverlayWrapper(env)
            env = overlay_wrapper
            print("[INFO] Video HUD overlay enabled (top-left corner).", flush=True)

        # Put videos in ./logs/videos/play by default (edit as you like)
        video_kwargs = {
            "video_folder": os.path.abspath(os.path.join("logs", "videos", args_cli.task, "play")),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    if args_cli.fast:
        fast_hint = "fast/proprio-only"
    elif _is_task_d and not args_cli.full_obs:
        fast_hint = "Task D lidar-only (no obs cameras)"
    else:
        fast_hint = "full-obs (4 cameras + lidar, SLOW)"
    print(f"[play] env.reset() starting ({fast_hint}) ...", flush=True)
    obs, _ = env.reset()
    print("[play] env.reset() done, entering control loop.", flush=True)
    if args_cli.video and not is_task_e:
        camera_follow(env)
    if hasattr(solution, "reset"):
        solution.reset(task=args_cli.task)
    if isinstance(args_cli.task, str) and "TaskD" in args_cli.task and hasattr(solution, "bind_env"):
        solution.bind_env(env)
        print("[play] Task D: bind_env() — nav uses sim robot/box pose.", flush=True)

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            if timestep == 0:
                print("[play] first solution.predicts() ...", flush=True)

            # ===== Your controller goes here =====
            resp = solution.predicts(obs, total_episode_reward)
            giveup = resp["giveup"]
            if giveup:
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device=args_cli.device).view(1, -1)

            if overlay_wrapper is not None:
                overlay_wrapper.set_overlay_lines(
                    _build_video_overlay_lines(
                        obs,
                        env,
                        solution,
                        timestep,
                        total_episode_reward,
                        total_elapsed_time,
                    )
                )

            obs, reward, terminated, truncated, info = env.step(actions)
            if not is_task_e and (args_cli.video or not args_cli.headless):
                camera_follow(env)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]  # simulation time from env as primary source
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt  # wall clock time as fallback

            if args_cli.debug:
                _debug_print_motion(obs, env, total_episode_reward, total_elapsed_time)

            done = (terminated.item() or truncated.item())
            if done:
                break

            timestep += 1
            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    env.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()
