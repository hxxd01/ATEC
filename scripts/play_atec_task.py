# Created by skywoodsz on 2026/02/07.

import argparse
import json
import os
import sys
import time
from datetime import datetime

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
    "--print-box-pose",
    action="store_true",
    default=False,
    help="Task D: print robot/box xyz (world + env-local) every sim step; also on video HUD.",
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
parser.add_argument(
    "--save-camera-views",
    action="store_true",
    default=False,
    help="Save each stream in obs['image'] as a separate mp4 file.",
)
parser.add_argument(
    "--camera-video-dir",
    type=str,
    default=None,
    help="Output directory for camera-view mp4 files. Defaults to logs/videos/<task>/camera_views/<timestamp>.",
)
parser.add_argument(
    "--camera-video-fps",
    type=int,
    default=25,
    help="FPS for saved camera-view videos.",
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
try:  # noqa: E402
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None

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


def _disable_lidar_keep_cameras(env_cfg) -> None:
    """Task D: keep camera observations, disable LiDAR extero."""
    if hasattr(env_cfg, "scene"):
        env_cfg.scene.lidar_sensor = None
    if hasattr(env_cfg, "observations"):
        env_cfg.observations.extero = None
    print("[play] Task D: lidar off, observation cameras kept.", flush=True)


# Task D box cuboid size in env_cfg.scene.box spawn (0.8, 1.0, 0.6) m.
_TASKD_BOX_HALF_EXT = (0.4, 0.5, 0.3)
_TASKD_BOX_DROP_CENTER_Z = 0.0  # teacher push-done: link-frame z < 0
_play_box_lowest_z0: float | None = None


def _box_corners_world(link_pos: torch.Tensor, link_quat: torch.Tensor) -> torch.Tensor:
    """Eight cuboid corners in world frame from PhysX link pose + env_cfg size."""
    from isaaclab.utils.math import quat_apply

    device = link_pos.device
    hx, hy, hz = _TASKD_BOX_HALF_EXT
    corners_b = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners_b.append([sx * hx, sy * hy, sz * hz])
    corners_b = torch.tensor(corners_b, device=device, dtype=torch.float32)
    quat = link_quat.unsqueeze(0).expand(8, -1)
    return quat_apply(quat, corners_b) + link_pos.unsqueeze(0)


def _taskd_scene_pose_snapshot(env) -> dict | None:
    """PhysX poses for env 0: link/COM/AABB (not a synthetic geometry-center guess)."""
    try:
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        try:
            box = unwrapped.scene["box"]
        except (AttributeError, KeyError):
            box = unwrapped.scene.rigid_objects["box"]
        device = robot.data.root_link_pos_w.device

        robot_link = robot.data.root_link_pos_w[0]
        box_link = box.data.root_link_pos_w[0]
        box_com = (
            box.data.root_com_pos_w[0]
            if hasattr(box.data, "root_com_pos_w")
            else box_link
        )
        box_quat = box.data.root_link_quat_w[0]
        corners_w = _box_corners_world(box_link, box_quat)
        aabb_min = corners_w.min(dim=0).values
        aabb_max = corners_w.max(dim=0).values
        # Center of the bottom face (4 corners at min-z side of the oriented box).
        z_eps = 1.0e-4
        bottom_mask = corners_w[:, 2] <= corners_w[:, 2].min() + z_eps
        bottom_ctr = corners_w[bottom_mask].mean(dim=0)

        box_v = torch.zeros(3, device=device)
        if hasattr(box.data, "root_com_lin_vel_w"):
            box_v = box.data.root_com_lin_vel_w[0]

        origin = torch.zeros(3, device=device)
        if hasattr(unwrapped.scene, "env_origins"):
            origin = unwrapped.scene.env_origins[0]

        def _to_list(t: torch.Tensor) -> list[float]:
            return t.detach().cpu().tolist()

        robot_w = _to_list(robot_link)
        dist_xy = float(
            torch.linalg.vector_norm(robot_link[:2] - box_link[:2]).detach().cpu()
        )
        speed = float(torch.linalg.vector_norm(box_v).detach().cpu())
        link_z = float(box_link[2].item())
        return {
            "robot_w": robot_w,
            "box_link": _to_list(box_link),
            "box_com": _to_list(box_com),
            "box_aabb_min": _to_list(aabb_min),
            "box_aabb_max": _to_list(aabb_max),
            "box_bottom_ctr": _to_list(bottom_ctr),
            "box_lowest_z": float(aabb_min[2].item()),
            "box_v": _to_list(box_v),
            "dist_xy": dist_xy,
            "box_speed": speed,
            "teacher_drop_link_z_lt0": link_z < _TASKD_BOX_DROP_CENTER_Z,
        }
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def _format_xyz(label: str, xyz: list[float]) -> str:
    return f"{label}=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})"


def _print_taskd_box_pose(env, timestep: int, *, on_done: bool = False) -> None:
    global _play_box_lowest_z0
    snap = _taskd_scene_pose_snapshot(env)
    if snap is None:
        print(f"[play] step={timestep} box pose unavailable (scene has no robot/box)", flush=True)
        return
    lowest_z = snap["box_lowest_z"]
    if _play_box_lowest_z0 is None:
        _play_box_lowest_z0 = lowest_z
    dz_low = lowest_z - _play_box_lowest_z0
    settled = snap["box_speed"] < 0.02
    tag = "done" if on_done else f"step={timestep}"
    bv = snap["box_v"]
    print(
        f"[play] {tag} "
        f"{_format_xyz('box_link', snap['box_link'])} "
        f"{_format_xyz('box_com', snap['box_com'])} "
        f"{_format_xyz('box_bottom', snap['box_bottom_ctr'])} "
        f"box_lowest_z={lowest_z:+.3f} dz_low={dz_low:+.3f} "
        f"aabb_min={snap['box_aabb_min'][0]:+.3f},{snap['box_aabb_min'][1]:+.3f},{snap['box_aabb_min'][2]:+.3f} "
        f"aabb_max={snap['box_aabb_max'][0]:+.3f},{snap['box_aabb_max'][1]:+.3f},{snap['box_aabb_max'][2]:+.3f} "
        f"box_v=({bv[0]:+.3f},{bv[1]:+.3f},{bv[2]:+.3f}) spd={snap['box_speed']:.4f} "
        f"settled={int(settled)} teacher_link_z_lt0={int(snap['teacher_drop_link_z_lt0'])} "
        f"dist_xy={snap['dist_xy']:.3f} {_format_xyz('robot_link', snap['robot_w'])}",
        flush=True,
    )


def _taskd_pose_overlay_lines(env) -> list[str]:
    snap = _taskd_scene_pose_snapshot(env)
    if snap is None:
        return []
    return [
        _format_xyz("box_link", snap["box_link"]),
        f"lowest_z={snap['box_lowest_z']:+.2f} bottom_z={snap['box_bottom_ctr'][2]:+.2f}",
        f"spd={snap['box_speed']:.3f}",
    ]


def _build_video_overlay_lines(
    obs,
    env,
    solution,
    timestep: int,
    total_episode_reward: float,
    total_elapsed_time: float,
    *,
    show_box_pose: bool = False,
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

    if show_box_pose:
        lines.extend(_taskd_pose_overlay_lines(env))

    if hasattr(solution, "get_video_overlay_lines"):
        lines.extend(solution.get_video_overlay_lines())

    return lines


def _debug_print_motion(
    obs,
    env,
    total_episode_reward: float,
    total_elapsed_time: float,
    solution=None,
    *,
    timestep: int = 0,
    print_box_pose: bool = False,
) -> None:
    """Print reward/time and measured velocities (after env.step)."""
    print(f"total_episode_reward:{total_episode_reward: .2f}")
    print(f"total_elapsed_time:{total_elapsed_time: .2f}")
    if print_box_pose:
        _print_taskd_box_pose(env, timestep)
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
    if solution is not None and hasattr(solution, "get_video_overlay_lines"):
        try:
            for line in solution.get_video_overlay_lines():
                print(f"[policy] {line}")
        except Exception:
            pass


def _print_done_reason(terminated, truncated, info, env=None) -> None:
    """Print done flags and best-effort termination cause."""
    term_val = bool(terminated.item() if hasattr(terminated, "item") else terminated)
    trunc_val = bool(truncated.item() if hasattr(truncated, "item") else truncated)
    print(f"[play] done: terminated={int(term_val)} truncated={int(trunc_val)}", flush=True)
    if not isinstance(info, dict):
        return

    # Common structured fields used by env wrappers/managers.
    for key in ("termination_terms", "terminations", "done_reasons", "episode_end"):
        if key in info:
            print(f"[play] {key}: {info[key]}", flush=True)

    # Fallback: surface likely termination-related true flags.
    true_flags: list[str] = []
    for key, value in info.items():
        key_l = str(key).lower()
        if not any(tok in key_l for tok in ("term", "done", "fall", "timeout", "trunc", "reach")):
            continue
        flag = None
        if isinstance(value, bool):
            flag = value
        elif isinstance(value, (int, float)):
            flag = bool(value)
        elif hasattr(value, "numel") and hasattr(value, "view"):
            try:
                if value.numel() > 0:
                    flag = bool(value.view(-1)[0].item())
            except Exception:
                flag = None
        if flag:
            true_flags.append(f"{key}={value}")

    if true_flags:
        print("[play] true termination-related flags:", flush=True)
        for line in true_flags:
            print(f"  - {line}", flush=True)

    # Final fallback: infer common Task D termination terms directly from env state.
    try:
        if env is None:
            return
        robot = env.unwrapped.scene.articulations["robot"]
        root_pos = robot.data.root_pos_w
        root_x = float(root_pos[0, 0].item())
        root_z = float(root_pos[0, 2].item())

        fall_thresh = 0.25
        x_thresh = 3.5
        try:
            cfg = env.unwrapped.cfg
            if getattr(cfg, "terminations", None) is not None:
                fall_cfg = getattr(cfg.terminations, "fall", None)
                x_cfg = getattr(cfg.terminations, "x_reached", None)
                if fall_cfg is not None and isinstance(getattr(fall_cfg, "params", None), dict):
                    fall_thresh = float(fall_cfg.params.get("minimum_height", fall_thresh))
                if x_cfg is not None and isinstance(getattr(x_cfg, "params", None), dict):
                    x_thresh = float(x_cfg.params.get("x_threshold", x_thresh))
        except Exception:
            pass

        fall_flag = root_z < fall_thresh
        x_reached_flag = root_x > x_thresh
        time_out_flag = False
        try:
            # episode_length_buf is per-env step counter in Isaac Lab envs.
            step_count = int(env.unwrapped.episode_length_buf[0].item())
            max_steps = int(env.unwrapped.max_episode_length)
            time_out_flag = step_count >= max_steps
            print(
                f"[play] infer: step={step_count}/{max_steps} "
                f"root_x={root_x:+.3f} (x_thresh={x_thresh:+.3f}) "
                f"root_z={root_z:+.3f} (fall_thresh={fall_thresh:+.3f})",
                flush=True,
            )
        except Exception:
            print(
                f"[play] infer: root_x={root_x:+.3f} (x_thresh={x_thresh:+.3f}) "
                f"root_z={root_z:+.3f} (fall_thresh={fall_thresh:+.3f})",
                flush=True,
            )

        print(
            f"[play] infer terms: fall={int(fall_flag)} x_reached={int(x_reached_flag)} time_out={int(time_out_flag)}",
            flush=True,
        )
    except Exception:
        pass


def _to_uint8_hwc(frame) -> np.ndarray | None:
    """Convert image tensor/array to uint8 HWC for video writing."""
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None

    # CHW -> HWC if needed
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        max_v = float(arr.max())
        min_v = float(arr.min())
        if max_v <= 1.5 and min_v >= 0.0:
            arr = arr * 255.0
        elif max_v > min_v:
            arr = (arr - min_v) / (max_v - min_v) * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


class CameraViewRecorder:
    """Record obs['image'] streams to mp4 files."""

    def __init__(self, out_dir: str, fps: int):
        self.out_dir = out_dir
        self.fps = int(fps)
        self.writers: dict[str, object] = {}
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[play] camera views output: {self.out_dir}", flush=True)

    def _get_writer(self, key: str):
        if key not in self.writers:
            path = os.path.join(self.out_dir, f"{key}.mp4")
            self.writers[key] = imageio.get_writer(path, fps=self.fps)
            print(f"[play] recording stream: {path}", flush=True)
        return self.writers[key]

    def write(self, obs: dict):
        if not isinstance(obs, dict):
            return
        image_obs = obs.get("image")
        if not isinstance(image_obs, dict):
            return
        for key, value in image_obs.items():
            frame = _to_uint8_hwc(value)
            if frame is None:
                continue
            self._get_writer(key).append_data(frame)

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


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
        # Task D default previously kept LiDAR only for scripted navigation.
        # For vision policies, if cameras are enabled, prefer camera obs and disable LiDAR.
        if args_cli.enable_cameras:
            _disable_lidar_keep_cameras(env_cfg)
        else:
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
        if args_cli.enable_cameras:
            fast_hint = "Task D camera-only (obs cameras, no lidar)"
        else:
            fast_hint = "Task D lidar-only (no obs cameras)"
    else:
        fast_hint = "full-obs (4 cameras + lidar, SLOW)"
    print(f"[play] env.reset() starting ({fast_hint}) ...", flush=True)
    obs, _ = env.reset()
    print("[play] env.reset() done, entering control loop.", flush=True)
    camera_recorder = None
    if args_cli.save_camera_views:
        if imageio is None:
            raise ImportError("imageio is required for --save-camera-views. Install with: pip install imageio")
        if args_cli.camera_video_dir is not None:
            out_dir = os.path.abspath(args_cli.camera_video_dir)
        else:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = os.path.abspath(os.path.join("logs", "videos", args_cli.task, "camera_views", stamp))
        camera_recorder = CameraViewRecorder(out_dir=out_dir, fps=args_cli.camera_video_fps)
        camera_recorder.write(obs)

    if args_cli.video and not is_task_e:
        camera_follow(env)
    if hasattr(solution, "reset"):
        solution.reset(task=args_cli.task)
    if isinstance(args_cli.task, str) and "TaskD" in args_cli.task and hasattr(solution, "bind_env"):
        solution.bind_env(env)
        print("[play] Task D: bind_env() — nav uses sim robot/box pose.", flush=True)

    show_box_pose = bool(_is_task_d and args_cli.print_box_pose)
    if show_box_pose:
        global _play_box_lowest_z0
        _play_box_lowest_z0 = None
        print(
            "[play] Task D: box pose from PhysX — link/COM + AABB corners (0.8x1.0x0.6); "
            "box_lowest_z = world min corner; teacher uses link z<0.",
            flush=True,
        )

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
                        show_box_pose=show_box_pose,
                    )
                )

            obs, reward, terminated, truncated, info = env.step(actions)
            if camera_recorder is not None:
                camera_recorder.write(obs)
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

            timestep += 1

            if args_cli.debug:
                _debug_print_motion(
                    obs,
                    env,
                    total_episode_reward,
                    total_elapsed_time,
                    solution=solution,
                    timestep=timestep,
                    print_box_pose=show_box_pose,
                )
            elif show_box_pose:
                _print_taskd_box_pose(env, timestep)

            done = (terminated.item() or truncated.item())
            if done:
                _print_done_reason(terminated, truncated, info, env=env)
                break

            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    if camera_recorder is not None:
        camera_recorder.close()
    env.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()
