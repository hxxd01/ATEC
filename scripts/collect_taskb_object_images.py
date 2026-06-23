#!/usr/bin/env python3
"""Collect Task-B garbage images grouped by class labels.

This script uses environment ground-truth object IDs (object_1..object_18) to
assign labels:
  - object_1..6   -> sugar
  - object_7..12  -> mustard
  - object_13..18 -> banana

It teleports the robot near each object, faces it, settles a few steps, then
stores RGB frames from `head_rgb` or `ee_rgb`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from datetime import datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Task-B object images by class.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--num-per-class", type=int, default=600, help="Images to save for each class.")
parser.add_argument(
    "--camera",
    type=str,
    default="head_rgb",
    choices=["head_rgb", "ee_rgb"],
    help="RGB camera stream key to save.",
)
parser.add_argument("--distance", type=float, default=1.2, help="Robot-object distance in meters.")
parser.add_argument(
    "--distance-min",
    type=float,
    default=0.7,
    help="Minimum robot-object distance for random sampling.",
)
parser.add_argument(
    "--distance-max",
    type=float,
    default=1.8,
    help="Maximum robot-object distance for random sampling.",
)
parser.add_argument("--settle-steps", type=int, default=8, help="Simulation steps after teleport before capture.")
parser.add_argument(
    "--squat",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use squat leg pose before capture (default: true).",
)
parser.add_argument("--squat-steps", type=int, default=12, help="Steps to hold squat pose before capture.")
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--max-episodes", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Camera tensors require camera-enabled app experience.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _to_uint8_hwc(frame) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        max_v = float(arr.max()) if arr.size else 0.0
        min_v = float(arr.min()) if arr.size else 0.0
        if max_v <= 1.5 and min_v >= 0.0:
            arr = arr * 255.0
        elif max_v > min_v:
            arr = (arr - min_v) / (max_v - min_v) * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _obj_class(obj_idx: int) -> str:
    if obj_idx <= 6:
        return "sugar"
    if obj_idx <= 12:
        return "mustard"
    return "banana"


def _prepare_env():
    print("[collect] stage: parse_env_cfg ...", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    print("[collect] stage: env_cfg ready", flush=True)
    # Speed up reset/step for dataset collection.
    if hasattr(env_cfg, "scene"):
        env_cfg.scene.lidar_sensor = None
        env_cfg.scene.ee_dual_camera = None
        # Keep only selected RGB camera to avoid extra renderer overhead.
        if args_cli.camera == "head_rgb":
            env_cfg.scene.ee_camera = None
        else:
            env_cfg.scene.head_camera = None
    if hasattr(env_cfg, "observations"):
        if getattr(env_cfg.observations, "extero", None) is not None:
            env_cfg.observations.extero = None
        img = getattr(env_cfg.observations, "image", None)
        if img is not None:
            if args_cli.camera == "head_rgb":
                img.ee_rgb = None
            else:
                img.head_rgb = None
            img.ee_dual_rgb = None
            img.ee_dual_depth = None
            img.head_depth = None
            img.ee_depth = None
    print("[collect] stage: gym.make ...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[collect] stage: gym.make done", flush=True)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        print("[collect] stage: converted MARL->single", flush=True)
    return env


def _action_dim_from_obs(obs: dict) -> int:
    proprio = obs["proprio"]
    if isinstance(proprio, torch.Tensor):
        p = proprio
    else:
        p = torch.as_tensor(proprio)
    return (int(p.shape[-1]) - 12) // 3


def _teleport_robot_near_object(env, robot, obj_pos: torch.Tensor, distance: float):
    device = obj_pos.device
    theta = random.uniform(-np.pi, np.pi)
    dx = float(np.cos(theta))
    dy = float(np.sin(theta))
    rx = float(obj_pos[0].item() - distance * dx)
    ry = float(obj_pos[1].item() - distance * dy)
    # Stand height for B2Piper in Task-B.
    rz = 0.68
    yaw = float(np.arctan2(float(obj_pos[1].item()) - ry, float(obj_pos[0].item()) - rx))

    zeros = torch.zeros(1, device=device)
    yaw_t = torch.tensor([yaw], dtype=torch.float32, device=device)
    quat = quat_from_euler_xyz(zeros, zeros, yaw_t)  # wxyz
    pose = torch.cat(
        [
            torch.tensor([[rx, ry, rz]], dtype=torch.float32, device=device),
            quat,
        ],
        dim=-1,
    )
    robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=device, dtype=torch.long))
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device), env_ids=torch.tensor([0], device=device, dtype=torch.long))


def _build_squat_action(action_dim: int, device: torch.device) -> torch.Tensor:
    squat_pose = [
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
        0.0, 0.5, -1.0,
    ]
    out = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
    out[0, :12] = torch.tensor(squat_pose, dtype=torch.float32, device=device)
    return out


def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    dist_min = float(args_cli.distance_min)
    dist_max = float(args_cli.distance_max)
    if dist_min > dist_max:
        dist_min, dist_max = dist_max, dist_min

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args_cli.output_dir is None:
        out_dir = os.path.abspath(os.path.join("datasets", "taskb_object_images", stamp))
    else:
        out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    for c in ("sugar", "mustard", "banana"):
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)
    meta_path = os.path.join(out_dir, "metadata.jsonl")

    print("[collect] stage: prepare env", flush=True)
    env = _prepare_env()
    print("[collect] stage: first reset ...", flush=True)
    obs, _ = env.reset()
    print("[collect] stage: first reset done", flush=True)
    image_obs = obs.get("image") if isinstance(obs, dict) else None
    if isinstance(image_obs, dict):
        print(f"[collect] available image keys: {list(image_obs.keys())}", flush=True)
    else:
        print("[collect] warning: obs['image'] missing after reset", flush=True)
    action_dim = _action_dim_from_obs(obs)
    zero_action = torch.zeros((1, action_dim), dtype=torch.float32, device=env.unwrapped.device)
    squat_action = _build_squat_action(action_dim, env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    counts = {"sugar": 0, "mustard": 0, "banana": 0}
    total_target = int(args_cli.num_per_class)
    episode = 0
    captured = 0

    with open(meta_path, "w", encoding="utf-8") as mf:
        print(
            f"[collect] start task={args_cli.task} camera={args_cli.camera} "
            f"distance_range=({dist_min:.2f},{dist_max:.2f}) squat={args_cli.squat} "
            f"target/class={total_target}",
            flush=True,
        )
        while episode < args_cli.max_episodes:
            if min(counts.values()) >= total_target:
                break
            episode += 1
            obs, _ = env.reset()

            obj_ids = list(range(1, 19))
            random.shuffle(obj_ids)
            for obj_idx in obj_ids:
                cls = _obj_class(obj_idx)
                if counts[cls] >= total_target:
                    continue

                obj = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"]
                obj_pos = obj.data.root_pos_w[0, :3].clone()
                sample_dist = float(random.uniform(dist_min, dist_max))
                _teleport_robot_near_object(env, robot, obj_pos, sample_dist)

                # Let dynamics and sensors settle.
                terminated = truncated = None
                act = zero_action
                if bool(args_cli.squat):
                    for _ in range(max(0, int(args_cli.squat_steps))):
                        obs, _, terminated, truncated, _ = env.step(squat_action)
                        if bool(terminated.item()) or bool(truncated.item()):
                            obs, _ = env.reset()
                            break
                    act = squat_action
                for _ in range(max(1, int(args_cli.settle_steps))):
                    obs, _, terminated, truncated, _ = env.step(act)
                    if bool(terminated.item()) or bool(truncated.item()):
                        obs, _ = env.reset()
                        break

                image_obs = obs.get("image") if isinstance(obs, dict) else None
                if not isinstance(image_obs, dict) or args_cli.camera not in image_obs:
                    continue
                frame = _to_uint8_hwc(image_obs[args_cli.camera])
                if frame is None:
                    continue

                idx = counts[cls]
                fname = f"{cls}_{idx:06d}.jpg"
                fpath = os.path.join(out_dir, cls, fname)
                try:
                    import imageio.v2 as imageio
                except ImportError:
                    import imageio
                imageio.imwrite(fpath, frame)

                rec = {
                    "file": os.path.relpath(fpath, out_dir),
                    "class": cls,
                    "object_index": obj_idx,
                    "camera_key": args_cli.camera,
                    "episode": episode,
                    "distance": sample_dist,
                    "object_pos_w": [float(x) for x in obj_pos.detach().cpu().tolist()],
                }
                mf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                counts[cls] += 1
                captured += 1
                print(
                    f"[collect] saved={captured} cls={cls} obj={obj_idx} "
                    f"dist={sample_dist:.2f} counts={counts}",
                    flush=True,
                )

                if min(counts.values()) >= total_target:
                    break

    print(f"[collect] done. output={out_dir}", flush=True)
    print(f"[collect] final counts={counts}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        print(
            f"[collect] parsed args: task={args_cli.task} num_per_class={args_cli.num_per_class} "
            f"camera={args_cli.camera} headless={args_cli.headless} enable_cameras={args_cli.enable_cameras}",
            flush=True,
        )
        main()
    except Exception:
        print("[collect] fatal error:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
