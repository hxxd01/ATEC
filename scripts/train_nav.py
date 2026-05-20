"""Train hierarchical navigation policy (Stage 2) for ATEC Task A.

Architecture:
  High-level nav policy (THIS script)  →  vel_cmd vx∈[0,1], vy±0.45, yaw±0.35
  Low-level loco policy (frozen .pt)   →  joint_pos [B, 20]
  Task-A env                           →  CrossXMulti reward

Sensor modes (fastest → slowest):
  --no_lidar     :    0 rays – proprio only (9-dim)
  default        :   75 rays – height scanner, GridPatternCfg (84-dim)  ← RECOMMENDED
  --front_lidar  :   40 rays – narrow LiDAR ±30°, compressed 36 bins (45-dim)
  --fast         :  240 rays – coarse 360° LiDAR, compressed 36 bins (45-dim)
  --camera       :  RGB 64×64 + CNN  (12297-dim raw, 265-dim after CNN encoding)

Usage:
  python train_nav.py --ll_policy demo/policy.pt --num_envs 64 --headless
  python train_nav.py --ll_policy demo/policy.pt --num_envs 8  --no_lidar --headless
  python train_nav.py --ll_policy demo/policy.pt --num_envs 4  --camera --enable_cameras --headless
  python train_nav.py --ll_policy demo/policy.pt --num_envs 4 --video --video_length 400 --video_interval 2000
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train nav policy (hierarchical, Stage 2).")
parser.add_argument("--ll_policy",    type=str,   required=True)
parser.add_argument("--num_envs",     type=int,   default=4)
parser.add_argument(
    "--nav_dt",
    type=float,
    default=1.0,
    help="Seconds between nav policy updates (default 1.0 Hz). Loco keeps 50 Hz inside each interval.",
)
parser.add_argument(
    "--inner_steps",
    type=int,
    default=None,
    help="Low-level env steps per nav decision. Default: nav_dt / 0.02 (e.g. 50 for nav_dt=1s).",
)
parser.add_argument("--max_iter",     type=int,   default=5000)
parser.add_argument("--resume",       type=str,   default=None)
# ── Sensor / speed knobs ──────────────────────────────────────────────────────
parser.add_argument("--fast",         action="store_true", help="Coarse 360° LiDAR (4ch, 6°).")
parser.add_argument("--lidar_horiz_res", type=float, default=None)
parser.add_argument("--steps_per_env",   type=int,   default=None)
parser.add_argument("--no_lidar",     action="store_true", help="Proprio only (fastest).")
parser.add_argument("--front_lidar",  action="store_true", help="Narrow front LiDAR.")
parser.add_argument("--camera",       action="store_true",
                    help="Use head_camera RGB 64×64 + CNN. Also pass --enable_cameras.")
parser.add_argument("--camera_hw",    type=int, default=64, help="Camera resize resolution (square).")
parser.add_argument("--episode_s",    type=float, default=60.0,
                    help="Episode length in seconds (default 60s).")
parser.add_argument("--stuck_time_s", type=float, default=3.0,
                    help="Terminate after this many seconds without +x progress (after grace).")
parser.add_argument("--stuck_grace_s", type=float, default=1.0,
                    help="Ignore stuck counter for this long after each reset (settling time).")
parser.add_argument("--no_stuck", action="store_true",
                    help="Disable stuck_no_progress termination (debug / early training).")
parser.add_argument("--no_random_spawn", action="store_true",
                    help="Reset on start flats: 16×16 grid (256 cells, ~2.4m×1.1m), env_id→fixed cell.")
parser.add_argument("--debug_timing", action="store_true")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training (saved under log_dir/videos/train/).")
parser.add_argument("--video_length", type=int, default=200,
                    help="Length of each recorded clip in low-level env steps (~50 Hz).")
parser.add_argument("--video_interval", type=int, default=2000,
                    help="Start a new clip every N low-level env steps.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Camera mode and video recording need Isaac Sim rendering enabled
if args_cli.camera or args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── All heavy imports AFTER AppLauncher ─────────────────────────────────────
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import atec_rl_lab.tasks  # noqa: F401 – registers ATEC gym envs
from atec_rl_lab.train.nav.hierarchical_env  import HierarchicalNavEnv
from atec_rl_lab.train.nav.nav_cfg           import NavPPORunnerCfg, NavCameraPPORunnerCfg
from atec_rl_lab.train.nav.nav_rsl_wrapper   import NavRslRlVecEnvWrapper
from atec_rl_lab.train.nav.speed_cfg         import (
    apply_nav_speed_cfg,
    apply_nav_training_env_cfg,
    apply_fast_ppo_cfg,
)
from atec_rl_lab.train.nav.nav_actor_critic  import ActorCriticWithCNN

# Inject ActorCriticWithCNN into rsl_rl runner's eval() namespace
import rsl_rl.runners.on_policy_runner as _runner_mod
_runner_mod.ActorCriticWithCNN = ActorCriticWithCNN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    device = args_cli.device if args_cli.device else "cuda"

    # ── Build Task-A env config ───────────────────────────────────────────────
    from atec_rl_lab.tasks.task_a.env_cfg import TaskAEnvB2Cfg
    env_cfg = TaskAEnvB2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── Camera mode: keep head_camera, disable others ─────────────────────────
    if args_cli.camera:
        # Resize camera to reduce memory / throughput cost
        env_cfg.scene.head_camera.height = args_cli.camera_hw
        env_cfg.scene.head_camera.width  = args_cli.camera_hw
        env_cfg.scene.ee_camera       = None
        env_cfg.scene.ee_dual_camera  = None
        # Keep env_cfg.observations.image.head_rgb but drop the rest
        env_cfg.observations.image.head_depth   = None
        env_cfg.observations.image.ee_rgb       = None
        env_cfg.observations.image.ee_depth     = None
        env_cfg.observations.image.ee_dual_rgb  = None
        env_cfg.observations.image.ee_dual_depth = None
        # We read the camera sensor directly in the wrapper (not via obs dict)
        # so disable the image obs group entirely to avoid the manager overhead
        env_cfg.observations.image = None
        print(f"[INFO] Camera mode: head_camera {args_cli.camera_hw}×{args_cli.camera_hw} RGB", flush=True)
    else:
        # No cameras needed
        env_cfg.scene.head_camera     = None
        env_cfg.scene.ee_camera       = None
        env_cfg.scene.ee_dual_camera  = None
        env_cfg.observations.image    = None

    # ── Nav training: random spawn, stuck termination, disable reward visuals ─
    apply_nav_training_env_cfg(
        env_cfg,
        stuck_time_s=args_cli.stuck_time_s,
        stuck_grace_s=args_cli.stuck_grace_s,
        enable_stuck=not args_cli.no_stuck,
        randomize_spawn=not args_cli.no_random_spawn,
    )

    # ── Episode length & nav control period ───────────────────────────────────
    env_cfg.episode_length_s = args_cli.episode_s
    # Task-A: decimation=4, sim.dt=0.005 → env step_dt = 0.02 s (50 Hz low-level)
    env_step_dt = float(env_cfg.decimation) * float(env_cfg.sim.dt)
    if args_cli.inner_steps is not None:
        inner_steps = max(1, int(args_cli.inner_steps))
    else:
        inner_steps = max(1, int(round(args_cli.nav_dt / env_step_dt)))
    nav_dt = inner_steps * env_step_dt
    nav_steps_max = max(1, int(round(args_cli.episode_s / nav_dt)))
    print(
        f"[INFO] Nav control: 1 decision every {nav_dt:.2f}s  "
        f"({inner_steps} inner steps @ {env_step_dt}s, loco ~{1.0/env_step_dt:.0f}Hz)",
        flush=True,
    )
    print(
        f"[INFO] Episode length: {args_cli.episode_s}s  "
        f"(~{nav_steps_max} nav steps/ep, reward summed per {nav_dt:.2f}s, "
        f"stuck>{args_cli.stuck_time_s}s)",
        flush=True,
    )

    # ── Apply speed / sensor config ────────────────────────────────────────────
    # In camera mode: still apply speed cfg so the LiDAR sensor is replaced with
    # the fast height scanner (default) instead of burning GPU on 5760 rays.
    camera_hw = args_cli.camera_hw
    C = 3
    env_cfg, extero_raw_dims, lidar_bins = apply_nav_speed_cfg(
        env_cfg,
        fast=args_cli.fast,
        lidar_horiz_res=args_cli.lidar_horiz_res,
        front_only=args_cli.front_lidar,
        no_lidar=args_cli.no_lidar,
    )
    if args_cli.camera:
        # Camera mode: image is the extero input; height scan runs but we ignore it
        nav_obs_dim = C * camera_hw * camera_hw + 9
    else:
        nav_obs_dim = max(extero_raw_dims, lidar_bins) + 9

    print(f"[INFO] Nav obs dim : {nav_obs_dim}  "
          f"(camera={args_cli.camera}, extero_raw={extero_raw_dims}, lidar_bins={lidar_bins})",
          flush=True)

    # ── PPO runner config (before log_dir / video path) ───────────────────────
    if args_cli.camera:
        agent_cfg = NavCameraPPORunnerCfg()
        # Sync img dims in case --camera_hw was changed
        agent_cfg.policy.img_flat_dim = C * camera_hw * camera_hw
        agent_cfg.policy.img_hw       = camera_hw
    else:
        agent_cfg = NavPPORunnerCfg()
        if args_cli.fast:
            apply_fast_ppo_cfg(agent_cfg)

    agent_cfg.max_iterations = args_cli.max_iter
    if args_cli.steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.steps_per_env

    # ── Log directory ─────────────────────────────────────────────────────────
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir  = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging to: {log_dir}", flush=True)

    # ── Create Isaac Lab environment ──────────────────────────────────────────
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("ATEC-TaskA-B2Piper", cfg=env_cfg, render_mode=render_mode)

    # Record on the base env (50 Hz); HierarchicalNavEnv steps it inner_steps times per nav step
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ── Wrap with hierarchical nav layer ─────────────────────────────────────
    nav_env = HierarchicalNavEnv(
        env,
        ll_policy_path=args_cli.ll_policy,
        device=device,
        inner_steps=inner_steps,
        lidar_bins=lidar_bins,
        extero_raw_dims=extero_raw_dims,
        camera_mode=args_cli.camera,
        camera_hw=camera_hw,
        camera_channels=C,
        debug_timing=args_cli.debug_timing,
    )

    # ── VecEnv bridge for rsl_rl ──────────────────────────────────────────────
    vec_env = NavRslRlVecEnvWrapper(nav_env)

    # ── Runner ────────────────────────────────────────────────────────────────
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    if args_cli.resume:
        print(f"[INFO] Resuming from: {args_cli.resume}", flush=True)
        runner.load(args_cli.resume)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"),   env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    t0 = time.time()
    print("[INFO] Starting training ...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    print(f"[INFO] Training done in {time.time() - t0:.0f}s", flush=True)

    nav_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
