"""Minimal Task D teacher training script.

Teacher setup:
- Actor obs: lidar + compact proprio + relative box geometry
- Critic obs: actor obs + privileged absolute robot/box world states
- High-level action: [vx, vy, yaw] in [-1, 1], mapped to vx in [-2, 2]
- Reward: stage progress + push-stage anti-skew shaping + contact bonus
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Task D teacher policy (hierarchical).")
parser.add_argument("--ll_policy", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--inner_steps", type=int, default=25, help="25 -> 50Hz/25 = 2Hz nav control")
parser.add_argument("--max_iter", type=int, default=8000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--steps_per_env", type=int, default=32)
parser.add_argument("--vx_min", type=float, default=-2.0)
parser.add_argument("--vx_max", type=float, default=2.0)
parser.add_argument("--curriculum_warmup_nav_steps", type=int, default=1500)
parser.add_argument("--curriculum_mid_nav_steps", type=int, default=3500)
parser.add_argument("--nav_log_interval", type=int, default=50, help="Print [TaskDTeacher] nav log every N nav steps (0=off).")
parser.add_argument(
    "--disable_extero",
    action="store_true",
    help="Disable exteroceptive observations (e.g., LiDAR) for load diagnosis.",
)
parser.add_argument(
    "--lidar_horizontal_res",
    type=float,
    default=None,
    help="Override LiDAR horizontal resolution (degrees). Larger means fewer rays.",
)
parser.add_argument(
    "--lidar_channels",
    type=int,
    default=None,
    help="Override LiDAR channel count. Smaller means fewer rays.",
)
parser.add_argument(
    "--lidar_update_period",
    type=float,
    default=None,
    help="Override LiDAR sensor update period (seconds). Larger means lower sensor rate.",
)
parser.add_argument(
    "--debug_lidar_runtime",
    action="store_true",
    help="Print runtime LiDAR sensor cfg and ray_hits tensor shape after env reset.",
)
parser.add_argument(
    "--rl_device",
    type=str,
    default=None,
    help="Device for PPO updates (e.g., cpu, cuda:0). If unset, uses --device.",
)
parser.add_argument(
    "--ppo_no_train",
    action="store_true",
    help="Disable PPO training updates; only run environment stepping for load diagnosis.",
)
parser.add_argument(
    "--no_train_steps",
    type=int,
    default=400,
    help="Number of high-level env steps when --ppo_no_train is enabled.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video.")
parser.add_argument("--video_length", type=int, default=300, help="Recorded video length in env steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
# RecordVideo needs rendering enabled even in headless mode.
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

import atec_rl_lab.tasks  # noqa: F401
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg, refresh_task_d_terrain_cfg
from atec_rl_lab.train.nav.taskd_teacher_env import TaskDTeacherEnv
from atec_rl_lab.train.nav.nav_cfg import TaskDTeacherPPORunnerCfg
from atec_rl_lab.train.nav.nav_rsl_wrapper import NavRslRlVecEnvWrapper


def _run_no_train_rollout(vec_env: NavRslRlVecEnvWrapper, steps: int) -> None:
    """Step the env with zero nav actions (no PPO update)."""
    obs, _ = vec_env.reset()
    del obs  # diagnostics mode: only use env stepping load.
    for i in range(int(steps)):
        actions = torch.zeros((vec_env.num_envs, vec_env.num_actions), device=vec_env.device, dtype=torch.float32)
        _, rew, dones, extras = vec_env.step(actions)
        if i % 50 == 0 or i == steps - 1:
            mean_rew = rew.float().mean().item()
            done_ratio = dones.float().mean().item()
            stage = extras.get("teacher_stage", "n/a") if isinstance(extras, dict) else "n/a"
            print(
                f"[NO-TRAIN] step={i+1:4d}/{steps} mean_rew={mean_rew:+.4f} done_ratio={done_ratio:.2f} stage={stage}",
                flush=True,
            )


def main():
    sim_device = args_cli.device if args_cli.device else "cuda"
    rl_device = args_cli.rl_device if args_cli.rl_device else sim_device

    env_cfg = TaskDEnvB2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    refresh_task_d_terrain_cfg(env_cfg)
    # Keep camera sensors off by default (speed + stability first).
    # "enable_cameras" comes from AppLauncher.add_app_launcher_args().
    if getattr(args_cli, "enable_cameras", False):
        print("[INFO] Camera sensors enabled for this run.", flush=True)
    else:
        env_cfg.scene.head_camera = None
        env_cfg.scene.ee_camera = None
        env_cfg.scene.ee_dual_camera = None
        env_cfg.observations.image = None
    if args_cli.disable_extero:
        env_cfg.observations.extero = None
    else:
        lidar_sensor = getattr(env_cfg.scene, "lidar_sensor", None)
        if lidar_sensor is not None:
            if args_cli.lidar_horizontal_res is not None:
                lidar_sensor.pattern_cfg.horizontal_res = float(args_cli.lidar_horizontal_res)
            if args_cli.lidar_channels is not None:
                lidar_sensor.pattern_cfg.channels = int(args_cli.lidar_channels)
            if args_cli.lidar_update_period is not None:
                lidar_sensor.update_period = float(args_cli.lidar_update_period)
            print(
                "[INFO] LiDAR config: "
                f"horizontal_res={lidar_sensor.pattern_cfg.horizontal_res}, "
                f"channels={lidar_sensor.pattern_cfg.channels}, "
                f"update_period={lidar_sensor.update_period}",
                flush=True,
            )

    agent_cfg = TaskDTeacherPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iter
    agent_cfg.num_steps_per_env = args_cli.steps_per_env

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("ATEC-TaskD-B2Piper", cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "teacher")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=int(args_cli.video_length),
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_dir}", flush=True)
    nav_env = TaskDTeacherEnv(
        env,
        ll_policy_path=args_cli.ll_policy,
        device=sim_device,
        inner_steps=args_cli.inner_steps,
        lidar_bins=36,
        vx_min=args_cli.vx_min,
        vx_max=args_cli.vx_max,
        curriculum_warmup_nav_steps=args_cli.curriculum_warmup_nav_steps,
        curriculum_mid_nav_steps=args_cli.curriculum_mid_nav_steps,
        nav_log_interval=args_cli.nav_log_interval,
    )
    if args_cli.debug_lidar_runtime and not args_cli.disable_extero:
        try:
            nav_env.reset()
            lidar_sensor = nav_env.env.unwrapped.scene["lidar_sensor"]
            ray_hits = lidar_sensor.data.ray_hits_w
            print(
                "[DEBUG] LiDAR runtime: "
                f"horizontal_res={lidar_sensor.cfg.pattern_cfg.horizontal_res}, "
                f"channels={lidar_sensor.cfg.pattern_cfg.channels}, "
                f"update_period={lidar_sensor.cfg.update_period}, "
                f"ray_hits_shape={tuple(ray_hits.shape)}",
                flush=True,
            )
        except Exception as exc:
            print(f"[WARN] Failed to inspect runtime LiDAR config: {exc}", flush=True)
    vec_env = NavRslRlVecEnvWrapper(nav_env)

    print(f"[INFO] Devices: sim={sim_device}, ppo={rl_device}", flush=True)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=rl_device)
    try:
        p = next(runner.alg.actor_critic.parameters())
        print(f"[INFO] PPO actor_critic param device: {p.device}", flush=True)
    except Exception as exc:
        print(f"[WARN] Failed to inspect PPO param device: {exc}", flush=True)
    if args_cli.resume:
        print(f"[INFO] Resuming from: {args_cli.resume}", flush=True)
        runner.load(args_cli.resume)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[INFO] Logging to: {log_dir}", flush=True)
    if args_cli.ppo_no_train:
        print("[INFO] PPO no-train mode enabled. Running rollout only...", flush=True)
        _run_no_train_rollout(vec_env, steps=args_cli.no_train_steps)
        nav_env.close()
        return

    print("[INFO] Start TaskD teacher training...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    nav_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

