"""Play / record video for Task D MARG depth pit DAgger student (MargDepthPitActorCritic)."""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ATEC_RL_LAB_SRC = os.path.join(_REPO_ROOT, "source", "atec_rl_lab")
if os.path.isdir(_ATEC_RL_LAB_SRC) and _ATEC_RL_LAB_SRC not in sys.path:
    sys.path.insert(0, _ATEC_RL_LAB_SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play Task D MARG depth pit DAgger student checkpoint (student-only inference)."
)
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (use 1 for cleaner video).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt checkpoint.")
parser.add_argument("--pit_width_min", type=float, default=0.4)
parser.add_argument("--pit_width_max", type=float, default=1.4)
parser.add_argument("--pit_curriculum_levels", type=int, default=11)
parser.add_argument(
    "--pit_level",
    type=int,
    default=10,
    help="Fixed pit-width row (0=narrowest). Default 10 = max width 1.4m (matches late DAgger train).",
)
parser.add_argument("--pit_curriculum", action="store_true", help="Full pit-width curriculum during play.")
parser.add_argument("--command_vx", type=float, default=0.6, help="Fixed forward velocity command (m/s).")
parser.add_argument("--spawn_x_offset", type=float, default=0.0)
parser.add_argument("--spawn_local_x", type=float, default=None)
parser.add_argument("--video", action="store_true", help="Record mp4 via gym RecordVideo wrapper.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded rollout length in env steps.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Output folder (default: <checkpoint_dir>/videos/play).",
)
parser.add_argument("--real-time", action="store_true", help="Sleep to match sim real-time (GUI mode).")
parser.add_argument("--stochastic", action="store_true", help="Sample actions (default: deterministic mean).")
parser.add_argument("--debug", action="store_true", help="Print action/cmd/velocity stats every 50 steps.")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=40,
    help="Steps before recording (fills MARG proprio history).",
)
parser.add_argument("--policy_img_h", type=int, default=24)
parser.add_argument("--policy_img_w", type=int, default=32)
parser.add_argument("--sim_camera_h", type=int, default=24)
parser.add_argument("--sim_camera_w", type=int, default=32)
parser.add_argument("--depth_only", action="store_true", default=True)
parser.add_argument("--depth_max", type=float, default=5.0)
parser.add_argument("--tiled_cameras", action="store_true", default=True)
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import atec_rl_lab.train  # noqa: F401
from atec_rl_lab.train.nav.marg_depth_pit_actor_critic import MargDepthPitActorCritic
from atec_rl_lab.train.nav.nav_cfg import TaskDMargDepthPitDaggerPPORunnerCfg
from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import (
    attach_dagger_depth_obs,
    configure_pit_e2e_cameras,
    configure_pit_e2e_dagger_env_cfg,
)
from atec_rl_lab.train.pit_marg.marg_pit_dagger_ppo import MargPitDaggerPPO
from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import apply_play_spawn, configure_play_terrain

import rsl_rl.runners.on_policy_runner as _runner_mod


def _register_modules() -> None:
    _runner_mod.MargDepthPitActorCritic = MargDepthPitActorCritic
    _runner_mod.MargPitDaggerPPO = MargPitDaggerPPO


def main():
    _register_modules()
    device = args_cli.device if args_cli.device else "cuda"
    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

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
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.pit_width_levels = None
    if hasattr(env_cfg.observations, "policy") and env_cfg.observations.policy is not None:
        env_cfg.observations.policy.enable_corruption = False

    decimation = int(getattr(env_cfg, "decimation", 4))
    sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
    phys_dt = decimation * sim_dt
    configure_pit_e2e_cameras(
        env_cfg,
        camera_height=int(args_cli.sim_camera_h),
        camera_width=int(args_cli.sim_camera_w),
        depth_only=bool(args_cli.depth_only),
        tiled=bool(args_cli.tiled_cameras),
        camera_far_clip=5.0,
        update_period=phys_dt,
    )
    attach_dagger_depth_obs(
        env_cfg,
        policy_h=int(args_cli.policy_img_h),
        policy_w=int(args_cli.policy_img_w),
        depth_max=float(args_cli.depth_max),
        depth_only=bool(args_cli.depth_only),
    )

    agent_cfg = TaskDMargDepthPitDaggerPPORunnerCfg()
    agent_cfg.device = device
    agent_cfg.policy.img_h = int(args_cli.policy_img_h)
    agent_cfg.policy.img_w = int(args_cli.policy_img_w)
    agent_cfg.policy.depth_channels = 1 if args_cli.depth_only else 4

    record_video = bool(args_cli.video)
    render_mode = "rgb_array" if record_video else None
    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg, render_mode=render_mode)

    if record_video:
        video_dir = args_cli.video_dir or os.path.join(os.path.dirname(ckpt), "videos", "play")
        video_dir = os.path.abspath(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == args_cli.warmup_steps,
            "video_length": int(args_cli.video_length),
            "disable_logger": True,
        }
        print("[TaskDMargDepthPitDAgger] Recording video:", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    print(f"[TaskDMargDepthPitDAgger] Loading checkpoint: {ckpt}", flush=True)
    runner.load(ckpt)
    runner.eval_mode()
    policy_nn = runner.alg.policy
    if args_cli.stochastic:
        policy = lambda obs: policy_nn.act(obs)
        print("[TaskDMargDepthPitDAgger] Stochastic policy.act()", flush=True)
    else:
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
        print("[TaskDMargDepthPitDAgger] Deterministic act_inference() (student-only)", flush=True)

    unwrapped = vec_env.unwrapped
    obs = vec_env.get_observations()
    dt = unwrapped.step_dt
    steps = 0
    warmup = max(0, int(args_cli.warmup_steps))
    max_steps = int(args_cli.video_length) if record_video else None

    print(
        f"[TaskDMargDepthPitDAgger] command_vx={args_cli.command_vx} m/s, "
        f"spawn_local_x={float(spawn_local[0]):.3f}, num_envs={args_cli.num_envs}, "
        f"pit_width={env_cfg.pit_width_range}, pit_level={args_cli.pit_level}, "
        f"depth={args_cli.policy_img_h}x{args_cli.policy_img_w}, stochastic={args_cli.stochastic}",
        flush=True,
    )
    if record_video and warmup > 0:
        print(f"[TaskDMargDepthPitDAgger] Warmup {warmup} steps before video", flush=True)

    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = policy(obs).to(unwrapped.device)
            if args_cli.debug and steps % 50 == 0:
                robot = unwrapped.scene["robot"]
                cmd_x = float(unwrapped.command_manager.get_command("base_velocity")[0, 0].item())
                vx = float(robot.data.root_lin_vel_w[0, 0].item())
                act_norm = float(actions[0].norm().item())
                ep_len = int(unwrapped.episode_length_buf[0].item())
                print(
                    f"[TaskDMargDepthPitDAgger debug] step={steps} cmd_x={cmd_x:.3f} vx={vx:.3f} "
                    f"|action|={act_norm:.3f} ep_len={ep_len}",
                    flush=True,
                )
            obs, _, dones, _ = vec_env.step(actions)
            policy_nn.reset(dones)
        steps += 1
        if steps == warmup and record_video:
            print(f"[TaskDMargDepthPitDAgger] Warmup done, recording starts.", flush=True)
        if max_steps is not None and steps >= warmup + max_steps:
            print(
                f"[TaskDMargDepthPitDAgger] Recorded {steps - warmup} steps after warmup, stopping.",
                flush=True,
            )
            break
        if args_cli.real_time:
            sleep_s = dt - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
