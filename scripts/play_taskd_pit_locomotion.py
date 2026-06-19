"""Play / record video for B2Piper Task D pit-crossing locomotion policy."""

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ATEC_RL_LAB_SRC = os.path.join(_REPO_ROOT, "source", "atec_rl_lab")
if os.path.isdir(_ATEC_RL_LAB_SRC) and _ATEC_RL_LAB_SRC not in sys.path:
    sys.path.insert(0, _ATEC_RL_LAB_SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Task D pit locomotion checkpoint.")
parser.add_argument("--num_envs", type=int, default=4, help="Parallel envs (use 1-4 for cleaner video).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt checkpoint.")
parser.add_argument("--pit_width_min", type=float, default=0.4)
parser.add_argument("--pit_width_max", type=float, default=1.4)
parser.add_argument("--pit_curriculum_levels", type=int, default=11)
parser.add_argument("--pit_level", type=int, default=0, help="Fixed pit-width row (0=narrowest). Ignored if --pit_curriculum.")
parser.add_argument(
    "--pit_curriculum",
    action="store_true",
    help="Keep pit-width curriculum enabled during play (default: fixed --pit_level).",
)
parser.add_argument("--command_vx", type=float, default=0.6, help="Fixed forward velocity command (m/s).")
parser.add_argument(
    "--spawn_x_offset",
    type=float,
    default=0.0,
    help="Shift spawn along env-local +x (m). Positive moves toward the pit.",
)
parser.add_argument(
    "--spawn_local_x",
    type=float,
    default=None,
    help="Override env-local spawn x (m). If set, ignores --spawn_x_offset.",
)
parser.add_argument("--video", action="store_true", help="Record mp4 via gym RecordVideo wrapper.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded rollout length in env steps.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Output folder (default: <checkpoint_dir>/videos/play).",
)
parser.add_argument("--real-time", action="store_true", help="Sleep to match sim real-time (GUI mode).")
parser.add_argument(
    "--marg",
    action="store_true",
    help="Load MARG asymmetric AC (estimator + elevation + privileged critic). Required for MARG checkpoints.",
)
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample actions from the policy (matches training rollouts). Default: deterministic mean.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print action/cmd/velocity stats every 50 steps.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=0,
    help="Steps before recording (fills MARG proprio history; try 30-60).",
)
parser.add_argument(
    "--pit_dr_light",
    action="store_true",
    help="Match train reset DR (joints, spawn xy/z/yaw, init velocity). Default: fixed spawn.",
)
parser.add_argument(
    "--pit_spawn_x_jitter",
    type=float,
    default=None,
    help="Override spawn x jitter (m). With --pit_dr_light defaults to 0.5.",
)
parser.add_argument("--pit_dr_spawn_y_jitter", type=float, default=None, help="Override robot spawn y jitter toward right (m), [0, value]. Default 0.5.")
parser.add_argument(
    "--pit_dr_spawn_z",
    type=float,
    default=None,
    help="DR spawn base z (m). Default 0.545 with --pit_dr_light.",
)
parser.add_argument("--pit_dr_spawn_z_jitter", type=float, default=None, help="DR spawn z jitter ± (m).")
parser.add_argument(
    "--pit_dr_yaw_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Yaw random range in rad (default -0.15 0.15 with --pit_dr_light).",
)
parser.add_argument(
    "--pit_dr_joint_scale",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Joint reset scale range (default 0.5 1.5 with --pit_dr_light).",
)
parser.add_argument(
    "--no_pit_box",
    action="store_true",
    help="Disable Task D push box (box on by default at teleop pushed pose).",
)
parser.add_argument(
    "--pit_box_default_spawn",
    action="store_true",
    help="Use Task D default box spawn (1.2, 1.6, 0.5) instead of teleop pushed pose.",
)
parser.add_argument(
    "--pit_box_x_jitter_down",
    type=float,
    default=None,
    help="Box x jitter downward from pushed max x (m). Default 0.5.",
)
parser.add_argument(
    "--pit_box_y_jitter",
    type=float,
    default=None,
    help="Box y jitter ± (m). Default 0.5.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import atec_rl_lab.train  # noqa: F401
from atec_rl_lab.tasks.task_d.locomotion.env_cfg import UnitreeB2PiperTaskDPitLocomotionEnvCfg
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.agents.rsl_rl_ppo_cfg import (
    UnitreeB2PiperTaskDPitLocomotionPPORunnerCfg,
)
from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import play_pit_marg


def main():
    if args_cli.marg:
        play_pit_marg(args_cli, simulation_app)
        return

    # Non-MARG flat policy play (legacy checkpoints).
    import time

    from isaaclab.utils.dict import print_dict

    device = args_cli.device if args_cli.device else "cuda"
    ckpt = os.path.abspath(args_cli.checkpoint)
    env_cfg = UnitreeB2PiperTaskDPitLocomotionEnvCfg()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.pit_width_range = (float(args_cli.pit_width_min), float(args_cli.pit_width_max))
    env_cfg.command_lin_vel_x_min = float(args_cli.command_vx)
    env_cfg.command_lin_vel_x_max = float(args_cli.command_vx)
    env_cfg.apply_command_config()

    agent_cfg = UnitreeB2PiperTaskDPitLocomotionPPORunnerCfg()
    agent_cfg.device = device
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir = args_cli.video_dir or os.path.join(os.path.dirname(ckpt), "videos", "play")
        os.makedirs(os.path.abspath(video_dir), exist_ok=True)
        video_kwargs = {
            "video_folder": os.path.abspath(video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": int(args_cli.video_length),
            "disable_logger": True,
        }
        print("[INFO] Recording video:", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(ckpt)
    runner.eval_mode()
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs = vec_env.get_observations()
    steps = 0
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = policy(obs).to(vec_env.unwrapped.device)
            obs, _, _, _ = vec_env.step(actions)
        steps += 1
        if args_cli.video and steps >= int(args_cli.video_length):
            break
        if args_cli.real_time:
            sleep_s = vec_env.unwrapped.step_dt - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
