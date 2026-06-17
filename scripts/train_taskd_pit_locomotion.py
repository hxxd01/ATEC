"""Train B2Piper Task D pit-crossing locomotion (no depth, pit-width curriculum)."""

import argparse
import os
import sys
from datetime import datetime

# Prefer this repo's source tree over an editable install from another checkout.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ATEC_RL_LAB_SRC = os.path.join(_REPO_ROOT, "source", "atec_rl_lab")
if os.path.isdir(_ATEC_RL_LAB_SRC) and _ATEC_RL_LAB_SRC not in sys.path:
    sys.path.insert(0, _ATEC_RL_LAB_SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Task D pit locomotion (B2Piper-flat rewards + priv obs).")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel envs (512 recommended; 4096 bakes 11x4096 tiles).")
parser.add_argument("--max_iterations", type=int, default=15000)
parser.add_argument("--pit_width_min", type=float, default=0.4)
parser.add_argument("--pit_width_max", type=float, default=1.4)
parser.add_argument("--pit_curriculum_levels", type=int, default=11)
parser.add_argument("--command_vx_max", type=float, default=4.0, help="Max forward velocity command (m/s) after curriculum.")
parser.add_argument(
    "--command_vx",
    type=float,
    default=None,
    help="Alias for --command_vx_max (e.g. --command_vx 4.0).",
)
parser.add_argument("--command_vx_min", type=float, default=0.0, help="Min forward velocity command (m/s).")
parser.add_argument(
    "--command_curriculum_start",
    type=float,
    default=0.1,
    help="Initial vx max = min + start_frac * (max-min); default 0.1 -> 0.4 m/s when max=4.",
)
parser.add_argument("--resume", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import atec_rl_lab.train  # noqa: F401
from atec_rl_lab.tasks.task_d.locomotion.env_cfg import (
    UnitreeB2PiperTaskDPitLocomotionEnvCfg,
    refresh_task_d_pit_locomotion_terrain_cfg,
)
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.agents.rsl_rl_ppo_cfg import (
    UnitreeB2PiperTaskDPitLocomotionPPORunnerCfg,
)


def main():
    device = args_cli.device if args_cli.device else "cuda"

    env_cfg = UnitreeB2PiperTaskDPitLocomotionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.pit_width_range = (float(args_cli.pit_width_min), float(args_cli.pit_width_max))
    env_cfg.pit_curriculum_levels = int(args_cli.pit_curriculum_levels)
    env_cfg.command_lin_vel_x_min = float(args_cli.command_vx_min)
    vx_max = args_cli.command_vx_max if args_cli.command_vx is None else args_cli.command_vx
    env_cfg.command_lin_vel_x_max = float(vx_max)
    env_cfg.command_curriculum_start_fraction = float(args_cli.command_curriculum_start)
    refresh_task_d_pit_locomotion_terrain_cfg(env_cfg)

    agent_cfg = UnitreeB2PiperTaskDPitLocomotionPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[INFO] Resuming from {args_cli.resume}", flush=True)
        runner.load(args_cli.resume)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[INFO] Logging to {log_dir}", flush=True)
    print(
        f"[INFO] pit_width={env_cfg.pit_width_range}, levels={env_cfg.pit_curriculum_levels}, "
        f"vx=[{env_cfg.command_lin_vel_x_min}, {env_cfg.command_lin_vel_x_max}] m/s "
        f"(curriculum start_frac={env_cfg.command_curriculum_start_fraction}), "
        f"num_envs={env_cfg.scene.num_envs}",
        flush=True,
    )
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
