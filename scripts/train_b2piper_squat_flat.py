"""Train a dedicated B2Piper stand->squat transition policy on flat terrain.

Design goals (per request):
1) Use B2Piper flat locomotion env (not Task B env)
2) Fix base_velocity command to zero (no walking objective)
3) Reward approaching the squat pose used in demo/solution.py
4) Add standing-nearby reset randomization (small roll/pitch and small base velocity)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch

from isaaclab.app import AppLauncher


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train B2Piper squat policy on flat terrain.")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=300)
    parser.add_argument("--video_interval", type=int, default=2000)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_argparser()
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that require simulator startup
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import atec_rl_lab.train  # noqa: F401 - ensure env registrations
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.agents.rsl_rl_ppo_cfg import (
    UnitreeB2PiperFlatPPORunnerCfg,
)
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.rough_env_cfg import (
    UnitreeB2PiperRoughEnvCfg,
)
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.flat_env_cfg import (
    UnitreeB2PiperFlatEnvCfg,
)

SQUAT_TASK_ID = "ATEC-Isaac-Velocity-Flat-Unitree-B2Piper-Squat-v0"


def squat_target_pose_exp(
    env,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_offsets: tuple[float, ...],
    std: float = 0.22,
):
    """Exponential reward for reaching a desired leg-joint offset from default pose."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    target = q_default + q.new_tensor(target_offsets).unsqueeze(0)
    err = torch.sum(torch.square(q - target), dim=1)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return torch.exp(-err / (std**2)) * upright


@configclass
class UnitreeB2PiperSquatFlatEnvCfg(UnitreeB2PiperFlatEnvCfg):
    """Flat B2Piper env tuned for stand->squat transition learning."""

    def __post_init__(self):
        super().__post_init__()

        # 1) Zero command (no walking objective)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # 2) Keep short episodes for transition shaping (150 env steps @ 0.02s)
        self.episode_length_s = 3.0

        # 3) Standing-nearby reset randomization (small attitude/speed noise)
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.20, 0.20),
            "y": (-0.20, 0.20),
            "z": (0.00, 0.05),
            "roll": (-0.12, 0.12),
            "pitch": (-0.12, 0.12),
            "yaw": (-3.14, 3.14),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.20, 0.20),
            "y": (-0.20, 0.20),
            "z": (-0.10, 0.10),
            "roll": (-0.20, 0.20),
            "pitch": (-0.20, 0.20),
            "yaw": (-0.20, 0.20),
        }
        self.events.randomize_reset_joints.params["position_range"] = (0.95, 1.05)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.10, 0.10)

        # Disable heavy disturbances for fast convergence on transition behavior.
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None

        # 4) Reward shaping:
        #    Keep posture stability, remove "stand at default joints" conflicts.
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.upward.weight = 2.0
        self.rewards.action_rate_l2.weight = -0.005

        # Squat target from demo/solution.py action-space intent:
        # action target: hip=0.34, thigh=0.78, calf=-1.45 with scale~0.5 in deployment.
        # Approx target joint offsets around default:
        #   hip: +/-0.17, thigh:+0.39, calf:-0.725
        squat_target_offsets = (
            -0.17, +0.39, -0.725,  # FR
            +0.17, +0.39, -0.725,  # FL
            -0.17, +0.39, -0.725,  # RR
            +0.17, +0.39, -0.725,  # RL
        )
        self.rewards.squat_target_pose = RewTerm(
            func=squat_target_pose_exp,
            weight=6.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UnitreeB2PiperRoughEnvCfg.joint_names),
                "target_offsets": squat_target_offsets,
                "std": 0.22,
            },
        )

        # 5) Safety termination for collapse/contact
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["base_link", ".*_hip", ".*_thigh"],
                ),
                "threshold": 1.0,
            },
        )
        self.terminations.fall = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.0,
            },
            time_out=False,
        )


def _register_squat_task() -> None:
    """Register a standalone squat task ID for this custom env config."""
    if SQUAT_TASK_ID in gym.registry:
        return
    gym.register(
        id=SQUAT_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:UnitreeB2PiperSquatFlatEnvCfg",
            "rsl_rl_cfg_entry_point": (
                "atec_rl_lab.train.locomotion.velocity.config.quadruped."
                "unitree_b2_piper.agents.rsl_rl_ppo_cfg:UnitreeB2PiperFlatPPORunnerCfg"
            ),
        },
    )


def main() -> None:
    _register_squat_task()
    env_cfg = UnitreeB2PiperSquatFlatEnvCfg()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)

    runner_cfg = UnitreeB2PiperFlatPPORunnerCfg()
    runner_cfg.seed = int(args_cli.seed)
    runner_cfg.max_iterations = int(args_cli.max_iterations)
    runner_cfg.experiment_name = "unitree_b2_piper_flat_squat_transition"
    runner_cfg.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner_cfg.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", runner_cfg.experiment_name))
    log_dir = os.path.join(log_root, runner_cfg.run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(
        SQUAT_TASK_ID,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda step: step % int(args_cli.video_interval) == 0,
            video_length=int(args_cli.video_length),
            disable_logger=True,
        )

    vec_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=log_dir, device=runner_cfg.device)
    runner.add_git_repo_to_log(__file__)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), runner_cfg)

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
