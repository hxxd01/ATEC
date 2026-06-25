import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeB2PiperFlatEnvCfg


def stand_target_pose_exp(
    env,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.9,
):
    """Exponential reward for returning to robot default standing leg pose."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    err = torch.sum(torch.square(q - q_default), dim=1)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return torch.exp(-err / (std**2)) * upright


@configclass
class UnitreeB2PiperStandFlatEnvCfg(UnitreeB2PiperFlatEnvCfg):
    """Flat B2Piper env tuned for squat-to-stand recovery / hold."""

    def __post_init__(self):
        super().__post_init__()

        # Zero command: only learn to recover/hold standing posture.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # Max episode length: 150 policy steps.
        self.episode_length_s = 150 * self.decimation * self.sim.dt

        # Larger reset randomization, including crouched starts.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.50, 0.50),
            "y": (-0.50, 0.50),
            "z": (-0.08, 0.12),
            "roll": (-0.35, 0.35),
            "pitch": (-0.35, 0.35),
            "yaw": (-3.14, 3.14),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.60, 0.60),
            "y": (-0.60, 0.60),
            "z": (-0.40, 0.40),
            "roll": (-0.60, 0.60),
            "pitch": (-0.60, 0.60),
            "yaw": (-0.60, 0.60),
        }
        # Covers stand to crouch-like starts while avoiding extreme folded postures.
        self.events.randomize_reset_joints.params["position_range"] = (0.65, 1.35)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.60, 0.60)

        self.events.randomize_push_robot.interval_range_s = (1.5, 3.0)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.22, 0.22),
            "y": (-0.22, 0.22),
            "z": (-0.22, 0.22),
        }
        self.events.randomize_apply_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["base_link"]
        )
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-14.0, 14.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-7.0, 7.0)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (-0.10, 0.10),
        }
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["(FR|FL|RR|RL)_(hip|thigh|calf)_joint"],
        )
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.45, 2.20)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.45, 2.20)

        disabled_reward_names = [
            "is_terminated",
            "joint_torques_l2",
            "joint_vel_l2",
            "joint_acc_l2",
            "joint_pos_limits",
            "joint_vel_limits",
            "joint_power",
            "stand_still",
            "joint_pos_penalty",
            "wheel_vel_penalty",
            "action_mirror",
            "action_sync",
            "applied_torque_limits",
            "undesired_contacts",
            "contact_forces",
            "feet_air_time",
            "feet_contact",
            "feet_contact_without_cmd",
            "feet_stumble",
            "feet_slide",
            "feet_height",
            "feet_height_body",
            "feet_gait",
            "upward",
        ]
        for name in disabled_reward_names:
            term = getattr(self.rewards, name, None)
            if term is not None:
                term.weight = 0.0

        self.rewards.stand_target_pose = RewTerm(
            func=stand_target_pose_exp,
            weight=16.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names),
                "std": 0.9,
            },
        )
        self.rewards.base_height_l2.params["target_height"] = 0.53
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.weight = -26.0
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -6.0
        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -1.0
        self.rewards.body_lin_acc_l2.weight = -0.3
        self.rewards.joint_mirror.weight = -2.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "FL_(hip|thigh|calf).*"],
            ["RR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FR_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ]
        self.rewards.action_rate_l2.weight = -0.02

        # TaskB-style termination behavior.
        self.terminations.terrain_out_of_bounds = None
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["base_link", ".*_hip", ".*_thigh"],
                ),
                "threshold": 1.0,
            },
            time_out=False,
        )
        self.terminations.fall = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.18,
            },
            time_out=False,
        )

        self.disable_zero_weight_rewards()
