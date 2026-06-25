import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeB2PiperFlatEnvCfg


def squat_target_pose_exp(
    env,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_offsets: tuple[float, ...],
    std: float = 0.8,
):
    """Exponential reward for reaching the desired squat pose."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    target = q_default + q.new_tensor(target_offsets).unsqueeze(0)
    err = torch.sum(torch.square(q - target), dim=1)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return torch.exp(-err / (std**2)) * upright


@configclass
class UnitreeB2PiperSquatFlatEnvCfg(UnitreeB2PiperFlatEnvCfg):
    """Flat B2Piper env tuned for stand-to-squat transition learning."""

    def __post_init__(self):
        super().__post_init__()

        # Zero command: focus on posture transition instead of locomotion tracking.
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

        # Standing-nearby reset randomization for transition robustness.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.40, 0.40),
            "y": (-0.40, 0.40),
            "z": (-0.05, 0.10),
            "roll": (-0.28, 0.28),
            "pitch": (-0.28, 0.28),
            "yaw": (-3.14, 3.14),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.50, 0.50),
            "y": (-0.50, 0.50),
            "z": (-0.30, 0.30),
            "roll": (-0.50, 0.50),
            "pitch": (-0.50, 0.50),
            "yaw": (-0.50, 0.50),
        }
        self.events.randomize_reset_joints.params["position_range"] = (0.95, 1.05)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.10, 0.10)
        # Add more reset diversity so leg policy remains stable when arm motion perturbs trunk.
        self.events.randomize_reset_joints.params["position_range"] = (0.80, 1.20)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.50, 0.50)

        # Mild disturbances to improve robustness when the arm starts moving.
        self.events.randomize_push_robot.interval_range_s = (1.8, 3.5)
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.18, 0.18),
            "y": (-0.18, 0.18),
            "z": (-0.20, 0.20),
        }
        self.events.randomize_apply_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["base_link"]
        )
        # 3D force randomization (including vertical/z component).
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-12.0, 12.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-6.0, 6.0)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.08, 0.08),
            "z": (-0.08, 0.08),
        }
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=["(FR|FL|RR|RL)_(hip|thigh|calf)_joint"],
        )
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.50, 2.00)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.50, 2.00)

        # Reward shaping:
        # Keep only a direct "lower base_z is better" objective.
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

        self.rewards.squat_target_pose = None
        # Track squat base height around z=0.18 instead of collapsing lower.
        self.rewards.base_height_l2.params["target_height"] = 0.18
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.weight = -36.0
        # Keep velocity tracking as an important secondary objective.
        self.rewards.track_lin_vel_xy_exp.weight = 8.0
        self.rewards.track_ang_vel_z_exp.weight = 4.0
        # Reduce body jitter/shake during squat.
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.6
        self.rewards.body_lin_acc_l2.weight = -0.2
        # Strong symmetry constraints for both left-right and front-back leveling.
        self.rewards.joint_mirror.weight = -4.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "FL_(hip|thigh|calf).*"],  # left-right (front)
            ["RR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],  # left-right (rear)
            ["FR_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],  # front-back (right)
            ["FL_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],  # front-back (left)
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
                "minimum_height": 0.15,
            },
            time_out=False,
        )

        # Drop any leftover zero-weight reward terms to avoid resolving
        # irrelevant configs in this specialized squat task.
        self.disable_zero_weight_rewards()
