import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.velocity.mdp as atec_mdp
from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG
from atec_rl_lab.tasks.task_b.env_cfg import TASK_B_ROBOT_SPAWN_LOCAL
from .flat_env_cfg import UnitreeB2PiperFlatEnvCfg

_HOLD_ARM_JOINT_NAMES = (
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
    "arm_joint7",
    "arm_joint8",
)
# Keep EE relatively low while pulling shoulder/elbow aside to reduce head-camera occlusion.
_LOW_CLEAR_HOLD_ARM_ACTION = (-1.2, 4.4, -2.6, 1.0, 0.0, 0.0, 0.0, 0.0)
_HOLD_ARM_PARAMS = {
    "asset_cfg": SceneEntityCfg("robot"),
    "arm_joint_names": _HOLD_ARM_JOINT_NAMES,
    "arm_action": _LOW_CLEAR_HOLD_ARM_ACTION,
    "action_scale": atec_mdp.DETECT_HOLD_ARM_ACTION_SCALE,
}

# Task-B B2Piper: fixed root on reset (no xy/yaw/z jitter).
_TASK_B_FIXED_RESET_BASE_PARAMS = {
    "pose_range": {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    },
    "velocity_range": {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    },
}


@configclass
class UnitreeB2PiperSquatFlatEnvCfg(UnitreeB2PiperFlatEnvCfg):
    """B2Piper flat locomotion + detect-arm hold + EE height reward for Task-B touch scoring."""

    def __post_init__(self):
        super().__post_init__()

        # Root spawn aligned with Task-B B2Piper (-10, -10, 0.68) env-local.
        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(pos=TASK_B_ROBOT_SPAWN_LOCAL),
        )

        # Default viewer looks at world origin; robot is at (-10,-10) so play/video is empty.
        sx, sy, sz = TASK_B_ROBOT_SPAWN_LOCAL
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (4.0, -4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.4)

        # Randomization aligned with Task-B B2Piper (no DR / no joint reset jitter).
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_reset_joints = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None
        self.events.randomize_reset_base.params = _TASK_B_FIXED_RESET_BASE_PARAMS

        self.observations.policy.enable_corruption = False
        if getattr(self.observations, "critic", None) is not None:
            self.observations.critic.enable_corruption = False

        # Velocity commands (final curriculum targets).
        self.commands.base_velocity.ranges.lin_vel_x = (-4.0, 4.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Detect hold during rollout only (interval PD target). Reset instant stays at USD
        # default arm joints, same as Task-B before the first detect action step.
        self.events.hold_detect_arm_interval = EventTerm(
            func=atec_mdp.hold_detect_arm_pose,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params=_HOLD_ARM_PARAMS,
        )

        # Rewards: inherit flat/rough weights; disable stand-biased terms; add EE-z only.
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.ee_height_exp = RewTerm(
            func=atec_mdp.ee_height_exp,
            weight=3.0,
            params={
                "target_height": 0.24,
                "std": 0.05,
                "ee_body_name": "gripper_base",
            },
        )

        # Terminations aligned with Task-B B2Piper.
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
                "minimum_height": 0.0,
            },
            time_out=False,
        )

        self.disable_zero_weight_rewards()
