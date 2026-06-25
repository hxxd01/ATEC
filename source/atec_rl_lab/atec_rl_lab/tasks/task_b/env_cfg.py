# Created by skywoodsz on 2026/02/09.

from __future__ import annotations

import copy

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from .terrain import (
    TASK_B_TERRAIN_CFG,
    TASK_B_CELL_SIZE,
    configure_task_b_terrain_for_num_envs,
    task_b_terrain_grid_shape,
)
from atec_rl_lab.tasks.task_base import BaseEnvCfg, BaseSceneCfg
from atec_rl_lab.tasks.task_base.envs_base_cfg import TerminationsCfg as BaseTerminationsCfg
from atec_rl_lab.assets.objects import Sugar_cfg, Mustard_cfg, Banana_cfg, Cracker_cfg
from atec_rl_lab.tasks.task_b.mdp.terminations import ObjectsInCircleDone
from atec_rl_lab.tasks.task_d.env_cfg import reset_root_state_at_env_origin, apply_task_d_camera_depth_clip
from atec_rl_lab.train.locomotion.velocity.mdp.events import (
    DETECT_HOLD_ARM_ACTION,
    DETECT_HOLD_ARM_ACTION_SCALE,
    hold_detect_arm_pose,
)
import atec_rl_lab.tasks.task_b.mdp as atec_mdp

TARGET_CENTER = (-3.0, -10.0)
TARGET_MARKER_Z = 0.06
TASK_B_ROBOT_SPAWN_LOCAL = (-10.0, -10.0, 0.68)
TASK_B_NAV_DEPTH_MAX = 5.0
TASK_B_PLATFORM_CAMERA_FAR = 50.0


def refresh_task_b_terrain_cfg(env_cfg: "TaskBEnvCfg") -> None:
    """Rebuild terrain grid after ``scene.num_envs`` is set. Call before ``gym.make``."""
    env_cfg.scene.terrain = env_cfg._build_terrain_cfg()


def apply_task_b_nav_train_overrides(env_cfg: "TaskBEnvCfg") -> None:
    """Disable base MDP rewards / success done; wrapper supplies nav rewards."""
    if getattr(env_cfg, "rewards", None) is not None:
        for name in ("objects_in_circle", "grasped_objects"):
            term = getattr(env_cfg.rewards, name, None)
            if term is not None:
                term.weight = 0.0
    if getattr(env_cfg, "terminations", None) is not None:
        env_cfg.terminations.objects_in_circle_done = None
        if getattr(env_cfg.terminations, "fall", None) is not None:
            env_cfg.terminations.fall.params["minimum_height"] = 0.0

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    objects_in_circle = RewTerm(
        func=atec_mdp.ObjectsInCircle,
        params={"center": TARGET_CENTER,
                "radius": 1.0,
                "reward_per_object": 1.0},
        weight=1.0,
    )
    grasped_objects = RewTerm(
        func=atec_mdp.GraspedObjectsByEE,
        params={
            "ee_body_name": "gripper_base",
            "grasp_dist_thresh": 0.20,
            "reward_per_object": 1.0,
        },
        weight=1.0,
    )


@configclass
class TaskBTerminationsCfg(BaseTerminationsCfg):
    objects_in_circle_done = DoneTerm(
        func=ObjectsInCircleDone,
        params={"center": TARGET_CENTER, "radius": 1.0},
        time_out=False,
    )


@configclass
class TaskBEnvCfg(BaseEnvCfg):
    """Environment base configuration for Task B."""

    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=2.5)

    def _build_terrain_cfg(self):
        num_envs = int(self.scene.num_envs)
        terrain_cfg = copy.deepcopy(TASK_B_TERRAIN_CFG)
        configure_task_b_terrain_for_num_envs(terrain_cfg, num_envs)
        nrow, ncol = task_b_terrain_grid_shape(num_envs)
        print(
            f"[TaskBEnv] terrain grid {nrow}x{ncol} for num_envs={num_envs} "
            f"(playable 20x20, cell={TASK_B_CELL_SIZE} incl. gap)",
            flush=True,
        )
        return terrain_cfg

    def _spawn_task_b_objects(self) -> None:
        import numpy as np

        rng = np.random.default_rng(seed=self.seed)
        SUGAR_QUAT = [0.0, 0.707, 0.0, 0.707]
        OTHER_QUAT = [0.0, 0.0, -0.707, 0.707]

        for i in range(18):
            x = rng.uniform(-15.0, -5.0)
            y = rng.uniform(-15.0, -5.0)
            if abs(x) < 1.0 and abs(y) < 1.0:
                x += 2.0
            name = f"Object{i + 1}"
            if i < 6:
                cfg = Sugar_cfg([x, y, 0.15], SUGAR_QUAT, name)
            elif i < 12:
                cfg = Mustard_cfg([x, y, 0.10], OTHER_QUAT, name)
            else:
                cfg = Banana_cfg([x, y, 0.10], OTHER_QUAT, name)
            setattr(self.scene, f"object_{i + 1}", cfg)

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = self._build_terrain_cfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        self.rewards = RewardsCfg()
        self.terminations = TaskBTerminationsCfg()

        # Turn off the DR and noise
        self.observations.proprio.enable_corruption = False
        self.observations.extero.enable_corruption = False
        self.observations.image.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints = None

        self._spawn_task_b_objects()


@configclass
class TaskBEnvG1Cfg(TaskBEnvCfg):
    """Environment configuration for Task C with Unitree g1."""

    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG
        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_G1_29DOF_DEX1_CFG.init_state.replace(
                pos=(-10, -10, 0.9),
            )
        )
        super().__post_init__()

        self.rewards.grasped_objects.params["ee_body_name"] = (
            "left_hand_base_link",
            "right_hand_base_link",
        )
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_G1_29DOF_DEX1_CFG.base_link_name,
            ".*_hip_(pitch|roll|yaw)_link"
        ]

        joint_names = UNITREE_G1_29DOF_DEX1_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_pos_leg.joint_names = joint_names
        self.actions.joint_vel_wheel = None
        self.actions.joint_pos_arm = None


@configclass
class TaskBEnvTron1Cfg(TaskBEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON1A_PIPER_CFG

        self.scene.robot = TRON1A_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON1A_PIPER_CFG.init_state.replace(
                pos=(-10, -10, 0.9 + 0.166),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            TRON1A_PIPER_CFG.base_link_name,
            "abad_[LR]_Link",
        ]

        joint_names = TRON1A_PIPER_CFG.joint_names
        leg_joint_names = TRON1A_PIPER_CFG.leg_joint_names
        wheel_joint_names = TRON1A_PIPER_CFG.wheel_joint_names
        arm_joint_names = TRON1A_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_pos_leg.joint_names = leg_joint_names
        self.actions.joint_vel_wheel.joint_names = wheel_joint_names
        self.actions.joint_pos_arm.joint_names = arm_joint_names

@configclass
class TaskBEnvB2Cfg(TaskBEnvCfg):
    def __post_init__(self):

        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=(-10, -10, 0.68),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh",
        ]

        joint_names = UNITREE_B2_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_B2_PIPER_CFG.leg_joint_names
        arm_joint_names = UNITREE_B2_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_pos_leg.joint_names = leg_joint_names
        self.actions.joint_pos_arm.joint_names = arm_joint_names
        self.actions.joint_vel_wheel = None


@configclass
class TaskBNavEnvB2Cfg(TaskBEnvB2Cfg):
    """Task B B2 nav training: multi-tile terrain, random trash on reset, detect arm hold."""

    scene: BaseSceneCfg = BaseSceneCfg(num_envs=512, env_spacing=float(TASK_B_CELL_SIZE[0]))

    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=TASK_B_ROBOT_SPAWN_LOCAL,
            ),
        )
        super().__post_init__()

        apply_task_b_nav_train_overrides(self)
        apply_task_d_camera_depth_clip(self.scene, TASK_B_PLATFORM_CAMERA_FAR)

        self.events.reset_robot_joints = None
        self.events.reset_robot_root = EventTerm(
            func=reset_root_state_at_env_origin,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "local_pos": TASK_B_ROBOT_SPAWN_LOCAL,
            },
        )
        self.events.randomize_task_b_objects = EventTerm(
            func=atec_mdp.randomize_task_b_objects,
            mode="reset",
        )
        hold_arm_params = {
            "asset_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": (
                "arm_joint1",
                "arm_joint2",
                "arm_joint3",
                "arm_joint4",
                "arm_joint5",
                "arm_joint6",
                "arm_joint7",
                "arm_joint8",
            ),
            "arm_action": DETECT_HOLD_ARM_ACTION,
            "action_scale": DETECT_HOLD_ARM_ACTION_SCALE,
        }
        self.events.hold_detect_arm_interval = EventTerm(
            func=hold_detect_arm_pose,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params=hold_arm_params,
        )


@configclass
class TaskBEnvB2WCfg(TaskBEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG

        self.scene.robot = UNITREE_B2W_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2W_PIPER_CFG.init_state.replace(
                pos=(-10, -10, 0.78),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2W_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh",
        ]

        joint_names = UNITREE_B2W_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_B2W_PIPER_CFG.leg_joint_names
        wheel_joint_names = UNITREE_B2W_PIPER_CFG.wheel_joint_names
        arm_joint_names = UNITREE_B2W_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_pos_leg.joint_names = leg_joint_names
        self.actions.joint_vel_wheel.joint_names = wheel_joint_names
        self.actions.joint_pos_arm.joint_names = arm_joint_names
