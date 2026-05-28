# Created by skywoodsz on 2026/02/06.

"""
Implementation of Task D environment configuration with different robots.
"""

import copy
import torch
from isaaclab.utils import configclass
import atec_rl_lab.tasks.task_d.mdp as atec_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_base import BaseEnvCfg
from atec_rl_lab.tasks.task_base.envs_base_cfg import BaseSceneCfg
from atec_rl_lab.tasks.task_base.envs_base_cfg import TerminationsCfg as BaseTerminationsCfg
from .terrain import (
    TASK_D_CELL_SIZE,
    TASK_D_TERRAIN_CFG,
    PitAndPlatformTerrainCfg,
    configure_task_d_terrain_for_num_envs,
    task_d_terrain_grid_shape,
)

# World spawn used by nav scripts / teacher; local offset is relative to each env's terrain origin.
# For num_rows=1,num_cols=1 pit tile, terrain_origins[0,0] ~= (-4.2, 0) after generator centering (see debug_taskd_env_origins.py).
_TASK_D_PIT_TERRAIN_ORIGIN_XY = (-4.2, 0.0)
_TASK_D_ROBOT_SPAWN_WORLD = (-3.0, 0.0, 0.8)
_TASK_D_BOX_SPAWN_WORLD = (-3.0, 1.6, 0.5)


def task_d_spawn_local(world_pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert nominal world spawn on the pit tile to env-local offset (env_origin + local)."""
    ox, oy = _TASK_D_PIT_TERRAIN_ORIGIN_XY
    return (world_pos[0] - ox, world_pos[1] - oy, world_pos[2])


TASK_D_ROBOT_SPAWN_LOCAL = task_d_spawn_local(_TASK_D_ROBOT_SPAWN_WORLD)
TASK_D_BOX_SPAWN_LOCAL = task_d_spawn_local(_TASK_D_BOX_SPAWN_WORLD)
_SPAWN_DIAG_PRINTED: set[str] = set()


def reset_root_state_at_env_origin(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    local_pos: tuple[float, float, float] | None = None,
):
    """Reset asset root pose to ``env_origin + local_pos`` (per-env world spawn on each pit tile)."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]
    if local_pos is None:
        positions = root_states[:, 0:3] + env_origins
    else:
        local = torch.tensor(local_pos, device=asset.device, dtype=root_states.dtype).view(1, 3).expand(n, 3)
        positions = env_origins + local
    orientations = root_states[:, 3:7]
    velocities = torch.zeros(n, 6, device=asset.device, dtype=root_states.dtype)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    if asset_cfg.name not in _SPAWN_DIAG_PRINTED:
        w0 = positions[0].detach().cpu().tolist()
        o0 = env_origins[0].detach().cpu().tolist()
        print(
            f"[TaskDSpawn] {asset_cfg.name}: env_origin={o0[:3]}  "
            f"local={list(local_pos) if local_pos is not None else 'default_root_state[:3]'}  "
            f"world={w0[:3]}  "
            f"(origins x:[{env_origins[:, 0].min():.2f},{env_origins[:, 0].max():.2f}])",
            flush=True,
        )
        _SPAWN_DIAG_PRINTED.add(asset_cfg.name)


def reset_root_state_absolute(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    world_pos: tuple[float, float, float] = (-3.0, 0.0, 0.8),
):
    """Reset asset root pose to a fixed world position (ignore env_origin offset)."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    positions = torch.tensor(world_pos, device=asset.device, dtype=root_states.dtype).view(1, 3).repeat(n, 1)
    orientations = root_states[:, 3:7]
    velocities = torch.zeros(n, 6, device=asset.device, dtype=root_states.dtype)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    achieve = RewTerm(
        func=atec_mdp.RewardCrossX,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "threshold": [-1.4, 2.0],
                "reward_value": [2, 20.0],
                "debug": False,
                "visual_assets": True,
                },
        weight=1.0,
    )
    box_in_target_x = RewTerm(
        func=atec_mdp.RewardBoxXInRange,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "x_min": [-0.7, -1.4],
            "x_max": [0.7, -0.7],
            "reward_value": 14.0,
            "one_time": True,
            "debug": False,
        },
        weight=1.0,
    )

@configclass
class TaskDTerminationsCfg(BaseTerminationsCfg):
    x_reached = DoneTerm(
        func=atec_mdp.robot_x_greater_than,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_threshold": 3.5,
        },
        time_out=False,
    )
    no_motion_timeout = DoneTerm(
        func=atec_mdp.NoMotionTimeout,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stuck_time_s": 1.5,
            "speed_eps": 0.03,
        },
        time_out=False,
    )
    stage_target_deviation = DoneTerm(
        func=atec_mdp.StageTargetDeviationTermination,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "max_dist": 0.8,
            "stage_idx_attr": "_nav_stage_idx",
        },
        time_out=False,
    )
    no_target_progress_timeout = DoneTerm(
        func=atec_mdp.NoTargetProgressTimeout,
        params={
            "stuck_time_s": 2.0,
            "progress_eps": 0.05,
        },
        time_out=False,
    )
    no_push_progress_timeout = DoneTerm(
        func=atec_mdp.PushStageStuckTimeout,
        params={
            "stuck_time_s": 2.0,
            "progress_eps": 0.05,
            "align_tol": 0.15,
        },
        time_out=False,
    )

def refresh_task_d_terrain_cfg(env_cfg: "TaskDEnvCfg") -> None:
    """Rebuild terrain grid after ``scene.num_envs`` is set. Call before ``gym.make``."""
    env_cfg.scene.terrain = env_cfg._build_terrain_cfg()


@configclass
class TaskDEnvCfg(BaseEnvCfg):
    """Task D defaults: 512 envs, 10 m clone spacing; terrain grid sized in ``_build_terrain_cfg``."""

    scene: BaseSceneCfg = BaseSceneCfg(num_envs=512, env_spacing=10.0)
    pit_width_range: tuple[float, float] = (1.3, 1.4)
    platform_height_range: tuple[float, float] = (1.0, 1.2)

    def _build_terrain_cfg(self):
        num_envs = int(self.scene.num_envs)
        terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
        configure_task_d_terrain_for_num_envs(terrain_cfg, num_envs)
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = self.pit_width_range
            pit_cfg.platform_height_range = self.platform_height_range
        nrow, ncol = task_d_terrain_grid_shape(num_envs)
        print(
            f"[TaskDEnv] terrain grid {nrow}x{ncol} for num_envs={num_envs} "
            f"(playable 12x8, cell={TASK_D_CELL_SIZE} incl. gap)",
            flush=True,
        )
        return terrain_cfg

    def __post_init__(self):
        super().__post_init__()
        refresh_task_d_terrain_cfg(self)

        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 1.0, 0.6),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=TASK_D_BOX_SPAWN_LOCAL,
            ),
        )
        self.sim.physics_material = self.scene.terrain.physics_material

        # Task D reward
        self.rewards = RewardsCfg()
        self.terminations = TaskDTerminationsCfg()

        # Turn off the DR and noise
        self.observations.proprio.enable_corruption = False
        self.observations.extero.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None

        # Trun off terminations
        self.terminations.illegal_contact = None
        self.terminations.fall.params["minimum_height"] = 0.25
        # Disable navigation-related terminations during nav training/debug.
        self.terminations.x_reached = None
        self.terminations.no_motion_timeout = None
        self.terminations.stage_target_deviation = None

        # Spawn at env_origin + local offset (same world pose as (-3,0)/( -3,1.6) on the reference pit tile).
        self.events.reset_robot_root = EventTerm(
            func=reset_root_state_at_env_origin,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "local_pos": TASK_D_ROBOT_SPAWN_LOCAL,
            },
        )
        self.events.reset_box_root = EventTerm(
            func=reset_root_state_at_env_origin,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "local_pos": TASK_D_BOX_SPAWN_LOCAL,
            },
        )

@configclass
class TaskDEnvG1Cfg(TaskDEnvCfg):
    """Environment configuration for Task C with Unitree g1."""

    pit_width_range: tuple[float, float] = (0.9, 1.0)
    platform_height_range: tuple[float, float] = (0.9, 1.0)

    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG

        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state = UNITREE_G1_29DOF_DEX1_CFG.init_state.replace(
                pos=TASK_D_ROBOT_SPAWN_LOCAL,
            )
        )
        super().__post_init__()

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     UNITREE_G1_29DOF_DEX1_CFG.base_link_name,
        #     ".*_hip_(pitch|roll|yaw)_link"
        # ]

        joint_names = UNITREE_G1_29DOF_DEX1_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_pos_leg.joint_names = joint_names
        self.actions.joint_vel_wheel = None
        self.actions.joint_pos_arm = None


@configclass
class TaskDEnvTron1Cfg(TaskDEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON1A_PIPER_CFG

        self.scene.robot = TRON1A_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state = TRON1A_PIPER_CFG.init_state.replace(
                pos=task_d_spawn_local((-3.0, 0.0, 0.8 + 0.166)),
            )
        )
        super().__post_init__()

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     TRON1A_PIPER_CFG.base_link_name,
        #     "abad_[LR]_Link",
        # ]

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
class TaskDEnvB2Cfg(TaskDEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=TASK_D_ROBOT_SPAWN_LOCAL,
            )
        )

        super().__post_init__()

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     UNITREE_B2_PIPER_CFG.base_link_name,
        #     ".*_hip",
        #     ".*_thigh",
        # ]

        joint_names = UNITREE_B2_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_B2_PIPER_CFG.leg_joint_names
        arm_joint_names = UNITREE_B2_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_pos_leg.joint_names = leg_joint_names
        self.actions.joint_pos_arm.joint_names = arm_joint_names
        self.actions.joint_vel_wheel = None


@configclass
class TaskDEnvB2WCfg(TaskDEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG

        self.scene.robot = UNITREE_B2W_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state = UNITREE_B2W_PIPER_CFG.init_state.replace(
                pos=task_d_spawn_local((-3.0, 0.0, 0.78)),
            )
        )
        super().__post_init__()

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     UNITREE_B2W_PIPER_CFG.base_link_name,
        #     ".*_hip",
        #     ".*_thigh",
        # ]

        joint_names = UNITREE_B2W_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_B2W_PIPER_CFG.leg_joint_names
        wheel_joint_names = UNITREE_B2W_PIPER_CFG.wheel_joint_names
        arm_joint_names = UNITREE_B2W_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_pos_leg.joint_names = leg_joint_names
        self.actions.joint_vel_wheel.joint_names = wheel_joint_names
        self.actions.joint_pos_arm.joint_names = arm_joint_names
