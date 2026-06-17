"""Task D pit crossing locomotion for B2Piper (flat-style rewards + privileged geometry obs)."""

from __future__ import annotations

import copy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import atec_rl_lab.tasks.task_d.locomotion.mdp as task_d_loco_mdp
import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.tasks.task_d.locomotion.terrain_curriculum import (
    TaskDPitCurriculumTerrainImporter,
    TaskDPitTerrainGenerator,
)
from atec_rl_lab.tasks.task_d.terrain import (
    TASK_D_TERRAIN_CFG,
    PitAndPlatformTerrainCfg,
    configure_task_d_terrain_for_num_envs,
)
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.rough_env_cfg import (
    UnitreeB2PiperRoughEnvCfg,
)
from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import (
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    TerminationsCfg,
)


@configclass
class TaskDPitLocomotionObservationsCfg(ObservationsCfg):
    """B2Piper-flat policy/critic obs plus privileged robot pose and pit geometry on both groups."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        robot_world_pos = ObsTerm(
            func=task_d_loco_mdp.robot_world_position,
            clip=(-500.0, 500.0),
            scale=0.01,
        )
        pit_geometry = ObsTerm(
            func=task_d_loco_mdp.pit_geometry,
            clip=(-500.0, 500.0),
            scale=0.1,
        )

        def __post_init__(self):
            super().__post_init__()
            self.height_scan = None
            self.enable_corruption = False
            for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
                term = getattr(self, term_name, None)
                if term is not None:
                    term.noise = None

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        robot_world_pos = ObsTerm(
            func=task_d_loco_mdp.robot_world_position,
            clip=(-500.0, 500.0),
            scale=0.01,
        )
        pit_geometry = ObsTerm(
            func=task_d_loco_mdp.pit_geometry,
            clip=(-500.0, 500.0),
            scale=0.1,
        )

        def __post_init__(self):
            super().__post_init__()
            self.height_scan = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TaskDPitLocomotionEventCfg(EventCfg):
    """Task D default spawn; DR cleared in env ``__post_init__`` after parent setup."""

    reset_robot_task_d = EventTerm(
        func=task_d_loco_mdp.reset_robot_at_task_d_spawn,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reset_joints_default = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TaskDPitLocomotionCurriculumCfg(CurriculumCfg):
    terrain_levels = None
    command_levels_ang_vel = None
    command_levels_lin_vel = CurrTerm(
        func=mdp.command_levels_lin_vel,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "range_multiplier": (0.1, 1.0),
            "success_fraction": 0.6,
        },
    )
    pit_width_levels = CurrTerm(
        func=task_d_loco_mdp.task_d_pit_width_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "success_fraction": 0.75,
        },
    )


@configclass
class TaskDPitLocomotionTerminationsCfg(TerminationsCfg):
    fall_in_pit = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.25, "asset_cfg": SceneEntityCfg("robot")},
        time_out=False,
    )


@configclass
class UnitreeB2PiperTaskDPitLocomotionEnvCfg(UnitreeB2PiperRoughEnvCfg):
    """B2Piper crosses Task D pit: flat rewards + forward progress, pit-width curriculum 0.4->1.4 m."""

    observations: TaskDPitLocomotionObservationsCfg = TaskDPitLocomotionObservationsCfg()
    events: TaskDPitLocomotionEventCfg = TaskDPitLocomotionEventCfg()
    curriculum: TaskDPitLocomotionCurriculumCfg = TaskDPitLocomotionCurriculumCfg()
    terminations: TaskDPitLocomotionTerminationsCfg = TaskDPitLocomotionTerminationsCfg()

    pit_width_range: tuple[float, float] = (0.4, 1.4)
    platform_height_range: tuple[float, float] = (1.1, 1.1)
    pit_curriculum_levels: int = 11
    command_lin_vel_x_min: float = 0.0
    command_lin_vel_x_max: float = 4.0
    command_curriculum_start_fraction: float = 0.1
    upward_reward_weight: float = 3.0
    track_lin_vel_xy_reward_weight: float = 6.0
    forward_progress_reward_weight: float = 8.0

    def _build_terrain_cfg(self):
        num_envs = int(self.scene.num_envs)
        terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
        terrain_cfg.class_type = TaskDPitCurriculumTerrainImporter
        terrain_cfg.terrain_generator.class_type = TaskDPitTerrainGenerator
        configure_task_d_terrain_for_num_envs(
            terrain_cfg,
            num_envs,
            curriculum_levels=self.pit_curriculum_levels,
        )
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = self.pit_width_range
            pit_cfg.platform_height_range = self.platform_height_range
        nrow = terrain_cfg.terrain_generator.num_rows
        ncol = terrain_cfg.terrain_generator.num_cols
        print(
            f"[TaskDPitLoco] terrain grid {nrow}x{ncol} for num_envs={num_envs} "
            f"pit_width={self.pit_width_range} curriculum_levels={self.pit_curriculum_levels}",
            flush=True,
        )
        return terrain_cfg

    def __post_init__(self):
        self.scene.terrain = self._build_terrain_cfg()
        self.scene.terrain.max_init_terrain_level = 0

        super().__post_init__()

        # Disable all domain randomization from rough defaults.
        for event_name in (
            "randomize_rigid_body_material",
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_others",
            "randomize_com_positions",
            "randomize_apply_external_force_torque",
            "randomize_actuator_gains",
            "randomize_reset_base",
            "randomize_reset_joints",
            "randomize_push_robot",
        ):
            setattr(self.events, event_name, None)

        # B2Piper-flat style: no height scan / plane-style reward sensor off.
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        self.disable_zero_weight_rewards()
        # Parent rough cfg disables command curriculum; restore for pit locomotion.
        self.curriculum.command_levels_lin_vel = CurrTerm(
            func=mdp.command_levels_lin_vel,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "range_multiplier": (
                    float(self.command_curriculum_start_fraction),
                    1.0,
                ),
                "success_fraction": 0.6,
            },
        )
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        self.rewards.forward_progress = RewTerm(
            func=task_d_loco_mdp.forward_world_x_progress,
            weight=float(self.forward_progress_reward_weight),
        )
        self.rewards.upward.weight = float(self.upward_reward_weight)
        self.rewards.track_lin_vel_xy_exp.weight = float(self.track_lin_vel_xy_reward_weight)

        vx_min = float(self.command_lin_vel_x_min)
        vx_max = float(self.command_lin_vel_x_max)
        start_frac = float(self.command_curriculum_start_fraction)
        self.commands.base_velocity.ranges.lin_vel_x = (vx_min, vx_max)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)

        self.scene.env_spacing = max(self.scene.env_spacing, 12.0)
        self.episode_length_s = 12.0

        # No cameras / depth for this stage.
        if hasattr(self.scene, "head_camera"):
            self.scene.head_camera = None
        if hasattr(self.scene, "ee_camera"):
            self.scene.ee_camera = None

        self.terminations.terrain_out_of_bounds = None

        init_vx_max = vx_min + (vx_max - vx_min) * start_frac
        print(
            f"[TaskDPitLoco] command vx curriculum: [{vx_min:.1f}, {init_vx_max:.1f}] -> [{vx_min:.1f}, {vx_max:.1f}] m/s, "
            f"rewards: forward={self.forward_progress_reward_weight}, "
            f"track_vel_xy={self.track_lin_vel_xy_reward_weight}, upward={self.upward_reward_weight}, "
            f"spawn=TaskD default, obs=flat+priv no-noise, no DR, no depth",
            flush=True,
        )


def refresh_task_d_pit_locomotion_terrain_cfg(env_cfg: UnitreeB2PiperTaskDPitLocomotionEnvCfg) -> None:
    """Rebuild terrain after ``scene.num_envs`` is changed."""
    env_cfg.scene.terrain = env_cfg._build_terrain_cfg()
    env_cfg.scene.terrain.max_init_terrain_level = 0
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.class_type = TaskDPitTerrainGenerator
        env_cfg.scene.terrain.terrain_generator.curriculum = True
