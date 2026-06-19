"""Task D pit crossing locomotion for B2Piper (flat rewards + forward progress + optional MARG obs)."""

from __future__ import annotations

import copy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab.sensors import MultiMeshRayCasterCfg, RayCasterCfg, patterns

import atec_rl_lab.tasks.task_d.locomotion.mdp as task_d_loco_mdp
import atec_rl_lab.train.locomotion.velocity.mdp as mdp
from atec_rl_lab.tasks.task_a.mdp.terminations import StuckNoProgress
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
class TaskDPitMargObservationsCfg:
    """MARG-style obs groups: proprio, history, height map, critic-only privileged state."""

    @configclass
    class ProprioCfg(ObsGroup):
        proprio = ObsTerm(
            func=task_d_loco_mdp.marg_proprio,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "joint_names": [
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ],
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioHistoryCfg(ObsGroup):
        proprio_history = ObsTerm(
            func=task_d_loco_mdp.marg_proprio_history,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "joint_names": [
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ],
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeightMapCfg(ObsGroup):
        height_map = ObsTerm(
            func=task_d_loco_mdp.marg_height_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-2.0, 2.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticPrivCfg(ObsGroup):
        critic_priv = ObsTerm(
            func=task_d_loco_mdp.marg_critic_privileged,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "joint_names": [
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ],
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        depth = ObsTerm(
            func=task_d_loco_mdp.marg_depth_flat,
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    height_map: HeightMapCfg = HeightMapCfg()
    critic_priv: CriticPrivCfg = CriticPrivCfg()
    depth: DepthCfg | None = None


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
            "post_cross_distance": 1.5,
            "success_term_name": "pit_cross_success",
        },
    )


@configclass
class TaskDPitLocomotionTerminationsCfg(TerminationsCfg):
    fall_in_pit = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.25, "asset_cfg": SceneEntityCfg("robot")},
        time_out=False,
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7, "asset_cfg": SceneEntityCfg("robot")},
        time_out=False,
    )
    pit_cross_success = DoneTerm(
        func=task_d_loco_mdp.pit_cross_local_x_success_done,
        params={"post_cross_distance": 1.5, "asset_cfg": SceneEntityCfg("robot")},
        time_out=False,
    )
    stuck_no_progress = DoneTerm(
        func=StuckNoProgress,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stuck_time_s": 2.5,
            "progress_eps": 0.03,
            "grace_time_s": 1.0,
            "command_name": "base_velocity",
            "min_cmd_speed": 0.2,
        },
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
    command_zero_speed_threshold: float = 0.0
    marg_track_lin_vel_xy_weight: float = 0.0
    marg_track_ang_vel_z_weight: float = 0.0
    marg_lin_vel_z_l2_weight: float = -2.0
    marg_ang_vel_xy_l2_weight: float = -0.05
    marg_joint_torque_l2_weight: float = -1.0e-5
    marg_action_rate_l2_weight: float = -0.01
    marg_joint_acc_l2_weight: float = -2.5e-7
    marg_collision_weight: float = -1.0
    marg_orientation_l2_weight: float = -0.2
    marg_joint_motion_limit_weight: float = -0.02
    marg_feet_air_time_weight: float = 1.0
    marg_feet_stumble_weight: float = -1.0
    marg_feet_center_weight: float = -0.01
    forward_progress_reward_weight: float = 8.0
    forward_vx_speed_capped_reward_weight: float = 10.0
    forward_vx_speed_cap_mps: float = 5.0
    upward_reward_weight: float = 2.0
    pit_cross_success_reward_weight: float = 1.0
    pit_cross_success_reward_value: float = 500.0
    pit_success_post_cross_distance: float = 1.5
    fall_minimum_height: float = 0.25
    bad_orientation_limit_angle: float = 0.7
    no_progress_stuck_time_s: float = 4.0
    no_progress_eps: float = 0.03
    no_progress_grace_time_s: float = 2.0
    no_progress_command_name: str = "base_velocity"
    no_progress_min_cmd_speed: float = 0.2
    disable_dr_and_obs_noise: bool = True

    def apply_command_config(self) -> None:
        """Sync ``commands.base_velocity.ranges`` from ``command_lin_vel_x_*`` fields.

        Call again after train/play CLI overrides (``env_cfg_cls()`` runs ``__post_init__`` first).
        """
        vx_min = float(self.command_lin_vel_x_min)
        vx_max = float(self.command_lin_vel_x_max)
        self.commands.base_velocity.ranges.lin_vel_x = (vx_min, vx_max)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.zero_cmd_speed_threshold = float(self.command_zero_speed_threshold)

    def _disable_dr_and_obs_noise(self) -> None:
        """Turn off domain randomization and observation noise/corruption only."""
        for name in dir(self.events):
            if name.startswith("_"):
                continue
            if name.startswith("randomize_") or name == "randomize_push_robot":
                setattr(self.events, name, None)

        obs_cfg = self.observations
        for group_name in ("policy", "critic", "proprio", "proprio_history", "height_map", "critic_priv"):
            group = getattr(obs_cfg, group_name, None)
            if group is None:
                continue
            if hasattr(group, "enable_corruption"):
                group.enable_corruption = False
            for term_name in dir(group):
                if term_name.startswith("_"):
                    continue
                term = getattr(group, term_name, None)
                if term is not None and hasattr(term, "noise"):
                    term.noise = None

    def _align_task_rewards(self) -> None:
        """Align reward terms to MARG Table I."""
        keep = {
            "forward_progress",
            "forward_vx_speed_capped",
            "upward",
            "pit_cross_success",
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "joint_torques_l2",
            "action_rate_l2",
            "joint_acc_l2",
            "undesired_contacts",
            "flat_orientation_l2",
            "joint_pos_penalty",
            "feet_air_time",
            "feet_stumble",
            "feet_center",
        }
        for reward_name in dir(self.rewards):
            if reward_name.startswith("_"):
                continue
            reward_term = getattr(self.rewards, reward_name, None)
            if reward_term is None or callable(reward_term):
                continue
            if reward_name not in keep:
                setattr(self.rewards, reward_name, None)

        self.rewards.forward_progress = RewTerm(
            func=task_d_loco_mdp.forward_world_x_progress,
            weight=float(self.forward_progress_reward_weight),
        )
        if float(self.forward_vx_speed_capped_reward_weight) > 0.0:
            self.rewards.forward_vx_speed_capped = RewTerm(
                func=task_d_loco_mdp.forward_world_x_speed_capped,
                weight=float(self.forward_vx_speed_capped_reward_weight),
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "v_cap": float(self.forward_vx_speed_cap_mps),
                },
            )
        else:
            self.rewards.forward_vx_speed_capped = None
        if float(self.upward_reward_weight) > 0.0:
            self.rewards.upward = RewTerm(
                func=mdp.upward,
                weight=float(self.upward_reward_weight),
            )
        else:
            self.rewards.upward = None
        if float(self.pit_cross_success_reward_weight) > 0.0:
            self.rewards.pit_cross_success = RewTerm(
                func=task_d_loco_mdp.PitCrossSuccessBonus,
                weight=float(self.pit_cross_success_reward_weight),
                params={
                    "reward_value": float(self.pit_cross_success_reward_value),
                    "post_cross_distance": float(self.pit_success_post_cross_distance),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
        else:
            self.rewards.pit_cross_success = None
        if float(self.marg_track_lin_vel_xy_weight) > 0.0:
            self.rewards.track_lin_vel_xy_exp = RewTerm(
                func=mdp.track_lin_vel_xy_exp,
                weight=float(self.marg_track_lin_vel_xy_weight),
                params={"command_name": "base_velocity", "std": 0.5},
            )
        else:
            self.rewards.track_lin_vel_xy_exp = None
        if float(self.marg_track_ang_vel_z_weight) > 0.0:
            self.rewards.track_ang_vel_z_exp = RewTerm(
                func=mdp.track_ang_vel_z_exp,
                weight=float(self.marg_track_ang_vel_z_weight),
                params={"command_name": "base_velocity", "std": 0.5},
            )
        else:
            self.rewards.track_ang_vel_z_exp = None
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=float(self.marg_lin_vel_z_l2_weight))
        self.rewards.ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=float(self.marg_ang_vel_xy_l2_weight))
        self.rewards.joint_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=float(self.marg_joint_torque_l2_weight),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)},
        )
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=float(self.marg_action_rate_l2_weight))
        self.rewards.joint_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=float(self.marg_joint_acc_l2_weight),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)},
        )
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=float(self.marg_collision_weight),
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[f"^(?!.*{self.foot_link_name}).*"]
                ),
                "threshold": 1.0,
            },
        )
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2,
            weight=float(self.marg_orientation_l2_weight),
        )
        self.rewards.joint_pos_penalty = RewTerm(
            func=mdp.joint_pos_penalty,
            weight=float(self.marg_joint_motion_limit_weight),
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names),
                "stand_still_scale": 1.0,
                "velocity_threshold": 0.0,
                "command_threshold": 0.0,
            },
        )
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=float(self.marg_feet_air_time_weight),
            params={
                "command_name": "base_velocity",
                "threshold": 0.5,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            },
        )
        self.rewards.feet_stumble = RewTerm(
            func=mdp.feet_stumble,
            weight=float(self.marg_feet_stumble_weight),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])},
        )
        self.rewards.feet_center = RewTerm(
            func=task_d_loco_mdp.marg_feet_center_penalty,
            weight=float(self.marg_feet_center_weight),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

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

        if self.disable_dr_and_obs_noise:
            self._disable_dr_and_obs_noise()

        # B2Piper-flat style: no height scan / plane-style reward sensor off.
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        policy_obs = getattr(self.observations, "policy", None)
        if policy_obs is not None:
            policy_obs.height_scan = None
        critic_obs = getattr(self.observations, "critic", None)
        if critic_obs is not None:
            critic_obs.height_scan = None
        self.curriculum.terrain_levels = None
        self.disable_zero_weight_rewards()
        # Enable velocity command curriculum only when xy tracking reward is active.
        if float(self.marg_track_lin_vel_xy_weight) > 0.0:
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
        else:
            self.curriculum.command_levels_lin_vel = None
        self.curriculum.pit_width_levels = CurrTerm(
            func=task_d_loco_mdp.task_d_pit_width_levels,
            params={
                "post_cross_distance": float(self.pit_success_post_cross_distance),
                "success_term_name": "pit_cross_success",
            },
        )
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        self._align_task_rewards()

        self.terminations.fall_in_pit = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": float(self.fall_minimum_height),
                "asset_cfg": SceneEntityCfg("robot"),
            },
            time_out=False,
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={
                "limit_angle": float(self.bad_orientation_limit_angle),
                "asset_cfg": SceneEntityCfg("robot"),
            },
            time_out=False,
        )
        self.terminations.pit_cross_success = DoneTerm(
            func=task_d_loco_mdp.pit_cross_local_x_success_done,
            params={
                "post_cross_distance": float(self.pit_success_post_cross_distance),
                "asset_cfg": SceneEntityCfg("robot"),
            },
            time_out=False,
        )
        self.terminations.stuck_no_progress = DoneTerm(
            func=StuckNoProgress,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stuck_time_s": float(self.no_progress_stuck_time_s),
                "progress_eps": float(self.no_progress_eps),
                "grace_time_s": float(self.no_progress_grace_time_s),
                "command_name": str(self.no_progress_command_name),
                "min_cmd_speed": float(self.no_progress_min_cmd_speed),
            },
            time_out=False,
        )

        vx_min = float(self.command_lin_vel_x_min)
        vx_max = float(self.command_lin_vel_x_max)
        start_frac = float(self.command_curriculum_start_fraction)
        self.apply_command_config()

        self.scene.env_spacing = max(self.scene.env_spacing, 12.0)
        self.episode_length_s = 12.0

        # No cameras / depth for this stage.
        if hasattr(self.scene, "head_camera"):
            self.scene.head_camera = None
        if hasattr(self.scene, "ee_camera"):
            self.scene.ee_camera = None

        self.terminations.terrain_out_of_bounds = None

        init_vx_max = vx_min + (vx_max - vx_min) * start_frac
        dr_msg = ", no DR/no-noise" if self.disable_dr_and_obs_noise else ""
        print(
            f"[TaskDPitLoco] command vx curriculum: [{vx_min:.1f}, {init_vx_max:.1f}] -> [{vx_min:.1f}, {vx_max:.1f}] m/s, "
            "rewards=MARG Table I aligned "
            f"+ task-drive(forward={self.forward_progress_reward_weight}, "
            f"vx_capped={self.forward_vx_speed_capped_reward_weight}@cap={self.forward_vx_speed_cap_mps:.1f}, "
            f"upward={self.upward_reward_weight}, "
            f"pit_success={self.pit_cross_success_reward_weight}x{self.pit_cross_success_reward_value:.0f}) "
            f"(lin_xy={self.marg_track_lin_vel_xy_weight}, ang_z={self.marg_track_ang_vel_z_weight}, "
            f"feet_center={self.marg_feet_center_weight}), "
            f"terminations=fall_in_pit(z<{self.fall_minimum_height})+"
            f"bad_orientation(>{self.bad_orientation_limit_angle:.2f}rad)+"
            f"stuck_no_progress({self.no_progress_stuck_time_s:.1f}s)+pit_cross_success, "
            f"spawn=TaskD default{dr_msg}, no depth",
            flush=True,
        )


def configure_pit_marg_height_scanner(env_cfg, *, include_box: bool = False) -> None:
    """Configure MARG elevation scanner: ground only, or ground + per-env Task D box."""
    prim_path = "{ENV_REGEX_NS}/Robot/" + env_cfg.base_link_name
    offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))
    pattern = patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0])

    if include_box:
        env_cfg.scene.height_scanner = MultiMeshRayCasterCfg(
            prim_path=prim_path,
            offset=offset,
            ray_alignment="yaw",
            pattern_cfg=pattern,
            debug_vis=False,
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/ground",
                    is_shared=True,
                    track_mesh_transforms=False,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Box",
                    track_mesh_transforms=True,
                ),
            ],
        )
    else:
        env_cfg.scene.height_scanner = RayCasterCfg(
            prim_path=prim_path,
            offset=offset,
            ray_alignment="yaw",
            pattern_cfg=pattern,
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
    env_cfg.scene.height_scanner.update_period = env_cfg.decimation * env_cfg.sim.dt


@configclass
class UnitreeB2PiperTaskDPitLocomotionMargEnvCfg(UnitreeB2PiperTaskDPitLocomotionEnvCfg):
    """Task D pit locomotion with MARG asymmetric AC observations (height map + privileged critic)."""

    observations: TaskDPitMargObservationsCfg = TaskDPitMargObservationsCfg()
    depth_policy_h: int = 24
    depth_policy_w: int = 32
    depth_render_h: int | None = None
    depth_render_w: int | None = None
    depth_max: float = 5.0
    depth_only: bool = True
    head_depth_only: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Height scanner for MARG elevation obs (not used by B2Piper-flat rewards).
        configure_pit_marg_height_scanner(self, include_box=False)

        print(
            "[TaskDPitLoco-MARG] obs=proprio(43)+history(258)+height(187), "
            "critic_priv(42), MARG Table-I rewards, MargActorCritic",
            flush=True,
        )


def refresh_task_d_pit_locomotion_terrain_cfg(env_cfg: UnitreeB2PiperTaskDPitLocomotionEnvCfg) -> None:
    """Rebuild terrain after ``scene.num_envs`` is changed."""
    env_cfg.scene.terrain = env_cfg._build_terrain_cfg()
    env_cfg.scene.terrain.max_init_terrain_level = 0
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.class_type = TaskDPitTerrainGenerator
        env_cfg.scene.terrain.terrain_generator.curriculum = True
