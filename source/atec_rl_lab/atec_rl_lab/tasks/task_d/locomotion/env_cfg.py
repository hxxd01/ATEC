"""Task D pit crossing locomotion for B2Piper (MARG Table I rewards + privileged geometry obs)."""

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

from isaaclab.sensors import RayCasterCfg, patterns

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

_MARG_LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
_MARG_FOOT_BODIES = [".*_foot"]
_MARG_TRUNK_HIP_BODIES = ["base_link", ".*_hip"]


@configclass
class TaskDPitMargRewardsCfg:
    """MARG Table I reward terms only (no rough / wheel_vel_penalty leftovers)."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_MARG_LEG_JOINTS)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_MARG_LEG_JOINTS)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_MARG_TRUNK_HIP_BODIES),
            "threshold": 1.0,
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_MARG_LEG_JOINTS)},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_MARG_FOOT_BODIES),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_MARG_FOOT_BODIES)},
    )
    feet_center = RewTerm(
        func=task_d_loco_mdp.marg_feet_center,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=_MARG_FOOT_BODIES),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=_MARG_FOOT_BODIES),
            "height_threshold": -0.2,
        },
    )
    pit_cross_success = RewTerm(
        func=task_d_loco_mdp.PitCrossSuccessBonus,
        weight=1.0,
        params={
            "reward_value": 5.0,
            "local_x_threshold": 2.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
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

    proprio: ProprioCfg = ProprioCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    height_map: HeightMapCfg = HeightMapCfg()
    critic_priv: CriticPrivCfg = CriticPrivCfg()


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
    """MARG terminations + Task D fall and local-x success."""

    bad_orientation = None
    foot_height_below = None

    fall_in_pit = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.25, "asset_cfg": SceneEntityCfg("robot")},
        time_out=False,
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_hip"]),
            "threshold": 1.0,
        },
        time_out=False,
    )
    pit_cross_success = DoneTerm(
        func=task_d_loco_mdp.pit_cross_local_x_success_done,
        params={"local_x_threshold": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        time_out=False,
    )
    stuck_timeout = DoneTerm(
        func=StuckNoProgress,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stuck_time_s": 20.0,
            "progress_eps": 0.03,
            "grace_time_s": 0.0,
        },
        time_out=False,
    )


@configclass
class UnitreeB2PiperTaskDPitLocomotionEnvCfg(UnitreeB2PiperRoughEnvCfg):
    """B2Piper crosses Task D pit with MARG Table I rewards and pit-width curriculum 0.4->1.4 m."""

    _skip_rough_reward_setup: bool = True

    observations: TaskDPitLocomotionObservationsCfg = TaskDPitLocomotionObservationsCfg()
    events: TaskDPitLocomotionEventCfg = TaskDPitLocomotionEventCfg()
    curriculum: TaskDPitLocomotionCurriculumCfg = TaskDPitLocomotionCurriculumCfg()
    terminations: TaskDPitLocomotionTerminationsCfg = TaskDPitLocomotionTerminationsCfg()
    rewards: TaskDPitMargRewardsCfg = TaskDPitMargRewardsCfg()

    pit_width_range: tuple[float, float] = (0.4, 1.4)
    platform_height_range: tuple[float, float] = (1.1, 1.1)
    pit_curriculum_levels: int = 11
    command_lin_vel_x_min: float = 0.0
    command_lin_vel_x_max: float = 4.0
    command_curriculum_start_fraction: float = 0.1
    disable_dr_and_obs_noise: bool = True
    marg_stuck_time_s: float = 20.0
    marg_episode_length_s: float = 20.0
    fall_minimum_height: float = 0.25
    pit_success_local_x: float = 2.0
    pit_cross_success_reward: float = 30.0

    def _ensure_height_scanner_for_marg_rewards(self) -> None:
        """MARG feet-center reward samples the 1.6 x 1.0 m height grid (187 rays)."""
        if self.scene.height_scanner is not None:
            return
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

    def _apply_marg_reward_table(self) -> None:
        """Enable height scanner for feet_center and restore MARG terminations."""
        self._ensure_height_scanner_for_marg_rewards()
        self._apply_marg_terminations()
        self._log_marg_reward_table()

    def _log_marg_reward_table(self) -> None:
        active = []
        for name, term in vars(self.rewards).items():
            if name.startswith("_") or term is None or callable(term):
                continue
            active.append(f"{name}={term.weight:g}")
        print(f"[TaskDPitLoco] MARG rewards ({len(active)}): {', '.join(sorted(active))}", flush=True)

    def _apply_marg_terminations(self) -> None:
        """Restore terminations (rough parent sets ``illegal_contact = None``)."""
        trunk_hip_bodies = [self.base_link_name, ".*_hip"]
        self.terminations.bad_orientation = None
        self.terminations.foot_height_below = None
        self.terminations.fall_in_pit = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": float(self.fall_minimum_height),
                "asset_cfg": SceneEntityCfg("robot"),
            },
            time_out=False,
        )
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=trunk_hip_bodies),
                "threshold": 1.0,
            },
            time_out=False,
        )
        self.terminations.pit_cross_success = DoneTerm(
            func=task_d_loco_mdp.pit_cross_local_x_success_done,
            params={
                "local_x_threshold": float(self.pit_success_local_x),
                "asset_cfg": SceneEntityCfg("robot"),
            },
            time_out=False,
        )
        self.terminations.stuck_timeout = DoneTerm(
            func=StuckNoProgress,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stuck_time_s": float(self.marg_stuck_time_s),
                "progress_eps": 0.03,
                "grace_time_s": 0.0,
            },
            time_out=False,
        )
        self.rewards.pit_cross_success = RewTerm(
            func=task_d_loco_mdp.PitCrossSuccessBonus,
            weight=1.0,
            params={
                "reward_value": float(self.pit_cross_success_reward),
                "local_x_threshold": float(self.pit_success_local_x),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

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

        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        policy_obs = getattr(self.observations, "policy", None)
        if policy_obs is not None:
            policy_obs.height_scan = None
        critic_obs = getattr(self.observations, "critic", None)
        if critic_obs is not None:
            critic_obs.height_scan = None
        self.curriculum.terrain_levels = None
        self._apply_marg_reward_table()
        # MARG table already nulls unused terms; skip disable_zero_weight_rewards (breaks on None attrs).
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
        self.episode_length_s = float(self.marg_episode_length_s)

        # No cameras / depth for this stage.
        if hasattr(self.scene, "head_camera"):
            self.scene.head_camera = None
        if hasattr(self.scene, "ee_camera"):
            self.scene.ee_camera = None

        self.terminations.terrain_out_of_bounds = None

        init_vx_max = vx_min + (vx_max - vx_min) * start_frac
        dr_msg = ", no DR/no-noise" if self.disable_dr_and_obs_noise else ""
        print(
            f"[TaskDPitLoco] MARG Table I rewards/terminations, command vx curriculum: "
            f"[{vx_min:.1f}, {init_vx_max:.1f}] -> [{vx_min:.1f}, {vx_max:.1f}] m/s, "
            f"episode={self.episode_length_s:.0f}s, stuck={self.marg_stuck_time_s:.0f}s, "
            f"spawn=TaskD default{dr_msg}, no depth",
            flush=True,
        )


@configclass
class UnitreeB2PiperTaskDPitLocomotionMargEnvCfg(UnitreeB2PiperTaskDPitLocomotionEnvCfg):
    """Task D pit locomotion with MARG asymmetric AC observations (height map + privileged critic)."""

    observations: TaskDPitMargObservationsCfg = TaskDPitMargObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Re-apply MARG height scanner (base env enables it for feet_center; ensure MARG grid params).
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        print(
            "[TaskDPitLoco-MARG] obs=proprio(43)+history(258)+height(187), "
            "critic_priv(42), MARG Table I rewards, estimator+elevation nets via MargActorCritic",
            flush=True,
        )


def refresh_task_d_pit_locomotion_terrain_cfg(env_cfg: UnitreeB2PiperTaskDPitLocomotionEnvCfg) -> None:
    """Rebuild terrain after ``scene.num_envs`` is changed."""
    env_cfg.scene.terrain = env_cfg._build_terrain_cfg()
    env_cfg.scene.terrain.max_init_terrain_level = 0
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.class_type = TaskDPitTerrainGenerator
        env_cfg.scene.terrain.terrain_generator.curriculum = True
