"""Task D platform end-to-end env: B2 locomotion rewards + MARG proprio obs (no perception)."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import atec_rl_lab.tasks.task_d.locomotion.mdp as task_d_loco_mdp
from atec_rl_lab.tasks.task_d.env_cfg import (
    TaskDEnvCfg,
    TaskDTerminationsCfg,
    TASK_D_ROBOT_SPAWN_LOCAL,
    refresh_task_d_terrain_cfg,
)
from atec_rl_lab.tasks.task_base.envs_base_cfg import ActionsCfg as TaskDActionsCfg
from atec_rl_lab.tasks.task_base.envs_base_cfg import ObservationsCfg
from atec_rl_lab.train.locomotion.marg.constants import (
    MARG_CRITIC_PRIV_DIM,
    MARG_HISTORY_DIM,
    MARG_PROPRIO_DIM,
)

E2E_EPISODE_LENGTH_S: float = 8.0
# Nav progress timeout (2s) is for teacher/high-level cmd; disabled in e2e — random 12D legs
# cannot shrink _nav_dist_to_target by 0.05m in 2s at retreat, causing 100% kill at step 100.
E2E_NO_TARGET_PROGRESS_TIMEOUT_S: float = 2.0  # unused while e2e disables this term

_B2_LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


@configclass
class TaskDE2EMargObservationsCfg(ObservationsCfg):
    """MARG proprio + history + privileged critic (no height map / depth / extero)."""

    @configclass
    class MargProprioCfg(ObsGroup):
        proprio = ObsTerm(
            func=task_d_loco_mdp.marg_proprio,
            params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": _B2_LEG_JOINTS},
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
            params={"asset_cfg": SceneEntityCfg("robot"), "joint_names": _B2_LEG_JOINTS},
            clip=(-100.0, 100.0),
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
                "joint_names": _B2_LEG_JOINTS,
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: MargProprioCfg = MargProprioCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    critic_priv: CriticPrivCfg = CriticPrivCfg()
    extero = None
    image = None


def _disable_perception_sensors_and_obs(env_cfg) -> None:
    """Disable lidar/cameras/height scan and unused extero/image obs groups."""
    env_cfg.scene.lidar_sensor = None
    env_cfg.scene.head_camera = None
    env_cfg.scene.ee_camera = None
    env_cfg.scene.ee_dual_camera = None
    env_cfg.scene.height_scanner = None
    env_cfg.scene.height_scanner_base = None

    obs = env_cfg.observations
    obs.extero = None
    obs.image = None


def apply_task_d_b2piper_action_layout(env_cfg) -> None:
    """Task D B2Piper: 12D leg + 8D arm (20D total); arm held at default via zero delta in wrapper."""
    from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

    actions = TaskDActionsCfg()
    actions.joint_pos_leg.joint_names = list(UNITREE_B2_PIPER_CFG.leg_joint_names)
    actions.joint_pos_arm.joint_names = list(UNITREE_B2_PIPER_CFG.arm_joint_names)
    actions.joint_pos_leg.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
    actions.joint_pos_leg.clip = {".*": (-100.0, 100.0)}
    actions.joint_vel_wheel = None
    env_cfg.actions = actions


def apply_b2_piper_flat_rewards_no_tracking(env_cfg) -> None:
    """Full B2Piper rough reward template (correct params) minus velocity tracking."""
    import copy

    from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import RewardsCfg

    base_link = getattr(env_cfg, "base_link_name", "base_link")
    foot = getattr(env_cfg, "foot_link_name", ".*_foot")
    r = copy.deepcopy(RewardsCfg())

    r.is_terminated.weight = 0
    r.lin_vel_z_l2.weight = -2.0
    r.ang_vel_xy_l2.weight = -0.05
    r.flat_orientation_l2.weight = 0
    r.base_height_l2.weight = 0
    r.base_height_l2.params["target_height"] = 0.53
    r.base_height_l2.params["asset_cfg"].body_names = [base_link]
    r.base_height_l2.params["sensor_cfg"] = None
    r.body_lin_acc_l2.weight = 0
    r.body_lin_acc_l2.params["asset_cfg"].body_names = [base_link]
    r.joint_torques_l2.weight = -1e-5
    r.joint_vel_l2.weight = 0
    r.joint_acc_l2.weight = -1e-7
    r.joint_pos_limits.weight = -5.0
    r.joint_vel_limits.weight = 0
    r.joint_power.weight = -1e-5
    # E2E stage policy uses zero velocity commands; this term otherwise strongly biases "do not move".
    r.stand_still.weight = 0.0
    r.joint_pos_penalty.weight = -1.0
    r.joint_mirror.weight = -0.05
    r.joint_mirror.params["mirror_joints"] = [
        ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
    ]
    r.action_rate_l2.weight = -0.01
    r.undesired_contacts.weight = -1.0
    r.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{foot}).*"]
    r.contact_forces.weight = -1.5e-4
    r.contact_forces.params["sensor_cfg"].body_names = [foot]
    r.track_lin_vel_xy_exp = None
    r.track_ang_vel_z_exp = None
    r.feet_air_time.weight = 0
    r.feet_air_time.params["threshold"] = 0.5
    r.feet_air_time.params["sensor_cfg"].body_names = [foot]
    r.feet_contact.weight = 0
    r.feet_contact.params["sensor_cfg"].body_names = [foot]
    r.feet_contact_without_cmd.weight = 0.0
    r.feet_contact_without_cmd.params["sensor_cfg"].body_names = [foot]
    r.feet_stumble.weight = 0
    r.feet_stumble.params["sensor_cfg"].body_names = [foot]
    r.feet_slide.weight = 0
    r.feet_slide.params["sensor_cfg"].body_names = [foot]
    r.feet_slide.params["asset_cfg"].body_names = [foot]
    r.feet_height.weight = 0
    r.feet_height.params["target_height"] = 0.05
    r.feet_height.params["asset_cfg"].body_names = [foot]
    r.feet_height_body.weight = -5.0
    r.feet_height_body.params["target_height"] = -0.4
    r.feet_height_body.params["asset_cfg"].body_names = [foot]
    r.feet_gait.weight = 0
    r.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
    r.upward.weight = 3.0

    env_cfg.rewards = r


class _TaskDE2ETrainingMixin:
    """Shared e2e MDP: zero velocity cmd, B2 rewards (no tracking), stage terminations."""

    base_link_name: str = "base_link"
    foot_link_name: str = ".*_foot"
    joint_names: list[str] = _B2_LEG_JOINTS

    def _zero_velocity_commands(self) -> None:
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)

    def _apply_stage_terminations(self) -> None:
        """E2e: instant corridor deviation + 2s no-target-progress timeout (strict)."""
        stage_terms = TaskDTerminationsCfg()
        self.terminations.stage_target_deviation = stage_terms.stage_target_deviation
        progress_term = stage_terms.no_target_progress_timeout
        progress_term.params["stuck_time_s"] = E2E_NO_TARGET_PROGRESS_TIMEOUT_S
        self.terminations.no_target_progress_timeout = progress_term
        self.terminations.no_motion_timeout = None
        self.terminations.no_push_progress_timeout = None
        if hasattr(self.terminations, "x_reached"):
            self.terminations.x_reached = None

    def _apply_b2_locomotion_rewards_no_tracking(self) -> None:
        """B2Piper rough locomotion rewards minus velocity tracking."""
        apply_b2_piper_flat_rewards_no_tracking(self)


@configclass
class TaskDE2EPlatformEnvCfg(_TaskDE2ETrainingMixin, TaskDEnvCfg):
    """Full Task D platform tile (slow startup at high num_envs)."""

    observations: TaskDE2EMargObservationsCfg = TaskDE2EMargObservationsCfg()
    episode_length_s: float = E2E_EPISODE_LENGTH_S

    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=TASK_D_ROBOT_SPAWN_LOCAL,
            ),
        )
        super().__post_init__()
        self._configure_e2e_platform()

    def _configure_e2e_platform(self) -> None:
        _disable_perception_sensors_and_obs(self)

        self._zero_velocity_commands()
        apply_task_d_b2piper_action_layout(self)

        self._apply_b2_locomotion_rewards_no_tracking()
        self._apply_stage_terminations()
        self.terminations.illegal_contact = None
        self.terminations.fall.params["minimum_height"] = 0.25

        print(
            f"[TaskDE2EPlatform] obs=proprio({MARG_PROPRIO_DIM})+history({MARG_HISTORY_DIM})+"
            f"critic_priv({MARG_CRITIC_PRIV_DIM}), B2 rewards w/o tracking, "
            f"stage_target_deviation + no_target_progress({E2E_NO_TARGET_PROGRESS_TIMEOUT_S:.0f}s), "
            f"episode={self.episode_length_s:.0f}s",
            flush=True,
        )


def _bind_e2e_mixin(env_cfg) -> _TaskDE2ETrainingMixin:
    """Bind env_cfg fields onto mixin helpers (rewards/terminations mutate in place)."""
    helper = _TaskDE2ETrainingMixin()
    helper.commands = env_cfg.commands
    helper.rewards = env_cfg.rewards
    helper.terminations = env_cfg.terminations
    helper.curriculum = env_cfg.curriculum
    helper.joint_names = env_cfg.joint_names
    helper.base_link_name = env_cfg.base_link_name
    helper.foot_link_name = env_cfg.foot_link_name
    return helper


def apply_e2e_pit_overrides(env_cfg) -> None:
    """Patch ``build_train_env_cfg`` output for e2e stage training (proprio-only + stage MDP)."""
    env_cfg.observations = TaskDE2EMargObservationsCfg()
    _disable_perception_sensors_and_obs(env_cfg)

    helper = _bind_e2e_mixin(env_cfg)
    helper._zero_velocity_commands()
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.apply_command_config()
    apply_task_d_b2piper_action_layout(env_cfg)
    # E2e base: B2Piper flat locomotion (no velocity tracking); stage rewards from wrapper.
    apply_b2_piper_flat_rewards_no_tracking(env_cfg)
    if hasattr(env_cfg, "disable_zero_weight_rewards"):
        env_cfg.disable_zero_weight_rewards()
    helper._apply_stage_terminations()
    env_cfg.terminations.pit_cross_success = None
    env_cfg.terminations.stuck_no_progress = None
    env_cfg.episode_length_s = E2E_EPISODE_LENGTH_S
    env_cfg.curriculum.pit_width_levels = None

    print(
        f"[TaskDE2EPit] e2e overrides on pit MARG base: obs=proprio({MARG_PROPRIO_DIM})+"
        f"history({MARG_HISTORY_DIM})+critic_priv({MARG_CRITIC_PRIV_DIM}), "
        f"pit_width={env_cfg.pit_width_range}, levels={env_cfg.pit_curriculum_levels}, "
        f"episode={env_cfg.episode_length_s:.0f}s, zero cmd, actions=20D(leg+arm), B2 flat + stage, "
        f"term=stage_target_deviation+no_target_progress({E2E_NO_TARGET_PROGRESS_TIMEOUT_S:.0f}s)",
        flush=True,
    )


def apply_e2e_single_terrain_overrides(env_cfg) -> None:
    """One fixed pit width; isolated tile per env (no width curriculum rows)."""
    import copy

    from atec_rl_lab.tasks.task_d.locomotion.terrain_curriculum import TaskDPitTerrainGenerator
    from atec_rl_lab.tasks.task_d.terrain import (
        PitAndPlatformTerrainCfg,
        TASK_D_TERRAIN_CFG,
        TaskDTerrainImporter,
        configure_task_d_terrain_for_num_envs,
    )

    num_envs = int(env_cfg.scene.num_envs)
    w0, w1 = float(env_cfg.pit_width_range[0]), float(env_cfg.pit_width_range[1])
    fixed_w = w0 if abs(w0 - w1) < 1e-6 else w1

    terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
    terrain_cfg.class_type = TaskDTerrainImporter
    terrain_cfg.terrain_generator.class_type = TaskDPitTerrainGenerator
    configure_task_d_terrain_for_num_envs(terrain_cfg, num_envs, curriculum_levels=None)
    terrain_cfg.terrain_generator.curriculum = False
    pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
    if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
        pit_cfg.pit_width_range = (fixed_w, fixed_w)
        pit_cfg.platform_height_range = env_cfg.platform_height_range
    env_cfg.scene.terrain = terrain_cfg
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.pit_width_range = (fixed_w, fixed_w)
    env_cfg.pit_curriculum_levels = 1
    env_cfg.curriculum.pit_width_levels = None

    nrow = terrain_cfg.terrain_generator.num_rows
    ncol = terrain_cfg.terrain_generator.num_cols
    print(
        f"[TaskDE2E] single terrain {nrow}x{ncol} ({nrow * ncol} tiles), "
        f"fixed pit_width={fixed_w:.3f}m; each env has own robot+box at Task D spawn",
        flush=True,
    )


def build_task_d_e2e_pit_env_cfg(args) -> object:
    """Same env path as ``train_taskd_pit_locomotion.py --marg`` + e2e stage overrides."""
    from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import apply_e2e_reset_overrides
    from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import build_train_env_cfg

    env_cfg = build_train_env_cfg(args)
    apply_e2e_pit_overrides(env_cfg)
    apply_e2e_single_terrain_overrides(env_cfg)
    apply_e2e_reset_overrides(env_cfg, args)
    return env_cfg


def refresh_task_d_e2e_terrain_cfg(env_cfg: TaskDE2EPlatformEnvCfg) -> None:
    refresh_task_d_terrain_cfg(env_cfg)
