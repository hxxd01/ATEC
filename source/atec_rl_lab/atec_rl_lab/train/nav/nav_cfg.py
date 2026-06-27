"""rsl_rl runner / PPO config for the hierarchical nav policy."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# ── Shared PPO algorithm config ───────────────────────────────────────────────
_PPO_ALG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)

# ── Standard (no camera) config ───────────────────────────────────────────────

@configclass
class NavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env  = 24
    max_iterations     = 5000
    save_interval      = 100
    experiment_name    = "nav_hierarchical_b2piper"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _PPO_ALG


# ── Camera (CNN) config ───────────────────────────────────────────────────────
# CameraActorCriticCfg extends RslRlPpoActorCriticCfg with CNN-specific fields.
# The extra fields are passed as **kwargs to ActorCriticWithCNN.__init__.

@configclass
class CameraActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticWithCNN"
    # Image dimensions (must match what HierarchicalNavEnv sends)
    img_flat_dim: int   = 3 * 64 * 64   # 12288
    img_hw: int         = 64
    img_channels: int   = 3
    cnn_feature_dim: int = 256


@configclass
class NavCameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env  = 16            # fewer steps per update (larger obs)
    max_iterations     = 5000
    save_interval      = 100
    experiment_name    = "nav_camera_b2piper"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = CameraActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,  # CNN has its own normalisation inside
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _PPO_ALG


@configclass
class TaskDTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for Task D teacher training (asymmetric actor/critic obs)."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "taskd_teacher_b2piper"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = _PPO_ALG


@configclass
class TaskDStudentActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "TaskDStudentActorCritic"
    img_h: int = 24
    img_w: int = 32
    img_channels: int = 4  # 4 = rgb+depth per camera, 1 = depth only
    proprio_dim: int = 9
    enc_dim: int = 128
    fuse_dim: int = 256
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1

    leg_action_dim: int = 12
    nav_action_dim: int = 3


@configclass
class TaskBStudentActorCriticCfg(TaskDStudentActorCriticCfg):
    """Task B nav student: inherits Task D student RNN actor + asymmetric RNN critic."""

    class_name: str = "TaskBStudentActorCritic"
    proprio_dim: int = 15  # base(9) + last_vel(3) + touch_t(1) + scored(1) + elapsed(1)
    nav_action_dim: int = 3
    leg_action_dim: int = 12


@configclass
class TaskDStudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for Task D student fine-tuning (no privileged critic)."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "taskd_student_b2piper"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = TaskDStudentActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = _PPO_ALG


@configclass
class TaskBStudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for Task B student nav: Task-D-style asymmetric RNN critic."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "taskb_student_b2piper"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = TaskBStudentActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )
    algorithm = _PPO_ALG


@configclass
class TaskDStudentPitE2EPPORunnerCfg(TaskDStudentPPORunnerCfg):
    """Legacy student GRU pit head (deprecated; use TaskDMargDepthPitE2EPPORunnerCfg)."""

    num_steps_per_env = 24
    max_iterations = 15000
    experiment_name = "taskd_student_pit_e2e_b2piper"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = TaskDStudentActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256, 128],
        activation="elu",
        leg_action_dim=12,
        nav_action_dim=3,
    )


@configclass
class MargDepthPitActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "MargDepthPitActorCritic"
    img_h: int = 24
    img_w: int = 32
    depth_channels: int = 1
    head_depth_only: bool = True
    enc_dim: int = 128
    elevation_out_dim: int = 16
    estimator_hidden_dims: list = [128]
    depth_hidden_dims: list = [128, 64]
    init_noise_std: float = 0.6
    max_noise_std: float = 2.0


@configclass
class MargDepthPitPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "MargPPO"
    reg_loss_coef: float = 5.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@configclass
class MargPitDaggerPpoAlgorithmCfg(MargDepthPitPpoAlgorithmCfg):
    class_name: str = "MargPitDaggerPPO"
    dagger_coef: float = 1.0
    dagger_beta: float = 1.0
    dagger_beta_end: float = 0.0
    dagger_beta_decay_iters: int = 4000


@configclass
class TaskDMargDepthPitE2EPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """MARG layout + depth CNN (replaces height map); pit env rewards/terminations."""

    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 100
    experiment_name = "taskd_marg_depth_pit_e2e_b2piper"
    obs_groups = {
        "policy": ["proprio", "proprio_history", "depth"],
        "critic": ["proprio", "depth", "critic_priv"],
    }

    policy = MargDepthPitActorCriticCfg(
        actor_obs_normalization=True,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="relu",
    )
    algorithm = MargDepthPitPpoAlgorithmCfg()


@configclass
class TaskDMargDepthPitDaggerPPORunnerCfg(TaskDMargDepthPitE2EPPORunnerCfg):
    """Depth student + frozen MARG height-map teacher (DAgger)."""

    experiment_name = "taskd_marg_depth_pit_dagger_b2piper"
    algorithm = MargPitDaggerPpoAlgorithmCfg(
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        entropy_coef=0.001,
    )


@configclass
class MargProprioActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "MargProprioActorCritic"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = False
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "relu"
    init_noise_std: float = 0.6
    max_noise_std: float = 2.0
    estimator_hidden_dims: list = [128]
    critic_task_dim: int = 21  # 17 + 4 stages (TaskDStudent-style priv)


@configclass
class TaskDE2EPlatformMargPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Task D platform e2e: MARG proprio (no elevation) + 4-stage wrapper rewards."""

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "taskd_e2e_platform_b2piper"
    obs_groups = {
        "policy": ["proprio", "proprio_history"],
        "critic": ["proprio", "critic_priv", "critic_task"],
    }
    policy = MargProprioActorCriticCfg()
    algorithm = MargDepthPitPpoAlgorithmCfg()
