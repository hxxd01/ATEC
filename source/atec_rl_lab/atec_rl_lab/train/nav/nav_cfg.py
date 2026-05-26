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
    img_hw: int = 64
    img_channels: int = 4  # 4 = rgb+depth per camera, 1 = depth only
    proprio_dim: int = 12  # lin/ang/gravity(9) + last_nav_cmd(3)
    lidar_bins: int = 0
    enc_dim: int = 128
    fuse_dim: int = 256
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


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
