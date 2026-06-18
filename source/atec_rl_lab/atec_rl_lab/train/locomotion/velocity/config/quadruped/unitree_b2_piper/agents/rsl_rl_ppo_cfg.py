from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeB2PiperRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_b2_piper_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeB2PiperFlatPPORunnerCfg(UnitreeB2PiperRoughPPORunnerCfg):
    experiment_name = "unitree_b2_piper_flat"


@configclass
class UnitreeB2PiperTaskDPitLocomotionPPORunnerCfg(UnitreeB2PiperFlatPPORunnerCfg):
    max_iterations = 15000
    experiment_name = "taskd_pit_locomotion_b2piper"


@configclass
class MargActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "MargActorCritic"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = False
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "relu"
    init_noise_std: float = 0.6
    max_noise_std: float = 2.0
    estimator_hidden_dims: list = [128]
    elevation_hidden_dims: list = [128, 64]


@configclass
class MargPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
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
class UnitreeB2PiperTaskDPitLocomotionMargPPORunnerCfg(UnitreeB2PiperFlatPPORunnerCfg):
    max_iterations = 15000
    experiment_name = "taskd_pit_locomotion_b2piper_marg"
    obs_groups = {
        "policy": ["proprio", "proprio_history", "height_map"],
        # Asymmetric critic uses proprio + elevation(height_map) + privileged state.
        # Proprio history is only for estimator in actor path.
        "critic": ["proprio", "height_map", "critic_priv"],
    }
    policy = MargActorCriticCfg()
    algorithm = MargPpoAlgorithmCfg()
