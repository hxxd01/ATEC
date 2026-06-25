import gymnasium as gym

from . import agents

gym.register(
    id="ATEC-Isaac-Velocity-Rough-Unitree-B2Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2PiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperRoughPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Unitree-B2Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Squat-Flat-Unitree-B2Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_flat_env_cfg:UnitreeB2PiperSquatFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperSquatFlatPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Stand-Flat-Unitree-B2Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_flat_env_cfg:UnitreeB2PiperStandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperStandFlatPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Unitree-B2Piper-Squat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_flat_env_cfg:UnitreeB2PiperSquatFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperSquatFlatPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Unitree-B2Piper-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_flat_env_cfg:UnitreeB2PiperStandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperStandFlatPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-TaskD-PitLocomotion-B2Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "atec_rl_lab.tasks.task_d.locomotion.env_cfg:UnitreeB2PiperTaskDPitLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2PiperTaskDPitLocomotionPPORunnerCfg",
    },
)
