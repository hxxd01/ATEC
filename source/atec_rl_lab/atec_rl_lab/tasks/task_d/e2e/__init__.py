"""Task D e2e gym registration (pit default, platform optional)."""

import gymnasium as gym

from .env_cfg import TaskDE2EPlatformEnvCfg

gym.register(
    id="ATEC-TaskD-E2E-Pit-B2Piper-v0",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "atec_rl_lab.tasks.task_d.locomotion.env_cfg:UnitreeB2PiperTaskDPitLocomotionMargEnvCfg"
        ),
    },
)

gym.register(
    id="ATEC-TaskD-E2E-B2Piper-v0",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:TaskDE2EPlatformEnvCfg"},
)

__all__ = ["TaskDE2EPlatformEnvCfg"]
