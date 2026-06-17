"""Rewards for Task D pit locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

from atec_rl_lab.tasks.task_d.locomotion.mdp.terminations import pit_cross_local_x_success_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_world_x_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for forward (+world x) velocity toward crossing the pit."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0)


class PitCrossSuccessBonus(ManagerTermBase):
    """One-time sparse bonus when local +x crosses the pit success threshold."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._reward_given.zero_()
        else:
            self._reward_given[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reward_value: float = 30.0,
        post_cross_distance: float = 0.5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        success = pit_cross_local_x_success_mask(
            env,
            post_cross_distance=post_cross_distance,
            asset_cfg=asset_cfg,
        )
        trigger = success & (~self._reward_given)
        self._reward_given |= success
        return trigger.float() * float(reward_value)
