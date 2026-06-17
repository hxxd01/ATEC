"""Rewards for Task D pit locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_world_x_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for forward (+world x) velocity toward crossing the pit."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0)
