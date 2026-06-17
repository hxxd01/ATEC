"""Termination terms for Task D pit locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_local_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot root +x relative to each env's ``env_origin``."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]


def pit_cross_local_x_success_mask(
    env: ManagerBasedRLEnv,
    local_x_threshold: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when robot local +x (env_origin offset) exceeds ``local_x_threshold``."""
    return robot_local_x(env, asset_cfg=asset_cfg) > float(local_x_threshold)


def pit_cross_local_x_success_done(
    env: ManagerBasedRLEnv,
    local_x_threshold: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on successful pit crossing (local x past threshold)."""
    return pit_cross_local_x_success_mask(
        env,
        local_x_threshold=local_x_threshold,
        asset_cfg=asset_cfg,
    )
