"""Termination terms for Task D pit locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import pit_success_local_x

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
    post_cross_distance: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when robot local +x passes the pit far edge (+ optional buffer).

    Uses the same analytic geometry as ``pit_width_levels`` curriculum (``pit_cross_world_x``),
    not a fixed local-x constant. Spawn is at local_x ~ 1.2; a hard threshold of 2.0 only
    requires ~0.8 m forward and is not a real pit crossing.
    """
    threshold = pit_success_local_x(env, post_cross_distance=post_cross_distance)
    return robot_local_x(env, asset_cfg=asset_cfg) > threshold


def pit_cross_local_x_success_done(
    env: ManagerBasedRLEnv,
    post_cross_distance: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on successful pit crossing (local x past pit far edge)."""
    return pit_cross_local_x_success_mask(
        env,
        post_cross_distance=post_cross_distance,
        asset_cfg=asset_cfg,
    )
