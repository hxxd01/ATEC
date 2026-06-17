"""Privileged observations for Task D pit locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import (
    pit_cross_world_x,
    pit_geometry_for_envs,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_world_position(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot root position in world frame, shape (num_envs, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def pit_geometry(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Pit center (world xyz) + pit size (length_y, width_x, depth_z), shape (num_envs, 6)."""
    center_w, size_lwh = pit_geometry_for_envs(env)
    return torch.cat([center_w, size_lwh], dim=-1)
