"""Per-env terrain origin offsets for Task D world-frame thresholds and waypoints."""

from __future__ import annotations

import torch

if False:  # TYPE_CHECKING
    from isaaclab.envs import ManagerBasedRLEnv


def task_d_env_origin_xy(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env terrain origin (x, y) in world frame, shape (num_envs,)."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    origins = base.scene.env_origins
    return origins[:, 0], origins[:, 1]


def task_d_offset_nominal_x(nominal_x: torch.Tensor, env_origin_x: torch.Tensor) -> torch.Tensor:
    """Add per-env origin to nominal world x (broadcast)."""
    return nominal_x + env_origin_x


def task_d_offset_nominal_y(nominal_y: torch.Tensor, env_origin_y: torch.Tensor) -> torch.Tensor:
    return nominal_y + env_origin_y


def task_d_offset_nominal_xy(
    nominal_x: torch.Tensor,
    nominal_y: torch.Tensor,
    env_origin_x: torch.Tensor,
    env_origin_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return nominal_x + env_origin_x, nominal_y + env_origin_y
