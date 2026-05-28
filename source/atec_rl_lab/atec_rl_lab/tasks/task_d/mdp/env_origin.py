"""Per-env terrain origin offsets for Task D world-frame thresholds and waypoints."""

from __future__ import annotations

import torch

if False:  # TYPE_CHECKING
    from isaaclab.envs import ManagerBasedRLEnv

# Reference pit tile origin (debug_taskd_env_origins.py); spawn local = nominal_world - this.
TASK_D_PIT_TERRAIN_ORIGIN_XY = (-4.2, 0.0)


def task_d_env_origin_xy(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env terrain origin (x, y) in world frame, shape (num_envs,)."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    origins = base.scene.env_origins
    return origins[:, 0], origins[:, 1]


def task_d_nominal_to_world(
    nominal_x: torch.Tensor,
    nominal_y: torch.Tensor,
    env_origin_x: torch.Tensor,
    env_origin_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map nominal world coords on the reference pit tile to another tile's world frame.

    Same convention as ``reset_root_state_at_env_origin``: ``world = env_origin + (nominal - pit_ref)``.
    """
    ox_ref, oy_ref = TASK_D_PIT_TERRAIN_ORIGIN_XY
    return (
        nominal_x + env_origin_x - ox_ref,
        nominal_y + env_origin_y - oy_ref,
    )


def task_d_nominal_x_to_world(nominal_x: torch.Tensor, env_origin_x: torch.Tensor) -> torch.Tensor:
    """Scalar / broadcast x only."""
    return nominal_x + env_origin_x - TASK_D_PIT_TERRAIN_ORIGIN_XY[0]


def task_d_nominal_y_to_world(nominal_y: torch.Tensor, env_origin_y: torch.Tensor) -> torch.Tensor:
    return nominal_y + env_origin_y - TASK_D_PIT_TERRAIN_ORIGIN_XY[1]


# Backward-compatible aliases (deprecated: use task_d_nominal_*_to_world).
def task_d_offset_nominal_x(nominal_x: torch.Tensor, env_origin_x: torch.Tensor) -> torch.Tensor:
    return task_d_nominal_x_to_world(nominal_x, env_origin_x)


def task_d_offset_nominal_y(nominal_y: torch.Tensor, env_origin_y: torch.Tensor) -> torch.Tensor:
    return task_d_nominal_y_to_world(nominal_y, env_origin_y)


def task_d_offset_nominal_xy(
    nominal_x: torch.Tensor,
    nominal_y: torch.Tensor,
    env_origin_x: torch.Tensor,
    env_origin_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return task_d_nominal_to_world(nominal_x, nominal_y, env_origin_x, env_origin_y)
