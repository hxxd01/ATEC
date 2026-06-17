"""Analytic pit geometry for Task D pit locomotion (matches ``pit_and_platform_terrain``)."""

from __future__ import annotations

import torch

from atec_rl_lab.tasks.task_d.terrain import (
    TASK_D_CELL_SIZE,
    TASK_D_PLAYABLE_SIZE,
)

PIT_BORDER_WIDTH = 1.0
PIT_DEPTH = 1.0
PIT_CROSS_EDGE_BUFFER = 0.25


def _cell_margins() -> tuple[float, float]:
    px, py = TASK_D_PLAYABLE_SIZE
    return (TASK_D_CELL_SIZE[0] - px) * 0.5, (TASK_D_CELL_SIZE[1] - py) * 0.5


def pit_center_local_offset() -> tuple[float, float, float]:
    """Pit gap center relative to each tile's ``env_origin``."""
    mx, my = _cell_margins()
    px, py = TASK_D_PLAYABLE_SIZE
    cx = mx + px * 0.5
    cy = my + py * 0.5
    origin_x = mx + px * 0.15
    origin_y = my + py / 2
    return (cx - origin_x, cy - origin_y, -PIT_DEPTH * 0.5)


def pit_length_y() -> float:
    px, py = TASK_D_PLAYABLE_SIZE
    del px
    return float(py - PIT_BORDER_WIDTH - 0.2)


def pit_width_from_level(level: torch.Tensor, width_range: tuple[float, float], max_level: int) -> torch.Tensor:
    """Map discrete curriculum level to pit opening width (meters)."""
    if max_level <= 0:
        difficulty = torch.zeros_like(level, dtype=torch.float32)
    else:
        difficulty = level.to(dtype=torch.float32) / float(max_level)
    w0, w1 = width_range
    return w0 + difficulty * (w1 - w0)


def _read_width_range(env) -> tuple[float, float]:
    cfg = getattr(env, "cfg", None)
    if cfg is not None and hasattr(cfg, "pit_width_range"):
        return tuple(cfg.pit_width_range)
    return (0.4, 1.4)


def _read_max_level(env) -> int:
    cfg = getattr(env, "cfg", None)
    if cfg is not None and hasattr(cfg, "pit_curriculum_levels"):
        return max(1, int(cfg.pit_curriculum_levels) - 1)
    terrain = env.scene.terrain
    if hasattr(terrain, "terrain_origins"):
        return max(1, int(terrain.terrain_origins.shape[0]) - 1)
    return 10


def _terrain_levels(env) -> torch.Tensor:
    terrain = env.scene.terrain
    levels = getattr(terrain, "terrain_levels", None)
    if levels is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    return levels


def pit_geometry_for_envs(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (center_world [N,3], size_lwh [N,3])."""
    device = env.device
    num_envs = env.num_envs
    ox, oy, oz = pit_center_local_offset()
    center_local = torch.tensor([ox, oy, oz], device=device, dtype=torch.float32).view(1, 3)
    origins = env.scene.env_origins[:, :3]
    center_w = origins + center_local

    width_range = _read_width_range(env)
    max_level = _read_max_level(env)
    levels = _terrain_levels(env)
    width = pit_width_from_level(levels, width_range, max_level)
    length = torch.full((num_envs,), pit_length_y(), device=device, dtype=torch.float32)
    depth = torch.full((num_envs,), PIT_DEPTH, device=device, dtype=torch.float32)
    size_lwh = torch.stack([length, width, depth], dim=-1)
    return center_w, size_lwh


def pit_cross_local_x(env) -> torch.Tensor:
    """Local +x from ``env_origin`` to the pit far edge plus a small buffer."""
    _, size_lwh = pit_geometry_for_envs(env)
    ox, _, _ = pit_center_local_offset()
    half_width = 0.5 * size_lwh[:, 1]
    return torch.full_like(half_width, float(ox)) + half_width + float(PIT_CROSS_EDGE_BUFFER)


def pit_cross_world_x(env) -> torch.Tensor:
    """World-frame x threshold: robot base x past pit far edge counts as crossed."""
    return env.scene.env_origins[:, 0] + pit_cross_local_x(env)


def pit_success_local_x(env, post_cross_distance: float = 0.5) -> torch.Tensor:
    """Local +x from ``env_origin`` required for a successful pit crossing."""
    return pit_cross_local_x(env) + float(post_cross_distance)
