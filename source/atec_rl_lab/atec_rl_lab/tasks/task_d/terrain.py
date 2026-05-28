from __future__ import annotations

import math
import random

import trimesh
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.terrains import (
    SubTerrainBaseCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
    MeshPlaneTerrainCfg,
)
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass

from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from atec_rl_lab.tasks.task_base import BetterTerrainGenerator, BetterTerrainImporter

# Playable pit footprint (same as legacy 12x8 tile). Cell size adds a dead gap between tiles.
TASK_D_PLAYABLE_SIZE = (12.0, 8.0)
TASK_D_TILE_GAP = (4.0, 4.0)
TASK_D_CELL_SIZE = (
    TASK_D_PLAYABLE_SIZE[0] + TASK_D_TILE_GAP[0],
    TASK_D_PLAYABLE_SIZE[1] + TASK_D_TILE_GAP[1],
)


def task_d_terrain_grid_shape(num_envs: int) -> tuple[int, int]:
    """Return (num_rows, num_cols) with num_rows * num_cols >= num_envs."""
    n = max(1, int(num_envs))
    num_cols = int(math.ceil(math.sqrt(n)))
    num_rows = int(math.ceil(n / num_cols))
    return num_rows, num_cols


def _pit_playable_size(cfg: PitAndPlatformTerrainCfg) -> tuple[float, float]:
    ps = getattr(cfg, "playable_size", None)
    if ps is None:
        return float(cfg.size[0]), float(cfg.size[1])
    return float(ps[0]), float(ps[1])


def _pit_cell_margin(cfg: PitAndPlatformTerrainCfg) -> tuple[float, float]:
    px, py = _pit_playable_size(cfg)
    return (float(cfg.size[0]) - px) * 0.5, (float(cfg.size[1]) - py) * 0.5


def pit_and_platform_terrain(
    difficulty: float, cfg: PitAndPlatformTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Pit + platform centered in the cell; margins stay empty so adjacent tiles do not connect."""
    mesh_list = []
    mx, my = _pit_cell_margin(cfg)
    px, py = _pit_playable_size(cfg)
    cx = mx + px * 0.5
    cy = my + py * 0.5
    pit_depth = cfg.pit_depth
    pit_width = cfg.pit_width_range[0] + difficulty * (
        cfg.pit_width_range[1] - cfg.pit_width_range[0]
    )
    platform_width = pit_width
    platform_height = cfg.platform_height_range[0] + difficulty * (
        cfg.platform_height_range[1] - cfg.platform_height_range[0]
    )
    mesh_list.extend(
        make_border(
            size=(px, py - cfg.border_width),
            inner_size=(pit_width, py - cfg.border_width - 0.2),
            height=pit_depth,
            position=(cx, cy, -pit_depth / 2),
        )
    )
    pit_bottom_thickness = 0.2
    pit_bottom = trimesh.creation.box(
        extents=(pit_width, py - cfg.border_width, pit_bottom_thickness),
        transform=trimesh.transformations.translation_matrix(
            (cx, cy, -pit_depth - pit_bottom_thickness / 2)
        ),
    )
    left_or_right = 0.75
    platform = trimesh.creation.box(
        extents=(platform_width, py / 2 - cfg.border_width, platform_height),
        transform=trimesh.transformations.translation_matrix(
            (cx, my + py * left_or_right, platform_height / 2)
        ),
    )
    mesh_list.append(pit_bottom)
    mesh_list.append(platform)
    origin = np.array([mx + px * 0.15, my + py / 2, 0.0])
    return mesh_list, origin


@configclass
class PitAndPlatformTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a pit and an adjacent platform."""

    function = pit_and_platform_terrain
    border_width: float = 1.0
    pit_depth: float = 1.0
    pit_width_range: tuple[float, float] = (1.6, 1.7)
    platform_height_range: tuple[float, float] = (1.4, 1.5)
    playable_size: tuple[float, float] = TASK_D_PLAYABLE_SIZE


def platform_terrain(difficulty: float, cfg: PlatformTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    platform_width = cfg.platform_width_range[0] + random.random() * (
        cfg.platform_width_range[1] - cfg.platform_width_range[0]
    )
    platform_height = cfg.platform_height_range[0] + difficulty * (
        cfg.platform_height_range[1] - cfg.platform_height_range[0]
    )
    mesh_list = []

    ground = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], 0.1),
        transform=trimesh.transformations.translation_matrix((cfg.size[0] / 2, cfg.size[1] / 2, -0.05)),
    )
    mesh_list.append(ground)

    platform = trimesh.creation.box(
        extents=(platform_width, cfg.size[1], platform_height),
        transform=trimesh.transformations.translation_matrix((cfg.size[0] / 2, cfg.size[1] / 2, platform_height / 2)),
    )

    mesh_list.append(platform)
    origin = np.array([cfg.size[0] * 0.15, cfg.size[1] / 2, platform_height])
    return mesh_list, origin


@configclass
class PlatformTerrainCfg(SubTerrainBaseCfg):
    function = platform_terrain
    platform_width_range: tuple[float, float] = (1.9, 2.0)
    platform_height_range: tuple[float, float] = (0.5, 0.6)


class TaskDTerrainImporter(BetterTerrainImporter):
    """One env per terrain tile (env_id -> row=env_id//cols, col=env_id%cols)."""

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        if origins is None:
            super().configure_env_origins(origins)
            return
        if isinstance(origins, np.ndarray):
            origins = torch.from_numpy(origins)
        self.terrain_origins = origins.to(self.device, dtype=torch.float)
        num_rows, num_cols, _ = self.terrain_origins.shape
        num_envs = int(self.cfg.num_envs)
        capacity = int(num_rows * num_cols)
        if capacity < num_envs:
            raise ValueError(
                f"Task D terrain grid {num_rows}x{num_cols}={capacity} < num_envs={num_envs}. "
                "Increase num_rows/num_cols in terrain generator."
            )
        env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
        row_idx = torch.div(env_ids, num_cols, rounding_mode="floor")
        col_idx = env_ids % num_cols
        self.terrain_levels = row_idx
        self.terrain_types = col_idx
        self.env_origins = self.terrain_origins[row_idx, col_idx]
        print(
            f"[TaskDTerrain] grid={num_rows}x{num_cols} ({capacity} cells), "
            f"cell_size={TASK_D_CELL_SIZE}, playable={TASK_D_PLAYABLE_SIZE}, gap={TASK_D_TILE_GAP}, "
            f"num_envs={num_envs}, one env per tile",
            flush=True,
        )


def configure_task_d_terrain_for_num_envs(terrain_cfg: TerrainImporterCfg, num_envs: int) -> TerrainImporterCfg:
    """Expand pit grid to cover ``num_envs`` with isolated tiles (gap between cells)."""
    num_rows, num_cols = task_d_terrain_grid_shape(num_envs)
    terrain_cfg.num_envs = int(num_envs)
    gen = terrain_cfg.terrain_generator
    gen.num_rows = num_rows
    gen.num_cols = num_cols
    gen.size = TASK_D_CELL_SIZE
    gen.use_cache = True
    gen.curriculum = False
    pit_cfg = gen.sub_terrains.get("pit_and_platform")
    if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
        pit_cfg.playable_size = TASK_D_PLAYABLE_SIZE
    return terrain_cfg


TASK_D_TERRAIN_CFG = TerrainImporterCfg(
    class_type=TaskDTerrainImporter,
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        class_type=BetterTerrainGenerator,
        seed=0,
        size=TASK_D_CELL_SIZE,
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        curriculum=False,
        sub_terrains={
            "pit_and_platform": PitAndPlatformTerrainCfg(proportion=1.0),
        },
    ),
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ATEC_ASSETS_MODEL_DIR}/scene/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)
