from __future__ import annotations

import math

from isaaclab.terrains import (
    SubTerrainBaseCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
import trimesh
import numpy as np
import torch
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

from atec_rl_lab.tasks.task_base import BetterTerrainGenerator, BetterTerrainImporter

# Playable Task-B footprint (20x20 m). Cell size adds dead gap between tiles (same idea as Task D).
TASK_B_PLAYABLE_SIZE = (20.0, 20.0)
TASK_B_TILE_GAP = (50.0, 50.0)
TASK_B_CELL_SIZE = (
    TASK_B_PLAYABLE_SIZE[0] + TASK_B_TILE_GAP[0],
    TASK_B_PLAYABLE_SIZE[1] + TASK_B_TILE_GAP[1],
)


def task_b_terrain_grid_shape(num_envs: int) -> tuple[int, int]:
    """Return (num_rows, num_cols) with num_rows * num_cols >= num_envs."""
    n = max(1, int(num_envs))
    num_cols = int(math.ceil(math.sqrt(n)))
    num_rows = int(math.ceil(n / num_cols))
    return num_rows, num_cols


def _playable_size(cfg: FlatTerrainWithTrashBinCfg) -> tuple[float, float]:
    ps = getattr(cfg, "playable_size", None)
    if ps is None:
        return float(cfg.size[0]), float(cfg.size[1])
    return float(ps[0]), float(ps[1])


def _cell_margin(cfg: FlatTerrainWithTrashBinCfg) -> tuple[float, float]:
    px, py = _playable_size(cfg)
    return (float(cfg.size[0]) - px) * 0.5, (float(cfg.size[1]) - py) * 0.5


def flat_terrain_with_trash_bin(
    difficulty: float, cfg: FlatTerrainWithTrashBinCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    del difficulty
    mesh_list = []
    mx, my = _cell_margin(cfg)
    px, py = _playable_size(cfg)
    cx = mx + px * 0.5
    cy = my + py * 0.5

    ground = trimesh.creation.box(
        extents=(px, py, 0.10),
        transform=trimesh.transformations.translation_matrix((cx, cy, -0.005)),
    )
    mesh_list.append(ground)

    bin_x = cx + float(cfg.trash_bin_x)
    bin_y = cy + float(getattr(cfg, "trash_bin_y", 0.0))

    bin_diameter = 2.0
    bin_radius = 0.5 * bin_diameter
    bin_height = 0.5
    wall_thickness = 0.02
    bottom_thickness = 0.05
    bin_color = np.asarray([255, 128, 0, 255], dtype=np.uint8)

    bottom = trimesh.creation.cylinder(
        radius=bin_radius,
        height=bottom_thickness,
        transform=trimesh.transformations.translation_matrix((bin_x, bin_y, bottom_thickness / 2)),
    )
    bottom.visual.vertex_colors = np.tile(bin_color, (bottom.vertices.shape[0], 1))
    mesh_list.append(bottom)

    wall = trimesh.creation.annulus(
        r_min=max(0.0, bin_radius - wall_thickness),
        r_max=bin_radius,
        height=bin_height,
        transform=trimesh.transformations.translation_matrix(
            (bin_x, bin_y, bottom_thickness + bin_height / 2)
        ),
    )
    wall.visual.vertex_colors = np.tile(bin_color, (wall.vertices.shape[0], 1))
    mesh_list.append(wall)

    origin = np.array([cx, cy, 0.0])
    return mesh_list, origin


@configclass
class FlatTerrainWithTrashBinCfg(SubTerrainBaseCfg):
    """Flat terrain with trash bin; playable area centered in the cell with outer gap."""

    function = flat_terrain_with_trash_bin
    trash_bin_x: float = 7.0
    trash_bin_y: float = 0.0
    playable_size: tuple[float, float] = TASK_B_PLAYABLE_SIZE


class TaskBTerrainImporter(BetterTerrainImporter):
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
                f"Task B terrain grid {num_rows}x{num_cols}={capacity} < num_envs={num_envs}. "
                "Increase num_rows/num_cols in terrain generator."
            )
        env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
        row_idx = torch.div(env_ids, num_cols, rounding_mode="floor")
        col_idx = env_ids % num_cols
        self.terrain_levels = row_idx
        self.terrain_types = col_idx
        self.env_origins = self.terrain_origins[row_idx, col_idx]
        print(
            f"[TaskBTerrain] grid={num_rows}x{num_cols} ({capacity} cells), "
            f"cell_size={TASK_B_CELL_SIZE}, playable={TASK_B_PLAYABLE_SIZE}, gap={TASK_B_TILE_GAP}, "
            f"num_envs={num_envs}, one env per tile",
            flush=True,
        )


def configure_task_b_terrain_for_num_envs(
    terrain_cfg: TerrainImporterCfg,
    num_envs: int,
) -> TerrainImporterCfg:
    """Expand flat grid to cover ``num_envs`` with isolated tiles (gap between cells)."""
    terrain_cfg.num_envs = int(num_envs)
    gen = terrain_cfg.terrain_generator
    gen.size = TASK_B_CELL_SIZE
    gen.use_cache = True
    flat_cfg = gen.sub_terrains.get("flat_with_bin")
    if isinstance(flat_cfg, FlatTerrainWithTrashBinCfg):
        flat_cfg.playable_size = TASK_B_PLAYABLE_SIZE
    num_rows, num_cols = task_b_terrain_grid_shape(num_envs)
    gen.num_rows = num_rows
    gen.num_cols = num_cols
    gen.curriculum = False
    return terrain_cfg


TASK_B_TERRAIN_CFG = TerrainImporterCfg(
    class_type=TaskBTerrainImporter,
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        class_type=BetterTerrainGenerator,
        seed=0,
        size=TASK_B_CELL_SIZE,
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        curriculum=False,
        sub_terrains={
            "flat_with_bin": FlatTerrainWithTrashBinCfg(proportion=1.0),
        },
    ),
    max_init_terrain_level=0,
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
