"""Task D pit terrain with per-env width curriculum via terrain row levels."""

from __future__ import annotations

import torch

from atec_rl_lab.tasks.task_base.terrain_base import BetterTerrainGenerator
from atec_rl_lab.tasks.task_d.terrain import TaskDTerrainImporter


class TaskDPitTerrainGenerator(BetterTerrainGenerator):
    """Row = pit-width curriculum level; avoids Isaac Lab curriculum bug for large ``num_cols``."""

    def _generate_curriculum_terrains(self):
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        if not sub_terrains_cfgs:
            raise ValueError("TaskDPitTerrainGenerator requires at least one sub-terrain.")
        sub_cfg = sub_terrains_cfgs[0]
        n_rows = int(self.cfg.num_rows)
        n_cols = int(self.cfg.num_cols)
        for sub_row in range(n_rows):
            difficulty = 0.0 if n_rows <= 1 else sub_row / (n_rows - 1)
            for sub_col in range(n_cols):
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)


class TaskDPitCurriculumTerrainImporter(TaskDTerrainImporter):
    """One env per tile; all envs start at the narrowest pit row and can move to wider rows."""

    def configure_env_origins(self, origins):
        super().configure_env_origins(origins)
        num_envs = int(self.cfg.num_envs)
        num_rows, num_cols, _ = self.terrain_origins.shape
        env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
        if num_cols < num_envs:
            raise ValueError(
                f"Task D pit curriculum needs num_cols >= num_envs ({num_cols} < {num_envs}). "
                "Reduce --num_envs or increase terrain columns."
            )
        # Each env owns one column on row 0; promotion moves it to wider pit rows.
        self.terrain_types = env_ids
        self.terrain_levels = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._apply_env_origins()

    def _apply_env_origins(self):
        row_idx = self.terrain_levels
        col_idx = self.terrain_types
        self.env_origins = self.terrain_origins[row_idx, col_idx]

    def update_env_origins_from_levels(self, new_levels: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Update terrain row (pit width level) and refresh ``env_origins``."""
        del env_ids
        self.terrain_levels = new_levels.to(device=self.device, dtype=torch.long)
        row_idx = self.terrain_levels
        col_idx = self.terrain_types
        self.env_origins = self.terrain_origins[row_idx, col_idx]
