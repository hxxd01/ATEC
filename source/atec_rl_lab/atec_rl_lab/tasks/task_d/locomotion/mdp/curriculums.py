"""Pit-width curriculum for Task D locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from atec_rl_lab.tasks.task_d.locomotion.mdp.terminations import pit_cross_local_x_success_mask
from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import pit_width_from_level, _read_max_level, _read_width_range

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_d_pit_width_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    post_cross_distance: float = 1.5,
    success_term_name: str = "pit_cross_success",
) -> torch.Tensor:
    """Promote only reset envs after true pit-cross success.

    This keeps pit-width curriculum aligned with episode success criteria and avoids
    global promotion from partial/temporary crossings in other environments.
    """
    terrain = env.scene.terrain
    if not hasattr(terrain, "terrain_levels") or not hasattr(terrain, "update_env_origins_from_levels"):
        return torch.tensor(0.0, device=env.device)

    env_ids_t = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    if env_ids_t.numel() == 0:
        return terrain.terrain_levels.float().mean()

    max_level = _read_max_level(env)
    width_range = _read_width_range(env)

    promote = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if success_term_name in env.termination_manager.active_terms:
        success_done = env.termination_manager.get_term(success_term_name)
        promote[env_ids_t] = success_done[env_ids_t]
    else:
        success = pit_cross_local_x_success_mask(env, post_cross_distance=post_cross_distance)
        promote[env_ids_t] = success[env_ids_t]

    promoted_ids = promote.nonzero(as_tuple=True)[0]
    if promoted_ids.numel() > 0:
        if hasattr(terrain, "promote_terrain_levels"):
            terrain.promote_terrain_levels(promoted_ids, max_level=max_level)
        else:
            new_levels = terrain.terrain_levels.clone()
            new_levels[promoted_ids] = torch.clamp(new_levels[promoted_ids] + 1, max=max_level)
            terrain.update_env_origins_from_levels(new_levels, env_ids=promoted_ids)

    mean_width = pit_width_from_level(terrain.terrain_levels, width_range, max_level).mean()
    return mean_width
