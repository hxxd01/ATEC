"""Pit-width curriculum for Task D locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import (
    pit_cross_world_x,
    pit_width_from_level,
    _read_max_level,
    _read_width_range,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_d_pit_width_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    success_fraction: float = 0.75,
) -> torch.Tensor:
    """Promote resetting envs to wider pits when tracking reward is strong or the pit was crossed."""
    terrain = env.scene.terrain
    if not hasattr(terrain, "terrain_levels") or not hasattr(terrain, "promote_terrain_levels"):
        return torch.tensor(0.0, device=env.device)

    env_ids_t = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    if env_ids_t.numel() == 0:
        return terrain.terrain_levels.float().mean()

    max_level = _read_max_level(env)
    width_range = _read_width_range(env)

    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    mean_rew = episode_sums[env_ids_t] / env.max_episode_length_s
    promote = mean_rew > success_fraction * reward_term_cfg.weight

    robot = env.scene["robot"]
    cross_x = pit_cross_world_x(env)
    promote |= robot.data.root_pos_w[env_ids_t, 0] >= cross_x[env_ids_t]

    promote_ids = env_ids_t[promote]
    if promote_ids.numel() > 0:
        terrain.promote_terrain_levels(promote_ids, max_level=max_level)

    mean_level = terrain.terrain_levels.float().mean()
    mean_width = pit_width_from_level(terrain.terrain_levels, width_range, max_level).mean()
    if promote_ids.numel() > 0 and env.common_step_counter % max(1, env.max_episode_length // 4) == 0:
        print(
            f"[TaskDPitCurriculum] mean_level={mean_level.item():.2f} "
            f"mean_width={mean_width.item():.3f}m "
            f"promoted={int(promote_ids.numel())}/{int(env_ids_t.numel())} resetting envs",
            flush=True,
        )
    return mean_width
