"""Reset events for Task D pit locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from atec_rl_lab.tasks.task_d.env_cfg import TASK_D_ROBOT_SPAWN_LOCAL

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_robot_at_task_d_spawn(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    local_pos: tuple[float, float, float] | None = None,
):
    """Reset robot to Task D default spawn: ``env_origin + TASK_D_ROBOT_SPAWN_LOCAL``."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    spawn = local_pos if local_pos is not None else TASK_D_ROBOT_SPAWN_LOCAL
    local = torch.tensor(spawn, device=asset.device, dtype=root_states.dtype).view(1, 3).expand(n, 3)
    positions = env.scene.env_origins[env_ids] + local
    orientations = root_states[:, 3:7]
    velocities = torch.zeros(n, 6, device=asset.device, dtype=root_states.dtype)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
