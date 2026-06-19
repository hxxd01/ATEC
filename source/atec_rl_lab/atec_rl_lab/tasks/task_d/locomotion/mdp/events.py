"""Reset events for Task D pit locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils import math as math_utils

from isaaclab.managers import SceneEntityCfg

from atec_rl_lab.tasks.task_d.env_cfg import TASK_D_ROBOT_SPAWN_LOCAL

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Pit MARG / pit-loco only: feet-on-ground standing height (teleop B2Piper ~z=0.529).
# Full Task D platform spawn stays at TASK_D_ROBOT_SPAWN_LOCAL z=0.8 in env_cfg.py.
TASK_D_PIT_MARG_SPAWN_Z = 0.529


def task_d_pit_marg_spawn_local(
    local_x: float | None = None,
    local_y: float | None = None,
) -> tuple[float, float, float]:
    """Pit locomotion spawn: Task D xy reference + MARG standing z."""
    base = TASK_D_ROBOT_SPAWN_LOCAL
    return (
        float(base[0] if local_x is None else local_x),
        float(base[1] if local_y is None else local_y),
        TASK_D_PIT_MARG_SPAWN_Z,
    )


def reset_robot_at_task_d_spawn(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    local_pos: tuple[float, float, float] | None = None,
    local_pos_x_jitter: float = 0.0,
    local_pos_y_jitter: float = 0.0,
    local_pos_y_jitter_right: float = 0.0,
    local_pos_z_jitter: float = 0.0,
    yaw_range: tuple[float, float] = (0.0, 0.0),
    lin_vel_x_range: tuple[float, float] = (0.0, 0.0),
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0),
    ang_vel_z_range: tuple[float, float] = (0.0, 0.0),
):
    """Reset robot to Task D spawn: ``env_origin + local_pos`` (+ optional xy/yaw/velocity DR)."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    spawn = local_pos if local_pos is not None else task_d_pit_marg_spawn_local()
    local = torch.tensor(spawn, device=asset.device, dtype=root_states.dtype).view(1, 3).expand(n, 3).clone()
    x_jitter = float(local_pos_x_jitter)
    if x_jitter > 0.0:
        local[:, 0] += torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            -x_jitter, x_jitter
        )
    y_jitter = float(local_pos_y_jitter)
    y_jitter_right = float(local_pos_y_jitter_right)
    if y_jitter_right > 0.0:
        # +y is left (see Task D sidestep_left); subtract to jitter toward robot right.
        local[:, 1] -= torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            0.0, y_jitter_right
        )
    elif y_jitter > 0.0:
        local[:, 1] += torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            -y_jitter, y_jitter
        )
    z_jitter = float(local_pos_z_jitter)
    if z_jitter > 0.0:
        local[:, 2] += torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            -z_jitter, z_jitter
        )
    positions = env.scene.env_origins[env_ids] + local
    orientations = root_states[:, 3:7]
    yaw_min, yaw_max = float(yaw_range[0]), float(yaw_range[1])
    if abs(yaw_max - yaw_min) > 1e-9:
        yaws = torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(yaw_min, yaw_max)
        orientations_delta = math_utils.quat_from_euler_xyz(
            torch.zeros(n, device=asset.device, dtype=root_states.dtype),
            torch.zeros(n, device=asset.device, dtype=root_states.dtype),
            yaws,
        )
        orientations = math_utils.quat_mul(orientations, orientations_delta)

    velocities = torch.zeros(n, 6, device=asset.device, dtype=root_states.dtype)
    vx_min, vx_max = float(lin_vel_x_range[0]), float(lin_vel_x_range[1])
    if abs(vx_max - vx_min) > 1e-9:
        velocities[:, 0] = torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            vx_min, vx_max
        )
    vy_min, vy_max = float(lin_vel_y_range[0]), float(lin_vel_y_range[1])
    if abs(vy_max - vy_min) > 1e-9:
        velocities[:, 1] = torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            vy_min, vy_max
        )
    wz_min, wz_max = float(ang_vel_z_range[0]), float(ang_vel_z_range[1])
    if abs(wz_max - wz_min) > 1e-9:
        velocities[:, 5] = torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            wz_min, wz_max
        )

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# Teleop ~step744 pushed box (env-local); world=(-1.067, +1.622, +0.300) with pit ref origin (-4.2, 0).
TASK_D_PIT_PUSHED_BOX_LOCAL = (3.133, 1.622, 0.300)
PIT_BOX_SPAWN_X_JITTER_DOWN = 0.5
PIT_BOX_SPAWN_Y_JITTER = 0.5


def reset_box_at_task_d_spawn(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    local_pos: tuple[float, float, float] | None = None,
    local_pos_x_jitter_down: float = 0.0,
    local_pos_y_jitter: float = 0.0,
    local_pos_z_jitter: float = 0.0,
):
    """Reset box to ``env_origin + local_pos`` (+ optional pushed-box DR)."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    spawn = local_pos if local_pos is not None else TASK_D_PIT_PUSHED_BOX_LOCAL
    local = torch.tensor(spawn, device=asset.device, dtype=root_states.dtype).view(1, 3).expand(n, 3).clone()
    x_down = float(local_pos_x_jitter_down)
    if x_down > 0.0:
        local[:, 0] -= torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(0.0, x_down)
    y_jitter = float(local_pos_y_jitter)
    if y_jitter > 0.0:
        local[:, 1] += torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            -y_jitter, y_jitter
        )
    z_jitter = float(local_pos_z_jitter)
    if z_jitter > 0.0:
        local[:, 2] += torch.empty(n, device=asset.device, dtype=root_states.dtype).uniform_(
            -z_jitter, z_jitter
        )
    positions = env.scene.env_origins[env_ids] + local
    orientations = root_states[:, 3:7]
    velocities = torch.zeros(n, 6, device=asset.device, dtype=root_states.dtype)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
