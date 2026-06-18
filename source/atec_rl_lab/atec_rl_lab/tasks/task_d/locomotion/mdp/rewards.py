"""Rewards for Task D pit locomotion."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

from atec_rl_lab.tasks.task_d.locomotion.mdp.terminations import pit_cross_local_x_success_mask
from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import pit_geometry_for_envs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_world_x_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for forward (+world x) velocity toward crossing the pit."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0)


def forward_world_x_speed_capped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    v_cap: float = 5.0,
) -> torch.Tensor:
    """Bounded forward-speed reward in [0, 1] with saturation at ``v_cap``.

    - Only rewards +x velocity.
    - Increases linearly for ``0 <= vx <= v_cap``.
    - For ``vx > v_cap``, reward does not increase further.
    """
    asset = env.scene[asset_cfg.name]
    vx = torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0)
    cap = max(float(v_cap), 1e-6)
    return torch.clamp(vx, max=cap) / cap


class PitCrossSuccessBonus(ManagerTermBase):
    """One-time sparse bonus when local +x crosses the pit success threshold."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._reward_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._reward_given.zero_()
        else:
            self._reward_given[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reward_value: float = 30.0,
        post_cross_distance: float = 1.5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        success = pit_cross_local_x_success_mask(
            env,
            post_cross_distance=post_cross_distance,
            asset_cfg=asset_cfg,
        )
        trigger = success & (~self._reward_given)
        self._reward_given |= success
        return trigger.float() * float(reward_value)


def marg_feet_center_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    d1: float = 0.05,
    d2: float = math.sqrt(0.005),
    c_t: float = 1.0,
) -> torch.Tensor:
    """Approximate MARG feet-center penalty from pit geometry.

    We sample 9 points per foot in world XY:
    - type-1: center point (not penalized directly)
    - type-2: 4 axis offsets at distance ``d1``
    - type-3: 4 diagonal offsets at distance ``d2``
    and mark a point unsafe if it falls inside the pit opening (terrain height < -0.2m).
    Penalty follows paper form ``c_t * (n2 + 2*n3)`` where ``n2``/``n3`` are episode-step flags.
    """
    asset = env.scene[asset_cfg.name]
    foot_ids = [i for i, name in enumerate(asset.data.body_names) if name.endswith("_foot")]
    if len(foot_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    foot_xy = asset.data.body_pos_w[:, foot_ids, :2]  # [N, F, 2]
    center_w, size_lwh = pit_geometry_for_envs(env)
    pit_xy = center_w[:, :2]  # [N, 2]
    pit_half_len_y = 0.5 * size_lwh[:, 0]  # [N]
    pit_half_wid_x = 0.5 * size_lwh[:, 1]  # [N]

    axis_offsets = torch.tensor(
        [[d1, 0.0], [-d1, 0.0], [0.0, d1], [0.0, -d1]],
        device=env.device,
        dtype=foot_xy.dtype,
    )  # [4, 2]
    diag_offsets = torch.tensor(
        [[d2, d2], [d2, -d2], [-d2, d2], [-d2, -d2]],
        device=env.device,
        dtype=foot_xy.dtype,
    )  # [4, 2]

    def _inside_pit(points_xy: torch.Tensor) -> torch.Tensor:
        # points_xy: [N, F, P, 2]
        dx = torch.abs(points_xy[..., 0] - pit_xy[:, None, None, 0])
        dy = torch.abs(points_xy[..., 1] - pit_xy[:, None, None, 1])
        return (dx <= pit_half_wid_x[:, None, None]) & (dy <= pit_half_len_y[:, None, None])

    axis_pts = foot_xy[:, :, None, :] + axis_offsets[None, None, :, :]  # [N, F, 4, 2]
    diag_pts = foot_xy[:, :, None, :] + diag_offsets[None, None, :, :]  # [N, F, 4, 2]
    n2 = _inside_pit(axis_pts).any(dim=(1, 2)).float()  # [N]
    n3 = _inside_pit(diag_pts).any(dim=(1, 2)).float()  # [N]
    return float(c_t) * (n2 + 2.0 * n3)
