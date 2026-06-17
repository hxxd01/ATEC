"""Rewards for Task D pit locomotion (MARG Table I + legacy helpers)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor, RayCaster

from atec_rl_lab.tasks.task_d.locomotion.mdp.terminations import pit_cross_local_x_success_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# MARG height-map grid: 1.6 m x 1.0 m @ 0.1 m -> 17 x 11 = 187
_MARG_HEIGHT_GRID_X = 17
_MARG_HEIGHT_GRID_Y = 11
_MARG_HEIGHT_RES = 0.1
_MARG_HEIGHT_X0 = -0.8
_MARG_HEIGHT_Y0 = -0.5


def _upright_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def _height_map_relative(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Relative height map (num_envs, 187), same convention as ``mdp.height_scan``."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5


def _sample_height_map_body(
    env: ManagerBasedRLEnv,
    body_xy: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bilinear-nearest sample of relative height map at body-frame (x, y) points."""
    height_map = _height_map_relative(env, sensor_cfg)
    ix = torch.round((body_xy[..., 0] - _MARG_HEIGHT_X0) / _MARG_HEIGHT_RES).long()
    iy = torch.round((body_xy[..., 1] - _MARG_HEIGHT_Y0) / _MARG_HEIGHT_RES).long()
    ix = ix.clamp(0, _MARG_HEIGHT_GRID_X - 1)
    iy = iy.clamp(0, _MARG_HEIGHT_GRID_Y - 1)
    idx = ix * _MARG_HEIGHT_GRID_Y + iy
    return height_map.gather(1, idx)


def marg_feet_center(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    height_threshold: float = -0.2,
    d1: float = 0.05,
    d2: float = math.sqrt(50.0) * 0.01,
) -> torch.Tensor:
    """MARG feet-center penalty: c_t * (n2 + 2*n3) for stepping near terrain edges."""
    asset: Articulation = env.scene[foot_asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    n_feet = len(foot_asset_cfg.body_ids)
    foot_pos_w = asset.data.body_link_pos_w[:, foot_asset_cfg.body_ids, :] - asset.data.root_link_pos_w.unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, n_feet, 3, device=env.device, dtype=foot_pos_w.dtype)
    root_quat_inv = math_utils.quat_conjugate(asset.data.root_link_quat_w)
    for i in range(n_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply(root_quat_inv, foot_pos_w[:, i, :])
    foot_pos_b = foot_pos_b[..., :2]

    diag = d2 / math.sqrt(2.0)
    offsets = torch.tensor(
        [
            [0.0, 0.0],
            [d1, 0.0],
            [-d1, 0.0],
            [0.0, d1],
            [0.0, -d1],
            [diag, diag],
            [diag, -diag],
            [-diag, diag],
            [-diag, -diag],
        ],
        device=env.device,
        dtype=foot_pos_b.dtype,
    )
    sample_xy = foot_pos_b.unsqueeze(2) + offsets.view(1, 1, 9, 2)
    sample_xy = sample_xy.reshape(env.num_envs, n_feet, 9, 2)

    heights = _sample_height_map_body(
        env,
        sample_xy.reshape(env.num_envs, n_feet * 9, 2),
        sensor_cfg,
    ).view(env.num_envs, n_feet, 9)

    # types 2 and 3 are offsets 1:5 and 5:9; penalize points below threshold.
    n2 = (heights[:, :, 1:5] < height_threshold).float().sum(dim=2)
    n3 = (heights[:, :, 5:9] < height_threshold).float().sum(dim=2)

    contacts = contact_sensor.data.net_forces_w_history[:, -1, contact_sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    penalty = (contacts.float() * (n2 + 2.0 * n3)).sum(dim=1)
    return penalty * _upright_scale(env)


def forward_world_x_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for forward (+world x) velocity toward crossing the pit."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0)


class PitCrossSuccessBonus(ManagerTermBase):
    """One-time sparse bonus when local +x (relative to env_origin) crosses the success threshold."""

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
        reward_value: float = 5.0,
        local_x_threshold: float = 2.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        success = pit_cross_local_x_success_mask(
            env,
            local_x_threshold=local_x_threshold,
            asset_cfg=asset_cfg,
        )
        trigger = success & (~self._reward_given)
        self._reward_given |= success
        return trigger.float() * float(reward_value)
