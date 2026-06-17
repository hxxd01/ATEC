"""Privileged observations for Task D pit locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

import atec_rl_lab.train.locomotion.velocity.mdp as vel_mdp

from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import (
    pit_cross_world_x,
    pit_geometry_for_envs,
)
from atec_rl_lab.train.locomotion.marg.constants import (
    MARG_CRITIC_PRIV_DIM,
    MARG_HISTORY_DIM,
    MARG_HISTORY_LEN,
    MARG_PROPRIO_DIM,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

_DEFAULT_LEG_JOINTS = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
_DEFAULT_FOOT_BODIES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


def robot_world_position(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot root position in world frame, shape (num_envs, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def pit_geometry(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Pit center (world xyz) + pit size (length_y, width_x, depth_z), shape (num_envs, 6)."""
    center_w, size_lwh = pit_geometry_for_envs(env)
    return torch.cat([center_w, size_lwh], dim=-1)


def _leg_joint_ids(env: ManagerBasedEnv, joint_names: list[str] | None) -> list[int]:
    if joint_names is None and hasattr(env.cfg, "joint_names"):
        joint_names = list(env.cfg.joint_names)
    names = joint_names if joint_names is not None else _DEFAULT_LEG_JOINTS
    asset: Articulation = env.scene["robot"]
    return [asset.data.joint_names.index(name) for name in names]


def _foot_body_ids(env: ManagerBasedEnv, body_names: list[str] | None) -> list[int]:
    asset: Articulation = env.scene["robot"]
    if body_names is not None:
        return [asset.data.body_names.index(name) for name in body_names]
    foot_ids = [i for i, name in enumerate(asset.data.body_names) if name.endswith("_foot")]
    if len(foot_ids) != 4:
        raise RuntimeError(f"Expected 4 foot bodies, found {len(foot_ids)}: {foot_ids}")
    return foot_ids


def _marg_proprio_vector(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """MARG actor proprio (43D): no base lin vel; cmd is forward vx only."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _leg_joint_ids(env, joint_names)
    n_j = len(joint_ids)

    ang_vel = vel_mdp.base_ang_vel(env, asset_cfg=asset_cfg)
    gravity = vel_mdp.projected_gravity(env, asset_cfg=asset_cfg)
    cmd = vel_mdp.generated_commands(env, command_name="base_velocity")[:, 0:1]
    joint_pos = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids]
    actions = vel_mdp.last_action(env)[..., :n_j]

    vec = torch.cat([ang_vel, gravity, cmd, joint_pos, joint_vel, actions], dim=-1)
    if vec.shape[-1] != MARG_PROPRIO_DIM:
        raise RuntimeError(
            f"MARG proprio dim {vec.shape[-1]} != expected {MARG_PROPRIO_DIM}. "
            f"components: ang={ang_vel.shape[-1]}, grav={gravity.shape[-1]}, cmd={cmd.shape[-1]}, "
            f"q={joint_pos.shape[-1]}, dq={joint_vel.shape[-1]}, act={actions.shape[-1]}"
        )
    return vec


def _get_proprio_history_buffer(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "_marg_proprio_history"):
        env._marg_proprio_history = torch.zeros(
            env.num_envs,
            MARG_HISTORY_LEN,
            MARG_PROPRIO_DIM,
            device=env.device,
            dtype=torch.float32,
        )
    return env._marg_proprio_history


def marg_proprio(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Current-step MARG proprio observation."""
    return _marg_proprio_vector(env, asset_cfg=asset_cfg, joint_names=joint_names)


def marg_proprio_history(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Stacked proprio history [o_t, o_{t-1}, ...], flattened to 258D."""
    current = _marg_proprio_vector(env, asset_cfg=asset_cfg, joint_names=joint_names)
    hist = _get_proprio_history_buffer(env)

    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            hist[reset_mask] = 0.0

    hist[:, 1:] = hist[:, :-1].clone()
    hist[:, 0] = current
    return hist.reshape(env.num_envs, MARG_HISTORY_DIM)


def marg_height_map(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
) -> torch.Tensor:
    """Egocentric relative height map (187D) for the elevation encoder."""
    return vel_mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)


def marg_critic_privileged(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    joint_names: list[str] | None = None,
    foot_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Privileged critic state (42D) + estimator regression targets in first 7 dims."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _leg_joint_ids(env, joint_names)
    foot_ids = _foot_body_ids(env, foot_body_names)

    base_lin_vel = vel_mdp.base_lin_vel(env, asset_cfg=asset_cfg)

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    foot_sensor_ids = contact_sensor_cfg.body_ids
    if foot_sensor_ids is None or len(foot_sensor_ids) == 0:
        foot_sensor_ids = list(range(len(foot_ids)))
    contact_forces = contact_sensor.data.net_forces_w_history[:, -1, foot_sensor_ids, :]
    foot_contact = (contact_forces.norm(dim=-1) > 1.0).to(dtype=torch.float32)

    foot_masses = asset.root_physx_view.get_masses()[:, foot_ids].to(device=env.device, dtype=torch.float32)

    friction = torch.ones(env.num_envs, 1, device=env.device, dtype=torch.float32)

    com_xy = asset.data.body_com_pos_b[:, 0, :2]

    base_name = "base_link"
    base_id = asset.data.body_names.index(base_name) if base_name in asset.data.body_names else 0
    disturbance_xy = asset.data.body_incoming_joint_wrench_b[:, base_id, :2]

    default_stiff = asset.data.default_joint_stiffness[:, joint_ids]
    default_damp = asset.data.default_joint_damping[:, joint_ids]
    stiff = asset.data.joint_stiffness[:, joint_ids] / (default_stiff + 1e-8)
    damp = asset.data.joint_damping[:, joint_ids] / (default_damp + 1e-8)
    motor_strength = torch.ones(env.num_envs, 1, device=env.device, dtype=torch.float32)
    offset_scale = torch.ones(env.num_envs, 1, device=env.device, dtype=torch.float32)
    k_pd = torch.cat([stiff, damp, motor_strength, offset_scale], dim=-1)

    priv = torch.cat(
        [
            base_lin_vel,
            foot_contact,
            foot_masses,
            friction,
            com_xy,
            disturbance_xy,
            k_pd,
        ],
        dim=-1,
    )
    if priv.shape[-1] != MARG_CRITIC_PRIV_DIM:
        raise RuntimeError(
            f"marg_critic_privileged dim {priv.shape[-1]} != expected {MARG_CRITIC_PRIV_DIM}"
        )
    return priv
