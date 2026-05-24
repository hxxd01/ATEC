# Created by skywoodsz on 4/4/26.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_x_greater_than(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold: float = 2.0,
) -> torch.Tensor:
    """Terminate when robot root x (world frame) is greater than threshold."""
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 0] > float(x_threshold)


class NoMotionTimeout(ManagerTermBase):
    """Terminate when robot horizontal speed stays below threshold for some time."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initialized = False
        self._asset_name = None
        self._stuck_counter = None
        self._stuck_step_threshold = 1
        self._speed_eps = 0.03

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        stuck_time_s: float = 1.5,
        speed_eps: float = 0.03,
    ) -> torch.Tensor:
        if not self._initialized:
            self._asset_name = asset_cfg.name
            self._stuck_step_threshold = max(1, int(float(stuck_time_s) / env.step_dt))
            self._speed_eps = float(speed_eps)
            self._stuck_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            self._initialized = True
            self.reset()

        robot = env.scene[self._asset_name]
        speed_xy = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
        moving = speed_xy > self._speed_eps
        self._stuck_counter = torch.where(
            moving, torch.zeros_like(self._stuck_counter), self._stuck_counter + 1
        )
        return self._stuck_counter >= self._stuck_step_threshold

    def reset(self, env_ids=None):
        if not self._initialized:
            return
        if env_ids is None:
            self._stuck_counter.zero_()
        else:
            self._stuck_counter[env_ids] = 0


class StageTargetDeviationTermination(ManagerTermBase):
    """Terminate when robot is too far from the current stage trajectory segment."""

    # Fallback nominal stage targets if wrapper has not synchronized trajectory segments yet.
    _STAGE_TARGETS = (
        (-4.00, 0.00),   # retreat
        (-4.00, 2.10),   # sidestep_left
        (-3.00, 2.10),   # advance
        (-3.00, 0.10),   # sidestep_right
        (-4.00, 0.10),   # retreat2
        (-4.00, -0.50),  # sidestep_right2
        (-1.25, -0.50),  # advance2
        (1.75, -0.50),   # final
    )

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initialized = False
        self._target_x = None
        self._target_y = None

    def _lazy_init(self, device: str | torch.device) -> None:
        if self._initialized:
            return
        xs = [p[0] for p in self._STAGE_TARGETS]
        ys = [p[1] for p in self._STAGE_TARGETS]
        self._target_x = torch.tensor(xs, device=device, dtype=torch.float32)
        self._target_y = torch.tensor(ys, device=device, dtype=torch.float32)
        self._initialized = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_dist: float = 1.2,
        stage_idx_attr: str = "_nav_stage_idx",
    ) -> torch.Tensor:
        self._lazy_init(env.device)
        robot = env.scene[robot_asset_cfg.name]
        pos = robot.data.root_pos_w.to(device=env.device, dtype=torch.float32)
        stage_idx = getattr(env, stage_idx_attr, None)
        if stage_idx is None:
            stage_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        else:
            stage_idx = stage_idx.to(device=env.device, dtype=torch.long).view(-1)
        stage_idx = torch.clamp(stage_idx, min=0, max=len(self._STAGE_TARGETS) - 1)

        seg_x0 = getattr(env, "_nav_seg_x0", None)
        seg_y0 = getattr(env, "_nav_seg_y0", None)
        seg_x1 = getattr(env, "_nav_seg_x1", None)
        seg_y1 = getattr(env, "_nav_seg_y1", None)
        has_synced_segments = all(
            isinstance(t, torch.Tensor) and int(t.shape[0]) == int(env.num_envs)
            for t in (seg_x0, seg_y0, seg_x1, seg_y1)
        )

        if has_synced_segments:
            x0 = seg_x0.to(device=env.device, dtype=torch.float32)
            y0 = seg_y0.to(device=env.device, dtype=torch.float32)
            x1 = seg_x1.to(device=env.device, dtype=torch.float32)
            y1 = seg_y1.to(device=env.device, dtype=torch.float32)
        else:
            # Fallback to point-distance when segment sync is unavailable.
            x0 = self._target_x[stage_idx]
            y0 = self._target_y[stage_idx]
            x1 = x0
            y1 = y0

        vx = x1 - x0
        vy = y1 - y0
        wx = pos[:, 0] - x0
        wy = pos[:, 1] - y0
        seg_len2 = vx * vx + vy * vy
        t = torch.where(seg_len2 > 1.0e-8, (wx * vx + wy * vy) / seg_len2, torch.zeros_like(seg_len2))
        t = torch.clamp(t, 0.0, 1.0)
        proj_x = x0 + t * vx
        proj_y = y0 + t * vy
        dist = torch.hypot(pos[:, 0] - proj_x, pos[:, 1] - proj_y)
        return dist > float(max_dist)


class TrajectoryDeviationTermination(ManagerTermBase):
    """Deprecated: kept for backward compatibility, delegates to stage-target logic."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._delegate = StageTargetDeviationTermination(cfg, env)

    def __call__(self, env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
        robot_max_dist = float(kwargs.pop("robot_max_dist", kwargs.pop("max_dist", 1.2)))
        return self._delegate(env, max_dist=robot_max_dist, **kwargs)
