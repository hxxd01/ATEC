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
