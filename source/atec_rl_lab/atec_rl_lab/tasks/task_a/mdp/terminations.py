from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_x_greater_than(
    env: ManagerBasedEnv,
    asset_cfg,
    x_threshold: float,
) -> torch.Tensor:
    """Terminate when robot world x is greater than x_threshold."""
    robot = env.scene[asset_cfg.name]
    robot_x = robot.data.root_pos_w[:, 0]   # world x

    return robot_x > x_threshold


class StuckNoProgress(ManagerTermBase):
    """Terminate when the robot makes no forward (+x) progress for ``stuck_time_s``.

    Only counts *after* ``grace_time_s`` from each episode reset so the loco policy can
    settle (stand up / recover from reset) without instant stuck timeouts.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initialized = False
        self._max_x = None
        self._stuck_counter = None
        self._episode_step = None
        self._stuck_step_threshold = 1
        self._grace_steps = 0
        self._asset_name = None
        self._progress_eps = 0.05

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg,
        stuck_time_s: float = 3.0,
        progress_eps: float = 0.03,
        grace_time_s: float = 1.0,
    ) -> torch.Tensor:
        if not self._initialized:
            self._stuck_step_threshold = max(1, int(stuck_time_s / env.step_dt))
            self._grace_steps = max(0, int(grace_time_s / env.step_dt))
            self._progress_eps = progress_eps
            self._asset_name = asset_cfg.name
            n = env.num_envs
            self._max_x = torch.zeros(n, device=env.device, dtype=torch.float32)
            self._stuck_counter = torch.zeros(n, device=env.device, dtype=torch.long)
            self._episode_step = torch.zeros(n, device=env.device, dtype=torch.long)
            self._initialized = True
            self.reset()

        self._episode_step += 1
        past_grace = self._episode_step > self._grace_steps

        robot = env.scene[asset_cfg.name]
        robot_x = robot.data.root_pos_w[:, 0]

        progressed = robot_x > (self._max_x + self._progress_eps)
        self._max_x = torch.maximum(self._max_x, robot_x)
        self._stuck_counter = torch.where(
            progressed,
            torch.zeros_like(self._stuck_counter),
            torch.where(
                past_grace,
                self._stuck_counter + 1,
                torch.zeros_like(self._stuck_counter),
            ),
        )
        return self._stuck_counter >= self._stuck_step_threshold

    def reset(self, env_ids=None):
        if not self._initialized:
            return

        robot_x = self._env.scene[self._asset_name].data.root_pos_w[:, 0]
        if env_ids is None:
            self._max_x.copy_(robot_x)
            self._stuck_counter.zero_()
            self._episode_step.zero_()
        else:
            self._max_x[env_ids] = robot_x[env_ids]
            self._stuck_counter[env_ids] = 0
            self._episode_step[env_ids] = 0
