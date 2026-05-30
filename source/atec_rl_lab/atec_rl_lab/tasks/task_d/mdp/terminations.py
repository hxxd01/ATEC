# Created by skywoodsz on 4/4/26.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

from .env_origin import (
    task_d_env_origin_xy,
    task_d_nominal_to_world,
    task_d_nominal_x_to_world,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_x_greater_than(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold: float = 2.0,
) -> torch.Tensor:
    """Terminate when robot root x (world frame) is greater than threshold + env origin."""
    robot = env.scene[asset_cfg.name]
    env_origin_x, _ = task_d_env_origin_xy(env)
    thresh = torch.full_like(env_origin_x, float(x_threshold))
    return robot.data.root_pos_w[:, 0] > task_d_nominal_x_to_world(thresh, env_origin_x)


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


class NoTargetProgressTimeout(ManagerTermBase):
    """Terminate when net progress toward stage target over a time window is too small."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initialized = False
        self._window_size = 1
        self._progress_eps = 0.05
        self._dist_history = None
        self._head = 0
        self._filled = None
        self._prev_stage_idx = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stuck_time_s: float = 1.0,
        progress_eps: float = 0.05,
        dist_attr: str = "_nav_dist_to_target",
        stage_idx_attr: str = "_nav_stage_idx",
        active_attr: str = "_nav_stage_active",
        push_active_attr: str = "_nav_push_stuck_active",
    ) -> torch.Tensor:
        if not self._initialized:
            self._window_size = max(1, int(float(stuck_time_s) / env.step_dt))
            self._progress_eps = float(progress_eps)
            self._dist_history = torch.full(
                (env.num_envs, self._window_size),
                float("nan"),
                device=env.device,
                dtype=torch.float32,
            )
            self._filled = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            self._prev_stage_idx = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
            self._head = 0
            self._initialized = True
            self.reset()

        root = env.unwrapped if hasattr(env, "unwrapped") else env
        dist_t = getattr(root, dist_attr, None)
        if dist_t is None or not isinstance(dist_t, torch.Tensor) or int(dist_t.shape[0]) != int(env.num_envs):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        dist = dist_t.to(device=env.device, dtype=torch.float32).view(-1)
        stage_idx = getattr(root, stage_idx_attr, None)
        if stage_idx is None:
            stage_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        else:
            stage_idx = stage_idx.to(device=env.device, dtype=torch.long).view(-1)

        active_t = getattr(root, active_attr, None)
        if active_t is None:
            active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        else:
            active = active_t.to(device=env.device, dtype=torch.bool).view(-1)

        stage_changed = stage_idx != self._prev_stage_idx
        reset_mask = stage_changed | (~active)
        if bool(reset_mask.any()):
            self._dist_history[reset_mask] = float("nan")
            self._filled[reset_mask] = 0

        oldest = self._head
        dist_oldest = self._dist_history[:, oldest]
        self._dist_history[:, self._head] = dist
        self._head = (self._head + 1) % self._window_size
        self._filled = torch.minimum(
            self._filled + 1,
            torch.full_like(self._filled, self._window_size),
        )
        self._prev_stage_idx = stage_idx.clone()

        window_ready = self._filled >= self._window_size
        net_progress = dist_oldest - dist
        no_progress = window_ready & (~torch.isnan(dist_oldest)) & (net_progress <= self._progress_eps)

        push_active_t = getattr(root, push_active_attr, None)
        if push_active_t is not None and isinstance(push_active_t, torch.Tensor):
            push_active = push_active_t.to(device=env.device, dtype=torch.bool).view(-1)
            no_progress = no_progress & (~push_active)

        return active & no_progress

    def reset(self, env_ids=None):
        if not self._initialized:
            return
        if env_ids is None:
            self._dist_history.fill_(float("nan"))
            self._filled.zero_()
            self._prev_stage_idx.fill_(-1)
            self._head = 0
        else:
            self._dist_history[env_ids] = float("nan")
            self._filled[env_ids] = 0
            self._prev_stage_idx[env_ids] = -1


class PushStageStuckTimeout(ManagerTermBase):
    """Terminate on push when box stalls on both axes (and not approaching when far/no contact)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initialized = False
        self._window_size = 1
        self._progress_eps = 0.05
        self._right_cap = 2.0
        self._right_history = None
        self._forward_history = None
        self._approach_history = None
        self._head = 0
        self._filled = None
        self._prev_stage_idx = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stuck_time_s: float = 2.0,
        progress_eps: float = 0.05,
        push_right_cap: float = 2.0,
        push_active_attr: str = "_nav_push_stuck_active",
        right_progress_attr: str = "_nav_push_box_right_progress",
        forward_progress_attr: str = "_nav_push_box_forward_progress",
        robot_box_dist_attr: str = "_nav_push_robot_box_dist",
        contact_attr: str = "_nav_push_in_contact",
        approach_dist_min: float = 1.2,
        stage_idx_attr: str = "_nav_stage_idx",
        active_attr: str = "_nav_stage_active",
    ) -> torch.Tensor:
        if not self._initialized:
            self._window_size = max(1, int(float(stuck_time_s) / env.step_dt))
            self._progress_eps = float(progress_eps)
            self._right_cap = float(push_right_cap)
            n = env.num_envs
            dev = env.device
            shape = (n, self._window_size)
            self._right_history = torch.full(shape, float("nan"), device=dev, dtype=torch.float32)
            self._forward_history = torch.full(shape, float("nan"), device=dev, dtype=torch.float32)
            self._approach_history = torch.full(shape, float("nan"), device=dev, dtype=torch.float32)
            self._filled = torch.zeros(n, device=dev, dtype=torch.long)
            self._prev_stage_idx = torch.full((n,), -1, device=dev, dtype=torch.long)
            self._head = 0
            self._initialized = True
            self.reset()

        root = env.unwrapped if hasattr(env, "unwrapped") else env
        push_active_t = getattr(root, push_active_attr, None)
        if push_active_t is None or not isinstance(push_active_t, torch.Tensor):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        push_active = push_active_t.to(device=env.device, dtype=torch.bool).view(-1)

        right_t = getattr(root, right_progress_attr, None)
        forward_t = getattr(root, forward_progress_attr, None)
        approach_t = getattr(root, robot_box_dist_attr, None)
        if (
            right_t is None
            or forward_t is None
            or approach_t is None
            or not all(
                isinstance(t, torch.Tensor) and int(t.shape[0]) == int(env.num_envs)
                for t in (right_t, forward_t, approach_t)
            )
        ):
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        right_prog = right_t.to(device=env.device, dtype=torch.float32).view(-1)
        forward_prog = forward_t.to(device=env.device, dtype=torch.float32).view(-1)
        approach = approach_t.to(device=env.device, dtype=torch.float32).view(-1)

        stage_idx = getattr(root, stage_idx_attr, None)
        if stage_idx is None:
            stage_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        else:
            stage_idx = stage_idx.to(device=env.device, dtype=torch.long).view(-1)

        active_t = getattr(root, active_attr, None)
        if active_t is None:
            active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        else:
            active = active_t.to(device=env.device, dtype=torch.bool).view(-1)

        stage_changed = stage_idx != self._prev_stage_idx
        reset_mask = stage_changed | (~active) | (~push_active)
        if bool(reset_mask.any()):
            self._right_history[reset_mask] = float("nan")
            self._forward_history[reset_mask] = float("nan")
            self._approach_history[reset_mask] = float("nan")
            self._filled[reset_mask] = 0

        oldest = self._head
        right_old = self._right_history[:, oldest]
        forward_old = self._forward_history[:, oldest]
        approach_old = self._approach_history[:, oldest]

        self._right_history[:, self._head] = right_prog
        self._forward_history[:, self._head] = forward_prog
        self._approach_history[:, self._head] = approach
        self._head = (self._head + 1) % self._window_size
        self._filled = torch.minimum(
            self._filled + 1,
            torch.full_like(self._filled, self._window_size),
        )
        self._prev_stage_idx = stage_idx.clone()

        window_ready = self._filled >= self._window_size
        right_delta = right_prog - right_old
        forward_delta = forward_prog - forward_old
        approach_delta = approach_old - approach

        right_capped = right_prog >= (self._right_cap - self._progress_eps)
        right_progressing = (~torch.isnan(right_old)) & (right_delta > self._progress_eps)
        forward_progressing = (~torch.isnan(forward_old)) & (forward_delta > self._progress_eps)
        approach_progressing = (~torch.isnan(approach_old)) & (approach_delta > self._progress_eps)

        contact_t = getattr(root, contact_attr, None)
        if contact_t is None or not isinstance(contact_t, torch.Tensor):
            in_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        else:
            in_contact = contact_t.to(device=env.device, dtype=torch.bool).view(-1)
        # While pushing in contact (or already close), distance naturally stays flat — do not
        # require monotonic approach; only check box axis progress.
        approach_required = (~in_contact) & (approach > float(approach_dist_min))

        # All box-axis checks must stall; skip right-axis after right cap is reached.
        right_stuck = torch.where(right_capped, torch.zeros_like(right_prog, dtype=torch.bool), ~right_progressing)
        forward_stuck = ~forward_progressing
        approach_stuck = approach_required & (~approach_progressing)
        box_axes_stuck = right_stuck & forward_stuck
        all_stuck = window_ready & box_axes_stuck & (~approach_required | approach_stuck)
        return active & push_active & all_stuck

    def reset(self, env_ids=None):
        if not self._initialized:
            return
        if env_ids is None:
            self._right_history.fill_(float("nan"))
            self._forward_history.fill_(float("nan"))
            self._approach_history.fill_(float("nan"))
            self._filled.zero_()
            self._prev_stage_idx.fill_(-1)
            self._head = 0
        else:
            self._right_history[env_ids] = float("nan")
            self._forward_history[env_ids] = float("nan")
            self._approach_history[env_ids] = float("nan")
            self._filled[env_ids] = 0
            self._prev_stage_idx[env_ids] = -1


class StageTargetDeviationTermination(ManagerTermBase):
    """Terminate when robot is too far from the current stage trajectory segment."""

    # Fallback nominal stage endpoints if wrapper has not synchronized trajectory segments yet.
    _STAGE_STARTS = (
        (-3.00, 0.00),   # retreat
        (-4.00, 0.00),   # sidestep_left
        (-4.00, 2.10),   # push
        (0.00, 0.10),    # final
    )
    _STAGE_TARGETS = (
        (-4.00, 0.00),   # retreat
        (-4.00, 2.10),   # sidestep_left
        (0.00, 0.10),    # push (2 m right + 4 m forward nominal)
        (3.00, 0.10),    # final (+3 m along x)
    )

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._initialized = False
        self._start_x = None
        self._start_y = None
        self._target_x = None
        self._target_y = None

    def _lazy_init(self, device: str | torch.device) -> None:
        if self._initialized:
            return
        self._start_x = torch.tensor([p[0] for p in self._STAGE_STARTS], device=device, dtype=torch.float32)
        self._start_y = torch.tensor([p[1] for p in self._STAGE_STARTS], device=device, dtype=torch.float32)
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
        root = env.unwrapped if hasattr(env, "unwrapped") else env
        robot = root.scene[robot_asset_cfg.name]
        pos = robot.data.root_pos_w.to(device=env.device, dtype=torch.float32)
        stage_idx = getattr(root, stage_idx_attr, None)
        if stage_idx is None:
            stage_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        else:
            stage_idx = stage_idx.to(device=env.device, dtype=torch.long).view(-1)
        stage_idx = torch.clamp(stage_idx, min=0, max=len(self._STAGE_TARGETS) - 1)

        seg_x0 = getattr(root, "_nav_seg_x0", None)
        seg_y0 = getattr(root, "_nav_seg_y0", None)
        seg_x1 = getattr(root, "_nav_seg_x1", None)
        seg_y1 = getattr(root, "_nav_seg_y1", None)
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
            env_origin_x, env_origin_y = task_d_env_origin_xy(env)
            x0, y0 = task_d_nominal_to_world(
                self._start_x[stage_idx], self._start_y[stage_idx], env_origin_x, env_origin_y
            )
            x1, y1 = task_d_nominal_to_world(
                self._target_x[stage_idx], self._target_y[stage_idx], env_origin_x, env_origin_y
            )

        vx = x1 - x0
        vy = y1 - y0
        wx = pos[:, 0] - x0
        wy = pos[:, 1] - y0
        seg_len2 = vx * vx + vy * vy
        valid_seg = seg_len2 > 1.0e-6
        t = torch.where(valid_seg, (wx * vx + wy * vy) / seg_len2, torch.zeros_like(seg_len2))
        t = torch.clamp(t, 0.0, 1.0)
        proj_x = x0 + t * vx
        proj_y = y0 + t * vy
        dist = torch.hypot(pos[:, 0] - proj_x, pos[:, 1] - proj_y)
        too_far = dist > float(max_dist)
        return valid_seg & too_far


class TrajectoryDeviationTermination(ManagerTermBase):
    """Deprecated: kept for backward compatibility, delegates to stage-target logic."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._delegate = StageTargetDeviationTermination(cfg, env)

    def __call__(self, env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
        robot_max_dist = float(kwargs.pop("robot_max_dist", kwargs.pop("max_dist", 1.2)))
        return self._delegate(env, max_dist=robot_max_dist, **kwargs)
