"""Task B student wrapper: Task-D student network + Task-B touch-nav rewards."""

from __future__ import annotations

import gymnasium as gym
import isaaclab.utils.math as math_utils
import numpy as np
import torch

from .taskd_student_env import TaskDStudentEnv
from .taskd_teacher_env import _LEG_DIM

TASK_B_NUM_OBJECTS = 18
TASK_B_GRASP_DIST = 0.20
TASK_B_PROPRIO_DIM = 9  # lin_vel(3) + ang_vel(3) + gravity(3) — embedded in policy flat obs
TASK_B_SCORED_MASK_DIM = TASK_B_NUM_OBJECTS
# Task-D student layout: critic = policy flat + priv extras (see _build_critic_obs).
# priv: robot pose(4) + ee xyz(3) + r_vel(2) + contact(1)
#       + min_xy_unscored(1) + min_3d_unscored(1) + scored_mask(18) + trash xyz(54)
TASK_B_CRITIC_EXTRA_DIM = (
    4 + 3 + 2 + 1 + 1 + 1 + TASK_B_SCORED_MASK_DIM + TASK_B_NUM_OBJECTS * 3
)
# Lower-profile arm hold keeps EE lower while pulling shoulder/elbow aside to
# reduce head-camera occlusion during nav training.
_TASK_B_LOW_CLEAR_ARM_ACTION = torch.tensor(
    (0.0, 5.4, -3.6, 0.0, -1.2, 0.0, 0.0, 0.0), dtype=torch.float32
)


class TaskBStudentEnv(TaskDStudentEnv):
    """Nav student for Task B: dense EE-to-unscored-trash approach, sparse platform touch score."""

    def __init__(
        self,
        env: gym.Env,
        ll_policy_path: str,
        device: str = "cuda",
        inner_steps: int = 25,
        vx_min: float = -2.0,
        vx_max: float = 2.0,
        vy_max: float = 1.0,
        wz_max: float = 1.0,
        image_h: int = 24,
        image_w: int = 32,
        image_hw: int | None = None,
        depth_max: float = 5.0,
        depth_only: bool = False,
        nav_log_interval: int = 10,
        w_dense_dist: float = 0.3,
        sparse_touch_reward: float = 10.0,
        grasp_dist_thresh: float = TASK_B_GRASP_DIST,
        time_penalty_per_env_step: float = 0.004,
        ee_body_name: str = "gripper_base",
        no_touch_timeout_s: float = 12.0,
        finished_reward: float = 100.0,
        visible_depth_tol: float = 0.25,
        visible_check_depth: bool = False,
        guide_milestone_thresholds: tuple[float, ...] = (1.2, 0.8, 0.5, 0.3),
        guide_milestone_rewards: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
    ):
        super().__init__(
            env=env,
            ll_policy_path=ll_policy_path,
            device=device,
            inner_steps=inner_steps,
            vx_min=vx_min,
            vx_max=vx_max,
            vy_max=vy_max,
            wz_max=wz_max,
            image_h=image_h,
            image_w=image_w,
            image_hw=image_hw,
            depth_max=depth_max,
            depth_only=depth_only,
            nav_log_interval=nav_log_interval,
        )
        self._nav_log_tag = "TaskBStudent"
        self._w_dense_dist = float(w_dense_dist)
        self._dense_enabled = self._w_dense_dist > 0.0
        self._sparse_touch_reward = float(sparse_touch_reward)
        self._grasp_dist_thresh = float(grasp_dist_thresh)
        self._grasp_dist_sq = self._grasp_dist_thresh * self._grasp_dist_thresh
        self._time_penalty_per_env_step = float(time_penalty_per_env_step)
        self._ee_body_name = str(ee_body_name)
        self._no_touch_timeout_s = float(no_touch_timeout_s)
        self._finished_reward = float(finished_reward)
        self._visible_depth_tol = float(visible_depth_tol)
        self._visible_check_depth = bool(visible_check_depth)
        self._env_step_dt = float(getattr(self.env.unwrapped, "step_dt", 0.02))
        if len(guide_milestone_thresholds) != len(guide_milestone_rewards):
            raise ValueError("guide_milestone_thresholds and guide_milestone_rewards must have same length.")
        self._guide_thresholds = torch.tensor(
            guide_milestone_thresholds, device=self._device, dtype=torch.float32
        )
        self._guide_rewards = torch.tensor(
            guide_milestone_rewards, device=self._device, dtype=torch.float32
        )

        # Critic (Task-D student): actor flat + privileged extras; RNN value head in AC.
        self._critic_extra_dim = TASK_B_CRITIC_EXTRA_DIM
        self._critic_dim = self._actor_dim + self._critic_extra_dim
        self.observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._actor_dim,), dtype=np.float32
                ),
                "critic": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._critic_dim,), dtype=np.float32
                ),
            }
        )

        self._ee_body_idx: int | None = None
        self._prev_obj_dist_3d = torch.full(
            (self.num_envs, TASK_B_NUM_OBJECTS),
            float("nan"),
            device=self._device,
            dtype=torch.float32,
        )
        self._prev_min_guide_dist = torch.full(
            (self.num_envs,), float("nan"), device=self._device, dtype=torch.float32
        )
        self._guide_milestone_hit = torch.zeros(
            (self.num_envs, self._guide_thresholds.numel()), device=self._device, dtype=torch.bool
        )
        self._scored = torch.zeros(
            (self.num_envs, TASK_B_NUM_OBJECTS), device=self._device, dtype=torch.bool
        )
        self._time_since_last_touch_s = torch.zeros(
            (self.num_envs,), device=self._device, dtype=torch.float32
        )
        self._episode_finished = torch.zeros(
            (self.num_envs,), device=self._device, dtype=torch.bool
        )
        self._logged_first_rollout = False
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_illegal = 0
        self._done_no_touch = 0
        self._done_finished = 0
        # Accumulators for "scored objects at episode termination" metric, flushed
        # into extras["log"] each nav step so rsl_rl logs it to the table.
        self._scored_at_term_sum = 0.0
        self._scored_at_term_count = 0
        # Per-step termination-type counters (no_touch / finished are wrapper-level
        # truncations not registered in the MDP termination manager, so we surface
        # them through extras["log"] too).
        self._illegal_at_term_count = 0
        self._fall_at_term_count = 0
        self._timeout_at_term_count = 0
        self._no_touch_at_term_count = 0
        self._finished_at_term_count = 0
        self._touch_total = 0
        print(
            f"[TaskBStudent] actor_dim={self._actor_dim}, critic_dim={self._critic_dim} "
            f"(+{self._critic_extra_dim} priv: robot/ee/r_vel/contact/min_dist/scored/trash) "
            f"guide_w_prog={self._w_dense_dist} guide=min_3d_progress+milestones "
            f"milestones={guide_milestone_thresholds}->{guide_milestone_rewards} "
            f"guide_on={self._dense_enabled} "
            f"sparse_touch={self._sparse_touch_reward} "
            f"grasp_dist={self._grasp_dist_thresh} time_pen={self._time_penalty_per_env_step}/step "
            f"no_touch_timeout_s={self._no_touch_timeout_s} finished_reward={self._finished_reward} "
            f"env_step_dt={self._env_step_dt:.4f} inner_steps={self.inner_steps}",
            flush=True,
        )

    def _ensure_ee_body_idx(self) -> int:
        if self._ee_body_idx is not None:
            return self._ee_body_idx
        robot = self.env.unwrapped.scene["robot"]
        body_ids, found_names = robot.find_bodies(self._ee_body_name)
        if len(body_ids) == 0:
            raise ValueError(f"Cannot find EE body '{self._ee_body_name}'.")
        self._ee_body_idx = int(body_ids[0])
        print(f"[TaskBStudent] ee body: {found_names[0]} (idx={self._ee_body_idx})", flush=True)
        return self._ee_body_idx

    def _ee_pos_w(self) -> torch.Tensor:
        ee_idx = self._ensure_ee_body_idx()
        robot = self.env.unwrapped.scene["robot"]
        return robot.data.body_pos_w[:, ee_idx, :3].to(device=self._device, dtype=torch.float32)

    def _object_root_pos_w(self) -> torch.Tensor:
        objs = []
        for obj_idx in range(1, TASK_B_NUM_OBJECTS + 1):
            obj = self.env.unwrapped.scene[f"object_{obj_idx}"]
            objs.append(obj.data.root_pos_w[:, :3])
        return torch.stack(objs, dim=1).to(device=self._device, dtype=torch.float32)

    def _nearest_unscored_trash_metrics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return min XY dist to nearest not-yet-scored trash, argmin idx, 3D dist, all XY dists."""
        ee_pos = self._ee_pos_w()
        obj_pos = self._object_root_pos_w()
        ee_xy = ee_pos[:, :2].unsqueeze(1)
        obj_xy = obj_pos[:, :, :2]
        diff_xy = obj_xy - ee_xy
        dist_xy = torch.linalg.norm(diff_xy, dim=-1)
        dist_xy_masked = dist_xy.masked_fill(self._scored, float("inf"))
        min_xy_dist, nearest_idx = dist_xy_masked.min(dim=1)

        nearest_obj = obj_pos.gather(
            1, nearest_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        dist_3d = torch.linalg.norm(nearest_obj - ee_pos, dim=-1)
        dist_3d = torch.where(torch.isfinite(min_xy_dist), dist_3d, torch.full_like(dist_3d, float("inf")))
        return min_xy_dist, nearest_idx, dist_3d, dist_xy

    def _camera_depth_map(self, cam_name: str) -> torch.Tensor | None:
        if not self._visible_check_depth:
            return None
        try:
            cam = self.env.unwrapped.scene[cam_name]
            out = cam.data.output
            if out is None or "depth" not in out:
                return None
            depth = out["depth"].to(device=self._device, dtype=torch.float32)
            if depth.ndim == 4 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            elif depth.ndim == 4 and depth.shape[1] == 1:
                depth = depth[:, 0]
            return depth
        except Exception:
            return None

    def _objects_in_camera_frustum(self, obj_pos_w: torch.Tensor, cam_name: str) -> torch.Tensor:
        """Fast frustum test: in front of cam and inside image bounds (no depth read)."""
        try:
            cam = self.env.unwrapped.scene[cam_name]
            cam_pos = cam.data.pos_w[:, :3].to(device=self._device, dtype=torch.float32)
            cam_quat = cam.data.quat_w_world.to(device=self._device, dtype=torch.float32)
            intrinsics = cam.data.intrinsic_matrices.to(device=self._device, dtype=torch.float32)
            height, width = cam.data.image_shape
        except Exception:
            return torch.zeros(obj_pos_w.shape[:2], device=self._device, dtype=torch.bool)

        batch, num_obj, _ = obj_pos_w.shape
        obj_flat = obj_pos_w.reshape(batch * num_obj, 3)
        cam_pos_rep = cam_pos.unsqueeze(1).expand(batch, num_obj, 3).reshape(batch * num_obj, 3)
        cam_quat_rep = cam_quat.unsqueeze(1).expand(batch, num_obj, 4).reshape(batch * num_obj, 4)
        obj_cam, _ = math_utils.subtract_frame_transforms(cam_pos_rep, cam_quat_rep, obj_flat)
        obj_cam = obj_cam.reshape(batch, num_obj, 3)
        proj = math_utils.project_points(obj_cam, intrinsics)
        u, v, obj_depth = proj[..., 0], proj[..., 1], proj[..., 2]
        in_front = obj_depth > 0.05
        in_bounds = (u >= 0.0) & (u <= float(width - 1)) & (v >= 0.0) & (v <= float(height - 1))
        visible = in_front & in_bounds

        if self._visible_check_depth:
            depth_map = self._camera_depth_map(cam_name)
            if depth_map is not None:
                ui = u.round().long().clamp(0, width - 1)
                vi = v.round().long().clamp(0, height - 1)
                env_ids = torch.arange(batch, device=self._device).unsqueeze(1).expand(batch, num_obj)
                sampled = depth_map[env_ids, vi, ui]
                sampled = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
                depth_ok = (sampled > 0.05) & (sampled + self._visible_depth_tol >= obj_depth)
                visible = visible & depth_ok
        return visible

    def _visible_unscored_mask(self, obj_pos_w: torch.Tensor) -> torch.Tensor:
        """Unscored trash visible in head or ee camera frustum (optional depth occlusion)."""
        visible = self._objects_in_camera_frustum(obj_pos_w, "head_camera")
        visible = visible | self._objects_in_camera_frustum(obj_pos_w, "ee_camera")
        return visible & (~self._scored)

    def _nearest_visible_unscored_3d(
        self, dist_3d_all: torch.Tensor, visible_unscored: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Min EE-3D among visible unscored trash (logging / critic-style oracle)."""
        masked = dist_3d_all.masked_fill(~visible_unscored, float("inf"))
        min_dist, _ = masked.min(dim=1)
        visible_count = visible_unscored.sum(dim=1).to(dtype=torch.float32)
        return min_dist, visible_count

    def _touch_score_sparse(
        self, dist_3d_all: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-time sparse reward when EE-object 3D distance <= grasp threshold (platform score)."""
        reached = dist_3d_all <= self._grasp_dist_thresh
        if active_mask is not None:
            reached = reached & active_mask.unsqueeze(-1)
        newly = reached & (~self._scored)
        self._scored |= reached
        count = newly.sum(dim=1).to(dtype=torch.float32)
        if bool(newly.any()):
            self._touch_total += int(newly.sum().item())
        return count * self._sparse_touch_reward, newly.any(dim=1)

    def _record_scored_at_termination(self, done_mask: torch.Tensor) -> None:
        """Accumulate scored-object counts for envs terminating this step.

        Called before any reset clears ``self._scored``. Each nav step the running
        mean is flushed into ``extras["log"]["Episode_Termination/scored_objects"]``.
        """
        if not bool(done_mask.any()):
            return
        self._scored_at_term_sum += float(self._scored[done_mask].sum(dim=1).sum().item())
        self._scored_at_term_count += int(done_mask.sum().item())

    def _update_no_touch_timer(self, newly_scored: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Return per-env truncated flags when sim time since last new touch exceeds threshold."""
        if self._no_touch_timeout_s <= 0.0:
            return torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        dt = torch.tensor(self._env_step_dt, device=self._device, dtype=torch.float32)
        active_f = active.to(dtype=self._time_since_last_touch_s.dtype)
        self._time_since_last_touch_s = torch.where(
            newly_scored & active,
            torch.zeros_like(self._time_since_last_touch_s),
            self._time_since_last_touch_s + dt * active_f,
        )
        return active & (self._time_since_last_touch_s >= self._no_touch_timeout_s)

    def _dense_visible_per_object_progress(
        self,
        dist_3d_all: torch.Tensor,
        visible_unscored: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense on visible unscored trash with per-object prev (no argmin jitter).

        Reward = ``w * sum_i max(0, prev_i - dist_i)`` over visible unscored objects.
        Each object keeps its own ``prev_i``; after touch only that object's prev clears.
        This matches "approach visible targets" without spiking when the nearest id switches.
        """
        valid = visible_unscored
        if active_mask is not None:
            valid = valid & active_mask.unsqueeze(-1)

        prev = self._prev_obj_dist_3d
        init = torch.isnan(prev) | ~valid
        delta = torch.clamp(prev - dist_3d_all, min=0.0)
        delta = torch.where(init | ~valid, torch.zeros_like(delta), delta)
        dense = self._w_dense_dist * delta.sum(dim=1)

        nan_fill = torch.full_like(self._prev_obj_dist_3d, float("nan"))
        self._prev_obj_dist_3d = torch.where(
            self._scored,
            nan_fill,
            torch.where(valid, dist_3d_all, nan_fill),
        )
        return dense

    def _guide_min_dist_reward(
        self,
        dist_3d_all: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse guidance from EE to nearest unscored object (no camera projection).

        - progress: ``w * max(0, prev_min - min_dist)``
        - milestones: one-time bonuses on first crossing distance thresholds
        """
        valid_obj = ~self._scored
        update_mask = torch.ones(self.num_envs, device=self._device, dtype=torch.bool)
        if active_mask is not None:
            valid_obj = valid_obj & active_mask.unsqueeze(-1)
            update_mask = active_mask

        masked = dist_3d_all.masked_fill(~valid_obj, float("inf"))
        min_dist, _ = masked.min(dim=1)
        has_target = torch.isfinite(min_dist)

        progress = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        prev = self._prev_min_guide_dist
        valid_prog = has_target & torch.isfinite(prev)
        progress = torch.where(
            valid_prog,
            self._w_dense_dist * torch.clamp(prev - min_dist, min=0.0),
            progress,
        )

        crossed = (
            has_target.unsqueeze(-1)
            & (~self._guide_milestone_hit)
            & (min_dist.unsqueeze(-1) <= self._guide_thresholds.unsqueeze(0))
        )
        milestone = (crossed.to(dtype=torch.float32) * self._guide_rewards.unsqueeze(0)).sum(dim=1)

        new_hit = self._guide_milestone_hit | crossed
        self._guide_milestone_hit = torch.where(
            update_mask.unsqueeze(-1),
            new_hit,
            self._guide_milestone_hit,
        )

        new_prev = torch.where(has_target, min_dist, torch.full_like(min_dist, float("nan")))
        self._prev_min_guide_dist = torch.where(update_mask, new_prev, self._prev_min_guide_dist)
        return progress, milestone

    def _env_origins(self) -> torch.Tensor:
        origins = self.env.unwrapped.scene.env_origins.to(device=self._device, dtype=torch.float32)
        return origins[:, :3]

    def _world_to_env_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World positions -> per-env local frame (world - env_origin)."""
        origins = self._env_origins()
        if pos_w.ndim == 2:
            return pos_w - origins
        return pos_w - origins.unsqueeze(1)

    def _build_env_action(self, ll_action_train: torch.Tensor) -> torch.Tensor:
        """Legs from the low-level policy + arm frozen at the DETECT hold pose.

        The base env action layout is [leg(12), arm(8)]. The locomotion policy only
        outputs leg targets; we fill the arm slots with ``DETECT_HOLD_ARM_ACTION`` so
        the action manager's PD target is ``default + 0.5 * DETECT_HOLD_ARM_ACTION``
        every sim step — identical to the squat-flat training. Without this the arm
        slots are 0 and the arm drifts back to its USD default instead of locking
        to the DETECT pose.
        """
        batch = ll_action_train.shape[0]
        action = torch.zeros(
            batch, _LEG_DIM + _TASK_B_LOW_CLEAR_ARM_ACTION.shape[0], device=self._device, dtype=torch.float32
        )
        action[:, :_LEG_DIM] = ll_action_train * self._t2e
        action[:, _LEG_DIM:] = _TASK_B_LOW_CLEAR_ARM_ACTION.to(self._device).expand(batch, -1)
        return action

    def _robot_pose_local(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = self.env.unwrapped.scene["robot"]
        pos_w = robot.data.root_pos_w.to(device=self._device, dtype=torch.float32)
        pos_local = self._world_to_env_local(pos_w[:, :3])
        quat = robot.data.root_quat_w.to(device=self._device, dtype=torch.float32)
        yaw = self._yaw_from_quat_wxyz(quat)
        return pos_local[:, 0:1], pos_local[:, 1:2], pos_local[:, 2:3], yaw

    def _object_root_pos_local(self) -> torch.Tensor:
        return self._world_to_env_local(self._object_root_pos_w())

    def _contact_on(self) -> torch.Tensor:
        cf = self.env.unwrapped.scene["contact_sensor"].data.net_forces_w
        return (cf.norm(dim=-1).max(dim=1).values > 2.0).to(dtype=torch.float32).unsqueeze(-1)

    def _build_critic_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Asymmetric critic obs aligned with TaskDStudentEnv: policy || priv extras."""
        robot = self.env.unwrapped.scene["robot"]
        r_vel = robot.data.root_lin_vel_w.to(device=self._device, dtype=torch.float32)[:, :2]
        rx, ry, rz, robot_yaw = self._robot_pose_local()
        ee_local = self._world_to_env_local(self._ee_pos_w())
        ee_x, ee_y, ee_z = ee_local[:, 0:1], ee_local[:, 1:2], ee_local[:, 2:3]
        min_xy, _, min_3d, _ = self._nearest_unscored_trash_metrics()
        min_xy = torch.where(torch.isfinite(min_xy), min_xy, torch.zeros_like(min_xy)).unsqueeze(-1)
        min_3d = torch.where(torch.isfinite(min_3d), min_3d, torch.zeros_like(min_3d)).unsqueeze(-1)
        scored_mask = self._scored.to(dtype=torch.float32)
        obj_local = self._object_root_pos_local().reshape(self.num_envs, -1)
        priv = torch.cat(
            [
                rx,
                ry,
                rz,
                robot_yaw,
                ee_x,
                ee_y,
                ee_z,
                r_vel,
                self._contact_on(),
                min_xy,
                min_3d,
                scored_mask,
                obj_local,
            ],
            dim=-1,
        )
        return torch.cat([actor_obs, priv], dim=-1)

    def _obs_dict(self, env_obs: dict):
        actor = self._build_actor_obs(env_obs)
        return {"policy": actor, "critic": self._build_critic_obs(actor)}

    def _partial_reset_envs(self, done_mask: torch.Tensor) -> None:
        """Reset base sim + wrapper state for envs finished by wrapper-level truncation."""
        if not bool(done_mask.any()):
            return
        env_ids = done_mask.nonzero(as_tuple=False).flatten()
        base = self.env.unwrapped
        # _reset_idx -> event_manager -> reset_root_state_at_env_origin needs a tensor
        # (PhysX set_root_transforms casts indices via .to(int32)); a python list crashes.
        base._reset_idx(env_ids)
        if base.sim.has_rtx_sensors() and int(getattr(base.cfg, "num_rerenders_on_reset", 0)) > 0:
            for _ in range(int(base.cfg.num_rerenders_on_reset)):
                base.sim.render()
        self._prev_obj_dist_3d[env_ids] = float("nan")
        self._prev_min_guide_dist[env_ids] = float("nan")
        self._guide_milestone_hit[env_ids] = False
        self._scored[env_ids] = False
        self._time_since_last_touch_s[env_ids] = 0.0
        self._episode_finished[env_ids] = False
        self._current_obs = base.observation_manager.compute(update_history=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self._prev_obj_dist_3d.fill_(float("nan"))
        self._prev_min_guide_dist.fill_(float("nan"))
        self._guide_milestone_hit.fill_(False)
        self._scored.fill_(False)
        self._time_since_last_touch_s.zero_()
        self._episode_finished.zero_()
        self._nav_step_count = 0
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_illegal = 0
        self._done_no_touch = 0
        self._done_finished = 0
        self._touch_total = 0
        self._scored_at_term_sum = 0.0
        self._scored_at_term_count = 0
        self._illegal_at_term_count = 0
        self._fall_at_term_count = 0
        self._timeout_at_term_count = 0
        self._no_touch_at_term_count = 0
        self._finished_at_term_count = 0
        self._logged_first_rollout = False
        return self._obs_dict(obs), info

    def step(self, nav_action: torch.Tensor):
        if not isinstance(nav_action, torch.Tensor):
            nav_action = torch.as_tensor(nav_action, dtype=torch.float32)
        nav_action = nav_action.to(self._device, dtype=torch.float32)
        if nav_action.ndim == 1:
            nav_action = nav_action.unsqueeze(0)

        executed_vel = self._nav_action_to_vel_cmd(nav_action)

        self._nav_step_count += 1
        if not self._logged_first_rollout:
            print("[TaskBStudent] rollout started (first nav step).", flush=True)
            self._logged_first_rollout = True

        total_reward = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_dense = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_guide_prog = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_guide_mile = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_sparse = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_finished = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_time_pen = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_action_rate_pen = action_rate_pen.clone()
        total_reward += action_rate_pen
        terminated = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        last_info = {}
        episode_done = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)

        ee_pos = self._ee_pos_w()
        obj_pos = self._object_root_pos_w()
        diff_3d = obj_pos - ee_pos.unsqueeze(1)
        dist_3d_all = torch.linalg.norm(diff_3d, dim=-1)

        for _ in range(self.inner_steps):
            active = ~episode_done
            if not bool(active.any()):
                break

            ll_obs = self._build_ll_obs(self._current_obs, nav_action)
            with torch.inference_mode():
                ll_act = self.ll_policy(ll_obs)
            env_action = self._build_env_action(ll_act)
            obs, _base_rew, term, trunc, info = self.env.step(env_action)
            self._current_obs = obs
            self._on_after_physics_step()

            step_term = term if isinstance(term, torch.Tensor) else torch.as_tensor(term, device=self._device)
            step_trunc = trunc if isinstance(trunc, torch.Tensor) else torch.as_tensor(trunc, device=self._device)
            step_term_1d = step_term.squeeze(-1).bool() if step_term.ndim > 1 else step_term.bool()
            step_trunc_1d = step_trunc.squeeze(-1).bool() if step_trunc.ndim > 1 else step_trunc.bool()
            done_now = step_term_1d | step_trunc_1d
            episode_done |= done_now
            terminated |= step_term_1d
            truncated |= step_trunc_1d
            last_info = info

            ee_pos = self._ee_pos_w()
            obj_pos = self._object_root_pos_w()
            diff_3d = obj_pos - ee_pos.unsqueeze(1)
            dist_3d_all = torch.linalg.norm(diff_3d, dim=-1)
            active_after_base_done = active & (~done_now)
            sparse, newly_scored = self._touch_score_sparse(
                dist_3d_all, active_mask=active_after_base_done
            )

            if self._dense_enabled:
                guide_prog, guide_mile = self._guide_min_dist_reward(
                    dist_3d_all, active_mask=active_after_base_done
                )
                dense = guide_prog + guide_mile
            else:
                guide_prog = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
                guide_mile = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
                dense = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
            time_pen = torch.full_like(dense, -self._time_penalty_per_env_step)

            all_scored = self._scored.all(dim=1)
            finished_now = active_after_base_done & all_scored & (~self._episode_finished)
            finished_bonus = torch.zeros_like(dense)
            if bool(finished_now.any()):
                self._episode_finished |= finished_now
                finished_bonus = finished_now.to(dtype=dense.dtype) * self._finished_reward
                episode_done |= finished_now
                terminated |= finished_now
                self._done_finished += int(finished_now.sum().item())
                self._record_scored_at_termination(finished_now)
                self._finished_at_term_count += int(finished_now.sum().item())

            no_touch_done = self._update_no_touch_timer(
                newly_scored, active_after_base_done & (~finished_now)
            )
            if bool(no_touch_done.any()):
                episode_done |= no_touch_done
                truncated |= no_touch_done
                self._done_no_touch += int(no_touch_done.sum().item())
                self._record_scored_at_termination(no_touch_done)
                self._no_touch_at_term_count += int(no_touch_done.sum().item())

            alive_f = active_after_base_done.to(dtype=dense.dtype)
            step_rew = (dense + sparse + time_pen) * alive_f + finished_bonus
            total_reward += step_rew
            total_dense += dense * alive_f
            total_guide_prog += guide_prog * alive_f
            total_guide_mile += guide_mile * alive_f
            total_sparse += sparse * alive_f
            total_finished += finished_bonus * alive_f
            total_time_pen += time_pen * alive_f

            if bool(done_now.any()):
                self._record_scored_at_termination(done_now)
                fall_flags = self._termination_term_flags().get("fall")
                timeout_flags = done_now & step_trunc_1d
                if fall_flags is None:
                    fall_flags = torch.zeros_like(done_now)
                fall_flags = done_now & (~step_trunc_1d) & fall_flags
                illegal_flags = done_now & (~step_trunc_1d) & (~fall_flags)
                self._illegal_at_term_count += int(illegal_flags.sum().item())
                self._fall_at_term_count += int(fall_flags.sum().item())
                self._timeout_at_term_count += int(timeout_flags.sum().item())
                self._prev_obj_dist_3d = torch.where(
                    done_now.unsqueeze(-1),
                    torch.full_like(self._prev_obj_dist_3d, float("nan")),
                    self._prev_obj_dist_3d,
                )
                self._prev_min_guide_dist = torch.where(
                    done_now,
                    torch.full_like(self._prev_min_guide_dist, float("nan")),
                    self._prev_min_guide_dist,
                )
                self._guide_milestone_hit = torch.where(
                    done_now.unsqueeze(-1),
                    torch.zeros_like(self._guide_milestone_hit),
                    self._guide_milestone_hit,
                )
                self._scored = torch.where(
                    done_now.unsqueeze(-1),
                    torch.zeros_like(self._scored),
                    self._scored,
                )
                self._time_since_last_touch_s = torch.where(
                    done_now,
                    torch.zeros_like(self._time_since_last_touch_s),
                    self._time_since_last_touch_s,
                )
                self._episode_finished = torch.where(
                    done_now,
                    torch.zeros_like(self._episode_finished),
                    self._episode_finished,
                )
                self._done_total += int(done_now.sum().item())
                self._done_timeout += int((done_now & step_trunc_1d).sum().item())
                if fall_flags is not None:
                    self._done_fall += int((done_now & fall_flags).sum().item())
                illegal = done_now
                if fall_flags is not None:
                    illegal = illegal & (~fall_flags)
                illegal = illegal & (~step_trunc_1d)
                self._done_illegal += int(illegal.sum().item())

        if self._nav_log_interval > 0 and self._nav_step_count % self._nav_log_interval == 0:
            min_xy, _, min_3d, _ = self._nearest_unscored_trash_metrics()
            scored = self._scored.sum(dim=1).float().mean().item()
            vel_cmd = self._nav_action_to_vel_cmd(nav_action)
            rx, ry, rz = self._robot_pose()
            base_denom = max(1, self._done_total)
            term_denom = max(1, self._scored_at_term_count)
            print(
                f"[TaskBStudent] nav={self._nav_step_count:5d} "
                f"rew={total_reward.mean().item():+.4f} "
                f"[guide/prog/mile/sparse/finished/time/rate]={total_dense.mean().item():+.3f}/"
                f"{total_guide_prog.mean().item():+.3f}/{total_guide_mile.mean().item():+.3f}/"
                f"{total_sparse.mean().item():+.3f}/{total_finished.mean().item():+.3f}/"
                f"{total_time_pen.mean().item():+.3f}/{total_action_rate_pen.mean().item():+.3f} "
                f"min_3d_oracle={min_3d[torch.isfinite(min_3d)].min().item() if torch.isfinite(min_3d).any() else float('nan'):.2f} "
                f"min_xy_oracle={min_xy[torch.isfinite(min_xy)].mean().item() if torch.isfinite(min_xy).any() else float('nan'):.2f} "
                f"scored_mean={scored:.2f}/18 touches={self._touch_total} "
                f"dones={int((terminated | truncated).sum())}/{self.num_envs} "
                f"pos0=({rx[0,0].item():+.1f},{ry[0,0].item():+.1f},{rz[0,0].item():+.2f}) "
                f"cmd0=({vel_cmd[0,0].item():+.2f},{vel_cmd[0,1].item():+.2f},{vel_cmd[0,2].item():+.2f}) "
                f"done[illegal/fall/timeout/no_touch/finished]={self._done_illegal}/{self._done_fall}/"
                f"{self._done_timeout}/{self._done_no_touch}/{self._done_finished} "
                f"ratio_base={self._done_illegal/base_denom:.2f}/{self._done_fall/base_denom:.2f}/"
                f"{self._done_timeout/base_denom:.2f} "
                f"ratio_wrap={self._done_no_touch/term_denom:.2f}/{self._done_finished/term_denom:.2f}",
                flush=True,
            )

        # Log mean scored objects over envs that terminated this nav step.
        # Inject termination metrics into extras["log"] so rsl_rl prints them in
        # the training table. scored_objects = mean #scored at termination;
        # no_touch / finished = fraction of terminations of each wrapper-level
        # type (these aren't in the MDP termination manager, unlike fall/time_out).
        if self._scored_at_term_count > 0:
            total_term = float(self._scored_at_term_count)
            mean_scored = self._scored_at_term_sum / total_term
            last_info = dict(last_info) if last_info is not None else {}
            log_dict = last_info.setdefault("log", {})
            # Keep all Episode_Termination/* ratios on one denominator so metrics
            # are directly comparable in the training table.
            log_dict["Episode_Termination/illegal_contact"] = self._illegal_at_term_count / total_term
            log_dict["Episode_Termination/fall"] = self._fall_at_term_count / total_term
            log_dict["Episode_Termination/time_out"] = self._timeout_at_term_count / total_term
            log_dict["Episode_Termination/scored_objects"] = mean_scored
            log_dict["Episode_Termination/no_touch"] = self._no_touch_at_term_count / total_term
            log_dict["Episode_Termination/finished"] = self._finished_at_term_count / total_term
            self._scored_at_term_sum = 0.0
            self._scored_at_term_count = 0
            self._illegal_at_term_count = 0
            self._fall_at_term_count = 0
            self._timeout_at_term_count = 0
            self._no_touch_at_term_count = 0
            self._finished_at_term_count = 0

        self._last_vel_cmd = executed_vel

        # Envs that terminated at any inner step were auto-reset by the base env.
        # Remaining inner steps may still advance them, so re-reset once at the end
        # so the next nav step starts from a clean initial state.
        if bool(episode_done.any()):
            self._partial_reset_envs(episode_done)

        return self._obs_dict(self._current_obs), total_reward, terminated, truncated, last_info
