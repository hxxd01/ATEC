"""Task B student wrapper: Task-D student network + Task-B touch-nav rewards."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .taskd_student_env import TaskDStudentEnv
from .taskd_teacher_env import _LIN_VEL_SLICE, _ANG_VEL_SLICE, _GRAVITY_SLICE

TASK_B_NUM_OBJECTS = 18
TASK_B_GRASP_DIST = 0.20


class TaskBStudentEnv(TaskDStudentEnv):
    """Nav student for Task B: dense EE-to-trash approach, sparse platform touch score, time penalty."""

    def __init__(
        self,
        env: gym.Env,
        ll_policy_path: str,
        device: str = "cuda",
        inner_steps: int = 25,
        vx_min: float = -4.0,
        vx_max: float = 4.0,
        vy_max: float = 2.0,
        wz_max: float = 1.0,
        image_h: int = 24,
        image_w: int = 32,
        image_hw: int | None = None,
        depth_max: float = 5.0,
        depth_only: bool = False,
        nav_log_interval: int = 10,
        w_dense_dist: float = 1.0,
        sparse_touch_reward: float = 10.0,
        grasp_dist_thresh: float = TASK_B_GRASP_DIST,
        time_penalty_per_env_step: float = 0.01,
        ee_body_name: str = "gripper_base",
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
        self._sparse_touch_reward = float(sparse_touch_reward)
        self._grasp_dist_thresh = float(grasp_dist_thresh)
        self._grasp_dist_sq = self._grasp_dist_thresh * self._grasp_dist_thresh
        self._time_penalty_per_env_step = float(time_penalty_per_env_step)
        self._ee_body_name = str(ee_body_name)

        # Privileged critic: robot(3) + nearest trash body rel(2) + min xy proj dist(1) + scored frac(1)
        self._critic_extra_dim = 7
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
        self._prev_min_xy_dist = torch.full(
            (self.num_envs,), float("nan"), device=self._device, dtype=torch.float32
        )
        self._scored = torch.zeros(
            (self.num_envs, TASK_B_NUM_OBJECTS), device=self._device, dtype=torch.bool
        )
        self._logged_first_rollout = False
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_illegal = 0
        self._touch_total = 0
        print(
            f"[TaskBStudent] actor_dim={self._actor_dim}, critic_dim={self._critic_dim} "
            f"dense_w={self._w_dense_dist} sparse_touch={self._sparse_touch_reward} "
            f"grasp_dist={self._grasp_dist_thresh} time_pen={self._time_penalty_per_env_step}/step "
            f"inner_steps={self.inner_steps}",
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

    def _nearest_trash_metrics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return min XY proj dist, argmin idx, nearest 3D dist, nearest XY dist per env."""
        ee_pos = self._ee_pos_w()
        obj_pos = self._object_root_pos_w()
        ee_xy = ee_pos[:, :2].unsqueeze(1)
        obj_xy = obj_pos[:, :, :2]
        diff_xy = obj_xy - ee_xy
        dist_xy = torch.linalg.norm(diff_xy, dim=-1)
        min_xy_dist, nearest_idx = dist_xy.min(dim=1)

        nearest_obj = obj_pos.gather(
            1, nearest_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        dist_3d = torch.linalg.norm(nearest_obj - ee_pos, dim=-1)
        return min_xy_dist, nearest_idx, dist_3d, dist_xy

    def _touch_score_sparse(self, dist_3d_all: torch.Tensor) -> torch.Tensor:
        """One-time sparse reward when EE-object 3D distance <= grasp threshold (platform score)."""
        reached = dist_3d_all <= self._grasp_dist_thresh
        newly = reached & (~self._scored)
        self._scored |= reached
        count = newly.sum(dim=1).to(dtype=torch.float32)
        if bool(newly.any()):
            self._touch_total += int(newly.sum().item())
        return count * self._sparse_touch_reward

    def _dense_xy_progress(self, min_xy_dist: torch.Tensor) -> torch.Tensor:
        prev = self._prev_min_xy_dist
        init = torch.isnan(prev)
        delta = torch.where(init, torch.zeros_like(min_xy_dist), prev - min_xy_dist)
        dense = self._w_dense_dist * delta
        self._prev_min_xy_dist = min_xy_dist.clone()
        return dense

    def _build_critic_obs(self, actor_obs: torch.Tensor):
        rx, ry, robot_yaw = self._robot_pose()
        min_xy_dist, nearest_idx, dist_3d, _ = self._nearest_trash_metrics()
        obj_pos = self._object_root_pos_w()
        nearest = obj_pos.gather(1, nearest_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        dx, dy = nearest[:, 0:1] - rx, nearest[:, 1:2] - ry
        cy, sy = torch.cos(robot_yaw), torch.sin(robot_yaw)
        rel_x = cy * dx + sy * dy
        rel_y = -sy * dx + cy * dy
        scored_frac = self._scored.sum(dim=1, keepdim=True).to(dtype=torch.float32) / float(
            TASK_B_NUM_OBJECTS
        )
        priv = torch.cat(
            [
                torch.cat([rx, ry, robot_yaw], dim=-1),
                rel_x,
                rel_y,
                min_xy_dist.unsqueeze(-1),
                scored_frac,
            ],
            dim=-1,
        )
        return torch.cat([actor_obs, priv], dim=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self._prev_min_xy_dist.fill_(float("nan"))
        self._scored.fill_(False)
        self._nav_step_count = 0
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_illegal = 0
        self._touch_total = 0
        self._logged_first_rollout = False
        return self._obs_dict(obs), info

    def step(self, nav_action: torch.Tensor):
        if not isinstance(nav_action, torch.Tensor):
            nav_action = torch.as_tensor(nav_action, dtype=torch.float32)
        nav_action = nav_action.to(self._device, dtype=torch.float32)
        if nav_action.ndim == 1:
            nav_action = nav_action.unsqueeze(0)

        self._nav_step_count += 1
        if not self._logged_first_rollout:
            print("[TaskBStudent] rollout started (first nav step).", flush=True)
            self._logged_first_rollout = True

        total_reward = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_dense = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_sparse = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_time_pen = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
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
            min_xy_dist, _, _, _ = self._nearest_trash_metrics()

            dense = self._dense_xy_progress(min_xy_dist)
            sparse = self._touch_score_sparse(dist_3d_all)
            time_pen = torch.full_like(dense, -self._time_penalty_per_env_step)

            alive_f = active.to(dtype=dense.dtype)
            step_rew = (dense + sparse + time_pen) * alive_f
            total_reward += step_rew
            total_dense += dense * alive_f
            total_sparse += sparse * alive_f
            total_time_pen += time_pen * alive_f

            if bool(done_now.any()):
                self._prev_min_xy_dist = torch.where(
                    done_now,
                    torch.full_like(self._prev_min_xy_dist, float("nan")),
                    self._prev_min_xy_dist,
                )
                self._scored = torch.where(
                    done_now.unsqueeze(-1),
                    torch.zeros_like(self._scored),
                    self._scored,
                )
                self._done_total += int(done_now.sum().item())
                self._done_timeout += int((done_now & step_trunc_1d).sum().item())
                fall_flags = self._termination_term_flags().get("fall")
                if fall_flags is not None:
                    self._done_fall += int((done_now & fall_flags).sum().item())
                illegal = done_now
                if fall_flags is not None:
                    illegal = illegal & (~fall_flags)
                illegal = illegal & (~step_trunc_1d)
                self._done_illegal += int(illegal.sum().item())

        if self._nav_log_interval > 0 and self._nav_step_count % self._nav_log_interval == 0:
            min_xy, _, min_3d, _ = self._nearest_trash_metrics()
            scored = self._scored.sum(dim=1).float().mean().item()
            vel_cmd = self._nav_action_to_vel_cmd(nav_action)
            rx, ry, rz = self._robot_pose()
            denom = max(1, self._done_total)
            print(
                f"[TaskBStudent] nav={self._nav_step_count:5d} "
                f"rew={total_reward.mean().item():+.4f} "
                f"[dense/sparse/time]={total_dense.mean().item():+.3f}/"
                f"{total_sparse.mean().item():+.3f}/{total_time_pen.mean().item():+.3f} "
                f"min_xy={min_xy.mean().item():.2f} min_3d={min_3d.min().item():.2f} "
                f"scored_mean={scored:.2f}/18 touches={self._touch_total} "
                f"dones={int((terminated | truncated).sum())}/{self.num_envs} "
                f"pos0=({rx[0,0].item():+.1f},{ry[0,0].item():+.1f},{rz[0,0].item():+.2f}) "
                f"cmd0=({vel_cmd[0,0].item():+.2f},{vel_cmd[0,1].item():+.2f},{vel_cmd[0,2].item():+.2f}) "
                f"done[illegal/fall/timeout]={self._done_illegal}/{self._done_fall}/"
                f"{self._done_timeout} ratio={self._done_illegal/denom:.2f}/"
                f"{self._done_fall/denom:.2f}/{self._done_timeout/denom:.2f}",
                flush=True,
            )

        return self._obs_dict(self._current_obs), total_reward, terminated, truncated, last_info
