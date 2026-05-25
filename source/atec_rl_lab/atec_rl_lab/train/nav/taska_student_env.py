"""Task A student wrapper: camera+proprio observations, no lidar."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .taskd_teacher_env import (
    TaskDTeacherEnv,
    _LIN_VEL_SLICE,
    _ANG_VEL_SLICE,
    _GRAVITY_SLICE,
)


class TaskAStudentEnv(TaskDTeacherEnv):
    def __init__(
        self,
        env: gym.Env,
        ll_policy_path: str,
        device: str = "cuda",
        inner_steps: int = 25,
        vx_min: float = -2.0,
        vx_max: float = 2.0,
        vy_max: float = 0.4,
        wz_max: float = 0.4,
        image_hw: int = 64,
        depth_max: float = 5.0,
        time_penalty_per_env_step: float = 0.002,
    ):
        super().__init__(
            env=env,
            ll_policy_path=ll_policy_path,
            device=device,
            inner_steps=inner_steps,
            lidar_bins=0,
            vx_min=vx_min,
            vx_max=vx_max,
            vy_max=vy_max,
            wz_max=wz_max,
        )
        self._image_hw = int(image_hw)
        self._depth_max = float(depth_max)
        self._time_penalty_per_env_step = float(time_penalty_per_env_step)

        self._student_img_flat = 8 * self._image_hw * self._image_hw
        self._actor_dim = self._student_img_flat + 9
        self._critic_dim = self._actor_dim
        self.observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._actor_dim,), dtype=np.float32),
                "critic": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._critic_dim,), dtype=np.float32),
            }
        )
        self._nav_step_count = 0
        self._logged_first_rollout = False
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_reach = 0
        self._done_illegal = 0
        print(
            f"[TaskAStudent] actor_dim={self._actor_dim}, critic_dim={self._critic_dim} "
            f"time_penalty={self._time_penalty_per_env_step:.4f}/env-step "
            f"inner_steps={self.inner_steps} "
            f"vel_limits(vx=[{self._vx_min:.2f},{self._vx_max:.2f}], vy=+/-{self._vy_max:.2f}, wz=+/-{self._wz_max:.2f})",
            flush=True,
        )

    def _prep_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] != self._image_hw or x.shape[-2] != self._image_hw:
            x = F.interpolate(x, size=(self._image_hw, self._image_hw), mode="bilinear", align_corners=False)
        return x

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x[..., 0]
        x = torch.nan_to_num(x, nan=self._depth_max, posinf=self._depth_max, neginf=0.0)
        x = torch.clamp(x, 0.05, self._depth_max)
        x = torch.log1p(x) / np.log1p(self._depth_max)
        x = x.unsqueeze(1)
        if x.shape[-1] != self._image_hw or x.shape[-2] != self._image_hw:
            x = F.interpolate(x, size=(self._image_hw, self._image_hw), mode="bilinear", align_corners=False)
        return x

    def _camera_4ch(self, cam_name: str, batch: int) -> torch.Tensor:
        try:
            cam = self.env.unwrapped.scene[cam_name]
            out = cam.data.output
            rgb = self._prep_rgb(out["rgb"].to(device=self._device))
            depth = self._prep_depth(out["depth"].to(device=self._device))
            return torch.cat([rgb, depth], dim=1)
        except Exception:
            return torch.zeros(batch, 4, self._image_hw, self._image_hw, device=self._device, dtype=torch.float32)

    def _build_actor_obs(self, env_obs: dict):
        proprio = env_obs["proprio"].to(self._device, dtype=torch.float32)
        batch = proprio.shape[0]
        lin_vel = proprio[:, _LIN_VEL_SLICE]
        ang_vel = proprio[:, _ANG_VEL_SLICE]
        gravity = proprio[:, _GRAVITY_SLICE]
        proprio_feat = torch.cat([lin_vel, ang_vel, gravity], dim=-1)
        head = self._camera_4ch("head_camera", batch).reshape(batch, -1)
        ee = self._camera_4ch("ee_camera", batch).reshape(batch, -1)
        return torch.cat([head, ee, proprio_feat], dim=-1)

    def _build_critic_obs(self, actor_obs: torch.Tensor):
        return actor_obs

    def _obs_dict(self, env_obs: dict):
        policy = self._build_actor_obs(env_obs)
        return {"policy": policy, "critic": policy}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_reach = 0
        self._done_illegal = 0
        return self._obs_dict(obs), info

    def _classify_done(self, step_term_1d: torch.Tensor, step_trunc_1d: torch.Tensor, rx, rz) -> None:
        if not bool((step_term_1d | step_trunc_1d).any()):
            return
        self._done_total += int((step_term_1d | step_trunc_1d).sum().item())
        self._done_timeout += int(step_trunc_1d.sum().item())
        reach_now = step_term_1d & (rx.squeeze(-1) > 140.0)
        fall_now = step_term_1d & (rz.squeeze(-1) < 0.35) & (~reach_now)
        illegal_now = step_term_1d & (~reach_now) & (~fall_now)
        self._done_reach += int(reach_now.sum().item())
        self._done_fall += int(fall_now.sum().item())
        self._done_illegal += int(illegal_now.sum().item())

    def step(self, nav_action: torch.Tensor):
        if not isinstance(nav_action, torch.Tensor):
            nav_action = torch.as_tensor(nav_action, dtype=torch.float32)
        nav_action = nav_action.to(self._device, dtype=torch.float32)
        if nav_action.ndim == 1:
            nav_action = nav_action.unsqueeze(0)

        self._nav_step_count += 1
        if not self._logged_first_rollout:
            print("[TaskAStudent] rollout started (first nav step).", flush=True)
            self._logged_first_rollout = True

        total_reward = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        terminated = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        last_info = {}
        progress_sum = 0.0

        for _ in range(self.inner_steps):
            ll_obs = self._build_ll_obs(self._current_obs, nav_action)
            with torch.inference_mode():
                ll_act = self.ll_policy(ll_obs)
            env_action = self._build_env_action(ll_act)
            obs, base_rew, term, trunc, info = self.env.step(env_action)
            self._current_obs = obs

            step_term = term if isinstance(term, torch.Tensor) else torch.as_tensor(term, device=self._device)
            step_trunc = trunc if isinstance(trunc, torch.Tensor) else torch.as_tensor(trunc, device=self._device)
            step_term_1d = step_term.squeeze(-1).bool() if step_term.ndim > 1 else step_term.bool()
            step_trunc_1d = step_trunc.squeeze(-1).bool() if step_trunc.ndim > 1 else step_trunc.bool()
            terminated |= step_term_1d
            truncated |= step_trunc_1d
            last_info = info

            robot = self.env.unwrapped.scene["robot"]
            rx = robot.data.root_pos_w.to(device=self._device, dtype=torch.float32)[:, 0:1]
            rz = robot.data.root_pos_w.to(device=self._device, dtype=torch.float32)[:, 2:3]
            self._classify_done(step_term_1d, step_trunc_1d, rx, rz)

            if isinstance(base_rew, torch.Tensor):
                base_rew_1d = base_rew.to(self._device, dtype=torch.float32)
            else:
                base_rew_1d = torch.as_tensor(base_rew, device=self._device, dtype=torch.float32)
            if base_rew_1d.ndim > 1:
                base_rew_1d = base_rew_1d.squeeze(-1)

            total_reward += base_rew_1d - self._time_penalty_per_env_step
            progress_sum += float(base_rew_1d.mean().item())

        done = terminated | truncated
        if self._nav_step_count <= 20 or self._nav_step_count % 50 == 0:
            vel_cmd = self._nav_action_to_vel_cmd(nav_action)
            ep_len_nav = int(self.episode_length_buf.float().mean().item())
            denom = max(1, self._done_total)
            print(
                f"[TaskAStudent] nav={self._nav_step_count:5d} "
                f"rew_mean={total_reward.mean().item():+.4f} prog_mean={progress_sum / self.inner_steps:+.4f} "
                f"dones={int(done.sum())}/{self.num_envs} "
                f"(term={int(terminated.sum())} trunc={int(truncated.sum())}) "
                f"ep_len_nav~{ep_len_nav} "
                f"x=[{rx.min().item():.1f},{rx.max().item():.1f}] z=[{rz.min().item():.2f},{rz.max().item():.2f}] "
                f"cmd0=({vel_cmd[0,0].item():+.2f},{vel_cmd[0,1].item():+.2f},{vel_cmd[0,2].item():+.2f}) "
                f"done[illegal/fall/timeout/reach]={self._done_illegal}/{self._done_fall}/"
                f"{self._done_timeout}/{self._done_reach} "
                f"ratio={self._done_illegal/denom:.2f}/{self._done_fall/denom:.2f}/"
                f"{self._done_timeout/denom:.2f}/{self._done_reach/denom:.2f}",
                flush=True,
            )

        return self._obs_dict(self._current_obs), total_reward, terminated, truncated, last_info
