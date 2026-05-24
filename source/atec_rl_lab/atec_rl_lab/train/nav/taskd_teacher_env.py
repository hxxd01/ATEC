"""Task D teacher wrapper with asymmetric actor/critic observations."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

_LIN_VEL_SLICE = slice(0, 3)
_ANG_VEL_SLICE = slice(3, 6)
_GRAVITY_SLICE = slice(9, 12)
_JPOS_SLICE = slice(12, 32)
_JVEL_SLICE = slice(32, 52)
_ACT_SLICE = slice(52, 72)
_LEG_DIM = 12
_TOTAL_ACTION_DIM = 20
_LIDAR_H = 360

_TRAIN_TO_ENV = torch.tensor([0.25, 0.5, 0.5] * 4, dtype=torch.float32)
_ENV_TO_TRAIN = torch.tensor([4.0, 2.0, 2.0] * 4, dtype=torch.float32)


class TaskDTeacherEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        ll_policy_path: str,
        device: str = "cuda",
        inner_steps: int = 25,
        lidar_bins: int = 36,
        vx_min: float = -2.0,
        vx_max: float = 2.0,
        vy_max: float = 1.2,
        wz_max: float = 0.6,
        curriculum_warmup_nav_steps: int = 1500,
        curriculum_mid_nav_steps: int = 3500,
    ):
        super().__init__(env)
        self._device = device
        self.inner_steps = int(inner_steps)
        self._lidar_bins = int(lidar_bins)
        self._vx_min = float(vx_min)
        self._vx_max = float(vx_max)
        self._vy_max = float(vy_max)
        self._wz_max = float(wz_max)
        self._curriculum_warmup_nav_steps = int(curriculum_warmup_nav_steps)
        self._curriculum_mid_nav_steps = int(curriculum_mid_nav_steps)
        self._nav_step_count = 0

        self.ll_policy = torch.jit.load(ll_policy_path, map_location=device)
        self.ll_policy.eval()
        self._t2e = _TRAIN_TO_ENV.to(device)
        self._e2t = _ENV_TO_TRAIN.to(device)

        # Keep stage script aligned with demo/solution.py.
        self.stage_specs = [
            dict(name="retreat", axis="x", sign=-0.7, dist=1.0, push=False, sparse_bonus=0.6),
            dict(name="sidestep_left", axis="y", sign=+1, dist=2.1, push=False, sparse_bonus=0.8),
            dict(
                name="advance",
                axis="x",
                sign=+0.7,
                dist=1.0,
                push=False,
                sparse_bonus=0.8,
                match_box_x_tol=0.15,
            ),
            dict(
                name="sidestep_right",
                axis="y",
                sign=-1,
                dist=2.0,
                push=True,
                sparse_bonus=1.2,
            ),
            dict(name="retreat2", axis="x", sign=-1, dist=1.0, push=False, sparse_bonus=0.8),
            dict(name="sidestep_right2", axis="y", sign=-1, dist=0.6, push=False, sparse_bonus=0.8),
            dict(
                name="advance2",
                axis="x",
                sign=+1,
                dist=4.0,
                push=True,
                sparse_bonus=1.6,
                box_x_slow_start=-1.7,
                box_x_stop=-1.25,
                stop_tol=0.08,
                wait_after_box_drop_z=-0.35,
                wait_after_box_drop_s=1.0,
            ),
            dict(name="final", axis="x", sign=+1, dist=3.0, push=False, sparse_bonus=1.0),
        ]
        self._num_stages = len(self.stage_specs)
        self._stage_names = [spec["name"] for spec in self.stage_specs]
        self._stage_axis_is_x = torch.tensor(
            [spec["axis"] == "x" for spec in self.stage_specs], device=self._device, dtype=torch.bool
        )
        self._stage_push = torch.tensor(
            [bool(spec["push"]) for spec in self.stage_specs], device=self._device, dtype=torch.bool
        )
        self._stage_sign = torch.tensor([float(spec["sign"]) for spec in self.stage_specs], device=self._device, dtype=torch.float32)
        self._stage_dist = torch.tensor([float(spec["dist"]) for spec in self.stage_specs], device=self._device, dtype=torch.float32)

        self._stage_idx_buf: torch.Tensor | None = None
        self._active_stage_count_buf: torch.Tensor | None = None
        self._curriculum_level_buf: torch.Tensor | None = None
        self._stage_origin_buf: torch.Tensor | None = None
        self._stage_progress_buf: torch.Tensor | None = None
        self._nav_step_count_buf: torch.Tensor | None = None
        self._step_wait_counter: torch.Tensor | None = None
        self._step_wait_armed: torch.Tensor | None = None
        self._stage_complete_bonus = 0.5
        self._current_obs: dict | None = None
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_x_reached = 0

        self._prev_robot_x = None
        self._prev_robot_y = None
        self._prev_box_x = None
        self._prev_box_y = None

        # actor: lidar + full proprio(9) + robot abs pose(3) + box abs pose(3) + rel body(3) + stage onehot + progress(1)
        self._actor_dim = self._lidar_bins + 9 + 3 + 3 + 3 + self._num_stages + 1
        # critic: actor + privileged world velocity terms + rel world + contact flag
        self._critic_dim = self._actor_dim + 2 + 2 + 2 + 1

        self.observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._actor_dim,), dtype=np.float32),
                "critic": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._critic_dim,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        env_dt = float(getattr(self.env.unwrapped, "step_dt", 0.02))
        nav_dt = self.inner_steps * env_dt
        print(
            f"[TaskDTeacher] nav_dt={nav_dt:.3f}s ({1.0/nav_dt:.2f}Hz), "
            f"vx=[{self._vx_min:.1f},{self._vx_max:.1f}] curriculum=[{self._curriculum_warmup_nav_steps},"
            f"{self._curriculum_mid_nav_steps}]",
            flush=True,
        )

    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_episode_length(self) -> int:
        base_max = getattr(self.env.unwrapped, "max_episode_length", 60000)
        return max(1, int(base_max) // self.inner_steps)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.unwrapped.episode_length_buf // self.inner_steps

    def _yaw_from_quat_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).unsqueeze(-1)

    def _robot_pose(self):
        robot = self.env.unwrapped.scene["robot"]
        pos = robot.data.root_pos_w.to(device=self._device, dtype=torch.float32)
        quat = robot.data.root_quat_w.to(device=self._device, dtype=torch.float32)
        return pos[:, 0:1], pos[:, 1:2], self._yaw_from_quat_wxyz(quat)

    def _box_pose(self):
        box = self.env.unwrapped.scene["box"]
        pos = box.data.root_pos_w.to(device=self._device, dtype=torch.float32)
        quat = box.data.root_quat_w.to(device=self._device, dtype=torch.float32)
        return pos[:, 0:1], pos[:, 1:2], self._yaw_from_quat_wxyz(quat)

    def _lidar_compact(self, extero: torch.Tensor | None, batch: int) -> torch.Tensor:
        if extero is None or extero.shape[-1] <= 0:
            return torch.zeros(batch, self._lidar_bins, device=self._device, dtype=torch.float32)
        x = extero.to(device=self._device, dtype=torch.float32)
        n = x.shape[-1]
        if n % _LIDAR_H == 0:
            x = x.view(batch, n // _LIDAR_H, _LIDAR_H).abs().max(dim=1).values
        else:
            x = x.abs()
        step = max(1, x.shape[-1] // self._lidar_bins)
        usable = min(x.shape[-1], step * self._lidar_bins)
        x = x[:, :usable]
        if usable < step * self._lidar_bins:
            pad = torch.zeros(batch, step * self._lidar_bins - usable, device=self._device, dtype=torch.float32)
            x = torch.cat([x, pad], dim=-1)
        return x.reshape(batch, self._lidar_bins, step).max(dim=-1).values

    def _nav_action_to_vel_cmd(self, nav_action: torch.Tensor) -> torch.Tensor:
        a = nav_action.clamp(-1.0, 1.0)
        vx = (a[:, 0] + 1.0) * 0.5 * (self._vx_max - self._vx_min) + self._vx_min
        return torch.stack([vx, a[:, 1] * self._vy_max, a[:, 2] * self._wz_max], dim=-1)

    def _build_ll_obs(self, env_obs: dict, nav_action: torch.Tensor) -> torch.Tensor:
        proprio = env_obs["proprio"].to(self._device, dtype=torch.float32)
        vel_cmd = self._nav_action_to_vel_cmd(nav_action)
        ang_vel = proprio[:, _ANG_VEL_SLICE]
        gravity = proprio[:, _GRAVITY_SLICE]
        jpos_leg = proprio[:, _JPOS_SLICE][:, :_LEG_DIM]
        jvel_leg = proprio[:, _JVEL_SLICE][:, :_LEG_DIM]
        act_leg_tr = proprio[:, _ACT_SLICE][:, :_LEG_DIM] * self._e2t
        return torch.cat([ang_vel * 0.25, gravity, vel_cmd, jpos_leg, jvel_leg * 0.05, act_leg_tr], dim=-1)

    def _build_env_action(self, ll_action_train: torch.Tensor) -> torch.Tensor:
        batch = ll_action_train.shape[0]
        action = torch.zeros(batch, _TOTAL_ACTION_DIM, device=self._device, dtype=torch.float32)
        action[:, :_LEG_DIM] = ll_action_train * self._t2e
        return action

    def _relative_box_body(self, rx, ry, yaw, bx, by):
        dx, dy = bx - rx, by - ry
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        bx_body = cy * dx + sy * dy
        by_body = -sy * dx + cy * dy
        return bx_body, by_body

    def _ensure_state_buffers(self, batch: int) -> None:
        if self._stage_idx_buf is not None and int(self._stage_idx_buf.shape[0]) == int(batch):
            return
        self._stage_idx_buf = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._active_stage_count_buf = torch.full(
            (batch,), min(2, self._num_stages), device=self._device, dtype=torch.long
        )
        self._curriculum_level_buf = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._stage_origin_buf = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._stage_progress_buf = torch.zeros(batch, device=self._device, dtype=torch.float32)
        self._nav_step_count_buf = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._step_wait_counter = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._step_wait_armed = torch.zeros(batch, device=self._device, dtype=torch.bool)

    def _reset_env_state(self, reset_mask: torch.Tensor, rx, ry, bx, by) -> None:
        if not bool(reset_mask.any()):
            return
        self._stage_idx_buf[reset_mask] = 0
        self._curriculum_level_buf[reset_mask] = 0
        self._active_stage_count_buf[reset_mask] = min(2, self._num_stages)
        self._stage_origin_buf[reset_mask] = float("nan")
        self._stage_progress_buf[reset_mask] = 0.0
        self._nav_step_count_buf[reset_mask] = 0
        self._step_wait_counter[reset_mask] = 0
        self._step_wait_armed[reset_mask] = False
        self._prev_robot_x[reset_mask] = rx[reset_mask]
        self._prev_robot_y[reset_mask] = ry[reset_mask]
        self._prev_box_x[reset_mask] = bx[reset_mask]
        self._prev_box_y[reset_mask] = by[reset_mask]

    def _current_stage_params(self):
        idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        axis_is_x = self._stage_axis_is_x[idx]
        push = self._stage_push[idx]
        sign = self._stage_sign[idx]
        dist = self._stage_dist[idx]
        valid = self._stage_idx_buf < self._active_stage_count_buf
        return axis_is_x, push, sign, dist, valid

    def _stage_coord(self, axis_is_x, push, rx, ry, bx, by):
        coord_robot = torch.where(axis_is_x, rx.squeeze(-1), ry.squeeze(-1))
        coord_box = torch.where(axis_is_x, bx.squeeze(-1), by.squeeze(-1))
        return torch.where(push, coord_box, coord_robot)

    def _stage_onehot(self, batch: int):
        x = torch.zeros(batch, self._num_stages, device=self._device, dtype=torch.float32)
        idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1).unsqueeze(-1)
        x.scatter_(1, idx, 1.0)
        return x

    def _build_actor_obs(self, env_obs: dict):
        proprio = env_obs["proprio"].to(self._device, dtype=torch.float32)
        batch = proprio.shape[0]
        lidar = self._lidar_compact(env_obs.get("extero", None), batch)
        lin_vel = proprio[:, _LIN_VEL_SLICE]
        ang_vel = proprio[:, _ANG_VEL_SLICE]
        gravity = proprio[:, _GRAVITY_SLICE]
        rx, ry, robot_yaw = self._robot_pose()
        bx, by, box_yaw = self._box_pose()
        bx_body, by_body = self._relative_box_body(rx, ry, robot_yaw, bx, by)
        rel_yaw = torch.atan2(torch.sin(box_yaw - robot_yaw), torch.cos(box_yaw - robot_yaw))
        stage_oh = self._stage_onehot(batch)
        stage_prog = self._stage_progress_buf.unsqueeze(-1)
        return torch.cat(
            [
                lidar,
                lin_vel,
                ang_vel,
                gravity,
                torch.cat([rx, ry, robot_yaw], dim=-1),
                torch.cat([bx, by, box_yaw], dim=-1),
                torch.cat([bx_body, by_body, rel_yaw], dim=-1),
                stage_oh,
                stage_prog,
            ],
            dim=-1,
        )

    def _build_critic_obs(self, actor_obs: torch.Tensor):
        robot = self.env.unwrapped.scene["robot"]
        box = self.env.unwrapped.scene["box"]
        r_vel = robot.data.root_lin_vel_w.to(device=self._device, dtype=torch.float32)[:, :2]
        b_vel = box.data.root_lin_vel_w.to(device=self._device, dtype=torch.float32)[:, :2]
        rx, ry, _ = self._robot_pose()
        bx, by, _ = self._box_pose()
        rel_world = torch.cat([bx - rx, by - ry], dim=-1)
        cf = self.env.unwrapped.scene["contact_sensor"].data.net_forces_w
        contact_on = (cf.norm(dim=-1).max(dim=1).values > 2.0).to(dtype=torch.float32).unsqueeze(-1)
        return torch.cat([actor_obs, r_vel, b_vel, rel_world, contact_on], dim=-1)

    def _obs_dict(self, env_obs: dict):
        policy = self._build_actor_obs(env_obs)
        critic = self._build_critic_obs(policy)
        return {"policy": policy, "critic": critic}

    def get_observations(self):
        return self._obs_dict(self._current_obs), {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_x_reached = 0
        rx, ry, _ = self._robot_pose()
        bx, by, _ = self._box_pose()
        self._ensure_state_buffers(rx.shape[0])
        self._prev_robot_x, self._prev_robot_y = rx.clone(), ry.clone()
        self._prev_box_x, self._prev_box_y = bx.clone(), by.clone()
        full_reset = torch.ones(rx.shape[0], device=self._device, dtype=torch.bool)
        self._reset_env_state(full_reset, rx, ry, bx, by)
        return self._obs_dict(obs), info

    def _update_curriculum(self):
        # Align with solution.py: always run full stage script.
        self._curriculum_level_buf.fill_(2)
        self._active_stage_count_buf.fill_(self._num_stages)

    def _compute_drop_wait_reached(
        self,
        valid: torch.Tensor,
        stage_idx: torch.Tensor,
        box_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reached = torch.zeros_like(valid)
        holding = torch.zeros_like(valid)
        for i, spec in enumerate(self.stage_specs):
            wait_z = spec.get("wait_after_box_drop_z", None)
            wait_s = spec.get("wait_after_box_drop_s", None)
            if wait_z is None or wait_s is None:
                continue
            m = valid & (stage_idx == i)
            if not bool(m.any()):
                continue
            dropped = box_z[:, 0] <= float(wait_z)
            active = m & dropped
            newly_armed = active & (~self._step_wait_armed)
            self._step_wait_armed[newly_armed] = True
            self._step_wait_counter[newly_armed] = 0
            not_dropped = m & (~dropped)
            self._step_wait_counter[not_dropped] = 0
            self._step_wait_armed[not_dropped] = False
            armed = m & self._step_wait_armed
            if bool(armed.any()):
                wait_steps = max(1, int(round(float(wait_s) / self.env.unwrapped.step_dt)))
                self._step_wait_counter[armed] += 1
                reached = reached | (armed & (self._step_wait_counter >= wait_steps))
                holding = holding | (armed & (self._step_wait_counter < wait_steps))
        return reached, holding

    def _reward_stage(self, rx, ry, bx, by, box_z, by_body, box_yaw, robot_yaw, contact_on):
        axis_is_x, push, sign, dist, valid = self._current_stage_params()
        coord_now = self._stage_coord(axis_is_x, push, rx, ry, bx, by)
        need_init = torch.isnan(self._stage_origin_buf) & valid
        self._stage_origin_buf = torch.where(need_init, coord_now, self._stage_origin_buf)

        progress = torch.where(sign > 0, coord_now - self._stage_origin_buf, self._stage_origin_buf - coord_now)
        progress = torch.where(valid, progress, torch.zeros_like(progress))
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)

        reached = (progress >= dist) & valid
        # match_box_x_tol step: reached when |robot_x - box_x| <= tol
        match_mask = torch.zeros_like(valid)
        match_err = torch.zeros_like(progress)
        for i, spec in enumerate(self.stage_specs):
            tol = spec.get("match_box_x_tol", None)
            if tol is None:
                continue
            m = valid & (stage_idx == i)
            if not bool(m.any()):
                continue
            err = (bx - rx).abs().squeeze(-1)
            match_mask = match_mask | m
            match_err = torch.where(m, err, match_err)
            reached = torch.where(m, err <= float(tol), reached)

        # stop-by-x step (advance2)
        for i, spec in enumerate(self.stage_specs):
            stop_x = spec.get("box_x_stop", None)
            if stop_x is None or spec.get("axis") != "x":
                continue
            m = valid & (stage_idx == i)
            if not bool(m.any()):
                continue
            stop_tol = float(spec.get("stop_tol", 0.1))
            if float(spec.get("sign", 1.0)) > 0:
                reached = reached | (m & (rx[:, 0] >= (float(stop_x) - stop_tol)))
            else:
                reached = reached | (m & (rx[:, 0] <= (float(stop_x) + stop_tol)))

        drop_reached, drop_holding = self._compute_drop_wait_reached(valid, stage_idx, box_z)
        reached = reached | drop_reached
        reached = reached & (~drop_holding)

        self._stage_progress_buf = torch.where(match_mask, match_err, progress)

        dbox_x = (bx - self._prev_box_x).squeeze(-1)
        dbox_y = (by - self._prev_box_y).squeeze(-1)
        push_dir = torch.where(axis_is_x, dbox_x, torch.where(sign < 0, -dbox_y, dbox_y))
        r_progress_push = 8.0 * torch.clamp(push_dir, -0.03, 0.03)

        prev_coord = self._stage_coord(axis_is_x, push, self._prev_robot_x, self._prev_robot_y, self._prev_box_x, self._prev_box_y)
        prev_prog = torch.where(sign > 0, prev_coord - self._stage_origin_buf, self._stage_origin_buf - prev_coord)
        # For match_box_x_tol step, reward reducing the alignment error.
        prev_match_err = (self._prev_box_x - self._prev_robot_x).abs().squeeze(-1)
        r_progress_nav = torch.where(
            match_mask,
            4.0 * torch.clamp(prev_match_err - match_err, -0.05, 0.05),
            4.0 * torch.clamp(progress - prev_prog, -0.05, 0.05),
        )
        r_progress = torch.where(push, r_progress_push, r_progress_nav)

        yaw_err = torch.atan2(torch.sin(box_yaw - robot_yaw), torch.cos(box_yaw - robot_yaw))
        r_yaw = -0.15 * yaw_err.squeeze(-1).abs()
        r_lat = -0.2 * by_body.squeeze(-1).abs()
        r_contact = 0.02 * contact_on.float()
        r_step = torch.full_like(r_progress, -0.002)
        r_reach_penalty = torch.where(reached, torch.full_like(r_progress, -0.2), torch.zeros_like(r_progress))
        r_sparse_bonus = torch.where(reached, torch.full_like(r_progress, self._stage_complete_bonus), torch.zeros_like(r_progress))

        reward = r_progress + r_step + r_reach_penalty + r_sparse_bonus
        reward = reward + torch.where(push, r_yaw + r_lat + r_contact, torch.zeros_like(reward))
        reward = torch.where(valid, reward, torch.zeros_like(reward))
        return reward, reached

    def step(self, nav_action: torch.Tensor):
        if not isinstance(nav_action, torch.Tensor):
            nav_action = torch.as_tensor(nav_action, dtype=torch.float32)
        nav_action = nav_action.to(self._device, dtype=torch.float32)
        if nav_action.ndim == 1:
            nav_action = nav_action.unsqueeze(0)
        self._ensure_state_buffers(self.num_envs)
        self._nav_step_count_buf += 1
        self._update_curriculum()

        total_reward = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        terminated = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        last_info = {}

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
            done_now = step_term_1d | step_trunc_1d
            terminated |= step_term_1d
            truncated |= step_trunc_1d
            last_info = info

            rx, ry, robot_yaw = self._robot_pose()
            bx, by, box_yaw = self._box_pose()
            box = self.env.unwrapped.scene["box"]
            box_z = box.data.root_pos_w.to(device=self._device, dtype=torch.float32)[:, 2:3]
            _, by_body = self._relative_box_body(rx, ry, robot_yaw, bx, by)
            cf = self.env.unwrapped.scene["contact_sensor"].data.net_forces_w
            contact_on = cf.norm(dim=-1).max(dim=1).values > 2.0

            x_reached_now = step_term_1d & (rx.squeeze(-1) > 3.5)
            timeout_now = step_trunc_1d
            fall_now = step_term_1d & (~x_reached_now)
            self._done_x_reached += int(x_reached_now.sum().item())
            self._done_timeout += int(timeout_now.sum().item())
            self._done_fall += int(fall_now.sum().item())
            self._done_total += int(done_now.sum().item())

            shaped, reached = self._reward_stage(rx, ry, bx, by, box_z, by_body, box_yaw, robot_yaw, contact_on)
            if isinstance(base_rew, torch.Tensor):
                base_rew_1d = base_rew.to(self._device)
                if base_rew_1d.ndim > 1:
                    base_rew_1d = base_rew_1d.squeeze(-1)
                shaped = shaped + 0.10 * base_rew_1d
            else:
                base_rew_1d = torch.as_tensor(base_rew, device=self._device, dtype=torch.float32)
                if base_rew_1d.ndim > 1:
                    base_rew_1d = base_rew_1d.squeeze(-1)
                shaped = shaped + 0.10 * base_rew_1d
            total_reward += shaped

            if bool(reached.any()):
                self._stage_idx_buf = torch.where(reached, self._stage_idx_buf + 1, self._stage_idx_buf)
                self._stage_origin_buf = torch.where(
                    reached, torch.full_like(self._stage_origin_buf, float("nan")), self._stage_origin_buf
                )
                completed = reached & (self._stage_idx_buf >= self._active_stage_count_buf)
                if bool(completed.any()):
                    total_reward[completed] += 2.0

            self._prev_robot_x, self._prev_robot_y = rx.clone(), ry.clone()
            self._prev_box_x, self._prev_box_y = bx.clone(), by.clone()

            self._reset_env_state(done_now, rx, ry, bx, by)

        idx0 = int(self._stage_idx_buf[0].item())
        active0 = int(self._active_stage_count_buf[0].item())
        level0 = int(self._curriculum_level_buf[0].item())
        nav0 = int(self._nav_step_count_buf[0].item())
        stage_name = self._stage_names[idx0] if idx0 < active0 and idx0 < self._num_stages else "done"
        if nav0 <= 20 or nav0 % 50 == 0:
            denom = max(1, self._done_total)
            print(
                f"[TaskDTeacher] nav={nav0:5d} curriculum={level0} "
                f"active={active0} stage={min(idx0 + 1, active0)}/{active0}({stage_name}) "
                f"prog={float(self._stage_progress_buf[0].item()):.2f} rew={total_reward.mean().item():+.4f} "
                f"done[f/t/x]={self._done_fall}/{self._done_timeout}/{self._done_x_reached} "
                f"ratio={self._done_fall/denom:.2f}/{self._done_timeout/denom:.2f}/{self._done_x_reached/denom:.2f}",
                flush=True,
            )

        obs_dict = self._obs_dict(self._current_obs)
        last_info = dict(last_info) if isinstance(last_info, dict) else {}
        last_info["teacher_stage"] = stage_name
        last_info["teacher_stage_idx"] = idx0
        last_info["teacher_curriculum_level"] = level0
        last_info["teacher_stage_progress"] = float(self._stage_progress_buf[0].item())
        return obs_dict, total_reward, terminated, truncated, last_info
