"""Task D teacher wrapper with asymmetric actor/critic observations."""

from __future__ import annotations

import math
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

_ROBOT_SPAWN_XY = (-3.0, 0.0)
_BOX_SPAWN_XY = (-3.0, 1.6)


def _build_nominal_waypoints(stage_specs: list[dict]) -> list[tuple[float, float]]:
    """Build a nominal world-frame polyline from scripted stage specs."""
    x, y = float(_ROBOT_SPAWN_XY[0]), float(_ROBOT_SPAWN_XY[1])
    box_x, box_y = float(_BOX_SPAWN_XY[0]), float(_BOX_SPAWN_XY[1])
    waypoints = [(x, y)]
    for spec in stage_specs:
        if spec.get("match_box_x_tol") is not None:
            x = box_x
        if spec.get("approach_y") is not None:
            y = float(spec["approach_y"])
        elif spec.get("match_box_y_tol") is not None:
            y = box_y
        elif spec.get("box_x_stop") is not None:
            x = float(spec["box_x_stop"])
        else:
            axis = spec.get("axis", "x")
            sign = float(spec.get("sign", 1.0))
            dist = float(spec.get("dist", 0.0))
            if axis == "xy":
                d = dist / math.sqrt(2.0)
                x += d if sign > 0 else -d
                y += d if sign > 0 else -d
            elif axis == "x":
                x += dist if sign > 0 else -dist
            else:
                y += dist if sign > 0 else -dist
        waypoints.append((x, y))
    return waypoints


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
        nav_log_interval: int = 50,
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
        self._nav_log_interval = max(0, int(nav_log_interval))
        self._nav_log_tag = "TaskDTeacher"
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
                match_box_lateral_tol=0.25,
                match_box_hold_s=0.3,
            ),
            dict(
                name="sidestep_right",
                axis="y",
                sign=-1,
                dist=2.0,
                push=True,
                sparse_bonus=1.2,
                align_x_with_box=True,
                relative_robot_target=True,
            ),
            dict(
                name="retreat2",
                axis="x",
                sign=-1,
                dist=1.0,
                push=False,
                sparse_bonus=0.8,
                relative_robot_target=True,
            ),
            dict(
                name="sidestep_right2",
                axis="y",
                sign=-1,
                dist=0.6,
                push=False,
                sparse_bonus=0.8,
                relative_robot_target=True,
                match_box_y_target=True,
                match_box_y_tol=0.15,
                match_box_lateral_tol=0.25,
                match_box_hold_s=0.3,
            ),
            dict(
                name="advance2",
                axis="x",
                sign=+1,
                dist=4.0,
                push=True,
                sparse_bonus=1.6,
                align_y_with_box=True,
                relative_robot_target=True,
                box_x_slow_start=-1.7,
                box_x_stop=-1.25,
                stop_tol=0.08,
                wait_after_box_drop_z=-0.35,
                wait_after_box_drop_s=1.0,
            ),
            dict(
                name="final",
                axis="x",
                sign=+1,
                dist=3.0,
                push=False,
                sparse_bonus=1.0,
                relative_robot_target=True,
            ),
        ]
        self._num_stages = len(self.stage_specs)
        self._stage_names = [spec["name"] for spec in self.stage_specs]
        self._stage_axis_is_x = torch.tensor(
            [spec["axis"] == "x" for spec in self.stage_specs], device=self._device, dtype=torch.bool
        )
        self._stage_push = torch.tensor(
            [bool(spec["push"]) for spec in self.stage_specs], device=self._device, dtype=torch.bool
        )
        self._stage_relative_target = torch.tensor(
            [bool(spec.get("relative_robot_target", False)) for spec in self.stage_specs],
            device=self._device,
            dtype=torch.bool,
        )
        self._stage_match_box_y_target = torch.tensor(
            [bool(spec.get("match_box_y_target", False)) for spec in self.stage_specs],
            device=self._device,
            dtype=torch.bool,
        )
        self._stage_match_box_x_tol = torch.tensor(
            [
                float(spec.get("match_box_x_tol"))
                if spec.get("match_box_x_tol") is not None
                else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._stage_match_box_y_tol = torch.tensor(
            [
                float(spec.get("match_box_y_tol"))
                if spec.get("match_box_y_tol") is not None
                else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._stage_match_box_lateral_tol = torch.tensor(
            [
                float(spec.get("match_box_lateral_tol"))
                if spec.get("match_box_lateral_tol") is not None
                else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._stage_match_box_hold_s = torch.tensor(
            [
                float(spec.get("match_box_hold_s"))
                if spec.get("match_box_hold_s") is not None
                else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._stage_approach_y = torch.tensor(
            [
                float(spec.get("approach_y"))
                if spec.get("approach_y") is not None
                else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._stage_face_box_tol = torch.tensor(
            [
                float(spec.get("face_box_tol"))
                if spec.get("face_box_tol") is not None
                else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._stage_sign = torch.tensor([float(spec["sign"]) for spec in self.stage_specs], device=self._device, dtype=torch.float32)
        self._stage_dist = torch.tensor([float(spec["dist"]) for spec in self.stage_specs], device=self._device, dtype=torch.float32)
        self._stage_box_x_stop = torch.tensor(
            [
                float(spec["box_x_stop"]) if spec.get("box_x_stop") is not None else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._stage_stop_tol = torch.tensor(
            [
                float(spec.get("stop_tol", 0.08)) if spec.get("box_x_stop") is not None else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._push_progress_tol = 0.08

        self._stage_idx_buf: torch.Tensor | None = None
        self._active_stage_count_buf: torch.Tensor | None = None
        self._curriculum_level_buf: torch.Tensor | None = None
        self._stage_origin_x_buf: torch.Tensor | None = None
        self._stage_origin_y_buf: torch.Tensor | None = None
        self._stage_progress_buf: torch.Tensor | None = None
        self._nav_step_count_buf: torch.Tensor | None = None
        self._step_wait_counter: torch.Tensor | None = None
        self._step_wait_armed: torch.Tensor | None = None
        self._current_obs: dict | None = None
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_x_reached = 0
        self._done_traj = 0
        self._done_stage_target = 0
        self._done_no_motion = 0
        self._done_no_target_progress = 0

        # Stage indices aligned with demo/solution.py nav_steps.
        self._idx_retreat = 0
        self._idx_sidestep_left = 1
        self._idx_advance_match = 2
        self._idx_sidestep_right = 3
        self._idx_retreat2 = 4
        self._idx_sidestep_right2 = 5
        self._idx_advance2 = 6
        self._idx_final = 7

        # Reward: progress toward stage target (bounded) + sparse bonus on reach.
        self._w_nav_dist = 3.0
        self._nav_dist_delta_clip = 0.05
        # Distance tolerance for waypoint stages; match_box_* stages use box alignment instead.
        self._stage_reach_tol = 0.35
        self._r_stage_complete = 1.0
        self._r_stage_complete_final = 2.0
        # Per-nav-step penalty (not per physics step) so inner_steps can fine-tune without 5x time tax.
        self._r_nav_step_penalty = 0.01
        # advance match_box: reward staying aligned (pairs with match_box_hold_s completion).
        self._w_match_box_hold = 0.04
        self._match_box_hold_speed_max = 0.35
        # sidestep_right: reward shrinking |robot_x - box_x| (progress-based).
        self._w_push_x_align = 6.0
        self._push_x_align_delta_clip = 0.05
        # advance2: reward shrinking |robot_y - box_y| (progress-based).
        self._w_push_y_align = 6.0
        self._push_y_align_delta_clip = 0.05
        # Push stages: reward box movement toward current stage target.
        self._w_push_box_target = 4.0
        self._push_box_target_delta_clip = 0.05
        # approach_box: reward turning to face the box (progress-based on yaw error).
        self._w_face_box = 2.0
        self._face_box_delta_clip = 0.05

        self._traj_waypoints = _build_nominal_waypoints(self.stage_specs)
        stage_targets = self._traj_waypoints[1 : self._num_stages + 1]
        stage_starts = self._traj_waypoints[: self._num_stages]
        self._stage_start_x = torch.tensor([p[0] for p in stage_starts], device=self._device, dtype=torch.float32)
        self._stage_start_y = torch.tensor([p[1] for p in stage_starts], device=self._device, dtype=torch.float32)
        self._stage_target_x = torch.tensor([p[0] for p in stage_targets], device=self._device, dtype=torch.float32)
        self._stage_target_y = torch.tensor([p[1] for p in stage_targets], device=self._device, dtype=torch.float32)
        self._traj_seg_x0 = None
        self._traj_seg_y0 = None
        self._traj_seg_dx = None
        self._traj_seg_dy = None
        self._traj_seg_len2 = None
        self._traj_prefix_len = None
        self._traj_total_len = 0.0
        self._build_traj_segments()
        self._prev_robot_s = None
        self._prev_rel_progress = None
        self._prev_dist_to_target: torch.Tensor | None = None
        self._prev_x_align_err: torch.Tensor | None = None
        self._prev_y_align_err: torch.Tensor | None = None
        self._prev_box_target_dist: torch.Tensor | None = None
        self._prev_face_box_yaw_err: torch.Tensor | None = None

        self._post_push_anchor_rx: torch.Tensor | None = None
        self._post_push_anchor_ry: torch.Tensor | None = None
        self._post_push_anchor_bx: torch.Tensor | None = None
        self._post_push_anchor_by: torch.Tensor | None = None
        self._pre_adv2_anchor_rx: torch.Tensor | None = None
        self._pre_adv2_anchor_ry: torch.Tensor | None = None
        self._pre_adv2_anchor_bx: torch.Tensor | None = None
        self._pre_adv2_anchor_by: torch.Tensor | None = None

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
            f"{self._curriculum_mid_nav_steps}] nav_log_interval={self._nav_log_interval}",
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

    def _build_traj_segments(self) -> None:
        if len(self._traj_waypoints) < 2:
            self._traj_seg_x0 = torch.zeros(1, device=self._device, dtype=torch.float32)
            self._traj_seg_y0 = torch.zeros(1, device=self._device, dtype=torch.float32)
            self._traj_seg_dx = torch.zeros(1, device=self._device, dtype=torch.float32)
            self._traj_seg_dy = torch.zeros(1, device=self._device, dtype=torch.float32)
            self._traj_seg_len2 = torch.ones(1, device=self._device, dtype=torch.float32)
            self._traj_prefix_len = torch.zeros(1, device=self._device, dtype=torch.float32)
            self._traj_total_len = 0.0
            return

        x0 = []
        y0 = []
        dx = []
        dy = []
        len2 = []
        prefix = []
        running = 0.0
        for i in range(len(self._traj_waypoints) - 1):
            ax, ay = self._traj_waypoints[i]
            bx, by = self._traj_waypoints[i + 1]
            vx = bx - ax
            vy = by - ay
            l2 = max(vx * vx + vy * vy, 1.0e-12)
            x0.append(ax)
            y0.append(ay)
            dx.append(vx)
            dy.append(vy)
            len2.append(l2)
            prefix.append(running)
            running += math.sqrt(l2)

        self._traj_seg_x0 = torch.tensor(x0, device=self._device, dtype=torch.float32)
        self._traj_seg_y0 = torch.tensor(y0, device=self._device, dtype=torch.float32)
        self._traj_seg_dx = torch.tensor(dx, device=self._device, dtype=torch.float32)
        self._traj_seg_dy = torch.tensor(dy, device=self._device, dtype=torch.float32)
        self._traj_seg_len2 = torch.tensor(len2, device=self._device, dtype=torch.float32)
        self._traj_prefix_len = torch.tensor(prefix, device=self._device, dtype=torch.float32)
        self._traj_total_len = running

    def _project_to_traj(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (arc-length projection s, minimum distance d) for each env."""
        px = x.squeeze(-1).to(device=self._device, dtype=torch.float32)
        py = y.squeeze(-1).to(device=self._device, dtype=torch.float32)
        seg = self._traj_seg_x0.shape[0]
        if seg == 1 and float(self._traj_seg_len2[0].item()) <= 1.0e-12:
            d = torch.hypot(px - self._traj_seg_x0[0], py - self._traj_seg_y0[0])
            return torch.zeros_like(d), d

        rx = px.unsqueeze(-1) - self._traj_seg_x0.unsqueeze(0)
        ry = py.unsqueeze(-1) - self._traj_seg_y0.unsqueeze(0)
        t = (rx * self._traj_seg_dx.unsqueeze(0) + ry * self._traj_seg_dy.unsqueeze(0)) / self._traj_seg_len2.unsqueeze(0)
        t = torch.clamp(t, 0.0, 1.0)
        proj_x = self._traj_seg_x0.unsqueeze(0) + t * self._traj_seg_dx.unsqueeze(0)
        proj_y = self._traj_seg_y0.unsqueeze(0) + t * self._traj_seg_dy.unsqueeze(0)
        dist = torch.hypot(px.unsqueeze(-1) - proj_x, py.unsqueeze(-1) - proj_y)
        min_dist, idx = dist.min(dim=-1)
        s = self._traj_prefix_len[idx] + t.gather(1, idx.unsqueeze(-1)).squeeze(-1) * torch.sqrt(self._traj_seg_len2[idx])
        return s, min_dist

    def _ensure_state_buffers(self, batch: int) -> None:
        if self._stage_idx_buf is not None and int(self._stage_idx_buf.shape[0]) == int(batch):
            return
        self._stage_idx_buf = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._active_stage_count_buf = torch.full(
            (batch,), self._num_stages, device=self._device, dtype=torch.long
        )
        self._curriculum_level_buf = torch.full((batch,), 2, device=self._device, dtype=torch.long)
        self._stage_origin_x_buf = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._stage_origin_y_buf = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._stage_progress_buf = torch.zeros(batch, device=self._device, dtype=torch.float32)
        self._nav_step_count_buf = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._step_wait_counter = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._step_wait_armed = torch.zeros(batch, device=self._device, dtype=torch.bool)
        self._match_box_align_hold_counter = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._prev_robot_s = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_rel_progress = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_dist_to_target = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_x_align_err = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_y_align_err = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_box_target_dist = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_face_box_yaw_err = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._post_push_anchor_rx = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._post_push_anchor_ry = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._post_push_anchor_bx = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._post_push_anchor_by = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._pre_adv2_anchor_rx = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._pre_adv2_anchor_ry = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._pre_adv2_anchor_bx = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._pre_adv2_anchor_by = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._box_stage_origin_x_buf = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._box_stage_origin_y_buf = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._stage_advanced_this_nav = torch.zeros(batch, device=self._device, dtype=torch.bool)
        self._ep_return_buf = torch.zeros(batch, device=self._device, dtype=torch.float32)
        self._ep_dense_buf = torch.zeros(batch, device=self._device, dtype=torch.float32)
        self._ep_sparse_buf = torch.zeros(batch, device=self._device, dtype=torch.float32)
        self._ep_nav_steps_buf = torch.zeros(batch, device=self._device, dtype=torch.long)

    def _reset_env_state(self, reset_mask: torch.Tensor, rx, ry, bx, by) -> None:
        if not bool(reset_mask.any()):
            return
        self._stage_idx_buf[reset_mask] = 0
        self._curriculum_level_buf[reset_mask] = 2
        self._active_stage_count_buf[reset_mask] = self._num_stages
        self._stage_origin_x_buf[reset_mask] = float("nan")
        self._stage_origin_y_buf[reset_mask] = float("nan")
        self._stage_progress_buf[reset_mask] = 0.0
        self._nav_step_count_buf[reset_mask] = 0
        self._step_wait_counter[reset_mask] = 0
        self._step_wait_armed[reset_mask] = False
        self._match_box_align_hold_counter[reset_mask] = 0
        self._prev_robot_x[reset_mask] = rx[reset_mask]
        self._prev_robot_y[reset_mask] = ry[reset_mask]
        self._prev_box_x[reset_mask] = bx[reset_mask]
        self._prev_box_y[reset_mask] = by[reset_mask]
        self._prev_robot_s[reset_mask] = float("nan")
        self._prev_rel_progress[reset_mask] = float("nan")
        self._prev_dist_to_target[reset_mask] = float("nan")
        self._prev_x_align_err[reset_mask] = float("nan")
        self._prev_y_align_err[reset_mask] = float("nan")
        self._prev_box_target_dist[reset_mask] = float("nan")
        self._prev_face_box_yaw_err[reset_mask] = float("nan")
        self._post_push_anchor_rx[reset_mask] = float("nan")
        self._post_push_anchor_ry[reset_mask] = float("nan")
        self._post_push_anchor_bx[reset_mask] = float("nan")
        self._post_push_anchor_by[reset_mask] = float("nan")
        self._pre_adv2_anchor_rx[reset_mask] = float("nan")
        self._pre_adv2_anchor_ry[reset_mask] = float("nan")
        self._pre_adv2_anchor_bx[reset_mask] = float("nan")
        self._pre_adv2_anchor_by[reset_mask] = float("nan")
        self._box_stage_origin_x_buf[reset_mask] = float("nan")
        self._box_stage_origin_y_buf[reset_mask] = float("nan")
        self._stage_advanced_this_nav[reset_mask] = False

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

    def _termination_term_flags(self) -> dict[str, torch.Tensor]:
        tm = getattr(self.env.unwrapped, "termination_manager", None)
        if tm is None:
            return {}
        out: dict[str, torch.Tensor] = {}
        for name in ("fall", "x_reached", "time_out", "stage_target_deviation", "trajectory_deviation", "no_motion_timeout", "no_target_progress_timeout"):
            try:
                out[name] = tm.get_term(name).view(-1)[: self.num_envs].to(
                    device=self._device, dtype=torch.bool
                )
            except Exception:
                pass
        return out

    def _count_done_terms(self, done_now: torch.Tensor) -> None:
        term_flags = self._termination_term_flags()
        if term_flags:
            if "fall" in term_flags:
                self._done_fall += int((done_now & term_flags["fall"]).sum().item())
            if "time_out" in term_flags:
                self._done_timeout += int((done_now & term_flags["time_out"]).sum().item())
            if "x_reached" in term_flags:
                self._done_x_reached += int((done_now & term_flags["x_reached"]).sum().item())
            if "trajectory_deviation" in term_flags:
                self._done_traj += int((done_now & term_flags["trajectory_deviation"]).sum().item())
            if "stage_target_deviation" in term_flags:
                self._done_stage_target += int((done_now & term_flags["stage_target_deviation"]).sum().item())
            if "no_motion_timeout" in term_flags:
                self._done_no_motion += int((done_now & term_flags["no_motion_timeout"]).sum().item())
            if "no_target_progress_timeout" in term_flags:
                self._done_no_target_progress += int((done_now & term_flags["no_target_progress_timeout"]).sum().item())
            return

        # Fallback when termination manager is unavailable.
        rx, _, _ = self._robot_pose()
        x_reached_now = done_now & (rx.squeeze(-1) > 3.5)
        self._done_x_reached += int(x_reached_now.sum().item())
        self._done_fall += int((done_now & (~x_reached_now)).sum().item())

    def get_observations(self):
        return self._obs_dict(self._current_obs), {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self._done_total = 0
        self._done_fall = 0
        self._done_timeout = 0
        self._done_x_reached = 0
        self._done_traj = 0
        self._done_stage_target = 0
        self._done_no_motion = 0
        self._done_no_target_progress = 0
        rx, ry, _ = self._robot_pose()
        bx, by, _ = self._box_pose()
        self._ensure_state_buffers(rx.shape[0])
        self._prev_robot_x, self._prev_robot_y = rx.clone(), ry.clone()
        self._prev_box_x, self._prev_box_y = bx.clone(), by.clone()
        full_reset = torch.ones(rx.shape[0], device=self._device, dtype=torch.bool)
        self._reset_env_state(full_reset, rx, ry, bx, by)
        self._sync_nav_stage_idx(rx, ry, bx, by)
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

    def _apply_match_box_hold(
        self,
        use_match: torch.Tensor,
        aligned_now: torch.Tensor,
        hold_s: torch.Tensor,
    ) -> torch.Tensor:
        """Require continuous box alignment for hold_s seconds before stage completion."""
        counter = self._match_box_align_hold_counter
        counter = torch.where(use_match, counter, torch.zeros_like(counter))
        misaligned = use_match & (~aligned_now)
        counter = torch.where(misaligned, torch.zeros_like(counter), counter)
        aligned = use_match & aligned_now
        hold_steps = torch.clamp(
            torch.round(hold_s / float(self.env.unwrapped.step_dt)).to(dtype=torch.long),
            min=1,
        )
        counter = torch.where(aligned, counter + 1, counter)
        self._match_box_align_hold_counter = counter
        held = aligned & (counter >= hold_steps)
        return held

    def _match_box_alignment_ok(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx1: torch.Tensor,
        ry1: torch.Tensor,
        bx1: torch.Tensor,
        by1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Box alignment for match_box_x_tol / match_box_y_tol stages (+ optional lateral tol)."""
        x_tol = self._stage_match_box_x_tol[stage_idx]
        y_tol = self._stage_match_box_y_tol[stage_idx]
        lat_tol = self._stage_match_box_lateral_tol[stage_idx]
        use_x_match = valid & (~torch.isnan(x_tol))
        use_y_match = valid & (~torch.isnan(y_tol))
        use_lat = ~torch.isnan(lat_tol)

        x_ok = (rx1 - bx1).abs() <= x_tol
        y_ok = (ry1 - by1).abs() <= y_tol
        lat_y_ok = (ry1 - by1).abs() <= lat_tol
        lat_x_ok = (rx1 - bx1).abs() <= lat_tol

        x_align = torch.where(use_x_match, x_ok, torch.zeros_like(valid))
        x_align = torch.where(use_x_match & use_lat, x_align & lat_y_ok, x_align)
        y_align = torch.where(use_y_match, y_ok, torch.zeros_like(valid))
        y_align = torch.where(use_y_match & use_lat, y_align & lat_x_ok, y_align)
        return x_align, y_align, use_x_match, use_y_match

    def _match_box_held_reached(
        self,
        aligned_now: torch.Tensor,
        use_match: torch.Tensor,
        stage_idx: torch.Tensor,
    ) -> torch.Tensor:
        hold_s_stage = self._stage_match_box_hold_s[stage_idx]
        use_hold = use_match & (~torch.isnan(hold_s_stage))
        held = aligned_now
        if bool(use_hold.any()):
            held = self._apply_match_box_hold(use_hold, aligned_now, hold_s_stage)
        return held

    def _update_stage_anchors(
        self,
        reached: torch.Tensor,
        stage_idx: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> None:
        rx1 = rx.squeeze(-1)
        ry1 = ry.squeeze(-1)
        bx1 = bx.squeeze(-1)
        by1 = by.squeeze(-1)
        finish_push1 = reached & (stage_idx == self._idx_sidestep_right)
        self._post_push_anchor_rx = torch.where(finish_push1, rx1, self._post_push_anchor_rx)
        self._post_push_anchor_ry = torch.where(finish_push1, ry1, self._post_push_anchor_ry)
        self._post_push_anchor_bx = torch.where(finish_push1, bx1, self._post_push_anchor_bx)
        self._post_push_anchor_by = torch.where(finish_push1, by1, self._post_push_anchor_by)

        finish_sidestep2 = reached & (stage_idx == self._idx_sidestep_right2)
        self._pre_adv2_anchor_rx = torch.where(finish_sidestep2, rx1, self._pre_adv2_anchor_rx)
        self._pre_adv2_anchor_ry = torch.where(finish_sidestep2, ry1, self._pre_adv2_anchor_ry)
        self._pre_adv2_anchor_bx = torch.where(finish_sidestep2, bx1, self._pre_adv2_anchor_bx)
        self._pre_adv2_anchor_by = torch.where(finish_sidestep2, by1, self._pre_adv2_anchor_by)

    def _relative_axis_progress(
        self,
        sign: torch.Tensor,
        coord_now: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        delta = coord_now - anchor
        signed = torch.where(sign > 0, delta, -delta)
        return torch.clamp(signed, min=0.0)

    def _ensure_push_box_origin(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> None:
        push_active = valid & self._stage_push[stage_idx]
        origin_unset = torch.isnan(self._box_stage_origin_x_buf) | torch.isnan(self._box_stage_origin_y_buf)
        arm = push_active & origin_unset
        if not bool(arm.any()):
            return
        self._box_stage_origin_x_buf = torch.where(arm, bx.squeeze(-1), self._box_stage_origin_x_buf)
        self._box_stage_origin_y_buf = torch.where(arm, by.squeeze(-1), self._box_stage_origin_y_buf)

    def _box_axis_progress(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_push_box_origin(stage_idx, valid, bx, by)
        axis_is_x = self._stage_axis_is_x[stage_idx]
        sign = self._stage_sign[stage_idx]
        coord = torch.where(axis_is_x, bx.squeeze(-1), by.squeeze(-1))
        origin = torch.where(axis_is_x, self._box_stage_origin_x_buf, self._box_stage_origin_y_buf)
        return self._relative_axis_progress(sign, coord, origin)

    def _task_progress_distance(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> torch.Tensor:
        rx1 = rx.squeeze(-1)
        ry1 = ry.squeeze(-1)
        bx1 = bx.squeeze(-1)
        by1 = by.squeeze(-1)
        target_x, target_y = self._compute_stage_target(stage_idx, valid, rx, ry, bx, by)
        robot_dist = torch.hypot(rx1 - target_x, ry1 - target_y)
        push_stage = valid & self._stage_push[stage_idx]
        dist_req = self._stage_dist[stage_idx]
        box_prog = self._box_axis_progress(stage_idx, valid, bx, by)
        box_remaining = torch.clamp(dist_req - box_prog, min=0.0)
        x_tol = self._stage_match_box_x_tol[stage_idx]
        y_tol = self._stage_match_box_y_tol[stage_idx]
        lat_tol = self._stage_match_box_lateral_tol[stage_idx]
        is_match_x = valid & (~torch.isnan(x_tol)) & (~push_stage)
        is_match_y = valid & (~torch.isnan(y_tol)) & (~push_stage)
        use_lat = ~torch.isnan(lat_tol)
        x_err = (rx1 - bx1).abs()
        y_err = (ry1 - by1).abs()
        align_x_err = x_err
        align_y_err = y_err
        if bool(use_lat.any()):
            align_x_err = torch.where(is_match_x & use_lat, torch.maximum(x_err, y_err), align_x_err)
            align_y_err = torch.where(is_match_y & use_lat, torch.maximum(y_err, x_err), align_y_err)
        waypoint_dist = torch.where(
            is_match_x,
            align_x_err,
            torch.where(is_match_y, align_y_err, robot_dist),
        )
        return torch.where(push_stage, box_remaining, waypoint_dist)

    def get_stage_distribution_log(self) -> dict[str, torch.Tensor]:
        """Snapshot: fraction of parallel envs at each stage (for PPO iteration log)."""
        if self._stage_idx_buf is None:
            return {}
        idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        counts = torch.bincount(idx, minlength=self._num_stages).to(dtype=torch.float32)
        total = max(float(counts.sum().item()), 1.0)
        fracs = counts / total
        out: dict[str, torch.Tensor] = {
            "Nav/live/stage_idx_mean": (idx.to(dtype=torch.float32) + 1.0).mean().unsqueeze(0),
            "Nav/live/stages_ge2_frac": (idx >= 1).to(dtype=torch.float32).mean().unsqueeze(0),
        }
        for i in range(self._num_stages):
            out[f"Nav/live/s{i + 1}_frac"] = fracs[i].unsqueeze(0)
        return out

    def format_stage_distribution_line(self) -> str:
        """Compact one-line stage mix for terminal logs."""
        if self._stage_idx_buf is None:
            return ""
        idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        counts = torch.bincount(idx, minlength=self._num_stages)
        total = max(int(counts.sum().item()), 1)
        parts = []
        for i in range(self._num_stages):
            frac = 100.0 * float(counts[i].item()) / float(total)
            if frac < 0.05:
                continue
            name = self._stage_names[i] if i < len(self._stage_names) else f"s{i + 1}"
            parts.append(f"s{i + 1}({name})={frac:.0f}%")
        mean_stage = float((idx.float() + 1.0).mean().item())
        mix = " ".join(parts) if parts else "s1=100%"
        return f"Nav/live stage_mix: {mix} mean={mean_stage:.2f}"

    def _publish_nav_traj_segments(
        self,
        sx: torch.Tensor,
        sy: torch.Tensor,
        ex: torch.Tensor,
        ey: torch.Tensor,
    ) -> None:
        base = self.env.unwrapped
        device = base.device
        for name in ("_nav_seg_x0", "_nav_seg_y0", "_nav_seg_x1", "_nav_seg_y1"):
            if (
                not hasattr(base, name)
                or not isinstance(getattr(base, name), torch.Tensor)
                or int(getattr(base, name).shape[0]) != int(self.num_envs)
            ):
                setattr(base, name, torch.zeros(self.num_envs, device=device, dtype=torch.float32))
        base._nav_seg_x0.copy_(sx.to(device=device, dtype=torch.float32))
        base._nav_seg_y0.copy_(sy.to(device=device, dtype=torch.float32))
        base._nav_seg_x1.copy_(ex.to(device=device, dtype=torch.float32))
        base._nav_seg_y1.copy_(ey.to(device=device, dtype=torch.float32))

    def _sync_nav_stage_idx(self, rx: torch.Tensor, ry: torch.Tensor, bx: torch.Tensor, by: torch.Tensor) -> None:
        """Expose current stage index and segment endpoints to base env termination manager."""
        base = self.env.unwrapped
        if (
            not hasattr(base, "_nav_stage_idx")
            or not isinstance(base._nav_stage_idx, torch.Tensor)
            or int(base._nav_stage_idx.shape[0]) != int(self.num_envs)
        ):
            base._nav_stage_idx = torch.zeros(self.num_envs, device=base.device, dtype=torch.long)
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        valid = self._stage_idx_buf < self._active_stage_count_buf
        base._nav_stage_idx.copy_(stage_idx)

        sx, sy, ex, ey = self._compute_stage_segment_endpoints(stage_idx, valid, rx, ry, bx, by)
        self._publish_nav_traj_segments(sx, sy, ex, ey)

        dist_to_target = self._task_progress_distance(stage_idx, valid, rx, ry, bx, by)
        self._stage_progress_buf = dist_to_target
        if (
            not hasattr(base, "_nav_dist_to_target")
            or not isinstance(base._nav_dist_to_target, torch.Tensor)
            or int(base._nav_dist_to_target.shape[0]) != int(self.num_envs)
        ):
            base._nav_dist_to_target = torch.zeros(self.num_envs, device=base.device, dtype=torch.float32)
        base._nav_dist_to_target.copy_(dist_to_target.to(device=base.device, dtype=torch.float32))
        if (
            not hasattr(base, "_nav_stage_active")
            or not isinstance(base._nav_stage_active, torch.Tensor)
            or int(base._nav_stage_active.shape[0]) != int(self.num_envs)
        ):
            base._nav_stage_active = torch.ones(self.num_envs, device=base.device, dtype=torch.bool)
        base._nav_stage_active.copy_(valid.to(device=base.device, dtype=torch.bool))

    def _ensure_relative_stage_origin(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
    ) -> None:
        rel_stage = valid & self._stage_relative_target[stage_idx]
        origin_init = torch.isnan(self._stage_origin_y_buf)
        arm = rel_stage & origin_init
        if bool(arm.any()):
            self._stage_origin_x_buf = torch.where(arm, rx.squeeze(-1), self._stage_origin_x_buf)
            self._stage_origin_y_buf = torch.where(arm, ry.squeeze(-1), self._stage_origin_y_buf)

    def _relative_stage_target(self, stage_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tx = self._stage_origin_x_buf.clone()
        ty = self._stage_origin_y_buf.clone()
        axis_is_x = self._stage_axis_is_x[stage_idx]
        sign = self._stage_sign[stage_idx]
        dist = self._stage_dist[stage_idx]
        tx = torch.where(axis_is_x, tx + sign * dist, tx)
        ty = torch.where(axis_is_x, ty, ty + sign * dist)
        return tx, ty

    def _compute_stage_segment_endpoints(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rx1 = rx.squeeze(-1)
        ry1 = ry.squeeze(-1)
        by1 = by.squeeze(-1)

        self._ensure_relative_stage_origin(stage_idx, valid, rx, ry)

        sx = self._stage_start_x[stage_idx]
        sy = self._stage_start_y[stage_idx]
        ex = self._stage_target_x[stage_idx]
        ey = self._stage_target_y[stage_idx]
        has_origin = (~torch.isnan(self._stage_origin_x_buf)) & (~torch.isnan(self._stage_origin_y_buf))

        # Only relative stages use previous-stage entry as segment start.
        # Early approach stages (retreat/sidestep_left/advance) stay absolute.
        entry_start = valid & has_origin & self._stage_relative_target[stage_idx]
        sx = torch.where(entry_start, self._stage_origin_x_buf, sx)
        sy = torch.where(entry_start, self._stage_origin_y_buf, sy)

        approach_y = self._stage_approach_y[stage_idx]
        approach_stage = valid & (~torch.isnan(approach_y))
        if bool(approach_stage.any()):
            ex = torch.where(approach_stage, bx.squeeze(-1), ex)
            ey = torch.where(approach_stage, approach_y, ey)

        match_box_y = valid & self._stage_match_box_y_target[stage_idx]
        if bool(match_box_y.any()):
            anchor_x = torch.where(has_origin, self._stage_origin_x_buf, rx1)
            anchor_y = torch.where(has_origin, self._stage_origin_y_buf, ry1)
            sx = torch.where(match_box_y, anchor_x, sx)
            sy = torch.where(match_box_y, anchor_y, sy)
            ex = torch.where(match_box_y, anchor_x, ex)
            ey = torch.where(match_box_y, by1, ey)

        rel_stage = (
            valid
            & self._stage_relative_target[stage_idx]
            & (~self._stage_match_box_y_target[stage_idx])
            & has_origin
        )
        if bool(rel_stage.any()):
            rel_tx, rel_ty = self._relative_stage_target(stage_idx)
            ex = torch.where(rel_stage, rel_tx, ex)
            ey = torch.where(rel_stage, rel_ty, ey)

        return sx, sy, ex, ey

    def _compute_stage_target(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, target_x, target_y = self._compute_stage_segment_endpoints(stage_idx, valid, rx, ry, bx, by)
        return target_x, target_y

    def _compute_stage_reached(
        self,
        rx,
        ry,
        bx,
        by,
        box_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        valid = self._stage_idx_buf < self._active_stage_count_buf
        rx1 = rx.squeeze(-1)
        ry1 = ry.squeeze(-1)
        bx1 = bx.squeeze(-1)
        by1 = by.squeeze(-1)
        _, _, robot_yaw = self._robot_pose()

        target_x, target_y = self._compute_stage_target(stage_idx, valid, rx, ry, bx, by)
        robot_dist = torch.hypot(rx1 - target_x, ry1 - target_y)
        push_stage = valid & self._stage_push[stage_idx]
        dist_req = self._stage_dist[stage_idx]
        box_prog = self._box_axis_progress(stage_idx, valid, bx, by)
        reached_push = push_stage & (box_prog >= (dist_req - self._push_progress_tol))

        stop_x = self._stage_box_x_stop[stage_idx]
        stop_tol = self._stage_stop_tol[stage_idx]
        use_box_x_stop = valid & (~torch.isnan(stop_x)) & (~torch.isnan(stop_tol))
        sign = self._stage_sign[stage_idx]
        reached_box_x_stop = torch.zeros_like(valid)
        if bool(use_box_x_stop.any()):
            reached_pos = bx1 >= (stop_x - stop_tol)
            reached_neg = bx1 <= (stop_x + stop_tol)
            reached_at_stop = torch.where(sign > 0, reached_pos, reached_neg)
            reached_box_x_stop = use_box_x_stop & reached_at_stop

        x_align, y_align, use_x_match, use_y_match = self._match_box_alignment_ok(
            stage_idx, valid, rx1, ry1, bx1, by1
        )
        match_x_reached = self._match_box_held_reached(x_align, use_x_match, stage_idx)
        match_y_reached = self._match_box_held_reached(y_align, use_y_match, stage_idx)

        push_reached = reached_push | reached_box_x_stop
        reached = torch.where(
            push_stage,
            push_reached,
            torch.where(
                use_x_match,
                match_x_reached,
                torch.where(
                    use_y_match,
                    match_y_reached,
                    valid & (robot_dist <= self._stage_reach_tol),
                ),
            ),
        )
        approach_y = self._stage_approach_y[stage_idx]
        use_approach_y = valid & (~torch.isnan(approach_y))
        if bool(use_approach_y.any()):
            y_ok = (ry1 - approach_y).abs() <= self._stage_reach_tol
            reached = torch.where(use_approach_y, reached & y_ok, reached)
        face_tol = self._stage_face_box_tol[stage_idx]
        use_face = valid & (~torch.isnan(face_tol))
        if bool(use_face.any()):
            target_yaw = torch.atan2(by.squeeze(-1) - ry1, bx1 - rx1)
            yaw_err = torch.atan2(
                torch.sin(robot_yaw.squeeze(-1) - target_yaw),
                torch.cos(robot_yaw.squeeze(-1) - target_yaw),
            ).abs()
            face_ok = yaw_err <= face_tol
            reached = torch.where(use_face, reached & face_ok, reached)

        if box_z is not None:
            drop_reached, _ = self._compute_drop_wait_reached(valid, stage_idx, box_z)
            reached = reached | (valid & drop_reached)

        dist_to_target = self._task_progress_distance(stage_idx, valid, rx, ry, bx, by)
        self._stage_progress_buf = dist_to_target
        return reached, dist_to_target

    def _compute_push_x_align_reward(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        bx: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        align_stage = valid & (stage_idx == self._idx_sidestep_right)
        x_err = (rx.squeeze(-1) - bx.squeeze(-1)).abs()
        prev_init = torch.isnan(self._prev_x_align_err)
        self._prev_x_align_err = torch.where(prev_init, x_err, self._prev_x_align_err)
        delta_x = torch.clamp(
            self._prev_x_align_err - x_err,
            -self._push_x_align_delta_clip,
            self._push_x_align_delta_clip,
        )
        align_dense = self._w_push_x_align * delta_x
        reset_prev = done_now | reached
        self._prev_x_align_err = torch.where(
            reset_prev,
            torch.full_like(self._prev_x_align_err, float("nan")),
            x_err,
        )
        return torch.where(align_stage, align_dense, torch.zeros_like(align_dense))

    def _compute_push_y_align_reward(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        ry: torch.Tensor,
        by: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        align_stage = valid & (stage_idx == self._idx_advance2)
        y_err = (ry.squeeze(-1) - by.squeeze(-1)).abs()
        prev_init = torch.isnan(self._prev_y_align_err)
        self._prev_y_align_err = torch.where(prev_init, y_err, self._prev_y_align_err)
        delta_y = torch.clamp(
            self._prev_y_align_err - y_err,
            -self._push_y_align_delta_clip,
            self._push_y_align_delta_clip,
        )
        align_dense = self._w_push_y_align * delta_y
        reset_prev = done_now | reached
        self._prev_y_align_err = torch.where(
            reset_prev,
            torch.full_like(self._prev_y_align_err, float("nan")),
            y_err,
        )
        return torch.where(align_stage, align_dense, torch.zeros_like(align_dense))

    def _compute_face_box_reward(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
        robot_yaw: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        face_tol = self._stage_face_box_tol[stage_idx]
        face_stage = valid & (~torch.isnan(face_tol))
        target_yaw = torch.atan2(by.squeeze(-1) - ry.squeeze(-1), bx.squeeze(-1) - rx.squeeze(-1))
        yaw_err = torch.atan2(
            torch.sin(robot_yaw.squeeze(-1) - target_yaw),
            torch.cos(robot_yaw.squeeze(-1) - target_yaw),
        ).abs()
        prev_init = torch.isnan(self._prev_face_box_yaw_err)
        self._prev_face_box_yaw_err = torch.where(prev_init, yaw_err, self._prev_face_box_yaw_err)
        delta_yaw = torch.clamp(
            self._prev_face_box_yaw_err - yaw_err,
            -self._face_box_delta_clip,
            self._face_box_delta_clip,
        )
        face_dense = self._w_face_box * delta_yaw
        reset_prev = done_now | reached
        self._prev_face_box_yaw_err = torch.where(
            reset_prev,
            torch.full_like(self._prev_face_box_yaw_err, float("nan")),
            yaw_err,
        )
        return torch.where(face_stage, face_dense, torch.zeros_like(face_dense))

    def _compute_push_box_target_reward(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        push_stage = valid & ((stage_idx == self._idx_sidestep_right) | (stage_idx == self._idx_advance2))
        _, _, target_x, target_y = self._compute_stage_segment_endpoints(stage_idx, valid, rx, ry, bx, by)
        box_dist = torch.hypot(bx.squeeze(-1) - target_x, by.squeeze(-1) - target_y)
        prev_init = torch.isnan(self._prev_box_target_dist)
        self._prev_box_target_dist = torch.where(prev_init, box_dist, self._prev_box_target_dist)
        delta = torch.clamp(
            self._prev_box_target_dist - box_dist,
            -self._push_box_target_delta_clip,
            self._push_box_target_delta_clip,
        )
        dense = self._w_push_box_target * delta
        reset_prev = done_now | reached
        self._prev_box_target_dist = torch.where(
            reset_prev,
            torch.full_like(self._prev_box_target_dist, float("nan")),
            box_dist,
        )
        return torch.where(push_stage, dense, torch.zeros_like(dense))

    def _compute_match_box_hold_reward(
        self,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> torch.Tensor:
        """Per-nav-step bonus for holding box alignment on match_box_hold stages."""
        if self._stage_idx_buf is None:
            return torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        valid = self._stage_idx_buf < self._active_stage_count_buf
        hold_s = self._stage_match_box_hold_s[stage_idx]
        hold_s = self._stage_match_box_hold_s[stage_idx]
        use_hold = valid & (~torch.isnan(hold_s))
        if not bool(use_hold.any()):
            return torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)

        rx1 = rx.squeeze(-1)
        ry1 = ry.squeeze(-1)
        bx1 = bx.squeeze(-1)
        by1 = by.squeeze(-1)
        x_align, y_align, use_x_match, use_y_match = self._match_box_alignment_ok(
            stage_idx, valid, rx1, ry1, bx1, by1
        )
        aligned = (use_x_match & x_align) | (use_y_match & y_align)

        robot = self.env.unwrapped.scene["robot"]
        speed_xy = torch.linalg.norm(
            robot.data.root_lin_vel_w.to(device=self._device, dtype=torch.float32)[:, :2],
            dim=1,
        )
        slow = speed_xy <= self._match_box_hold_speed_max
        active = use_hold & aligned & slow
        return torch.where(
            active,
            torch.full((self.num_envs,), self._w_match_box_hold, device=self._device, dtype=torch.float32),
            torch.zeros(self.num_envs, device=self._device, dtype=torch.float32),
        )

    def _compute_reward(
        self,
        dist_to_target: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        valid = self._stage_idx_buf < self._active_stage_count_buf
        _, _, robot_yaw = self._robot_pose()
        prev_init = torch.isnan(self._prev_dist_to_target)
        self._prev_dist_to_target = torch.where(prev_init, dist_to_target, self._prev_dist_to_target)
        delta_dist = torch.clamp(
            self._prev_dist_to_target - dist_to_target,
            -self._nav_dist_delta_clip,
            self._nav_dist_delta_clip,
        )
        dense = self._w_nav_dist * delta_dist
        dense = dense + self._compute_push_x_align_reward(stage_idx, valid, rx, bx, done_now, reached)
        dense = dense + self._compute_push_y_align_reward(stage_idx, valid, ry, by, done_now, reached)
        dense = dense + self._compute_face_box_reward(stage_idx, valid, rx, ry, bx, by, robot_yaw, done_now, reached)
        dense = dense + self._compute_push_box_target_reward(stage_idx, valid, rx, ry, bx, by, done_now, reached)
        reward = torch.where(valid, dense, torch.zeros_like(dense))
        reset_prev = done_now | reached
        self._prev_dist_to_target = torch.where(
            reset_prev,
            torch.full_like(self._prev_dist_to_target, float("nan")),
            dist_to_target,
        )
        zero_time = torch.zeros_like(dense)
        return reward, dense, zero_time

    def _stage_sparse_bonus(self, reached: torch.Tensor) -> torch.Tensor:
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        final_reach = reached & (stage_idx == self._idx_final)
        bonus = torch.where(
            final_reach,
            torch.full((self.num_envs,), self._r_stage_complete_final, device=self._device, dtype=torch.float32),
            torch.full((self.num_envs,), self._r_stage_complete, device=self._device, dtype=torch.float32),
        )
        bonus = torch.where(reached, bonus, torch.zeros_like(bonus))
        return bonus

    def _on_after_physics_step(self) -> None:
        """Hook after each low-level env.step (override in subclasses)."""
        return

    def step(self, nav_action: torch.Tensor):
        if not isinstance(nav_action, torch.Tensor):
            nav_action = torch.as_tensor(nav_action, dtype=torch.float32)
        nav_action = nav_action.to(self._device, dtype=torch.float32)
        if nav_action.ndim == 1:
            nav_action = nav_action.unsqueeze(0)
        self._ensure_state_buffers(self.num_envs)
        self._nav_step_count_buf += 1
        self._update_curriculum()
        self._stage_advanced_this_nav.fill_(False)

        total_reward = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_dense = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_sparse = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        total_time_pen = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        terminated = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
        last_info = {}
        episode_log: dict = {}
        # Debug snapshots for env-0 logging.
        dbg_rx0 = float("nan")
        dbg_ry0 = float("nan")
        dbg_tx0 = float("nan")
        dbg_ty0 = float("nan")
        dbg_sx0 = float("nan")
        dbg_sy0 = float("nan")
        dbg_ex0 = float("nan")
        dbg_ey0 = float("nan")
        dbg_dseg0 = float("nan")
        dbg_done_reason0 = "none"

        for _ in range(self.inner_steps):
            rx, ry, _ = self._robot_pose()
            bx, by, _ = self._box_pose()
            self._sync_nav_stage_idx(rx, ry, bx, by)
            # Snapshot values exactly at the moment used by base-env termination checks.
            stage_idx_dbg_pre = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
            valid_dbg_pre = self._stage_idx_buf < self._active_stage_count_buf
            sx_pre, sy_pre, ex_pre, ey_pre = self._compute_stage_segment_endpoints(stage_idx_dbg_pre, valid_dbg_pre, rx, ry, bx, by)
            tx_pre, ty_pre = self._compute_stage_target(stage_idx_dbg_pre, valid_dbg_pre, rx, ry, bx, by)
            vx_pre = ex_pre - sx_pre
            vy_pre = ey_pre - sy_pre
            wx_pre = rx.squeeze(-1) - sx_pre
            wy_pre = ry.squeeze(-1) - sy_pre
            seg_len2_pre = vx_pre * vx_pre + vy_pre * vy_pre
            t_pre = torch.where(seg_len2_pre > 1.0e-6, (wx_pre * vx_pre + wy_pre * vy_pre) / seg_len2_pre, torch.zeros_like(seg_len2_pre))
            t_pre = torch.clamp(t_pre, 0.0, 1.0)
            proj_x_pre = sx_pre + t_pre * vx_pre
            proj_y_pre = sy_pre + t_pre * vy_pre
            dseg_pre = torch.hypot(rx.squeeze(-1) - proj_x_pre, ry.squeeze(-1) - proj_y_pre)
            dbg_rx0 = float(rx[0, 0].item())
            dbg_ry0 = float(ry[0, 0].item())
            dbg_tx0 = float(tx_pre[0].item())
            dbg_ty0 = float(ty_pre[0].item())
            dbg_sx0 = float(sx_pre[0].item())
            dbg_sy0 = float(sy_pre[0].item())
            dbg_ex0 = float(ex_pre[0].item())
            dbg_ey0 = float(ey_pre[0].item())
            dbg_dseg0 = float(dseg_pre[0].item())
            ll_obs = self._build_ll_obs(self._current_obs, nav_action)
            with torch.inference_mode():
                ll_act = self.ll_policy(ll_obs)
            env_action = self._build_env_action(ll_act)
            obs, base_rew, term, trunc, info = self.env.step(env_action)
            self._current_obs = obs
            self._on_after_physics_step()

            step_term = term if isinstance(term, torch.Tensor) else torch.as_tensor(term, device=self._device)
            step_trunc = trunc if isinstance(trunc, torch.Tensor) else torch.as_tensor(trunc, device=self._device)
            step_term_1d = step_term.squeeze(-1).bool() if step_term.ndim > 1 else step_term.bool()
            step_trunc_1d = step_trunc.squeeze(-1).bool() if step_trunc.ndim > 1 else step_trunc.bool()
            done_now = step_term_1d | step_trunc_1d
            terminated |= step_term_1d
            truncated |= step_trunc_1d
            last_info = info
            if done_now.any():
                step_log = info.get("log", {}) if isinstance(info, dict) else {}
                if step_log:
                    episode_log = step_log

            rx, ry, _ = self._robot_pose()
            bx, by, _ = self._box_pose()
            box_z = self.env.unwrapped.scene["box"].data.root_pos_w.to(device=self._device, dtype=torch.float32)[:, 2:3]
            if done_now.any():
                self._count_done_terms(done_now)
                if bool(done_now[0].item()):
                    term_flags = self._termination_term_flags()
                    reasons0: list[str] = []
                    for name in ("time_out", "fall", "x_reached", "stage_target_deviation", "trajectory_deviation", "no_motion_timeout", "no_target_progress_timeout"):
                        if name in term_flags and bool(term_flags[name][0].item()):
                            reasons0.append(name)
                    if bool(step_trunc_1d[0].item()):
                        reasons0.append("trunc")
                    if not reasons0 and bool(step_term_1d[0].item()):
                        reasons0.append("term_unknown")
                    dbg_done_reason0 = "|".join(reasons0) if reasons0 else "none"
            self._done_total += int(done_now.sum().item())

            reached, dist_to_target = self._compute_stage_reached(rx, ry, bx, by, box_z=box_z)
            reached = reached & (~self._stage_advanced_this_nav)
            shaped, dense, time_pen = self._compute_reward(dist_to_target, done_now, reached, rx, ry, bx, by)
            total_reward += shaped
            total_dense += dense
            total_time_pen += time_pen

            if bool(reached.any()):
                stage_idx_now = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
                self._update_stage_anchors(reached, stage_idx_now, rx, ry, bx, by)
                stage_bonus = self._stage_sparse_bonus(reached)
                total_reward = total_reward + stage_bonus
                total_sparse += stage_bonus
                rx1 = rx.squeeze(-1)
                ry1 = ry.squeeze(-1)
                self._stage_origin_x_buf = torch.where(reached, rx1, self._stage_origin_x_buf)
                self._stage_origin_y_buf = torch.where(reached, ry1, self._stage_origin_y_buf)
                self._box_stage_origin_x_buf = torch.where(
                    reached,
                    torch.full_like(self._box_stage_origin_x_buf, float("nan")),
                    self._box_stage_origin_x_buf,
                )
                self._box_stage_origin_y_buf = torch.where(
                    reached,
                    torch.full_like(self._box_stage_origin_y_buf, float("nan")),
                    self._box_stage_origin_y_buf,
                )
                self._stage_idx_buf = torch.where(reached, self._stage_idx_buf + 1, self._stage_idx_buf)
                self._stage_advanced_this_nav = torch.where(reached, torch.ones_like(self._stage_advanced_this_nav), self._stage_advanced_this_nav)
                self._sync_nav_stage_idx(rx, ry, bx, by)
                self._prev_rel_progress = torch.where(
                    reached, torch.full_like(self._prev_rel_progress, float("nan")), self._prev_rel_progress
                )

            self._prev_robot_x, self._prev_robot_y = rx.clone(), ry.clone()
            self._prev_box_x, self._prev_box_y = bx.clone(), by.clone()

            self._reset_env_state(done_now, rx, ry, bx, by)

        hold_rew = self._compute_match_box_hold_reward(rx, ry, bx, by)
        total_reward = total_reward + hold_rew
        total_dense += hold_rew

        nav_step_penalty = torch.full_like(total_reward, self._r_nav_step_penalty)
        total_reward = total_reward - nav_step_penalty
        total_time_pen = total_time_pen + nav_step_penalty

        self._ep_return_buf += total_reward.detach()
        self._ep_dense_buf += total_dense.detach()
        self._ep_sparse_buf += total_sparse.detach()
        self._ep_nav_steps_buf += 1

        episode_done = terminated | truncated
        if bool(episode_done.any()):
            self._ep_return_buf[episode_done] = 0.0
            self._ep_dense_buf[episode_done] = 0.0
            self._ep_sparse_buf[episode_done] = 0.0
            self._ep_nav_steps_buf[episode_done] = 0

        idx0 = int(self._stage_idx_buf[0].item())
        active0 = int(self._active_stage_count_buf[0].item())
        level0 = int(self._curriculum_level_buf[0].item())
        nav0 = int(self._nav_step_count_buf[0].item())
        stage_name = self._stage_names[idx0] if idx0 < self._num_stages else "done"
        if self._nav_log_interval > 0 and nav0 % self._nav_log_interval == 0:
            denom = max(1, self._done_total)
            print(
                f"[{self._nav_log_tag}] nav={nav0:5d} curriculum={level0} "
                f"stage={idx0 + 1}/{self._num_stages}({stage_name}) active={active0} "
                f"prog={float(self._stage_progress_buf[0].item()):.2f} rew={total_reward.mean().item():+.4f} "
                f"nav[dense/sparse/time_pen]={total_dense.mean().item():+.3f}/"
                f"{total_sparse.mean().item():+.3f}/{total_time_pen.mean().item():+.3f} "
                f"ep_ret_running={self._ep_return_buf.mean().item():+.3f} "
                f"pos=({dbg_rx0:+.2f},{dbg_ry0:+.2f}) tgt=({dbg_tx0:+.2f},{dbg_ty0:+.2f}) "
                f"seg=({dbg_sx0:+.2f},{dbg_sy0:+.2f})->({dbg_ex0:+.2f},{dbg_ey0:+.2f}) dseg={dbg_dseg0:.2f} "
                f"why0={dbg_done_reason0} "
                f"done[f/t/x/stg/nm/np]={self._done_fall}/{self._done_timeout}/{self._done_x_reached}/"
                f"{self._done_stage_target}/{self._done_no_motion}/{self._done_no_target_progress} "
                f"ratio={self._done_fall/denom:.2f}/{self._done_timeout/denom:.2f}/"
                f"{self._done_x_reached/denom:.2f}/{self._done_stage_target/denom:.2f}/"
                f"{self._done_no_motion/denom:.2f}/{self._done_no_target_progress/denom:.2f}",
                flush=True,
            )

        obs_dict = self._obs_dict(self._current_obs)
        last_info = dict(last_info) if isinstance(last_info, dict) else {}
        if episode_log:
            last_info["log"] = episode_log
        last_info["teacher_stage"] = stage_name
        last_info["teacher_stage_idx"] = idx0
        last_info["teacher_curriculum_level"] = level0
        last_info["teacher_stage_progress"] = float(self._stage_progress_buf[0].item())
        return obs_dict, total_reward, terminated, truncated, last_info
