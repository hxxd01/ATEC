"""Task D teacher wrapper with asymmetric actor/critic observations."""

from __future__ import annotations

import math
import gymnasium as gym
import numpy as np
import torch

from atec_rl_lab.tasks.task_d.mdp.env_origin import TASK_D_PIT_TERRAIN_ORIGIN_XY

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
        elif spec.get("push_combined"):
            x += float(spec.get("push_forward_dist", 4.0))
            y -= float(spec.get("push_right_dist", 2.0))
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
        push_box_drop_com_z: float = 0.295,
        adjust_box_behind_x: float = 0.4,
        adjust_box_z_settle_eps: float = 1.0e-3,
        w_adjust_yaw_face_x: float = 2.0,
        adjust_yaw_delta_clip: float = 0.05,
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
                name="push",
                axis="y",
                sign=-1,
                dist=2.0,
                push=True,
                push_combined=True,
                push_right_dist=2.0,
                push_forward_dist=4.0,
                sparse_bonus=1.6,
                relative_robot_target=True,
            ),
            dict(
                name="adjust",
                axis="x",
                sign=0.0,
                dist=0.0,
                push=False,
                sparse_bonus=0.8,
                relative_robot_target=True,
                follow_box_target=True,
                yaw_face_plus_x=True,
                box_behind_x=0.4,
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
        self._stage_follow_box_target = torch.tensor(
            [bool(spec.get("follow_box_target", False)) for spec in self.stage_specs],
            device=self._device,
            dtype=torch.bool,
        )
        self._stage_yaw_face_plus_x = torch.tensor(
            [bool(spec.get("yaw_face_plus_x", False)) for spec in self.stage_specs],
            device=self._device,
            dtype=torch.bool,
        )
        self._stage_box_behind_x = torch.tensor(
            [
                float(spec.get("box_behind_x"))
                if spec.get("box_behind_x") is not None
                else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )
        self._stage_sign = torch.tensor([float(spec["sign"]) for spec in self.stage_specs], device=self._device, dtype=torch.float32)
        self._stage_dist = torch.tensor([float(spec["dist"]) for spec in self.stage_specs], device=self._device, dtype=torch.float32)
        self._stage_push_right_dist = torch.tensor(
            [
                float(spec["push_right_dist"]) if spec.get("push_right_dist") is not None else float("nan")
                for spec in self.stage_specs
            ],
            device=self._device,
            dtype=torch.float32,
        )

        self._stage_idx_buf: torch.Tensor | None = None
        # Per-env cumulative nav reward while in each stage (current episode).
        self._ep_stage_reward_buf: torch.Tensor | None = None
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
        self._done_no_push_progress = 0
        # Per-stage episode-end counts (last index = all stages completed).
        self._done_stage_counts: list[int] = [0] * (self._num_stages + 1)

        # Stage indices: retreat → sidestep_left → push → adjust → final.
        self._idx_retreat = 0
        self._idx_sidestep_left = 1
        self._idx_push = 2
        self._idx_adjust = 3
        self._idx_final = 4
        self._push_box_drop_z = float(push_box_drop_com_z)
        self._adjust_box_behind_x = float(adjust_box_behind_x)
        self._adjust_box_z_settle_eps = float(adjust_box_z_settle_eps)
        self._w_adjust_yaw_face_x = float(w_adjust_yaw_face_x)
        self._adjust_yaw_delta_clip = float(adjust_yaw_delta_clip)

        # Reward: progress toward stage target (bounded) + sparse bonus on reach.
        self._w_nav_dist = 3.0
        self._nav_dist_delta_clip = 0.05
        # Distance tolerance for stage completion (non-push stages).
        self._stage_reach_tol = 0.35
        self._r_stage_complete = 1.0
        self._r_stage_complete_final = 2.0
        # Small per-step penalty to encourage reaching targets quickly.
        self._r_step_penalty = -0.01
        # Push stages: box axis progress (solution-style) + approach when no contact.
        self._w_push_box_axis = 4.0
        self._push_box_axis_delta_clip = 0.05
        self._w_approach_box = 3.0
        self._approach_box_delta_clip = 0.05
        self._contact_force_thresh = 2.0
        # approach_box: reward turning to face the box (progress-based on yaw error).
        self._w_face_box = 2.0
        self._face_box_delta_clip = 0.05
        # final: reward robot x increasing after box has landed.
        self._w_final_robot_x = 4.0
        self._final_robot_x_delta_clip = 0.05

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
        self._bind_env_origins()
        self._prev_robot_s = None
        self._prev_rel_progress = None
        self._prev_dist_to_target: torch.Tensor | None = None
        self._prev_box_axis_progress: torch.Tensor | None = None
        self._prev_push_right_progress: torch.Tensor | None = None
        self._prev_push_forward_progress: torch.Tensor | None = None
        self._prev_final_robot_x_progress: torch.Tensor | None = None
        self._prev_robot_box_dist: torch.Tensor | None = None
        self._box_push_origin_x_buf: torch.Tensor | None = None
        self._box_push_origin_y_buf: torch.Tensor | None = None
        self._prev_sync_stage_idx: torch.Tensor | None = None
        self._prev_face_box_yaw_err: torch.Tensor | None = None
        self._prev_adjust_yaw_err: torch.Tensor | None = None
        self._prev_box_com_z: torch.Tensor | None = None

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
            f"{self._curriculum_mid_nav_steps}] stages={self._num_stages} "
            f"push_drop_com_z<{self._push_box_drop_z:.3f} adjust_behind_x={self._adjust_box_behind_x:.2f} "
            f"nav_log_interval={self._nav_log_interval}",
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
        if hasattr(box.data, "root_com_pos_w"):
            pos = box.data.root_com_pos_w.to(device=self._device, dtype=torch.float32)
        else:
            pos = box.data.root_pos_w.to(device=self._device, dtype=torch.float32)
        quat = box.data.root_quat_w.to(device=self._device, dtype=torch.float32)
        return pos[:, 0:1], pos[:, 1:2], pos[:, 2:3], self._yaw_from_quat_wxyz(quat)

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

    def _bind_env_origins(self) -> None:
        """Cache terrain env_origins for per-tile world-frame stage targets and thresholds."""
        base = self.env.unwrapped
        origins = base.scene.env_origins.to(device=self._device, dtype=torch.float32)
        if int(origins.shape[0]) != int(self.num_envs):
            raise ValueError(
                f"scene.env_origins has {int(origins.shape[0])} entries but num_envs={int(self.num_envs)}"
            )
        self._env_origin_x = origins[:, 0].clone()
        self._env_origin_y = origins[:, 1].clone()
        self._pit_ref_ox, self._pit_ref_oy = TASK_D_PIT_TERRAIN_ORIGIN_XY

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
        self._ep_stage_reward_buf = torch.zeros(
            batch, self._num_stages, device=self._device, dtype=torch.float32
        )
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
        self._prev_robot_s = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_rel_progress = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_dist_to_target = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_box_axis_progress = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_push_right_progress = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_push_forward_progress = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_final_robot_x_progress = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_robot_box_dist = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._box_push_origin_x_buf = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._box_push_origin_y_buf = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_sync_stage_idx = torch.full((batch,), -1, device=self._device, dtype=torch.long)
        self._prev_face_box_yaw_err = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_adjust_yaw_err = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)
        self._prev_box_com_z = torch.full((batch,), float("nan"), device=self._device, dtype=torch.float32)

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
        self._prev_robot_x[reset_mask] = rx[reset_mask]
        self._prev_robot_y[reset_mask] = ry[reset_mask]
        self._prev_box_x[reset_mask] = bx[reset_mask]
        self._prev_box_y[reset_mask] = by[reset_mask]
        self._prev_robot_s[reset_mask] = float("nan")
        self._prev_rel_progress[reset_mask] = float("nan")
        self._prev_dist_to_target[reset_mask] = float("nan")
        self._prev_box_axis_progress[reset_mask] = float("nan")
        self._prev_push_right_progress[reset_mask] = float("nan")
        self._prev_push_forward_progress[reset_mask] = float("nan")
        self._prev_final_robot_x_progress[reset_mask] = float("nan")
        self._prev_robot_box_dist[reset_mask] = float("nan")
        self._box_push_origin_x_buf[reset_mask] = float("nan")
        self._box_push_origin_y_buf[reset_mask] = float("nan")
        self._prev_sync_stage_idx[reset_mask] = -1
        self._prev_face_box_yaw_err[reset_mask] = float("nan")
        self._prev_adjust_yaw_err[reset_mask] = float("nan")
        self._prev_box_com_z[reset_mask] = float("nan")
        if self._ep_stage_reward_buf is not None:
            self._ep_stage_reward_buf[reset_mask] = 0.0

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
        bx, by, _, box_yaw = self._box_pose()
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
        bx, by, _, _ = self._box_pose()
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
        for name in (
            "fall",
            "x_reached",
            "time_out",
            "stage_target_deviation",
            "trajectory_deviation",
            "no_motion_timeout",
            "no_target_progress_timeout",
            "no_push_progress_timeout",
        ):
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
            if "no_push_progress_timeout" in term_flags:
                self._done_no_push_progress += int((done_now & term_flags["no_push_progress_timeout"]).sum().item())
            return

        # Fallback when termination manager is unavailable.
        rx, _, _ = self._robot_pose()
        x_reached_now = done_now & (
            rx.squeeze(-1) > (3.5 + self._env_origin_x - self._pit_ref_ox)
        )
        self._done_x_reached += int(x_reached_now.sum().item())
        self._done_fall += int((done_now & (~x_reached_now)).sum().item())

    def _count_done_stages(self, done_now: torch.Tensor) -> None:
        if not bool(done_now.any()):
            return
        stage_idx = torch.clamp(self._stage_idx_buf[done_now], min=0, max=self._num_stages)
        for i in range(self._num_stages):
            self._done_stage_counts[i] += int((stage_idx == i).sum().item())
        self._done_stage_counts[self._num_stages] += int((stage_idx >= self._num_stages).sum().item())

    def _stage_at_done_episode_log(self, done_now: torch.Tensor) -> dict[str, float]:
        """Fraction of terminating envs per nav stage (for rsl_rl Episode_Termination_Stage/*)."""
        total = int(done_now.sum().item())
        if total <= 0:
            return {}
        stage_idx = torch.clamp(self._stage_idx_buf[done_now], min=0, max=self._num_stages)
        out: dict[str, float] = {}
        for i, name in enumerate(self._stage_names):
            out[f"Episode_Termination_Stage/{name}"] = float((stage_idx == i).sum().item()) / total
        finished = int((stage_idx >= self._num_stages).sum().item())
        if finished > 0:
            out["Episode_Termination_Stage/finished"] = finished / total
        return out

    def _accumulate_episode_stage_reward(self, step_reward: torch.Tensor, active_mask: torch.Tensor) -> None:
        """Add per-step nav reward to the current stage bucket for each active env."""
        if self._ep_stage_reward_buf is None:
            return
        valid = self._stage_idx_buf < self._num_stages
        active = active_mask & valid
        if not bool(active.any()):
            return
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        rew = step_reward * active.to(dtype=step_reward.dtype)
        self._ep_stage_reward_buf.scatter_add_(1, stage_idx.unsqueeze(1), rew.unsqueeze(1))

    def _stage_reward_at_episode_log(self, done_now: torch.Tensor) -> dict[str, float]:
        """Mean per-stage cumulative episode reward over terminating envs (Episode_StageReward/*)."""
        if self._ep_stage_reward_buf is None:
            return {}
        total = int(done_now.sum().item())
        if total <= 0:
            return {}
        ep_rew = self._ep_stage_reward_buf[done_now]
        return {
            f"Episode_StageReward/{name}": float(ep_rew[:, i].mean().item())
            for i, name in enumerate(self._stage_names)
        }

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
        self._done_no_push_progress = 0
        self._done_stage_counts = [0] * (self._num_stages + 1)
        rx, ry, _ = self._robot_pose()
        bx, by, _, _ = self._box_pose()
        self._ensure_state_buffers(rx.shape[0])
        self._prev_robot_x, self._prev_robot_y = rx.clone(), ry.clone()
        self._prev_box_x, self._prev_box_y = bx.clone(), by.clone()
        self._bind_env_origins()
        full_reset = torch.ones(rx.shape[0], device=self._device, dtype=torch.bool)
        self._reset_env_state(full_reset, rx, ry, bx, by)
        bx, by, bz, _ = self._box_pose()
        self._sync_nav_stage_idx(rx, ry, bx, by, bz)
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

    def _relative_axis_progress(
        self,
        sign: torch.Tensor,
        coord_now: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        delta = coord_now - anchor
        signed = torch.where(sign > 0, delta, -delta)
        return torch.clamp(signed, min=0.0)

    def _main_push_stage_mask(self, stage_idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return valid & (stage_idx == self._idx_push)

    def _adjust_stage_mask(self, stage_idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return valid & (stage_idx == self._idx_adjust)

    def _final_stage_mask(self, stage_idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return valid & (stage_idx == self._idx_final)

    def _compute_final_robot_x_progress(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
    ) -> torch.Tensor:
        final = self._final_stage_mask(stage_idx, valid)
        sign = torch.full_like(rx.squeeze(-1), 1.0)
        raw = self._relative_axis_progress(sign, rx.squeeze(-1), self._stage_origin_x_buf)
        return torch.where(final, raw, torch.zeros_like(raw))

    def _compute_push_right_progress(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        by: torch.Tensor,
    ) -> torch.Tensor:
        push = self._main_push_stage_mask(stage_idx, valid)
        sign = torch.full_like(by.squeeze(-1), -1.0)
        raw = self._relative_axis_progress(sign, by.squeeze(-1), self._box_push_origin_y_buf)
        cap = self._stage_push_right_dist[stage_idx]
        capped = torch.clamp(raw, max=cap)
        return torch.where(push, capped, torch.zeros_like(capped))

    def _compute_push_forward_progress(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        bx: torch.Tensor,
    ) -> torch.Tensor:
        push = self._main_push_stage_mask(stage_idx, valid)
        sign = torch.full_like(bx.squeeze(-1), 1.0)
        raw = self._relative_axis_progress(sign, bx.squeeze(-1), self._box_push_origin_x_buf)
        return torch.where(push, raw, torch.zeros_like(raw))

    def _contact_on(self) -> torch.Tensor:
        cf = self.env.unwrapped.scene["contact_sensor"].data.net_forces_w
        return cf.norm(dim=-1).max(dim=1).values > self._contact_force_thresh

    def _ensure_box_push_origin(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> None:
        main_push = self._main_push_stage_mask(stage_idx, valid)
        if self._prev_sync_stage_idx is None:
            return
        stage_changed = stage_idx != self._prev_sync_stage_idx
        if bool(stage_changed.any()):
            self._box_push_origin_x_buf = torch.where(
                stage_changed,
                torch.full_like(self._box_push_origin_x_buf, float("nan")),
                self._box_push_origin_x_buf,
            )
            self._box_push_origin_y_buf = torch.where(
                stage_changed,
                torch.full_like(self._box_push_origin_y_buf, float("nan")),
                self._box_push_origin_y_buf,
            )
        origin_init = torch.isnan(self._box_push_origin_x_buf)
        arm = main_push & origin_init
        if bool(arm.any()):
            self._box_push_origin_x_buf = torch.where(arm, bx.squeeze(-1), self._box_push_origin_x_buf)
            self._box_push_origin_y_buf = torch.where(arm, by.squeeze(-1), self._box_push_origin_y_buf)

    def _publish_push_stuck_signals(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_box_push_origin(stage_idx, valid, bx, by)
        main_push = self._main_push_stage_mask(stage_idx, valid)
        right_progress = self._compute_push_right_progress(stage_idx, valid, by)
        forward_progress = self._compute_push_forward_progress(stage_idx, valid, bx)
        robot_box_dist = torch.hypot(rx.squeeze(-1) - bx.squeeze(-1), ry.squeeze(-1) - by.squeeze(-1))
        self._prev_sync_stage_idx = stage_idx.clone()
        return main_push, right_progress, forward_progress, robot_box_dist

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

    def _sync_nav_stage_idx(
        self, rx: torch.Tensor, ry: torch.Tensor, bx: torch.Tensor, by: torch.Tensor, bz: torch.Tensor
    ) -> None:
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

        push_stuck_active, push_right_progress, push_forward_progress, push_robot_box_dist = (
            self._publish_push_stuck_signals(stage_idx, valid, rx, ry, bx, by)
        )
        _, dist_to_target = self._compute_stage_reached(rx, ry, bx, by, bz)

        device = base.device
        float_attrs = (
            ("_nav_dist_to_target", dist_to_target),
            ("_nav_push_box_right_progress", push_right_progress),
            ("_nav_push_box_forward_progress", push_forward_progress),
            ("_nav_push_robot_box_dist", push_robot_box_dist),
        )
        for attr, val in float_attrs:
            if (
                not hasattr(base, attr)
                or not isinstance(getattr(base, attr), torch.Tensor)
                or int(getattr(base, attr).shape[0]) != int(self.num_envs)
            ):
                setattr(base, attr, torch.zeros(self.num_envs, device=device, dtype=torch.float32))
            getattr(base, attr).copy_(val.to(device=device, dtype=torch.float32))
        if (
            not hasattr(base, "_nav_push_stuck_active")
            or not isinstance(base._nav_push_stuck_active, torch.Tensor)
            or int(base._nav_push_stuck_active.shape[0]) != int(self.num_envs)
        ):
            base._nav_push_stuck_active = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
        base._nav_push_stuck_active.copy_(push_stuck_active.to(device=device, dtype=torch.bool))

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

        ox = self._env_origin_x
        oy = self._env_origin_y
        pit_ox = self._pit_ref_ox
        pit_oy = self._pit_ref_oy
        sx = self._stage_start_x[stage_idx] + ox - pit_ox
        sy = self._stage_start_y[stage_idx] + oy - pit_oy
        ex = self._stage_target_x[stage_idx] + ox - pit_ox
        ey = self._stage_target_y[stage_idx] + oy - pit_oy
        has_origin = (~torch.isnan(self._stage_origin_x_buf)) & (~torch.isnan(self._stage_origin_y_buf))

        # Only relative stages use previous-stage entry as segment start.
        # Early approach stages (retreat/sidestep_left) stay absolute.
        entry_start = valid & has_origin & self._stage_relative_target[stage_idx]
        sx = torch.where(entry_start, self._stage_origin_x_buf, sx)
        sy = torch.where(entry_start, self._stage_origin_y_buf, sy)

        approach_y = self._stage_approach_y[stage_idx]
        approach_stage = valid & (~torch.isnan(approach_y))
        if bool(approach_stage.any()):
            ex = torch.where(approach_stage, bx.squeeze(-1), ex)
            ey = torch.where(approach_stage, approach_y + oy - pit_oy, ey)

        match_box_y = valid & self._stage_match_box_y_target[stage_idx]
        if bool(match_box_y.any()):
            anchor_x = torch.where(has_origin, self._stage_origin_x_buf, rx1)
            anchor_y = torch.where(has_origin, self._stage_origin_y_buf, ry1)
            sx = torch.where(match_box_y, anchor_x, sx)
            sy = torch.where(match_box_y, anchor_y, sy)
            ex = torch.where(match_box_y, anchor_x, ex)
            ey = torch.where(match_box_y, by1, ey)

        follow_box = valid & self._stage_follow_box_target[stage_idx]
        if bool(follow_box.any()):
            behind_x = self._stage_box_behind_x[stage_idx]
            behind_x = torch.where(torch.isnan(behind_x), self._adjust_box_behind_x, behind_x)
            sx = torch.where(follow_box, rx1, sx)
            sy = torch.where(follow_box, ry1, sy)
            ex = torch.where(follow_box, bx.squeeze(-1) - behind_x, ex)
            ey = torch.where(follow_box, by1, ey)

        rel_stage = (
            valid
            & self._stage_relative_target[stage_idx]
            & (~self._stage_match_box_y_target[stage_idx])
            & (~self._stage_follow_box_target[stage_idx])
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

    def _compute_stage_reached(self, rx, ry, bx, by, bz) -> tuple[torch.Tensor, torch.Tensor]:
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        valid = self._stage_idx_buf < self._active_stage_count_buf
        rx1 = rx.squeeze(-1)
        ry1 = ry.squeeze(-1)
        pit_oy = self._pit_ref_oy
        _, _, robot_yaw = self._robot_pose()

        main_push = self._main_push_stage_mask(stage_idx, valid)
        adjust = self._adjust_stage_mask(stage_idx, valid)
        self._ensure_box_push_origin(stage_idx, valid, bx, by)
        right_progress = self._compute_push_right_progress(stage_idx, valid, by)
        forward_progress = self._compute_push_forward_progress(stage_idx, valid, bx)
        stage_dist = self._stage_dist[stage_idx]

        target_x, target_y = self._compute_stage_target(stage_idx, valid, rx, ry, bx, by)
        dist_to_target = torch.hypot(rx1 - target_x, ry1 - target_y)
        reached_nav = valid & (dist_to_target <= self._stage_reach_tol)
        reached_push = main_push & (bz.squeeze(-1) < self._push_box_drop_z)
        bz1 = bz.squeeze(-1)
        prev_bz = self._prev_box_com_z
        box_z_settled = (~torch.isnan(prev_bz)) & (
            (bz1 - prev_bz).abs() <= self._adjust_box_z_settle_eps
        )
        reached_adjust = adjust & reached_nav & box_z_settled
        reached = torch.where(
            main_push,
            reached_push,
            torch.where(adjust, reached_adjust, reached_nav),
        )

        # Keep important alignment constraints at stage completion (non-push stages).
        x_tol = self._stage_match_box_x_tol[stage_idx]
        use_x_tol = valid & (~main_push) & (~adjust) & (~torch.isnan(x_tol))
        if bool(use_x_tol.any()):
            x_ok = (rx.squeeze(-1) - bx.squeeze(-1)).abs() <= x_tol
            reached = torch.where(use_x_tol, reached & x_ok, reached)
        y_tol = self._stage_match_box_y_tol[stage_idx]
        use_y_tol = valid & (~main_push) & (~adjust) & (~torch.isnan(y_tol))
        if bool(use_y_tol.any()):
            y_ok = (ry.squeeze(-1) - by.squeeze(-1)).abs() <= y_tol
            reached = torch.where(use_y_tol, reached & y_ok, reached)
        approach_y = self._stage_approach_y[stage_idx]
        use_approach_y = valid & (~torch.isnan(approach_y))
        if bool(use_approach_y.any()):
            y_ok = (ry1 - (approach_y + self._env_origin_y - pit_oy)).abs() <= self._stage_reach_tol
            reached = torch.where(use_approach_y, reached & y_ok, reached)
        face_tol = self._stage_face_box_tol[stage_idx]
        use_face = valid & (~torch.isnan(face_tol))
        if bool(use_face.any()):
            target_yaw = torch.atan2(by.squeeze(-1) - ry1, bx.squeeze(-1) - rx1)
            yaw_err = torch.atan2(
                torch.sin(robot_yaw.squeeze(-1) - target_yaw),
                torch.cos(robot_yaw.squeeze(-1) - target_yaw),
            ).abs()
            face_ok = yaw_err <= face_tol
            reached = torch.where(use_face, reached & face_ok, reached)

        right_cap = self._stage_push_right_dist[stage_idx]
        right_norm = torch.where(
            right_cap > 1.0e-6,
            torch.clamp(right_progress / right_cap, 0.0, 1.0),
            torch.zeros_like(right_progress),
        )
        push_prog = 0.5 * right_norm + 0.5 * torch.clamp(forward_progress / torch.clamp(stage_dist, min=1.0e-6), 0.0, 1.0)
        final_mask = self._final_stage_mask(stage_idx, valid)
        final_x_prog = self._compute_final_robot_x_progress(stage_idx, valid, rx)
        final_prog = torch.where(
            stage_dist > 1.0e-6,
            torch.clamp(final_x_prog / stage_dist, 0.0, 1.0),
            torch.zeros_like(final_x_prog),
        )
        self._stage_progress_buf = torch.where(
            main_push,
            push_prog * stage_dist,
            torch.where(final_mask, final_prog * stage_dist, dist_to_target),
        )
        return reached, dist_to_target

    def _compute_adjust_yaw_face_x_reward(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        robot_yaw: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        adjust = self._adjust_stage_mask(stage_idx, valid)
        yaw_face = valid & self._stage_yaw_face_plus_x[stage_idx]
        active = adjust & yaw_face
        yaw_err = torch.atan2(
            torch.sin(robot_yaw.squeeze(-1)),
            torch.cos(robot_yaw.squeeze(-1)),
        ).abs()
        prev_init = torch.isnan(self._prev_adjust_yaw_err)
        self._prev_adjust_yaw_err = torch.where(prev_init, yaw_err, self._prev_adjust_yaw_err)
        delta_yaw = torch.clamp(
            self._prev_adjust_yaw_err - yaw_err,
            -self._adjust_yaw_delta_clip,
            self._adjust_yaw_delta_clip,
        )
        dense = self._w_adjust_yaw_face_x * delta_yaw
        reset_prev = done_now | reached
        self._prev_adjust_yaw_err = torch.where(
            reset_prev,
            torch.full_like(self._prev_adjust_yaw_err, float("nan")),
            yaw_err,
        )
        return torch.where(active, dense, torch.zeros_like(dense))

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

    def _compute_approach_box_reward(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
        contact: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        main_push = self._main_push_stage_mask(stage_idx, valid)
        approach_stage = main_push & (~contact)
        robot_box_dist = torch.hypot(rx.squeeze(-1) - bx.squeeze(-1), ry.squeeze(-1) - by.squeeze(-1))
        prev_init = torch.isnan(self._prev_robot_box_dist)
        self._prev_robot_box_dist = torch.where(prev_init, robot_box_dist, self._prev_robot_box_dist)
        delta = torch.clamp(
            self._prev_robot_box_dist - robot_box_dist,
            -self._approach_box_delta_clip,
            self._approach_box_delta_clip,
        )
        dense = self._w_approach_box * delta
        reset_prev = done_now | reached
        self._prev_robot_box_dist = torch.where(
            reset_prev,
            torch.full_like(self._prev_robot_box_dist, float("nan")),
            robot_box_dist,
        )
        return torch.where(approach_stage, dense, torch.zeros_like(dense))

    def _compute_push_dual_axis_progress_reward(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
        bz: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        main_push = self._main_push_stage_mask(stage_idx, valid)
        self._ensure_box_push_origin(stage_idx, valid, bx, by)
        right_progress = self._compute_push_right_progress(stage_idx, valid, by)
        forward_progress = self._compute_push_forward_progress(stage_idx, valid, bx)
        right_cap = self._stage_push_right_dist[stage_idx]
        right_active = main_push & (right_progress < right_cap - 1.0e-4)
        forward_active = main_push & (bz.squeeze(-1) >= self._push_box_drop_z)

        prev_r_init = torch.isnan(self._prev_push_right_progress)
        self._prev_push_right_progress = torch.where(
            prev_r_init, right_progress, self._prev_push_right_progress
        )
        prev_f_init = torch.isnan(self._prev_push_forward_progress)
        self._prev_push_forward_progress = torch.where(
            prev_f_init, forward_progress, self._prev_push_forward_progress
        )
        delta_r = torch.clamp(
            right_progress - self._prev_push_right_progress,
            -self._push_box_axis_delta_clip,
            self._push_box_axis_delta_clip,
        )
        delta_f = torch.clamp(
            forward_progress - self._prev_push_forward_progress,
            -self._push_box_axis_delta_clip,
            self._push_box_axis_delta_clip,
        )
        reset_prev = done_now | reached
        self._prev_push_right_progress = torch.where(
            reset_prev,
            torch.full_like(self._prev_push_right_progress, float("nan")),
            right_progress,
        )
        self._prev_push_forward_progress = torch.where(
            reset_prev,
            torch.full_like(self._prev_push_forward_progress, float("nan")),
            forward_progress,
        )
        dense = torch.where(right_active, self._w_push_box_axis * delta_r, torch.zeros_like(delta_r))
        dense = dense + torch.where(forward_active, self._w_push_box_axis * delta_f, torch.zeros_like(delta_f))
        return dense

    def _compute_final_robot_x_progress_reward(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
    ) -> torch.Tensor:
        final = self._final_stage_mask(stage_idx, valid)
        x_progress = self._compute_final_robot_x_progress(stage_idx, valid, rx)
        prev_init = torch.isnan(self._prev_final_robot_x_progress)
        self._prev_final_robot_x_progress = torch.where(
            prev_init, x_progress, self._prev_final_robot_x_progress
        )
        delta_x = torch.clamp(
            x_progress - self._prev_final_robot_x_progress,
            -self._final_robot_x_delta_clip,
            self._final_robot_x_delta_clip,
        )
        dense = self._w_final_robot_x * delta_x
        reset_prev = done_now | reached
        self._prev_final_robot_x_progress = torch.where(
            reset_prev,
            torch.full_like(self._prev_final_robot_x_progress, float("nan")),
            x_progress,
        )
        return torch.where(final, dense, torch.zeros_like(dense))

    def _compute_reward(
        self,
        dist_to_target: torch.Tensor,
        done_now: torch.Tensor,
        reached: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
        bz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        valid = self._stage_idx_buf < self._active_stage_count_buf
        _, _, robot_yaw = self._robot_pose()
        main_push = self._main_push_stage_mask(stage_idx, valid)
        contact = self._contact_on()

        prev_init = torch.isnan(self._prev_dist_to_target)
        self._prev_dist_to_target = torch.where(prev_init, dist_to_target, self._prev_dist_to_target)
        delta_dist = torch.clamp(
            self._prev_dist_to_target - dist_to_target,
            -self._nav_dist_delta_clip,
            self._nav_dist_delta_clip,
        )
        nav_dense = self._w_nav_dist * delta_dist
        final_mask = self._final_stage_mask(stage_idx, valid)
        skip_nav_dist = main_push | final_mask
        dense = torch.where(skip_nav_dist, torch.zeros_like(nav_dense), nav_dense)
        dense = dense + self._compute_approach_box_reward(
            stage_idx, valid, rx, ry, bx, by, contact, done_now, reached
        )
        dense = dense + self._compute_push_dual_axis_progress_reward(
            stage_idx, valid, bx, by, bz, done_now, reached
        )
        dense = dense + self._compute_adjust_yaw_face_x_reward(
            stage_idx, valid, robot_yaw, done_now, reached
        )
        dense = dense + self._compute_face_box_reward(stage_idx, valid, rx, ry, bx, by, robot_yaw, done_now, reached)
        dense = dense + self._compute_final_robot_x_progress_reward(
            stage_idx, valid, rx, done_now, reached
        )
        step_penalty = torch.full_like(dense, self._r_step_penalty)
        reward = torch.where(valid, dense + step_penalty, torch.zeros_like(dense))
        reset_prev = done_now | reached
        self._prev_dist_to_target = torch.where(
            reset_prev,
            torch.full_like(self._prev_dist_to_target, float("nan")),
            dist_to_target,
        )
        return reward, dense, torch.where(valid, step_penalty, torch.zeros_like(step_penalty))

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

        # Per-env mask: once Isaac resets an env mid-inner-loop, do not mix the next
        # episode's reward/stats into this nav-step return (rsl_rl sees one transition).
        episode_done = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)

        for _ in range(self.inner_steps):
            active = ~episode_done
            if not bool(active.any()):
                break

            rx, ry, _ = self._robot_pose()
            bx, by, bz, _ = self._box_pose()
            self._sync_nav_stage_idx(rx, ry, bx, by, bz)
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
            new_done = done_now & active
            episode_done |= done_now
            terminated |= step_term_1d
            truncated |= step_trunc_1d
            last_info = info
            if new_done.any():
                step_log = info.get("log", {}) if isinstance(info, dict) else {}
                merged_log = dict(episode_log)
                if step_log:
                    merged_log.update(step_log)
                merged_log.update(self._stage_at_done_episode_log(new_done))
                episode_log = merged_log

            rx, ry, _ = self._robot_pose()
            bx, by, bz, _ = self._box_pose()
            if new_done.any():
                self._count_done_terms(new_done)
                self._count_done_stages(new_done)
                if bool(new_done[0].item()):
                    term_flags = self._termination_term_flags()
                    reasons0: list[str] = []
                    for name in (
                        "time_out",
                        "fall",
                        "x_reached",
                        "stage_target_deviation",
                        "trajectory_deviation",
                        "no_motion_timeout",
                        "no_target_progress_timeout",
                        "no_push_progress_timeout",
                    ):
                        if name in term_flags and bool(term_flags[name][0].item()):
                            reasons0.append(name)
                    if bool(step_trunc_1d[0].item()):
                        reasons0.append("trunc")
                    if not reasons0 and bool(step_term_1d[0].item()):
                        reasons0.append("term_unknown")
                    dbg_done_reason0 = "|".join(reasons0) if reasons0 else "none"
            self._done_total += int(new_done.sum().item())

            reached, dist_to_target = self._compute_stage_reached(rx, ry, bx, by, bz)
            reached = reached & active
            shaped, dense, time_pen = self._compute_reward(
                dist_to_target, done_now, reached, rx, ry, bx, by, bz
            )
            alive_f = active.to(dtype=shaped.dtype)
            inner_rew = shaped * alive_f
            total_reward += inner_rew
            total_dense += dense * alive_f
            total_time_pen += time_pen * alive_f

            if bool(reached.any()):
                stage_idx_now = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
                valid_now = self._stage_idx_buf < self._active_stage_count_buf
                finish_push = reached & self._main_push_stage_mask(stage_idx_now, valid_now)
                self._box_push_origin_x_buf = torch.where(
                    finish_push,
                    torch.full_like(self._box_push_origin_x_buf, float("nan")),
                    self._box_push_origin_x_buf,
                )
                self._box_push_origin_y_buf = torch.where(
                    finish_push,
                    torch.full_like(self._box_push_origin_y_buf, float("nan")),
                    self._box_push_origin_y_buf,
                )
                stage_bonus = self._stage_sparse_bonus(reached)
                inner_rew = inner_rew + stage_bonus
                total_reward = total_reward + stage_bonus
                total_sparse += stage_bonus
                rx1 = rx.squeeze(-1)
                ry1 = ry.squeeze(-1)
                self._stage_origin_x_buf = torch.where(reached, rx1, self._stage_origin_x_buf)
                self._stage_origin_y_buf = torch.where(reached, ry1, self._stage_origin_y_buf)
                self._stage_idx_buf = torch.where(reached, self._stage_idx_buf + 1, self._stage_idx_buf)
                self._sync_nav_stage_idx(rx, ry, bx, by, bz)
                self._prev_rel_progress = torch.where(
                    reached, torch.full_like(self._prev_rel_progress, float("nan")), self._prev_rel_progress
                )
            self._accumulate_episode_stage_reward(inner_rew, active)
            if new_done.any():
                episode_log.update(self._stage_reward_at_episode_log(new_done))

            self._prev_robot_x, self._prev_robot_y = rx.clone(), ry.clone()
            self._prev_box_x, self._prev_box_y = bx.clone(), by.clone()
            self._prev_box_com_z = torch.where(
                reached,
                torch.full_like(self._prev_box_com_z, float("nan")),
                bz.squeeze(-1),
            )

            self._reset_env_state(new_done, rx, ry, bx, by)

        self._nav_step_count += 1
        idx0 = int(self._stage_idx_buf[0].item())
        active0 = int(self._active_stage_count_buf[0].item())
        level0 = int(self._curriculum_level_buf[0].item())
        nav0 = self._nav_step_count
        stage_name = self._stage_names[idx0] if idx0 < self._num_stages else "done"
        if self._nav_log_interval > 0 and nav0 % self._nav_log_interval == 0:
            denom = max(1, self._done_total)
            stage_ratio_parts = [
                f"{self._stage_names[i]}={self._done_stage_counts[i] / denom:.2f}"
                for i in range(self._num_stages)
            ]
            if self._done_stage_counts[self._num_stages] > 0:
                stage_ratio_parts.append(f"finished={self._done_stage_counts[self._num_stages] / denom:.2f}")
            stage_at_done = " ".join(stage_ratio_parts)
            print(
                f"[{self._nav_log_tag}] nav={nav0:5d} curriculum={level0} "
                f"stage={idx0 + 1}/{self._num_stages}({stage_name}) active={active0} "
                f"prog={float(self._stage_progress_buf[0].item()):.2f} rew={total_reward.mean().item():+.4f} "
                f"nav[dense/sparse/time]={total_dense.mean().item():+.3f}/"
                f"{total_sparse.mean().item():+.3f}/{total_time_pen.mean().item():+.3f} "
                f"pos=({dbg_rx0:+.2f},{dbg_ry0:+.2f}) tgt=({dbg_tx0:+.2f},{dbg_ty0:+.2f}) "
                f"seg=({dbg_sx0:+.2f},{dbg_sy0:+.2f})->({dbg_ex0:+.2f},{dbg_ey0:+.2f}) dseg={dbg_dseg0:.2f} "
                f"why0={dbg_done_reason0} "
                f"done[f/t/x/stg/nm/np/pp]={self._done_fall}/{self._done_timeout}/{self._done_x_reached}/"
                f"{self._done_stage_target}/{self._done_no_motion}/{self._done_no_target_progress}/"
                f"{self._done_no_push_progress} "
                f"ratio={self._done_fall/denom:.2f}/{self._done_timeout/denom:.2f}/"
                f"{self._done_x_reached/denom:.2f}/{self._done_stage_target/denom:.2f}/"
                f"{self._done_no_motion/denom:.2f}/{self._done_no_target_progress/denom:.2f}/"
                f"{self._done_no_push_progress/denom:.2f} "
                f"stage_at_done[{stage_at_done}]",
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
