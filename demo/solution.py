"""
Task B — waypoint navigation with proprio odometry only (no LiDAR / no camera).

Run play:
  cp "demo/solution copy.py" demo/solution.py
  python scripts/play_atec_task.py --task ATEC-TaskB-B2Piper --video
"""

import os
import torch


class AlgSolution:

    ACTION_SCALE = 0.5

    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        self.device = "cuda"

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5] * 4, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0] * 4, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim), device=self.device, dtype=torch.float32
        )

        # Odometry + waypoint nav (Task B B2Piper spawn)
        self.dt = 0.02
        self.nav_init_world_x = -10.0
        self.nav_init_world_y = -10.0
        self.nav_vx_max = 0.85
        self.nav_vy_lim = 0.40
        self.nav_k_yaw = 1.1
        self.nav_k_wz = 0.25
        self.nav_wz_lim = 0.40
        self.wp_arrival_thresh = 0.45
        self.wp_k_v = 0.85
        self.wp_k_lat = 0.65

        self.waypoints = [
            (-10.0, -10.0),
            (-13.0, -13.0),
            (-15.0, -10.0),
            (-15.0, -6.0),
            (-10.0, -6.0),
            (-6.0, -10.0),
            (-6.0, -14.0),
            (-10.0, -14.0),
            (-3.0, -10.0),  # trash bin center (task_b TARGET_CENTER)
        ]
        self._wp_tensor = torch.tensor(self.waypoints, device=self.device, dtype=torch.float32)

        self.yaw_est = None
        self.y_est = None
        self.x_est = None
        self.wp_idx = 0
        self._debug_step = 0
        self._extero = None

        # Optional stuck recovery (flat field)
        self.recovery_stuck_vx_thresh = 0.03
        self.recovery_stuck_steps = 50
        self.recovery_duration_steps = 50
        self.recovery_vx_cmd = 0.15
        self.recovery_yaw_mag = 0.25
        self._slow_vx_accum = None
        self._recovery_left = None
        self._recovery_next_yaw = None
        self._active_recovery_yaw = None
        self._predicts_calls = 0

        print(
            "[TaskB] AlgSolution ready (waypoint odom, no lidar/camera). "
            "First step may be slow if env renders 4 cameras + video.",
            flush=True,
        )

    def reset(self, **kwargs):
        self.yaw_est = None
        self.y_est = None
        self.x_est = None
        self.wp_idx = 0
        self._debug_step = 0
        self._predicts_calls = 0
        self._slow_vx_accum = None
        self._recovery_left = None
        self._recovery_next_yaw = None
        self._active_recovery_yaw = None
        print("[TaskB] reset() odom cleared.", flush=True)

    def _advance_waypoint(self, x_est: torch.Tensor, y_est: torch.Tensor) -> int:
        idx = self.wp_idx
        thresh2 = self.wp_arrival_thresh ** 2
        while idx < len(self.waypoints) - 1:
            wx, wy = self.waypoints[idx]
            dx = float(x_est[0, 0].item()) - wx
            dy = float(y_est[0, 0].item()) - wy
            if dx * dx + dy * dy < thresh2:
                idx += 1
            else:
                break
        self.wp_idx = idx
        return idx

    def _waypoint_vel_cmd(
        self,
        x_est: torch.Tensor,
        y_est: torch.Tensor,
        yaw_est: torch.Tensor,
        wz: torch.Tensor,
        device,
        dtype,
        b: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self._advance_waypoint(x_est, y_est)
        wp = self._wp_tensor[idx].to(device=device, dtype=dtype)

        dx = wp[0] - x_est[:, 0]
        dy = wp[1] - y_est[:, 0]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6).view(b, 1)

        target_yaw = torch.atan2(dy, dx).view(b, 1)
        yaw_err = target_yaw - yaw_est
        yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

        vx = (self.wp_k_v * dist * torch.cos(yaw_err)).clamp(0.0, self.nav_vx_max)
        vy = (self.wp_k_lat * dist * torch.sin(yaw_err)).clamp(-self.nav_vy_lim, self.nav_vy_lim)
        yaw_cmd = (self.nav_k_yaw * yaw_err - self.nav_k_wz * wz).clamp(
            -self.nav_wz_lim, self.nav_wz_lim
        )
        return vx, vy, yaw_cmd

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        device = proprio.device
        dtype = proprio.dtype
        b = proprio.shape[0]

        vx_body = proprio[:, 0:1]
        vy_body = proprio[:, 1:2]
        wz = proprio[:, 3:6][:, 2:3]

        if self.yaw_est is None or self.yaw_est.shape[0] != b:
            self.yaw_est = torch.zeros((b, 1), device=device, dtype=dtype)
            self.y_est = torch.full(
                (b, 1), float(self.nav_init_world_y), device=device, dtype=dtype
            )
            self.x_est = torch.full(
                (b, 1), float(self.nav_init_world_x), device=device, dtype=dtype
            )
            zl = torch.zeros((b, 1), device=device, dtype=torch.long)
            self._slow_vx_accum = zl.clone()
            self._recovery_left = zl.clone()
            self._recovery_next_yaw = torch.ones((b, 1), device=device, dtype=dtype)
            self._active_recovery_yaw = torch.ones((b, 1), device=device, dtype=dtype)
            self.wp_idx = 0
        else:
            self.yaw_est = self.yaw_est.to(device=device, dtype=dtype)
            self.y_est = self.y_est.to(device=device, dtype=dtype)
            self.x_est = self.x_est.to(device=device, dtype=dtype)
            self._slow_vx_accum = self._slow_vx_accum.to(device=device)
            self._recovery_left = self._recovery_left.to(device=device)
            self._recovery_next_yaw = self._recovery_next_yaw.to(device=device, dtype=dtype)
            self._active_recovery_yaw = self._active_recovery_yaw.to(device=device, dtype=dtype)

        in_recovery_before = self._recovery_left > 0

        self.yaw_est = torch.where(
            in_recovery_before,
            self.yaw_est,
            self.yaw_est + wz * self.dt,
        )
        self.yaw_est = torch.atan2(torch.sin(self.yaw_est), torch.cos(self.yaw_est))

        cos_y, sin_y = torch.cos(self.yaw_est), torch.sin(self.yaw_est)
        vx_world = cos_y * vx_body - sin_y * vy_body
        vy_world = sin_y * vx_body + cos_y * vy_body
        self.x_est = self.x_est + vx_world * self.dt
        self.y_est = torch.where(
            in_recovery_before,
            self.y_est,
            self.y_est + vy_world * self.dt,
        )

        thresh = torch.tensor(self.recovery_stuck_vx_thresh, device=device, dtype=dtype)
        vx_low = vx_world < thresh
        self._slow_vx_accum = torch.where(
            in_recovery_before,
            self._slow_vx_accum,
            torch.where(vx_low, self._slow_vx_accum + 1, torch.zeros_like(self._slow_vx_accum)),
        )
        trigger = (~in_recovery_before) & (self._slow_vx_accum >= self.recovery_stuck_steps)
        self._active_recovery_yaw = torch.where(
            trigger, self._recovery_next_yaw.to(dtype=dtype), self._active_recovery_yaw
        )
        self._recovery_next_yaw = torch.where(
            trigger, -self._recovery_next_yaw, self._recovery_next_yaw
        )
        self._recovery_left = torch.where(
            trigger,
            torch.full_like(self._recovery_left, self.recovery_duration_steps),
            self._recovery_left,
        )
        self._slow_vx_accum = torch.where(
            trigger, torch.zeros_like(self._slow_vx_accum), self._slow_vx_accum
        )
        in_recovery = self._recovery_left > 0

        vx_cmd, vy_cmd, yaw_cmd = self._waypoint_vel_cmd(
            self.x_est, self.y_est, self.yaw_est, wz, device, dtype, b
        )

        vx_rec = torch.full((b, 1), float(self.recovery_vx_cmd), device=device, dtype=dtype)
        vy_rec = torch.zeros((b, 1), device=device, dtype=dtype)
        yaw_rec = self._active_recovery_yaw.to(dtype=dtype) * float(self.recovery_yaw_mag)
        vx_cmd = torch.where(in_recovery, vx_rec, vx_cmd)
        vy_cmd = torch.where(in_recovery, vy_rec, vy_cmd)
        yaw_cmd = torch.where(in_recovery, yaw_rec, yaw_cmd)

        was_recovery = in_recovery.squeeze(-1)
        self._recovery_left = self._recovery_left - was_recovery.unsqueeze(-1).long()
        self._recovery_left = torch.clamp(self._recovery_left, min=0)
        just_finished = was_recovery.unsqueeze(-1) & (self._recovery_left == 0)
        self._slow_vx_accum = torch.where(
            just_finished, torch.zeros_like(self._slow_vx_accum), self._slow_vx_accum
        )

        cmd = torch.cat([vx_cmd, vy_cmd, yaw_cmd], dim=-1)
        self._debug_step += 1
        if self._debug_step <= 10 or self._debug_step % 25 == 1:
            wp = self.waypoints[self.wp_idx]
            print(
                f"[TaskB nav step={self._debug_step:5d}] "
                f"wp={self.wp_idx}/{len(self.waypoints)-1} ({wp[0]:.1f},{wp[1]:.1f})  "
                f"x={self.x_est[0,0].item():7.2f}  y={self.y_est[0,0].item():+7.2f}  "
                f"yaw={self.yaw_est[0,0].item():+5.3f}  "
                f"| cmd=({vx_cmd[0,0].item():.2f}, {vy_cmd[0,0].item():+.3f}, {yaw_cmd[0,0].item():+.3f})",
                flush=True,
            )
        return cmd

    def _extract_policy_obs(self, obs, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        idx = 3
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 6
        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]
        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)

        velocity_commands = self._get_velocity_commands(proprio)

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def predicts(self, obs, current_score):
        self._predicts_calls += 1
        if self._predicts_calls <= 5:
            print(f"[TaskB] predicts() #{self._predicts_calls}", flush=True)

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        return {"action": action_env.cpu().numpy().tolist(), "giveup": False}
