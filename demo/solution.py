import os
import torch


class AlgSolution:
    """Leg policy wrapper + simple velocity-command navigation."""

    # Task A: segmented forward speed by world-x strip
    _TASK_A_STRIP_X0 = -150.0
    _TASK_A_STRIP_DX = 20.0
    _TASK_A_VX = (
        2.0, 2.0,
        1.2, 1.2, 1.2, 1.2,
        1.0, 1.0, 1.0, 1.0,
        0.7, 0.7, 0.7, 0.7,
        2.0,
    )

    # Task B: B2 spawn + patrol / drop waypoints (world frame, axis-aligned)
    _TASK_B_SPAWN = (-10.0, -10.0)
    _TASK_B_WAYPOINTS = (
        (-12.0, -10.0),
        (-12.0, -12.0),
        (-7.0, -12.0),
        (-7.0, -7.0),
        (-3.0, -10.0),  # scoring circle center
    )

    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        self.device = "cuda"

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.leg_action_dim = 12
        self.arm_action_dim = 8

        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5] * 4, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0] * 4, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim), device=self.device, dtype=torch.float32
        )

        self.dt = 0.02
        self.nav_task = "A"

        # Task A
        self.task_a_spawn_x = -141.0
        self.task_a_vx_default = 0.5
        self.task_a_k_lat = 1.0
        self.task_a_k_yaw = 0.8
        self.task_a_k_wz = 0.25
        self.task_a_y_leak_len = 2.0
        self.task_a_vy_lim = 0.45
        self.task_a_wz_lim = 0.35

        strip_starts = [
            self._TASK_A_STRIP_X0 + i * self._TASK_A_STRIP_DX
            for i in range(len(self._TASK_A_VX))
        ]
        self._task_a_strip_starts = torch.tensor(
            strip_starts, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self._task_a_strip_vx = torch.tensor(
            self._TASK_A_VX, device=self.device, dtype=torch.float32
        ).view(1, -1)

        # Task B: unicycle-style (vx forward + yaw); no holonomic vy strafe
        self.task_b_vx_cruise = 0.35
        self.task_b_vx_creep = 0.08
        self.task_b_k_yaw = 0.65
        self.task_b_k_wz = 0.25
        self.task_b_align_thresh = 0.45  # ~26°: below this allow cruise vx
        self.task_b_wz_lim = 0.28
        self.task_b_wp_tol = 0.55

        self._reset_nav_state()

    def set_device(self, device: str) -> None:
        """Move policy + buffers to match Isaac Lab env device (e.g. cuda:1)."""
        device = str(device)
        if device == self.device:
            return
        self.device = device
        self.policy = self.policy.to(device)
        self.train_to_env_action_scale = self.train_to_env_action_scale.to(device)
        self.env_to_train_action_scale = self.env_to_train_action_scale.to(device)
        self.arm_default_action = self.arm_default_action.to(device)
        self._task_a_strip_starts = self._task_a_strip_starts.to(device)
        self._task_a_strip_vx = self._task_a_strip_vx.to(device)
        self._reset_nav_state()

    def reset(self, **kwargs):
        task = kwargs.get("task") or os.environ.get("ATEC_TASK", "")
        if isinstance(task, str):
            if "TaskB" in task:
                self.nav_task = "B"
            elif "TaskA" in task:
                self.nav_task = "A"
        self._reset_nav_state()

    def _reset_nav_state(self):
        self.yaw_est = None
        self.pos_x = None
        self.pos_y = None
        self._wp_idx = 0
        self._last_nav_cmd = (0.0, 0.0, 0.0)

    def _ensure_odom(self, proprio, device, dtype, batch_size: int):
        if self.yaw_est is not None and self.yaw_est.shape[0] == batch_size:
            return

        if self.nav_task == "B":
            x0, y0 = self._TASK_B_SPAWN
        else:
            x0, y0 = self.task_a_spawn_x, 0.0

        self.yaw_est = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        self.pos_x = torch.full((batch_size, 1), float(x0), device=device, dtype=dtype)
        self.pos_y = torch.full((batch_size, 1), float(y0), device=device, dtype=dtype)
        self._wp_idx = 0

    def _integrate_odom(self, vx_body, vy_body, wz):
        self.yaw_est = self.yaw_est + wz * self.dt
        self.yaw_est = torch.atan2(torch.sin(self.yaw_est), torch.cos(self.yaw_est))

        cos_y = torch.cos(self.yaw_est)
        sin_y = torch.sin(self.yaw_est)
        vx_world = cos_y * vx_body - sin_y * vy_body
        vy_world = sin_y * vx_body + cos_y * vy_body

        self.pos_x = self.pos_x + vx_world * self.dt
        if self.nav_task == "B":
            self.pos_y = self.pos_y + vy_world * self.dt
        else:
            leak = torch.exp(-torch.abs(vx_world) * self.dt / float(self.task_a_y_leak_len))
            self.pos_y = leak * self.pos_y + vy_world * self.dt
            self.pos_y = self.pos_y.clamp(-1.0, 1.0)

    def _task_a_strip_vx(self, world_x: torch.Tensor, dtype) -> torch.Tensor:
        sx = world_x.unsqueeze(-1)
        idx = (sx >= self._task_a_strip_starts.to(device=world_x.device, dtype=dtype)).sum(dim=-1) - 1
        idx = idx.clamp(0, self._task_a_strip_vx.shape[-1] - 1).long().reshape(-1)
        return self._task_a_strip_vx.squeeze(0)[idx].view(world_x.shape[0], 1)

    def _nav_task_a(self, vy_body, wz, gravity_y, device, dtype):
        vx_cmd = self._task_a_strip_vx(self.pos_x, dtype).to(device=device)
        vy_cmd = (
            -self.task_a_k_lat * self.pos_y - 0.4 * vy_body
        ).clamp(-self.task_a_vy_lim, self.task_a_vy_lim)

        yaw_grav = (0.4 * gravity_y).clamp(-0.1, 0.1)
        yaw_grav = torch.where(gravity_y.abs() > 0.03, yaw_grav, torch.zeros_like(yaw_grav))
        yaw_cmd = (
            -self.task_a_k_yaw * self.yaw_est - self.task_a_k_wz * wz - yaw_grav
        ).clamp(-self.task_a_wz_lim, self.task_a_wz_lim)
        return vx_cmd, vy_cmd, yaw_cmd

    def _nav_task_b(self, vy_body, wz, device, dtype):
        tx, ty = self._TASK_B_WAYPOINTS[self._wp_idx]
        dx = torch.full((1, 1), float(tx), device=device, dtype=dtype) - self.pos_x
        dy = torch.full((1, 1), float(ty), device=device, dtype=dtype) - self.pos_y
        dist = torch.sqrt(dx * dx + dy * dy)

        if dist[0, 0].item() < self.task_b_wp_tol:
            self._wp_idx = (self._wp_idx + 1) % len(self._TASK_B_WAYPOINTS)
            tx, ty = self._TASK_B_WAYPOINTS[self._wp_idx]
            dx = torch.full((1, 1), float(tx), device=device, dtype=dtype) - self.pos_x
            dy = torch.full((1, 1), float(ty), device=device, dtype=dtype) - self.pos_y
            dist = torch.sqrt(dx * dx + dy * dy)

        desired_yaw = torch.atan2(dy, dx)
        yaw_err = torch.atan2(
            torch.sin(desired_yaw - self.yaw_est),
            torch.cos(desired_yaw - self.yaw_est),
        )
        yaw_abs = yaw_err.abs()

        # Turn toward waypoint first; only cruise vx when roughly aligned.
        aligned = yaw_abs < self.task_b_align_thresh
        vx_cmd = torch.where(
            aligned,
            torch.full_like(dist, self.task_b_vx_cruise),
            torch.full_like(dist, self.task_b_vx_creep),
        )
        vx_cmd = vx_cmd * torch.clamp(dist / 1.2, 0.35, 1.0)

        # Quadruped policy: avoid lateral strafe; small damping only.
        vy_cmd = (-0.25 * vy_body).clamp(-0.12, 0.12)

        yaw_cmd = (
            self.task_b_k_yaw * yaw_err - self.task_b_k_wz * wz
        ).clamp(-self.task_b_wz_lim, self.task_b_wz_lim)
        return vx_cmd, vy_cmd, yaw_cmd

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        device = proprio.device
        dtype = proprio.dtype
        batch_size = proprio.shape[0]

        base_lin_vel = proprio[:, 0:3]
        base_ang_vel = proprio[:, 3:6]
        projected_gravity = proprio[:, 9:12]

        vx_body = base_lin_vel[:, 0:1]
        vy_body = base_lin_vel[:, 1:2]
        wz = base_ang_vel[:, 2:3]
        gravity_y = projected_gravity[:, 1:2]

        self._ensure_odom(proprio, device, dtype, batch_size)
        self._integrate_odom(vx_body, vy_body, wz)

        if self.nav_task == "B":
            vx_cmd, vy_cmd, yaw_cmd = self._nav_task_b(vy_body, wz, device, dtype)
        else:
            vx_cmd, vy_cmd, yaw_cmd = self._nav_task_a(vy_body, wz, gravity_y, device, dtype)

        cmd = torch.cat([vx_cmd, vy_cmd, yaw_cmd], dim=-1)
        self._last_nav_cmd = (
            float(cmd[0, 0].item()),
            float(cmd[0, 1].item()),
            float(cmd[0, 2].item()),
        )
        return cmd

    def get_video_overlay_lines(self) -> list[str]:
        if self.pos_x is None:
            return []
        vx, vy, yaw = self._last_nav_cmd
        lines = [
            f"task={self.nav_task}  x={self.pos_x[0, 0].item():.2f}  "
            f"y={self.pos_y[0, 0].item():+.3f}  yaw={self.yaw_est[0, 0].item():+.3f}",
            f"cmd=({vx:.2f}, {vy:+.3f}, {yaw:+.3f})",
        ]
        if self.nav_task == "B":
            tx, ty = self._TASK_B_WAYPOINTS[self._wp_idx]
            lines.append(f"wp={self._wp_idx}→({tx:.1f},{ty:.1f})")
        return lines

    def _extract_policy_obs(self, obs, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        idx = 6  # skip base_lin_vel + base_ang_vel + vel_cmd
        base_ang_vel = proprio[:, 3:6]
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
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros(
            (action_train.shape[0], action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(action_train.shape[0], 1)
        return action_env

    def predicts(self, obs, current_score):
        proprio = obs["proprio"]
        if str(proprio.device) != self.device:
            self.set_device(str(proprio.device))
        proprio = proprio.to(self.device)
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
