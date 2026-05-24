import os
import torch


class AlgSolution:
    """Task D step 1: retreat, sidestep until box leaves front cone, then push-ready."""

    ROBOT_SPAWN_XY = (-3.0, 0.0)
    BOX_SPAWN_XY = (-3.0, 1.6)
    PUSH_YAW = 0.0
    # Box size in env_cfg: 0.8 x 1.0 x 0.6 (x x y x z).
    BOX_HALF_X = 0.40
    BOX_HALF_Y = 0.50

    # LiDAR layout (same as Task A height-scan)
    _LIDAR_H = 360
    _LIDAR_FRONT_HALF = 25
    _LIDAR_GROUND_REF = 0.08
    _LIDAR_BOX_DETECT_DELTA = 0.12

    def __init__(self):
        policy_path = os.path.dirname(os.path.abspath(__file__)) + "/policy.pt"
        self.device = "cuda"

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        self.dt = 0.02
        self.nav_k_yaw = 1.2
        self.nav_k_wz = 0.25
        self.nav_wz_lim = 0.35

        # Phase retreat: back up along −x (world) before sidestep.
        self.retreat_dist = 0.45
        self.retreat_vx = -0.30
        # Phase sidestep: +vy (body left) until front LiDAR loses box.
        self.sidestep_vy = 0.35
        self.sidestep_max_steps = 400
        self.sidestep_no_box_steps = 8

        self.phase = "retreat"
        self.x_est = None
        self.y_est = None
        self.yaw_est = None
        self._retreat_start_x = None
        self._sidestep_steps = 0
        self._sidestep_clear_count = 0
        self._env = None
        self._robot = None
        self._box = None
        self._extero = None
        self._debug_step = 0
        self._last_nav_cmd = (0.0, 0.0, 0.0)
        self._last_box_xy = (0.0, 0.0)
        self._last_front_lidar = 0.0
        self._last_box_in_front = False
        self._last_box_geom = False
        self._last_box_lidar = False
        self._approach_done = False

    def set_device(self, device: str) -> None:
        self.device = device
        self.policy = self.policy.to(device)
        self.train_to_env_action_scale = self.train_to_env_action_scale.to(device)
        self.env_to_train_action_scale = self.env_to_train_action_scale.to(device)
        self.arm_default_action = self.arm_default_action.to(device)

    def bind_env(self, env) -> None:
        self._env = env
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self._robot = unwrapped.scene.articulations["robot"]
        self._box = unwrapped.scene.rigid_objects["box"]

    def reset(self, **kwargs):
        self.phase = "retreat"
        self.x_est = None
        self.y_est = None
        self.yaw_est = None
        self._retreat_start_x = None
        self._sidestep_steps = 0
        self._sidestep_clear_count = 0
        self._debug_step = 0
        self._approach_done = False
        self._last_nav_cmd = (0.0, 0.0, 0.0)

    @staticmethod
    def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).unsqueeze(-1)

    def _box_pose_xy(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self._box is not None:
            box_xy = self._box.data.root_pos_w[:, :2].to(device=device, dtype=dtype)
            return box_xy[:, 0:1], box_xy[:, 1:2]
        bx = torch.full((1, 1), self.BOX_SPAWN_XY[0], device=device, dtype=dtype)
        by = torch.full((1, 1), self.BOX_SPAWN_XY[1], device=device, dtype=dtype)
        return bx, by

    def _update_odom(self, proprio: torch.Tensor) -> None:
        device, dtype = proprio.device, proprio.dtype
        b = proprio.shape[0]
        vx_body = proprio[:, 0:1]
        vy_body = proprio[:, 1:2]
        wz = proprio[:, 5:6]

        if self.x_est is None or self.x_est.shape[0] != b:
            self.x_est = torch.full((b, 1), self.ROBOT_SPAWN_XY[0], device=device, dtype=dtype)
            self.y_est = torch.full((b, 1), self.ROBOT_SPAWN_XY[1], device=device, dtype=dtype)
            self.yaw_est = torch.zeros((b, 1), device=device, dtype=dtype)
        else:
            self.x_est = self.x_est.to(device=device, dtype=dtype)
            self.y_est = self.y_est.to(device=device, dtype=dtype)
            self.yaw_est = self.yaw_est.to(device=device, dtype=dtype)

        self.yaw_est = self.yaw_est + wz * self.dt
        self.yaw_est = torch.atan2(torch.sin(self.yaw_est), torch.cos(self.yaw_est))

        cos_y, sin_y = torch.cos(self.yaw_est), torch.sin(self.yaw_est)
        vx_world = cos_y * vx_body - sin_y * vy_body
        vy_world = sin_y * vx_body + cos_y * vy_body
        self.x_est = self.x_est + vx_world * self.dt
        self.y_est = self.y_est + vy_world * self.dt

    def _robot_xy_yaw(self, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = proprio.device, proprio.dtype
        if self._robot is not None:
            pos = self._robot.data.root_pos_w.to(device=device, dtype=dtype)
            quat = self._robot.data.root_quat_w.to(device=device, dtype=dtype)
            return pos[:, 0:1], pos[:, 1:2], self._yaw_from_quat_wxyz(quat)
        self._update_odom(proprio)
        return self.x_est, self.y_est, self.yaw_est

    def _sector_mean(self, ch0, ch1, ch2, center: int, half_width: int) -> torch.Tensor:
        H = self._LIDAR_H
        idx = torch.arange(center - half_width, center + half_width + 1, device=ch0.device) % H
        vals = torch.cat([ch0[idx], ch1[idx], ch2[idx]], dim=0).clamp(min=-1.0, max=3.0)
        k = max(1, vals.numel() // 5)
        return torch.topk(vals, k).values.mean()

    def _front_lidar_metric(self, extero: torch.Tensor | None) -> tuple[float, bool]:
        """Return (front_sector_mean, box_detected_in_front).

        Flat ground ≈ 0.08; box top/face in front cone → value deviates strongly.
        """
        if extero is None or extero.shape[-1] < self._LIDAR_H:
            return self._LIDAR_GROUND_REF, False

        device = extero.device
        dtype = extero.dtype
        rays = extero[0].to(device=device, dtype=dtype)
        rays = rays.nan_to_num(nan=self._LIDAR_GROUND_REF, posinf=3.0, neginf=0.0)
        H = self._LIDAR_H
        ch0 = rays[0 * H : 1 * H]
        ch1 = rays[1 * H : 2 * H]
        ch2 = rays[2 * H : 3 * H]
        front_val = self._sector_mean(ch0, ch1, ch2, 180, self._LIDAR_FRONT_HALF)
        fv = float(front_val.item())
        delta = abs(fv - self._LIDAR_GROUND_REF)
        box_det = delta > self._LIDAR_BOX_DETECT_DELTA or fv < -0.05
        return fv, box_det

    def _box_in_front_geometric(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        yaw: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
    ) -> torch.Tensor:
        """True when box AABB overlaps the robot forward cone (body frame)."""
        dx = box_x - x
        dy = box_y - y
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        bx_body = cos_y * dx + sin_y * dy
        by_body = -sin_y * dx + cos_y * dy
        half_x = self.BOX_HALF_X + 0.15
        half_y = self.BOX_HALF_Y + 0.20
        in_front = (bx_body > -half_x) & (bx_body < 2.5) & (by_body.abs() < half_y)
        return in_front

    def _yaw_hold_cmd(self, yaw: torch.Tensor, wz: torch.Tensor, device, dtype) -> torch.Tensor:
        yaw_err = torch.atan2(
            torch.sin(torch.full_like(yaw, self.PUSH_YAW) - yaw),
            torch.cos(torch.full_like(yaw, self.PUSH_YAW) - yaw),
        )
        return (self.nav_k_yaw * yaw_err - self.nav_k_wz * wz).clamp(
            -self.nav_wz_lim, self.nav_wz_lim
        )

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        device, dtype = proprio.device, proprio.dtype
        b = proprio.shape[0]
        wz = proprio[:, 5:6]

        x, y, yaw = self._robot_xy_yaw(proprio)
        box_x, box_y = self._box_pose_xy(device, dtype)
        self._last_box_xy = (float(box_x[0, 0].item()), float(box_y[0, 0].item()))

        front_val, box_lidar = self._front_lidar_metric(self._extero)
        box_geom = self._box_in_front_geometric(x, y, yaw, box_x, box_y)
        # Local Isaac Lab RayCaster only hits ground; eval may include box in mesh.
        if self._box is not None:
            box_in_front = bool(box_geom.all())
        else:
            box_in_front = box_lidar
        self._last_front_lidar = front_val
        self._last_box_lidar = bool(box_lidar)
        self._last_box_geom = bool(box_geom.all())
        self._last_box_in_front = bool(box_in_front)

        if self._retreat_start_x is None:
            self._retreat_start_x = x.clone()

        yaw_cmd = self._yaw_hold_cmd(yaw, wz, device, dtype)

        if self.phase == "retreat":
            retreated = (self._retreat_start_x - x) >= self.retreat_dist
            if bool(retreated.all()):
                self.phase = "sidestep"
                self._sidestep_steps = 0
                self._sidestep_clear_count = 0
            vx_cmd = torch.full((b, 1), float(self.retreat_vx), device=device, dtype=dtype)
            vy_cmd = torch.zeros((b, 1), device=device, dtype=dtype)

        elif self.phase == "sidestep":
            self._sidestep_steps += 1
            if not box_in_front:
                self._sidestep_clear_count += 1
            else:
                self._sidestep_clear_count = 0

            cleared = self._sidestep_clear_count >= self.sidestep_no_box_steps
            timed_out = self._sidestep_steps >= self.sidestep_max_steps
            if cleared or timed_out:
                self.phase = "push_ready"
                self._approach_done = True
                vx_cmd = torch.zeros((b, 1), device=device, dtype=dtype)
                vy_cmd = torch.zeros((b, 1), device=device, dtype=dtype)
            else:
                vx_cmd = torch.zeros((b, 1), device=device, dtype=dtype)
                vy_cmd = torch.full((b, 1), float(self.sidestep_vy), device=device, dtype=dtype)

        elif self.phase == "push_ready":
            vx_cmd = torch.zeros((b, 1), device=device, dtype=dtype)
            vy_cmd = torch.zeros((b, 1), device=device, dtype=dtype)
        else:
            vx_cmd = torch.zeros((b, 1), device=device, dtype=dtype)
            vy_cmd = torch.zeros((b, 1), device=device, dtype=dtype)

        cmd = torch.cat([vx_cmd, vy_cmd, yaw_cmd], dim=-1)
        self._last_nav_cmd = (
            float(cmd[0, 0].item()),
            float(cmd[0, 1].item()),
            float(cmd[0, 2].item()),
        )
        self._debug_step += 1
        if self._debug_step % 50 == 1:
            print(
                f"[taskd {self.phase} step={self._debug_step:5d}] "
                f"pos=({x[0,0].item():+.2f},{y[0,0].item():+.2f}) "
                f"box=({box_x[0,0].item():+.2f},{box_y[0,0].item():+.2f}) "
                f"front_hs={front_val:.3f} geom={int(self._last_box_geom)} "
                f"lidar={int(self._last_box_lidar)} box_det={int(self._last_box_in_front)} "
                f"ret_dx={(self._retreat_start_x[0,0]-x[0,0]).item():.2f} "
                f"done={int(self._approach_done)} "
                f"cmd=({vx_cmd[0,0].item():.2f},{vy_cmd[0,0].item():+.2f},{yaw_cmd[0,0].item():+.2f})"
            )
        return cmd

    def get_video_overlay_lines(self) -> list[str]:
        lines = [
            f"phase={self.phase}  done={int(self._approach_done)}",
            f"box=({self._last_box_xy[0]:+.2f},{self._last_box_xy[1]:+.2f})",
            f"front_hs={self._last_front_lidar:.3f}  geom={int(self._last_box_geom)}  "
            f"lidar={int(self._last_box_lidar)}  box_det={int(self._last_box_in_front)}",
        ]
        vx, vy, yaw = self._last_nav_cmd
        lines.append(f"cmd=({vx:.2f},{vy:+.2f},{yaw:+.2f})")
        return lines

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        idx = 0
        idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3
        idx += 3
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
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def predicts(self, obs, current_score):
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        raw_extero = obs.get("extero", None)
        if raw_extero is not None:
            self._extero = raw_extero.to(self.device, dtype=torch.float32)
        else:
            self._extero = None
            if self._debug_step == 0:
                print("[taskd WARNING] no extero — sidestep uses geometry/timeout only")

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
