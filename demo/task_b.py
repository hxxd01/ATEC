import os
import torch


def _build_grasp_matrix(long_axis: torch.Tensor, grip_z: torch.Tensor) -> torch.Tensor:
    jaw_dir = torch.linalg.cross(long_axis, grip_z)
    jaw_dir = jaw_dir / jaw_dir.norm().clamp(min=1e-6)
    align_dir = torch.linalg.cross(jaw_dir, grip_z)
    align_dir = align_dir / align_dir.norm().clamp(min=1e-6)
    return torch.stack([align_dir, jaw_dir, grip_z], dim=1)


def _compute_grasp_quat(obj_quat_w: torch.Tensor, device: str) -> torch.Tensor:
    """Top-down grasp orientation aligned with object long axis."""
    from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

    default_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    r_obj = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)
    grip_z = torch.tensor([0.0, 0.0, -1.0], device=device)

    norms, axes_xy = [], []
    for col in range(3):
        ax = torch.tensor([r_obj[0, col].item(), r_obj[1, col].item(), 0.0], device=device)
        norms.append(ax.norm().item())
        axes_xy.append(ax)

    best_norm = max(norms)
    candidates = [
        axes_xy[c] / max(norms[c], 1e-6)
        for c in range(3)
        if norms[c] >= best_norm - 1e-3
    ]

    best_cos = -2.0
    long_axis = candidates[0]
    for cand in candidates:
        q_cand = quat_from_matrix(_build_grasp_matrix(cand, grip_z).unsqueeze(0)).squeeze(0)
        cos_sim = torch.abs((q_cand * default_quat).sum()).item()
        if cos_sim > best_cos:
            best_cos = cos_sim
            long_axis = cand

    return quat_from_matrix(_build_grasp_matrix(long_axis, grip_z).unsqueeze(0)).squeeze(0)


class _TaskBPickSM:
    """Minimal pick sequence: PRE_GRASP → REACH → CLOSE → LIFT."""

    _ORDER = ("PRE_GRASP", "REACH", "CLOSE", "LIFT")
    _STEPS = {"PRE_GRASP": 120, "REACH": 100, "CLOSE": 40, "LIFT": 120}

    def __init__(self, grasp_quat: torch.Tensor, device: str):
        self._device = device
        self._grasp_quat = grasp_quat
        self._state_idx = 0
        self._count = 0
        self._cached_obj_pos = None
        self.done = False

    @property
    def state(self) -> str:
        return self._ORDER[self._state_idx]

    def tick(self, obj_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        state = self.state
        if state == "PRE_GRASP" and self._count == 0:
            self._cached_obj_pos = obj_pos.clone()
        if state in ("REACH", "CLOSE") and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos

        if state == "PRE_GRASP":
            target = obj_pos.clone()
            target[2] = _TaskBPickSM._carry_z()
            gripper = "open"
        elif state == "REACH":
            target = obj_pos.clone()
            target[2] += _TaskBPickSM._grasp_z_offset()
            gripper = "open"
        elif state == "CLOSE":
            target = obj_pos.clone()
            target[2] += _TaskBPickSM._grasp_z_offset()
            gripper = "close"
        else:  # LIFT
            target = obj_pos.clone()
            target[2] = _TaskBPickSM._carry_z()
            gripper = "close"

        ee_quat = self._grasp_quat if state in ("REACH", "CLOSE", "LIFT") else torch.tensor(
            [0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=self._device
        )

        self._count += 1
        if self._count >= self._STEPS[state]:
            self._count = 0
            if state == "LIFT":
                self.done = True
            else:
                self._state_idx += 1

        return target, ee_quat, gripper

    @staticmethod
    def _carry_z() -> float:
        return 0.55

    @staticmethod
    def _grasp_z_offset() -> float:
        return 0.10


class AlgSolution:
    """Leg policy + Task A strip nav / Task B turn-then-drive waypoint nav."""

    _TASK_A_STRIP_X0 = -150.0
    _TASK_A_STRIP_DX = 20.0
    _TASK_A_STRIP_TERRAINS = (
        "flat", "flat",
        "random_rough", "random_rough", "random_rough", "random_rough",
        "hf_pyramid_slope", "hf_pyramid_slope_inv",
        "hf_pyramid_slope", "hf_pyramid_slope_inv",
        "pyramid_stairs", "pyramid_stairs_inv",
        "pyramid_stairs", "pyramid_stairs_inv",
        "flat",
    )
    _TASK_A_VX = (
        2.0, 2.0,
        1.2, 1.2, 1.2, 1.2,
        1.0, 1.0, 1.0, 1.0,
        0.7, 0.7, 0.7, 0.7,
        2.0,
    )

    _TASK_B_SPAWN = (-10.0, -10.0)
    _TASK_B_DROP = (-3.0, -10.0)
    _TASK_B_NUM_OBJECTS = 18
    _TASK_B_WAYPOINTS = (
        (-12.0, -12.0),
        (-7.0, -12.0),
        (-7.0, -7.0),
        (-12.0, -7.0),
        (-3.0, -10.0),
    )
    _TASK_B_EE_BODY = "gripper_base"
    _TASK_B_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    _TASK_B_GRIPPER_JOINTS = ["joint7", "joint8"]
    _TASK_B_IK_ACTION_SCALE = 0.5
    _TASK_B_GRIPPER_OPEN = (0.035, -0.035)
    _TASK_B_GRIPPER_CLOSE = (-0.015, 0.015)
    _TASK_B_APPROACH_DIST = 0.85

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
        self.task_a_k_lat = 1.0
        self.task_a_k_yaw = 0.8
        self.task_a_k_wz = 0.25
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
        self._task_a_strip_is_inv = torch.tensor(
            [
                1.0 if k in ("pyramid_stairs_inv", "hf_pyramid_slope_inv") else 0.0
                for k in self._TASK_A_STRIP_TERRAINS
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        # Task B: turn in place first, then drive forward only (no lateral vy)
        self.task_b_vx = 0.35
        self.task_b_k_yaw_turn = 0.6
        self.task_b_k_yaw_drive = 0.25
        self.task_b_k_wz = 0.2
        self.task_b_yaw_align = 0.20  # rad, ~11° — must face target before walking
        self.task_b_wz_lim = 0.30
        self.task_b_wp_tol = 0.50

        self._env = None
        self._ik_ctrl = None
        self._arm_ids = None
        self._gripper_ids = None
        self._default_joint_pos = None
        self._task_b_ik_enabled = False
        self._reset_nav_state()

    def bind_env(self, env) -> None:
        """Attach sim env for privileged Task B IK (Route A)."""
        self._env = env
        self._ik_ctrl = None
        self._arm_ids = None
        self._gripper_ids = None
        self._default_joint_pos = None
        if self.nav_task == "B":
            self._task_b_ik_enabled = True

    def set_device(self, device: str) -> None:
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
        self._task_a_strip_is_inv = self._task_a_strip_is_inv.to(device)
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
        self._task_b_mode = "turn"
        self._last_nav_cmd = (0.0, 0.0, 0.0)
        self._task_b_phase = "nav"
        self._pick_obj_idx = None
        self._pick_sm = None
        self._objects_done: set[int] = set()
        self._hold_arm_action = None
        self._last_pick_state = "-"
        if self._ik_ctrl is not None:
            self._ik_ctrl.reset()

    def _ensure_odom(self, device, dtype, batch_size: int):
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
        self._task_b_mode = "turn"

    def _integrate_odom(self, vx_body, vy_body, wz):
        self.yaw_est = self.yaw_est + wz * self.dt
        self.yaw_est = torch.atan2(torch.sin(self.yaw_est), torch.cos(self.yaw_est))

        cos_y = torch.cos(self.yaw_est)
        sin_y = torch.sin(self.yaw_est)
        vx_world = cos_y * vx_body - sin_y * vy_body
        vy_world = sin_y * vx_body + cos_y * vy_body

        self.pos_x = self.pos_x + vx_world * self.dt
        self.pos_y = self.pos_y + vy_world * self.dt

    def _task_a_strip_vx(self, world_x: torch.Tensor, dtype) -> torch.Tensor:
        sx = world_x.unsqueeze(-1)
        idx = (sx >= self._task_a_strip_starts.to(device=world_x.device, dtype=dtype)).sum(dim=-1) - 1
        idx = idx.clamp(0, self._task_a_strip_vx.shape[-1] - 1).long().reshape(-1)
        return self._task_a_strip_vx.squeeze(0)[idx].view(world_x.shape[0], 1)

    def _nav_task_a(self, vy_body, wz, gravity_y, device, dtype, batch_size):
        vx_cmd = self._task_a_strip_vx(self.pos_x, dtype).to(device=device)

        strip_idx = (
            (self.pos_x.unsqueeze(-1) >= self._task_a_strip_starts.to(device=device, dtype=dtype))
            .sum(dim=-1)
            .sub(1)
            .clamp(0, self._task_a_strip_vx.shape[-1] - 1)
            .long()
            .reshape(-1)
        )
        inv_mask = self._task_a_strip_is_inv.squeeze(0)[strip_idx].view(batch_size, 1)
        grav_sign = 1.0 - 2.0 * inv_mask

        vy_cmd = (
            -self.task_a_k_lat * self.pos_y - 0.4 * vy_body
        ).clamp(-self.task_a_vy_lim, self.task_a_vy_lim)

        yaw_grav = (0.4 * gravity_y).clamp(-0.1, 0.1)
        yaw_grav = torch.where(gravity_y.abs() > 0.03, yaw_grav, torch.zeros_like(yaw_grav))
        yaw_cmd = (
            -self.task_a_k_yaw * self.yaw_est - self.task_a_k_wz * wz - grav_sign * yaw_grav
        ).clamp(-self.task_a_wz_lim, self.task_a_wz_lim)
        return vx_cmd, vy_cmd, yaw_cmd

    def _robot_xy_sim(self) -> tuple[float, float] | None:
        if self._env is None:
            return None
        try:
            robot = self._env.unwrapped.scene["robot"]
            pos = robot.data.root_pos_w[0]
            return float(pos[0].item()), float(pos[1].item())
        except (AttributeError, KeyError):
            return None

    def _get_object_asset(self, obj_idx: int):
        scene = self._env.unwrapped.scene
        key = f"object_{obj_idx}"
        if hasattr(scene, "rigid_objects") and key in scene.rigid_objects:
            return scene.rigid_objects[key]
        return scene[key]

    def _nearest_available_object(self) -> tuple[int, torch.Tensor, float] | None:
        if self._env is None:
            return None
        robot_xy = self._robot_xy_sim()
        if robot_xy is None:
            return None
        rx, ry = robot_xy
        best = None
        for obj_idx in range(1, self._TASK_B_NUM_OBJECTS + 1):
            if obj_idx in self._objects_done:
                continue
            pos = self._get_object_asset(obj_idx).data.root_pos_w[0]
            dx = float(pos[0].item()) - rx
            dy = float(pos[1].item()) - ry
            dist = (dx * dx + dy * dy) ** 0.5
            if best is None or dist < best[2]:
                best = (obj_idx, pos.clone(), dist)
        return best

    def _ensure_ik(self) -> None:
        if self._ik_ctrl is not None or self._env is None:
            return
        from atec_rl_lab.utils import CartesianController

        env = self._env.unwrapped
        robot = env.scene.articulations["robot"]
        self._arm_ids, _ = robot.find_joints(self._TASK_B_ARM_JOINTS)
        self._gripper_ids, _ = robot.find_joints(self._TASK_B_GRIPPER_JOINTS)
        self._default_joint_pos = robot.data.default_joint_pos.clone()
        self._ik_ctrl = CartesianController(
            robot=robot,
            ee_body_name=self._TASK_B_EE_BODY,
            arm_joint_names=self._TASK_B_ARM_JOINTS,
            num_envs=env.num_envs,
            device=str(env.device),
        )
        self._ik_ctrl.reset()

    def _start_pick(self, obj_idx: int) -> None:
        obj = self._get_object_asset(obj_idx)
        obj_quat = obj.data.root_state_w[0, 3:7]
        grasp_quat = _compute_grasp_quat(obj_quat, self.device)
        self._pick_obj_idx = obj_idx
        self._pick_sm = _TaskBPickSM(grasp_quat, self.device)
        self._task_b_phase = "pick"
        self._last_pick_state = "PRE_GRASP"
        self._ensure_ik()
        self._ik_ctrl.reset()

    def _update_task_b_phase(self) -> None:
        if not self._task_b_ik_enabled or self._env is None:
            return
        if self._task_b_phase != "nav":
            if self._task_b_phase == "carry":
                robot_xy = self._robot_xy_sim()
                if robot_xy is None:
                    return
                dx = self._TASK_B_DROP[0] - robot_xy[0]
                dy = self._TASK_B_DROP[1] - robot_xy[1]
                if (dx * dx + dy * dy) ** 0.5 < self.task_b_wp_tol:
                    self._task_b_phase = "drop"
                    self._drop_open_steps = 60
                    self._ensure_ik()
                    self._ik_ctrl.reset()
            return

        nearest = self._nearest_available_object()
        if nearest is None:
            return
        obj_idx, _, dist = nearest
        if dist <= self._TASK_B_APPROACH_DIST:
            self._start_pick(obj_idx)

    def _task_b_nav_target(self) -> tuple[float, float]:
        if self._task_b_ik_enabled:
            if self._task_b_phase == "carry":
                return self._TASK_B_DROP
            nearest = self._nearest_available_object()
            if nearest is not None:
                pos = nearest[1]
                return float(pos[0].item()), float(pos[1].item())
            return self._TASK_B_DROP
        tx, ty = self._TASK_B_WAYPOINTS[self._wp_idx]
        return tx, ty

    def _nav_task_b(self, wz, device, dtype):
        """Turn toward waypoint first; only drive forward once aligned."""
        if self._task_b_ik_enabled:
            self._update_task_b_phase()
            if self._task_b_phase in ("pick", "drop"):
                z = torch.zeros((1, 1), device=device, dtype=dtype)
                return z, z, z

        tx, ty = self._task_b_nav_target()
        dx = torch.tensor([[float(tx)]], device=device, dtype=dtype) - self.pos_x
        dy = torch.tensor([[float(ty)]], device=device, dtype=dtype) - self.pos_y
        dist = torch.sqrt(dx * dx + dy * dy)

        if not self._task_b_ik_enabled and dist.item() < self.task_b_wp_tol:
            self._wp_idx = (self._wp_idx + 1) % len(self._TASK_B_WAYPOINTS)
            self._task_b_mode = "turn"
            tx, ty = self._TASK_B_WAYPOINTS[self._wp_idx]
            dx = torch.tensor([[float(tx)]], device=device, dtype=dtype) - self.pos_x
            dy = torch.tensor([[float(ty)]], device=device, dtype=dtype) - self.pos_y
        elif self._task_b_ik_enabled and self._task_b_phase == "carry" and dist.item() < self.task_b_wp_tol:
            self._task_b_mode = "hold"

        desired_yaw = torch.atan2(dy, dx)
        yaw_err = torch.atan2(
            torch.sin(desired_yaw - self.yaw_est),
            torch.cos(desired_yaw - self.yaw_est),
        )
        yaw_abs = yaw_err.abs().item()

        if yaw_abs > self.task_b_yaw_align:
            self._task_b_mode = "turn"
            vx_cmd = torch.zeros((1, 1), device=device, dtype=dtype)
            vy_cmd = torch.zeros((1, 1), device=device, dtype=dtype)
            yaw_cmd = (
                self.task_b_k_yaw_turn * yaw_err - self.task_b_k_wz * wz
            ).clamp(-self.task_b_wz_lim, self.task_b_wz_lim)
        else:
            self._task_b_mode = "drive"
            vx_cmd = torch.full((1, 1), float(self.task_b_vx), device=device, dtype=dtype)
            vy_cmd = torch.zeros((1, 1), device=device, dtype=dtype)
            yaw_cmd = (
                self.task_b_k_yaw_drive * yaw_err - self.task_b_k_wz * wz
            ).clamp(-self.task_b_wz_lim * 0.5, self.task_b_wz_lim * 0.5)

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

        self._ensure_odom(device, dtype, batch_size)
        self._integrate_odom(vx_body, vy_body, wz)

        if self.nav_task == "B":
            vx_cmd, vy_cmd, yaw_cmd = self._nav_task_b(wz, device, dtype)
        else:
            vx_cmd, vy_cmd, yaw_cmd = self._nav_task_a(
                vy_body, wz, gravity_y, device, dtype, batch_size
            )

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
            tx, ty = self._task_b_nav_target()
            dx = tx - self.pos_x[0, 0].item()
            dy = ty - self.pos_y[0, 0].item()
            dist = (dx * dx + dy * dy) ** 0.5
            lines.append(
                f"phase={self._task_b_phase}  mode={self._task_b_mode}  "
                f"tgt=({tx:.1f},{ty:.1f})  dist={dist:.2f}m"
            )
            if self._task_b_ik_enabled:
                lines.append(
                    f"ik={'on' if self._env is not None else 'off'}  "
                    f"pick_obj={self._pick_obj_idx}  pick_state={self._last_pick_state}  "
                    f"done={len(self._objects_done)}/{self._TASK_B_NUM_OBJECTS}"
                )
            else:
                next_idx = (self._wp_idx + 1) % len(self._TASK_B_WAYPOINTS)
                nx, ny = self._TASK_B_WAYPOINTS[next_idx]
                lines.append(f"next_tgt[{next_idx}]=({nx:.1f},{ny:.1f})")
        return lines

    def _gripper_tensor(self, cmd: str, device: str, dtype) -> torch.Tensor:
        vals = self._TASK_B_GRIPPER_OPEN if cmd == "open" else self._TASK_B_GRIPPER_CLOSE
        return torch.tensor([list(vals)], device=device, dtype=dtype)

    def _arm_targets_to_env_action(self, arm_jpos_des: torch.Tensor, gripper_target: torch.Tensor) -> torch.Tensor:
        full_target = self._default_joint_pos.clone()
        full_target[:, self._arm_ids] = arm_jpos_des
        full_target[:, self._gripper_ids] = gripper_target
        env_action = (full_target - self._default_joint_pos) / self._TASK_B_IK_ACTION_SCALE
        arm_env = env_action[:, self.arm_joint_indices]
        return arm_env.to(device=self.device, dtype=torch.float32)

    def _task_b_ik_arm_action(self) -> torch.Tensor | None:
        if self._env is None:
            return None
        self._ensure_ik()
        robot = self._env.unwrapped.scene.articulations["robot"]
        device = str(robot.device)
        dtype = robot.data.joint_pos.dtype

        if self._task_b_phase == "pick" and self._pick_sm is not None and self._pick_obj_idx is not None:
            obj_pos = self._get_object_asset(self._pick_obj_idx).data.root_pos_w[0].clone()
            ee_pos_des, ee_quat_des, gripper_cmd = self._pick_sm.tick(obj_pos)
            self._last_pick_state = self._pick_sm.state
            arm_jpos_des = self._ik_ctrl.compute(
                ee_pos_des.unsqueeze(0).to(device=device, dtype=dtype),
                ee_quat_des.unsqueeze(0).to(device=device, dtype=dtype),
            )
            gripper_target = self._gripper_tensor(gripper_cmd, device, dtype)
            arm_env = self._arm_targets_to_env_action(arm_jpos_des, gripper_target)
            self._hold_arm_action = arm_env.clone()
            if self._pick_sm.done:
                self._objects_done.add(self._pick_obj_idx)
                self._pick_obj_idx = None
                self._pick_sm = None
                self._task_b_phase = "carry"
                self._task_b_mode = "turn"
            return arm_env

        if self._task_b_phase == "drop":
            drop_pos = torch.tensor(
                [[self._TASK_B_DROP[0], self._TASK_B_DROP[1], 0.25]],
                device=device,
                dtype=dtype,
            )
            drop_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype)
            arm_jpos_des = self._ik_ctrl.compute(drop_pos, drop_quat)
            gripper_target = self._gripper_tensor("open", device, dtype)
            arm_env = self._arm_targets_to_env_action(arm_jpos_des, gripper_target)
            self._drop_open_steps -= 1
            if self._drop_open_steps <= 0:
                self._task_b_phase = "nav"
                self._hold_arm_action = None
            return arm_env

        if self._task_b_phase == "carry" and self._hold_arm_action is not None:
            return self._hold_arm_action.clone()

        return None

    def _extract_policy_obs(self, obs, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        base_ang_vel = proprio[:, 3:6]
        projected_gravity = proprio[:, 9:12]
        idx = 12

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
        if self.nav_task == "B" and self._task_b_ik_enabled and self._env is not None:
            arm_env = self._task_b_ik_arm_action()
            if arm_env is not None:
                action_env[:, self.arm_joint_indices] = arm_env
        return {"action": action_env.cpu().numpy().tolist(), "giveup": False}
