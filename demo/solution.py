import math
import os

import torch


def build_taskd_nominal_waypoints(
    nav_steps: list[dict],
    start_xy: tuple[float, float] = (-3.0, 0.0),
    box_spawn_xy: tuple[float, float] = (-3.0, 1.6),
) -> list[tuple[float, float]]:
    """Build nominal world-frame polyline from scripted nav_steps."""
    x, y = float(start_xy[0]), float(start_xy[1])
    box_x, box_y = float(box_spawn_xy[0]), float(box_spawn_xy[1])
    waypoints: list[tuple[float, float]] = [(x, y)]
    for step in nav_steps:
        if step.get("match_box_x_tol") is not None:
            x = box_x
        elif step.get("match_box_y_tol") is not None:
            y = box_y
        elif step.get("box_x_stop") is not None:
            x = float(step["box_x_stop"])
        else:
            axis = step.get("axis", "x")
            sign = float(step.get("sign", 1.0))
            dist = float(step.get("dist", 0.0))
            if axis == "xy":
                # Legacy diagonal step: dist along combined axis.
                scale = math.sqrt(2.0)
                dx = dist / scale
                dy = dist / scale
                x += dx if sign > 0 else -dx
                y += dy if sign > 0 else -dy
            elif axis == "x":
                x += dist if sign > 0 else -dist
            else:
                y += dist if sign > 0 else -dist
        waypoints.append((x, y))
    return waypoints


def _point_to_segment_distance(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    vx, vy = x1 - x0, y1 - y0
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1.0e-12:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * vx + (py - y0) * vy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x0 + t * vx
    proj_y = y0 + t * vy
    return math.hypot(px - proj_x, py - proj_y)


def distance_to_reference_trajectory(
    x: float,
    y: float,
    waypoints: list[tuple[float, float]],
) -> float:
    if len(waypoints) < 2:
        return 0.0
    return min(
        _point_to_segment_distance(x, y, waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1])
        for i in range(len(waypoints) - 1)
    )


def trajectory_deviation_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    waypoints: list[tuple[float, float]],
) -> torch.Tensor:
    """Minimum distance from each env position to the nominal trajectory polyline."""
    if len(waypoints) < 2:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    dev = torch.full((x.shape[0],), float("inf"), device=x.device, dtype=x.dtype)
    px = x.squeeze(-1)
    py = y.squeeze(-1)
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        vx = x1 - x0
        vy = y1 - y0
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq <= 1.0e-12:
            dist = torch.hypot(px - x0, py - y0)
        else:
            t = ((px - x0) * vx + (py - y0) * vy) / seg_len_sq
            t = torch.clamp(t, 0.0, 1.0)
            proj_x = x0 + t * vx
            proj_y = y0 + t * vy
            dist = torch.hypot(px - proj_x, py - proj_y)
        dev = torch.minimum(dev, dist)
    return dev


class AlgSolution:
    """Task D: scripted approach via a tunable nav step table → push_ready."""

    ROBOT_SPAWN_XY = (-3.0, 0.0)
    BOX_SPAWN_XY = (-3.0, 1.6)
    BOX_SPAWN_Z = 0.5
    FALL_MIN_HEIGHT = 0.25
    PUSH_YAW = 0.0
    BOX_HALF_X = 0.40
    BOX_HALF_Y = 0.50
    ENABLE_TRAJECTORY_TERMINATION = True
    TRAJECTORY_DEVIATION_TOL = 1.50  # meters, <=0 disables

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
        self.high_level_hz = 2.0
        self.nav_k_yaw = 1.2
        self.nav_k_wz = 0.25
        self.nav_wz_lim = 0.35
        self.nav_max_steps = 500  # per-step timeout (~10 s @ 0.02 s)

        # push correction (forward advance2 + lateral sidestep_right)
        self.push_enable_lidar_correction = False  # True: fl/fr/right radar → yaw/vy
        self.push_k_yaw = 0.9          # follow box heading (rad/s per rad error)
        self.push_k_lidar_skew = 0.5   # forward push: (fl - fr) → yaw
        self.push_k_lateral = 0.35     # forward push: body y offset → vy
        self.push_k_world_y_align = 0.6  # forward push: keep robot_y close to box_y
        self.push_k_lateral_sidestep = 0.45  # lateral push: body y offset → vy
        self.push_k_lidar_right = 0.35  # lateral push: right radar (fallback, no sim box)
        self.push_k_lidar_lat_skew = 0.4  # lateral push: (fr - fl) → yaw
        self.push_k_longitudinal = 0.35   # lateral push: keep box alongside → vx
        self.push_k_world_x_align = 0.6   # lateral push: keep robot_x close to box_x
        self.push_lateral_lidar_target = 0.14  # desired (right - ground) when in contact
        self.push_target_bx_body = 0.35  # want box slightly ahead during lateral push
        self.push_lateral_vy_min = 0.75  # never drop below this |vy| during lateral push
        self.push_vx_min_scale = 0.45
        self.push_vy_min_scale = 0.55
        self.push_skew_slowdown = 0.55
        self.push_yaw_lim = 0.30

        # ------------------------------------------------------------------
        # Navigation script — edit here only
        # Each step: name, world axis ("x"|"y"), sign (+1 / -1), dist (m), vx, vy
        #   axis "x", sign +1  → forward;  sign -1 → backward (后移)
        #   axis "y", sign +1  → left;     sign -1 → right (右移)
        #   push=True → dist tracks box along step axis, auto skew correction
        #   forward push (axis x): box yaw + geom; optional fl/fr lidar
        #   lateral push (axis y): box yaw + geom; optional fr/fl/right lidar
        # ------------------------------------------------------------------
        self.nav_steps = [
            dict(name="retreat",         axis="x", sign=-0.7, dist=1.0, vx=-2.0, vy=0.0),
            dict(name="sidestep_left",  axis="y", sign=+1, dist=2.1, vx=0.0,  vy=1.0),
            # match_box_x_tol: move until |robot_x - box_x| <= tol (dist ignored for this step)
            dict(name="advance",        axis="x", sign=+0.7, dist=1.0, vx=2.0,  vy=0.0, match_box_x_tol=0.15),
            dict(
                name="sidestep_right",
                axis="y",
                sign=-1,
                dist=2.0,
                vx=0.0,
                vy=-1.0,
                push=True,
                align_x_with_box=True,
            ),
            dict(name="retreat2",       axis="x", sign=-1, dist=1.0, vx=-2.0, vy=0.0),
            dict(name="sidestep_right2", axis="y", sign=-1, dist=0.6, vx=0.0, vy=-1.0, match_box_y_tol=0.25),
            dict(
                name="advance2",
                axis="x",
                sign=+1,
                dist=4.0,
                vx=1.5,
                vy=0.0,
                push=True,
                align_y_with_box=True,
                # Keep this push conservative; stop earlier to avoid over-dropping the box.
                box_x_slow_start=-1.7,  # start slowing down when robot pos.x approaches this value
                box_x_stop=-1.25,       # stop/push-ready when robot pos.x reaches this value
                stop_tol=0.08,
                wait_after_box_drop_z=-0.35,  # transition only after box drops below this z.
                wait_after_box_drop_s=1.0,    # then hold still >= 1s before entering final stages.
            ),

            dict(
                name="final",
                axis="x",
                sign=+1,
                dist=3.0,
                vx=0.3,
                vy=0.0,
                # Crossing step: do NOT use push correction here, just go straight and slow.
                push=False,
            ),
        ]
        '''self.nav_steps = [
            dict(name="retreat1",         axis="xy", sign=-1, dist=2.23, vx=-1.0, vy=2.0),
            #dict(name="sidestep_left",  axis="y", sign=+1, dist=2.0, vx=0.0,  vy=1.0),
            #dict(name="advance",        axis="x", sign=+1, dist=1.0, vx=2.0,  vy=0.0), 
            #dict(name="sidestep_right", axis="y", sign=-1, dist=3.0, vx=1.0,  vy=-1.0, push=True),
            #dict(name="retreat2",       axis="x", sign=-1, dist=1.0, vx=-2.0, vy=0.0),
            #dict(name="sidestep_right2", axis="y", sign=-1, dist=0.6, vx=0.0, vy=-1.0),
            #dict(name="advance2", axis="x", sign=+1, dist=4.0, vx=1.5, vy=0.0, push=True),
        ]'''

        self._reference_waypoints = build_taskd_nominal_waypoints(
            self.nav_steps,
            start_xy=self.ROBOT_SPAWN_XY,
            box_spawn_xy=self.BOX_SPAWN_XY,
        )
        self._reset_nav_state()

    def _reset_nav_state(self) -> None:
        # Per-env navigation state (allocated lazily on first batch).
        self._pe_batch_size = 0
        self._nav_step_idx = None
        self._nav_step_steps = None
        self._nav_origin = None
        self._nav_origin_valid = None
        self._nav_step_target = None
        self._step_wait_counter = None
        self._step_wait_armed = None
        self._approach_done_batch = None
        self.phase = "push_ready" if not self.nav_steps else self.nav_steps[0]["name"]
        self._nav_step_progress = 0.0

        self.x_est = None
        self.y_est = None
        self.yaw_est = None
        self._env = None
        self._robot = None
        self._box = None
        self._extero = None
        self._debug_step = 0
        self._last_nav_cmd = (0.0, 0.0, 0.0)
        self._last_high_level_cmd_batch = None
        self._hl_cmd_step_counter = 0
        self._hl_cmd_hold_steps = max(1, int(round(1.0 / (self.high_level_hz * self.dt))))
        self._hl_cmd_cached = None
        self._hl_cmd_force_refresh = True
        self._last_pos_xy = (self.ROBOT_SPAWN_XY[0], self.ROBOT_SPAWN_XY[1])
        self._last_box_xy = (0.0, 0.0)
        self._last_box_z = self.BOX_SPAWN_Z
        self._last_root_z = self.FALL_MIN_HEIGHT
        self._last_front_lidar = 0.0
        self._last_box_in_front = False
        self._last_box_geom = False
        self._last_box_lidar = False
        self._last_lidar_fl = 0.0
        self._last_lidar_fr = 0.0
        self._last_lidar_right = 0.0
        self._last_lidar_delta = 0.0
        self._last_has_extero = False
        self._last_box_det_src = "none"
        self._last_box_rel_yaw = 0.0
        self._last_push_skew = 0.0
        self._last_push_right_err = 0.0
        self._last_align_x_err = 0.0
        self._last_align_x_delta = 0.0
        self._last_align_y_delta = 0.0
        self._push_mode = "none"
        self._last_stage_idx_batch = None
        self._trajectory_deviated_batch = None
        self._last_trajectory_dev = 0.0

    def _ensure_pe_state(self, batch_size: int, device, dtype) -> None:
        if (
            self._nav_step_idx is not None
            and self._pe_batch_size == batch_size
            and self._nav_step_idx.device == device
        ):
            return
        self._pe_batch_size = batch_size
        self._nav_step_idx = torch.zeros((batch_size,), device=device, dtype=torch.long)
        self._nav_step_steps = torch.zeros((batch_size,), device=device, dtype=torch.long)
        self._nav_origin = torch.zeros((batch_size, 2), device=device, dtype=dtype)
        self._nav_origin_valid = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        self._nav_step_target = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        self._step_wait_counter = torch.zeros((batch_size,), device=device, dtype=torch.long)
        self._step_wait_armed = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        self._approach_done_batch = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        self._trajectory_deviated_batch = torch.zeros((batch_size,), device=device, dtype=torch.bool)

    @property
    def _approach_done(self) -> bool:
        if self._approach_done_batch is None:
            return False
        return bool(self._approach_done_batch.all().item())

    def get_stage_batch(self) -> list[int] | None:
        if self._nav_step_idx is None:
            return None
        return self._nav_step_idx.detach().cpu().tolist()

    def _phase_name_for_env(self, env_idx: int) -> str:
        if self._nav_step_idx is None:
            return self.phase
        step_idx = int(self._nav_step_idx[env_idx].item())
        if step_idx >= len(self.nav_steps):
            return "push_ready"
        return self.nav_steps[step_idx]["name"]

    def _sync_phase_from_env0(self) -> None:
        if self._nav_step_idx is None:
            return
        self.phase = self._phase_name_for_env(0)
        if self._nav_step_idx.numel() > 1:
            unique = self._nav_step_idx.unique()
            if unique.numel() > 1:
                self.phase = f"{self.phase}+mixed"

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
        self._reset_nav_state()

    def reset_env_batch(self, env_mask: torch.Tensor) -> None:
        """Reset navigation state for env indices that finished and auto-reset in sim."""
        if self._nav_step_idx is None or env_mask is None:
            return
        if not isinstance(env_mask, torch.Tensor):
            env_mask = torch.as_tensor(env_mask, device=self._nav_step_idx.device, dtype=torch.bool)
        env_mask = env_mask.view(-1).to(device=self._nav_step_idx.device, dtype=torch.bool)
        if env_mask.shape[0] != self._nav_step_idx.shape[0]:
            return
        if not bool(env_mask.any()):
            return
        self._nav_step_idx[env_mask] = 0
        self._nav_step_steps[env_mask] = 0
        self._nav_origin_valid[env_mask] = False
        self._step_wait_counter[env_mask] = 0
        self._step_wait_armed[env_mask] = False
        self._approach_done_batch[env_mask] = False
        if self._trajectory_deviated_batch is not None:
            self._trajectory_deviated_batch[env_mask] = False
        self._hl_cmd_force_refresh = True

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

    def _current_nav_step(self) -> dict | None:
        """Step config for env-0 (debug / overlay)."""
        if self._nav_step_idx is None:
            return self.nav_steps[0] if self.nav_steps else None
        step_idx = int(self._nav_step_idx[0].item())
        if step_idx >= len(self.nav_steps):
            return None
        return self.nav_steps[step_idx]

    def _begin_nav_step_batch(
        self,
        mask: torch.Tensor,
        step: dict,
        x: torch.Tensor,
        y: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
    ) -> None:
        if not bool(mask.any()):
            return
        axis = step.get("axis", "x")
        if self._is_push_step(step):
            ox, oy = box_x, box_y
        else:
            ox, oy = x, y
        origin = torch.cat([ox, oy], dim=-1)
        self._nav_origin[mask] = origin[mask]
        self._nav_step_target[mask] = float(step["dist"])
        self._nav_step_steps[mask] = 0
        self._step_wait_counter[mask] = 0
        self._step_wait_armed[mask] = False
        self._nav_origin_valid[mask] = True

    def _nav_progress_batch(
        self,
        step: dict,
        x: torch.Tensor,
        y: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
    ) -> torch.Tensor:
        axis = step.get("axis", "x")
        if self._is_push_step(step):
            cx, cy = box_x, box_y
        else:
            cx, cy = x, y

        if axis == "xy":
            coord = torch.cat([cx, cy], dim=-1)
            origin = self._nav_origin
            return torch.linalg.norm(coord - origin, dim=-1, keepdim=True)
        if axis == "x":
            coord = cx
            origin = self._nav_origin[:, 0:1]
        else:
            coord = cy
            origin = self._nav_origin[:, 1:2]

        if step["sign"] > 0:
            return coord - origin
        return origin - coord

    def _advance_nav_step_batch(self, mask: torch.Tensor) -> None:
        if not bool(mask.any()):
            return
        self._nav_step_idx[mask] += 1
        self._nav_origin_valid[mask] = False
        self._nav_step_steps[mask] = 0
        self._step_wait_counter[mask] = 0
        self._step_wait_armed[mask] = False
        finished = mask & (self._nav_step_idx >= len(self.nav_steps))
        if bool(finished.any()):
            self._approach_done_batch[finished] = True

    def _compute_drop_wait_reached(
        self,
        mask: torch.Tensor,
        step: dict,
        box_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reached, holding) per-env for wait_after_box_drop_* steps."""
        wait_z = step.get("wait_after_box_drop_z", None)
        wait_s = step.get("wait_after_box_drop_s", None)
        if wait_z is None or wait_s is None:
            return torch.zeros_like(mask), torch.zeros_like(mask)

        dropped = box_z[:, 0] <= float(wait_z)
        reached = torch.zeros_like(mask)
        holding = torch.zeros_like(mask)

        active = mask & dropped
        if bool(active.any()):
            newly_armed = active & (~self._step_wait_armed)
            self._step_wait_armed[newly_armed] = True
            self._step_wait_counter[newly_armed] = 0

        not_dropped = mask & (~dropped)
        if bool(not_dropped.any()):
            self._step_wait_counter[not_dropped] = 0
            self._step_wait_armed[not_dropped] = False

        armed = mask & self._step_wait_armed
        if bool(armed.any()):
            wait_steps = max(1, int(round(float(wait_s) / self.dt)))
            self._step_wait_counter[armed] += 1
            reached = reached | (armed & (self._step_wait_counter >= wait_steps))
            holding = holding | (armed & (self._step_wait_counter < wait_steps))

        return reached, holding

    def _compute_step_commands(
        self,
        step: dict,
        mask: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
        box_z: torch.Tensor,
        device,
        dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute vx/vy/reached/holding/progress for envs on this nav step."""
        b = x.shape[0]
        vx = torch.zeros((b, 1), device=device, dtype=dtype)
        vy = torch.zeros((b, 1), device=device, dtype=dtype)
        reached = torch.zeros((b,), device=device, dtype=torch.bool)
        holding = torch.zeros((b,), device=device, dtype=torch.bool)
        progress = torch.zeros((b, 1), device=device, dtype=dtype)

        if not bool(mask.any()):
            return vx, vy, reached, holding, progress

        if self._is_match_box_x_step(step):
            tol = float(step["match_box_x_tol"])
            align_delta = box_x - x
            align_err = align_delta.abs()
            progress = align_err
            reached = (align_err[:, 0] <= tol) & mask
            vmax = abs(float(step["vx"]))
            kx = float(step.get("match_box_x_k", 1.0))
            min_speed = float(step.get("match_box_x_min_speed", 0.15))
            vx = (kx * (box_x - x)).clamp(-vmax, vmax)
            slow_mask = vx.abs() < min_speed
            vx = torch.where(
                slow_mask,
                torch.sign(vx).clamp(min=-1.0, max=1.0) * min_speed,
                vx,
            )
            vy = torch.full((b, 1), float(step["vy"]), device=device, dtype=dtype)
        elif self._is_match_box_y_step(step):
            tol = float(step["match_box_y_tol"])
            align_delta = box_y - y
            align_err = align_delta.abs()
            progress = align_err
            reached = (align_err[:, 0] <= tol) & mask
            vmax = abs(float(step["vy"]))
            ky = float(step.get("match_box_y_k", 1.0))
            min_speed = float(step.get("match_box_y_min_speed", 0.12))
            vy = (ky * (box_y - y)).clamp(-vmax, vmax)
            slow_mask = vy.abs() < min_speed
            vy = torch.where(
                slow_mask,
                torch.sign(vy).clamp(min=-1.0, max=1.0) * min_speed,
                vy,
            )
            vx = torch.full((b, 1), float(step["vx"]), device=device, dtype=dtype)
        else:
            progress = self._nav_progress_batch(step, x, y, box_x, box_y)
            reached = (progress[:, 0] >= self._nav_step_target[:, 0]) & mask
            vx = torch.full((b, 1), float(step["vx"]), device=device, dtype=dtype)
            box_x_slow_start = step.get("box_x_slow_start", None)
            box_x_stop = step.get("box_x_stop", None)
            if (
                box_x_slow_start is not None
                and box_x_stop is not None
                and step.get("axis") == "x"
            ):
                slow_start = float(box_x_slow_start)
                stop_x = float(box_x_stop)
                denom = max(abs(stop_x - slow_start), 1e-6)
                if step["sign"] > 0:
                    scale = ((stop_x - x) / denom).clamp(0.0, 1.0)
                else:
                    scale = ((x - stop_x) / denom).clamp(0.0, 1.0)
                vx = vx * scale
            vy = torch.full((b, 1), float(step["vy"]), device=device, dtype=dtype)

        box_x_stop = step.get("box_x_stop", None)
        if box_x_stop is not None and step.get("axis") == "x":
            stop_x = float(box_x_stop)
            stop_tol = float(step.get("stop_tol", 0.1))
            if step["sign"] > 0:
                reached = reached | ((x[:, 0] >= (stop_x - stop_tol)) & mask)
            else:
                reached = reached | ((x[:, 0] <= (stop_x + stop_tol)) & mask)

        drop_reached, drop_holding = self._compute_drop_wait_reached(mask, step, box_z)
        reached = reached | drop_reached
        holding = holding | drop_holding

        require_box_z_below = step.get("require_box_z_below", None)
        if require_box_z_below is not None:
            not_ready = mask & (box_z[:, 0] > float(require_box_z_below))
            holding = holding | not_ready
            reached = reached & (~not_ready)

        # Only apply commands on active envs for this step.
        m = mask.unsqueeze(-1)
        vx = torch.where(m, vx, torch.zeros_like(vx))
        vy = torch.where(m, vy, torch.zeros_like(vy))
        return vx, vy, reached, holding, progress

    def _sector_mean(self, ch0, ch1, ch2, center: int, half_width: int) -> torch.Tensor:
        h = self._LIDAR_H
        idx = torch.arange(center - half_width, center + half_width + 1, device=ch0.device) % h
        vals = torch.cat([ch0[idx], ch1[idx], ch2[idx]], dim=0).clamp(min=-1.0, max=3.0)
        k = max(1, vals.numel() // 5)
        return torch.topk(vals, k).values.mean()

    def _lidar_sectors_and_box(self, extero: torch.Tensor | None) -> dict:
        empty = {
            "has_extero": False,
            "front": self._LIDAR_GROUND_REF,
            "front_left": self._LIDAR_GROUND_REF,
            "front_right": self._LIDAR_GROUND_REF,
            "right": self._LIDAR_GROUND_REF,
            "delta": 0.0,
            "box_lidar": False,
        }
        if extero is None or extero.shape[-1] < self._LIDAR_H:
            return empty

        device = extero.device
        dtype = extero.dtype
        rays = extero[0].to(device=device, dtype=dtype)
        rays = rays.nan_to_num(nan=self._LIDAR_GROUND_REF, posinf=3.0, neginf=0.0)
        h = self._LIDAR_H
        ch0 = rays[0 * h : 1 * h]
        ch1 = rays[1 * h : 2 * h]
        ch2 = rays[2 * h : 3 * h]

        front = float(self._sector_mean(ch0, ch1, ch2, 180, self._LIDAR_FRONT_HALF).item())
        front_left = float(self._sector_mean(ch0, ch1, ch2, 230, 25).item())
        front_right = float(self._sector_mean(ch0, ch1, ch2, 130, 25).item())
        right = float(self._sector_mean(ch0, ch1, ch2, 90, 20).item())
        delta = abs(front - self._LIDAR_GROUND_REF)
        box_lidar = delta > self._LIDAR_BOX_DETECT_DELTA or front < -0.05
        return {
            "has_extero": True,
            "front": front,
            "front_left": front_left,
            "front_right": front_right,
            "right": right,
            "delta": delta,
            "box_lidar": box_lidar,
        }

    def _box_in_front_geometric(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        yaw: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
    ) -> torch.Tensor:
        dx = box_x - x
        dy = box_y - y
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        bx_body = cos_y * dx + sin_y * dy
        by_body = -sin_y * dx + cos_y * dy
        half_x = self.BOX_HALF_X + 0.15
        half_y = self.BOX_HALF_Y + 0.20
        return (bx_body > -half_x) & (bx_body < 2.5) & (by_body.abs() < half_y)

    def _is_push_step(self, step: dict | None) -> bool:
        return step is not None and bool(step.get("push", False))

    def _push_mode_for_step(self, step: dict | None) -> str:
        if not self._is_push_step(step):
            return "none"
        mode = step.get("push_mode")
        if mode in ("forward", "lateral"):
            return mode
        return "lateral" if step.get("axis") == "y" else "forward"

    def _is_match_box_x_step(self, step: dict | None) -> bool:
        return step is not None and ("match_box_x_tol" in step)

    def _is_match_box_y_step(self, step: dict | None) -> bool:
        return step is not None and ("match_box_y_tol" in step)

    def _box_relative_yaw(
        self, yaw: torch.Tensor, device, dtype
    ) -> torch.Tensor | None:
        if self._box is None:
            return None
        box_yaw = self._yaw_from_quat_wxyz(
            self._box.data.root_quat_w.to(device=device, dtype=dtype)
        )
        return torch.atan2(torch.sin(box_yaw - yaw), torch.cos(box_yaw - yaw))

    def _box_lateral_offset_body(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        yaw: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
    ) -> torch.Tensor:
        dx = box_x - x
        dy = box_y - y
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        return -sin_y * dx + cos_y * dy

    def _box_longitudinal_offset_body(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        yaw: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
    ) -> torch.Tensor:
        dx = box_x - x
        dy = box_y - y
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        return cos_y * dx + sin_y * dy

    def _yaw_correction_from_box(
        self,
        yaw_cmd: torch.Tensor,
        yaw: torch.Tensor,
        device,
        dtype,
    ) -> torch.Tensor:
        rel_yaw = self._box_relative_yaw(yaw, device, dtype)
        yaw_corr = torch.zeros_like(yaw_cmd)
        if rel_yaw is not None:
            self._last_box_rel_yaw = float(rel_yaw[0, 0].item())
            yaw_corr = yaw_corr + self.push_k_yaw * rel_yaw
        else:
            self._last_box_rel_yaw = 0.0
        return yaw_corr

    def _apply_forward_push_correction(
        self,
        vx_cmd: torch.Tensor,
        vy_cmd: torch.Tensor,
        yaw_cmd: torch.Tensor,
        step: dict,
        yaw: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
        lidar: dict,
        device,
        dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep robot aligned behind the box while pushing forward."""
        rel_yaw = self._box_relative_yaw(yaw, device, dtype)
        yaw_corr = self._yaw_correction_from_box(yaw_cmd, yaw, device, dtype)
        skew = 0.0
        if self.push_enable_lidar_correction and lidar.get("has_extero"):
            skew = float(lidar["front_left"] - lidar["front_right"])

        self._last_push_skew = skew
        self._last_push_right_err = 0.0
        if self.push_enable_lidar_correction and abs(skew) > 1e-4:
            yaw_corr = yaw_corr + self.push_k_lidar_skew * torch.full_like(
                yaw_cmd, skew, device=device, dtype=dtype
            )

        by_body = self._box_lateral_offset_body(x, y, yaw, box_x, box_y)
        vy_corr = (-self.push_k_lateral * by_body).clamp(-0.35, 0.35)
        if bool(step.get("align_y_with_box", False)):
            # World-y alignment: keep robot and box y close during forward push.
            y_err_world = (box_y - y).clamp(-0.4, 0.4)
            self._last_align_y_delta = float(y_err_world[0, 0].item())
            vy_corr = vy_corr + (self.push_k_world_y_align * y_err_world).clamp(-0.35, 0.35)
        else:
            self._last_align_y_delta = 0.0

        yaw_cmd = (yaw_cmd + yaw_corr).clamp(-self.push_yaw_lim, self.push_yaw_lim)
        vy_cmd = (vy_cmd + vy_corr).clamp(-0.35, 0.35)

        if rel_yaw is not None:
            yaw_mag = rel_yaw.abs()
        else:
            yaw_mag = torch.zeros_like(yaw)
        vx_scale = torch.clamp(
            1.0 - self.push_skew_slowdown * yaw_mag,
            min=self.push_vx_min_scale,
        )
        vx_cmd = vx_cmd * vx_scale
        return vx_cmd, vy_cmd, yaw_cmd

    def _apply_lateral_push_correction(
        self,
        vx_cmd: torch.Tensor,
        vy_cmd: torch.Tensor,
        yaw_cmd: torch.Tensor,
        step: dict,
        yaw: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
        lidar: dict,
        device,
        dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sidestep push: geom by_body primary, right radar as fallback."""
        base_vy = float(step["vy"])
        vy_lim = max(abs(base_vy), 1.0)
        vy_sign = 1.0 if base_vy >= 0.0 else -1.0

        rel_yaw = self._box_relative_yaw(yaw, device, dtype)
        yaw_corr = self._yaw_correction_from_box(yaw_cmd, yaw, device, dtype)

        right_signal = self._LIDAR_GROUND_REF
        lat_skew = 0.0
        if self.push_enable_lidar_correction and lidar.get("has_extero"):
            right_signal = max(float(lidar["right"]), float(lidar["front_right"]))
            lat_skew = float(lidar["front_right"] - lidar["front_left"])

        right_err = self.push_lateral_lidar_target - (right_signal - self._LIDAR_GROUND_REF)
        self._last_push_right_err = right_err if self.push_enable_lidar_correction else 0.0
        self._last_push_skew = lat_skew

        if self.push_enable_lidar_correction and abs(lat_skew) > 1e-4:
            yaw_corr = yaw_corr + self.push_k_lidar_lat_skew * torch.full_like(
                yaw_cmd, lat_skew, device=device, dtype=dtype
            )

        by_body = self._box_lateral_offset_body(x, y, yaw, box_x, box_y)
        bx_body = self._box_longitudinal_offset_body(x, y, yaw, box_x, box_y)

        # Box in front during sidestep: center laterally via geom, not right-only lidar.
        vy_corr = (-self.push_k_lateral_sidestep * by_body).clamp(-0.25, 0.25)
        if (
            self.push_enable_lidar_correction
            and self._box is None
            and lidar.get("has_extero")
        ):
            vy_corr = vy_corr + self.push_k_lidar_right * right_err * vy_sign

        vx_corr = (self.push_k_longitudinal * (self.push_target_bx_body - bx_body)).clamp(-0.35, 0.35)
        if bool(step.get("align_x_with_box", False)):
            # During lateral push, allow slight forward/backward correction to keep robot x aligned with box x.
            x_err_world = (box_x - x).clamp(-0.4, 0.4)
            self._last_align_x_delta = float(x_err_world[0, 0].item())
            vx_corr = vx_corr + (self.push_k_world_x_align * x_err_world).clamp(-0.35, 0.35)

        yaw_cmd = (yaw_cmd + yaw_corr).clamp(-self.push_yaw_lim, self.push_yaw_lim)
        vy_cmd = (vy_cmd + vy_corr).clamp(-vy_lim, vy_lim)
        vx_cmd = (vx_cmd + vx_corr).clamp(-0.5, 0.5)

        if rel_yaw is not None:
            yaw_mag = rel_yaw.abs()
            self._last_box_rel_yaw = float(rel_yaw[0, 0].item())
        else:
            yaw_mag = torch.zeros_like(yaw)
            self._last_box_rel_yaw = 0.0
        vy_scale = torch.clamp(
            1.0 - self.push_skew_slowdown * yaw_mag,
            min=self.push_vy_min_scale,
        )
        vy_cmd = vy_cmd * vy_scale

        min_vy = self.push_lateral_vy_min * vy_sign
        slow_mask = vy_cmd.abs() < self.push_lateral_vy_min
        vy_cmd = torch.where(
            slow_mask,
            torch.full_like(vy_cmd, min_vy) * vy_scale,
            vy_cmd,
        )
        return vx_cmd, vy_cmd, yaw_cmd

    def _apply_push_box_correction(
        self,
        vx_cmd: torch.Tensor,
        vy_cmd: torch.Tensor,
        yaw_cmd: torch.Tensor,
        step: dict,
        yaw: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        box_x: torch.Tensor,
        box_y: torch.Tensor,
        lidar: dict,
        device,
        dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = self._push_mode_for_step(step)
        self._push_mode = mode
        if mode == "lateral":
            return self._apply_lateral_push_correction(
                vx_cmd,
                vy_cmd,
                yaw_cmd,
                step,
                yaw,
                x,
                y,
                box_x,
                box_y,
                lidar,
                device,
                dtype,
            )
        return self._apply_forward_push_correction(
            vx_cmd,
            vy_cmd,
            yaw_cmd,
            step,
            yaw,
            x,
            y,
            box_x,
            box_y,
            lidar,
            device,
            dtype,
        )

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
        self._ensure_pe_state(b, device, dtype)

        x, y, yaw = self._robot_xy_yaw(proprio)
        if self._robot is not None:
            root_z = self._robot.data.root_pos_w[:, 2:3].to(device=device, dtype=dtype)
        else:
            root_z = torch.full((b, 1), self.FALL_MIN_HEIGHT, device=device, dtype=dtype)
        box_x, box_y = self._box_pose_xy(device, dtype)
        self._last_pos_xy = (float(x[0, 0].item()), float(y[0, 0].item()))
        self._last_root_z = float(root_z[0, 0].item())
        if self._box is not None:
            box_z = self._box.data.root_pos_w[:, 2:3].to(device=device, dtype=dtype)
        else:
            box_z = torch.full((b, 1), self.BOX_SPAWN_Z, device=device, dtype=dtype)
        self._last_box_xy = (float(box_x[0, 0].item()), float(box_y[0, 0].item()))
        self._last_box_z = float(box_z[0, 0].item())
        self._last_stage_idx_batch = self._nav_step_idx.detach().clone()
        if self.ENABLE_TRAJECTORY_TERMINATION and self.TRAJECTORY_DEVIATION_TOL > 0.0:
            dev = trajectory_deviation_batch(x, y, self._reference_waypoints)
            self._last_trajectory_dev = float(dev[0].item())
            self._trajectory_deviated_batch = dev > float(self.TRAJECTORY_DEVIATION_TOL)
        else:
            self._last_trajectory_dev = 0.0
            self._trajectory_deviated_batch = torch.zeros((b,), device=device, dtype=torch.bool)

        lidar = self._lidar_sectors_and_box(self._extero)
        front_val = lidar["front"]
        box_lidar = lidar["box_lidar"]
        box_geom = self._box_in_front_geometric(x, y, yaw, box_x, box_y)
        if self._box is not None:
            box_in_front = bool(box_geom.all())
            box_det_src = "geom"
        else:
            box_in_front = box_lidar
            box_det_src = "lidar" if lidar["has_extero"] else "none"

        self._last_front_lidar = front_val
        self._last_lidar_fl = lidar["front_left"]
        self._last_lidar_fr = lidar["front_right"]
        self._last_lidar_right = lidar.get("right", self._LIDAR_GROUND_REF)
        self._last_lidar_delta = lidar["delta"]
        self._last_has_extero = lidar["has_extero"]
        self._last_box_lidar = bool(box_lidar)
        self._last_box_geom = bool(box_geom.all())
        self._last_box_in_front = bool(box_in_front)
        self._last_box_det_src = box_det_src

        yaw_cmd = self._yaw_hold_cmd(yaw, wz, device, dtype)
        vx_cmd = torch.zeros((b, 1), device=device, dtype=dtype)
        vy_cmd = torch.zeros((b, 1), device=device, dtype=dtype)

        push_ready_mask = (self._nav_step_idx >= len(self.nav_steps)) | self._approach_done_batch
        active_mask = ~push_ready_mask
        if self._trajectory_deviated_batch is not None:
            active_mask = active_mask & (~self._trajectory_deviated_batch)

        need_begin = active_mask & (~self._nav_origin_valid)
        if bool(need_begin.any()):
            for s_idx, step in enumerate(self.nav_steps):
                begin_mask = need_begin & (self._nav_step_idx == s_idx)
                if bool(begin_mask.any()):
                    self._begin_nav_step_batch(begin_mask, step, x, y, box_x, box_y)

        self._nav_step_steps[active_mask] += 1

        advance_mask = torch.zeros((b,), device=device, dtype=torch.bool)
        for s_idx, step in enumerate(self.nav_steps):
            mask = active_mask & (self._nav_step_idx == s_idx)
            if not bool(mask.any()):
                continue

            vx_s, vy_s, reached, holding, progress = self._compute_step_commands(
                step, mask, x, y, box_x, box_y, box_z, device, dtype
            )
            timed_out = mask & (self._nav_step_steps >= self.nav_max_steps)
            to_advance = (reached | timed_out) & mask
            advance_mask = advance_mask | to_advance

            move_mask = mask & (~to_advance) & (~holding)
            m_move = move_mask.unsqueeze(-1)
            vx_step = vx_s
            vy_step = vy_s
            if self._is_push_step(step):
                vx_p, vy_p, yaw_p = self._apply_push_box_correction(
                    vx_step,
                    vy_step,
                    yaw_cmd,
                    step,
                    yaw,
                    x,
                    y,
                    box_x,
                    box_y,
                    lidar,
                    device,
                    dtype,
                )
                m = mask.unsqueeze(-1)
                vx_step = torch.where(m, vx_p, vx_step)
                vy_step = torch.where(m, vy_p, vy_step)
                yaw_cmd = torch.where(m, yaw_p, yaw_cmd)
            vx_cmd = torch.where(m_move, vx_step, vx_cmd)
            vy_cmd = torch.where(m_move, vy_step, vy_cmd)

            if bool((self._nav_step_idx == s_idx)[0]):
                    self._nav_step_progress = float(progress[0, 0].item())
                    if self._is_match_box_x_step(step):
                        self._last_align_x_delta = float((box_x - x)[0, 0].item())
                        self._last_align_x_err = float(progress[0, 0].item())
                    elif self._is_match_box_y_step(step):
                        self._last_align_y_delta = float((box_y - y)[0, 0].item())

        if bool(advance_mask.any()):
            self._advance_nav_step_batch(advance_mask)

        self._sync_phase_from_env0()

        cmd = torch.cat([vx_cmd, vy_cmd, yaw_cmd], dim=-1)
        self._last_nav_cmd = (
            float(cmd[0, 0].item()),
            float(cmd[0, 1].item()),
            float(cmd[0, 2].item()),
        )
        self._debug_step += 1
        if self._debug_step % 50 == 1:
            step_info = ""
            cur = self._current_nav_step()
            step_idx0 = int(self._nav_step_idx[0].item())
            if cur is not None and bool(self._nav_origin_valid[0].item()):
                if self._is_match_box_x_step(cur):
                    step_info = (
                        f" [{step_idx0 + 1}/{len(self.nav_steps)} "
                        f"{cur['name']} dx={self._last_align_x_delta:+.3f} x_err={self._last_align_x_err:.3f}"
                        f"/tol={float(cur['match_box_x_tol']):.3f} "
                        f"cmd=({cur['vx']:.1f},{cur['vy']:+.1f})]"
                    )
                elif self._is_match_box_y_step(cur):
                    step_info = (
                        f" [{step_idx0 + 1}/{len(self.nav_steps)} "
                        f"{cur['name']} dy={self._last_align_y_delta:+.3f} y_err={self._nav_step_progress:.3f}"
                        f"/tol={float(cur['match_box_y_tol']):.3f} "
                        f"cmd=({cur['vx']:.1f},{cur['vy']:+.1f})]"
                    )
                else:
                    step_info = (
                        f" [{step_idx0 + 1}/{len(self.nav_steps)} "
                        f"{cur['name']} prog={self._nav_step_progress:.2f}/{cur['dist']:.2f} "
                        f"cmd=({cur['vx']:.1f},{cur['vy']:+.1f})]"
                    )
            mixed = ""
            if self._nav_step_idx.unique().numel() > 1:
                mixed = f" mixed_steps={self._nav_step_idx.unique().numel()}"
            print(
                f"[taskd {self.phase} step={self._debug_step:5d}] "
                f"pos=({x[0,0].item():+.2f},{y[0,0].item():+.2f}) "
                f"box=({box_x[0,0].item():+.2f},{box_y[0,0].item():+.2f},{box_z[0,0].item():+.2f}) "
                f"done={int(self._approach_done)}"
                f"{mixed}{step_info} "
                f"out=({vx_cmd[0,0].item():.2f},{vy_cmd[0,0].item():+.2f},{yaw_cmd[0,0].item():+.2f})"
            )
            if self._is_push_step(cur):
                if self._push_mode == "lateral":
                    print(
                        f"  push[lateral] box_yaw_rel={self._last_box_rel_yaw:+.3f} "
                        f"right_err={self._last_push_right_err:+.3f} "
                        f"dx={self._last_align_x_delta:+.3f} "
                        f"fr-fl={self._last_push_skew:+.3f} "
                        f"right={self._last_lidar_right:.2f}"
                    )
                else:
                    print(
                        f"  push[forward] box_yaw_rel={self._last_box_rel_yaw:+.3f} "
                        f"dy={self._last_align_y_delta:+.3f} "
                        f"skew={self._last_push_skew:+.3f}"
                    )
        return cmd

    def _get_velocity_commands_held(self, proprio: torch.Tensor) -> torch.Tensor:
        """Update high-level cmd at high_level_hz; hold between updates."""
        b = proprio.shape[0]
        need_refresh = (
            bool(getattr(self, "_hl_cmd_force_refresh", False))
            or self._hl_cmd_cached is None
            or (not isinstance(self._hl_cmd_cached, torch.Tensor))
            or self._hl_cmd_cached.shape[0] != b
            or (self._hl_cmd_step_counter % self._hl_cmd_hold_steps == 0)
        )
        if need_refresh:
            self._hl_cmd_cached = self._get_velocity_commands(proprio)
            self._hl_cmd_force_refresh = False
        self._hl_cmd_step_counter += 1
        return self._hl_cmd_cached

    def get_video_overlay_lines(self) -> list[str]:
        lines = [
            f"phase={self.phase}  done={int(self._approach_done)}",
            f"pos=({self._last_pos_xy[0]:+.2f},{self._last_pos_xy[1]:+.2f})",
            f"box=({self._last_box_xy[0]:+.2f},{self._last_box_xy[1]:+.2f},{self._last_box_z:+.2f})",
        ]
        if self.ENABLE_TRAJECTORY_TERMINATION and self.TRAJECTORY_DEVIATION_TOL > 0.0:
            lines.append(
                f"traj_dev={self._last_trajectory_dev:.2f}m / tol={self.TRAJECTORY_DEVIATION_TOL:.2f}m"
            )
        cur = self._current_nav_step()
        if cur is not None:
            step_idx0 = int(self._nav_step_idx[0].item()) if self._nav_step_idx is not None else 0
            lines.append(
                f"nav [{step_idx0 + 1}/{len(self.nav_steps)}] {cur['name']} "
                f"{self._nav_step_progress:.2f}/{cur['dist']:.2f}m "
                f"cmd=({cur['vx']:.1f},{cur['vy']:+.1f})"
            )
        else:
            lines.append("nav: push_ready")
        if self._is_push_step(cur):
            lidar_flag = "on" if self.push_enable_lidar_correction else "off"
            if self._push_mode == "lateral":
                lines.append(
                    f"push[lateral] lidar={lidar_flag} rel_yaw={self._last_box_rel_yaw:+.3f} "
                    f"right_err={self._last_push_right_err:+.3f} "
                    f"dx={self._last_align_x_delta:+.3f} "
                    f"fr-fl={self._last_push_skew:+.3f} "
                    f"right={self._last_lidar_right:.2f}"
                )
            else:
                lines.append(
                    f"push[forward] lidar={lidar_flag} rel_yaw={self._last_box_rel_yaw:+.3f} "
                    f"dy={self._last_align_y_delta:+.3f} "
                    f"skew={self._last_push_skew:+.3f}"
                )
        if self.phase == "final":
            lines.append(
                f"final z: root_z={self._last_root_z:+.3f} fall_thresh={self.FALL_MIN_HEIGHT:+.3f}"
            )
        vx, vy, yaw = self._last_nav_cmd
        lines.append(f"out=({vx:.2f},{vy:+.2f},{yaw:+.2f})")
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

        velocity_commands = self._get_velocity_commands_held(proprio)
        # Cache per-env high-level command batch for dataset collection/debugging.
        self._last_high_level_cmd_batch = velocity_commands.detach().clone()

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

        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        giveup = False
        if self.ENABLE_TRAJECTORY_TERMINATION and self._trajectory_deviated_batch is not None:
            giveup = bool(self._trajectory_deviated_batch.any().item())
        return {
            "action": action_env.detach().cpu().numpy().tolist(),
            "action_tensor": action_env.detach(),
            "giveup": giveup,
        }
