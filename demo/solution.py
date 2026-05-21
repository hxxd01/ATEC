import os
import torch

class AlgSolution:

    #思路：y_est积分：维持4s的滑动窗口泄露，通过本体y计算（这里默认角度纠偏正确，如果用当前朝向的y分速度很可能出现偏差）。
    #yaw_est用长时间积分计算，可能会因为打滑而出现偏差。因此在粗糙地面的恢复过程中冻结y_est和yaw_est。
    #在金字塔坡度的时候会有重力导致y的变化，给重力投影的反向速度来抵消。
    '''地形中心 x	x 范围	地形类型	对应分数约
    -140	-150 ~ -130	平地	0
    -120	-130 ~ -110	平地	0.85 ~ 2.25
    -100	-110 ~ -90	随机粗糙地形	2.25 ~ 3.25
    -80	-90 ~ -70	随机粗糙地形	3.25 ~ 4.25
    -60	-70 ~ -50	随机粗糙地形	4.25 ~ 5.25
    -40	-50 ~ -30	随机粗糙地形	5.25 ~ 6.50
    -20	-30 ~ -10	金字塔坡	6.50 ~ 8.50
    0	-10 ~ 10	倒金字塔坡	8.50 ~ 10.50
    20	10 ~ 30	金字塔坡	10.50 ~ 12.50
    40	30 ~ 50	倒金字塔坡	12.50 ~ 14.50
    60	50 ~ 70	金字塔楼梯	14.50 ~ 16.50
    80	70 ~ 90	倒金字塔楼梯	16.50 ~ 18.50
    100	90 ~ 110	金字塔楼梯	18.50 ~ 20.50
    120	110 ~ 130	倒金字塔楼梯	20.50 ~ 23.33
    140	130 ~ 150	平地/终点段	23.33 ~ 26'''
    _TASK_A_STRIP_X0 = -150.0
    _TASK_A_STRIP_DX = 20.0
    _TASK_A_STRIP_TERRAINS = (
        "flat",
        "flat",
        "random_rough",
        "random_rough",
        "random_rough",
        "random_rough",
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
        "hf_pyramid_slope",
        "hf_pyramid_slope_inv",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "pyramid_stairs",
        "pyramid_stairs_inv",
        "flat",
    )

    ACTION_SCALE = 0.5
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )

    def __init__(self):
        policy_path = os.path.dirname(os.path.abspath(__file__)) + '/policy.pt'
        self.device = 'cuda'

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

        # Task A navigation: approximate odometry + closed-loop cmds (vx, vy, yaw_rate)
        self.dt = 0.02
        self.nav_vx_max = 1.0  # cap forward cmd (matches loco-friendly training range)
        self.nav_vx_target = 0.5
        self.nav_use_task_a_strip_vx = True
        # World-x seed for integrating position (matches ~Task A B2Piper spawn): (-141, ...)
        self.nav_init_world_x = -141.0
        self.nav_goal_x = 145.0
        self.nav_goal_slow_x = 125.0  # start easing vx before the finish strip
        self.nav_k_lat = 1.0 #y_est反馈
        self.nav_vy_lim = 0.45 #横向速度命令限幅
        self.nav_k_yaw = 0.8 #角度反馈
        self.nav_k_wz = 0.25 #角速度阻尼
        self.nav_wz_lim = 0.35 #yaw_cmd限幅度
        # Yaw sign test: set e.g. 0.3 to force constant yaw_cmd; None = normal nav.
        self.debug_fixed_yaw = None
        self.nav_cmd_ema_alpha = 0.35  # smooth vel_cmd at 50 Hz (0=hold, 1=no filter)
        self.yaw_est = None
        self.y_est = None
        self.x_est = None
        self._cmd_ema = None
        self._robot = None  # optional: set via bind_robot() for true world-x strip lookup
        self._debug_step = 0

        # Stuck → short recovery burst (cmds override normal nav loop)
        self.recovery_stuck_vx_thresh = 0.03
        self.recovery_stuck_steps = 50  # ~1 s @ dt=0.02
        self.recovery_duration_steps = 50
        self.recovery_vx_cmd = 0.15  # small forward nudge (backward drifts off the strip)
        self.recovery_yaw_mag = 0.30
        self._slow_vx_accum = None
        self._recovery_left = None
        self._recovery_next_yaw = None  # ±1, toggles on each new trigger
        self._active_recovery_yaw = None

        # Target vx per terrain category (tune freely)
        self.nav_vx_by_terrain_kind = dict(
            flat=2.0,
            random_rough=1.2,
            hf_pyramid_slope=1.0,
            hf_pyramid_slope_inv=1.0,
            pyramid_stairs=0.7,
            pyramid_stairs_inv=0.7,
        )
        vx_per_strip = [
            float(self.nav_vx_by_terrain_kind[k]) for k in self._TASK_A_STRIP_TERRAINS
        ]
        strip_starts = [
            self._TASK_A_STRIP_X0 + i * self._TASK_A_STRIP_DX for i in range(len(vx_per_strip))
        ]
        self._task_a_strip_starts_t = torch.tensor(
            strip_starts,
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        self._task_a_strip_vx_t = torch.tensor(
            vx_per_strip,
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        # 1.0 on pyramid_stairs_inv strips (x≈80, 120): flip gravity + LiDAR lateral sign.
        self._task_a_strip_is_stairs_inv_t = torch.tensor(
            [1.0 if k == "pyramid_stairs_inv" else 0.0 for k in self._TASK_A_STRIP_TERRAINS],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

    def reset(self, **kwargs):
        """Clear odometry when starting a new episode (optional; play script may not call)."""
        self.yaw_est = None
        self.y_est = None
        self.x_est = None
        self._slow_vx_accum = None
        self._recovery_left = None
        self._recovery_next_yaw = None
        self._active_recovery_yaw = None
        self._debug_step = 0

    def _resolve_joint_ids(self, candidates: tuple[list[str], ...]) -> list[int]:
        last_error = None
        for names in candidates:
            try:
                ids, found_names = self.robot.find_joints(names)
            except ValueError as err:
                last_error = err
                continue
            if len(ids) == len(names):
                if candidates is self.ARM_JOINT_NAME_CANDIDATES:
                    self.arm_joint_names = list(found_names)
                return list(ids)
        raise ValueError(
            f"Cannot resolve required joints from candidates: {candidates}. Last error: {last_error}"
        )

    def _resolve_ee_body_name(self) -> str:
        last_error = None
        for name in self.EE_BODY_NAME_CANDIDATES:
            try:
                body_ids, _ = self.robot.find_bodies(name)
            except ValueError as err:
                last_error = err
                continue
            if len(body_ids) == 1:
                return name
        raise ValueError(
            f"Cannot resolve EE body from candidates: {self.EE_BODY_NAME_CANDIDATES}. Last error: {last_error}"
        )

    def _ensure_cartesian_targets(self):
        self.cartesian_ctrl.reset()

    def _compute_arm_overlay_action(self) -> torch.Tensor:
        self._ensure_cartesian_targets()

        arm_jpos_des = self.cartesian_ctrl.compute_base(
            self.ee_pos_target_b,
            self.ee_quat_target_b,
        )

        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = self.gripper_open_pos.repeat(full_target.shape[0], 1)

        return (full_target - self.default_joint_pos) / self.ACTION_SCALE

    def _task_a_strip_indices(self, world_x: torch.Tensor) -> torch.Tensor:
        """Strip index [0 .. n-1] from world-frame x coordinate (broadcast on batch dim)."""
        # count = # {start <= x}; idx = count - 1
        sx = world_x.unsqueeze(-1)
        count = (sx >= self._task_a_strip_starts_t.to(device=world_x.device, dtype=world_x.dtype)).sum(dim=-1)
        n = self._task_a_strip_vx_t.shape[-1]
        idx = count - 1
        idx = idx.clamp(0, n - 1)
        return idx

    def _vx_cmd_from_strip(self, world_x: torch.Tensor, dtype) -> torch.Tensor:
        """Pick target vx for each batch row from segmented table or global fallback."""
        b = world_x.shape[0]
        if not self.nav_use_task_a_strip_vx:
            return torch.full((b, 1), float(self.nav_vx_target), device=world_x.device, dtype=dtype)
        vx_row = self._task_a_strip_vx_t.to(device=world_x.device, dtype=dtype)
        idx = self._task_a_strip_indices(world_x).long().reshape(-1)
        vx_tab = vx_row.squeeze(0)
        vx = vx_tab[idx].view(world_x.shape[0], 1)
        return vx

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Body odometry + lateral / heading loop; feed policy a consistent command vector."""
        # proprio: [base_lin_vel(3), base_ang_vel(3), vel_cmd(3), gravity(3), ...]
        device = proprio.device
        dtype = proprio.dtype
        b = proprio.shape[0]

        base_lin_vel = proprio[:, 0:3]
        base_ang_vel = proprio[:, 3:6]
        # proprio[:, 6:9] is vel_cmd (skipped); gravity is at [9:12]
        projected_gravity = proprio[:, 9:12]
        vx_body = base_lin_vel[:, 0:1]
        vy_body = base_lin_vel[:, 1:2]
        wz = base_ang_vel[:, 2:3]
        # gravity_y: body-frame lateral component of gravity.
        # Negative → right side lower (slope pushes robot rightward).
        # Feed-forward: push back opposite to gravity lean.
        gravity_y = projected_gravity[:, 1:2]
        gravity_x = projected_gravity[:, 0:1]   # pitch: negative when climbing

        if self.yaw_est is None or self.yaw_est.shape[0] != b:
            self.yaw_est = torch.zeros((b, 1), device=device, dtype=dtype)
            self.y_est = torch.zeros((b, 1), device=device, dtype=dtype)
            self.x_est = torch.full(
                (b, 1), float(self.nav_init_world_x), device=device, dtype=dtype
            )
            zl = torch.zeros((b, 1), device=device, dtype=torch.long)
            self._slow_vx_accum = zl.clone()
            self._recovery_left = zl.clone()
            self._recovery_next_yaw = torch.ones((b, 1), device=device, dtype=dtype)
            self._active_recovery_yaw = torch.ones((b, 1), device=device, dtype=dtype)
        else:
            self.yaw_est = self.yaw_est.to(device=device, dtype=dtype)
            self.y_est = self.y_est.to(device=device, dtype=dtype)
            self.x_est = self.x_est.to(device=device, dtype=dtype)
            self._slow_vx_accum = self._slow_vx_accum.to(device=device)
            self._recovery_left = self._recovery_left.to(device=device)
            self._recovery_next_yaw = self._recovery_next_yaw.to(device=device, dtype=dtype)
            self._active_recovery_yaw = self._active_recovery_yaw.to(device=device, dtype=dtype)

        in_recovery_before = self._recovery_left > 0

        # Freeze yaw/y during recovery: wz from turning maneuver would pollute heading estimate.
        self.yaw_est = torch.where(
            in_recovery_before,
            self.yaw_est,
            self.yaw_est + wz * self.dt,
        )
        self.yaw_est = torch.atan2(torch.sin(self.yaw_est), torch.cos(self.yaw_est))

        cos_y, sin_y = torch.cos(self.yaw_est), torch.sin(self.yaw_est)
        vx_world = cos_y * vx_body - sin_y * vy_body
        self.x_est = self.x_est + vx_world * self.dt

        # Use vy_body directly (not world_vy) to avoid yaw drift contaminating y_est.
        # Leaky integrator (τ ≈ 4 s) + clamp to prevent wind-up.
        self.y_est = torch.where(
            in_recovery_before,
            self.y_est,
            0.995 * self.y_est + vy_body * self.dt,
        )
        self.y_est = self.y_est.clamp(-0.6, 0.6)
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

        vx_cmd = self._vx_cmd_from_strip(self.x_est, dtype).to(device=device)
        strip_idx = self._task_a_strip_indices(self.x_est).long().reshape(-1)
        stairs_inv_mask = (
            self._task_a_strip_is_stairs_inv_t.to(device=device, dtype=dtype)
            .squeeze(0)[strip_idx]
            .view(b, 1)
        )
      
        grav_sign = 1.0 - 2.0 * stairs_inv_mask

        # Three-term lateral correction:
        #   1. y_est position feedback (unreliable on slopes, but still useful on flat)
        #   2. vy_body velocity damping (direct, no integration error)
        #   3. gravity_y feed-forward (sign flipped on pyramid_stairs_inv)
        vy_cmd = (
            -self.nav_k_lat * self.y_est
            - 0.4 * vy_body
            - grav_sign * gravity_y *1.2
        ).clamp(-self.nav_vy_lim, self.nav_vy_lim)
        # +yaw_cmd = turn left. Normal: left lean → subtract yaw_grav → turn right.
        # pyramid_stairs_inv: grav_sign=−1 → add yaw_grav → turn left toward centre.
        yaw_grav = (0.5 * gravity_y).clamp(-0.2, 0.2)
        yaw_grav = torch.where(gravity_y.abs() > 0.06, yaw_grav, torch.zeros_like(yaw_grav))
        yaw_cmd = (
            -self.nav_k_yaw * self.yaw_est - self.nav_k_wz * wz - grav_sign * yaw_grav
        ).clamp(-self.nav_wz_lim, self.nav_wz_lim)

        vx_rec = torch.full(
            (b, 1),
            float(self.recovery_vx_cmd),
            device=device,
            dtype=dtype,
        )
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

        # Pitch-based speed reduction: mild, only for extreme tilt (>25°).
        pitch_reduction = ((gravity_x.abs() - 0.40) * 0.8).clamp(0.0, 0.3)
        vx_cmd = vx_cmd * (1.0 - pitch_reduction)

        if self.debug_fixed_yaw is not None:
            yaw_cmd = torch.full(
                (b, 1), float(self.debug_fixed_yaw), device=device, dtype=dtype
            )
            if self._debug_step == 0:
                print(
                    f"[yaw sign test] forcing yaw_cmd={self.debug_fixed_yaw:+.3f} rad/s "
                    f"(from above: + = CCW / typically turn LEFT if z-up)"
                )

        cmd = torch.cat([vx_cmd, vy_cmd, yaw_cmd], dim=-1)
        self._debug_step += 1
        if self._debug_step % 50 == 1:
            print(
                f"[nav dbg step={self._debug_step:5d}] "
                f"x={self.x_est[0,0].item():7.2f}  "
                f"y_est={self.y_est[0,0].item():+6.3f}  "
                f"yaw={self.yaw_est[0,0].item():+5.3f}  "
                f"vx_body={vx_body[0,0].item():+5.3f}  "
                f"vy_body={vy_body[0,0].item():+5.3f}  "
                f"grav_y={gravity_y[0,0].item():+5.3f}  "
                f"grav_x={gravity_x[0,0].item():+5.3f}  "
                f"wz={wz[0,0].item():+5.3f}  "
                f"| cmd=({vx_cmd[0,0].item():.2f}, {vy_cmd[0,0].item():+.3f}, {yaw_cmd[0,0].item():+.3f})"
                f"  inv={stairs_inv_mask[0,0].item():.0f}"
            )

        return cmd

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        expected_dim = 3 + 3 + 3 + 3 + action_dim + action_dim + action_dim

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        _velocity_commands_env = proprio[:, idx:idx + 3]
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

        policy_obs = torch.cat(
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

        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Map training-time 12D leg action to current env 20D full-body action."""
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale

        action_env = torch.zeros(
            (num_envs, action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)

        return action_env

    def predicts(self, obs, current_score):
        """Run policy inference and return current-env full-body action."""
        #if current_score > 1:
            #return {'action': [], 'giveup': True}
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(
                action_train, device=self.device, dtype=torch.float32
            )

        action_train = action_train.to(device=self.device, dtype=torch.float32)

        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        action_env = action_env.cpu().numpy().tolist()
        return {'action': action_env, 'giveup': False}
