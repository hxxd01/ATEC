"""
HierarchicalNavEnv: wraps Task-A env to train a second-stage navigation policy.

Architecture:
  High-level nav policy  →  vel_cmd [vx, vy, yaw]  (what we train here)
  Low-level loco policy  →  joint_pos [B, 20]       (frozen policy.pt)
  Task-A env             →  CrossXMulti reward

Exteroception modes:
  camera_mode=True        → 64×64 RGB from head_camera + CNN (img_flat=12288, obs=12297)
  lidar_bins=36           → compress spherical LiDAR into 36 bins (obs=45)
  extero_raw_dims=75      → raw height-scan (obs=84)  ← DEFAULT
  all zeros               → proprio only (obs=9, fastest)

Observation layout: [extero_flat | lin_vel(3) | ang_vel(3) | gravity(3)]
Action (3): nav output in [-1, 1] → vel_cmd [vx, vy, yaw] with vx∈[0,1] m/s

One nav step = inner_steps low-level env steps (default 50 → 1.0 s @ 0.02 s).
The same vel_cmd is held for the whole interval (no command jumps at 10 Hz).
Reward returned to PPO is the sum of env rewards over that interval (one scalar / s).

Isaac Lab auto-resets terminated envs inside each env.step(); the observation
returned after a nav step is always post-reset for envs that finished this step.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────────────────────────────────────
# Constants (B2Piper)
# ──────────────────────────────────────────────────────────────────────────────
# proprio layout: [lin_vel(3), ang_vel(3), vel_cmd(3), gravity(3),
#                  joint_pos(20), joint_vel(20), last_actions(20)]  → 72 dims
_LIN_VEL_SLICE    = slice(0,  3)
_ANG_VEL_SLICE    = slice(3,  6)
_GRAVITY_SLICE    = slice(9, 12)
_JPOS_SLICE       = slice(12, 32)   # all 20 joints
_JVEL_SLICE       = slice(32, 52)
_ACT_SLICE        = slice(52, 72)
_LEG_DIM          = 12              # first 12 joints are legs
_TOTAL_ACTION_DIM = 20

# Scale factors matching loco policy training (solution.py)
_TRAIN_TO_ENV = torch.tensor([0.25, 0.5, 0.5] * 4, dtype=torch.float32)   # [12]
_ENV_TO_TRAIN = torch.tensor([4.0,  2.0, 2.0] * 4, dtype=torch.float32)   # [12]

# Nav action in [-1, 1] → physical vel_cmd for policy.pt
_VX_CMD_MIN = 0.0
_VX_CMD_MAX = 1.0
_VEL_CMD_SCALE = torch.tensor([0.45, 0.35], dtype=torch.float32)   # vy, yaw_rate (symmetric)


class HierarchicalNavEnv(gym.Wrapper):
    """
    Gymnasium wrapper around Task-A env for training a nav policy.
    Compatible with isaaclab_rl's RslRlVecEnvWrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        ll_policy_path: str,
        device: str = "cuda",
        inner_steps: int = 50,
        lidar_bins: int = 0,
        extero_raw_dims: int = 75,
        camera_mode: bool = False,
        camera_hw: int = 64,
        camera_channels: int = 3,
        debug_timing: bool = False,
    ):
        super().__init__(env)
        self._device = device
        self.inner_steps = inner_steps
        self._lidar_bins = lidar_bins
        self._extero_raw_dims = extero_raw_dims
        self._camera_mode = camera_mode
        self._camera_hw = camera_hw
        self._camera_channels = camera_channels
        self._img_flat_dim = camera_channels * camera_hw * camera_hw if camera_mode else 0
        if camera_mode:
            self._nav_obs_dim = self._img_flat_dim + 9
        else:
            self._nav_obs_dim = max(lidar_bins, extero_raw_dims) + 9

        self.ll_policy = torch.jit.load(ll_policy_path, map_location=device)
        self.ll_policy.eval()

        self._t2e = _TRAIN_TO_ENV.to(device)
        self._e2t = _ENV_TO_TRAIN.to(device)
        self._vx_cmd_min = _VX_CMD_MIN
        self._vx_cmd_max = _VX_CMD_MAX
        self._vel_cmd_scale = _VEL_CMD_SCALE.to(device)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._nav_obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,), dtype=np.float32,
        )

        self._current_obs: dict | None = None
        self._nav_step_count = 0
        self._logged_first_rollout = False
        self._debug_timing = debug_timing

        base = self.env.unwrapped
        self._env_dt = float(getattr(base, "step_dt", 0.02))
        self._nav_dt = self.inner_steps * self._env_dt

        print(
            f"[INFO] HierarchicalNavEnv: nav_dt={self._nav_dt:.2f}s  "
            f"inner_steps={self.inner_steps}  env_dt={self._env_dt}s  "
            f"(nav policy {1.0/self._nav_dt:.2f} Hz, loco {1.0/self._env_dt:.0f} Hz)",
            flush=True,
        )
        print("[INFO] Warming up low-level policy (first CUDA/JIT forward)...", flush=True)
        t_warm = time.perf_counter()
        dummy_obs = torch.zeros(1, 45, device=device, dtype=torch.float32)
        with torch.inference_mode():
            _ = self.ll_policy(dummy_obs)
        print(f"[INFO] Low-level policy warmup done in {time.perf_counter() - t_warm:.1f}s", flush=True)

    # ─────────────────────────────── Properties ──────────────────────────────

    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def max_episode_length(self) -> int:
        """Nav-level max steps (Isaac Lab timeout is in base env steps)."""
        base_max = getattr(self.env.unwrapped, "max_episode_length", 60000)
        return max(1, int(base_max) // self.inner_steps)

    @property
    def device(self) -> str:
        return self._device

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Nav-step counter derived from the base env (do not maintain a separate buffer)."""
        base_buf = self.env.unwrapped.episode_length_buf
        return base_buf // self.inner_steps

    # ─────────────────────────────── Helpers ─────────────────────────────────

    def _compress_lidar(self, extero: torch.Tensor) -> torch.Tensor:
        B = extero.shape[0]
        n = extero.shape[-1]
        if n <= 0:
            return torch.zeros(B, self._lidar_bins, device=self._device, dtype=extero.dtype)

        if n % 360 == 0:
            ch, horiz = n // 360, 360
        elif n % 72 == 0:
            ch, horiz = n // 72, 72
        elif n % 60 == 0:
            ch, horiz = n // 60, 60
        else:
            lidar_1d = extero.abs()
            step = max(1, lidar_1d.shape[-1] // self._lidar_bins)
            chunks = lidar_1d.unfold(-1, step, step)[..., : self._lidar_bins]
            return chunks.max(dim=-1).values

        lidar = extero.view(B, ch, horiz).abs()
        lidar_1d = lidar.max(dim=1).values
        rays_per_bin = max(1, horiz // self._lidar_bins)
        usable = rays_per_bin * self._lidar_bins
        return lidar_1d[..., :usable].view(B, self._lidar_bins, rays_per_bin).max(dim=-1).values

    def _get_camera_image(self, B: int) -> torch.Tensor:
        try:
            cam = self.env.unwrapped.scene["head_camera"]
            rgb = cam.data.output["rgb"]
            rgb = rgb.to(self._device, dtype=torch.float32)
            rgb = rgb.permute(0, 3, 1, 2)
            if rgb.shape[-1] != self._camera_hw or rgb.shape[-2] != self._camera_hw:
                rgb = F.interpolate(
                    rgb, size=(self._camera_hw, self._camera_hw),
                    mode="bilinear", align_corners=False,
                )
            return rgb.reshape(B, -1)
        except Exception:
            return torch.zeros(B, self._img_flat_dim, device=self._device, dtype=torch.float32)

    def _build_nav_obs(self, env_obs: dict) -> torch.Tensor:
        proprio = env_obs["proprio"].to(self._device, dtype=torch.float32)
        B = proprio.shape[0]

        lin_vel = proprio[:, _LIN_VEL_SLICE]
        ang_vel = proprio[:, _ANG_VEL_SLICE]
        gravity = proprio[:, _GRAVITY_SLICE]
        proprio_feat = torch.cat([lin_vel, ang_vel, gravity], dim=-1)

        if self._camera_mode:
            img_flat = self._get_camera_image(B)
            return torch.cat([img_flat, proprio_feat], dim=-1)

        extero = env_obs.get("extero", None)

        if self._lidar_bins > 0:
            if extero is not None and extero.shape[-1] > 0:
                extero_feat = self._compress_lidar(extero.to(self._device, dtype=torch.float32))
            else:
                extero_feat = torch.zeros(B, self._lidar_bins, device=self._device, dtype=torch.float32)
            return torch.cat([extero_feat, proprio_feat], dim=-1)

        if self._extero_raw_dims > 0:
            if extero is not None and extero.shape[-1] > 0:
                raw = extero.to(self._device, dtype=torch.float32)
                n = raw.shape[-1]
                if n >= self._extero_raw_dims:
                    extero_feat = raw[..., : self._extero_raw_dims]
                else:
                    pad = torch.zeros(B, self._extero_raw_dims - n, device=self._device, dtype=torch.float32)
                    extero_feat = torch.cat([raw, pad], dim=-1)
            else:
                extero_feat = torch.zeros(B, self._extero_raw_dims, device=self._device, dtype=torch.float32)
            return torch.cat([extero_feat, proprio_feat], dim=-1)

        return proprio_feat

    def nav_action_to_vel_cmd(self, nav_cmd: torch.Tensor) -> torch.Tensor:
        """Map nav policy output [-1,1]³ to low-level ``vel_cmd`` (vx, vy, yaw_rate).

        vx: action -1→0 m/s, +1→1 m/s (forward only, no reverse).
        vy / yaw: symmetric scale ±0.45 / ±0.35 rad/s.
        """
        nav_cmd = nav_cmd.clamp(-1.0, 1.0)
        vx = (nav_cmd[:, 0] + 1.0) * 0.5 * (self._vx_cmd_max - self._vx_cmd_min) + self._vx_cmd_min
        vy = nav_cmd[:, 1] * self._vel_cmd_scale[0]
        wz = nav_cmd[:, 2] * self._vel_cmd_scale[1]
        return torch.stack(
            [
                vx.clamp(self._vx_cmd_min, self._vx_cmd_max),
                vy.clamp(-self._vel_cmd_scale[0], self._vel_cmd_scale[0]),
                wz.clamp(-self._vel_cmd_scale[1], self._vel_cmd_scale[1]),
            ],
            dim=-1,
        )

    def _build_ll_obs(self, env_obs: dict, nav_cmd: torch.Tensor) -> torch.Tensor:
        proprio = env_obs["proprio"].to(self._device, dtype=torch.float32)

        ang_vel     = proprio[:, _ANG_VEL_SLICE]
        gravity     = proprio[:, _GRAVITY_SLICE]
        jpos_leg    = proprio[:, _JPOS_SLICE][:, :_LEG_DIM]
        jvel_leg    = proprio[:, _JVEL_SLICE][:, :_LEG_DIM]
        act_leg_env = proprio[:, _ACT_SLICE][:, :_LEG_DIM]
        act_leg_tr  = act_leg_env * self._e2t

        vel_cmd = self.nav_action_to_vel_cmd(nav_cmd)

        return torch.cat([
            ang_vel  * 0.25,
            gravity,
            vel_cmd,
            jpos_leg,
            jvel_leg * 0.05,
            act_leg_tr,
        ], dim=-1)

    def _build_env_action(self, ll_action_train: torch.Tensor) -> torch.Tensor:
        B = ll_action_train.shape[0]
        env_action = torch.zeros(B, _TOTAL_ACTION_DIM, device=self._device, dtype=torch.float32)
        env_action[:, :_LEG_DIM] = ll_action_train * self._t2e
        return env_action

    def _to_bool_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.squeeze(-1).to(self._device).bool()
        return torch.as_tensor(x, device=self._device).bool()

    def _merge_episode_log(self, episode_log: dict, info: dict) -> dict:
        """Keep the latest Isaac Lab episode stats from a terminating inner step."""
        step_log = info.get("log", {}) if isinstance(info, dict) else {}
        if step_log:
            return step_log
        return episode_log

    # ──────────────────── rsl_rl compatibility interface ─────────────────────

    def get_observations(self):
        nav_obs = self._build_nav_obs(self._current_obs)
        return {"policy": nav_obs}, {}

    # ─────────────────────────────── Core API ────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        nav_obs = self._build_nav_obs(obs)
        return {"policy": nav_obs}, info

    def step(self, nav_action: torch.Tensor):
        """
        Hold nav_action fixed for inner_steps low-level steps (~nav_dt seconds).

        Returns one reward per nav decision: sum of env rewards over the interval
        (total progress in that second). Isaac Lab auto-resets on done; we OR
        termination flags across inner steps for rsl_rl.
        """
        if not isinstance(nav_action, torch.Tensor):
            nav_action = torch.as_tensor(nav_action, dtype=torch.float32)
        nav_action = nav_action.to(self._device, dtype=torch.float32)

        self._nav_step_count += 1
        if not self._logged_first_rollout:
            print(
                "[INFO] HierarchicalNavEnv: rollout started (first nav step).",
                flush=True,
            )
            self._logged_first_rollout = True

        B = self.num_envs
        total_reward = torch.zeros(B, device=self._device, dtype=torch.float32)
        terminated = torch.zeros(B, device=self._device, dtype=torch.bool)
        truncated = torch.zeros(B, device=self._device, dtype=torch.bool)
        last_info: dict = {}
        episode_log: dict = {}

        for k in range(self.inner_steps):
            ll_obs = self._build_ll_obs(self._current_obs, nav_action)
            with torch.inference_mode():
                ll_act_tr = self.ll_policy(ll_obs)
            env_action = self._build_env_action(ll_act_tr)

            obs, rew, term, trunc, info = self.env.step(env_action)
            self._current_obs = obs

            if isinstance(rew, torch.Tensor):
                total_reward += rew.squeeze(-1).to(self._device)
            else:
                total_reward += torch.as_tensor(rew, device=self._device).squeeze(-1)

            step_term = self._to_bool_tensor(term)
            step_trunc = self._to_bool_tensor(trunc)

            if (step_term | step_trunc).any():
                episode_log = self._merge_episode_log(episode_log, info)

            terminated |= step_term
            truncated |= step_trunc
            last_info = info

        done = terminated | truncated

        if self._nav_step_count <= 20 or self._nav_step_count % 50 == 0:
            try:
                robot_x = self.env.unwrapped.scene["robot"].data.root_pos_w[:, 0]
                x_str = f"  x=[{robot_x.min().item():.1f},{robot_x.max().item():.1f}]"
            except Exception:
                x_str = ""
            ep_len_nav = int(self.episode_length_buf.float().mean().item())
            print(
                f"[INFO] nav#{self._nav_step_count:4d}  "
                f"rew_mean={total_reward.mean().item():+.4f}  "
                f"dones={int(done.sum())}/{B}  "
                f"(term={int(terminated.sum())} trunc={int(truncated.sum())})  "
                f"ep_len_nav~{ep_len_nav}{x_str}",
                flush=True,
            )

        nav_obs_dict = {"policy": self._build_nav_obs(self._current_obs)}

        if episode_log and isinstance(last_info, dict):
            merged = dict(last_info)
            merged["log"] = episode_log
            last_info = merged

        return nav_obs_dict, total_reward, terminated, truncated, last_info
