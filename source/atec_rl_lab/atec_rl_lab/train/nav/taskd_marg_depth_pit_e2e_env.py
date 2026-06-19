"""Task D pit e2e env: MARG proprio/history/priv + depth cameras (no height map)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import atec_rl_lab.tasks.task_d.locomotion.mdp as task_d_loco_mdp
from atec_rl_lab.train.locomotion.marg.constants import (
    MARG_CRITIC_PRIV_DIM,
    MARG_HEIGHT_MAP_DIM,
    MARG_HISTORY_DIM,
    MARG_PROPRIO_DIM,
)
from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import (
    LEG_ACTION_DIM,
    configure_pit_e2e_cameras,
    configure_pit_e2e_dagger_env_cfg,
    configure_pit_e2e_env_cfg,
)

_demo_dir = Path(__file__).resolve().parents[5] / "demo"
if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))
from depth_preprocess import prep_depth as _prep_depth_shared  # noqa: E402

_MARG_LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


class TaskDMargDepthPitE2EEnv(gym.Wrapper):
    """MARG obs groups + depth; pit locomotion rewards/terminations; 12-D leg actions."""

    inner_steps = 1
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env: gym.Env,
        *,
        device: str = "cuda",
        image_h: int = 24,
        image_w: int = 32,
        depth_max: float = 5.0,
        depth_only: bool = True,
        depth_render_h: int | None = None,
        depth_render_w: int | None = None,
        include_teacher_obs: bool = False,
        marg_from_obs_manager: bool = False,
        head_depth_only: bool | None = None,
    ):
        super().__init__(env)
        self._device = device
        self._include_teacher_obs = bool(include_teacher_obs)
        self._marg_from_obs_manager = bool(marg_from_obs_manager)
        if head_depth_only is None:
            head_depth_only = bool(getattr(self.unwrapped.cfg, "head_depth_only", True))
        self._head_depth_only = bool(head_depth_only)
        self._image_h = int(image_h)
        self._image_w = int(image_w)
        self._depth_render_h = int(depth_render_h) if depth_render_h is not None else self._image_h
        self._depth_render_w = int(depth_render_w) if depth_render_w is not None else self._image_w
        self._depth_max = float(depth_max)
        self._depth_only = bool(depth_only)
        self._img_channels = 1 if self._depth_only else 4
        self._depth_flat_per_cam = self._img_channels * self._image_h * self._image_w
        num_cams = 1 if self._head_depth_only else 2
        self._depth_dim = num_cams * self._depth_flat_per_cam

        self.num_envs = int(self.unwrapped.num_envs)
        self.device = self._device
        self.max_episode_length = int(self.unwrapped.max_episode_length)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(LEG_ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(MARG_PROPRIO_DIM,), dtype=np.float32
                ),
                "proprio_history": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(MARG_HISTORY_DIM,), dtype=np.float32
                ),
                "depth": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._depth_dim,), dtype=np.float32
                ),
                "critic_priv": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(MARG_CRITIC_PRIV_DIM,), dtype=np.float32
                ),
            }
        )
        if self._include_teacher_obs:
            self.observation_space.spaces["height_map"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(MARG_HEIGHT_MAP_DIM,), dtype=np.float32
            )
        self._cached_obs: dict[str, torch.Tensor] | None = None
        teacher_msg = f"+height_map({MARG_HEIGHT_MAP_DIM})" if self._include_teacher_obs else ""
        marg_src = "obs_manager" if self._marg_from_obs_manager else "wrapper"
        cam_tag = "head" if self._head_depth_only else "head+ee"
        print(
            f"[TaskDMargDepthPitE2E] obs=proprio({MARG_PROPRIO_DIM})+history({MARG_HISTORY_DIM})+"
            f"depth({self._depth_dim})+critic_priv({MARG_CRITIC_PRIV_DIM}){teacher_msg}, "
            f"marg_src={marg_src}, depth={self._img_channels}ch {cam_tag} @ {self._image_h}x{self._image_w}, "
            f"actions={LEG_ACTION_DIM}",
            flush=True,
        )

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf.copy_(value)

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        return _prep_depth_shared(
            x,
            image_h=self._image_h,
            image_w=self._image_w,
            depth_max=self._depth_max,
        )

    def _clear_marg_history(self, env_ids: torch.Tensor | None = None) -> None:
        """Only used when building MARG obs manually (non-Marg env cfg)."""
        if self._marg_from_obs_manager:
            return
        u = self.unwrapped
        hist = getattr(u, "_marg_proprio_history", None)
        if hist is None:
            return
        if env_ids is None:
            hist.zero_()
        elif env_ids.numel() > 0:
            hist[env_ids] = 0.0

    def _marg_obs_from_manager(self) -> dict[str, torch.Tensor]:
        """Read MARG groups from Isaac Lab obs_buf (computed during env.step/reset)."""
        buf = self.unwrapped.obs_buf
        obs = {
            "proprio": buf["proprio"].to(device=self._device),
            "proprio_history": buf["proprio_history"].to(device=self._device),
            "critic_priv": buf["critic_priv"].to(device=self._device),
        }
        if self._include_teacher_obs:
            obs["height_map"] = buf["height_map"].to(device=self._device)
        return obs

    def _marg_obs_manual(self, batch: int) -> dict[str, torch.Tensor]:
        """Build MARG obs by hand (legacy path when env has no Marg obs groups)."""
        u = self.unwrapped
        joint_names = _MARG_LEG_JOINTS
        obs = {
            "proprio": task_d_loco_mdp.marg_proprio(u, joint_names=joint_names),
            "proprio_history": task_d_loco_mdp.marg_proprio_history(u, joint_names=joint_names),
            "critic_priv": task_d_loco_mdp.marg_critic_privileged(u, joint_names=joint_names),
        }
        if self._include_teacher_obs:
            obs["height_map"] = task_d_loco_mdp.marg_height_map(u)
        return obs

    def _camera_depth_flat(self, cam_name: str, batch: int) -> torch.Tensor:
        try:
            cam = self.unwrapped.scene[cam_name]
            out = cam.data.output
            if "depth" not in out:
                raise KeyError(f"{cam_name} missing depth")
            return self._prep_depth(out["depth"].to(device=self._device)).reshape(batch, -1)
        except Exception:
            return torch.zeros(
                batch,
                self._depth_flat_per_cam,
                device=self._device,
                dtype=torch.float32,
            )

    def _build_obs(self) -> dict[str, torch.Tensor]:
        batch = self.num_envs
        if self._marg_from_obs_manager:
            obs = self._marg_obs_from_manager()
        else:
            obs = self._marg_obs_manual(batch)
        head = self._camera_depth_flat("head_camera", batch)
        if self._head_depth_only:
            obs["depth"] = head
        else:
            ee = self._camera_depth_flat("ee_camera", batch)
            obs["depth"] = torch.cat([head, ee], dim=-1)
        return obs

    def get_observations(self) -> tuple[dict[str, torch.Tensor], dict]:
        # Return cached obs: rsl_rl calls this before rollout without a physics step.
        if self._cached_obs is None:
            self._cached_obs = self._build_obs()
        return self._cached_obs, {}

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        self._clear_marg_history()
        self._cached_obs = self._build_obs()
        return self._cached_obs, info

    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = actions.to(self._device, dtype=torch.float32)
        _, reward, terminated, truncated, info = self.env.step(actions)
        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            self._clear_marg_history(done_ids)
        self._cached_obs = self._build_obs()
        return self._cached_obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()


__all__ = [
    "TaskDMargDepthPitE2EEnv",
    "configure_pit_e2e_cameras",
    "configure_pit_e2e_env_cfg",
    "configure_pit_e2e_dagger_env_cfg",
    "LEG_ACTION_DIM",
]
