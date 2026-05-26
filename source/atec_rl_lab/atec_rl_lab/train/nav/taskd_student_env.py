"""Task D student wrapper: image(+depth)+proprio observations, no privileged critic."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .taskd_teacher_env import (
    TaskDTeacherEnv,
    _LIN_VEL_SLICE,
    _ANG_VEL_SLICE,
    _GRAVITY_SLICE,
)

_PROPRIO_BASE_DIM = 9
_LAST_NAV_CMD_DIM = 3


class TaskDStudentEnv(TaskDTeacherEnv):
    def __init__(
        self,
        env: gym.Env,
        ll_policy_path: str,
        device: str = "cuda",
        inner_steps: int = 25,
        vx_min: float = -2.0,
        vx_max: float = 2.0,
        image_hw: int = 64,
        depth_max: float = 5.0,
        depth_only: bool = False,
        lidar_bins: int = 0,
        nav_log_interval: int = 10,
    ):
        self._student_lidar_bins = int(lidar_bins)
        super().__init__(
            env=env,
            ll_policy_path=ll_policy_path,
            device=device,
            inner_steps=inner_steps,
            lidar_bins=self._student_lidar_bins,
            vx_min=vx_min,
            vx_max=vx_max,
            nav_log_interval=nav_log_interval,
        )
        self._nav_log_tag = "TaskDStudent"
        self._image_hw = int(image_hw)
        self._depth_max = float(depth_max)
        self._depth_only = bool(depth_only)
        self._img_channels = 1 if self._depth_only else 4
        self._student_img_flat = 2 * self._img_channels * self._image_hw * self._image_hw
        self._proprio_dim = _PROPRIO_BASE_DIM + _LAST_NAV_CMD_DIM
        self._actor_dim = self._student_lidar_bins + self._student_img_flat + self._proprio_dim
        # privileged extras:
        # robot pose(3) + box pose(3) + rel body(3) + r_vel(2) + b_vel(2) + rel_world(2) + contact(1)
        # + stage onehot(num_stages) + stage_progress(1)
        self._critic_extra_dim = 17 + self._num_stages
        self._critic_dim = self._actor_dim + self._critic_extra_dim
        self._last_nav_cmd_buf: torch.Tensor | None = None
        self.observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._actor_dim,), dtype=np.float32),
                "critic": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._critic_dim,), dtype=np.float32),
            }
        )
        print(
            f"[TaskDStudent] actor_dim={self._actor_dim}, critic_dim={self._critic_dim} "
            f"img={self._img_channels}ch x2 lidar_bins={self._student_lidar_bins} "
            f"proprio={self._proprio_dim} (incl last_nav_cmd) depth_only={self._depth_only} "
            f"(includes {self._critic_extra_dim} privileged dims)",
            flush=True,
        )

    def _ensure_last_nav_cmd_buf(self, batch: int) -> None:
        if self._last_nav_cmd_buf is None or int(self._last_nav_cmd_buf.shape[0]) != int(batch):
            self._last_nav_cmd_buf = torch.zeros(
                batch, _LAST_NAV_CMD_DIM, device=self._device, dtype=torch.float32
            )

    def _reset_env_state(self, reset_mask: torch.Tensor, rx, ry, bx, by) -> None:
        super()._reset_env_state(reset_mask, rx, ry, bx, by)
        if self._last_nav_cmd_buf is not None and bool(reset_mask.any()):
            self._last_nav_cmd_buf[reset_mask] = 0.0

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self._ensure_last_nav_cmd_buf(self.num_envs)
        self._last_nav_cmd_buf.zero_()
        return self._obs_dict(self._current_obs), info

    def step(self, nav_action: torch.Tensor):
        if not isinstance(nav_action, torch.Tensor):
            nav_action = torch.as_tensor(nav_action, dtype=torch.float32)
        nav_action = nav_action.to(self._device, dtype=torch.float32)
        if nav_action.ndim == 1:
            nav_action = nav_action.unsqueeze(0)
        nav_action = nav_action.clamp(-1.0, 1.0)

        obs_dict, reward, terminated, truncated, info = super().step(nav_action)

        self._ensure_last_nav_cmd_buf(nav_action.shape[0])
        self._last_nav_cmd_buf.copy_(nav_action)
        done = terminated | truncated
        if bool(done.any()):
            self._last_nav_cmd_buf[done] = 0.0

        return self._obs_dict(self._current_obs), reward, terminated, truncated, info

    def _prep_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] != self._image_hw or x.shape[-2] != self._image_hw:
            x = F.interpolate(x, size=(self._image_hw, self._image_hw), mode="bilinear", align_corners=False)
        return x

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        src_is_int = not x.dtype.is_floating_point
        if x.dtype != torch.float32:
            x = x.float()
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        elif x.ndim == 4 and x.shape[-1] == 1:
            x = x[..., 0]
        x = torch.nan_to_num(x, nan=self._depth_max, posinf=self._depth_max, neginf=0.0)
        if src_is_int:
            x = torch.clamp(x / 255.0, 0.0, 1.0)
        elif x.max() > 1.5:
            x = torch.clamp(x, 0.05, self._depth_max)
            x = torch.log1p(x) / np.log1p(self._depth_max)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[-1] != self._image_hw or x.shape[-2] != self._image_hw:
            x = F.interpolate(x, size=(self._image_hw, self._image_hw), mode="bilinear", align_corners=False)
        return x

    def _policy_depth_gray(self, cam_tensor: torch.Tensor, env_idx: int) -> np.ndarray:
        """Single-camera policy-input depth as uint8 HxW."""
        if self._depth_only:
            depth = cam_tensor[env_idx, 0].detach().float().clamp(0.0, 1.0)
        else:
            depth = cam_tensor[env_idx, 3].detach().float().clamp(0.0, 1.0)
        return (depth.cpu().numpy() * 255.0).astype(np.uint8)

    @staticmethod
    def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
        return np.stack([gray, gray, gray], axis=-1)

    def get_depth_video_frames(self, env_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """RGB previews of policy-input depth for head and ee cameras."""
        batch = self.num_envs
        ei = max(0, min(int(env_idx), batch - 1))
        head = self._camera_tensor("head_camera", batch)
        ee = self._camera_tensor("ee_camera", batch)
        head_rgb = self._gray_to_rgb(self._policy_depth_gray(head, ei))
        ee_rgb = self._gray_to_rgb(self._policy_depth_gray(ee, ei))
        return head_rgb, ee_rgb

    def enable_combined_video(self, env_idx: int = 0, max_frames: int | None = None) -> None:
        """Record RGB|head_depth|ee_depth stitched frames at each physics step."""
        self._combined_video_enabled = True
        self._combined_video_env_idx = max(0, min(int(env_idx), self.num_envs - 1))
        self._combined_video_max_frames = int(max_frames) if max_frames is not None else None
        self._combined_video_frames: list[np.ndarray] = []

    def disable_combined_video(self) -> None:
        self._combined_video_enabled = False
        self._combined_video_frames = []

    @property
    def combined_video_frame_count(self) -> int:
        return len(getattr(self, "_combined_video_frames", []))

    @staticmethod
    def _resize_rgb_panel(img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == target_h:
            return img
        target_w = max(1, int(round(w * target_h / h)))
        try:
            import cv2

            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception:
            t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
            t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

    @staticmethod
    def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.5:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _get_sim_render_rgb(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            return np.zeros((self._image_hw, self._image_hw, 3), dtype=np.uint8)
        return self._to_uint8_rgb(frame)

    def stitch_rgb_head_ee_frame(self, env_idx: int | None = None) -> np.ndarray:
        """Horizontally stack sim RGB | head depth | ee depth (same physics-step snapshot)."""
        ei = self._combined_video_env_idx if env_idx is None else max(0, min(int(env_idx), self.num_envs - 1))
        rgb = self._get_sim_render_rgb()
        head_rgb, ee_rgb = self.get_depth_video_frames(ei)
        target_h = max(rgb.shape[0], head_rgb.shape[0], ee_rgb.shape[0])
        panels = [
            self._resize_rgb_panel(rgb, target_h),
            self._resize_rgb_panel(head_rgb, target_h),
            self._resize_rgb_panel(ee_rgb, target_h),
        ]
        return np.concatenate(panels, axis=1)

    def _on_after_physics_step(self) -> None:
        if not getattr(self, "_combined_video_enabled", False):
            return
        max_frames = getattr(self, "_combined_video_max_frames", None)
        if max_frames is not None and len(self._combined_video_frames) >= max_frames:
            return
        self._combined_video_frames.append(self.stitch_rgb_head_ee_frame())

    def _camera_tensor(self, cam_name: str, batch: int) -> torch.Tensor:
        try:
            cam = self.env.unwrapped.scene[cam_name]
            out = cam.data.output
            if self._depth_only:
                if "depth" not in out:
                    raise KeyError(f"{cam_name} missing depth output")
                return self._prep_depth(out["depth"].to(device=self._device))
            rgb = self._prep_rgb(out["rgb"].to(device=self._device))
            depth = self._prep_depth(out["depth"].to(device=self._device))
            return torch.cat([rgb, depth], dim=1)
        except Exception:
            return torch.zeros(
                batch,
                self._img_channels,
                self._image_hw,
                self._image_hw,
                device=self._device,
                dtype=torch.float32,
            )

    def _build_actor_obs(self, env_obs: dict):
        proprio = env_obs["proprio"].to(self._device, dtype=torch.float32)
        batch = proprio.shape[0]
        lin_vel = proprio[:, _LIN_VEL_SLICE]
        ang_vel = proprio[:, _ANG_VEL_SLICE]
        gravity = proprio[:, _GRAVITY_SLICE]
        self._ensure_last_nav_cmd_buf(batch)
        proprio_feat = torch.cat([lin_vel, ang_vel, gravity, self._last_nav_cmd_buf[:batch]], dim=-1)

        parts = []
        if self._student_lidar_bins > 0:
            parts.append(self._lidar_compact(env_obs.get("extero", None), batch))
        head = self._camera_tensor("head_camera", batch).reshape(batch, -1)
        ee = self._camera_tensor("ee_camera", batch).reshape(batch, -1)
        parts.extend([head, ee, proprio_feat])
        return torch.cat(parts, dim=-1)

    def _build_critic_obs(self, actor_obs: torch.Tensor):
        robot = self.env.unwrapped.scene["robot"]
        box = self.env.unwrapped.scene["box"]
        r_vel = robot.data.root_lin_vel_w.to(device=self._device, dtype=torch.float32)[:, :2]
        b_vel = box.data.root_lin_vel_w.to(device=self._device, dtype=torch.float32)[:, :2]
        rx, ry, robot_yaw = self._robot_pose()
        bx, by, box_yaw = self._box_pose()
        bx_body, by_body = self._relative_box_body(rx, ry, robot_yaw, bx, by)
        rel_yaw = torch.atan2(torch.sin(box_yaw - robot_yaw), torch.cos(box_yaw - robot_yaw))
        rel_world = torch.cat([bx - rx, by - ry], dim=-1)
        cf = self.env.unwrapped.scene["contact_sensor"].data.net_forces_w
        contact_on = (cf.norm(dim=-1).max(dim=1).values > 2.0).to(dtype=torch.float32).unsqueeze(-1)
        stage_oh = self._stage_onehot(actor_obs.shape[0])
        stage_prog = self._stage_progress_buf.unsqueeze(-1)

        priv = torch.cat(
            [
                torch.cat([rx, ry, robot_yaw], dim=-1),
                torch.cat([bx, by, box_yaw], dim=-1),
                torch.cat([bx_body, by_body, rel_yaw], dim=-1),
                r_vel,
                b_vel,
                rel_world,
                contact_on,
                stage_oh,
                stage_prog,
            ],
            dim=-1,
        )
        return torch.cat([actor_obs, priv], dim=-1)
