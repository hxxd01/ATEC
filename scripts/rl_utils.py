import numpy as np
import torch
import gymnasium as gym


def to_uint8_hwc(frame) -> np.ndarray | None:
    """Convert render/obs image tensor or array to uint8 HWC RGB."""
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None

    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        max_v = float(arr.max())
        min_v = float(arr.min())
        if max_v <= 1.5 and min_v >= 0.0:
            arr = arr * 255.0
        elif max_v > min_v:
            arr = (arr - min_v) / (max_v - min_v) * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _rgb_panel_uint8(frame: np.ndarray | None) -> np.ndarray:
    """Return a contiguous HWC uint8 panel safe for OpenCV."""
    if frame is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    out = to_uint8_hwc(frame)
    if out is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    return np.ascontiguousarray(out)


def resize_rgb_panel(frame: np.ndarray | None, target_h: int) -> np.ndarray:
    """Resize an RGB panel to target height (keep aspect ratio)."""
    if frame is None:
        target_w = max(1, int(round(target_h * 4 / 3)))
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    frame = _rgb_panel_uint8(frame)
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, max(1, target_h), 3), dtype=np.uint8)
    target_w = max(1, int(round(w * target_h / h)))
    if h == target_h and w == target_w:
        return frame.copy()
    try:
        import cv2

        return np.ascontiguousarray(
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        )
    except ImportError:
        from PIL import Image

        return np.ascontiguousarray(
            np.asarray(Image.fromarray(frame).resize((target_w, target_h), Image.BILINEAR))
        )


def obs_image_rgb(obs: dict | None, key: str) -> np.ndarray | None:
    if not isinstance(obs, dict):
        return None
    image_obs = obs.get("image")
    if not isinstance(image_obs, dict) or key not in image_obs:
        return None
    return to_uint8_hwc(image_obs[key])


def stitch_global_head_ee_frame(
    global_frame: np.ndarray | None,
    obs: dict | None,
    *,
    panel_labels: tuple[str, str, str] = ("global", "head", "ee"),
) -> np.ndarray | None:
    """Horizontally stack viewport | head_rgb | ee_rgb from the same step."""
    global_rgb = to_uint8_hwc(global_frame)
    head_rgb = obs_image_rgb(obs, "head_rgb")
    ee_rgb = obs_image_rgb(obs, "ee_rgb")
    if global_rgb is None and head_rgb is None and ee_rgb is None:
        return None

    ref = global_rgb if global_rgb is not None else (head_rgb if head_rgb is not None else ee_rgb)
    target_h = max(
        ref.shape[0],
        head_rgb.shape[0] if head_rgb is not None else 0,
        ee_rgb.shape[0] if ee_rgb is not None else 0,
        240,
    )
    panels = [
        resize_rgb_panel(global_rgb, target_h),
        resize_rgb_panel(head_rgb, target_h),
        resize_rgb_panel(ee_rgb, target_h),
    ]
    try:
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        labeled: list[np.ndarray] = []
        for panel, label in zip(panels, panel_labels):
            canvas = np.ascontiguousarray(panel.copy())
            cv2.putText(
                canvas,
                label,
                (8, 24),
                font,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            labeled.append(canvas)
        panels = labeled
    except (ImportError, Exception):
        pass
    return np.ascontiguousarray(np.concatenate(panels, axis=1))


def draw_text_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw semi-transparent HUD text on the top-left of an RGB frame."""
    if frame is None or len(lines) == 0:
        return frame

    img = np.ascontiguousarray(frame)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    line_h = max(14, h // 36)
    pad = 6
    box_h = pad * 2 + line_h * len(lines)
    box_w = min(w - 2 * pad, max(480, int(w * 0.55)))

    try:
        import cv2

        out = img.copy()
        overlay = out.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.38, h / 900.0)
        thickness = 1
        y = pad + line_h
        for line in lines:
            cv2.putText(
                out,
                line,
                (pad + 4, y - 4),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y += line_h
        return out
    except ImportError:
        pass

    try:
        from PIL import Image, ImageDraw, ImageFont

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil, "RGBA")
        draw.rectangle((pad, pad, pad + box_w, pad + box_h), fill=(0, 0, 0, 140))
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", max(12, line_h - 2))
        except OSError:
            font = ImageFont.load_default()
        y = pad + 2
        for line in lines:
            draw.text((pad + 4, y), line, fill=(255, 255, 255, 255), font=font)
            y += line_h
        return np.asarray(pil)
    except ImportError:
        return img


class RenderOverlayWrapper(gym.Wrapper):
    """Inject HUD lines into env.render() frames (use inside RecordVideo)."""

    def __init__(self, env: gym.Env, *, multi_view: bool = False):
        super().__init__(env)
        self._overlay_lines: list[str] = []
        self._multi_view = bool(multi_view)
        self._last_obs: dict | None = None

    def set_overlay_lines(self, lines: list[str]) -> None:
        self._overlay_lines = list(lines)

    def set_latest_obs(self, obs: dict | None) -> None:
        self._last_obs = obs if isinstance(obs, dict) else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.set_latest_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.set_latest_obs(obs)
        return obs, reward, terminated, truncated, info

    def render(self):
        global_frame = self.env.render()
        if self._multi_view:
            frame = stitch_global_head_ee_frame(global_frame, self._last_obs)
            if frame is None:
                frame = to_uint8_hwc(global_frame)
        else:
            frame = global_frame
        if frame is None:
            return global_frame
        if self._overlay_lines:
            return draw_text_overlay(frame, self._overlay_lines)
        return frame


def _unwrap_play_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def ground_camera_follow(env, robot_name: str = "robot", env_index: int = 0, alpha: float = 0.15):
    """Low side-view camera that follows the robot (for ground_view.mp4)."""
    unwrapped = _unwrap_play_env(env)
    if not hasattr(unwrapped, "scene"):
        return
    try:
        ground_cam = unwrapped.scene["ground_camera"]
        robot = unwrapped.scene[robot_name]
    except KeyError:
        return

    from isaaclab.utils.math import quat_apply

    device = unwrapped.device
    robot_pos = robot.data.root_pos_w[env_index]
    robot_quat = robot.data.root_quat_w[env_index : env_index + 1]

    # Body-frame offsets: behind-left, near ground; look at torso/arm height.
    eye_local = torch.tensor([[-1.6, -1.3, 0.32]], device=device, dtype=torch.float32)
    target_local = torch.tensor([[0.9, 0.0, 0.38]], device=device, dtype=torch.float32)
    eye = quat_apply(robot_quat, eye_local).squeeze(0) + robot_pos
    lookat = quat_apply(robot_quat, target_local).squeeze(0) + robot_pos
    eye[2] = torch.clamp(eye[2], min=0.12)

    if not hasattr(ground_camera_follow, "_smooth_eye"):
        ground_camera_follow._smooth_eye = {}
        ground_camera_follow._smooth_look = {}
    if env_index not in ground_camera_follow._smooth_eye:
        ground_camera_follow._smooth_eye[env_index] = eye.clone()
        ground_camera_follow._smooth_look[env_index] = lookat.clone()

    smooth_eye = (1.0 - alpha) * ground_camera_follow._smooth_eye[env_index] + alpha * eye
    smooth_look = (1.0 - alpha) * ground_camera_follow._smooth_look[env_index] + alpha * lookat
    ground_camera_follow._smooth_eye[env_index] = smooth_eye
    ground_camera_follow._smooth_look[env_index] = smooth_look

    ground_cam.set_world_poses_from_view(
        smooth_eye.unsqueeze(0),
        smooth_look.unsqueeze(0),
        env_ids=[env_index],
    )


def camera_follow(env, robot_name: str = "robot", env_index: int = 0, alpha: float = 0.15):
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, "viewport_camera_controller"):
        return

    try:
        robot = unwrapped.scene[robot_name]
    except KeyError as e:
        raise KeyError(
            f"Robot asset '{robot_name}' not found in env.unwrapped.scene."
        ) from e

    device = unwrapped.device

    robot_pos = robot.data.root_pos_w[env_index]
    # Top-down spectator view for precise foothold inspection.
    top_offset = torch.tensor([-0.6, 0.0, 4.0], dtype=torch.float32, device=device)
    target_camera_pos = robot_pos + top_offset
    target_camera_pos[2] = torch.clamp(target_camera_pos[2], min=1.2)

    if not hasattr(camera_follow, "_smooth_pos"):
        camera_follow._smooth_pos = {}

    if env_index not in camera_follow._smooth_pos:
        camera_follow._smooth_pos[env_index] = target_camera_pos.clone()

    smooth_camera_pos = camera_follow._smooth_pos[env_index]
    smooth_camera_pos = (1.0 - alpha) * smooth_camera_pos + alpha * target_camera_pos
    camera_follow._smooth_pos[env_index] = smooth_camera_pos

    unwrapped.viewport_camera_controller.set_view_env_index(env_index=env_index)
    unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.detach().cpu().numpy(),
        lookat=robot_pos.detach().cpu().numpy(),
    )