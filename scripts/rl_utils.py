import numpy as np
import torch
import gymnasium as gym


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

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._overlay_lines: list[str] = []

    def set_overlay_lines(self, lines: list[str]) -> None:
        self._overlay_lines = list(lines)

    def render(self):
        frame = self.env.render()
        if frame is None or not self._overlay_lines:
            return frame
        return draw_text_overlay(frame, self._overlay_lines)


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