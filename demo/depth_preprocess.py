"""Shared depth preprocessing: train (taskd_student_env) and deploy (solution.py).

Pipeline (must match taskd_student_env._prep_depth):
  1. -> Bx1xHxW, nan_to_num
  2. optional bilinear to depth_render_h x depth_render_w (platform 480x640 -> train sim 24x24)
  3. uint8 / log1p normalize (meters if max > 1.5)
  4. bilinear to image_hw x image_hw
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4 and x.shape[1] == 1:
        return x.float()
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[..., 0].unsqueeze(1).float()
    if x.ndim == 3:
        return x.unsqueeze(1).float()
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(1).float()
    raise ValueError(f"unexpected depth shape: {tuple(x.shape)}")


def prep_depth(
    depth: torch.Tensor,
    *,
    image_hw: int,
    depth_render_h: int | None = None,
    depth_render_w: int | None = None,
    depth_max: float = 5.0,
) -> torch.Tensor:
    """Preprocess depth to policy input [B, 1, image_hw, image_hw]."""
    src_is_int = not depth.dtype.is_floating_point
    x = _to_bchw(depth)

    x = torch.nan_to_num(x, nan=depth_max, posinf=depth_max, neginf=0.0)

    if depth_render_h is not None and depth_render_w is not None:
        rh, rw = int(depth_render_h), int(depth_render_w)
        if x.shape[-2] != rh or x.shape[-1] != rw:
            x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)

    if src_is_int:
        x = torch.clamp(x / 255.0, 0.0, 1.0)
    elif x.max() > 1.5:
        x = torch.clamp(x, 0.05, depth_max)
        x = torch.log1p(x) / torch.log1p(
            torch.tensor(depth_max, device=x.device, dtype=x.dtype)
        )
    else:
        x = torch.clamp(x, 0.0, 1.0)

    ih = int(image_hw)
    if x.shape[-2] != ih or x.shape[-1] != ih:
        x = F.interpolate(x, size=(ih, ih), mode="bilinear", align_corners=False)
    return x


# Back-compat alias
def preprocess_depth(
    depth: torch.Tensor,
    *,
    output_hw: int = 24,
    max_depth: float = 5.0,
    depth_render_h: int | None = None,
    depth_render_w: int | None = None,
) -> torch.Tensor:
    return prep_depth(
        depth,
        image_hw=output_hw,
        depth_render_h=depth_render_h,
        depth_render_w=depth_render_w,
        depth_max=max_depth,
    )
