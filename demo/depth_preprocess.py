"""Shared depth preprocessing: train (taskd_student_env) and deploy (solution.py).

Pipeline (must match taskd_student_env._prep_depth):
  1. -> Bx1xHxW, nan_to_num
  2. bilinear to image_h x image_w if needed (platform 480x640 -> 24x32, keeps aspect)
  3. uint8 / log1p normalize (meters if max > 1.5)
  4. output [B, 1, image_h, image_w] — no square squash
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
    image_h: int,
    image_w: int,
    depth_max: float = 5.0,
    image_hw: int | None = None,
) -> torch.Tensor:
    """Preprocess depth to policy input [B, 1, image_h, image_w]."""
    if image_hw is not None:
        image_h = image_w = int(image_hw)
    ih, iw = int(image_h), int(image_w)
    src_is_int = not depth.dtype.is_floating_point
    x = _to_bchw(depth)

    x = torch.nan_to_num(x, nan=depth_max, posinf=depth_max, neginf=0.0)
    # Isaac mdp.image() maps inf -> 0 before obs reaches deploy code; treat as far clip (depth_max),
    # matching raw camera buffers that prep_depth sees during training (inf/nan -> depth_max).
    if not src_is_int:
        invalid = (~torch.isfinite(x)) | (x <= 0.0)
        if invalid.any():
            x = torch.where(invalid, torch.full_like(x, depth_max), x)

    if x.shape[-2] != ih or x.shape[-1] != iw:
        x = F.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)

    if src_is_int:
        x = torch.clamp(x / 255.0, 0.0, 1.0)
    elif x.max() > 1.5:
        x = torch.clamp(x, 0.05, depth_max)
        x = torch.log1p(x) / torch.log1p(
            torch.tensor(depth_max, device=x.device, dtype=x.dtype)
        )
    else:
        x = torch.clamp(x, 0.0, 1.0)

    return x


# Back-compat alias
def preprocess_depth(
    depth: torch.Tensor,
    *,
    output_hw: int = 24,
    max_depth: float = 5.0,
    image_h: int | None = None,
    image_w: int | None = None,
) -> torch.Tensor:
    if image_h is not None and image_w is not None:
        ih, iw = int(image_h), int(image_w)
    else:
        ih = iw = int(output_hw)
    return prep_depth(depth, image_h=ih, image_w=iw, depth_max=max_depth)
