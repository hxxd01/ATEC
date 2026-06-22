"""LiDAR bearing + head depth range for Task B trash-bin approach."""

from __future__ import annotations

import numpy as np
from scipy.stats import circmean


def get_dustbin_direction(z_hits, *, debug: bool = False) -> float | None:
    """Trash-bin azimuth (rad) in robot base frame; 0=forward, +=left."""
    channels = 16
    h_res = 1.0
    h_steps = int(360 / h_res)

    v_angles_deg = np.linspace(-20.0, 20.0, channels)
    h_angles_deg = np.linspace(-180.0, 180.0 - h_res, h_steps)
    v_rad = np.radians(v_angles_deg)
    h_rad = np.radians(h_angles_deg)
    h_mesh, _v_mesh = np.meshgrid(h_rad, v_rad)
    h_flat = h_mesh.flatten()

    z = np.asarray(z_hits, dtype=np.float64).reshape(-1)
    finite = np.isfinite(z)
    if not finite.any():
        return None

    rounded = np.round(z[finite], 2)
    values, counts = np.unique(rounded, return_counts=True)
    ground_ref = float(values[np.argmax(counts)])
    if debug:
        print(f"[lidar] ground={ground_ref:.3f}", flush=True)

    if h_flat.size != z.size:
        h_flat = h_flat[: z.size]

    obstacle = finite & (z < ground_ref - 0.01)
    h_valid = h_flat[obstacle]
    if h_valid.size < 8:
        return None

    span_deg = np.degrees(float(np.max(h_valid) - np.min(h_valid)))
    if span_deg > 120.0:
        return None

    bearing = float(circmean(h_valid, high=np.pi, low=-np.pi))
    if debug:
        print(f"[lidar] bearing={np.degrees(bearing):+.1f}° rays={h_valid.size}", flush=True)
    return bearing


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def depth_range_at_bearing(
    depth: np.ndarray,
    bearing_rad: float,
    K: np.ndarray,
    *,
    half_deg: float = 20.0,
    v0: int = 80,
    v1: int = 400,
    min_depth: float = 0.2,
    max_depth: float = 12.0,
) -> float | None:
    """Median depth (m) along head camera cone at LiDAR bearing — bin range, not trash."""
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        return None

    fx, cx = float(K[0, 0]), float(K[0, 2])
    h, w = depth.shape
    v0 = max(0, min(v0, h - 1))
    v1 = max(v0 + 1, min(v1, h))

    cols: list[int] = []
    for u in range(w):
        ang = np.arctan2(-(u - cx), fx)
        diff = _wrap_angle(ang - bearing_rad)
        if abs(diff) <= np.radians(half_deg):
            cols.append(u)
    if not cols:
        return None

    patch = depth[v0:v1, cols]
    valid = np.isfinite(patch) & (patch > min_depth) & (patch < max_depth)
    if not valid.any():
        return None
    return float(np.median(patch[valid]))


def approach_dustbin(
    z_hits: np.ndarray | None,
    head_depth: np.ndarray | None,
    K: np.ndarray,
    *,
    default_range: float = 4.0,
    debug: bool = False,
) -> dict:
    """GO_BIN: LiDAR yaw + head depth distance to bin (no trash clustering)."""
    bearing = get_dustbin_direction(z_hits, debug=debug) if z_hits is not None else None
    dist = None
    if bearing is not None and head_depth is not None:
        dist = depth_range_at_bearing(head_depth, bearing, K)

    if bearing is None:
        return {"source": "none", "target": None, "dist": None, "bearing": None}

    use_dist = float(dist) if dist is not None else default_range
    target = np.array([use_dist * np.cos(bearing), use_dist * np.sin(bearing)], dtype=np.float64)
    source = "lidar+depth_range" if dist is not None else "lidar_bearing"
    return {
        "source": source,
        "target": target,
        "dist": dist,
        "bearing": bearing,
    }
