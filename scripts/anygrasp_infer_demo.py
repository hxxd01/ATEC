#!/usr/bin/env python3
"""Run AnyGrasp on one RGB-D frame and print top-k grasps.

Usage example:
  python scripts/anygrasp_infer_demo.py \
    --rgb path/to/rgb.png \
    --depth path/to/depth.png \
    --checkpoint path/to/anygrasp.ckpt \
    --fx 458.12 --fy 458.12 --cx 320 --cy 240 \
    --depth-scale 1000 --topk 5
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any

import imageio.v3 as iio
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def _to_rgb_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"RGB image must be HxWxC or HxW, got shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.float32)
        max_v = float(np.nanmax(arr)) if arr.size else 0.0
        min_v = float(np.nanmin(arr)) if arr.size else 0.0
        if max_v <= 1.5 and min_v >= 0.0:
            arr = arr * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _load_rgb(path: str) -> np.ndarray:
    arr = iio.imread(path)
    return _to_rgb_hwc_uint8(np.asarray(arr))


def _load_depth(path: str, depth_scale: float) -> np.ndarray:
    arr = iio.imread(path)
    depth = np.asarray(arr)
    if depth.ndim == 3:
        # Some formats store single-channel depth as HxWx1.
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be HxW, got shape={depth.shape}")
    depth = depth.astype(np.float32) / float(depth_scale)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(depth)


def _load_intrinsics(args: argparse.Namespace) -> CameraIntrinsics:
    if args.intrinsics_json:
        with open(args.intrinsics_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        return CameraIntrinsics(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
        )
    if None in (args.fx, args.fy, args.cx, args.cy):
        raise ValueError("Provide --intrinsics-json OR all of --fx --fy --cx --cy")
    return CameraIntrinsics(
        fx=float(args.fx),
        fy=float(args.fy),
        cx=float(args.cx),
        cy=float(args.cy),
    )


def _try_import_anygrasp() -> tuple[type[Any], str]:
    # Known module names vary across AnyGrasp releases.
    candidates: list[tuple[str, str]] = [
        ("anygrasp", "AnyGrasp"),
        ("gsnet", "AnyGrasp"),
        ("graspnetAPI.anygrasp", "AnyGrasp"),
    ]
    last_error = None
    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            return cls, mod_name
        except Exception as e:  # pragma: no cover
            last_error = e
            continue
    raise ImportError(
        "Cannot import AnyGrasp. Install the AnyGrasp package/SDK first."
    ) from last_error


def _construct_anygrasp(
    cls: type[Any],
    checkpoint: str,
    max_gripper_width: float,
    gripper_height: float,
) -> Any:
    # Support different ctor signatures across releases.
    kwargs = {
        "checkpoint_path": checkpoint,
        "checkpoint": checkpoint,
        "model_path": checkpoint,
        "max_gripper_width": max_gripper_width,
        "gripper_height": gripper_height,
    }
    sig = inspect.signature(cls)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    try:
        return cls(**accepted)
    except TypeError:
        # Last fallback: positional checkpoint only.
        return cls(checkpoint)


def _call_infer(
    model: Any,
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    workspace: np.ndarray,
    topk: int,
) -> Any:
    # Different versions expose different API names.
    candidates = [
        "get_grasp",
        "inference",
        "predict",
        "infer",
    ]
    last_error = None
    for fn_name in candidates:
        if not hasattr(model, fn_name):
            continue
        fn = getattr(model, fn_name)
        try:
            return fn(
                rgb=rgb,
                depth=depth,
                camera_intrinsic=K,
                workspace_limits=workspace,
                topk=topk,
            )
        except TypeError:
            try:
                return fn(rgb, depth, K, workspace)
            except Exception as e:  # pragma: no cover
                last_error = e
                continue
        except Exception as e:  # pragma: no cover
            last_error = e
            continue
    raise RuntimeError(
        "AnyGrasp inference API not matched. Check your installed AnyGrasp version."
    ) from last_error


def _as_python(v: Any) -> Any:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _extract_grasp_list(out: Any) -> list[Any]:
    if out is None:
        return []
    if isinstance(out, (list, tuple)):
        # Common pattern: (gg, cloud) or (grasp_list, extra)
        if len(out) == 0:
            return []
        first = out[0]
        if isinstance(first, (list, tuple)):
            return list(first)
        return list(out)
    if hasattr(out, "grasp_group"):
        return list(getattr(out, "grasp_group"))
    if hasattr(out, "grasps"):
        return list(getattr(out, "grasps"))
    if hasattr(out, "__iter__"):
        return list(out)
    return [out]


def _grasp_score(g: Any) -> float:
    for key in ("score", "grasp_score", "confidence"):
        if hasattr(g, key):
            return float(getattr(g, key))
        if isinstance(g, dict) and key in g:
            return float(g[key])
    return 0.0


def _print_topk(grasps: list[Any], topk: int) -> None:
    if not grasps:
        print("No grasps returned.")
        return
    # Sort by score descending if score exists.
    ordered = sorted(grasps, key=_grasp_score, reverse=True)
    n = min(topk, len(ordered))
    print(f"Total grasps: {len(ordered)}; printing top-{n}")
    for i in range(n):
        g = ordered[i]
        score = _grasp_score(g)
        row = {"rank": i + 1, "score": score}
        # Best-effort fields across common grasp structures.
        fields = [
            "translation",
            "rotation_matrix",
            "width",
            "depth",
            "height",
            "center",
            "position",
        ]
        for f in fields:
            if hasattr(g, f):
                row[f] = _as_python(getattr(g, f))
            elif isinstance(g, dict) and f in g:
                row[f] = _as_python(g[f])
        print(json.dumps(row, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AnyGrasp single-frame RGB-D inference demo")
    p.add_argument("--rgb", type=str, required=True, help="Path to one RGB image.")
    p.add_argument("--depth", type=str, required=True, help="Path to one depth image.")
    p.add_argument("--checkpoint", type=str, required=True, help="AnyGrasp model checkpoint.")
    p.add_argument("--intrinsics-json", type=str, default=None, help="JSON with keys fx,fy,cx,cy.")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--depth-scale", type=float, default=1000.0, help="Depth unit scale to meters.")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-gripper-width", type=float, default=0.085)
    p.add_argument("--gripper-height", type=float, default=0.03)
    p.add_argument(
        "--workspace-limits",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=(-0.5, 0.5, -0.5, 0.5, 0.0, 1.2),
        help="Workspace bounds in camera frame (meters).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not os.path.isfile(args.rgb):
        raise FileNotFoundError(args.rgb)
    if not os.path.isfile(args.depth):
        raise FileNotFoundError(args.depth)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    intr = _load_intrinsics(args)
    rgb = _load_rgb(args.rgb)
    depth = _load_depth(args.depth, args.depth_scale)
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB/Depth shape mismatch: rgb={rgb.shape}, depth={depth.shape}")

    K = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    workspace = np.array(args.workspace_limits, dtype=np.float32).reshape(3, 2)

    AnyGraspCls, module_name = _try_import_anygrasp()
    print(f"Imported AnyGrasp from module: {module_name}")
    model = _construct_anygrasp(
        AnyGraspCls,
        checkpoint=args.checkpoint,
        max_gripper_width=float(args.max_gripper_width),
        gripper_height=float(args.gripper_height),
    )

    # Some versions require explicit load call.
    if hasattr(model, "load_net"):
        model.load_net()
    elif hasattr(model, "load"):
        try:
            model.load()
        except TypeError:
            model.load(args.checkpoint)

    out = _call_infer(
        model=model,
        rgb=rgb,
        depth=depth,
        K=K,
        workspace=workspace,
        topk=int(args.topk),
    )
    grasps = _extract_grasp_list(out)
    _print_topk(grasps, int(args.topk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

