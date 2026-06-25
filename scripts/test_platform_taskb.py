#!/usr/bin/env python3
"""Smoke-test Task B solution the same way demo/server.py calls it (no bind_env)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "demo"))

os.environ.setdefault("DISABLE_O3D_VIS", "1")
os.environ.setdefault("ATEC_ENABLE_GRASPNET", "0")

from solution import AlgSolution  # noqa: E402


def _fake_obs(*, use_head: bool, proprio_dim: int = 72) -> dict:
    h, w = 480, 640
    proprio = torch.zeros(1, proprio_dim, dtype=torch.float32)
    extero = torch.zeros(1, 5760, dtype=torch.float32)
    ee_rgb = torch.zeros(1, h, w, 3, dtype=torch.uint8)
    ee_depth = torch.full((1, h, w, 1), 1.5, dtype=torch.float32)
    if use_head:
        return {
            "proprio": proprio,
            "extero": extero,
            "image": {
                "head_rgb": torch.zeros(1, h, w, 3, dtype=torch.uint8),
                "head_depth": ee_depth.clone(),
                "ee_rgb": ee_rgb,
                "ee_depth": ee_depth,
            },
        }
    return {
        "proprio": proprio,
        "extero": extero,
        "image": {
            "video_rgb": torch.zeros(1, h, w, 3, dtype=torch.uint8),
            "video_depth": ee_depth.clone(),
            "ee_rgb": ee_rgb,
            "ee_depth": ee_depth,
        },
    }


def _run_case(label: str, obs: dict) -> None:
    agent = AlgSolution()
    agent.reset()
    resp = agent.predicts(obs=obs, current_score=0.0)
    action = resp["action"]
    assert isinstance(action, list), f"{label}: action must be list"
    assert len(action) == 20, f"{label}: expected 20-dim B2Piper action, got {len(action)}"
    assert resp["giveup"] is False
    print(f"[ok] {label}: action_dim={len(action)} status={agent.status.name}")


def main() -> None:
    _run_case("head_rgb path", _fake_obs(use_head=True))
    _run_case("video_rgb path", _fake_obs(use_head=False))
    print("platform Task B smoke test passed")


if __name__ == "__main__":
    main()
