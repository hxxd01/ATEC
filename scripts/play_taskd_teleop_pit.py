#!/usr/bin/env python3
"""Play Task D: teleop traj replay → pit MARG depth student via play_atec_task + solution_marg_depth_pit."""

from __future__ import annotations

import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PLAY = os.path.join(_ROOT, "scripts", "play_atec_task.py")

DEFAULT_TRAJ = os.path.join(_ROOT, "scripts", "ATEC-TaskD-B2Piper_20260619_153715.json")
DEFAULT_CKPT = os.path.join(
    _ROOT,
    "logs/rsl_rl/taskd_marg_depth_pit_dagger_b2piper/2026-06-19_04-34-32/model_800.pt",
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Task D teleop→pit play (play_atec_task wrapper).")
    parser.add_argument("--traj", type=str, default=DEFAULT_TRAJ)
    parser.add_argument("--pit_ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--pit_command_vx", type=float, default=0.6)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=600, help="Pit phase steps; teleop steps added automatically.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num_envs", type=int, default=1)
    args, extra = parser.parse_known_args()

    cmd = [
        sys.executable,
        _PLAY,
        "--task",
        "ATEC-TaskD-B2Piper",
        "--teleop_traj",
        os.path.abspath(args.traj),
        "--pit_ckpt",
        os.path.abspath(args.pit_ckpt),
        "--pit_command_vx",
        str(float(args.pit_command_vx)),
        "--num_envs",
        str(int(args.num_envs)),
        "--platform_depth",
    ]
    if args.video:
        cmd += ["--video", "--video_length", str(int(args.video_length))]
    if args.headless:
        cmd.append("--headless")
    cmd.extend(extra)

    print("[play_taskd_teleop_pit]", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=_ROOT))


if __name__ == "__main__":
    main()
