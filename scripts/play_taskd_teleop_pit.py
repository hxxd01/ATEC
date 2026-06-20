#!/usr/bin/env python3
"""Play Task D: hierarchical nav student → pit MARG depth student (play_atec_task wrapper)."""

from __future__ import annotations

import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PLAY = os.path.join(_ROOT, "scripts", "play_atec_task.py")

DEFAULT_NAV_CKPT = os.path.join(
    _ROOT,
    "logs/rsl_rl/taskd_student_b2piper_depth/2026-06-20_16-12-47/model_1600.pt",
)
DEFAULT_TRAJ = os.path.join(_ROOT, "scripts", "ATEC-TaskD-B2Piper_20260619_153715.json")
DEFAULT_PIT_CKPT = os.path.join(
    _ROOT,
    "logs/rsl_rl/taskd_marg_depth_pit_dagger_b2piper/2026-06-19_04-34-32/model_800.pt",
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Task D nav→pit play (hierarchical nav phase 1 + pit MARG student phase 2)."
    )
    parser.add_argument(
        "--nav_ckpt",
        type=str,
        default=None,
        help=f"Nav student checkpoint for phase 1 (default: {DEFAULT_NAV_CKPT}).",
    )
    parser.add_argument(
        "--traj",
        type=str,
        default=None,
        help="Legacy: teleop JSON instead of nav student (mutually exclusive with --nav_ckpt).",
    )
    parser.add_argument("--pit_ckpt", type=str, default=DEFAULT_PIT_CKPT)
    parser.add_argument("--pit_command_vx", type=float, default=0.6)
    parser.add_argument(
        "--pit_handoff_warmup",
        type=int,
        default=0,
        help="Pit steps after nav/teleop with ramped actions + history fill (0 = full pit policy immediately).",
    )
    parser.add_argument(
        "--nav_only",
        action="store_true",
        help="Disable nav→pit handoff; run nav student only until episode ends.",
    )
    parser.add_argument(
        "--ee_depth",
        action="store_true",
        help="Use head+ee depth for pit student. Nav phase always uses head+ee.",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--video_length",
        type=int,
        default=600,
        help="Pit phase steps; nav/teleop steps added automatically for --video.",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num_envs", type=int, default=1)
    args, extra = parser.parse_known_args()

    if args.traj and args.nav_ckpt:
        parser.error("Use either --nav_ckpt or --traj (legacy teleop), not both.")
    if args.nav_only and args.traj:
        parser.error("--nav_only requires --nav_ckpt (not legacy --traj)")

    nav_ckpt = os.path.abspath(args.nav_ckpt or DEFAULT_NAV_CKPT)
    cmd = [
        sys.executable,
        _PLAY,
        "--task",
        "ATEC-TaskD-B2Piper",
        "--num_envs",
        str(int(args.num_envs)),
        "--platform_depth",
    ]
    if args.nav_only:
        cmd += ["--nav_only", "--nav_ckpt", nav_ckpt]
    else:
        cmd += [
            "--pit_ckpt",
            os.path.abspath(args.pit_ckpt),
            "--pit_command_vx",
            str(float(args.pit_command_vx)),
        ]
        if args.traj:
            cmd += ["--teleop_traj", os.path.abspath(args.traj)]
        else:
            cmd += ["--nav_ckpt", nav_ckpt]
    if args.video:
        cmd += ["--video", "--video_length", str(int(args.video_length))]
    if args.headless:
        cmd.append("--headless")
    if args.ee_depth:
        cmd.append("--ee_depth")
    if not args.nav_only:
        cmd.extend(["--pit_handoff_warmup", str(int(args.pit_handoff_warmup))])
    cmd.extend(extra)

    print("[play_taskd_teleop_pit]", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=_ROOT))


if __name__ == "__main__":
    main()
