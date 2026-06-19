#!/usr/bin/env python3
"""Pit MARG depth student ablation — **same chain as DAgger train / play**.

Uses ``play_taskd_marg_depth_pit_dagger.py`` directly:
  - env: ATEC-TaskD-PitLocomotion-B2Piper (12D leg actions)
  - obs: obs_manager ``proprio`` + ``proprio_history`` + ``depth`` (no manual solution obs)
  - policy: OnPolicyRunner + ``act_inference`` (MargDepthPitActorCritic)
  - cameras: same as train (default native 24x32, far clip 50m; auto-read params/env.yaml)

Task D B2Piper teleop→pit deploy remains ``play_taskd_teleop_pit.py`` (20D + box OOD).
"""

from __future__ import annotations

import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DAGGER_PLAY = os.path.join(_ROOT, "scripts", "play_taskd_marg_depth_pit_dagger.py")

DEFAULT_CKPT = os.path.join(
    _ROOT,
    "logs/rsl_rl/taskd_marg_depth_pit_dagger_b2piper/2026-06-19_21-36-25/model_400.pt",
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Pit student ablation on PitLocomotion (training-identical play chain). "
            "Not Task D B2Piper — use play_taskd_teleop_pit.py for full Task D deploy."
        )
    )
    parser.add_argument("--pit_ckpt", type=str, default=DEFAULT_CKPT, help="Alias for --checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="DAgger student checkpoint.")
    parser.add_argument("--pit_command_vx", type=float, default=0.6, help="Alias for --command_vx.")
    parser.add_argument("--command_vx", type=float, default=None)
    parser.add_argument("--pit_level", type=int, default=10)
    parser.add_argument("--pit_width_min", type=float, default=1.4)
    parser.add_argument("--pit_width_max", type=float, default=1.4)
    parser.add_argument("--pit_curriculum", action="store_true")
    parser.add_argument("--spawn_x_offset", type=float, default=0.0)
    parser.add_argument("--spawn_local_x", type=float, default=None)
    parser.add_argument("--ee_depth", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=600)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--policy_img_h", type=int, default=24)
    parser.add_argument("--policy_img_w", type=int, default=32)
    parser.add_argument("--sim_camera_h", type=int, default=24)
    parser.add_argument("--sim_camera_w", type=int, default=32)
    parser.add_argument("--depth_max", type=float, default=5.0)
    parser.add_argument(
        "--camera_far_clip",
        type=float,
        default=None,
        help="Camera far clip (m). Default 50 (matches DAgger train).",
    )
    parser.add_argument(
        "--pit_platform_depth",
        "--platform_depth_train",
        action="store_true",
        help="Match train with 480x640 sim cameras (auto from checkpoint env.yaml if omitted).",
    )
    parser.add_argument("--platform_depth_h", type=int, default=480)
    parser.add_argument("--platform_depth_w", type=int, default=640)
    parser.add_argument("--no_tiled_cameras", action="store_true")
    args, extra = parser.parse_known_args()

    ckpt = args.checkpoint or args.pit_ckpt
    cmd_vx = args.command_vx if args.command_vx is not None else args.pit_command_vx

    cmd = [
        sys.executable,
        _DAGGER_PLAY,
        "--checkpoint",
        os.path.abspath(ckpt),
        "--command_vx",
        str(float(cmd_vx)),
        "--pit_level",
        str(int(args.pit_level)),
        "--pit_width_min",
        str(float(args.pit_width_min)),
        "--pit_width_max",
        str(float(args.pit_width_max)),
        "--num_envs",
        str(int(args.num_envs)),
        "--warmup_steps",
        str(int(args.warmup_steps)),
        "--policy_img_h",
        str(int(args.policy_img_h)),
        "--policy_img_w",
        str(int(args.policy_img_w)),
        "--sim_camera_h",
        str(int(args.sim_camera_h)),
        "--sim_camera_w",
        str(int(args.sim_camera_w)),
        "--depth_max",
        str(float(args.depth_max)),
    ]
    if args.camera_far_clip is not None:
        cmd += ["--camera_far_clip", str(float(args.camera_far_clip))]
    if getattr(args, "pit_platform_depth", False) or getattr(args, "platform_depth_train", False):
        cmd.append("--pit_platform_depth")
        cmd += [
            "--platform_depth_h",
            str(int(args.platform_depth_h)),
            "--platform_depth_w",
            str(int(args.platform_depth_w)),
        ]
    cmd += [
        "--spawn_x_offset",
        str(float(args.spawn_x_offset)),
    ]
    if args.spawn_local_x is not None:
        cmd += ["--spawn_local_x", str(float(args.spawn_local_x))]
    if args.pit_curriculum:
        cmd.append("--pit_curriculum")
    if args.video:
        cmd += ["--video", "--video_length", str(int(args.video_length))]
    if args.video_dir:
        cmd += ["--video_dir", os.path.abspath(args.video_dir)]
    if args.headless:
        cmd.append("--headless")
    if args.debug:
        cmd.append("--debug")
    if args.real_time:
        cmd.append("--real-time")
    if args.stochastic:
        cmd.append("--stochastic")
    if args.ee_depth:
        cmd.append("--ee_depth")
    if not args.no_tiled_cameras:
        cmd.append("--tiled_cameras")
    cmd.append("--depth_only")
    cmd.extend(extra)

    print("[play_taskd_pit_student_platform] train-chain play:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=_ROOT))


if __name__ == "__main__":
    main()
