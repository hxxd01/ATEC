"""Play / record video for Task D MARG depth pit DAgger student (MargDepthPitActorCritic)."""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ATEC_RL_LAB_SRC = os.path.join(_REPO_ROOT, "source", "atec_rl_lab")
if os.path.isdir(_ATEC_RL_LAB_SRC) and _ATEC_RL_LAB_SRC not in sys.path:
    sys.path.insert(0, _ATEC_RL_LAB_SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play Task D MARG depth pit DAgger student checkpoint (student-only inference)."
)
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (use 1 for cleaner video).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt checkpoint.")
parser.add_argument("--pit_width_min", type=float, default=0.4)
parser.add_argument("--pit_width_max", type=float, default=1.4)
parser.add_argument("--pit_curriculum_levels", type=int, default=11)
parser.add_argument(
    "--pit_level",
    type=int,
    default=10,
    help="Fixed pit-width row (0=narrowest). Default 10 = max width 1.4m (matches late DAgger train).",
)
parser.add_argument("--pit_curriculum", action="store_true", help="Full pit-width curriculum during play.")
parser.add_argument("--command_vx", type=float, default=0.6, help="Fixed forward velocity command (m/s).")
parser.add_argument("--spawn_x_offset", type=float, default=0.0)
parser.add_argument("--spawn_local_x", type=float, default=None)
parser.add_argument(
    "--pit_dr_light",
    action="store_true",
    help="Match train reset DR (joint/spawn/yaw/init-vel jitter).",
)
parser.add_argument(
    "--no_pit_box",
    action="store_true",
    help="Disable Task D push box in pit env.",
)
parser.add_argument(
    "--pit_box_pushed_spawn",
    action="store_true",
    help="Use teleop pushed box pose+jitter (new behavior). Default keeps legacy static box spawn.",
)
parser.add_argument("--video", action="store_true", help="Record mp4 via gym RecordVideo wrapper.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded rollout length in env steps.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Output folder (default: <checkpoint_dir>/videos/play).",
)
parser.add_argument("--real-time", action="store_true", help="Sleep to match sim real-time (GUI mode).")
parser.add_argument("--stochastic", action="store_true", help="Sample actions (default: deterministic mean).")
parser.add_argument("--debug", action="store_true", help="Print action/cmd/velocity stats every 50 steps.")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=0,
    help="Steps before recording (fills MARG proprio history; 0 = record from spawn).",
)
parser.add_argument("--policy_img_h", type=int, default=24)
parser.add_argument("--policy_img_w", type=int, default=32)
parser.add_argument("--sim_camera_h", type=int, default=24)
parser.add_argument("--sim_camera_w", type=int, default=32)
parser.add_argument("--depth_only", action="store_true", default=True)
parser.add_argument("--depth_max", type=float, default=5.0)
parser.add_argument(
    "--camera_far_clip",
    type=float,
    default=None,
    help="Camera far clip (m). Default 50 (matches train). Use 5 only to hide distant pits.",
)
parser.add_argument(
    "--platform_depth_train",
    "--pit_platform_depth",
    action="store_true",
    help="480x640 sim cameras + prep_depth -> policy size (must match train if used).",
)
parser.add_argument("--platform_depth_h", type=int, default=480)
parser.add_argument("--platform_depth_w", type=int, default=640)
parser.add_argument("--tiled_cameras", action="store_true", default=True)
parser.add_argument(
    "--ee_depth",
    action="store_true",
    help="Head+ee depth (768D->1536D). Default head-only; use for old head+ee checkpoints.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
# Keep play default compatible with known-good runs: static Task D box spawn (no pushed jitter).
args_cli.pit_box_default_spawn = not bool(getattr(args_cli, "pit_box_pushed_spawn", False))
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import atec_rl_lab.train  # noqa: F401
from atec_rl_lab.train.nav.marg_depth_pit_actor_critic import MargDepthPitActorCritic
from atec_rl_lab.train.nav.nav_cfg import TaskDMargDepthPitDaggerPPORunnerCfg
from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import (
    configure_pit_dagger_play_env_cfg,
    resolve_pit_depth_pipeline,
)
from atec_rl_lab.train.pit_marg.marg_pit_dagger_ppo import MargPitDaggerPPO

import rsl_rl.runners.on_policy_runner as _runner_mod


def _register_modules() -> None:
    _runner_mod.MargDepthPitActorCritic = MargDepthPitActorCritic
    _runner_mod.MargPitDaggerPPO = MargPitDaggerPPO


def _parse_isaaclab_env_yaml_hints(env_yaml_path: str) -> dict:
    """Extract play hints without full YAML load (Isaac Lab env.yaml uses ``!!python/tuple``)."""
    import re

    with open(env_yaml_path, encoding="utf-8") as f:
        text = f.read()

    hints: dict = {}

    m = re.search(r"^head_depth_only:\s*(\S+)", text, re.MULTILINE)
    if m:
        hints["head_depth_only"] = m.group(1).lower() in ("true", "yes", "1")

    idx = text.find("  head_camera:")
    if idx >= 0:
        chunk = text[idx : idx + 2500]
        w_m = re.search(r"^\s+width:\s*(\d+)\s*$", chunk, re.MULTILINE)
        h_m = re.search(r"^\s+height:\s*(\d+)\s*$", chunk, re.MULTILINE)
        if w_m and h_m:
            sim_w, sim_h = int(w_m.group(1)), int(h_m.group(1))
            hints["sim_camera_w"] = sim_w
            hints["sim_camera_h"] = sim_h
            hints["platform_depth_train"] = sim_h >= 240 or sim_w >= 320

    if re.search(r"^\s{2}ee_camera:\s*null\s*$", text, re.MULTILINE):
        hints.setdefault("head_depth_only", True)

    return hints


def _load_ckpt_agent_hints(ckpt_path: str) -> dict:
    """Optional overrides from ``params/*.yaml`` next to checkpoint."""
    import yaml

    params_dir = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "params")
    hints: dict = {}
    agent_yaml = os.path.join(params_dir, "agent.yaml")
    if os.path.isfile(agent_yaml):
        with open(agent_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        policy = data.get("policy") or {}
        if "head_depth_only" in policy:
            hints["head_depth_only"] = bool(policy["head_depth_only"])
        if "img_h" in policy:
            hints["img_h"] = int(policy["img_h"])
        if "img_w" in policy:
            hints["img_w"] = int(policy["img_w"])

    deploy_yaml = os.path.join(params_dir, "deploy_pit_e2e.yaml")
    if os.path.isfile(deploy_yaml):
        with open(deploy_yaml, encoding="utf-8") as f:
            deploy = yaml.safe_load(f) or {}
        if "head_depth_only" in deploy:
            hints["head_depth_only"] = bool(deploy["head_depth_only"])

    env_yaml = os.path.join(params_dir, "env.yaml")
    if os.path.isfile(env_yaml):
        hints.update(_parse_isaaclab_env_yaml_hints(env_yaml))
    return hints


def _apply_ckpt_hints(args, hints: dict) -> None:
    """Align play CLI with checkpoint train cfg unless user explicitly overrode."""
    if not hints:
        return
    if hints.get("head_depth_only") is True:
        if getattr(args, "ee_depth", False):
            print("[WARN] --ee_depth ignored: checkpoint trained head-only", flush=True)
        args.ee_depth = False
    elif hints.get("head_depth_only") is False and not getattr(args, "ee_depth", False):
        print("[WARN] checkpoint is head+ee; pass --ee_depth if load fails", flush=True)

    platform = hints.get("platform_depth_train")
    user_platform = bool(
        getattr(args, "platform_depth_train", False) or getattr(args, "pit_platform_depth", False)
    )
    if platform is True and not user_platform:
        args.platform_depth_train = True
        print(
            f"[TaskDMargDepthPitDAgger] auto --pit_platform_depth "
            f"(train sim={hints.get('sim_camera_h')}x{hints.get('sim_camera_w')})",
            flush=True,
        )
    elif platform is False and user_platform:
        print(
            "[WARN] --pit_platform_depth set but checkpoint env.yaml used native sim cameras",
            flush=True,
        )


def main():
    _register_modules()
    device = args_cli.device if args_cli.device else "cuda"
    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    ckpt_hints = _load_ckpt_agent_hints(ckpt)
    _apply_ckpt_hints(args_cli, ckpt_hints)

    env_cfg, spawn_local = configure_pit_dagger_play_env_cfg(args_cli)
    _, _, policy_h, policy_w, depth_mode = resolve_pit_depth_pipeline(args_cli)
    head_depth_only = not bool(getattr(args_cli, "ee_depth", False))

    agent_cfg = TaskDMargDepthPitDaggerPPORunnerCfg()
    agent_cfg.device = device
    agent_cfg.policy.img_h = int(ckpt_hints.get("img_h", policy_h))
    agent_cfg.policy.img_w = int(ckpt_hints.get("img_w", policy_w))
    agent_cfg.policy.depth_channels = 1 if args_cli.depth_only else 4
    agent_cfg.policy.head_depth_only = head_depth_only

    record_video = bool(args_cli.video)
    render_mode = "rgb_array" if record_video else None
    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg, render_mode=render_mode)

    if record_video:
        video_dir = args_cli.video_dir or os.path.join(os.path.dirname(ckpt), "videos", "play")
        video_dir = os.path.abspath(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == args_cli.warmup_steps,
            "video_length": int(args_cli.video_length),
            "disable_logger": True,
        }
        print("[TaskDMargDepthPitDAgger] Recording video:", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    print(f"[TaskDMargDepthPitDAgger] Loading checkpoint: {ckpt}", flush=True)
    runner.load(ckpt)
    runner.eval_mode()
    policy_nn = runner.alg.policy
    if args_cli.stochastic:
        policy = lambda obs: policy_nn.act(obs)
        print("[TaskDMargDepthPitDAgger] Stochastic policy.act()", flush=True)
    else:
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
        print("[TaskDMargDepthPitDAgger] Deterministic act_inference() (student-only)", flush=True)

    unwrapped = vec_env.unwrapped
    obs = vec_env.get_observations()
    dt = unwrapped.step_dt
    steps = 0
    warmup = max(0, int(args_cli.warmup_steps))
    max_steps = int(args_cli.video_length) if record_video else None

    print(
        f"[TaskDMargDepthPitDAgger] command_vx={args_cli.command_vx} m/s, "
        f"spawn_local_x={float(spawn_local[0]):.3f}, num_envs={args_cli.num_envs}, "
        f"pit_width={env_cfg.pit_width_range}, pit_level={args_cli.pit_level}, "
        f"depth={depth_mode} {policy_h}x{policy_w}, head_only={head_depth_only}, "
        f"stochastic={args_cli.stochastic}",
        flush=True,
    )
    if record_video and warmup > 0:
        print(f"[TaskDMargDepthPitDAgger] Warmup {warmup} steps before video", flush=True)

    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = policy(obs).to(unwrapped.device)
            if args_cli.debug and steps % 50 == 0:
                robot = unwrapped.scene["robot"]
                cmd_x = float(unwrapped.command_manager.get_command("base_velocity")[0, 0].item())
                vx = float(robot.data.root_lin_vel_w[0, 0].item())
                act_norm = float(actions[0].norm().item())
                depth_norm = float(obs["depth"][0].norm().item()) if "depth" in obs else -1.0
                ep_len = int(unwrapped.episode_length_buf[0].item())
                print(
                    f"[TaskDMargDepthPitDAgger debug] step={steps} cmd_x={cmd_x:.3f} vx={vx:.3f} "
                    f"|action|={act_norm:.3f} |depth|={depth_norm:.3f} ep_len={ep_len}",
                    flush=True,
                )
            obs, _, dones, _ = vec_env.step(actions)
            policy_nn.reset(dones)
        steps += 1
        if steps == warmup and record_video:
            print(f"[TaskDMargDepthPitDAgger] Warmup done, recording starts.", flush=True)
        if max_steps is not None and steps >= warmup + max_steps:
            print(
                f"[TaskDMargDepthPitDAgger] Recorded {steps - warmup} steps after warmup, stopping.",
                flush=True,
            )
            break
        if args_cli.real_time:
            sleep_s = dt - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
