"""Debug hierarchical nav: print actions and robot state every inner step (num_envs=1).

Usage:
  python scripts/debug_nav_step.py --ll_policy demo/policy.pt --headless
  python scripts/debug_nav_step.py --ll_policy demo/policy.pt --headless --nav_cmd 1,0,0
  python scripts/debug_nav_step.py --ll_policy demo/policy.pt --headless --nav_steps 30
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug nav env step-by-step (1 env).")
parser.add_argument("--ll_policy", type=str, required=True)
parser.add_argument("--nav_dt", type=float, default=1.0, help="Seconds per nav decision.")
parser.add_argument("--inner_steps", type=int, default=None)
parser.add_argument("--nav_steps", type=int, default=20, help="Number of high-level nav steps.")
parser.add_argument(
    "--nav_cmd",
    type=str,
    default="1,0,0",
    help="Fixed nav command as vx,vy,yaw in [-1,1]. Use 'rand' for Gaussian noise.",
)
parser.add_argument("--no_random_spawn", action="store_true", help="Spawn at default -141 only.")
parser.add_argument("--no_lidar", action="store_true")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import atec_rl_lab.tasks  # noqa: F401
from atec_rl_lab.tasks.task_a.env_cfg import TaskAEnvB2Cfg
from atec_rl_lab.train.nav.hierarchical_env import HierarchicalNavEnv
from atec_rl_lab.train.nav.speed_cfg import apply_nav_speed_cfg, apply_nav_training_env_cfg


def _parse_nav_cmd(s: str) -> torch.Tensor:
    if s.strip().lower() == "rand":
        return torch.randn(1, 3)
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--nav_cmd must be 'vx,vy,yaw' or 'rand'")
    return torch.tensor(parts, dtype=torch.float32).view(1, 3)


def _term_flags(base) -> dict:
    out = {}
    if hasattr(base, "reset_terminated"):
        t = base.reset_terminated[0].item() if base.reset_terminated.numel() else False
        out["terminated"] = bool(t)
    if hasattr(base, "reset_time_outs"):
        to = base.reset_time_outs[0].item() if base.reset_time_outs.numel() else False
        out["time_out"] = bool(to)
    if hasattr(base, "termination_manager"):
        tm = base.termination_manager
        for name in getattr(tm, "active_terms", tm._term_names):
            try:
                out[name] = bool(tm.get_term(name)[0].item())
            except Exception:
                pass
    return out


def _robot_state(base) -> dict:
    r = base.scene["robot"].data
    pos = r.root_pos_w[0].cpu().tolist()
    vel = r.root_lin_vel_w[0].cpu().tolist()
    quat = r.root_quat_w[0].cpu().tolist()
    return {"pos_w": pos, "lin_vel_w": vel, "quat_w": quat}


def main():
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda"

    env_cfg = TaskAEnvB2Cfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.head_camera = None
    env_cfg.scene.ee_camera = None
    env_cfg.scene.ee_dual_camera = None
    env_cfg.observations.image = None
    env_cfg.episode_length_s = 120.0
    env_step_dt = float(env_cfg.decimation) * float(env_cfg.sim.dt)
    inner_steps = (
        max(1, int(args_cli.inner_steps))
        if args_cli.inner_steps is not None
        else max(1, int(round(args_cli.nav_dt / env_step_dt)))
    )
    print(
        f"[DEBUG] nav_dt={inner_steps * env_step_dt:.2f}s  inner_steps={inner_steps}",
        flush=True,
    )

    apply_nav_training_env_cfg(
        env_cfg,
        stuck_time_s=99.0,  # disable stuck during this debug run
        randomize_spawn=not args_cli.no_random_spawn,
    )
    # Remove stuck term so we only see fall / timeout
    if hasattr(env_cfg.terminations, "stuck_no_progress"):
        env_cfg.terminations.stuck_no_progress = None

    env_cfg, extero_raw_dims, lidar_bins = apply_nav_speed_cfg(
        env_cfg, no_lidar=args_cli.no_lidar,
    )

    print("[DEBUG] Creating env (num_envs=1)...", flush=True)
    env = gym.make("ATEC-TaskA-B2Piper", cfg=env_cfg, render_mode=None)
    nav_env = HierarchicalNavEnv(
        env,
        ll_policy_path=args_cli.ll_policy,
        device=device,
        inner_steps=inner_steps,
        lidar_bins=lidar_bins,
        extero_raw_dims=extero_raw_dims,
    )

    base = env.unwrapped
    nav_cmd_fixed = _parse_nav_cmd(args_cli.nav_cmd).to(device)

    obs_dict, _ = nav_env.reset()
    print(f"[DEBUG] reset done. robot={_robot_state(base)}", flush=True)

    scale_t2e = nav_env._t2e

    for nav_i in range(1, args_cli.nav_steps + 1):
        if args_cli.nav_cmd.strip().lower() == "rand":
            nav_action = torch.randn(1, 3, device=device).clamp(-1.0, 1.0)
        else:
            nav_action = nav_cmd_fixed.clone()

        nav_action = nav_action.clamp(-1.0, 1.0)
        vel_cmd = nav_env.nav_action_to_vel_cmd(nav_action)

        print(
            f"\n{'='*72}\n"
            f"NAV step {nav_i}  nav_action={nav_action[0].tolist()}  "
            f"vel_cmd(scaled)={vel_cmd[0].tolist()}",
            flush=True,
        )

        total_rew = 0.0
        term_any = False
        trunc_any = False

        for k in range(nav_env.inner_steps):
            proprio = nav_env._current_obs["proprio"].to(device)
            env_vel_cmd = proprio[0, 6:9].tolist()
            ll_obs = nav_env._build_ll_obs(nav_env._current_obs, nav_action)
            with torch.inference_mode():
                ll_act_tr = nav_env.ll_policy(ll_obs)
            env_action = nav_env._build_env_action(ll_act_tr)

            pos_before = _robot_state(base)

            obs, rew, term, trunc, info = env.step(env_action)
            nav_env._current_obs = obs

            r = float(rew.squeeze().item()) if isinstance(rew, torch.Tensor) else float(rew)
            total_rew += r
            t = bool(term.squeeze().item()) if isinstance(term, torch.Tensor) else bool(term)
            tr = bool(trunc.squeeze().item()) if isinstance(trunc, torch.Tensor) else bool(trunc)
            term_any |= t
            trunc_any |= tr

            pos_after = _robot_state(base)
            flags = _term_flags(base)

            print(
                f"  inner#{k}  rew={r:+.5f}  term={t} trunc={tr}  flags={flags}",
                flush=True,
            )
            print(
                f"           pos_before x={pos_before['pos_w'][0]:.3f} "
                f"y={pos_before['pos_w'][1]:.3f} z={pos_before['pos_w'][2]:.3f}  "
                f"vx_w={pos_before['lin_vel_w'][0]:+.3f}",
                flush=True,
            )
            print(
                f"           pos_after  x={pos_after['pos_w'][0]:.3f} "
                f"y={pos_after['pos_w'][1]:.3f} z={pos_after['pos_w'][2]:.3f}  "
                f"vx_w={pos_after['lin_vel_w'][0]:+.3f}",
                flush=True,
            )
            print(
                f"           proprio_vel_cmd(env)={env_vel_cmd}  "
                f"ll_obs_vel_cmd={ll_obs[0, 6:9].tolist()}",
                flush=True,
            )
            print(
                f"           ll_act_tr[0:4]={ll_act_tr[0, :4].tolist()}  "
                f"env_leg_raw[0:4]={env_action[0, :4].tolist()}",
                flush=True,
            )
            print(
                f"           gravity={ll_obs[0, 3:6].tolist()}  "
                f"ang_vel*0.25={ll_obs[0, :3].tolist()}",
                flush=True,
            )

            if t or tr:
                log = info.get("log", {}) if isinstance(info, dict) else {}
                if log:
                    print(f"           episode_log keys={list(log.keys())[:8]}...", flush=True)
                break

        ep_nav = int(nav_env.episode_length_buf[0].item())
        print(
            f"  NAV#{nav_i} summary: total_rew={total_rew:+.5f}  "
            f"term={term_any} trunc={trunc_any}  ep_len_nav={ep_nav}",
            flush=True,
        )
        if term_any or trunc_any:
            print("[DEBUG] Episode ended — env auto-reset; continuing with new episode.", flush=True)

    env.close()
    print("\n[DEBUG] Done.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
