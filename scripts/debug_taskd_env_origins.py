"""Quick check: scene.env_origins vs env_spacing / terrain grid (Task D)."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print Task D env_origins for spacing/terrain debug.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--env_spacing", type=float, default=10.0)
parser.add_argument("--show", type=int, default=16, help="How many env rows to print.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import atec_rl_lab.tasks  # noqa: F401
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg, refresh_task_d_terrain_cfg


def _print_env_origins_debug(base_env, *, env_spacing: float, show: int) -> None:
    scene = base_env.unwrapped.scene
    origins = scene.env_origins
    n = min(int(show), int(origins.shape[0]))
    xy = origins[:n, :2].detach().cpu().numpy()
    tg = getattr(getattr(scene, "terrain", None), "terrain_generator", None)
    tg_cfg = getattr(tg, "cfg", None) if tg is not None else None
    rows = getattr(tg_cfg, "num_rows", None)
    cols = getattr(tg_cfg, "num_cols", None)
    print(
        f"[TaskDDebug] num_envs={origins.shape[0]}  scene.env_spacing(cfg)={env_spacing}  "
        f"terrain num_rows={rows} num_cols={cols}",
        flush=True,
    )
    if getattr(scene, "terrain", None) is not None and scene.terrain.terrain_origins is not None:
        t_orig = scene.terrain.terrain_origins.detach().cpu().numpy()
        print(f"[TaskDDebug] terrain_origins grid shape={t_orig.shape}", flush=True)
    print(f"[TaskDDebug] env_origins[:{n}, :2]:\n{xy}", flush=True)
    rounded = [tuple(row) for row in xy.round(3)]
    uniq = len(set(rounded))
    print(f"[TaskDDebug] unique xy (3dp): {uniq}/{n}", flush=True)
    if uniq <= 1:
        print(
            "[TaskDDebug] WARNING: env_origins are identical -> terrain is still 1x1 (or all envs map to one tile). "
            "env_spacing alone does not create separate pits; need multi-tile terrain.",
            flush=True,
        )
    else:
        print("[TaskDDebug] env_origins differ across envs (terrain grid or assignment OK).", flush=True)


def main():
    env_cfg = TaskDEnvB2Cfg()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    refresh_task_d_terrain_cfg(env_cfg)
    if env_cfg.observations is not None:
        env_cfg.observations.image = None
        env_cfg.observations.extero = None
    if getattr(env_cfg.scene, "lidar_sensor", None) is not None:
        env_cfg.scene.lidar_sensor = None
    env_cfg.scene.head_camera = None
    env_cfg.scene.ee_camera = None

    print(
        f"[TaskDDebug] Creating ATEC-TaskD-B2Piper num_envs={env_cfg.scene.num_envs} "
        f"env_spacing={env_cfg.scene.env_spacing}",
        flush=True,
    )
    env = gym.make("ATEC-TaskD-B2Piper", cfg=env_cfg)
    _print_env_origins_debug(env, env_spacing=env_cfg.scene.env_spacing, show=args_cli.show)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
