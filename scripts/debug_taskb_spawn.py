"""Print Task-B spawn geometry: env_origin, robot world pos, each object world pos, ground footprint.

Usage:
    python scripts/debug_taskb_spawn.py --task ATEC-TaskB-B2Piper --num_envs 1 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print Task B spawn coordinates for terrain/spawn debug.")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--show", type=int, default=4, help="How many envs to print objects for.")
parser.add_argument("--nav", action="store_true", help="Use TaskBNavEnvB2Cfg (nav training cfg).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401,E402


def main():
    if args_cli.nav:
        from atec_rl_lab.tasks.task_b.env_cfg import TaskBNavEnvB2Cfg, refresh_task_b_terrain_cfg
        env_cfg = TaskBNavEnvB2Cfg()
        env_cfg.scene.num_envs = int(args_cli.num_envs)
        refresh_task_b_terrain_cfg(env_cfg)
        if env_cfg.observations is not None:
            env_cfg.observations.image = None
            env_cfg.observations.extero = None
        if getattr(env_cfg.scene, "lidar_sensor", None) is not None:
            env_cfg.scene.lidar_sensor = None
    else:
        # Build cfg via the same path play uses.
        try:
            from isaaclab_tasks.utils import parse_env_cfg
            env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
        except Exception as e:
            print(f"[TaskBDebug] parse_env_cfg failed ({e}); constructing TaskBEnvB2Cfg directly", flush=True)
            from atec_rl_lab.tasks.task_b.env_cfg import TaskBEnvB2Cfg
            env_cfg = TaskBEnvB2Cfg()
            env_cfg.scene.num_envs = int(args_cli.num_envs)

    print(
        f"[TaskBDebug] cfg class={type(env_cfg).__name__} scene.num_envs={env_cfg.scene.num_envs}",
        flush=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    scene = base.scene

    # Force a reset so reset-time events (reset_robot_root, randomize_task_b_objects)
    # fire and place the robot/objects at their env_origin-relative positions.
    print("\n[TaskBDebug] calling env.reset() ...", flush=True)
    env.reset()
    print("[TaskBDebug] env.reset() done.", flush=True)

    origins = scene.env_origins
    n_envs = int(origins.shape[0])
    n_show = min(int(args_cli.show), n_envs)
    print(f"\n[TaskBDebug] num_envs={n_envs}", flush=True)
    print(f"[TaskBDebug] env_origins[:{n_show}]:\n{origins[:n_show].detach().cpu().numpy()}", flush=True)

    robot = scene["robot"]
    rpos = robot.data.root_pos_w
    print(
        f"\n[TaskBDebug] robot.root_pos_w[:{n_show}]:\n{rpos[:n_show].detach().cpu().numpy()}",
        flush=True,
    )

    # Terrain origins / ground info if available.
    terrain = getattr(scene, "terrain", None)
    if terrain is not None:
        t_orig = getattr(terrain, "terrain_origins", None)
        if t_orig is not None:
            print(
                f"\n[TaskBDebug] terrain_origins shape={tuple(t_orig.shape)}:\n{t_orig.reshape(-1,3)[:n_show].detach().cpu().numpy()}",
                flush=True,
            )

    def _get_obj(idx):
        for attr in ("keys",):
            pass
        try:
            return scene[f"object_{idx}"]
        except Exception:
            return None

    # Each object world pos.
    print("\n[TaskBDebug] object root_pos_w (env 0):", flush=True)
    for obj_idx in range(1, 19):
        obj = _get_obj(obj_idx)
        if obj is None:
            continue
        try:
            opos = obj.data.root_pos_w
            row = opos[0].detach().cpu().tolist()
        except Exception as e:
            row = f"<err {e}>"
        print(f"  object_{obj_idx}: {row}", flush=True)

    # Compute, relative to env_origin[0], the xy of robot + all objects (env 0).
    o0 = origins[0].detach().cpu().numpy()
    print(f"\n[TaskBDebug] === env0 local (world - env_origin[0]={o0.tolist()}) ===", flush=True)
    rloc = (rpos[0].detach().cpu().numpy() - o0).tolist()
    print(f"  robot local: {rloc}", flush=True)
    xs, ys = [], []
    for obj_idx in range(1, 19):
        obj = _get_obj(obj_idx)
        if obj is None:
            continue
        opos = obj.data.root_pos_w[0].detach().cpu().numpy()
        loc = (opos - o0).tolist()
        xs.append(loc[0]); ys.append(loc[1])
        print(f"  object_{obj_idx} local: {loc}", flush=True)
    if xs:
        print(f"  objects x range: [{min(xs):.2f}, {max(xs):.2f}]  y range: [{min(ys):.2f}, {max(ys):.2f}]", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
