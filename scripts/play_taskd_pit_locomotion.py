"""Play / record video for B2Piper Task D pit-crossing locomotion policy."""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ATEC_RL_LAB_SRC = os.path.join(_REPO_ROOT, "source", "atec_rl_lab")
if os.path.isdir(_ATEC_RL_LAB_SRC) and _ATEC_RL_LAB_SRC not in sys.path:
    sys.path.insert(0, _ATEC_RL_LAB_SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Task D pit locomotion checkpoint.")
parser.add_argument("--num_envs", type=int, default=4, help="Parallel envs (use 1-4 for cleaner video).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt checkpoint.")
parser.add_argument("--pit_width_min", type=float, default=0.4)
parser.add_argument("--pit_width_max", type=float, default=1.4)
parser.add_argument("--pit_curriculum_levels", type=int, default=11)
parser.add_argument("--pit_level", type=int, default=0, help="Fixed pit-width row (0=narrowest). Ignored if --pit_curriculum.")
parser.add_argument(
    "--pit_curriculum",
    action="store_true",
    help="Keep pit-width curriculum enabled during play (default: fixed --pit_level).",
)
parser.add_argument("--command_vx", type=float, default=0.6, help="Fixed forward velocity command (m/s).")
parser.add_argument("--video", action="store_true", help="Record mp4 via gym RecordVideo wrapper.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded rollout length in env steps.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Output folder (default: <checkpoint_dir>/videos/play).",
)
parser.add_argument("--real-time", action="store_true", help="Sleep to match sim real-time (GUI mode).")
parser.add_argument(
    "--marg",
    action="store_true",
    help="Load MARG asymmetric AC (estimator + elevation + privileged critic).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import atec_rl_lab.train  # noqa: F401
from atec_rl_lab.tasks.task_d.locomotion.env_cfg import (
    UnitreeB2PiperTaskDPitLocomotionEnvCfg,
    UnitreeB2PiperTaskDPitLocomotionMargEnvCfg,
)
from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import pit_width_from_level
from atec_rl_lab.tasks.task_d.locomotion.terrain_curriculum import TaskDPitTerrainGenerator
from atec_rl_lab.tasks.task_d.terrain import TaskDTerrainImporter, configure_task_d_terrain_for_num_envs
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.agents.rsl_rl_ppo_cfg import (
    UnitreeB2PiperTaskDPitLocomotionMargPPORunnerCfg,
    UnitreeB2PiperTaskDPitLocomotionPPORunnerCfg,
)
from atec_rl_lab.train.locomotion.marg.marg_actor_critic import MargActorCritic
from atec_rl_lab.train.locomotion.marg.marg_ppo import MargPPO

import rsl_rl.runners.on_policy_runner as _runner_mod

_runner_mod.MargActorCritic = MargActorCritic
_runner_mod.MargPPO = MargPPO


def _pit_width_for_level(level: int, width_range: tuple[float, float], curriculum_levels: int) -> float:
    max_level = max(0, int(curriculum_levels) - 1)
    level_t = torch.tensor([max(0, min(int(level), max_level))], dtype=torch.long)
    return float(pit_width_from_level(level_t, width_range, max_level).item())


def _configure_play_terrain(env_cfg: UnitreeB2PiperTaskDPitLocomotionEnvCfg, *, use_pit_curriculum: bool) -> None:
    """Minimal terrain for play: fixed pit width bakes only ~num_envs tiles, not 11×num_envs."""
    num_envs = int(env_cfg.scene.num_envs)
    width_range = tuple(env_cfg.pit_width_range)
    if use_pit_curriculum:
        terrain_cfg = env_cfg._build_terrain_cfg()
        nrow = terrain_cfg.terrain_generator.num_rows
        ncol = terrain_cfg.terrain_generator.num_cols
        print(
            f"[TaskDPitPlay] full curriculum terrain {nrow}x{ncol} ({nrow * ncol} tiles)",
            flush=True,
        )
    else:
        import copy

        from atec_rl_lab.tasks.task_d.terrain import TASK_D_TERRAIN_CFG, PitAndPlatformTerrainCfg

        level = int(args_cli.pit_level)
        fixed_w = _pit_width_for_level(level, width_range, env_cfg.pit_curriculum_levels)
        terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
        terrain_cfg.class_type = TaskDTerrainImporter
        terrain_cfg.terrain_generator.class_type = TaskDPitTerrainGenerator
        configure_task_d_terrain_for_num_envs(terrain_cfg, num_envs, curriculum_levels=None)
        terrain_cfg.terrain_generator.curriculum = False
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = (fixed_w, fixed_w)
            pit_cfg.platform_height_range = env_cfg.platform_height_range
        nrow = terrain_cfg.terrain_generator.num_rows
        ncol = terrain_cfg.terrain_generator.num_cols
        print(
            f"[TaskDPitPlay] minimal terrain {nrow}x{ncol} ({nrow * ncol} tiles), "
            f"fixed pit_width={fixed_w:.3f}m (level={level})",
            flush=True,
        )

    env_cfg.scene.terrain = terrain_cfg
    env_cfg.scene.terrain.max_init_terrain_level = 0


def main():
    device = args_cli.device if args_cli.device else "cuda"
    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    env_cfg_cls = (
        UnitreeB2PiperTaskDPitLocomotionMargEnvCfg
        if args_cli.marg
        else UnitreeB2PiperTaskDPitLocomotionEnvCfg
    )
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.pit_width_range = (float(args_cli.pit_width_min), float(args_cli.pit_width_max))
    env_cfg.pit_curriculum_levels = int(args_cli.pit_curriculum_levels)
    env_cfg.command_lin_vel_x_min = float(args_cli.command_vx)
    env_cfg.command_lin_vel_x_max = float(args_cli.command_vx)
    env_cfg.command_curriculum_start_fraction = 1.0
    _configure_play_terrain(env_cfg, use_pit_curriculum=bool(args_cli.pit_curriculum))

    # Play: no DR, no command/pit curriculum updates.
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.pit_width_levels = None
    if hasattr(env_cfg.observations, "policy") and env_cfg.observations.policy is not None:
        env_cfg.observations.policy.enable_corruption = False

    agent_cfg = (
        UnitreeB2PiperTaskDPitLocomotionMargPPORunnerCfg()
        if args_cli.marg
        else UnitreeB2PiperTaskDPitLocomotionPPORunnerCfg()
    )
    agent_cfg.device = device

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = args_cli.video_dir
        if video_dir is None:
            video_dir = os.path.join(os.path.dirname(ckpt), "videos", "play")
        video_dir = os.path.abspath(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": int(args_cli.video_length),
            "disable_logger": True,
        }
        print("[INFO] Recording video:", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    print(f"[INFO] Loading checkpoint: {ckpt}", flush=True)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    unwrapped = vec_env.unwrapped
    if args_cli.pit_curriculum:
        terrain = unwrapped.scene.terrain
        if hasattr(terrain, "terrain_levels"):
            level = int(args_cli.pit_level)
            max_level = max(0, int(args_cli.pit_curriculum_levels) - 1)
            level = max(0, min(level, max_level))
            terrain.terrain_levels[:] = level
            terrain.update_env_origins_from_levels(terrain.terrain_levels.clone())
            print(f"[INFO] Initial pit level={level} for all envs", flush=True)

    obs = vec_env.get_observations()
    dt = unwrapped.step_dt
    steps = 0
    max_steps = int(args_cli.video_length) if args_cli.video else None

    print(
        f"[INFO] command_vx={args_cli.command_vx} m/s, num_envs={args_cli.num_envs}, "
        f"pit_width={env_cfg.pit_width_range}",
        flush=True,
    )

    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            policy_nn.reset(dones)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            print(f"[INFO] Recorded {steps} steps, stopping.", flush=True)
            break
        if args_cli.real_time:
            sleep_s = dt - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
