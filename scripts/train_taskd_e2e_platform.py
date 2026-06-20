"""Train Task D end-to-end policy (MARG proprio, 4-stage rewards, direct leg actions).

Pit env uses ``build_train_env_cfg`` from ``train_taskd_pit_locomotion.py --marg`` verbatim,
then applies e2e stage overrides. Pass the same pit CLI flags as pit locomotion training.
"""

import argparse
import os
import sys
from datetime import datetime

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ATEC_RL_LAB_SRC = os.path.join(_REPO_ROOT, "source", "atec_rl_lab")
if os.path.isdir(_ATEC_RL_LAB_SRC) and _ATEC_RL_LAB_SRC not in sys.path:
    sys.path.insert(0, _ATEC_RL_LAB_SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Train Task D e2e (pit MARG env from train_taskd_pit_locomotion --marg + 4-stage)."
)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--max_iterations", type=int, default=8000)
parser.add_argument("--steps_per_env", type=int, default=24)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--stage_reach_tol", type=float, default=0.1)
parser.add_argument("--return_still_s", type=float, default=1.0)
parser.add_argument("--stage_log_interval", type=int, default=100)
parser.add_argument(
    "--e2e_reward_mode",
    type=str,
    default="base_only",
    choices=["base_only", "stage_only", "base_plus_stage"],
    help="Reward used for PPO update in E2E wrapper.",
)
parser.add_argument("--pit_width_min", type=float, default=0.4)
parser.add_argument("--pit_width_max", type=float, default=1.4)
parser.add_argument("--pit_curriculum_levels", type=int, default=11)
parser.add_argument("--command_vx_max", type=float, default=4.0)
parser.add_argument("--command_vx", type=float, default=None)
parser.add_argument("--command_vx_min", type=float, default=0.0)
parser.add_argument("--command_curriculum_start", type=float, default=0.1)
parser.add_argument(
    "--sim_easy",
    action="store_true",
    help="Same as train_taskd_pit_locomotion --sim_easy (fixed vx, relaxed penalties).",
)
parser.add_argument("--sim_easy_vx", type=float, default=1.0)
parser.add_argument("--pit_dr_light", action="store_true")
parser.add_argument("--pit_spawn_x_offset", type=float, default=0.0)
parser.add_argument("--pit_spawn_local_x", type=float, default=None)
parser.add_argument("--pit_spawn_x_jitter", type=float, default=None)
parser.add_argument("--no_pit_box", action="store_true")
parser.add_argument("--pit_box_default_spawn", action="store_true")
parser.add_argument("--pit_box_x_jitter_down", type=float, default=None)
parser.add_argument("--pit_box_y_jitter", type=float, default=None)
parser.add_argument(
    "--platform",
    action="store_true",
    help="Use full Task D platform terrain (slow; default is pit MARG from train_taskd_pit_locomotion).",
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument(
    "--video_interval",
    type=int,
    default=0,
    help="Record every N env steps (0 = once at step 0 only).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from rsl_rl.runners import OnPolicyRunner

import atec_rl_lab.tasks  # noqa: F401
import rsl_rl.runners.on_policy_runner as _runner_mod
from atec_rl_lab.tasks.task_d.e2e.env_cfg import (
    TaskDE2EPlatformEnvCfg,
    build_task_d_e2e_pit_env_cfg,
    refresh_task_d_e2e_terrain_cfg,
)
from atec_rl_lab.train.locomotion.marg.marg_ppo import MargPPO
from atec_rl_lab.train.locomotion.marg.marg_proprio_actor_critic import MargProprioActorCritic
from atec_rl_lab.train.nav.nav_cfg import TaskDE2EPlatformMargPPORunnerCfg
from atec_rl_lab.train.nav.taskd_e2e_stage_env import TaskDE2EStageEnv, TaskDE2ERslVecEnvWrapper


def register_marg_proprio_modules() -> None:
    _runner_mod.MargProprioActorCritic = MargProprioActorCritic
    _runner_mod.MargPPO = MargPPO


def main():
    device = args_cli.device if args_cli.device else "cuda"
    register_marg_proprio_modules()

    if args_cli.platform:
        env_cfg = TaskDE2EPlatformEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        refresh_task_d_e2e_terrain_cfg(env_cfg)
        gym_id = "ATEC-TaskD-E2E-B2Piper-v0"
        env_tag = "platform"
    else:
        args_cli.pit_sim_easy = args_cli.sim_easy
        args_cli.pit_sim_easy_vx = args_cli.sim_easy_vx
        env_cfg = build_task_d_e2e_pit_env_cfg(args_cli)
        gym_id = "ATEC-TaskD-E2E-Pit-B2Piper-v0"
        env_tag = "pit"

    if args_cli.video:
        # Keep recorded view anchored on env0 so multi-env runs don't drift to empty sky.
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (6.0, -6.0, 4.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.8)

    agent_cfg = TaskDE2EPlatformMargPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.num_steps_per_env = args_cli.steps_per_env
    agent_cfg.device = device
    if env_tag == "pit":
        agent_cfg.experiment_name = "taskd_e2e_pit_b2piper"

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    render_mode = "rgb_array" if args_cli.video else None
    print(
        f"[INFO] gym={gym_id} env={env_tag} num_envs={env_cfg.scene.num_envs} "
        f"pit_width={getattr(env_cfg, 'pit_width_range', None)} "
        f"levels={getattr(env_cfg, 'pit_curriculum_levels', None)} "
        f"(base=train_taskd_pit_locomotion --marg)",
        flush=True,
    )
    env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "train")
        os.makedirs(video_dir, exist_ok=True)
        interval = int(args_cli.video_interval)
        if interval > 0:
            step_trigger = lambda step, n=interval: step % n == 0
        else:
            step_trigger = lambda step: step == 0
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=step_trigger,
            video_length=int(args_cli.video_length),
            disable_logger=True,
        )
        print(
            f"[INFO] Recording video to: {video_dir} "
            f"(length={int(args_cli.video_length)} steps, interval={interval or 'once'})",
            flush=True,
        )

    stage_env = TaskDE2EStageEnv(
        env,
        device=device,
        stage_reach_tol=float(args_cli.stage_reach_tol),
        return_still_s=float(args_cli.return_still_s),
        stage_log_interval=int(args_cli.stage_log_interval),
        reward_mode=str(args_cli.e2e_reward_mode),
    )
    vec_env = TaskDE2ERslVecEnvWrapper(stage_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[INFO] Resuming from {args_cli.resume}", flush=True)
        runner.load(args_cli.resume)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"[INFO] Logging to {log_dir}", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
