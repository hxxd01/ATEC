"""Train Task-D student nav policy and optionally warm-start from BC checkpoint."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Task D student policy (BC warm-start + PPO).")
parser.add_argument("--ll_policy", type=str, required=True, help="Low-level locomotion policy (.pt)")
parser.add_argument("--bc_ckpt", type=str, default=None, help="BC checkpoint path (best.pt/last.pt)")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--inner_steps", type=int, default=25)
parser.add_argument("--max_iter", type=int, default=8000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--steps_per_env", type=int, default=24)
parser.add_argument("--vx_min", type=float, default=-2.0)
parser.add_argument("--vx_max", type=float, default=2.0)
parser.add_argument("--camera_hw", type=int, default=64)
parser.add_argument("--depth_max", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Student policy always reads head/ee camera buffers.
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

import atec_rl_lab.tasks  # noqa: F401
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg
from atec_rl_lab.train.nav.taskd_student_env import TaskDStudentEnv
from atec_rl_lab.train.nav.nav_cfg import TaskDStudentPPORunnerCfg
from atec_rl_lab.train.nav.nav_rsl_wrapper import NavRslRlVecEnvWrapper
from atec_rl_lab.train.nav.taskd_student_actor_critic import TaskDStudentActorCritic

import rsl_rl.runners.on_policy_runner as _runner_mod

_runner_mod.TaskDStudentActorCritic = TaskDStudentActorCritic


def _load_bc_into_actor_critic(actor_critic, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    bc_sd = ckpt.get("model", ckpt)
    model_sd = actor_critic.state_dict()

    prefixes = ("head_encoder.", "ee_encoder.", "proprio_mlp.", "fuse.")
    loaded = 0
    skipped = []
    for k, v in bc_sd.items():
        if not k.startswith(prefixes):
            continue
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            model_sd[k] = v
            loaded += 1
        else:
            skipped.append(k)

    # GRU warm-start: BC gru -> actor memory GRU
    gru_map = {
        "gru.weight_ih_l0": "memory_a.rnn.weight_ih_l0",
        "gru.weight_hh_l0": "memory_a.rnn.weight_hh_l0",
        "gru.bias_ih_l0": "memory_a.rnn.bias_ih_l0",
        "gru.bias_hh_l0": "memory_a.rnn.bias_hh_l0",
    }
    for src, dst in gru_map.items():
        if src in bc_sd and dst in model_sd and tuple(model_sd[dst].shape) == tuple(bc_sd[src].shape):
            model_sd[dst] = bc_sd[src]
            loaded += 1
        else:
            skipped.append(src)

    # Initialize critic memory from actor memory.
    cc_map = {
        "memory_c.rnn.weight_ih_l0": "memory_a.rnn.weight_ih_l0",
        "memory_c.rnn.weight_hh_l0": "memory_a.rnn.weight_hh_l0",
        "memory_c.rnn.bias_ih_l0": "memory_a.rnn.bias_ih_l0",
        "memory_c.rnn.bias_hh_l0": "memory_a.rnn.bias_hh_l0",
    }
    for dst, src in cc_map.items():
        if dst in model_sd and src in model_sd and tuple(model_sd[dst].shape) == tuple(model_sd[src].shape):
            model_sd[dst] = model_sd[src].clone()

    # Actor head warm-start: BC head -> PPO actor sequential.
    head_map = {
        "head.0.weight": "actor.0.weight",
        "head.0.bias": "actor.0.bias",
        "head.2.weight": "actor.2.weight",
        "head.2.bias": "actor.2.bias",
    }
    for src, dst in head_map.items():
        if src in bc_sd and dst in model_sd and tuple(model_sd[dst].shape) == tuple(bc_sd[src].shape):
            model_sd[dst] = bc_sd[src]
            loaded += 1
        else:
            skipped.append(src)

    actor_critic.load_state_dict(model_sd, strict=False)
    print(
        f"[INFO] BC warm-start loaded {loaded} encoder tensors from {ckpt_path}. "
        f"Skipped={len(skipped)}",
        flush=True,
    )
    if skipped:
        print(f"[INFO] Example skipped keys: {skipped[:5]}", flush=True)


def main():
    device = args_cli.device if args_cli.device else "cuda"

    env_cfg = TaskDEnvB2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # Keep camera sensors but drop image observation manager (we read sensor buffers directly).
    if env_cfg.observations is not None:
        env_cfg.observations.image = None
    if getattr(env_cfg.scene, "head_camera", None) is not None:
        env_cfg.scene.head_camera.height = args_cli.camera_hw
        env_cfg.scene.head_camera.width = args_cli.camera_hw
    if getattr(env_cfg.scene, "ee_camera", None) is not None:
        env_cfg.scene.ee_camera.height = args_cli.camera_hw
        env_cfg.scene.ee_camera.width = args_cli.camera_hw

    agent_cfg = TaskDStudentPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iter
    agent_cfg.num_steps_per_env = args_cli.steps_per_env
    agent_cfg.policy.img_hw = args_cli.camera_hw

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env = gym.make("ATEC-TaskD-B2Piper", cfg=env_cfg, render_mode=None)
    nav_env = TaskDStudentEnv(
        env=env,
        ll_policy_path=args_cli.ll_policy,
        device=device,
        inner_steps=args_cli.inner_steps,
        vx_min=args_cli.vx_min,
        vx_max=args_cli.vx_max,
        image_hw=args_cli.camera_hw,
        depth_max=args_cli.depth_max,
    )
    vec_env = NavRslRlVecEnvWrapper(nav_env)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[INFO] Resuming PPO from: {args_cli.resume}", flush=True)
        runner.load(args_cli.resume)

    if args_cli.bc_ckpt:
        _load_bc_into_actor_critic(runner.alg.actor_critic, args_cli.bc_ckpt)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[INFO] Logging to: {log_dir}", flush=True)
    print("[INFO] Start TaskD student fine-tuning...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    nav_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

