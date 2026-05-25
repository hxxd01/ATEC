"""Train Task-D student nav policy and optionally warm-start from BC checkpoint."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Task D student policy (BC warm-start + PPO).")
parser.add_argument("--ll_policy", type=str, required=True, help="Low-level locomotion policy (.pt)")
parser.add_argument("--bc_ckpt", type=str, default=None, help="BC checkpoint path (best.pt/last.pt)")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--inner_steps", type=int, default=5, help="Low-level sim steps per nav step (5 -> 10Hz nav).")
parser.add_argument("--max_iter", type=int, default=8000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--steps_per_env", type=int, default=24)
parser.add_argument("--vx_min", type=float, default=-2.0)
parser.add_argument("--vx_max", type=float, default=2.0)
parser.add_argument("--camera_hw", type=int, default=64)
parser.add_argument("--depth_max", type=float, default=5.0)
parser.add_argument("--ppo_no_train", action="store_true", help="Disable PPO updates; run rollout only.")
parser.add_argument("--no_train_steps", type=int, default=300, help="High-level rollout steps for --ppo_no_train.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video.")
parser.add_argument("--video_length", type=int, default=300, help="Recorded video length in env steps.")
parser.add_argument(
    "--no_train_ckpt",
    type=str,
    default=None,
    help="Checkpoint used only for --ppo_no_train rollout inference.",
)
parser.add_argument("--nav_log_interval", type=int, default=50, help="Print [TaskDTeacher] nav log every N nav steps (0=off).")
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


def _get_policy_module(alg):
    """Return the policy module across rsl_rl versions (`policy` vs legacy `actor_critic`)."""
    if hasattr(alg, "policy"):
        return alg.policy
    if hasattr(alg, "actor_critic"):
        return alg.actor_critic
    raise AttributeError("PPO algorithm has neither `policy` nor `actor_critic`.")


def _merge_state_dict_partial(model_sd: dict, ckpt_sd: dict) -> tuple[dict, list[str], list[str], list[str]]:
    """Load checkpoint weights; tolerate critic priv dim changes from stage-count edits."""
    merged = {k: v.clone() for k, v in model_sd.items()}
    loaded_keys: list[str] = []
    partial_keys: list[str] = []
    skipped_keys: list[str] = []

    for k, v in ckpt_sd.items():
        if k not in merged:
            skipped_keys.append(k)
            continue
        cur = merged[k]
        if tuple(cur.shape) == tuple(v.shape):
            merged[k] = v
            loaded_keys.append(k)
            continue
        if k == "critic_priv_mlp.0.weight" and cur.ndim == 2 and v.ndim == 2:
            # Priv obs layout: 16 fixed dims + stage onehot + stage progress.
            n_shared = 16
            n_copy = min(n_shared, cur.shape[1], v.shape[1])
            merged[k][:, :n_copy] = v[:, :n_copy]
            old_prog_col = v.shape[1] - 1
            new_prog_col = cur.shape[1] - 1
            if old_prog_col >= n_shared and new_prog_col >= n_shared:
                merged[k][:, new_prog_col] = v[:, old_prog_col]
            partial_keys.append(k)
            continue
        skipped_keys.append(k)

    return merged, loaded_keys, partial_keys, skipped_keys


def _load_ppo_checkpoint(runner, ckpt_path: str, *, load_optimizer: bool = True) -> None:
    loaded = torch.load(ckpt_path, map_location=runner.device, weights_only=False)
    ckpt_sd = loaded["model_state_dict"]
    policy = _get_policy_module(runner.alg)
    model_sd = policy.state_dict()
    merged, loaded_keys, partial_keys, skipped_keys = _merge_state_dict_partial(model_sd, ckpt_sd)
    policy.load_state_dict(merged, strict=False)

    can_load_optimizer = len(skipped_keys) == 0
    if load_optimizer and can_load_optimizer and "optimizer_state_dict" in loaded:
        runner.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
    elif load_optimizer and not can_load_optimizer:
        print(
            "[WARN] Optimizer state skipped because checkpoint architecture differs "
            f"({len(skipped_keys)} tensors not loaded).",
            flush=True,
        )

    if can_load_optimizer and "iter" in loaded:
        runner.current_learning_iteration = loaded["iter"]
    elif "iter" in loaded:
        print(
            f"[WARN] Checkpoint iter={loaded['iter']} ignored due to partial load; restarting from iter 0.",
            flush=True,
        )

    print(
        f"[INFO] Checkpoint merge: exact={len(loaded_keys)}, partial={len(partial_keys)}, "
        f"skipped={len(skipped_keys)}",
        flush=True,
    )
    if partial_keys:
        print(f"[INFO] Partially loaded keys: {partial_keys}", flush=True)
    if skipped_keys:
        print(f"[INFO] Skipped keys (shape mismatch): {skipped_keys[:8]}", flush=True)


def _run_no_train_rollout(vec_env: NavRslRlVecEnvWrapper, steps: int, policy=None, policy_nn=None) -> None:
    """Step the student env without PPO updates (optionally with policy inference)."""
    obs, _ = vec_env.reset()
    for i in range(int(steps)):
        if policy is not None:
            # Use no_grad (not inference_mode) so recurrent hidden state can be reset in-place.
            with torch.no_grad():
                actions = policy(obs)
        else:
            actions = torch.zeros((vec_env.num_envs, vec_env.num_actions), device=vec_env.device, dtype=torch.float32)
        obs, rew, dones, _ = vec_env.step(actions)
        if policy_nn is not None and hasattr(policy_nn, "reset"):
            policy_nn.reset(dones)
        if i % 50 == 0 or i == steps - 1:
            print(
                f"[NO-TRAIN] step={i+1:4d}/{steps} mean_rew={rew.float().mean().item():+.4f} "
                f"done_ratio={dones.float().mean().item():.2f}",
                flush=True,
            )


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
    # Keep camera sensors but drop image/lidar observation managers (read camera buffers directly).
    if env_cfg.observations is not None:
        env_cfg.observations.image = None
        env_cfg.observations.extero = None
    if getattr(env_cfg.scene, "lidar_sensor", None) is not None:
        env_cfg.scene.lidar_sensor = None
    print("[INFO] Student sim: LiDAR disabled (extero obs + lidar_sensor off).", flush=True)
    if getattr(env_cfg.scene, "head_camera", None) is not None:
        env_cfg.scene.head_camera.height = args_cli.camera_hw
        env_cfg.scene.head_camera.width = args_cli.camera_hw
    if getattr(env_cfg.scene, "ee_camera", None) is not None:
        env_cfg.scene.ee_camera.height = args_cli.camera_hw
        env_cfg.scene.ee_camera.width = args_cli.camera_hw

    # Render cameras once per nav step (not every physics substep).
    decimation = int(getattr(env_cfg, "decimation", 4))
    sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
    nav_dt = float(args_cli.inner_steps) * decimation * sim_dt
    for cam_name in ("head_camera", "ee_camera"):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is not None:
            cam.update_period = nav_dt
    print(
        f"[INFO] Student sim: nav_dt={nav_dt:.3f}s ({1.0 / nav_dt:.1f}Hz), "
        f"camera_hw={args_cli.camera_hw}, num_envs={args_cli.num_envs}, inner_steps={args_cli.inner_steps}",
        flush=True,
    )

    agent_cfg = TaskDStudentPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iter
    agent_cfg.num_steps_per_env = args_cli.steps_per_env
    agent_cfg.policy.img_hw = args_cli.camera_hw

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("ATEC-TaskD-B2Piper", cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "student")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=int(args_cli.video_length),
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_dir}", flush=True)
    nav_env = TaskDStudentEnv(
        env=env,
        ll_policy_path=args_cli.ll_policy,
        device=device,
        inner_steps=args_cli.inner_steps,
        vx_min=args_cli.vx_min,
        vx_max=args_cli.vx_max,
        image_hw=args_cli.camera_hw,
        depth_max=args_cli.depth_max,
        nav_log_interval=args_cli.nav_log_interval,
    )
    vec_env = NavRslRlVecEnvWrapper(nav_env)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[INFO] Resuming PPO from: {args_cli.resume}", flush=True)
        _load_ppo_checkpoint(runner, args_cli.resume, load_optimizer=True)
    if args_cli.no_train_ckpt:
        print(f"[INFO] Loading no-train inference checkpoint: {args_cli.no_train_ckpt}", flush=True)
        _load_ppo_checkpoint(runner, args_cli.no_train_ckpt, load_optimizer=False)

    if args_cli.bc_ckpt:
        _load_bc_into_actor_critic(_get_policy_module(runner.alg), args_cli.bc_ckpt)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[INFO] Logging to: {log_dir}", flush=True)
    if args_cli.ppo_no_train:
        print("[INFO] PPO no-train mode enabled. Running rollout only...", flush=True)
        policy = runner.get_inference_policy(device=vec_env.device)
        try:
            policy_nn = _get_policy_module(runner.alg)
        except AttributeError:
            policy_nn = None
        rollout_steps = int(args_cli.no_train_steps)
        if args_cli.video:
            rollout_steps = min(rollout_steps, int(args_cli.video_length))
            print(
                f"[INFO] Video mode: limiting no-train rollout to {rollout_steps} steps.",
                flush=True,
            )
        _run_no_train_rollout(
            vec_env,
            steps=rollout_steps,
            policy=policy,
            policy_nn=policy_nn,
        )
        nav_env.close()
        return

    print("[INFO] Start TaskD student fine-tuning...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    nav_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

