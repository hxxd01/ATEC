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
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="InteractiveScene env_spacing (GridCloner). With terrain generator, env_origins still come from terrain tiles.",
)
parser.add_argument(
    "--debug_env_origins",
    action="store_true",
    help="Print scene.env_origins[:16] after env creation (check 1x1 terrain vs multi-tile).",
)
parser.add_argument("--inner_steps", type=int, default=5, help="Low-level sim steps per nav step (5 -> 10Hz nav).")
parser.add_argument("--max_iter", type=int, default=8000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--steps_per_env", type=int, default=24)
parser.add_argument("--vx_min", type=float, default=-2.0)
parser.add_argument("--vx_max", type=float, default=2.0)
parser.add_argument("--camera_hw", type=int, default=64)
parser.add_argument("--depth_max", type=float, default=5.0)
parser.add_argument(
    "--depth_only",
    action="store_true",
    help="Use depth maps only (head+ee). Isaac cameras render depth only (no RGB).",
)
parser.add_argument(
    "--tiled_cameras",
    action="store_true",
    help="Use TiledCameraCfg for head/ee (one tiled render product per camera type; scales to more envs).",
)
parser.add_argument("--ppo_no_train", action="store_true", help="Disable PPO updates; run rollout only.")
parser.add_argument("--no_train_steps", type=int, default=300, help="High-level rollout steps for --ppo_no_train.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video.")
parser.add_argument(
    "--video_length",
    type=int,
    default=300,
    help="Max physics frames for --ppo_no_train --video (caps rollout nav steps).",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Output mp4 path or folder for --ppo_no_train --video (default: <log_dir>/videos/rgb_head_ee.mp4).",
)
parser.add_argument(
    "--depth_video_fps",
    type=float,
    default=0.0,
    help="FPS for --ppo_no_train --video combined mp4 (0 = physics step rate, ~50Hz).",
)
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
from atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvB2Cfg, refresh_task_d_terrain_cfg
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


def _write_depth_video(frames: list, path: str, fps: float) -> None:
    if not frames:
        print(f"[WARN] Video not saved (0 frames): {path}", flush=True)
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, frames, fps=float(fps))
    except Exception as exc:
        print(f"[WARN] imageio video failed ({exc}); trying cv2...", flush=True)
        try:
            import cv2

            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (w, h),
            )
            for fr in frames:
                writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception as exc2:
            print(f"[WARN] video not saved: {exc2}", flush=True)
            return
    print(f"[INFO] Video saved: {path} ({len(frames)} frames @ {fps:.1f} fps)", flush=True)


def _run_no_train_rollout(
    vec_env: NavRslRlVecEnvWrapper,
    steps: int,
    policy=None,
    policy_nn=None,
    *,
    nav_env=None,
    video_env: int = 0,
    early_stop_on_done: bool = True,
) -> None:
    """Step the student env without PPO updates (optionally with policy inference)."""
    env_idx = max(0, min(int(video_env), vec_env.num_envs - 1))
    obs, _ = vec_env.reset()
    max_steps = int(steps)
    for i in range(max_steps):
        if policy is not None:
            # Use no_grad (not inference_mode) so recurrent hidden state can be reset in-place.
            with torch.no_grad():
                actions = policy(obs)
        else:
            actions = torch.zeros((vec_env.num_envs, vec_env.num_actions), device=vec_env.device, dtype=torch.float32)
        obs, rew, dones, _ = vec_env.step(actions)
        if policy_nn is not None and hasattr(policy_nn, "reset"):
            policy_nn.reset(dones)
        done_ratio = dones.float().mean().item()
        if (
            nav_env is not None
            and getattr(nav_env, "_combined_video_enabled", False)
            and getattr(nav_env, "_combined_video_max_frames", None) is not None
            and nav_env.combined_video_frame_count >= nav_env._combined_video_max_frames
        ):
            print(
                f"[NO-TRAIN] stop: recorded {nav_env.combined_video_frame_count} physics frames.",
                flush=True,
            )
            break
        if i % 50 == 0 or i == max_steps - 1 or done_ratio >= 1.0:
            print(
                f"[NO-TRAIN] step={i+1:4d}/{max_steps} mean_rew={rew.float().mean().item():+.4f} "
                f"done_ratio={done_ratio:.2f}",
                flush=True,
            )
        if early_stop_on_done and bool(dones.all().item()):
            print(f"[NO-TRAIN] early stop: all envs done at nav step {i+1}.", flush=True)
            break


def _load_bc_into_actor_critic(actor_critic, ckpt_path: str, *, depth_only: bool = False) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    bc_sd = ckpt.get("model", ckpt)
    model_sd = actor_critic.state_dict()

    prefixes = ("proprio_mlp.", "fuse.")
    if not depth_only:
        prefixes = ("head_encoder.", "ee_encoder.",) + prefixes
    else:
        print(
            "[WARN] depth_only=True: skipping BC head/ee encoder warm-start (in_ch mismatch).",
            flush=True,
        )
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


def _print_env_origins_debug(base_env, *, env_spacing: float, show: int = 16) -> None:
    """Print env_origins sample to verify terrain grid vs 1x1 pit."""
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
    uniq = len({tuple(row) for row in xy.round(3)})
    print(f"[TaskDDebug] unique xy (3dp): {uniq}/{n}", flush=True)
    if uniq <= 1:
        print(
            "[TaskDDebug] WARNING: env_origins identical -> terrain likely 1x1; "
            "env_spacing does not spread pits.",
            flush=True,
        )


def _configure_student_cameras(
    env_cfg,
    camera_hw: int,
    depth_only: bool,
    tiled: bool,
) -> None:
    """Resize cameras; optionally depth-only and/or tiled rendering."""
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    cam_cfg_cls = TiledCameraCfg if tiled else CameraCfg
    data_types = ["depth"] if depth_only else ["rgb", "depth"]
    for cam_name in ("head_camera", "ee_camera"):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is None:
            continue
        setattr(
            env_cfg.scene,
            cam_name,
            cam_cfg_cls(
                prim_path=cam.prim_path,
                spawn=cam.spawn,
                offset=cam.offset,
                height=int(camera_hw),
                width=int(camera_hw),
                data_types=data_types,
                update_period=cam.update_period,
            ),
        )


def main():
    device = args_cli.device if args_cli.device else "cuda"

    env_cfg = TaskDEnvB2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    refresh_task_d_terrain_cfg(env_cfg)
    scene_env_spacing = float(env_cfg.scene.env_spacing)
    # Keep camera sensors but drop image/lidar observation managers (read camera buffers directly).
    if env_cfg.observations is not None:
        env_cfg.observations.image = None
        env_cfg.observations.extero = None
    if getattr(env_cfg.scene, "lidar_sensor", None) is not None:
        env_cfg.scene.lidar_sensor = None
    print("[INFO] Student sim: LiDAR disabled (extero obs + lidar_sensor off).", flush=True)
    _configure_student_cameras(
        env_cfg,
        args_cli.camera_hw,
        args_cli.depth_only,
        args_cli.tiled_cameras,
    )
    if args_cli.tiled_cameras:
        print(
            "[INFO] Student sim: tiled_cameras=True (TiledCameraCfg for head/ee; "
            "one tiled render product per camera type).",
            flush=True,
        )
    if args_cli.depth_only:
        print("[INFO] Student sim: depth_only mode (cameras data_types=['depth']).", flush=True)
    if args_cli.ppo_no_train and not args_cli.video:
        print("[INFO] ppo_no_train: no video (add --video for global|head|ee stitched MP4).", flush=True)

    decimation = int(getattr(env_cfg, "decimation", 4))
    sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
    phys_dt = decimation * sim_dt
    nav_dt = float(args_cli.inner_steps) * phys_dt
    cam_update_period = phys_dt if (args_cli.ppo_no_train and args_cli.video) else nav_dt
    for cam_name in ("head_camera", "ee_camera"):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is not None:
            cam.update_period = cam_update_period
    print(
        f"[INFO] Student sim: nav_dt={nav_dt:.3f}s ({1.0 / nav_dt:.1f}Hz), "
        f"cam_update={cam_update_period:.3f}s, "
        f"camera_hw={args_cli.camera_hw}, depth_only={args_cli.depth_only}, "
        f"tiled_cameras={args_cli.tiled_cameras}, "
        f"num_envs={args_cli.num_envs}, env_spacing={scene_env_spacing}, inner_steps={args_cli.inner_steps}",
        flush=True,
    )

    agent_cfg = TaskDStudentPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iter
    agent_cfg.num_steps_per_env = args_cli.steps_per_env
    agent_cfg.policy.img_hw = args_cli.camera_hw
    agent_cfg.policy.img_channels = 1 if args_cli.depth_only else 4
    if args_cli.depth_only:
        agent_cfg.experiment_name = "taskd_student_b2piper_depth"

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    record_combined_video = bool(args_cli.ppo_no_train and args_cli.video)
    inner_steps = int(args_cli.inner_steps)
    max_phys_frames = int(args_cli.video_length)
    no_train_video_nav_steps = min(
        int(args_cli.no_train_steps),
        max(1, (max_phys_frames + inner_steps - 1) // inner_steps),
    )
    render_mode = "rgb_array" if record_combined_video else None
    env = gym.make("ATEC-TaskD-B2Piper", cfg=env_cfg, render_mode=render_mode)
    if record_combined_video:
        print(
            "[INFO] ppo_no_train + --video: global|head|ee stitched MP4 at physics step rate.",
            flush=True,
        )
    nav_env = TaskDStudentEnv(
        env=env,
        ll_policy_path=args_cli.ll_policy,
        device=device,
        inner_steps=args_cli.inner_steps,
        vx_min=args_cli.vx_min,
        vx_max=args_cli.vx_max,
        image_hw=args_cli.camera_hw,
        depth_render_h=int(args_cli.camera_hw),
        depth_render_w=int(args_cli.camera_hw),
        depth_max=args_cli.depth_max,
        depth_only=args_cli.depth_only,
        nav_log_interval=args_cli.nav_log_interval,
    )
    vec_env = NavRslRlVecEnvWrapper(nav_env)
    if args_cli.debug_env_origins:
        _print_env_origins_debug(env, env_spacing=scene_env_spacing, show=16)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[INFO] Resuming PPO from: {args_cli.resume}", flush=True)
        _load_ppo_checkpoint(runner, args_cli.resume, load_optimizer=True)
    if args_cli.no_train_ckpt:
        print(f"[INFO] Loading no-train inference checkpoint: {args_cli.no_train_ckpt}", flush=True)
        _load_ppo_checkpoint(runner, args_cli.no_train_ckpt, load_optimizer=False)

    if args_cli.bc_ckpt:
        _load_bc_into_actor_critic(
            _get_policy_module(runner.alg),
            args_cli.bc_ckpt,
            depth_only=args_cli.depth_only,
        )

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
        combined_video_path = None
        phys_fps = (
            float(args_cli.depth_video_fps) if args_cli.depth_video_fps > 0 else 1.0 / max(phys_dt, 1.0e-6)
        )
        if record_combined_video:
            rollout_steps = no_train_video_nav_steps
            if args_cli.video_dir:
                combined_video_path = os.path.abspath(args_cli.video_dir)
                if combined_video_path.endswith(os.sep) or os.path.isdir(combined_video_path):
                    os.makedirs(combined_video_path, exist_ok=True)
                    combined_video_path = os.path.join(combined_video_path, "rgb_head_ee.mp4")
            else:
                combined_video_path = os.path.join(log_dir, "videos", "rgb_head_ee.mp4")
            nav_env.enable_combined_video(env_idx=0, max_frames=max_phys_frames)
            print(
                f"[INFO] Combined video: {combined_video_path} "
                f"(<= {max_phys_frames} physics frames @ {phys_fps:.1f} Hz, "
                f"up to {rollout_steps} nav steps x {inner_steps} inner); "
                f"continues after episode done (auto-reset).",
                flush=True,
            )
        _run_no_train_rollout(
            vec_env,
            steps=rollout_steps,
            policy=policy,
            policy_nn=policy_nn,
            nav_env=nav_env,
            video_env=0,
            early_stop_on_done=not record_combined_video,
        )
        if record_combined_video and combined_video_path is not None:
            _write_depth_video(nav_env._combined_video_frames, combined_video_path, phys_fps)
            nav_env.disable_combined_video()
        nav_env.close()
        return

    print("[INFO] Start TaskD student fine-tuning...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    nav_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

