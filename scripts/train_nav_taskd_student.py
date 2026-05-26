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
parser.add_argument(
    "--lidar_bins",
    type=int,
    default=0,
    help="LiDAR bins in actor obs (0=off). With --lidar_bins 36, default pattern is 360 ring "
    "at 36 rays/env (use --lidar_fast/--lidar_coarse to override).",
)
parser.add_argument(
    "--lidar_horizontal_res",
    type=float,
    default=None,
    help="Override LiDAR horizontal resolution (degrees). Larger means fewer rays.",
)
parser.add_argument(
    "--lidar_channels",
    type=int,
    default=None,
    help="Override LiDAR channel count. Smaller means fewer rays.",
)
parser.add_argument(
    "--lidar_update_period",
    type=float,
    default=None,
    help="Override LiDAR sensor update period (seconds). Larger means lower sensor rate.",
)
parser.add_argument(
    "--lidar_fast",
    action="store_true",
    help="Low-ray LiDAR preset: front +/-30 deg, 3 deg res, 2 channels (~40 rays/env).",
)
parser.add_argument(
    "--lidar_coarse",
    action="store_true",
    help="Coarse 360 deg LiDAR: 6 deg res, 4 channels (~240 rays/env vs default ~5760).",
)
parser.add_argument(
    "--lidar_ring",
    action="store_true",
    help="360 ring LiDAR: horizontal_res=360/lidar_bins, 1 channel (rays/env == lidar_bins). "
    "Default when --lidar_bins>0 unless --lidar_fast/--lidar_coarse is set.",
)
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
parser.add_argument(
    "--no_progress_s",
    type=float,
    default=3.0,
    help="no_target_progress_timeout window (seconds). Default 3.0 for student (env_cfg default is 2.0).",
)
parser.add_argument(
    "--no_progress_eps",
    type=float,
    default=0.05,
    help="Min net progress toward stage target (m) within the window to avoid termination.",
)
parser.add_argument(
    "--disable_no_progress",
    action="store_true",
    help="Disable no_target_progress_timeout (not recommended for student RL).",
)
parser.add_argument(
    "--depth_video_fps",
    type=float,
    default=0.0,
    help="FPS for --ppo_no_train --video combined mp4 (0 = physics step rate, ~50Hz).",
)
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


def _write_depth_video(frames: list, path: str, fps: float) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(path, frames, fps=float(fps))
    except Exception as exc:
        print(f"[WARN] imageio depth video failed ({exc}); trying cv2...", flush=True)
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
            print(f"[WARN] depth video not saved: {exc2}", flush=True)
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
) -> None:
    """Step the student env without PPO updates (optionally with policy inference)."""
    num_envs = vec_env.num_envs
    env_idx = max(0, min(int(video_env), num_envs - 1))
    max_stage_1idx = torch.ones(num_envs, device=vec_env.device, dtype=torch.long)
    ep_count = 0

    obs, _ = vec_env.reset()
    if nav_env is not None and getattr(nav_env, "_stage_idx_buf", None) is not None:
        max_stage_1idx = nav_env._stage_idx_buf.to(dtype=torch.long) + 1
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
        if nav_env is not None and getattr(nav_env, "_stage_idx_buf", None) is not None:
            cur_stage_1idx = nav_env._stage_idx_buf.to(dtype=torch.long) + 1
            max_stage_1idx = torch.maximum(max_stage_1idx, cur_stage_1idx)
            if bool(dones.any()):
                done_ids = dones.nonzero(as_tuple=False).view(-1)
                for ei in done_ids.tolist():
                    si = int(nav_env._stage_idx_buf[ei].item())
                    sn = nav_env._stage_names[si] if 0 <= si < nav_env._num_stages else "done"
                    ep_count += 1
                    print(
                        f"[NO-TRAIN] episode_done #{ep_count} env{ei} "
                        f"end_stage={si + 1}({sn}) max_stage={int(max_stage_1idx[ei].item())} "
                        f"rew_step={rew[ei].item():+.3f}",
                        flush=True,
                    )
                max_stage_1idx[done_ids] = nav_env._stage_idx_buf[done_ids].to(dtype=torch.long) + 1
        step_no = i + 1
        if (
            nav_env is not None
            and getattr(nav_env, "_combined_video_enabled", False)
            and getattr(nav_env, "_combined_video_max_frames", None) is not None
            and nav_env.combined_video_frame_count >= nav_env._combined_video_max_frames
        ):
            break
        if i % 50 == 0 or i == steps - 1:
            stage_hint = ""
            mix_hint = ""
            if nav_env is not None and getattr(nav_env, "_stage_idx_buf", None) is not None:
                si = int(nav_env._stage_idx_buf[env_idx].item())
                sn = nav_env._stage_names[si] if 0 <= si < nav_env._num_stages else "?"
                prog = float(nav_env._stage_progress_buf[env_idx].item())
                stage_hint = f" env{env_idx}_stage={si + 1}({sn}) prog={prog:.2f} max={int(max_stage_1idx[env_idx].item())}"
                if hasattr(nav_env, "format_stage_distribution_line"):
                    mix_hint = " " + nav_env.format_stage_distribution_line()
            print(
                f"[NO-TRAIN] step={step_no:4d}/{steps} mean_rew={rew.float().mean().item():+.4f} "
                f"done_ratio={dones.float().mean().item():.2f}{stage_hint}{mix_hint}",
                flush=True,
            )


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


def _estimate_lidar_rays(pattern_cfg) -> int:
    """Approximate rays/sensor for LidarPatternCfg (degrees-based)."""
    from isaaclab.sensors import patterns

    if not isinstance(pattern_cfg, patterns.LidarPatternCfg):
        return -1
    h_min, h_max = pattern_cfg.horizontal_fov_range
    hres = max(float(pattern_cfg.horizontal_res), 1.0e-6)
    horiz = max(1, int(round((float(h_max) - float(h_min)) / hres)))
    return horiz * int(pattern_cfg.channels)


def _apply_ring_lidar_pattern(lidar_sensor, lidar_bins: int) -> None:
    """360 deg ring with exactly one ray per policy bin (fastest full-ring mode)."""
    from isaaclab.sensors import patterns

    bins = max(1, int(lidar_bins))
    horizontal_res = 360.0 / float(bins)
    lidar_sensor.pattern_cfg = patterns.LidarPatternCfg(
        vertical_fov_range=(-10.0, 10.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=horizontal_res,
        channels=1,
    )


def _apply_student_lidar_cfg(env_cfg, args_cli, nav_dt: float) -> None:
    from isaaclab.sensors import patterns

    lidar_sensor = getattr(env_cfg.scene, "lidar_sensor", None)
    if lidar_sensor is None:
        raise ValueError("--lidar_bins > 0 but env_cfg.scene.lidar_sensor is missing.")

    preset_flags = (args_cli.lidar_fast, args_cli.lidar_coarse, args_cli.lidar_ring)
    if sum(int(x) for x in preset_flags) > 1:
        raise ValueError("Use only one of --lidar_fast, --lidar_coarse, or --lidar_ring.")

    lidar_mode = "ring360"
    if args_cli.lidar_fast:
        lidar_mode = "front_cone"
        lidar_sensor.pattern_cfg = patterns.LidarPatternCfg(
            vertical_fov_range=(-10.0, 10.0),
            horizontal_fov_range=(-30.0, 30.0),
            horizontal_res=3.0,
            channels=2,
        )
    elif args_cli.lidar_coarse:
        lidar_mode = "coarse360"
        lidar_sensor.pattern_cfg = patterns.LidarPatternCfg(
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=6.0,
            channels=4,
        )
    else:
        _apply_ring_lidar_pattern(lidar_sensor, args_cli.lidar_bins)

    if args_cli.lidar_horizontal_res is not None:
        lidar_sensor.pattern_cfg.horizontal_res = float(args_cli.lidar_horizontal_res)
    if args_cli.lidar_channels is not None:
        lidar_sensor.pattern_cfg.channels = int(args_cli.lidar_channels)
    if args_cli.lidar_update_period is not None:
        lidar_sensor.update_period = float(args_cli.lidar_update_period)
    elif lidar_sensor.update_period < nav_dt:
        lidar_sensor.update_period = nav_dt

    rays = _estimate_lidar_rays(lidar_sensor.pattern_cfg)
    pat = lidar_sensor.pattern_cfg
    align = "1:1 with lidar_bins" if rays == int(args_cli.lidar_bins) else "pooled to lidar_bins"
    print(
        "[INFO] Student sim: LiDAR enabled "
        f"(mode={lidar_mode}, lidar_bins={args_cli.lidar_bins}, hfov={pat.horizontal_fov_range}, "
        f"horizontal_res={pat.horizontal_res}, channels={pat.channels}, "
        f"update_period={lidar_sensor.update_period}, ~rays/env={rays}, {align}, "
        f"~total_rays={rays * args_cli.num_envs if rays > 0 else 'n/a'}).",
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
    if args_cli.disable_no_progress:
        env_cfg.terminations.no_target_progress_timeout = None
        print("[INFO] Student sim: no_target_progress_timeout disabled.", flush=True)
    else:
        np_term = env_cfg.terminations.no_target_progress_timeout
        np_term.params["stuck_time_s"] = float(args_cli.no_progress_s)
        np_term.params["progress_eps"] = float(args_cli.no_progress_eps)
        decimation = int(getattr(env_cfg, "decimation", 4))
        sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
        phys_dt = decimation * sim_dt
        window_steps = max(1, int(float(args_cli.no_progress_s) / phys_dt))
        print(
            f"[INFO] Student sim: no_target_progress_timeout enabled "
            f"(stuck_time_s={args_cli.no_progress_s}, progress_eps={args_cli.no_progress_eps}, "
            f"~{window_steps} physics steps / {window_steps / max(1, args_cli.inner_steps):.0f} nav steps).",
            flush=True,
        )
    fall_min_h = env_cfg.terminations.fall.params.get("minimum_height", "?")
    print(
        f"[INFO] Student sim: active terminations: fall(min_h={fall_min_h}), time_out, "
        f"no_target_progress={not args_cli.disable_no_progress} "
        f"(x_reached/no_motion/stage_target_deviation remain off in TaskDEnvCfg).",
        flush=True,
    )
    # Keep camera sensors but drop image observation managers (read camera buffers directly).
    if env_cfg.observations is not None:
        env_cfg.observations.image = None
    if int(args_cli.lidar_bins) > 0:
        decimation = int(getattr(env_cfg, "decimation", 4))
        sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
        nav_dt = float(args_cli.inner_steps) * decimation * sim_dt
        _apply_student_lidar_cfg(env_cfg, args_cli, nav_dt)
    else:
        if env_cfg.observations is not None:
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
        print("[INFO] ppo_no_train: no video output (add --video for RGB|head|ee stitched MP4).", flush=True)

    # Training: render cameras once per nav step; combined debug video uses physics rate.
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
        f"lidar_bins={args_cli.lidar_bins}, tiled_cameras={args_cli.tiled_cameras}, "
        f"num_envs={args_cli.num_envs}, inner_steps={args_cli.inner_steps}",
        flush=True,
    )

    agent_cfg = TaskDStudentPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iter
    agent_cfg.num_steps_per_env = args_cli.steps_per_env
    agent_cfg.policy.img_hw = args_cli.camera_hw
    agent_cfg.policy.img_channels = 1 if args_cli.depth_only else 4
    agent_cfg.policy.proprio_dim = 12
    agent_cfg.policy.lidar_bins = int(args_cli.lidar_bins)
    if args_cli.depth_only:
        agent_cfg.experiment_name = "taskd_student_b2piper_depth"
    if int(args_cli.lidar_bins) > 0:
        agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_lidar{int(args_cli.lidar_bins)}"

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    record_combined_video = bool(args_cli.ppo_no_train and args_cli.video)
    record_rgb_video = bool(args_cli.video and not args_cli.ppo_no_train)
    inner_steps = int(args_cli.inner_steps)
    max_phys_frames = int(args_cli.video_length)
    no_train_video_nav_steps = min(
        int(args_cli.no_train_steps),
        max(1, (max_phys_frames + inner_steps - 1) // inner_steps),
    )
    render_mode = "rgb_array" if (record_rgb_video or record_combined_video) else None
    env = gym.make("ATEC-TaskD-B2Piper", cfg=env_cfg, render_mode=render_mode)
    if record_rgb_video:
        video_dir = os.path.join(log_dir, "videos", "student")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=int(args_cli.video_length),
            disable_logger=True,
        )
        print(
            f"[INFO] Recording RGB rollout video (RecordVideo) to: {video_dir} "
            f"({int(args_cli.video_length)} physics steps)",
            flush=True,
        )
    if record_combined_video:
        print(
            "[INFO] ppo_no_train + --video: RGB|head_depth|ee_depth stitched MP4 at physics step rate.",
            flush=True,
        )
    nav_log_interval = 0 if args_cli.ppo_no_train else args_cli.nav_log_interval
    nav_env = TaskDStudentEnv(
        env=env,
        ll_policy_path=args_cli.ll_policy,
        device=device,
        inner_steps=args_cli.inner_steps,
        vx_min=args_cli.vx_min,
        vx_max=args_cli.vx_max,
        image_hw=args_cli.camera_hw,
        depth_max=args_cli.depth_max,
        depth_only=args_cli.depth_only,
        lidar_bins=int(args_cli.lidar_bins),
        nav_log_interval=nav_log_interval,
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
        phys_fps = float(args_cli.depth_video_fps) if args_cli.depth_video_fps > 0 else 1.0 / max(phys_dt, 1.0e-6)
        if record_combined_video:
            rollout_steps = no_train_video_nav_steps
            combined_video_path = os.path.join(log_dir, "videos", "rgb_head_ee.mp4")
            nav_env.enable_combined_video(env_idx=0, max_frames=max_phys_frames)
            print(
                f"[INFO] Combined video: {combined_video_path} "
                f"(<= {max_phys_frames} physics frames @ {phys_fps:.1f} Hz, "
                f"{rollout_steps} nav steps x {inner_steps} inner)",
                flush=True,
            )
        _run_no_train_rollout(
            vec_env,
            steps=rollout_steps,
            policy=policy,
            policy_nn=policy_nn,
            nav_env=nav_env,
            video_env=0,
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

