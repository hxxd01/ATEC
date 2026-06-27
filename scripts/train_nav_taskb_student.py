"""Train Task-B student nav policy (Task-D student network + Task-B touch rewards)."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Task B student policy (PPO, Task-D student network).")
parser.add_argument("--task", type=str, default="ATEC-TaskB-B2Piper")
parser.add_argument("--ll_policy", type=str, required=True, help="Low-level locomotion policy JIT (.pt)")
parser.add_argument("--bc_ckpt", type=str, default=None, help="Optional BC checkpoint (best.pt/last.pt)")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="InteractiveScene env_spacing (terrain tiles still use Task-B cell gap).",
)
parser.add_argument("--inner_steps", type=int, default=5, help="Low-level sim steps per nav step.")
parser.add_argument("--max_iter", type=int, default=8000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--steps_per_env", type=int, default=24)
parser.add_argument("--vx_min", type=float, default=-1.0)
parser.add_argument("--vx_max", type=float, default=1.0)
parser.add_argument("--vy_max", type=float, default=1.0)
parser.add_argument("--wz_max", type=float, default=1.0)
parser.add_argument("--policy_img_h", type=int, default=24)
parser.add_argument("--policy_img_w", type=int, default=32)
parser.add_argument("--camera_hw", type=int, default=None, help="Legacy square policy input (overrides h/w).")
parser.add_argument(
    "--sim_camera_h",
    type=int,
    default=24,
    help="Native sim head/ee camera height.",
)
parser.add_argument(
    "--sim_camera_w",
    type=int,
    default=32,
    help="Native sim head/ee camera width.",
)
parser.add_argument("--depth_max", type=float, default=5.0)
parser.add_argument(
    "--depth_only",
    action="store_true",
    help="Depth-only (1ch/cam). Default without this flag: head+ee dual cam, rgb+depth (4ch/cam).",
)
parser.add_argument("--tiled_cameras", action="store_true", help="Use TiledCameraCfg for head/ee.")
parser.add_argument("--camera_far_clip", type=float, default=50.0)
parser.add_argument("--nav_log_interval", type=int, default=10)
parser.add_argument(
    "--w_dense_dist",
    type=float,
    default=0.1,
    help="Guide progress scale: w * max(0, prev_min_3d - min_3d_unscored).",
)
parser.add_argument(
    "--sparse_only",
    action="store_true",
    help="Disable dense shaping (force w_dense_dist=0, skip visibility/depth dense logic).",
)
parser.add_argument(
    "--sparse_touch_reward",
    type=float,
    default=10.0,
    help="Sparse reward per newly scored object (EE-object 3D dist <= grasp threshold).",
)
parser.add_argument("--grasp_dist_thresh", type=float, default=0.20, help="Platform touch score distance (m).")
parser.add_argument(
    "--time_penalty_per_env_step",
    type=float,
    default=0.0,
    help="Time penalty each low-level env step (0=off).",
)
parser.add_argument(
    "--no_touch_timeout_s",
    type=float,
    default=0.0,
    help="Truncate episode if no new platform touch score for this many sim seconds (0=off).",
)
parser.add_argument(
    "--max_vel_cmd_delta",
    type=float,
    default=0.0,
    help="Max change in physical vel_cmd [vx,vy,wz] per nav step (m/s, rad/s). 0=disable rate limit.",
)
parser.add_argument(
    "--w_action_rate",
    type=float,
    default=0.01,
    help="Soft penalty weight on ||vel_cmd_t - vel_cmd_{t-1}||^2 per nav step (0=off).",
)
parser.add_argument(
    "--illegal_contact_penalty",
    type=float,
    default=0.0,
    help="Extra penalty added once when an episode ends with illegal_contact.",
)
parser.add_argument(
    "--finished_reward",
    type=float,
    default=100.0,
    help="One-time reward when all 18 objects are platform-scored in one episode.",
)
parser.add_argument(
    "--visible_depth_tol",
    type=float,
    default=0.25,
    help="(Deprecated) kept for backward compatibility.",
)
parser.add_argument(
    "--visible_check_depth",
    action="store_true",
    help="(Deprecated) kept for backward compatibility.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record rollout video(s) during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=300,
    help="Recorded clip length in low-level physics steps (per RecordVideo trigger).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=0,
    help="Re-record every N physics steps; 0 = only once at physics step 0.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

import atec_rl_lab.tasks  # noqa: F401
from atec_rl_lab.tasks.task_b.env_cfg import (
    TaskBNavEnvB2Cfg,
    apply_task_d_camera_depth_clip,
    refresh_task_b_terrain_cfg,
)
from atec_rl_lab.train.nav.taskb_student_env import TaskBStudentEnv
from atec_rl_lab.train.nav.nav_cfg import TaskBStudentPPORunnerCfg
from atec_rl_lab.train.nav.nav_rsl_wrapper import NavRslRlVecEnvWrapper
from atec_rl_lab.train.nav.taskb_student_actor_critic import TaskBStudentActorCritic

import rsl_rl.runners.on_policy_runner as _runner_mod

_runner_mod.TaskBStudentActorCritic = TaskBStudentActorCritic

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_policy_path(path: str) -> str:
    if not path:
        raise ValueError("--ll_policy must be a non-empty path")
    candidates = []
    raw = os.path.expanduser(path)
    candidates.append(raw)
    if not os.path.isabs(raw):
        candidates.append(os.path.join(_REPO_ROOT, raw))
    if raw.startswith("/demo/"):
        candidates.append(os.path.join(_REPO_ROOT, raw.lstrip("/")))
    seen = set()
    for cand in candidates:
        cand = os.path.abspath(cand)
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            if cand != raw:
                print(f"[INFO] Resolved ll_policy: {path} -> {cand}", flush=True)
            return cand
    raise FileNotFoundError(f"ll_policy not found: {path}\nTried: {', '.join(sorted(seen))}")


def _configure_student_cameras(env_cfg, *, camera_height, camera_width, depth_only, tiled, camera_far_clip):
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    apply_task_d_camera_depth_clip(env_cfg.scene, float(camera_far_clip))
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
                height=int(camera_height),
                width=int(camera_width),
                data_types=data_types,
                update_period=cam.update_period,
            ),
        )


def _load_bc_into_actor_critic(actor_critic, ckpt_path: str, *, depth_only: bool) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    bc_sd = ckpt.get("model", ckpt)
    model_sd = actor_critic.state_dict()
    img_ch = 1 if depth_only else 4
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
    cc_map = {
        "memory_c.rnn.weight_ih_l0": "memory_a.rnn.weight_ih_l0",
        "memory_c.rnn.weight_hh_l0": "memory_a.rnn.weight_hh_l0",
        "memory_c.rnn.bias_ih_l0": "memory_a.rnn.bias_ih_l0",
        "memory_c.rnn.bias_hh_l0": "memory_a.rnn.bias_hh_l0",
    }
    for dst, src in cc_map.items():
        if dst in model_sd and src in model_sd and tuple(model_sd[dst].shape) == tuple(model_sd[src].shape):
            model_sd[dst] = model_sd[src].clone()
            loaded += 1
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
    actor_critic.load_state_dict(model_sd, strict=False)
    print(
        f"[INFO] BC warm-start loaded {loaded} tensors from {ckpt_path} "
        f"(img_ch={img_ch}). Skipped={len(skipped)}",
        flush=True,
    )


def main():
    device = args_cli.device if args_cli.device else "cuda"
    if args_cli.sparse_only:
        args_cli.w_dense_dist = 0.0
        args_cli.visible_check_depth = False

    policy_h = int(args_cli.policy_img_h)
    policy_w = int(args_cli.policy_img_w)
    if args_cli.camera_hw is not None:
        policy_h = policy_w = int(args_cli.camera_hw)
    cam_h = int(args_cli.sim_camera_h)
    cam_w = int(args_cli.sim_camera_w)

    env_cfg = TaskBNavEnvB2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    refresh_task_b_terrain_cfg(env_cfg)

    # When recording video, point the viewport camera at env 0 from an elevated
    # side angle so the whole playable area (robot at centre, trash in +/-5,
    # bin at +7) is visible. Default viewer looks at the world origin, which is
    # far from env 0 in multi-tile training -> it ends up pointing at the sky.
    if args_cli.video:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (8.0, -8.0, 9.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.resolution = (1280, 720)
        print(
            "[INFO] Video viewer camera -> env0 local eye=(8,-8,9) lookat=(0,0,0)",
            flush=True,
        )

    if env_cfg.observations is not None:
        env_cfg.observations.image = None
        env_cfg.observations.extero = None
    if getattr(env_cfg.scene, "lidar_sensor", None) is not None:
        env_cfg.scene.lidar_sensor = None

    _configure_student_cameras(
        env_cfg,
        camera_height=cam_h,
        camera_width=cam_w,
        depth_only=args_cli.depth_only,
        tiled=args_cli.tiled_cameras,
        camera_far_clip=args_cli.camera_far_clip,
    )

    decimation = int(getattr(env_cfg, "decimation", 4))
    sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
    nav_dt = float(args_cli.inner_steps) * decimation * sim_dt
    for cam_name in ("head_camera", "ee_camera"):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is not None:
            cam.update_period = nav_dt

    agent_cfg = TaskBStudentPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iter
    agent_cfg.num_steps_per_env = args_cli.steps_per_env
    agent_cfg.policy.img_h = policy_h
    agent_cfg.policy.img_w = policy_w
    agent_cfg.policy.img_channels = 1 if args_cli.depth_only else 4
    if args_cli.depth_only:
        agent_cfg.experiment_name = "taskb_student_b2piper_depth"

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    ll_policy_path = _resolve_policy_path(args_cli.ll_policy)

    img_ch = 1 if args_cli.depth_only else 4
    cam_mode = "depth-only 1ch" if args_cli.depth_only else "rgb+depth 4ch"
    print(
        f"[INFO] TaskB cameras: head+ee dual cam, {cam_mode}/cam, "
        f"sim={cam_h}x{cam_w} -> policy={policy_h}x{policy_w}",
        flush=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "train")
        os.makedirs(video_dir, exist_ok=True)
        interval = int(args_cli.video_interval)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=(lambda step: step == 0) if interval <= 0 else (lambda step: step % interval == 0),
            video_length=int(args_cli.video_length),
            disable_logger=True,
        )
        print(
            f"[INFO] Recording video to: {video_dir} "
            f"(length={int(args_cli.video_length)} physics steps, interval={interval or 'once'})",
            flush=True,
        )
    nav_env = TaskBStudentEnv(
        env=env,
        ll_policy_path=ll_policy_path,
        device=device,
        inner_steps=args_cli.inner_steps,
        vx_min=args_cli.vx_min,
        vx_max=args_cli.vx_max,
        vy_max=args_cli.vy_max,
        wz_max=args_cli.wz_max,
        image_h=policy_h,
        image_w=policy_w,
        depth_max=args_cli.depth_max,
        depth_only=args_cli.depth_only,
        nav_log_interval=args_cli.nav_log_interval,
        w_dense_dist=args_cli.w_dense_dist,
        sparse_touch_reward=args_cli.sparse_touch_reward,
        grasp_dist_thresh=args_cli.grasp_dist_thresh,
        time_penalty_per_env_step=args_cli.time_penalty_per_env_step,
        no_touch_timeout_s=args_cli.no_touch_timeout_s,
        finished_reward=args_cli.finished_reward,
        max_vel_cmd_delta=args_cli.max_vel_cmd_delta,
        w_action_rate=args_cli.w_action_rate,
        illegal_contact_penalty=args_cli.illegal_contact_penalty,
        visible_depth_tol=args_cli.visible_depth_tol,
        visible_check_depth=args_cli.visible_check_depth,
    )
    vec_env = NavRslRlVecEnvWrapper(nav_env)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[INFO] Resuming PPO from: {args_cli.resume}", flush=True)
        runner.load(args_cli.resume)
    if args_cli.bc_ckpt:
        _load_bc_into_actor_critic(
            runner.alg.actor_critic,
            args_cli.bc_ckpt,
            depth_only=args_cli.depth_only,
        )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(f"[INFO] Logging to: {log_dir}", flush=True)
    print(
        f"[INFO] TaskB nav train: num_envs={args_cli.num_envs} inner_steps={args_cli.inner_steps} "
        f"sim_cam={cam_h}x{cam_w} policy={policy_h}x{policy_w} "
        f"cams=head+ee img={img_ch}ch depth_only={args_cli.depth_only} "
        f"rewards guide_prog={args_cli.w_dense_dist} sparse={args_cli.sparse_touch_reward} "
        f"sparse_only={args_cli.sparse_only} "
        f"time_pen={args_cli.time_penalty_per_env_step} no_touch_timeout_s={args_cli.no_touch_timeout_s} "
        f"max_vel_cmd_delta={args_cli.max_vel_cmd_delta} w_action_rate={args_cli.w_action_rate} "
        f"illegal_pen={args_cli.illegal_contact_penalty} "
        f"finished_reward={args_cli.finished_reward}",
        flush=True,
    )
    print("[INFO] Start Task B student training...", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    nav_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
