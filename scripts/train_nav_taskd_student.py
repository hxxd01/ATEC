"""Train Task-D student nav policy and optionally warm-start from BC checkpoint."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Task D student policy (BC warm-start + PPO).")
parser.add_argument(
    "--ll_policy",
    type=str,
    default=None,
    help="Low-level locomotion JIT (.pt). Required for nav student; not used for pit e2e / pit MARG modes.",
)
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
parser.add_argument(
    "--resume-no-optimizer",
    action="store_true",
    help="Load model weights only; reset Adam (safer if resume destabilizes).",
)
parser.add_argument("--steps_per_env", type=int, default=24)
parser.add_argument("--vx_min", type=float, default=-2.0)
parser.add_argument("--vx_max", type=float, default=2.0)
parser.add_argument(
    "--policy_img_h",
    type=int,
    default=24,
    help="Policy depth input height (default 24, matches platform 480:640 aspect with policy_img_w=32).",
)
parser.add_argument(
    "--policy_img_w",
    type=int,
    default=32,
    help="Policy depth input width (default 32).",
)
parser.add_argument(
    "--camera_hw",
    type=int,
    default=None,
    help="Legacy square policy input (sets policy_img_h/w). Overrides --policy_img_h/w when set.",
)
parser.add_argument(
    "--sim_camera_h",
    type=int,
    default=None,
    help="Native sim head/ee camera height (e.g. 24). Ignored with --platform_depth_train.",
)
parser.add_argument(
    "--sim_camera_w",
    type=int,
    default=None,
    help="Native sim head/ee camera width (e.g. 32). Ignored with --platform_depth_train.",
)
parser.add_argument(
    "--platform_depth_train",
    "--pit_platform_depth",
    action="store_true",
    help=(
        "Sim cameras at platform resolution (default 480x640), prep_depth bilinear+log1p -> "
        "--policy_img_h/w (e.g. 24x32). Use with --student_pit_dagger/--student_pit_e2e to match "
        "demo/server.py deploy before platform fine-tune."
    ),
)
parser.add_argument(
    "--platform_depth_h",
    type=int,
    default=480,
    help="Sim camera H when --platform_depth_train (demo/server.py SERVER_DEPTH_H).",
)
parser.add_argument(
    "--platform_depth_w",
    type=int,
    default=640,
    help="Sim camera W when --platform_depth_train (demo/server.py SERVER_DEPTH_W).",
)
parser.add_argument(
    "--depth_render_h",
    type=int,
    default=None,
    help="Deprecated alias for --policy_img_h (kept for old scripts).",
)
parser.add_argument(
    "--depth_render_w",
    type=int,
    default=None,
    help="Deprecated alias for --policy_img_w (kept for old scripts).",
)
parser.add_argument(
    "--depth_max",
    type=float,
    default=5.0,
    help="prep_depth log1p clamp (meters). Not in server.py; default 5.0 matches existing BC/train.",
)
parser.add_argument(
    "--camera_far_clip",
    type=float,
    default=None,
    help="Camera far clip (m). Default 50 (platform/play). Use 5 only to hide distant pits in multi-env.",
)
parser.add_argument(
    "--depth_only",
    action="store_true",
    help="Use depth maps only. Isaac cameras render depth only (no RGB). Pit default: head only.",
)
parser.add_argument(
    "--ee_depth",
    action="store_true",
    help="Also render ee camera depth (default pit student: head depth only).",
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
parser.add_argument(
    "--push-box-drop-com-z",
    type=float,
    default=0.295,
    help="End push stage when box center-of-mass world z drops below this value.",
)
parser.add_argument(
    "--push-min-box-nominal-x",
    type=float,
    default=-0.8,
    help="Push completes on box drop only if box nominal x (env-relative) exceeds this value.",
)
parser.add_argument(
    "--push-right-reward-dist",
    type=float,
    default=1.0,
    help="Cap lateral push progress/reward at this many meters.",
)

# ── Task D pit MARG locomotion (parallel path; does NOT use TaskDStudentActorCritic) ──
pit_grp = parser.add_argument_group(
    "pit MARG locomotion",
    "Train/play pit-crossing with MargActorCritic + pit env rewards/terminations. "
    "Independent from depth student nav below.",
)
pit_grp.add_argument(
    "--pit_marg_locomotion",
    action="store_true",
    help="Train pit MARG (height_map 187D). Exits before student nav setup.",
)
pit_grp.add_argument(
    "--pit_marg_play",
    action="store_true",
    help="Play/eval pit MARG checkpoint (same env + network as --pit_marg_locomotion).",
)
pit_grp.add_argument(
    "--pit_checkpoint",
    type=str,
    default=None,
    help="Checkpoint for --pit_marg_play (or use --checkpoint).",
)
pit_grp.add_argument("--checkpoint", type=str, default=None, help="Alias for --pit_checkpoint in pit play mode.")
pit_grp.add_argument("--pit_width_min", type=float, default=0.4)
pit_grp.add_argument("--pit_width_max", type=float, default=1.4)
pit_grp.add_argument("--pit_curriculum_levels", type=int, default=11)
pit_grp.add_argument("--pit_level", type=int, default=0, help="Fixed pit-width row for --pit_marg_play.")
pit_grp.add_argument("--pit_curriculum", action="store_true", help="Full pit-width curriculum during play.")
pit_grp.add_argument("--command_vx_min", type=float, default=0.0, help="Pit train: min forward cmd (m/s).")
pit_grp.add_argument("--command_vx_max", type=float, default=4.0, help="Pit train: max forward cmd (m/s).")
pit_grp.add_argument(
    "--command_vx",
    type=float,
    default=None,
    help="Pit play: fixed forward cmd (m/s). Pit train: alias for --command_vx_max.",
)
pit_grp.add_argument(
    "--command_curriculum_start",
    type=float,
    default=0.1,
    help="Pit train: initial vx curriculum fraction.",
)
pit_grp.add_argument("--pit_sim_easy", action="store_true", help="Pit train: sim-easy crossing overrides.")
pit_grp.add_argument("--pit_sim_easy_vx", type=float, default=1.0)
pit_grp.add_argument("--pit_spawn_x_offset", type=float, default=0.0)
pit_grp.add_argument("--pit_spawn_local_x", type=float, default=None)
pit_grp.add_argument(
    "--pit_spawn_x_jitter",
    type=float,
    default=None,
    help=(
        "Uniform random ± jitter on env-local spawn x at reset (m). "
        "Default 0 (off); with --pit_dr_light defaults to 0.5 m."
    ),
)
pit_grp.add_argument(
    "--pit_dr_light",
    action="store_true",
    help=(
        "Pit reset DR: joint scale 0.5-1.5, spawn xy jitter ±0.5 m, "
        "z=0.545±0.025 m, yaw ±0.15 rad, small initial vx/vy/wz (sim-only; keeps obs noise off)."
    ),
)
pit_grp.add_argument("--pit_dr_spawn_y_jitter", type=float, default=None, help="Override --pit_dr_light robot spawn y jitter toward right (m), [0, value]. Default 0.5.")
pit_grp.add_argument(
    "--pit_dr_spawn_z",
    type=float,
    default=None,
    help="DR spawn base z (m). Default 0.545 (teleop stand ~0.529, slightly higher).",
)
pit_grp.add_argument(
    "--pit_dr_spawn_z_jitter",
    type=float,
    default=None,
    help="Uniform ± z jitter on spawn (m). Default 0.025 with --pit_dr_light.",
)
pit_grp.add_argument(
    "--pit_dr_yaw_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Override yaw random range in rad (default -0.15 0.15).",
)
pit_grp.add_argument(
    "--pit_dr_joint_scale",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Override reset_joints_by_scale range (default 0.5 1.5).",
)
pit_grp.add_argument("--pit_warmup_steps", type=int, default=0, help="MARG history warmup before pit play video.")
pit_grp.add_argument("--pit_stochastic", action="store_true", help="Stochastic actions during pit play.")
pit_grp.add_argument("--pit_debug", action="store_true", help="Print pit play stats every 50 steps.")
pit_grp.add_argument("--pit_real_time", action="store_true", help="Real-time pit play loop.")
pit_grp.add_argument(
    "--no_pit_box",
    action="store_true",
    help="Disable Task D push box in pit env (box is on by default at teleop pushed pose).",
)
pit_grp.add_argument(
    "--pit_box_default_spawn",
    action="store_true",
    help="Use Task D default box spawn (1.2, 1.6, 0.5) instead of teleop pushed pose.",
)
pit_grp.add_argument(
    "--pit_box_x_jitter_down",
    type=float,
    default=None,
    help="Box x jitter downward from pushed max x (m). Default 0.5.",
)
pit_grp.add_argument(
    "--pit_box_y_jitter",
    type=float,
    default=None,
    help="Box y jitter ± (m). Default 0.5.",
)

e2e_grp = parser.add_argument_group(
    "student pit e2e",
    "Shared student depth encoder + GRU; pit_actor outputs 12 leg actions (demo/server.py obs). "
    "Same pit env rewards/terminations; no --ll_policy.",
)
e2e_grp.add_argument(
    "--student_pit_e2e",
    action="store_true",
    help=(
        "Single-stage PPO fine-tune for MargDepthPitActorCritic (100% student rollouts, no teacher). "
        "Same Marg env + obs_manager as DAgger/play. "
        "With --resume on a DAgger BC ckpt: loads weights only, iter resets to 0 (finetune phase)."
    ),
)
e2e_grp.add_argument(
    "--finetune_from_dagger",
    action="store_true",
    help=(
        "Explicit: --student_pit_e2e + --resume <dagger model_*.pt>. "
        "Loads BC weights, fresh optimizer, iter=0, env/cmd matched to DAgger (vx≈0.6 unless overridden)."
    ),
)
e2e_grp.add_argument(
    "--student_pit_dagger",
    action="store_true",
    help=(
        "DAgger train depth student with frozen MARG height-map teacher (--pit_teacher_ckpt required). "
        "Add --pit_platform_depth (alias --platform_depth_train) to fine-tune with 480x640→24x32 depth "
        "matching platform deploy; default is native sim_camera=policy size (24x32)."
    ),
)
e2e_grp.add_argument(
    "--pit_teacher_ckpt",
    type=str,
    default=None,
    help="MARG pit teacher checkpoint (MargActorCritic) for --student_pit_dagger.",
)
e2e_grp.add_argument("--dagger_coef", type=float, default=1.0, help="BC loss weight on teacher leg actions.")
e2e_grp.add_argument(
    "--dagger_beta",
    type=float,
    default=1.0,
    help="Initial rollout mix: P(execute teacher action). Decays to --dagger_beta_end.",
)
e2e_grp.add_argument("--dagger_beta_end", type=float, default=0.0)
e2e_grp.add_argument("--dagger_beta_decay_iters", type=int, default=4000)
e2e_grp.add_argument(
    "--no_teacher_warmstart",
    action="store_true",
    help="Skip copying teacher estimator into depth student at init.",
)
e2e_grp.add_argument(
    "--student_encoder_ckpt",
    type=str,
    default=None,
    help="Warm-start shared encoder/GRU from nav student or BC ckpt (loads encoders only, fresh pit head).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.pit_marg_locomotion and args_cli.pit_marg_play:
    parser.error("Use only one of --pit_marg_locomotion or --pit_marg_play.")
if args_cli.pit_marg_play and not (args_cli.pit_checkpoint or args_cli.checkpoint):
    parser.error("--pit_marg_play requires --pit_checkpoint or --checkpoint.")
if args_cli.student_pit_dagger and not args_cli.pit_teacher_ckpt:
    parser.error("--student_pit_dagger requires --pit_teacher_ckpt.")
if args_cli.student_pit_dagger and args_cli.student_pit_e2e:
    parser.error("Use only one of --student_pit_dagger or --student_pit_e2e.")
if args_cli.finetune_from_dagger and args_cli.student_pit_dagger:
    parser.error("Use --finetune_from_dagger with --student_pit_e2e, not --student_pit_dagger.")
if args_cli.finetune_from_dagger and not args_cli.resume:
    parser.error("--finetune_from_dagger requires --resume <dagger_bc_checkpoint>.")
if args_cli.finetune_from_dagger:
    args_cli.student_pit_e2e = True
_nav_modes = (
    not args_cli.pit_marg_locomotion
    and not args_cli.pit_marg_play
    and not args_cli.student_pit_e2e
    and not args_cli.student_pit_dagger
)
if _nav_modes and not args_cli.ll_policy:
    parser.error("--ll_policy is required for Task D nav student training.")
# Student / pit-e2e use cameras; pit MARG train uses height_scanner only.
if _nav_modes or args_cli.student_pit_e2e or args_cli.student_pit_dagger or (args_cli.pit_marg_play and args_cli.video):
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

import atec_rl_lab.tasks  # noqa: F401
from atec_rl_lab.tasks.task_d.env_cfg import (
    TASK_D_PLATFORM_CAMERA_FAR,
    TaskDEnvB2Cfg,
    apply_task_d_camera_depth_clip,
    refresh_task_d_terrain_cfg,
)
from atec_rl_lab.train.nav.taskd_student_env import TaskDStudentEnv
from atec_rl_lab.train.nav.nav_cfg import (
    TaskDStudentPPORunnerCfg,
    TaskDMargDepthPitE2EPPORunnerCfg,
    TaskDMargDepthPitDaggerPPORunnerCfg,
)
from atec_rl_lab.train.nav.nav_rsl_wrapper import NavRslRlVecEnvWrapper
from atec_rl_lab.train.nav.taskd_student_actor_critic import TaskDStudentActorCritic
from atec_rl_lab.train.nav.taskd_marg_depth_pit_e2e_env import configure_pit_e2e_cameras
from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import (
    attach_dagger_depth_obs,
    configure_pit_e2e_dagger_env_cfg,
    configure_pit_e2e_env_cfg,
)
from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import register_marg_modules
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from atec_rl_lab.train.nav.marg_depth_pit_actor_critic import MargDepthPitActorCritic
from atec_rl_lab.train.locomotion.marg.marg_ppo import MargPPO
from atec_rl_lab.train.pit_marg.marg_pit_dagger_ppo import MargPitDaggerPPO
from atec_rl_lab.train.pit_marg.marg_teacher_loader import (
    load_marg_teacher_checkpoint,
    warmstart_depth_student_from_teacher,
)

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


def _sync_ppo_learning_rate(runner) -> float:
    """Keep PPO.adaptive schedule in sync with the optimizer LR loaded from checkpoint."""
    lr = float(runner.alg.optimizer.param_groups[0]["lr"])
    runner.alg.learning_rate = lr
    return lr


def _load_ppo_checkpoint(runner, ckpt_path: str, *, load_optimizer: bool = True) -> dict:
    loaded = torch.load(ckpt_path, map_location=runner.device, weights_only=False)
    ckpt_sd = loaded["model_state_dict"]
    policy = _get_policy_module(runner.alg)
    model_sd = policy.state_dict()
    merged, loaded_keys, partial_keys, skipped_keys = _merge_state_dict_partial(model_sd, ckpt_sd)
    policy.load_state_dict(merged, strict=False)

    can_load_optimizer = len(skipped_keys) == 0
    if load_optimizer and can_load_optimizer and "optimizer_state_dict" in loaded:
        runner.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        lr = _sync_ppo_learning_rate(runner)
        print(f"[INFO] Optimizer restored; learning_rate={lr:.2e}", flush=True)
    elif load_optimizer and not can_load_optimizer:
        print(
            "[WARN] Optimizer state skipped because checkpoint architecture differs "
            f"({len(skipped_keys)} tensors not loaded).",
            flush=True,
        )
    elif not load_optimizer:
        print("[INFO] Optimizer not loaded (--resume-no-optimizer); using fresh Adam.", flush=True)

    if can_load_optimizer and "iter" in loaded:
        # Checkpoints are saved after iteration `iter` completes; continue from the next one.
        runner.current_learning_iteration = int(loaded["iter"]) + 1
        print(
            f"[INFO] Resume from checkpoint iter={loaded['iter']} -> next learning iteration "
            f"{runner.current_learning_iteration}",
            flush=True,
        )
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
    return loaded


def _pit_marg_depth_ckpt_kind(ckpt_path: str) -> str | None:
    """Infer 'dagger' | 'e2e' from checkpoint log folder name."""
    ckpt_dir = os.path.abspath(os.path.dirname(os.path.abspath(ckpt_path)))
    exp_name = os.path.basename(os.path.dirname(ckpt_dir))
    if "dagger" in exp_name:
        return "dagger"
    if "e2e" in exp_name:
        return "e2e"
    return None


def _is_finetune_from_dagger(args_cli) -> bool:
    """Load DAgger BC ckpt into single-stage PPO (not continue DAgger)."""
    if bool(getattr(args_cli, "finetune_from_dagger", False)):
        return True
    if not getattr(args_cli, "resume", None) or getattr(args_cli, "student_pit_dagger", False):
        return False
    if not getattr(args_cli, "student_pit_e2e", False):
        return False
    return _pit_marg_depth_ckpt_kind(args_cli.resume) == "dagger"


def _validate_pit_marg_depth_resume(ckpt_path: str, *, dagger: bool, finetune_from_dagger: bool = False) -> None:
    """Reject resume when checkpoint training mode does not match CLI flags."""
    ckpt = os.path.abspath(ckpt_path)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
    kind = _pit_marg_depth_ckpt_kind(ckpt)
    if finetune_from_dagger:
        if kind != "dagger":
            raise ValueError(
                f"--finetune_from_dagger expects a DAgger BC checkpoint, got path kind={kind!r}: {ckpt}"
            )
        print(
            f"[TaskDMargDepthPitFinetune] DAgger BC ckpt -> single-stage PPO "
            f"(weights only, iter=0, same obs as play)",
            flush=True,
        )
        return
    mode = "dagger" if dagger else "e2e"
    if kind is None:
        print(
            f"[WARN] Cannot infer checkpoint mode from path ({ckpt}); "
            f"ensure --student_pit_{mode} matches how it was trained.",
            flush=True,
        )
        return
    if kind != mode:
        if kind == "dagger" and not dagger:
            raise ValueError(
                f"Checkpoint {ckpt} is from DAgger BC training. "
                "To fine-tune with single-stage PPO use --student_pit_e2e --resume <ckpt> "
                "(or --finetune_from_dagger). To continue DAgger use --student_pit_dagger --resume <ckpt>."
            )
        other_flag = "--student_pit_dagger" if kind == "dagger" else "--student_pit_e2e"
        this_flag = "--student_pit_dagger" if dagger else "--student_pit_e2e"
        raise ValueError(
            f"Checkpoint {ckpt} is from a {kind} run but you passed {this_flag}. "
            f"Use {other_flag} (same obs manager + Marg env as play)."
        )


def _restore_dagger_schedule(runner) -> None:
    """Sync dagger_beta decay with resumed learning iteration."""
    alg = runner.alg
    if not hasattr(alg, "set_dagger_iteration"):
        return
    iter_idx = int(runner.current_learning_iteration)
    alg._dagger_updates = iter_idx
    alg.set_dagger_iteration(iter_idx)
    print(
        f"[TaskDMargDepthPitDAgger] restored schedule at iter={iter_idx}, "
        f"dagger_beta={alg.dagger_beta:.4f}",
        flush=True,
    )


def _load_pit_finetune_from_dagger(runner, ckpt_path: str) -> None:
    """Load DAgger BC policy weights; fresh PPO optimizer; restart iteration counter."""
    loaded = torch.load(ckpt_path, map_location=runner.device, weights_only=False)
    ckpt_sd = loaded["model_state_dict"]
    policy = _get_policy_module(runner.alg)
    model_sd = policy.state_dict()
    merged, loaded_keys, partial_keys, skipped_keys = _merge_state_dict_partial(model_sd, ckpt_sd)
    policy.load_state_dict(merged, strict=False)
    runner.current_learning_iteration = 0
    src_iter = loaded.get("iter")
    print(
        f"[TaskDMargDepthPitFinetune] loaded BC weights from iter={src_iter} "
        f"(exact={len(loaded_keys)}, partial={len(partial_keys)}, skipped={len(skipped_keys)}); "
        "optimizer=fresh MargPPO, learning starts at iter=0",
        flush=True,
    )
    if skipped_keys:
        print(f"[WARN] Finetune skipped tensors: {skipped_keys[:8]}", flush=True)
    print(
        "[INFO] Finetune obs: MargEnvCfg + obs_manager "
        "(proprio, proprio_history, depth); 100% student rollouts.",
        flush=True,
    )


def _load_pit_marg_depth_resume(runner, ckpt_path: str, *, dagger: bool, load_optimizer: bool) -> None:
    loaded = _load_ppo_checkpoint(runner, ckpt_path, load_optimizer=load_optimizer)
    if dagger:
        _restore_dagger_schedule(runner)
    resume_iter = loaded.get("iter")
    if resume_iter is not None:
        print(
            "[INFO] Resume obs path: MargEnvCfg + obs_manager "
            "(proprio, proprio_history, depth, critic_priv); play uses the same stack.",
            flush=True,
        )


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


_ENCODER_PREFIXES = (
    "head_encoder.",
    "ee_encoder.",
    "proprio_mlp.",
    "fuse.",
    "memory_a.",
    "actor_obs_normalizer.",
)


def _load_student_encoders_for_pit(actor_critic, ckpt_path: str) -> None:
    """Warm-start pit e2e shared trunk from nav student / BC checkpoint."""
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_sd = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
    model_sd = actor_critic.state_dict()
    loaded_n = 0
    for key, val in ckpt_sd.items():
        mapped = key
        if key.startswith("actor.") and not key.startswith("actor_obs_normalizer."):
            continue
        if mapped not in model_sd:
            continue
        if not any(mapped.startswith(p) for p in _ENCODER_PREFIXES):
            continue
        if tuple(model_sd[mapped].shape) != tuple(val.shape):
            continue
        model_sd[mapped] = val
        loaded_n += 1
    actor_critic.load_state_dict(model_sd, strict=False)
    print(
        f"[INFO] Pit e2e encoder warm-start: loaded {loaded_n} tensors from {ckpt_path} "
        "(pit_actor head remains randomly initialized).",
        flush=True,
    )


def _run_student_pit_e2e_train(args_cli, device: str, *, dagger: bool = False) -> None:
    import rsl_rl.runners.on_policy_runner as _runner_mod

    finetune = _is_finetune_from_dagger(args_cli)
    if args_cli.resume:
        _validate_pit_marg_depth_resume(
            args_cli.resume, dagger=dagger, finetune_from_dagger=finetune
        )

    _runner_mod.MargDepthPitActorCritic = MargDepthPitActorCritic
    _runner_mod.MargPPO = MargPPO
    if dagger:
        _runner_mod.MargPitDaggerPPO = MargPitDaggerPPO

    cam_h, cam_w, policy_h, policy_w, depth_mode = _resolve_depth_pipeline(args_cli)
    camera_far_clip = _resolve_camera_far_clip(args_cli.camera_far_clip)
    head_depth_only = not bool(getattr(args_cli, "ee_depth", False))
    if dagger or finetune:
        # Finetune from DAgger BC: keep same cmd/env as BC (fixed vx≈0.6 unless CLI overrides).
        env_cfg = configure_pit_e2e_dagger_env_cfg(args_cli)
    else:
        env_cfg = configure_pit_e2e_env_cfg(args_cli)
    decimation = int(getattr(env_cfg, "decimation", 4))
    sim_dt = float(getattr(env_cfg.sim, "dt", 0.005))
    phys_dt = decimation * sim_dt
    configure_pit_e2e_cameras(
        env_cfg,
        camera_height=cam_h,
        camera_width=cam_w,
        depth_only=args_cli.depth_only,
        tiled=args_cli.tiled_cameras,
        camera_far_clip=camera_far_clip,
        update_period=phys_dt,
        head_depth_only=head_depth_only,
    )
    attach_dagger_depth_obs(
        env_cfg,
        policy_h=policy_h,
        policy_w=policy_w,
        depth_max=float(args_cli.depth_max),
        depth_only=bool(args_cli.depth_only),
        depth_render_h=cam_h if depth_mode == "platform" else None,
        depth_render_w=cam_w if depth_mode == "platform" else None,
        head_depth_only=head_depth_only,
    )
    mode_tag = "DAgger" if dagger else ("Finetune-PPO" if finetune else "PPO")
    box_tag = "with Task D box" if not bool(getattr(args_cli, "no_pit_box", False)) else "no box"
    depth_cam_tag = "head" if head_depth_only else "head+ee"
    print(
        f"[TaskDMargDepthPit] mode={mode_tag}, vec_env=RslRlVecEnvWrapper, "
        f"obs=proprio+proprio_history+depth({depth_cam_tag})(+height_map teacher-only), "
        f"env={box_tag}, "
        f"depth pipeline={depth_mode}, sim={cam_h}x{cam_w}, "
        f"policy={policy_h}x{policy_w}, depth_only={args_cli.depth_only}, "
        f"pit_width={env_cfg.pit_width_range}, success_post={env_cfg.pit_success_post_cross_distance}m, "
        f"vx=[{env_cfg.command_lin_vel_x_min}, {env_cfg.command_lin_vel_x_max}] m/s, "
        f"network=MARG(estimator+depthCNN)+12leg",
        flush=True,
    )

    if dagger:
        agent_cfg = TaskDMargDepthPitDaggerPPORunnerCfg()
        agent_cfg.algorithm.dagger_coef = float(args_cli.dagger_coef)
        agent_cfg.algorithm.dagger_beta = float(args_cli.dagger_beta)
        agent_cfg.algorithm.dagger_beta_end = float(args_cli.dagger_beta_end)
        agent_cfg.algorithm.dagger_beta_decay_iters = int(args_cli.dagger_beta_decay_iters)
        register_marg_modules()
    else:
        agent_cfg = TaskDMargDepthPitE2EPPORunnerCfg()
    agent_cfg.max_iterations = int(args_cli.max_iter)
    agent_cfg.num_steps_per_env = int(args_cli.steps_per_env)
    agent_cfg.policy.img_h = policy_h
    agent_cfg.policy.img_w = policy_w
    agent_cfg.policy.depth_channels = 1 if args_cli.depth_only else 4
    agent_cfg.policy.head_depth_only = head_depth_only

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg)
    clip_actions = getattr(agent_cfg, "clip_actions", None)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    if dagger:
        obs_sample = {k: v for k, v in vec_env.get_observations().items()}
        teacher_ckpt = os.path.abspath(args_cli.pit_teacher_ckpt)
        teacher = load_marg_teacher_checkpoint(teacher_ckpt, obs_sample, device)
        runner.alg.teacher = teacher
        if not args_cli.no_teacher_warmstart and not args_cli.resume:
            warmstart_depth_student_from_teacher(runner.alg.policy, teacher_ckpt)
        elif args_cli.resume:
            print("[INFO] Resume: skip teacher estimator warm-start (using checkpoint weights).", flush=True)
        with torch.no_grad():
            obs0 = {k: v for k, v in vec_env.get_observations().items()}
            t_act = teacher.act_inference(
                {
                    "proprio": obs0["proprio"],
                    "proprio_history": obs0["proprio_history"],
                    "height_map": obs0["height_map"],
                }
            )
            s_act = runner.alg.policy.act_inference(obs0)
            act_mse = torch.mean((t_act - s_act) ** 2).item()
        print(
            f"[TaskDMargDepthPitDAgger] teacher={teacher_ckpt}, "
            f"dagger_coef={args_cli.dagger_coef}, beta={args_cli.dagger_beta}->{args_cli.dagger_beta_end} "
            f"over {args_cli.dagger_beta_decay_iters} iters, init student-vs-teacher action MSE={act_mse:.4f}",
            flush=True,
        )
        # Quick teacher-only rollout: verify MARG history + height_map obs before PPO loop.
        obs_roll = {k: v.clone() for k, v in vec_env.get_observations().items()}
        teacher_obs_keys = ("proprio", "proprio_history", "height_map")
        warmup = 6
        for _ in range(warmup):
            with torch.inference_mode():
                t_act = teacher.act_inference({k: obs_roll[k] for k in teacher_obs_keys})
            obs_roll, _, _, _ = vec_env.step(t_act)
            obs_roll = {k: v for k, v in obs_roll.items()}
        robot = vec_env.unwrapped.scene["robot"]
        x0 = robot.data.root_pos_w[:, 0].clone()
        for _ in range(24):
            with torch.inference_mode():
                t_act = teacher.act_inference({k: obs_roll[k] for k in teacher_obs_keys})
            obs_roll, _, _, _ = vec_env.step(t_act)
            obs_roll = {k: v for k, v in obs_roll.items()}
        dx = float((robot.data.root_pos_w[:, 0] - x0).mean().item())
        vx = float(robot.data.root_lin_vel_w[:, 0].mean().item())
        act_norm = float(t_act.norm(dim=-1).mean().item())
        print(
            f"[TaskDMargDepthPitDAgger] teacher-only sanity (warmup={warmup}, then 24 steps): "
            f"mean_dx={dx:.3f}m, mean_vx={vx:.3f}m/s, |action|={act_norm:.3f}",
            flush=True,
        )
        if dx < 0.05:
            print(
                "[TaskDMargDepthPitDAgger] WARNING: teacher barely moved — check command_vx, "
                "history reset, and teacher ckpt.",
                flush=True,
            )
        vec_env.reset()

    if args_cli.student_encoder_ckpt or args_cli.bc_ckpt:
        print(
            "[WARN] --student_encoder_ckpt/--bc_ckpt ignored for MARG-depth pit e2e "
            "(different architecture from nav student GRU).",
            flush=True,
        )
    if args_cli.resume:
        if finetune:
            print(f"[INFO] Finetune from DAgger BC: {args_cli.resume}", flush=True)
            _load_pit_finetune_from_dagger(runner, args_cli.resume)
        else:
            mode_label = "DAgger" if dagger else "e2e"
            print(f"[INFO] Resuming pit MARG-depth {mode_label} from {args_cli.resume}", flush=True)
            _load_pit_marg_depth_resume(
                runner,
                args_cli.resume,
                dagger=dagger,
                load_optimizer=not args_cli.resume_no_optimizer,
            )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if finetune:
        mode_name = "marg_depth_pit_finetune_from_dagger"
    else:
        mode_name = "marg_depth_pit_dagger" if dagger else "marg_depth_pit_e2e"
    deploy_notes = {
        "mode": mode_name,
        "init_ckpt": os.path.abspath(args_cli.resume) if finetune and args_cli.resume else None,
        "teacher_ckpt": os.path.abspath(args_cli.pit_teacher_ckpt) if dagger else None,
        "head_depth_only": head_depth_only,
        "depth_pipeline": {
            "mode": depth_mode,
            "sim_camera_h": int(cam_h),
            "sim_camera_w": int(cam_w),
            "policy_img_h": int(policy_h),
            "policy_img_w": int(policy_w),
            "prep": "demo/depth_preprocess.prep_depth (bilinear + log1p, depth_max=5)",
        },
        "obs_groups": {
            "proprio": "43D MARG leg proprio + cmd",
            "proprio_history": "258D",
            "depth": f"{depth_cam_tag} depth flat {policy_h}x{policy_w}",
            "critic_priv": "42D privileged (train only)",
        },
        "network": (
            f"MargDepthPitActorCritic: estimator(history)->7, depthCNN({depth_cam_tag})->16, "
            "actor MLP 512-256-128 -> 12 legs (same layout as MargActorCritic w/ depth replacing elevation)"
        ),
        "deploy": "demo/server.py 480x640 depth + MARG proprio; solution_marg_depth_pit.py prep_depth->24x32",
        "nav_student": "unchanged (TaskDStudentActorCritic separate)",
    }
    dump_yaml(os.path.join(log_dir, "params", "deploy_pit_e2e.yaml"), deploy_notes)
    _dump_pit_marg_deploy_agent_yaml(
        log_dir,
        policy_h=policy_h,
        policy_w=policy_w,
        depth_only=bool(args_cli.depth_only),
        depth_max=float(args_cli.depth_max),
        head_depth_only=head_depth_only,
        depth_mode=depth_mode,
        sim_cam_h=cam_h,
        sim_cam_w=cam_w,
        platform_h=int(args_cli.platform_depth_h),
        platform_w=int(args_cli.platform_depth_w),
    )

    phase = "DAgger" if dagger else ("Finetune" if finetune else "e2e")
    print(f"[INFO] Pit MARG-depth {phase} logging to {log_dir}", flush=True)
    # MARG proprio history must align with episode resets; random ep len breaks estimator input.
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    vec_env.close()


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


def _resolve_camera_far_clip(cli_value: float | None) -> float:
    return float(TASK_D_PLATFORM_CAMERA_FAR if cli_value is None else cli_value)


def _resolve_policy_img_size(args_cli) -> tuple[int, int]:
    """Return (policy_h, policy_w) for rect policy input."""
    if args_cli.camera_hw is not None:
        side = int(args_cli.camera_hw)
        return side, side
    ph = int(args_cli.depth_render_h) if args_cli.depth_render_h is not None else int(args_cli.policy_img_h)
    pw = int(args_cli.depth_render_w) if args_cli.depth_render_w is not None else int(args_cli.policy_img_w)
    return ph, pw


def _resolve_depth_pipeline(args_cli) -> tuple[int, int, int, int, str]:
    """Return (sim_cam_h, sim_cam_w, policy_h, policy_w, mode_label)."""
    policy_h, policy_w = _resolve_policy_img_size(args_cli)
    if args_cli.platform_depth_train:
        if args_cli.sim_camera_h is not None or args_cli.sim_camera_w is not None:
            print(
                "[WARN] --sim_camera_h/w ignored when --platform_depth_train is set "
                "(using --platform_depth_h/w for sim cameras).",
                flush=True,
            )
        cam_h = int(args_cli.platform_depth_h)
        cam_w = int(args_cli.platform_depth_w)
        return cam_h, cam_w, policy_h, policy_w, "platform"

    cam_h = int(args_cli.sim_camera_h) if args_cli.sim_camera_h is not None else policy_h
    cam_w = int(args_cli.sim_camera_w) if args_cli.sim_camera_w is not None else policy_w
    return cam_h, cam_w, policy_h, policy_w, "native"


def _configure_student_cameras(
    env_cfg,
    *,
    camera_height: int,
    camera_width: int,
    depth_only: bool,
    tiled: bool,
    camera_far_clip: float,
) -> None:
    """Set head/ee camera resolution; optionally depth-only and/or tiled rendering."""
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


def _dump_deploy_agent_yaml(
    log_dir: str,
    *,
    policy_h: int,
    policy_w: int,
    depth_only: bool,
    depth_max: float,
    platform: bool,
    platform_h: int,
    platform_w: int,
) -> None:
    """Write demo-compatible policy block (solution.py reads demo/agent.yaml)."""
    deploy = {
        "policy": {
            "img_h": int(policy_h),
            "img_w": int(policy_w),
            "img_channels": 1 if depth_only else 4,
            "platform_depth_h": int(platform_h),
            "platform_depth_w": int(platform_w),
            "depth_max": float(depth_max),
            "proprio_dim": 9,
            "enc_dim": 128,
            "fuse_dim": 256,
            "rnn_type": "gru",
            "rnn_hidden_dim": 256,
            "rnn_num_layers": 1,
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "actor_hidden_dims": [256],
            "critic_hidden_dims": [256, 128],
            "init_noise_std": 0.6,
            "noise_std_type": "scalar",
        }
    }
    dump_yaml(os.path.join(log_dir, "params", "deploy_agent.yaml"), deploy)


def _dump_pit_marg_deploy_agent_yaml(
    log_dir: str,
    *,
    policy_h: int,
    policy_w: int,
    depth_only: bool,
    depth_max: float,
    head_depth_only: bool,
    depth_mode: str,
    sim_cam_h: int,
    sim_cam_w: int,
    platform_h: int,
    platform_w: int,
) -> None:
    """Write demo/agent_marg_depth_pit.yaml-compatible deploy block for solution_marg_depth_pit.py."""
    deploy = {
        "policy": {
            "img_h": int(policy_h),
            "img_w": int(policy_w),
            "depth_channels": 1 if depth_only else 4,
            "head_depth_only": bool(head_depth_only),
            "depth_max": float(depth_max),
            "depth_render_h": int(sim_cam_h),
            "depth_render_w": int(sim_cam_w),
            "platform_depth_h": int(platform_h),
            "platform_depth_w": int(platform_w),
            "depth_pipeline": depth_mode,
            "enc_dim": 128,
            "elevation_out_dim": 16,
            "estimator_hidden_dims": [128],
            "depth_hidden_dims": [128, 64],
            "actor_obs_normalization": True,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 0.6,
            "max_noise_std": 2.0,
            "noise_std_type": "scalar",
        }
    }
    dump_yaml(os.path.join(log_dir, "params", "deploy_agent.yaml"), deploy)
    dump_yaml(os.path.join(log_dir, "params", "agent_marg_depth_pit.yaml"), deploy)


def main():
    if args_cli.pit_marg_locomotion or args_cli.pit_marg_play:
        from atec_rl_lab.train.pit_marg.taskd_pit_marg_runner import play_pit_marg, train_pit_marg

        if args_cli.pit_marg_play:
            play_pit_marg(args_cli, simulation_app)
        else:
            log_dir = train_pit_marg(args_cli)
            print(f"[INFO] Pit MARG train done. log_dir={log_dir}", flush=True)
            print(
                "[INFO] Deploy/play: "
                "python scripts/train_nav_taskd_student.py --pit_marg_play "
                f"--pit_checkpoint {log_dir}/model_<iter>.pt --headless --num_envs 1",
                flush=True,
            )
        return

    device = args_cli.device if args_cli.device else "cuda"

    if args_cli.student_pit_dagger:
        _run_student_pit_e2e_train(args_cli, device, dagger=True)
        return

    if args_cli.student_pit_e2e:
        _run_student_pit_e2e_train(args_cli, device, dagger=False)
        return

    env_cfg = TaskDEnvB2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    refresh_task_d_terrain_cfg(env_cfg)
    scene_env_spacing = float(env_cfg.scene.env_spacing)
    cam_h, cam_w, policy_h, policy_w, depth_mode = _resolve_depth_pipeline(args_cli)
    camera_far_clip = _resolve_camera_far_clip(args_cli.camera_far_clip)
    # Keep camera sensors but drop image/lidar observation managers (read camera buffers directly).
    if env_cfg.observations is not None:
        env_cfg.observations.image = None
        env_cfg.observations.extero = None
    if getattr(env_cfg.scene, "lidar_sensor", None) is not None:
        env_cfg.scene.lidar_sensor = None
    print("[INFO] Student sim: LiDAR disabled (extero obs + lidar_sensor off).", flush=True)
    _configure_student_cameras(
        env_cfg,
        camera_height=cam_h,
        camera_width=cam_w,
        depth_only=args_cli.depth_only,
        tiled=args_cli.tiled_cameras,
        camera_far_clip=camera_far_clip,
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
    cam_update_period = nav_dt
    if args_cli.ppo_no_train and args_cli.video:
        print(
            "[INFO] ppo_no_train + --video: cam_update=nav_dt (match training obs timing).",
            flush=True,
        )
    for cam_name in ("head_camera", "ee_camera"):
        cam = getattr(env_cfg.scene, cam_name, None)
        if cam is not None:
            cam.update_period = cam_update_period
    if depth_mode == "platform":
        print(
            f"[INFO] Depth pipeline (platform, match demo/server.py): sim {cam_h}x{cam_w} float32 "
            f"-> bilinear {policy_h}x{policy_w} -> log1p -> policy {policy_h}x{policy_w}",
            flush=True,
        )
    else:
        print(
            f"[INFO] Depth pipeline (native): sim {cam_h}x{cam_w} -> bilinear {policy_h}x{policy_w} "
            f"-> log1p -> policy {policy_h}x{policy_w}",
            flush=True,
        )
    print(
        f"[INFO] Student sim: nav_dt={nav_dt:.3f}s ({1.0 / nav_dt:.1f}Hz), "
        f"cam_update={cam_update_period:.3f}s, depth_mode={depth_mode}, "
        f"sim_camera={cam_h}x{cam_w}, policy_img={policy_h}x{policy_w}, "
        f"depth_only={args_cli.depth_only}, "
        f"depth_max={args_cli.depth_max}, camera_far_clip={camera_far_clip}, "
        f"tiled_cameras={args_cli.tiled_cameras}, "
        f"num_envs={args_cli.num_envs}, env_spacing={scene_env_spacing}, inner_steps={args_cli.inner_steps}",
        flush=True,
    )

    agent_cfg = TaskDStudentPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iter
    agent_cfg.num_steps_per_env = args_cli.steps_per_env
    agent_cfg.policy.img_h = policy_h
    agent_cfg.policy.img_w = policy_w
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
        image_h=policy_h,
        image_w=policy_w,
        depth_max=args_cli.depth_max,
        depth_only=args_cli.depth_only,
        nav_log_interval=args_cli.nav_log_interval,
        push_box_drop_com_z=args_cli.push_box_drop_com_z,
        push_min_box_nominal_x=args_cli.push_min_box_nominal_x,
        push_right_reward_dist=args_cli.push_right_reward_dist,
    )
    vec_env = NavRslRlVecEnvWrapper(nav_env)
    if args_cli.debug_env_origins:
        _print_env_origins_debug(env, env_spacing=scene_env_spacing, show=16)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[INFO] Resuming PPO from: {args_cli.resume}", flush=True)
        _load_ppo_checkpoint(
            runner,
            args_cli.resume,
            load_optimizer=not args_cli.resume_no_optimizer,
        )
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
    _dump_deploy_agent_yaml(
        log_dir,
        policy_h=policy_h,
        policy_w=policy_w,
        depth_only=args_cli.depth_only,
        depth_max=args_cli.depth_max,
        platform=depth_mode == "platform",
        platform_h=int(args_cli.platform_depth_h),
        platform_w=int(args_cli.platform_depth_w),
    )

    print(f"[INFO] Logging to: {log_dir}", flush=True)
    if args_cli.ppo_no_train:
        if not args_cli.no_train_ckpt and not args_cli.resume:
            print(
                "[WARN] --ppo_no_train without --no_train_ckpt (or --resume): "
                "nav policy weights are random. Pass --no_train_ckpt path/to/model_XXX.pt",
                flush=True,
            )
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

