import argparse
import os
import pickle
import sys
import time
from datetime import datetime

# Repo root contains package `demo/`; running `python scripts/...` only puts scripts/ on sys.path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from isaaclab.app import AppLauncher
from demo.solution import AlgSolution


parser = argparse.ArgumentParser(description="Collect Task-D trajectories with scripted policy.")
parser.add_argument("--task", type=str, default="ATEC-TaskD-B2Piper", help="Task name.")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel env count.")
parser.add_argument(
    "--total_steps",
    type=int,
    default=30000,
    help="Total simulation steps to collect (all envs run in parallel until this cap).",
)
parser.add_argument(
    "--log_every_steps",
    type=int,
    default=500,
    help="Print avg score of all finished episodes every N sim steps.",
)
parser.add_argument(
    "--collect_hz",
    type=float,
    default=10.0,
    help="Trajectory sampling frequency in Hz (downsampled from env stepping rate).",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Output dir for pkl trajectory files. Defaults to logs/datasets/<task>/<timestamp>.",
)
parser.add_argument(
    "--store_images",
    action="store_true",
    default=False,
    help="Store obs['image'] payloads in dataset (very large files).",
)
parser.add_argument(
    "--image_every",
    type=int,
    default=1,
    help="Store images every K collected samples (only when --store_images).",
)
parser.add_argument(
    "--store_uint8_images",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Convert image tensors to uint8 before storing (use --no-store-uint8-images for float depth).",
)
parser.add_argument(
    "--disable_lidar",
    action="store_true",
    default=True,
    help="Disable LiDAR sensor and extero observations during collection (default: on).",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _to_uint8_batch(image_arr: np.ndarray) -> np.ndarray:
    arr = image_arr
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    max_v = float(arr.max()) if arr.size > 0 else 1.0
    min_v = float(arr.min()) if arr.size > 0 else 0.0
    if max_v <= 1.5 and min_v >= 0.0:
        arr = arr * 255.0
    elif max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v) * 255.0
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def _extract_obs_features(obs: dict) -> dict:
    proprio = obs["proprio"]
    if not isinstance(proprio, torch.Tensor):
        proprio = torch.as_tensor(proprio)
    # [base_lin_vel(3), base_ang_vel(3), velocity_commands(3), projected_gravity(3), ...]
    return {
        "base_lin_vel": proprio[:, 0:3].detach().cpu().numpy().astype(np.float32),
        "base_ang_vel": proprio[:, 3:6].detach().cpu().numpy().astype(np.float32),
        "projected_gravity": proprio[:, 9:12].detach().cpu().numpy().astype(np.float32),
    }


def _extract_images(obs: dict, store_uint8: bool) -> dict:
    image_dict = obs.get("image", None)
    if not isinstance(image_dict, dict):
        return {}
    out = {}
    for key, value in image_dict.items():
        arr = _to_numpy(value)
        if store_uint8:
            arr = _to_uint8_batch(arr)
        out[key] = arr
    return out


def _safe_bool_tensor(v, num_envs: int, device) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=torch.bool).view(-1)
    return torch.full((num_envs,), bool(v), device=device, dtype=torch.bool)


def _resolve_stage_batch(solution, num_envs: int) -> np.ndarray:
    stage_batch = getattr(solution, "get_stage_batch", lambda: None)()
    if stage_batch is not None:
        return np.asarray(stage_batch, dtype=np.int32)
    nav_idx = getattr(solution, "_nav_step_idx", None)
    if isinstance(nav_idx, torch.Tensor):
        return nav_idx.detach().cpu().numpy().astype(np.int32)
    return np.zeros((num_envs,), dtype=np.int32)


def _read_done_from_env(
    env,
    num_envs: int,
    device,
    terminated,
    truncated,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read done flags from env buffers (authoritative) with fallback to step() returns."""
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    term_t = _safe_bool_tensor(terminated, num_envs, device)
    trunc_t = _safe_bool_tensor(truncated, num_envs, device)
    if hasattr(unwrapped, "reset_terminated"):
        term_t = _safe_bool_tensor(unwrapped.reset_terminated, num_envs, device)
    if hasattr(unwrapped, "reset_time_outs"):
        trunc_t = _safe_bool_tensor(unwrapped.reset_time_outs, num_envs, device)
    if hasattr(unwrapped, "reset_buf"):
        done_t = _safe_bool_tensor(unwrapped.reset_buf, num_envs, device)
    else:
        done_t = term_t | trunc_t
    return term_t, trunc_t, done_t


def _termination_thresholds(env) -> tuple[float, float]:
    fall_thresh, x_thresh = 0.25, 3.5
    try:
        cfg = env.unwrapped.cfg if hasattr(env, "unwrapped") else env.cfg
        terms = getattr(cfg, "terminations", None)
        if terms is not None:
            fall_cfg = getattr(terms, "fall", None)
            x_cfg = getattr(terms, "x_reached", None)
            if fall_cfg is not None and isinstance(getattr(fall_cfg, "params", None), dict):
                fall_thresh = float(fall_cfg.params.get("minimum_height", fall_thresh))
            if x_cfg is not None and isinstance(getattr(x_cfg, "params", None), dict):
                x_thresh = float(x_cfg.params.get("x_threshold", x_thresh))
    except Exception:
        pass
    return fall_thresh, x_thresh


def _termination_term_flags(env, num_envs: int, device) -> dict[str, torch.Tensor]:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    tm = getattr(unwrapped, "termination_manager", None)
    if tm is None:
        return {}
    out: dict[str, torch.Tensor] = {}
    for name in ("fall", "x_reached", "time_out"):
        try:
            out[name] = tm.get_term(name).view(-1)[:num_envs].to(device=device, dtype=torch.bool)
        except Exception:
            pass
    return out


def _termination_reasons(env, env_idx: int) -> str:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    tm = getattr(unwrapped, "termination_manager", None)
    if tm is None:
        return ""
    try:
        terms = tm.get_active_iterable_terms(env_idx)
        fired = [name for name, val in terms if val and float(val[0]) > 0.5]
        return ",".join(fired) if fired else "none"
    except Exception:
        return ""


def _robot_xyz(solution, env_idx: int = 0) -> tuple[float, float, float] | None:
    robot = getattr(solution, "_robot", None)
    if robot is None:
        return None
    pos = robot.data.root_pos_w[env_idx]
    return float(pos[0].item()), float(pos[1].item()), float(pos[2].item())


def _infer_done_from_pose(
    solution,
    fall_thresh: float,
    x_thresh: float,
    num_envs: int,
    device,
) -> dict[str, torch.Tensor]:
    robot = getattr(solution, "_robot", None)
    if robot is None:
        z = torch.zeros(num_envs, device=device, dtype=torch.bool)
        x = torch.zeros(num_envs, device=device, dtype=torch.bool)
        return {"infer_fall": z, "infer_x_reached": x}
    pos = robot.data.root_pos_w[:num_envs]
    return {
        "infer_fall": pos[:, 2] < fall_thresh,
        "infer_x_reached": pos[:, 0] > x_thresh,
    }


def _robot_xy(solution) -> tuple[float, float] | None:
    robot = getattr(solution, "_robot", None)
    if robot is None:
        return None
    pos = robot.data.root_pos_w
    return float(pos[0, 0].item()), float(pos[0, 1].item())


def _episode_lengths(env, num_envs: int) -> torch.Tensor | None:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    buf = getattr(unwrapped, "episode_length_buf", None)
    if not isinstance(buf, torch.Tensor):
        return None
    return buf.view(-1)[:num_envs]


def _print_score_stats(
    step: int,
    total_steps: int,
    completed_scores: list[float],
    completed_scores_valid: list[float],
    episode_score: torch.Tensor,
    ep_lengths: torch.Tensor | None,
    max_episode_length: int | None,
    done_this_step: int,
    num_envs: int,
    collected_idx: int,
    elapsed: float,
    robot_diag: str = "",
) -> None:
    n_done = len(completed_scores)
    running_mean = float(episode_score.mean().item())
    running_max = float(episode_score.max().item())
    if n_done > 0:
        done_avg = float(np.mean(completed_scores))
        done_min = float(np.min(completed_scores))
        done_max = float(np.max(completed_scores))
        done_str = f"finished_avg={done_avg:.3f} (min={done_min:.3f} max={done_max:.3f} n={n_done})"
    else:
        done_str = "finished_avg=n/a (no episode ended yet)"
    eps_per_env = n_done / max(num_envs, 1)
    n_valid = len(completed_scores_valid)
    valid_per_env = n_valid / max(num_envs, 1)
    valid_str = f" valid_episodes={n_valid} ({valid_per_env:.2f}/env)"
    if n_valid > 0:
        valid_avg = float(np.mean(completed_scores_valid))
        valid_min = float(np.min(completed_scores_valid))
        valid_max = float(np.max(completed_scores_valid))
        valid_str += f" valid_avg={valid_avg:.3f} (min={valid_min:.3f} max={valid_max:.3f})"
    ep_len_str = ""
    if ep_lengths is not None and ep_lengths.numel() > 0:
        ep_len_str = (
            f" ep_len={int(ep_lengths.min().item())}-{int(ep_lengths.max().item())}"
        )
        if max_episode_length is not None:
            ep_len_str += f"/{max_episode_length}"
    print(
        f"[collect] step={step:6d}/{total_steps} done_episodes={n_done} "
        f"({eps_per_env:.2f}/env) done_now={done_this_step} "
        f"running_score={running_mean:.3f} (max={running_max:.3f}) {done_str}{valid_str}"
        f"{ep_len_str}{robot_diag} samples={collected_idx} elapsed={elapsed:.1f}s",
        flush=True,
    )


def _log_episode_done(
    step: int,
    done_mask: torch.Tensor,
    episode_score: torch.Tensor,
    term_t: torch.Tensor,
    trunc_t: torch.Tensor,
    solution,
    env,
) -> None:
    idxs = done_mask.nonzero(as_tuple=False).view(-1).tolist()
    if not idxs:
        return
    for idx in idxs:
        score = float(episode_score[idx].item())
        term = bool(term_t[idx].item())
        trunc = bool(trunc_t[idx].item())
        reasons = _termination_reasons(env, idx)
        reason = "timeout" if trunc else ("terminated" if term else "unknown")
        xyz = _robot_xyz(solution, idx)
        pos_str = f" pos=({xyz[0]:+.2f},{xyz[1]:+.2f},{xyz[2]:+.2f})" if xyz else ""
        print(
            f"[collect] episode_end step={step} env={idx} score={score:.3f} "
            f"reason={reason} terms=[{reasons}] term={int(term)} trunc={int(trunc)}{pos_str}",
            flush=True,
        )


def _log_env_reset(
    solution,
    done_mask: torch.Tensor,
    *,
    spawn_xyz: tuple[float, float, float] = (-3.0, 0.0, 0.8),
    spawn_tol: float = 0.75,
) -> None:
    if not bool(done_mask.any()):
        return
    robot = getattr(solution, "_robot", None)
    if robot is None:
        return
    idxs = done_mask.nonzero(as_tuple=False).view(-1).tolist()
    for idx in idxs:
        row = robot.data.root_pos_w[idx]
        x, y, z = float(row[0].item()), float(row[1].item()), float(row[2].item())
        dist = ((x - spawn_xyz[0]) ** 2 + (y - spawn_xyz[1]) ** 2 + (z - spawn_xyz[2]) ** 2) ** 0.5
        warn = " WARN: far from spawn — root reset may have failed" if dist > spawn_tol else ""
        print(
            f"[collect] env_reset env={idx} -> robot_pos=({x:+.2f},{y:+.2f},{z:+.2f}) "
            f"spawn_dist={dist:.2f}{warn}",
            flush=True,
        )


def main() -> None:
    solution = AlgSolution()
    if hasattr(solution, "set_device"):
        solution.set_device(args_cli.device)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    if args_cli.disable_lidar:
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "lidar_sensor"):
            env_cfg.scene.lidar_sensor = None
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "extero"):
            env_cfg.observations.extero = None
        print("[collect] LiDAR disabled (scene.lidar_sensor=None, observations.extero=None).", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if hasattr(solution, "bind_env") and "TaskD" in str(args_cli.task):
        solution.bind_env(env)

    if args_cli.out_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.abspath(os.path.join("logs", "datasets", args_cli.task, stamp))
    else:
        out_dir = os.path.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[collect] output dir: {out_dir}", flush=True)

    num_envs = int(args_cli.num_envs)
    total_steps = int(args_cli.total_steps)
    log_every = max(1, int(args_cli.log_every_steps))

    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    max_episode_length = int(getattr(unwrapped, "max_episode_length", 0)) or None
    max_episode_length_s = float(getattr(unwrapped, "max_episode_length_s", 0.0))
    if max_episode_length is not None:
        print(
            f"[collect] env max_episode_length={max_episode_length} "
            f"(~{max_episode_length_s:.0f}s) — timeout only after this many steps",
            flush=True,
        )
    fall_thresh, x_thresh = _termination_thresholds(env)
    print(
        f"[collect] termination thresholds: fall_z<{fall_thresh:.2f}, x_reached>{x_thresh:.2f}; "
        f"score=16 is partial reward (2+14), not episode success",
        flush=True,
    )

    obs, _ = env.reset()
    if hasattr(solution, "bind_env") and "TaskD" in str(args_cli.task):
        solution.bind_env(env)
    if hasattr(solution, "reset"):
        solution.reset(task=args_cli.task)
    if hasattr(solution, "bind_env") and "TaskD" in str(args_cli.task):
        solution.bind_env(env)

    data_file = os.path.join(out_dir, "trajectories.pkl")
    print(f"[collect] total_steps={total_steps} num_envs={num_envs} -> {data_file}", flush=True)

    dt = float(getattr(solution, "dt", 0.02))
    collect_hz = max(float(args_cli.collect_hz), 1e-6)
    sample_interval_steps = max(1, int(round((1.0 / collect_hz) / dt)))
    effective_hz = 1.0 / (sample_interval_steps * dt)
    print(
        f"[collect] sampling: target={collect_hz:.2f}Hz, dt={dt:.4f}s, "
        f"interval={sample_interval_steps} steps, effective={effective_hz:.2f}Hz",
        flush=True,
    )
    print(
        "[collect] store filter: only keep env samples with current episode_score > 0.",
        flush=True,
    )
    print(
        "[collect] pkl schema: meta(dict) -> step(dict, filtered by episode_score>0, with env_indices) -> summary(dict).",
        flush=True,
    )

    episode_score = torch.zeros((num_envs,), device=args_cli.device, dtype=torch.float32)
    prev_high_level_cmd = torch.zeros((num_envs, 3), device=args_cli.device, dtype=torch.float32)
    prev_done_t = torch.zeros((num_envs,), device=args_cli.device, dtype=torch.bool)
    completed_scores: list[float] = []
    completed_scores_valid: list[float] = []
    collected_idx = 0
    step = 0
    collect_start = time.time()
    next_log_step = log_every

    with open(data_file, "wb") as f:
        pickle.dump(
            {
                "type": "meta",
                "task": args_cli.task,
                "num_envs": num_envs,
                "total_steps": total_steps,
                "dt": dt,
                "collect_hz_target": collect_hz,
                "sample_interval_steps": sample_interval_steps,
                "collect_hz_effective": effective_hz,
                "log_every_steps": log_every,
                "max_episode_length": max_episode_length,
                "max_episode_length_s": max_episode_length_s,
                "feature_layout": {
                    "proprio_slice.base_lin_vel": [0, 3],
                    "proprio_slice.base_ang_vel": [3, 6],
                    "proprio_slice.projected_gravity": [9, 12],
                    "prev_high_level_cmd": "shape [num_envs, 3]",
                    "high_level_cmd": "body-frame [vx, vy, wz] fed to policy.pt",
                    "stage_idx": "shape [num_envs], per-env nav stage",
                },
                "store_images": bool(args_cli.store_images),
                "image_every": int(args_cli.image_every),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        while simulation_app.is_running() and step < total_steps:
            step += 1
            feats = _extract_obs_features(obs)

            if step == 1:
                print("[collect] first solution.predicts() ...", flush=True)
            resp = solution.predicts(obs, float(episode_score.mean().item()))
            stage_idx = _resolve_stage_batch(solution, num_envs)
            phase = str(getattr(solution, "phase", "unknown"))
            if bool(resp.get("giveup", False)):
                print("[collect] giveup=True from solution, stopping early.", flush=True)
                break

            action_tensor = resp.get("action_tensor", None)
            if isinstance(action_tensor, torch.Tensor):
                actions = action_tensor.to(device=args_cli.device, dtype=torch.float32)
            else:
                actions = torch.as_tensor(resp["action"], dtype=torch.float32, device=args_cli.device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            if actions.shape[0] != num_envs:
                raise RuntimeError(
                    f"Action batch mismatch: got {actions.shape[0]} actions for num_envs={num_envs}"
                )

            current_cmd = getattr(solution, "_last_high_level_cmd_batch", None)
            if isinstance(current_cmd, torch.Tensor):
                current_cmd = current_cmd.to(device=args_cli.device, dtype=torch.float32)
            else:
                current_cmd = torch.zeros((num_envs, 3), device=args_cli.device, dtype=torch.float32)

            next_obs, reward, terminated, truncated, info = env.step(actions)

            reward_t = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(reward, device=args_cli.device)
            reward_t = reward_t.view(-1).to(device=args_cli.device, dtype=torch.float32)

            sim_dt = info.get("Step_dt", 1.0) if isinstance(info, dict) else 1.0
            if isinstance(sim_dt, torch.Tensor):
                sim_dt = float(sim_dt.view(-1)[0].item())
            sim_dt = float(sim_dt)
            score_increment = reward_t / max(sim_dt, 1e-8)
            episode_score = episode_score + score_increment

            term_t, trunc_t, done_t = _read_done_from_env(
                env, num_envs, args_cli.device, terminated, truncated
            )
            term_flags = _termination_term_flags(env, num_envs, args_cli.device)
            infer = _infer_done_from_pose(solution, fall_thresh, x_thresh, num_envs, args_cli.device)

            if bool((infer["infer_fall"] & (~done_t)).any()):
                for idx in (infer["infer_fall"] & (~done_t)).nonzero(as_tuple=False).view(-1).tolist():
                    xyz = _robot_xyz(solution, idx)
                    z_str = f"{xyz[2]:+.3f}" if xyz else "?"
                    print(
                        f"[collect] WARN step={step} env={idx}: root_z={z_str} < {fall_thresh:.2f} "
                        f"but env done=0 — fall term may not be firing",
                        flush=True,
                    )

            newly_done = done_t & (~prev_done_t)
            if bool(newly_done.any()):
                _log_episode_done(
                    step=step,
                    done_mask=newly_done,
                    episode_score=episode_score,
                    term_t=term_t,
                    trunc_t=trunc_t,
                    solution=solution,
                    env=env,
                )
                finished = episode_score[newly_done].detach().cpu().numpy().astype(np.float64)
                completed_scores.extend(float(s) for s in finished.tolist())
                completed_scores_valid.extend(float(s) for s in finished.tolist() if s > 0.0)

            if bool(done_t.any()) and hasattr(solution, "reset_env_batch"):
                solution.reset_env_batch(done_t)
                setattr(solution, "_hl_cmd_force_refresh", True)

            if step % sample_interval_steps == 0:
                collected_idx += 1
                keep_mask_t = episode_score > 0.0
                keep_idx = keep_mask_t.nonzero(as_tuple=False).view(-1).detach().cpu().numpy().astype(np.int64)
                if keep_idx.size > 0:
                    stage_idx_np = np.asarray(stage_idx, dtype=np.int32)[keep_idx]
                    step_record = {
                        "type": "step",
                        "step": step,
                        "collect_index": collected_idx,
                        "env_indices": keep_idx,
                        "phase": phase,
                        "stage_idx": stage_idx_np,
                        "base_lin_vel": feats["base_lin_vel"][keep_idx],
                        "base_ang_vel": feats["base_ang_vel"][keep_idx],
                        "projected_gravity": feats["projected_gravity"][keep_idx],
                        "prev_high_level_cmd": prev_high_level_cmd.detach().cpu().numpy().astype(np.float32)[keep_idx],
                        "high_level_cmd": current_cmd.detach().cpu().numpy().astype(np.float32)[keep_idx],
                        "actions_low_level": actions.detach().cpu().numpy().astype(np.float32)[keep_idx],
                        "reward": reward_t.detach().cpu().numpy().astype(np.float32)[keep_idx],
                        "done": done_t.detach().cpu().numpy()[keep_idx],
                        "terminated": term_t.detach().cpu().numpy()[keep_idx],
                        "truncated": trunc_t.detach().cpu().numpy()[keep_idx],
                    }
                    if args_cli.store_images and (collected_idx % max(int(args_cli.image_every), 1) == 0):
                        imgs = _extract_images(obs, store_uint8=bool(args_cli.store_uint8_images))
                        step_record["images"] = {k: v[keep_idx] for k, v in imgs.items()}
                    pickle.dump(step_record, f, protocol=pickle.HIGHEST_PROTOCOL)

            episode_score[done_t] = 0.0
            prev_high_level_cmd = current_cmd
            obs = next_obs

            if bool(newly_done.any()):
                _log_env_reset(solution, newly_done)

            prev_done_t = done_t.clone()

            if step >= next_log_step or step == total_steps:
                robot_diag = ""
                xyz = _robot_xyz(solution, 0)
                if xyz is not None and num_envs == 1:
                    fall_flag = bool(infer["infer_fall"][0].item())
                    x_flag = bool(infer["infer_x_reached"][0].item())
                    term_parts = []
                    if "fall" in term_flags:
                        term_parts.append(f"fall={int(term_flags['fall'][0].item())}")
                    if "x_reached" in term_flags:
                        term_parts.append(f"x_reached={int(term_flags['x_reached'][0].item())}")
                    if "time_out" in term_flags:
                        term_parts.append(f"time_out={int(term_flags['time_out'][0].item())}")
                    term_str = " ".join(term_parts) if term_parts else "terms=n/a"
                    robot_diag = (
                        f" robot=({xyz[0]:+.2f},{xyz[1]:+.2f},z={xyz[2]:+.2f})"
                        f" infer_fall={int(fall_flag)} infer_x={int(x_flag)} {term_str}"
                    )
                _print_score_stats(
                    step=step,
                    total_steps=total_steps,
                    completed_scores=completed_scores,
                    completed_scores_valid=completed_scores_valid,
                    episode_score=episode_score,
                    ep_lengths=_episode_lengths(env, num_envs),
                    max_episode_length=max_episode_length,
                    done_this_step=int(newly_done.sum().item()),
                    num_envs=num_envs,
                    collected_idx=collected_idx,
                    elapsed=time.time() - collect_start,
                    robot_diag=robot_diag,
                )
                next_log_step = ((step // log_every) + 1) * log_every

        elapsed = time.time() - collect_start
        if len(completed_scores) > 0:
            final_avg = float(np.mean(completed_scores))
            final_min = float(np.min(completed_scores))
            final_max = float(np.max(completed_scores))
        else:
            final_avg = final_min = final_max = 0.0
        if len(completed_scores_valid) > 0:
            final_valid_avg = float(np.mean(completed_scores_valid))
            final_valid_min = float(np.min(completed_scores_valid))
            final_valid_max = float(np.max(completed_scores_valid))
        else:
            final_valid_avg = final_valid_min = final_valid_max = 0.0

        pickle.dump(
            {
                "type": "summary",
                "total_steps": step,
                "num_envs": num_envs,
                "completed_episodes": len(completed_scores),
                "completed_episodes_valid": len(completed_scores_valid),
                "avg_score": final_avg,
                "min_score": final_min,
                "max_score": final_max,
                "avg_score_valid": final_valid_avg,
                "min_score_valid": final_valid_min,
                "max_score_valid": final_valid_max,
                "samples": collected_idx,
                "elapsed_sec": elapsed,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    env.close()
    if len(completed_scores) == 0:
        print(
            "[collect] note: no env episode ended (need x_reached / fall / timeout). "
            f"Task D timeout is ~{max_episode_length} steps; total_steps was {step}.",
            flush=True,
        )
    print(
        f"[collect] done. steps={step} episodes={len(completed_scores)} valid_episodes={len(completed_scores_valid)} "
        f"finished_avg={final_avg:.3f} valid_avg={final_valid_avg:.3f} samples={collected_idx} file={data_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
