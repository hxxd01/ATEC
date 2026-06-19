"""Shared Task D teleop trajectory load/replay helpers (used by replay_teleop_traj + pit solution play)."""

from __future__ import annotations

import json
import time
from typing import Any

import torch


def load_teleop_trajectory(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "samples" not in data or not isinstance(data["samples"], list):
        raise ValueError(f"Invalid teleop trajectory (missing samples): {path}")
    return data


class TrajectoryReplayer:
    """Replay recorded (vx, vy, wz) commands with an optional pre-play delay."""

    def __init__(self, samples: list[dict], *, replay_delay_s: float = 2.0) -> None:
        if not samples:
            raise ValueError("Trajectory has no samples.")
        self.samples = sorted(samples, key=lambda s: float(s.get("t", 0.0)))
        self.replay_delay_s = float(replay_delay_s)
        self.num_samples = len(self.samples)
        t0 = float(self.samples[0].get("t", 0.0))
        t1 = float(self.samples[-1].get("t", t0))
        self.duration = max(0.0, t1 - t0)
        self.total_duration = self.replay_delay_s + self.duration

    def _traj_time(self, sim_time: float) -> float:
        return float(sim_time) - self.replay_delay_s

    def _sample_index_for_traj_time(self, traj_time: float) -> int:
        idx = 0
        for i, sample in enumerate(self.samples):
            if float(sample.get("t", 0.0)) <= traj_time + 1e-9:
                idx = i
            else:
                break
        return idx

    def cmd_at(self, sim_time: float) -> tuple[float, float, float]:
        if sim_time < self.replay_delay_s - 1e-9:
            return 0.0, 0.0, 0.0
        traj_time = self._traj_time(sim_time)
        if traj_time > self.duration + 1e-6:
            return 0.0, 0.0, 0.0
        sample = self.samples[self._sample_index_for_traj_time(traj_time)]
        cmd = sample.get("cmd", [0.0, 0.0, 0.0])
        return float(cmd[0]), float(cmd[1]), float(cmd[2])

    def sample_index_at(self, sim_time: float) -> int:
        if sim_time < self.replay_delay_s - 1e-9:
            return -1
        return self._sample_index_for_traj_time(self._traj_time(sim_time))


def read_robot_spawn_w(env) -> list[float]:
    robot = env.unwrapped.scene["robot"]
    pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def read_scene_snapshot(env) -> tuple[list[float] | None, None, list[float] | None]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    robot_pos = robot.data.root_pos_w[0].detach().cpu().tolist()[:3]
    robot_pos = [float(v) for v in robot_pos]
    try:
        box = unwrapped.scene["box"]
        box_pos = box.data.root_pos_w[0].detach().cpu().tolist()[:3]
        box_pos = [float(v) for v in box_pos]
    except KeyError:
        box_pos = None
    return robot_pos, None, box_pos


class SpawnPointMarker:
    """Lightweight marker hook (full USD marker optional; logs in headless)."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def mark(self, spawn_w: list[float] | tuple[float, ...]) -> None:
        xyz = tuple(float(v) for v in spawn_w[:3])
        print(f"[replay] spawn marker at ({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})", flush=True)


def run_teleop_replay(
    env,
    controller,
    replayer: TrajectoryReplayer,
    *,
    simulation_app,
    real_time: bool = False,
    compare: bool = False,
    headless: bool = True,
    device: str | None = None,
    obs=None,
) -> tuple[Any, float, bool]:
    """Play trajectory in-place; returns (last_obs, total_sim_time, finished_ok)."""
    if obs is None:
        obs, _ = env.reset()
    controller.reset()

    dt = getattr(env.unwrapped, "step_dt", None)
    total_reward = 0.0
    total_time = 0.0
    timestep = 0
    last_compared_idx = -1
    replay_done = False
    playback_started = False
    finished_ok = False
    dev = device or getattr(controller, "device", "cuda")
    camera_follow = None
    if not headless:
        try:
            from rl_utils import camera_follow as _camera_follow

            camera_follow = _camera_follow
        except Exception:
            camera_follow = None

    while simulation_app.is_running():
        start = time.time()
        vx, vy, wz = replayer.cmd_at(total_time)
        if not playback_started and total_time >= replayer.replay_delay_s - 1e-9:
            playback_started = True
            print(f"[replay] playback started at sim t={total_time:.2f}s", flush=True)
        if total_time > replayer.total_duration + 1e-6:
            vx, vy, wz = 0.0, 0.0, 0.0
            replay_done = True

        controller.set_velocity_command(vx, vy, wz)
        with torch.inference_mode():
            resp = controller.predicts(obs, total_reward)
            actions = torch.tensor(resp["action"], dtype=torch.float32, device=dev).view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)

        if camera_follow is not None:
            camera_follow(env)

        sim_dt = info.get("Step_dt", dt) if isinstance(info, dict) else dt
        if sim_dt is not None:
            if isinstance(reward, torch.Tensor):
                total_reward += reward.mean().item() / float(sim_dt)
            else:
                total_reward += float(reward) / float(sim_dt)

        if isinstance(info, dict) and "Elapsed_Time" in info:
            elapsed = info["Elapsed_Time"]
            total_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
        elif dt is not None:
            total_time += dt

        if compare:
            idx = replayer.sample_index_at(total_time)
            if idx >= 0 and idx != last_compared_idx:
                ref = replayer.samples[idx]
                robot_pos, _, box_pos = read_scene_snapshot(env)
                print(
                    f"[replay] compare t={ref.get('t', 0.0):.2f} idx={idx} "
                    f"robot={robot_pos} ref={ref.get('robot_pos_w')} "
                    f"box={box_pos} ref={ref.get('box_pos_w')} score_ref={ref.get('score')}",
                    flush=True,
                )
                last_compared_idx = idx

        timestep += 1
        done = bool(terminated.item() if hasattr(terminated, "item") else terminated) or bool(
            truncated.item() if hasattr(truncated, "item") else truncated
        )
        if done:
            print(f"[replay] episode done at step={timestep}, score={total_reward:.2f}", flush=True)
            break
        if replay_done and total_time >= replayer.total_duration + 0.5:
            print(
                f"[replay] trajectory finished at t={total_time:.2f}s, score={total_reward:.2f}",
                flush=True,
            )
            finished_ok = True
            break

        if real_time and dt is not None:
            sleep_time = float(dt) - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    return obs, total_time, finished_ok
