"""Load / replay Task D teleop trajectory JSON (platform-local, no scripts/ deps)."""

from __future__ import annotations

import json
from typing import Any


def load_teleop_trajectory(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "samples" not in data or not isinstance(data["samples"], list):
        raise ValueError(f"Invalid teleop trajectory (missing samples): {path}")
    return data


class TrajectoryReplayer:
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

    def cmd_at(self, sim_time: float) -> tuple[float, float, float]:
        if sim_time < self.replay_delay_s - 1e-9:
            return 0.0, 0.0, 0.0
        traj_time = float(sim_time) - self.replay_delay_s
        if traj_time > self.duration + 1e-6:
            return 0.0, 0.0, 0.0
        idx = 0
        for i, sample in enumerate(self.samples):
            if float(sample.get("t", 0.0)) <= traj_time + 1e-9:
                idx = i
            else:
                break
        cmd = self.samples[idx].get("cmd", [0.0, 0.0, 0.0])
        return float(cmd[0]), float(cmd[1]), float(cmd[2])
