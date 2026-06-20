"""Task D platform competition score (RewardCrossX + RewardBoxXInRange milestones)."""

from __future__ import annotations

import torch

from .env_origin import task_d_env_origin_xy, task_d_nominal_x_to_world

# Defaults match TaskDEnvCfg.rewards (achieve + box_in_target_x).
DEFAULT_CROSS_THRESHOLDS = (-1.4, 2.0)
DEFAULT_CROSS_VALUES = (2.0, 20.0)
DEFAULT_BOX_X_RANGES = ((-0.7, 0.7), (-1.4, -0.7))
DEFAULT_BOX_VALUE = 14.0


class TaskDPlatformScoreTracker:
    """Track one-time Task D platform milestone score during play/deploy."""

    def __init__(
        self,
        *,
        cross_thresholds: tuple[float, ...] = DEFAULT_CROSS_THRESHOLDS,
        cross_values: tuple[float, ...] = DEFAULT_CROSS_VALUES,
        box_x_ranges: tuple[tuple[float, float], ...] = DEFAULT_BOX_X_RANGES,
        box_value: float = DEFAULT_BOX_VALUE,
    ) -> None:
        if len(cross_thresholds) != len(cross_values):
            raise ValueError("cross_thresholds and cross_values must have the same length.")
        self._cross_thresholds = tuple(float(x) for x in cross_thresholds)
        self._cross_values = tuple(float(x) for x in cross_values)
        self._box_x_ranges = tuple((float(a), float(b)) for a, b in box_x_ranges)
        self._box_value = float(box_value)
        self.reset()

    def reset(self) -> None:
        self.score = 0.0
        self._cross_given = [False] * len(self._cross_thresholds)
        self._box_given = [False] * len(self._box_x_ranges)

    @property
    def cross_given(self) -> list[bool]:
        return list(self._cross_given)

    @property
    def box_given(self) -> list[bool]:
        return list(self._box_given)

    def breakdown_str(self) -> str:
        cross_pts = sum(v for g, v in zip(self._cross_given, self._cross_values) if g)
        box_pts = sum(self._box_value for g in self._box_given if g)
        return (
            f"cross={cross_pts:.0f} ({self._cross_given}) "
            f"box={box_pts:.0f} ({self._box_given}) total={self.score:.1f}"
        )

    def update(self, unwrapped, *, with_box: bool = True) -> float:
        robot = unwrapped.scene["robot"]
        root_x = robot.data.root_pos_w[0, 0]
        env_origin_x, _ = task_d_env_origin_xy(unwrapped)
        ox = env_origin_x[0]

        for i, (nominal_th, value) in enumerate(zip(self._cross_thresholds, self._cross_values)):
            th_world = task_d_nominal_x_to_world(
                torch.tensor(float(nominal_th), device=root_x.device),
                ox.unsqueeze(0),
            )[0]
            if root_x > th_world and not self._cross_given[i]:
                self._cross_given[i] = True
                self.score += float(value)

        if with_box:
            try:
                box = unwrapped.scene["box"]
            except KeyError:
                box = None
            if box is not None:
                box_x = box.data.root_pos_w[0, 0]
                for i, (x_min, x_max) in enumerate(self._box_x_ranges):
                    mn_w = task_d_nominal_x_to_world(
                        torch.tensor(float(x_min), device=box_x.device),
                        ox.unsqueeze(0),
                    )[0]
                    mx_w = task_d_nominal_x_to_world(
                        torch.tensor(float(x_max), device=box_x.device),
                        ox.unsqueeze(0),
                    )[0]
                    lo = torch.minimum(mn_w, mx_w)
                    hi = torch.maximum(mn_w, mx_w)
                    if box_x >= lo and box_x <= hi and not self._box_given[i]:
                        self._box_given[i] = True
                        self.score += self._box_value

        return self.score


def scene_has_task_d_box(unwrapped) -> bool:
    try:
        unwrapped.scene["box"]
    except KeyError:
        return False
    return True
