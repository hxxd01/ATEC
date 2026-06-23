"""Task E grasp IK helpers (same pipeline as scripts/act/collect_demos_task_e.py).

Grasp-only subset: PRE_GRASP → REACH → CLOSE → LIFT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ACT_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "act"
if str(_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACT_ROOT))

from task_e.config import (  # noqa: E402
    ACTION_SCALE,
    CARRY_Z,
    DEFAULT_PLACE_QUAT_W,
    GRASP_Z_OFFSET,
    GRIPPER_CLOSE_POS,
    GRIPPER_OPEN_POS,
    STEPS,
)
from task_e.state_machine import compute_grasp_quat  # noqa: E402

GRASP_STATE_ORDER = ("PRE_GRASP", "REACH", "CLOSE", "LIFT")

__all__ = [
    "ACTION_SCALE",
    "GRIPPER_CLOSE_POS",
    "GRIPPER_OPEN_POS",
    "GRASP_STATE_ORDER",
    "STEPS",
    "TaskEGraspOnlySM",
    "compute_grasp_quat",
    "gripper_tensor",
]


def gripper_tensor(cmd: str, device: str, dtype) -> torch.Tensor:
    vals = GRIPPER_OPEN_POS if cmd == "open" else GRIPPER_CLOSE_POS
    return torch.tensor([list(vals)], device=device, dtype=dtype)


class TaskEGraspOnlySM:
    """Same tick logic as Task E PickPlaceStateMachine, grasp states only."""

    def __init__(
        self,
        grasp_quat: torch.Tensor,
        device: str,
        *,
        carry_z: float | None = None,
        grasp_z_offset: float | None = None,
    ):
        self._device = device
        self._grasp_quat = grasp_quat
        self._carry_z = float(carry_z if carry_z is not None else CARRY_Z)
        self._grasp_z_offset = float(grasp_z_offset if grasp_z_offset is not None else GRASP_Z_OFFSET)
        self._state_idx = 0
        self._count = 0
        self._cached_obj_pos: torch.Tensor | None = None
        self.done = False

    @property
    def state(self) -> str:
        return GRASP_STATE_ORDER[self._state_idx]

    def tick(self, obj_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        s = self.state
        d = self._device

        if s == "PRE_GRASP" and self._count == 0:
            self._cached_obj_pos = obj_pos.clone()
            if self._carry_z <= 0.0:
                self._carry_z = float(obj_pos[2].item()) + 0.40
        if s in ("REACH", "CLOSE") and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos

        ee_pos, gripper = self._target_pos_gripper(s, obj_pos, d)
        ee_quat = self._target_quat(s, d)

        self._count += 1
        if self._count >= STEPS[s]:
            self._count = 0
            if s == "LIFT":
                self.done = True
            else:
                self._state_idx += 1

        return ee_pos, ee_quat, gripper

    def _target_pos_gripper(
        self, s: str, obj_pos: torch.Tensor, d: str
    ) -> tuple[torch.Tensor, str]:
        if s == "PRE_GRASP":
            p = obj_pos.clone()
            p[2] = self._carry_z
            return p, "open"
        if s == "REACH":
            p = obj_pos.clone()
            p[2] += self._grasp_z_offset
            return p, "open"
        if s == "CLOSE":
            p = obj_pos.clone()
            p[2] += self._grasp_z_offset
            return p, "close"
        if s == "LIFT":
            p = obj_pos.clone()
            p[2] = self._carry_z
            return p, "close"
        raise ValueError(f"Unknown grasp state: {s}")

    def _target_quat(self, s: str, d: str) -> torch.Tensor:
        default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d)
        if s in ("REACH", "CLOSE", "LIFT"):
            return self._grasp_quat
        return default_quat
