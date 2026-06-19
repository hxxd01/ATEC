"""Load a frozen MargActorCritic teacher checkpoint for pit DAgger / distillation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from atec_rl_lab.train.locomotion.marg.marg_actor_critic import MargActorCritic


def build_marg_teacher(obs_sample: dict, device: str) -> MargActorCritic:
    """Construct MargActorCritic matching pit MARG obs layout."""
    obs_groups = {
        "policy": ["proprio", "proprio_history", "height_map"],
        "critic": ["proprio", "height_map", "critic_priv"],
    }
    num_actions = 12
    teacher = MargActorCritic(obs_sample, obs_groups, num_actions)
    return teacher.to(device)


def load_marg_teacher_checkpoint(
    ckpt_path: str | Path,
    obs_sample: dict,
    device: str,
) -> MargActorCritic:
    """Load MargActorCritic weights from an OnPolicyRunner checkpoint."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")

    teacher = build_marg_teacher(obs_sample, device)
    loaded = torch.load(path, map_location=device, weights_only=False)
    sd = loaded.get("model_state_dict", loaded)
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    if missing:
        print(f"[MargTeacher] missing keys ({len(missing)}): {missing[:5]}", flush=True)
    if unexpected:
        print(f"[MargTeacher] unexpected keys ({len(unexpected)}): {unexpected[:5]}", flush=True)

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"[MargTeacher] loaded {path} (iter={loaded.get('iter', '?')})", flush=True)
    return teacher


def warmstart_depth_student_from_teacher(student: nn.Module, ckpt_path: str | Path) -> int:
    """Copy estimator only (actor MLP input differs: elevation vs depth CNN)."""
    path = Path(ckpt_path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    teacher_sd = loaded.get("model_state_dict", loaded)
    student_sd = student.state_dict()
    prefixes = ("estimator.",)
    loaded_n = 0
    for key, val in teacher_sd.items():
        if not any(key.startswith(p) for p in prefixes):
            continue
        if key not in student_sd:
            continue
        if tuple(student_sd[key].shape) != tuple(val.shape):
            continue
        student_sd[key] = val
        loaded_n += 1
    student.load_state_dict(student_sd, strict=False)
    print(
        f"[MargTeacher] warm-started {loaded_n} estimator tensors from {path} "
        "(actor/depth CNN left random — elevation vs depth input mismatch).",
        flush=True,
    )
    return loaded_n
