"""Task B MDP events (object randomization on reset)."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedEnv

TASK_B_NUM_OBJECTS = 18
TASK_B_SUGAR_QUAT = (0.0, 0.707, 0.0, 0.707)
TASK_B_OTHER_QUAT = (0.0, 0.0, -0.707, 0.707)
TASK_B_SUGAR_Z = 0.15
TASK_B_OTHER_Z = 0.10


def randomize_task_b_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None = None,
    x_range: tuple[float, float] = (-15.0, -5.0),
    y_range: tuple[float, float] = (-15.0, -5.0),
    pocket_radius: float = 1.0,
    pocket_shift: float = 2.0,
) -> None:
    """Randomize trash object poses on reset (same distribution as Task B test cfg)."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    n = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])

    for obj_idx in range(1, TASK_B_NUM_OBJECTS + 1):
        obj: RigidObject = env.scene[f"object_{obj_idx}"]
        x = torch.rand(n, device=env.device) * (x_max - x_min) + x_min
        y = torch.rand(n, device=env.device) * (y_max - y_min) + y_min
        pocket = (x.abs() < pocket_radius) & (y.abs() < pocket_radius)
        x = torch.where(pocket, x + pocket_shift, x)
        z = TASK_B_SUGAR_Z if obj_idx <= 6 else TASK_B_OTHER_Z
        local = torch.stack([x, y, torch.full_like(x, z)], dim=-1)
        positions = env_origins + local

        if obj_idx <= 6:
            quat = torch.tensor(TASK_B_SUGAR_QUAT, device=env.device, dtype=torch.float32).view(1, 4)
        else:
            quat = torch.tensor(TASK_B_OTHER_QUAT, device=env.device, dtype=torch.float32).view(1, 4)
        quat = quat.expand(n, 4)

        root_states = obj.data.default_root_state[env_ids].clone()
        root_states[:, 0:3] = positions
        root_states[:, 3:7] = quat
        root_states[:, 7:13] = 0.0
        obj.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
        obj.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)
