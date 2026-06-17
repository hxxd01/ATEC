"""Verify pit_cross_success termination geometry in sim (teleport sanity check)."""

import argparse
import copy
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ATEC_RL_LAB_SRC = os.path.join(_REPO_ROOT, "source", "atec_rl_lab")
if os.path.isdir(_ATEC_RL_LAB_SRC) and _ATEC_RL_LAB_SRC not in sys.path:
    sys.path.insert(0, _ATEC_RL_LAB_SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug pit_cross_success termination.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import atec_rl_lab.tasks  # noqa: F401
from atec_rl_lab.tasks.task_d.locomotion.env_cfg import UnitreeB2PiperTaskDPitLocomotionEnvCfg
from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import (
    log_pit_threshold_reference,
    pit_cross_world_x,
    pit_geometry_for_envs,
    pit_success_local_x,
)
from atec_rl_lab.tasks.task_d.locomotion.terrain_curriculum import TaskDPitTerrainGenerator
from atec_rl_lab.tasks.task_d.terrain import TASK_D_TERRAIN_CFG, PitAndPlatformTerrainCfg, TaskDTerrainImporter
from atec_rl_lab.tasks.task_d.terrain import configure_task_d_terrain_for_num_envs


def _build_fixed_terrain(env_cfg, num_envs: int):
    terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
    terrain_cfg.class_type = TaskDTerrainImporter
    terrain_cfg.terrain_generator.class_type = TaskDPitTerrainGenerator
    configure_task_d_terrain_for_num_envs(terrain_cfg, num_envs, curriculum_levels=None)
    terrain_cfg.terrain_generator.curriculum = False
    pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
    if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
        w = float(env_cfg.pit_width_range[0])
        pit_cfg.pit_width_range = (w, w)
        pit_cfg.platform_height_range = env_cfg.platform_height_range
    return terrain_cfg


def main():
    env_cfg = UnitreeB2PiperTaskDPitLocomotionEnvCfg()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.terrain = _build_fixed_terrain(env_cfg, env_cfg.scene.num_envs)
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.pit_width_levels = None

    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg)
    unwrapped = env.unwrapped

    # Warm up one reset.
    obs, _ = env.reset()
    del obs

    robot = unwrapped.scene["robot"]
    origins = unwrapped.scene.env_origins
    cross_x = pit_cross_world_x(unwrapped)
    success_local = pit_success_local_x(unwrapped)
    success_x = origins[:, 0] + success_local

    center_w, size_lwh = pit_geometry_for_envs(unwrapped)
    local_x = robot.data.root_pos_w[:, 0] - origins[:, 0]

    log_pit_threshold_reference(unwrapped)

    print("[DebugPitCross] env_origins[0]:", origins[0].tolist())
    print("[DebugPitCross] spawn root_pos_w[0]:", robot.data.root_pos_w[0].tolist())
    print("[DebugPitCross] spawn local_x[0]:", local_x[0].item())
    print("[DebugPitCross] pit center_w[0]:", center_w[0].tolist())
    print("[DebugPitCross] pit width[0]:", size_lwh[0, 1].item())
    print("[DebugPitCross] cross_x[0]:", cross_x[0].item())
    print("[DebugPitCross] success local_x[0]:", success_local[0].item())
    print("[DebugPitCross] success world_x[0]:", success_x[0].item())

    term_fn = unwrapped.termination_manager.get_term
    before = term_fn("pit_cross_success").clone()

    # Teleport env 0 past success line, keep standing height.
    env_ids = torch.tensor([0], device=unwrapped.device, dtype=torch.long)
    pose = robot.data.root_pos_w[0].clone()
    pose[0] = success_x[0] + 0.05
    pose[2] = 0.8
    quat = robot.data.root_quat_w[0]
    robot.write_root_pose_to_sim(torch.cat([pose, quat]).unsqueeze(0), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=unwrapped.device), env_ids=env_ids)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.step(render=False)
    unwrapped.scene.update(dt=unwrapped.physics_dt)

    after_tp = term_fn("pit_cross_success")
    print("[DebugPitCross] pit_cross_success after teleport (pre-step):", after_tp[0].item())

    action = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
    obs, rew, terminated, truncated, info = env.step(action)
    term = unwrapped.termination_manager.get_term("pit_cross_success")
    print("[DebugPitCross] pit_cross_success after step:", term[0].item())
    print("[DebugPitCross] terminated:", terminated[0].item(), "truncated:", truncated[0].item())
    if "log" in info:
        for k, v in sorted(info["log"].items()):
            if "Termination" in k:
                print(f"  {k}: {v}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
