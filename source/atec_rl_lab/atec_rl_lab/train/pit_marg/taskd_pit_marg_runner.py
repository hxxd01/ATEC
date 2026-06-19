"""Shared Task D pit MARG locomotion train/play (same env rewards/terminations + MargActorCritic).

Used by ``scripts/train_taskd_pit_locomotion.py``, ``scripts/play_taskd_pit_locomotion.py``,
and ``scripts/train_nav_taskd_student.py --pit_marg_locomotion|--pit_marg_play``.
"""

from __future__ import annotations

import copy
import os
import time
from datetime import datetime
from typing import Any

import gymnasium as gym
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import atec_rl_lab.train  # noqa: F401
import atec_rl_lab.tasks.task_d.locomotion.mdp as task_d_loco_mdp
from atec_rl_lab.tasks.task_d.env_cfg import TASK_D_ROBOT_SPAWN_LOCAL
from atec_rl_lab.tasks.task_d.locomotion.mdp.events import task_d_pit_marg_spawn_local
from atec_rl_lab.tasks.task_d.locomotion.env_cfg import (
    UnitreeB2PiperTaskDPitLocomotionMargEnvCfg,
    refresh_task_d_pit_locomotion_terrain_cfg,
)
from atec_rl_lab.tasks.task_d.locomotion.pit_geometry import pit_width_from_level
from atec_rl_lab.tasks.task_d.locomotion.terrain_curriculum import TaskDPitTerrainGenerator
from atec_rl_lab.tasks.task_d.terrain import (
    TASK_D_TERRAIN_CFG,
    PitAndPlatformTerrainCfg,
    TaskDTerrainImporter,
    configure_task_d_terrain_for_num_envs,
)
from atec_rl_lab.train.locomotion.marg.marg_actor_critic import MargActorCritic
from atec_rl_lab.train.locomotion.marg.marg_ppo import MargPPO
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.agents.rsl_rl_ppo_cfg import (
    UnitreeB2PiperTaskDPitLocomotionMargPPORunnerCfg,
)

import rsl_rl.runners.on_policy_runner as _runner_mod


def register_marg_modules() -> None:
    """Register MargActorCritic / MargPPO with rsl_rl OnPolicyRunner."""
    _runner_mod.MargActorCritic = MargActorCritic
    _runner_mod.MargPPO = MargPPO


def _pit_width_for_level(level: int, width_range: tuple[float, float], curriculum_levels: int) -> float:
    max_level = max(0, int(curriculum_levels) - 1)
    level_t = torch.tensor([max(0, min(int(level), max_level))], dtype=torch.long)
    return float(pit_width_from_level(level_t, width_range, max_level).item())


def _apply_sim_easy_mode(env_cfg, args: Any) -> None:
    """Bias training toward fast pit-cross learning (simulation-only)."""
    env_cfg.disable_dr_and_obs_noise = True
    if hasattr(env_cfg, "_disable_dr_and_obs_noise"):
        env_cfg._disable_dr_and_obs_noise()

    vx = float(getattr(args, "pit_sim_easy_vx", getattr(args, "sim_easy_vx", 1.0)))
    env_cfg.command_lin_vel_x_min = vx
    env_cfg.command_lin_vel_x_max = vx
    env_cfg.command_curriculum_start_fraction = 1.0
    env_cfg.apply_command_config()

    env_cfg.pit_width_range = (float(env_cfg.pit_width_range[0]), float(env_cfg.pit_width_range[1]))

    env_cfg.marg_track_lin_vel_xy_weight = 1.8
    env_cfg.marg_track_ang_vel_z_weight = 0.0
    env_cfg.marg_collision_weight = -0.2
    env_cfg.marg_orientation_l2_weight = -0.08
    env_cfg.marg_joint_motion_limit_weight = -0.005
    env_cfg.marg_feet_stumble_weight = -0.3
    env_cfg.marg_feet_center_weight = -0.003
    env_cfg.marg_action_rate_l2_weight = -0.005
    env_cfg.marg_feet_air_time_weight = 0.0
    if hasattr(env_cfg, "_align_task_rewards"):
        env_cfg._align_task_rewards()

    env_cfg.rewards.forward_progress = RewTerm(
        func=task_d_loco_mdp.forward_world_x_progress,
        weight=8.0,
    )
    env_cfg.rewards.pit_cross_success = RewTerm(
        func=task_d_loco_mdp.PitCrossSuccessBonus,
        weight=1.0,
        params={
            "reward_value": 30.0,
            "post_cross_distance": float(env_cfg.pit_success_post_cross_distance),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    env_cfg.curriculum.command_levels_lin_vel = None


def build_train_env_cfg(args: Any):
    """Build MARG pit env cfg with CLI overrides (rewards/terminations from env_cfg defaults)."""
    env_cfg = UnitreeB2PiperTaskDPitLocomotionMargEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.pit_width_range = (float(_get(args, "pit_width_min", 0.4)), float(_get(args, "pit_width_max", 1.4)))
    env_cfg.pit_curriculum_levels = int(_get(args, "pit_curriculum_levels", 11))

    vx_min = float(_get(args, "command_vx_min", 0.0))
    vx_max = _get(args, "command_vx_max", None)
    if vx_max is None:
        vx_max = _get(args, "command_vx", 4.0)
    env_cfg.command_lin_vel_x_min = vx_min
    env_cfg.command_lin_vel_x_max = float(vx_max)
    env_cfg.command_curriculum_start_fraction = float(
        _get(args, "command_curriculum_start", _get(args, "command_curriculum_start_fraction", 0.1))
    )

    sim_easy = bool(_get(args, "pit_sim_easy", False) or _get(args, "sim_easy", False))
    if sim_easy:
        _apply_sim_easy_mode(env_cfg, args)
    env_cfg.apply_command_config()
    refresh_task_d_pit_locomotion_terrain_cfg(env_cfg)
    from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import (
        apply_pit_box_if_requested,
        apply_pit_dr_light,
        apply_pit_train_spawn,
    )

    apply_pit_train_spawn(env_cfg, args)
    apply_pit_box_if_requested(env_cfg, args)
    apply_pit_dr_light(env_cfg, args)
    return env_cfg


def build_play_env_cfg(args: Any):
    """Play/eval cfg: fixed command, optional fixed pit width, same reward/termination terms as train."""
    env_cfg = UnitreeB2PiperTaskDPitLocomotionMargEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.pit_width_range = (float(_get(args, "pit_width_min", 0.4)), float(_get(args, "pit_width_max", 1.4)))
    env_cfg.pit_curriculum_levels = int(_get(args, "pit_curriculum_levels", 11))

    command_vx = float(_get(args, "command_vx", _get(args, "pit_command_vx", 0.6)))
    env_cfg.command_lin_vel_x_min = command_vx
    env_cfg.command_lin_vel_x_max = command_vx
    env_cfg.command_curriculum_start_fraction = 1.0
    env_cfg.apply_command_config()

    configure_play_terrain(
        env_cfg,
        pit_level=int(_get(args, "pit_level", 0)),
        use_pit_curriculum=bool(_get(args, "pit_curriculum", False)),
    )
    apply_play_spawn(
        env_cfg,
        spawn_x_offset=float(_get(args, "pit_spawn_x_offset", _get(args, "spawn_x_offset", 0.0))),
        spawn_local_x=_get(args, "pit_spawn_local_x", _get(args, "spawn_local_x", None)),
    )

    from atec_rl_lab.train.nav.taskd_student_pit_e2e_env import (
        apply_pit_box_if_requested,
        apply_pit_dr_light,
    )

    apply_pit_box_if_requested(env_cfg, args)
    apply_pit_dr_light(env_cfg, args)

    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.pit_width_levels = None
    if hasattr(env_cfg.observations, "policy") and env_cfg.observations.policy is not None:
        env_cfg.observations.policy.enable_corruption = False
    return env_cfg


def configure_play_terrain(env_cfg, *, pit_level: int, use_pit_curriculum: bool) -> None:
    """Minimal terrain for play: fixed pit width bakes only ~num_envs tiles."""
    num_envs = int(env_cfg.scene.num_envs)
    width_range = tuple(env_cfg.pit_width_range)
    if use_pit_curriculum:
        terrain_cfg = env_cfg._build_terrain_cfg()
        nrow = terrain_cfg.terrain_generator.num_rows
        ncol = terrain_cfg.terrain_generator.num_cols
        print(
            f"[TaskDPitPlay] full curriculum terrain {nrow}x{ncol} ({nrow * ncol} tiles)",
            flush=True,
        )
    else:
        fixed_w = _pit_width_for_level(pit_level, width_range, env_cfg.pit_curriculum_levels)
        terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
        terrain_cfg.class_type = TaskDTerrainImporter
        terrain_cfg.terrain_generator.class_type = TaskDPitTerrainGenerator
        configure_task_d_terrain_for_num_envs(terrain_cfg, num_envs, curriculum_levels=None)
        terrain_cfg.terrain_generator.curriculum = False
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = (fixed_w, fixed_w)
            pit_cfg.platform_height_range = env_cfg.platform_height_range
        nrow = terrain_cfg.terrain_generator.num_rows
        ncol = terrain_cfg.terrain_generator.num_cols
        print(
            f"[TaskDPitPlay] minimal terrain {nrow}x{ncol} ({nrow * ncol} tiles), "
            f"fixed pit_width={fixed_w:.3f}m (level={pit_level})",
            flush=True,
        )

    env_cfg.scene.terrain = terrain_cfg
    env_cfg.scene.terrain.max_init_terrain_level = 0


def _robot_spawn_reset_params(env_cfg):
    """Pit locomotion uses reset_robot_task_d; full Task D uses reset_robot_root."""
    events = env_cfg.events
    if getattr(events, "reset_robot_task_d", None) is not None:
        return events.reset_robot_task_d.params
    if getattr(events, "reset_robot_root", None) is not None:
        return events.reset_robot_root.params
    raise AttributeError("env_cfg.events has neither reset_robot_task_d nor reset_robot_root")


def apply_play_spawn(
    env_cfg,
    *,
    spawn_x_offset: float,
    spawn_local_x: float | None,
) -> tuple[float, float, float]:
    """Configure Task D play spawn on env-local +x (positive = toward pit)."""
    base = tuple(float(v) for v in TASK_D_ROBOT_SPAWN_LOCAL)
    if spawn_local_x is not None:
        local_x = float(spawn_local_x)
    else:
        local_x = base[0] + float(spawn_x_offset)
    # Pit-loco / MARG student envs only; full Task D platform keeps env_cfg z=0.8.
    if getattr(env_cfg.events, "reset_robot_task_d", None) is not None:
        local_pos = task_d_pit_marg_spawn_local(local_x=local_x)
    else:
        local_pos = (local_x, base[1], base[2])
    _robot_spawn_reset_params(env_cfg)["local_pos"] = local_pos
    print(
        f"[TaskDPitPlay] spawn local_pos={local_pos} "
        f"(default={base}, offset_x={float(spawn_x_offset):+.3f})",
        flush=True,
    )
    return local_pos


def build_agent_cfg(device: str, *, max_iterations: int | None = None) -> UnitreeB2PiperTaskDPitLocomotionMargPPORunnerCfg:
    agent_cfg = UnitreeB2PiperTaskDPitLocomotionMargPPORunnerCfg()
    agent_cfg.device = device
    if max_iterations is not None:
        agent_cfg.max_iterations = int(max_iterations)
    return agent_cfg


def _get(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _resolve_checkpoint(args: Any) -> str:
    ckpt = _get(args, "pit_checkpoint", None) or _get(args, "checkpoint", None)
    if not ckpt:
        raise ValueError("Pit MARG play requires --pit_checkpoint or --checkpoint")
    path = os.path.abspath(str(ckpt))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def train_pit_marg(args: Any) -> str:
    """Train pit MARG locomotion; returns log_dir."""
    register_marg_modules()
    device = args.device if getattr(args, "device", None) else "cuda"
    env_cfg = build_train_env_cfg(args)
    max_iter = int(_get(args, "max_iterations", _get(args, "max_iter", 15000)))
    agent_cfg = build_agent_cfg(device, max_iterations=max_iter)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=device)

    resume = _get(args, "resume", None)
    if resume:
        print(f"[TaskDPitMarg] Resuming from {resume}", flush=True)
        runner.load(resume)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    _dump_pit_marg_deploy_notes(log_dir, env_cfg)

    print(f"[TaskDPitMarg] Logging to {log_dir}", flush=True)
    print(
        f"[TaskDPitMarg] pit_width={env_cfg.pit_width_range}, levels={env_cfg.pit_curriculum_levels}, "
        f"vx=[{env_cfg.command_lin_vel_x_min}, {env_cfg.command_lin_vel_x_max}] m/s "
        f"(curriculum start_frac={env_cfg.command_curriculum_start_fraction}), "
        f"success_post={env_cfg.pit_success_post_cross_distance}m, "
        f"num_envs={env_cfg.scene.num_envs}, MargActorCritic + pit env rewards/terminations",
        flush=True,
    )
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    return log_dir


def play_pit_marg(args: Any, simulation_app) -> None:
    """Play/eval pit MARG checkpoint (same env + network as train)."""
    register_marg_modules()
    device = args.device if getattr(args, "device", None) else "cuda"
    ckpt = _resolve_checkpoint(args)
    env_cfg = build_play_env_cfg(args)
    spawn_local = env_cfg.events.reset_robot_task_d.params["local_pos"]
    agent_cfg = build_agent_cfg(device)

    record_video = bool(_get(args, "video", False))
    render_mode = "rgb_array" if record_video else None
    env = gym.make("ATEC-TaskD-PitLocomotion-B2Piper-v0", cfg=env_cfg, render_mode=render_mode)

    if record_video:
        video_dir = _get(args, "video_dir", None)
        if video_dir is None:
            video_dir = os.path.join(os.path.dirname(ckpt), "videos", "play")
        video_dir = os.path.abspath(str(video_dir))
        os.makedirs(video_dir, exist_ok=True)
        video_length = int(_get(args, "video_length", 600))
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[TaskDPitMarg] Recording video:", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    print(f"[TaskDPitMarg] Loading checkpoint: {ckpt}", flush=True)
    runner.load(ckpt)
    runner.eval_mode()
    policy_nn = runner.alg.policy
    stochastic = bool(_get(args, "pit_stochastic", _get(args, "stochastic", False)))
    if stochastic:
        policy = lambda obs: policy_nn.act(obs)
        print("[TaskDPitMarg] Stochastic policy.act() (training rollouts)", flush=True)
    else:
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
        print("[TaskDPitMarg] Deterministic act_inference()", flush=True)

    unwrapped = vec_env.unwrapped
    if bool(_get(args, "pit_curriculum", False)):
        terrain = unwrapped.scene.terrain
        if hasattr(terrain, "terrain_levels"):
            level = int(_get(args, "pit_level", 0))
            max_level = max(0, int(_get(args, "pit_curriculum_levels", 11)) - 1)
            level = max(0, min(level, max_level))
            terrain.terrain_levels[:] = level
            terrain.update_env_origins_from_levels(terrain.terrain_levels.clone())
            print(f"[TaskDPitMarg] Initial pit level={level} for all envs", flush=True)

    obs = vec_env.get_observations()
    dt = unwrapped.step_dt
    steps = 0
    warmup = max(0, int(_get(args, "pit_warmup_steps", _get(args, "warmup_steps", 0))))
    max_steps = int(_get(args, "video_length", 600)) if record_video else None
    debug = bool(_get(args, "pit_debug", _get(args, "debug", False)))
    real_time = bool(_get(args, "pit_real_time", _get(args, "real_time", False)))
    command_vx = float(_get(args, "command_vx", _get(args, "pit_command_vx", 0.6)))

    print(
        f"[TaskDPitMarg] command_vx={command_vx} m/s, spawn_local_x={float(spawn_local[0]):.3f}, "
        f"num_envs={args.num_envs}, pit_width={env_cfg.pit_width_range}, "
        f"success_post={env_cfg.pit_success_post_cross_distance}m, stochastic={stochastic}, "
        f"pit_dr_light={bool(_get(args, 'pit_dr_light', False))}",
        flush=True,
    )
    if record_video and warmup > 0:
        print(f"[TaskDPitMarg] Warmup {warmup} steps before video (MARG history)", flush=True)

    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = policy(obs).to(unwrapped.device)
            if debug and steps % 50 == 0:
                robot = unwrapped.scene["robot"]
                cmd_x = float(unwrapped.command_manager.get_command("base_velocity")[0, 0].item())
                vx = float(robot.data.root_lin_vel_w[0, 0].item())
                act_norm = float(actions[0].norm().item())
                ep_len = int(unwrapped.episode_length_buf[0].item())
                print(
                    f"[TaskDPitMarg debug] step={steps} cmd_x={cmd_x:.3f} vx={vx:.3f} "
                    f"|action|={act_norm:.3f} ep_len={ep_len}",
                    flush=True,
                )
            obs, _, dones, _ = vec_env.step(actions)
            policy_nn.reset(dones)
        steps += 1
        if steps == warmup and record_video:
            print(f"[TaskDPitMarg] Warmup done ({warmup} steps), recording starts.", flush=True)
        if max_steps is not None and steps >= warmup + max_steps:
            print(f"[TaskDPitMarg] Recorded {steps - warmup} steps after warmup, stopping.", flush=True)
            break
        if real_time:
            sleep_s = dt - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    vec_env.close()


def _dump_pit_marg_deploy_notes(log_dir: str, env_cfg) -> None:
    """Write play/deploy hints (sim play, not demo/server.py depth nav)."""
    notes = {
        "network": "MargActorCritic (proprio + proprio_history + height_map 187D)",
        "train_env": "UnitreeB2PiperTaskDPitLocomotionMargEnvCfg",
        "play_script": "scripts/play_taskd_pit_locomotion.py --marg --checkpoint <model.pt> [--pit_dr_light]",
        "play_via_student_script": (
            "scripts/train_nav_taskd_student.py --pit_marg_play --pit_checkpoint <model.pt> "
            "[--video --pit_level N --command_vx 0.6 --pit_dr_light]"
        ),
        "pit_success_post_cross_distance": float(env_cfg.pit_success_post_cross_distance),
        "not_compatible_with": "demo/server.py depth student nav (different obs + network)",
    }
    dump_yaml(os.path.join(log_dir, "params", "deploy_pit_marg.yaml"), notes)
