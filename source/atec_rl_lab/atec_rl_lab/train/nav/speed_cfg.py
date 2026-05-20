"""Speed-oriented overrides for Task-A nav training.

Also see :func:`apply_nav_training_env_cfg` for reset / termination tweaks used
during hierarchical nav policy training.

Sensor performance hierarchy (fastest → slowest):
  --no_lidar     :    0 extero rays – proprio only
  default        :   75 downward rays (GridPatternCfg height scanner)  ← NEW DEFAULT
  --front_lidar  :   40 angled rays (LiDAR ±30°, 3° res, 2 ch)
  --fast         :  240 angled rays (LiDAR 360°, 6° res, 4 ch)
  full quality   : 5760 angled rays (LiDAR 360°, 1° res, 16 ch)  – unusable

The default height scanner replaces the spherical LiDAR with a forward-facing
grid of *downward* rays (GridPatternCfg). All rays are parallel and share BVH
cache lines → 50-100× faster than the same number of omnidirectional LiDAR rays
on the complex Task-A pyramid-stair terrain.
"""

from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from atec_rl_lab.train.locomotion.velocity.mdp import events as vel_events
import atec_rl_lab.tasks.task_a.mdp as atec_mdp


# Task-A track world x (terrain spans -150 ~ +150 in world frame)
TASK_A_START_X = -141.0   # robot default spawn world x
TASK_A_GOAL_X  = 145.0
TASK_A_TRACK_LENGTH = TASK_A_GOAL_X - TASK_A_START_X  # 286 m

# World x range considered "safe on terrain" for random spawning.
# Terrain spans world x ∈ [-150, +150].  Keep margins from edge and goal.
TASK_A_SPAWN_X_MIN = -135.0   # 5 m past terrain row-0 centre (-140)
TASK_A_SPAWN_X_MAX =  130.0   # 15 m before goal (145)

# Standing height for B2 Piper (from init_state.pos z)
B2_STANDING_Z = 0.58
# Extra clearance above terrain to avoid spawning inside sloped tiles
SPAWN_Z_EXTRA  = 0.20

# ── Start flat tiles (terrain_sequence[0:2] are "flat", 20 m cells, see task_a/terrain.py) ──
# World AABB covering both flats: x ∈ [-150,-110], y ∈ [-10,10] with 1 m inset from tile edges.
FLAT_GRID_COLS = 16
FLAT_GRID_ROWS = 16
FLAT_GRID_SLOTS = FLAT_GRID_COLS * FLAT_GRID_ROWS  # 256 spawn cells
FLAT_WORLD_X_MIN = -149.0
FLAT_WORLD_X_MAX = -111.0
FLAT_WORLD_Y_MIN = -9.0
FLAT_WORLD_Y_MAX = 9.0
# Cell size ≈ 2.4 m × 1.1 m — enough separation for B2 (~0.8 m body width)


def _flat_grid_world_xy(env_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each env_id to a unique cell centre on the two start flat terrain tiles."""
    slot = env_ids.long() % FLAT_GRID_SLOTS
    col = slot % FLAT_GRID_COLS
    row = slot // FLAT_GRID_COLS
    span_x = FLAT_WORLD_X_MAX - FLAT_WORLD_X_MIN
    span_y = FLAT_WORLD_Y_MAX - FLAT_WORLD_Y_MIN
    wx = FLAT_WORLD_X_MIN + (col.to(dtype=torch.float32) + 0.5) / FLAT_GRID_COLS * span_x
    wy = FLAT_WORLD_Y_MIN + (row.to(dtype=torch.float32) + 0.5) / FLAT_GRID_ROWS * span_y
    return wx.to(device), wy.to(device)


def reset_nav_spawn_absolute(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float] = (-1.5, 1.5),
    yaw_range: tuple[float, float] = (-0.25, 0.25),
    standing_z: float = B2_STANDING_Z,
    z_extra: float = SPAWN_Z_EXTRA,
    y_stagger: str = "random",
    _warned: list = [],
) -> None:
    """Spawn robot at absolute world (x, y) positions along the Task-A track.

    Unlike ``reset_root_state_uniform`` this function ignores the
    ``default_root_state + env_origin`` double-offset and writes the absolute
    world position directly via ``write_root_pose_to_sim``.

    This is necessary for Task A because the env_origins (centre of the
    terrain sub-tile assigned to each env) are typically around (-140, 0, 0),
    so ``default_root_state.x(-141) + env_origin.x(-140) + offset`` would
    place the robot at roughly world x = -281 + offset, pushing most spawns
    off the terrain (world x < -150 = void).

    ``y_stagger``:
      - ``"random"``: uniform random y in ``world_y_range`` (independent per reset)
      - ``"flat_grid"``: ``env_id % 256`` → centre of one of 16×16 cells on the two start
        flat tiles (``--no_random_spawn``); ~2.4 m × 1.1 m per cell, no overlap up to 256 envs
    """
    robot = env.scene[asset_cfg.name]
    n = len(env_ids)

    # Print a one-time diagnostic on the first reset call.
    if not _warned:
        env_origins = env.scene.env_origins[env_ids]
        default_pos = robot.data.default_root_state[env_ids, :3]
        old_world_x = (default_pos + env_origins)[:, 0]
        if y_stagger == "flat_grid":
            cell_x = (FLAT_WORLD_X_MAX - FLAT_WORLD_X_MIN) / FLAT_GRID_COLS
            cell_y = (FLAT_WORLD_Y_MAX - FLAT_WORLD_Y_MIN) / FLAT_GRID_ROWS
            spawn_msg = (
                f"flat_grid {FLAT_GRID_COLS}×{FLAT_GRID_ROWS}={FLAT_GRID_SLOTS} cells "
                f"on start flats x∈[{FLAT_WORLD_X_MIN},{FLAT_WORLD_X_MAX}] "
                f"y∈[{FLAT_WORLD_Y_MIN},{FLAT_WORLD_Y_MAX}] "
                f"(~{cell_x:.2f}m×{cell_y:.2f}m/cell)"
            )
        else:
            spawn_msg = f"world_x∈{world_x_range}  y_stagger={y_stagger}  y∈{world_y_range}"
        print(
            f"[NavSpawn] env_origin_x range: "
            f"[{env_origins[:,0].min():.1f}, {env_origins[:,0].max():.1f}]  "
            f"default_world_x: [{old_world_x.min():.1f}, {old_world_x.max():.1f}]  "
            f"→ {spawn_msg}",
            flush=True,
        )
        if y_stagger == "flat_grid" and env.num_envs > FLAT_GRID_SLOTS:
            print(
                f"[NavSpawn] WARNING: num_envs={env.num_envs} > {FLAT_GRID_SLOTS}; "
                f"env_ids wrap with modulo (two envs may share a cell).",
                flush=True,
            )
        _warned.append(True)

    env_origins = env.scene.env_origins[env_ids]

    if y_stagger == "flat_grid":
        wx, wy = _flat_grid_world_xy(env_ids, env.device)
        # Start flats are MeshPlane at world z≈0
        wz = torch.full((n,), standing_z + z_extra, device=env.device, dtype=torch.float32)
    else:
        # World x (fixed when min == max)
        if world_x_range[0] == world_x_range[1]:
            wx = torch.full((n,), world_x_range[0], device=env.device, dtype=torch.float32)
        else:
            wx = torch.rand(n, device=env.device) * (world_x_range[1] - world_x_range[0]) + world_x_range[0]
        wy = torch.rand(n, device=env.device) * (world_y_range[1] - world_y_range[0]) + world_y_range[0]
        wz = env_origins[:, 2] + standing_z + z_extra

    positions = torch.stack([wx, wy, wz], dim=-1)  # [N, 3] absolute world

    # Orientation: default quaternion + random yaw
    root_states = robot.data.default_root_state[env_ids].clone()
    yaw   = torch.rand(n, device=env.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    zeros = torch.zeros_like(yaw)
    ori_delta = quat_from_euler_xyz(zeros, zeros, yaw)
    orientations = quat_mul(root_states[:, 3:7], ori_delta)

    velocities = torch.zeros(n, 6, device=env.device)

    robot.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# ── Height scanner constants ──────────────────────────────────────────────────
# Grid: 3 m forward × 1 m lateral, 0.2 m resolution → 15 × 5 = 75 rays
HEIGHT_SCAN_FWD  = 3.0   # metres ahead of robot centre
HEIGHT_SCAN_LAT  = 1.0   # metres across robot
HEIGHT_SCAN_RES  = 0.2   # grid cell size
HEIGHT_SCAN_DIMS = int(HEIGHT_SCAN_FWD / HEIGHT_SCAN_RES) * int(HEIGHT_SCAN_LAT / HEIGHT_SCAN_RES)  # 75


def apply_nav_speed_cfg(
    env_cfg,
    *,
    fast: bool = False,
    lidar_horiz_res: float | None = None,
    front_only: bool = False,
    no_lidar: bool = False,
) -> tuple:
    """Apply speed overrides to env_cfg and return (env_cfg, extero_raw_dims, lidar_compress_bins).

    Callers should pass these to HierarchicalNavEnv:
        extero_raw_dims   > 0  → pass raw extero obs directly (height scanner)
        lidar_compress_bins > 0 → compress spherical LiDAR scan into N angular bins
        both == 0              → proprio only
    """
    if no_lidar:
        env_cfg.scene.lidar_sensor = None
        env_cfg.observations.extero.lidar_scan = None
        env_cfg.observations.extero = None
        _apply_common_speed_cfg(env_cfg, fast)
        return env_cfg, 0, 0   # (env_cfg, extero_raw_dims, lidar_bins)

    lidar = env_cfg.scene.lidar_sensor
    if lidar is None:
        return env_cfg, 0, 0

    # ── Decide which sensor pattern to use ───────────────────────────────────

    if front_only:
        # Narrow-cone LiDAR: ±30° horizontal, 2 vertical channels → 40 rays
        lidar.update_period = 0.1
        lidar.pattern_cfg = patterns.LidarPatternCfg(
            vertical_fov_range=(-10.0, 10.0),
            horizontal_fov_range=(-30.0, 30.0),
            horizontal_res=3.0,
            channels=2,
        )
        _apply_common_speed_cfg(env_cfg, fast)
        return env_cfg, 0, 36  # compress into 36 angular bins

    if fast or lidar_horiz_res is not None:
        # Coarse 360° LiDAR: 4 channels, 6° (or user-specified) resolution → 240 rays
        res = float(lidar_horiz_res) if lidar_horiz_res else 6.0
        lidar.update_period = 0.1
        lidar.pattern_cfg = patterns.LidarPatternCfg(
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=res,
            channels=4,
        )
        _apply_common_speed_cfg(env_cfg, fast)
        return env_cfg, 0, 36  # compress into 36 angular bins

    # ── DEFAULT: replace spherical LiDAR with fast forward height scanner ────
    # GridPatternCfg casts rays straight down (parallel) → much faster BVH.
    # update_period=0.02 = every env step (cheap enough to do every step).
    lidar.update_period = 0.02
    lidar.pattern_cfg = patterns.GridPatternCfg(
        resolution=HEIGHT_SCAN_RES,
        size=[HEIGHT_SCAN_FWD, HEIGHT_SCAN_LAT],
    )
    # Offset: 1 m ahead so the scan covers [0 m, 3 m] in front of the robot;
    # 20 m up so the rays fall from above and hit terrain going downward.
    lidar.offset = RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0))
    # attach_yaw_only=True → grid rotates with robot yaw but stays level (no tilt).
    lidar.attach_yaw_only = True

    _apply_common_speed_cfg(env_cfg, fast)
    return env_cfg, HEIGHT_SCAN_DIMS, 0  # raw 75-dim height scan


def _apply_common_speed_cfg(env_cfg, fast: bool):
    """Shared lightweight tweaks (applied regardless of sensor mode)."""
    if fast:
        env_cfg.episode_length_s = 120.0

    cs = env_cfg.scene.contact_sensor
    if cs is not None:
        cs.history_length = 1
        cs.track_air_time = False

    if hasattr(env_cfg, "sim") and env_cfg.sim is not None:
        if hasattr(env_cfg.sim, "physx"):
            env_cfg.sim.physx.enable_external_forces_every_iteration = False


def apply_nav_training_env_cfg(
    env_cfg,
    *,
    stuck_time_s: float = 3.0,
    stuck_grace_s: float = 1.0,
    enable_stuck: bool = True,
    randomize_spawn: bool = True,
):
    """Nav-training overrides: random spawn along track, optional stuck termination, sane fall."""
    # CrossXMulti visual cuboids are expensive and not needed for RL
    if hasattr(env_cfg, "rewards") and env_cfg.rewards is not None:
        pr = getattr(env_cfg.rewards, "progress_reward", None)
        if pr is not None and hasattr(pr, "params"):
            pr.params["visual_assets"] = False
            pr.params["debug"] = False

    # Fall: Task A default minimum_height=-20 effectively disables fall; use 0.15 for B2 (~0.58 m spawn).
    if hasattr(env_cfg, "terminations") and env_cfg.terminations is not None:
        fall = getattr(env_cfg.terminations, "fall", None)
        if fall is not None and hasattr(fall, "params"):
            fall.params["minimum_height"] = 0.15

        if enable_stuck:
            env_cfg.terminations.stuck_no_progress = DoneTerm(
                func=atec_mdp.StuckNoProgress,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "stuck_time_s": stuck_time_s,
                    "progress_eps": 0.03,
                    "grace_time_s": stuck_grace_s,
                },
                time_out=False,
            )
        elif hasattr(env_cfg.terminations, "stuck_no_progress"):
            env_cfg.terminations.stuck_no_progress = None

    if randomize_spawn and hasattr(env_cfg, "events"):
        # Use absolute world-coordinate spawn to avoid the double-negative offset:
        #   reset_root_state_uniform: world_x = default_root_x(-141) + env_origin_x(-140) + rand_x
        # → all spawns land near world x ≈ -281+rand, far off the terrain (-150~+150)!
        # reset_nav_spawn_absolute sets world positions directly.
        env_cfg.events.randomize_reset_base = EventTerm(
            func=reset_nav_spawn_absolute,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "world_x_range": (TASK_A_SPAWN_X_MIN, TASK_A_SPAWN_X_MAX),
                "world_y_range": (-1.5, 1.5),
                "yaw_range": (-0.25, 0.25),
                "standing_z": B2_STANDING_Z,
                "z_extra": SPAWN_Z_EXTRA,
                "y_stagger": "random",
            },
        )
    else:
        # 16×16 grid on the two start flat tiles (256 cells, ~2.4 m × 1.1 m each)
        if hasattr(env_cfg, "events"):
            env_cfg.events.randomize_reset_base = EventTerm(
                func=reset_nav_spawn_absolute,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "world_x_range": (TASK_A_START_X, TASK_A_START_X),
                    "world_y_range": (0.0, 0.0),
                    "yaw_range": (0.0, 0.0),
                    "standing_z": B2_STANDING_Z,
                    "z_extra": SPAWN_Z_EXTRA,
                    "y_stagger": "flat_grid",
                },
            )
        # Keep every env on the first flat row during training (no rough/stairs at init)
        terrain = getattr(env_cfg.scene, "terrain", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 0

    if hasattr(env_cfg, "events") and getattr(env_cfg.events, "randomize_reset_base", None) is not None:
        # Joint reset helps the loco policy recover after a fall / stuck episode
        import isaaclab.envs.mdp as isaac_mdp

        env_cfg.events.reset_robot_joints = EventTerm(
            func=isaac_mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.95, 1.05),
                "velocity_range": (0.0, 0.0),
            },
        )

    return env_cfg


def apply_fast_ppo_cfg(agent_cfg):
    """Smaller rollout / network for quicker iteration during debugging."""
    agent_cfg.num_steps_per_env = 12
    agent_cfg.policy.actor_hidden_dims = [256, 128]
    agent_cfg.policy.critic_hidden_dims = [256, 128]
    agent_cfg.algorithm.num_learning_epochs = 3
    agent_cfg.algorithm.num_mini_batches = 2
    return agent_cfg
