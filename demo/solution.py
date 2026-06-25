import os
import importlib
from enum import Enum
from typing import Any

import numpy as np
import torch

try:
    from utils import approach_dustbin
except ImportError:
    from demo.utils import approach_dustbin

try:
    from server_obs import SERVER_DEPTH_H, SERVER_DEPTH_W
except ImportError:
    try:
        from demo.server_obs import SERVER_DEPTH_H, SERVER_DEPTH_W
    except ImportError:
        SERVER_DEPTH_H, SERVER_DEPTH_W = 480, 640

_O3D = None
_O3D_IMPORT_FAILED = False


def _try_import_o3d():
    """Lazy import: headless servers often lack libX11 even when open3d is pip-installed."""
    global _O3D, _O3D_IMPORT_FAILED
    if _O3D_IMPORT_FAILED:
        return None
    if _O3D is not None:
        return _O3D
    try:
        import open3d as o3d  # noqa: WPS433

        _O3D = o3d
        return o3d
    except (ImportError, OSError) as exc:
        _O3D_IMPORT_FAILED = True
        print(f"[TaskB] open3d unavailable ({exc}); using numpy/sklearn fallbacks.", flush=True)
        return None
try:
    from demo.graspnet_pick import (
        BANANA_VIEW_SPECS,
        GraspNetStructuredSM,
        GraspNetTaskBRunner,
        GraspNetWaypointSM,
        GraspPlanConfig,
        MUSTARD_VIEW_SPECS,
        MustardGraspConfig,
        calibrate_gripper_camera_extrinsic,
        make_transform,
        quat_wxyz_to_matrix,
        world_camera_from_gripper_extrinsic,
    )
    _GRASPNET_PICK_AVAILABLE = True
except Exception:  # pragma: no cover
    GraspNetTaskBRunner = None
    GraspNetStructuredSM = None
    GraspNetWaypointSM = None
    GraspPlanConfig = None
    MustardGraspConfig = None
    MUSTARD_VIEW_SPECS = []
    BANANA_VIEW_SPECS = []
    calibrate_gripper_camera_extrinsic = None
    world_camera_from_gripper_extrinsic = None
    make_transform = None
    quat_wxyz_to_matrix = None
    _GRASPNET_PICK_AVAILABLE = False
try:
    from demo.grasp_ik_task_e import (
        ACTION_SCALE as TASKE_ACTION_SCALE,
        GRASP_STATE_ORDER,
        GRIPPER_CLOSE_POS,
        GRIPPER_OPEN_POS,
        STEPS as TASKE_GRASP_STEPS,
        TaskEGraspOnlySM,
        compute_grasp_quat,
        gripper_tensor,
    )
except Exception:  # pragma: no cover
    TASKE_ACTION_SCALE = 0.5
    GRASP_STATE_ORDER = ("PRE_GRASP", "REACH", "CLOSE", "LIFT")
    GRIPPER_CLOSE_POS = [-0.015, 0.015]
    GRIPPER_OPEN_POS = [0.035, -0.035]
    TASKE_GRASP_STEPS = {}
    TaskEGraspOnlySM = None
    compute_grasp_quat = None
    gripper_tensor = None
try:
    from atec_rl_lab.utils import CartesianController
    from isaaclab.utils.math import subtract_frame_transforms
except Exception:  # pragma: no cover
    CartesianController = None
    subtract_frame_transforms = None
class Status(Enum):
    SEARCH = 1
    LOCK = 2
    DETECT = 3
    GRASP = 4
    STAND = 5
    GO_BIN = 6


class AlgSolution:
    ACTION_SCALE = 0.5
    _SQUAT_STEPS = int(os.environ.get("ATEC_SQUAT_STEPS", "140"))
    _LOCK_POLICY_STEPS = int(os.environ.get("ATEC_LOCK_POLICY_STEPS", "50"))
    _LOCK_DETECT_MIN_ROOT_Z = float(os.environ.get("ATEC_LOCK_DETECT_MIN_ROOT_Z", "0.19"))
    _LEG_ACTION_SCALE = 0.5
    _PICK_ARM_STEPS = 25
    _PICK_SCRIPTED_PRE_STEPS = 34
    _PICK_SCRIPTED_HOLD_STEPS = 18
    _STAND_STEPS = 40
    _BIN_ARRIVE_DIST = 0.4
    _GO_BIN_ALIGN_RAD = 0.3
    _GO_BIN_WZ_GAIN = 0.8
    _GO_BIN_VX = 1.5
    _BIN_GUARD_ENABLE = os.environ.get("ATEC_BIN_GUARD_ENABLE", "0").strip().lower() in ("1", "true", "yes", "on")
    _BIN_GUARD_DIST = float(os.environ.get("ATEC_BIN_GUARD_DIST", "0.30"))
    _BIN_GUARD_REVERSE_VX = float(os.environ.get("ATEC_BIN_GUARD_REVERSE_VX", "-0.25"))
    _BIN_GUARD_WZ_GAIN = float(os.environ.get("ATEC_BIN_GUARD_WZ_GAIN", "1.2"))
    _BIN_GUARD_MAX_WZ = float(os.environ.get("ATEC_BIN_GUARD_MAX_WZ", "0.60"))
    _SEARCH_CRUISE_VX = 1.5
    _SEARCH_DECEL_START_DIST = 1.8
    _SEARCH_SLOW_VX = 0.4
    _SEARCH_NO_TARGET_WZ = float(os.environ.get("ATEC_SEARCH_NO_TARGET_WZ", "0.45"))
    _SEARCH_LOST_MEMORY_STEPS = int(os.environ.get("ATEC_SEARCH_LOST_MEMORY_STEPS", "2"))
    _SEARCH_LOST_MEMORY_VX_CAP = float(os.environ.get("ATEC_SEARCH_LOST_MEMORY_VX_CAP", "0.25"))
    _SEARCH_LOST_MEMORY_WZ_CAP = float(os.environ.get("ATEC_SEARCH_LOST_MEMORY_WZ_CAP", "0.6"))
    _LOCK_ARRIVE_DIST = float(os.environ.get("ATEC_LOCK_ARRIVE_DIST", "0.4"))
    _LOCK_ARRIVE_CONFIRM_STEPS = int(os.environ.get("ATEC_LOCK_ARRIVE_CONFIRM_STEPS", "1"))
    _LOCK_COMMIT_CONFIRM_STEPS = int(os.environ.get("ATEC_LOCK_COMMIT_CONFIRM_STEPS", "1"))
    _LOCK_LOST_COMMIT_DIST = float(os.environ.get("ATEC_LOCK_LOST_COMMIT_DIST", "0.4"))
    _LOCK_LOST_COMMIT_STEPS = int(os.environ.get("ATEC_LOCK_LOST_COMMIT_STEPS", "8"))
    _LOCK_PRE_ZERO_STEPS = int(os.environ.get("ATEC_LOCK_PRE_ZERO_STEPS", "40"))
    _LOCK_PRE_ZERO_MAX_STEPS = int(os.environ.get("ATEC_LOCK_PRE_ZERO_MAX_STEPS", "150"))
    _LOCK_PRE_ZERO_SETTLE_STEPS = int(os.environ.get("ATEC_LOCK_PRE_ZERO_SETTLE_STEPS", "10"))
    _LOCK_PRE_ZERO_MAX_HSPEED = float(os.environ.get("ATEC_LOCK_PRE_ZERO_MAX_HSPEED", "0.08"))
    _LOCK_PRE_ZERO_MAX_VSPEED = float(os.environ.get("ATEC_LOCK_PRE_ZERO_MAX_VSPEED", "0.12"))
    _LOCK_PRE_ZERO_FORCE_HSPEED = float(os.environ.get("ATEC_LOCK_PRE_ZERO_FORCE_HSPEED", "0.10"))
    _LOCK_PRE_ZERO_FORCE_VSPEED = float(os.environ.get("ATEC_LOCK_PRE_ZERO_FORCE_VSPEED", "0.18"))
    _LOCK_RECENTER_STEPS = int(os.environ.get("ATEC_LOCK_RECENTER_STEPS", "5"))
    _LOCK_COMMIT_VX_CAP = float(os.environ.get("ATEC_LOCK_COMMIT_VX_CAP", "0.35"))
    _SEARCH_TRACK_ENABLE_DIST = float(os.environ.get("ATEC_SEARCH_TRACK_ENABLE_DIST", "1.0"))
    _SEARCH_TRACK_MAX_JUMP = float(os.environ.get("ATEC_SEARCH_TRACK_MAX_JUMP", "0.85"))
    _SEARCH_COVERAGE_ENABLE = os.environ.get("ATEC_SEARCH_COVERAGE_ENABLE", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )
    _SEARCH_NO_TARGET_WP_TIMEOUT_STEPS = int(os.environ.get("ATEC_SEARCH_NO_TARGET_WP_TIMEOUT_STEPS", "18"))
    _SEARCH_WP_REACHED_DIST = float(os.environ.get("ATEC_SEARCH_WP_REACHED_DIST", "0.9"))
    _SEARCH_WP_VX_CAP = float(os.environ.get("ATEC_SEARCH_WP_VX_CAP", "0.7"))
    _SEARCH_WP_WZ_CAP = float(os.environ.get("ATEC_SEARCH_WP_WZ_CAP", "0.7"))
    _SEARCH_CLUSTER_MAX_POINTS = int(os.environ.get("ATEC_SEARCH_CLUSTER_MAX_POINTS", "40000"))
    _SEARCH_BIN_FILTER_ENABLE = os.environ.get("ATEC_SEARCH_BIN_FILTER_ENABLE", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )
    _SEARCH_BIN_FILTER_MIN_HEIGHT = float(os.environ.get("ATEC_SEARCH_BIN_FILTER_MIN_HEIGHT", "0.3"))
    _SEARCH_BIN_AVOID_ENABLE = os.environ.get("ATEC_SEARCH_BIN_AVOID_ENABLE", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )
    _SEARCH_BIN_AVOID_TARGET_ZMAX = float(os.environ.get("ATEC_SEARCH_BIN_AVOID_TARGET_ZMAX", "0.3"))
    _SEARCH_BIN_AVOID_REVERSE_VX = float(os.environ.get("ATEC_SEARCH_BIN_AVOID_REVERSE_VX", "-0.22"))
    _SEARCH_BIN_AVOID_WZ_GAIN = float(os.environ.get("ATEC_SEARCH_BIN_AVOID_WZ_GAIN", "1.2"))
    _SEARCH_BIN_AVOID_MAX_WZ = float(os.environ.get("ATEC_SEARCH_BIN_AVOID_MAX_WZ", "0.7"))
    _SEARCH_BIN_RGB_ENABLE = os.environ.get("ATEC_SEARCH_BIN_RGB_ENABLE", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )
    _SEARCH_BIN_RGB_BEARING_TOL = float(os.environ.get("ATEC_SEARCH_BIN_RGB_BEARING_TOL", "0.75"))
    _SEARCH_BIN_RGB_HALF_DEG = float(os.environ.get("ATEC_SEARCH_BIN_RGB_HALF_DEG", "16.0"))
    _SEARCH_BIN_RGB_V0 = int(os.environ.get("ATEC_SEARCH_BIN_RGB_V0", "100"))
    _SEARCH_BIN_RGB_V1 = int(os.environ.get("ATEC_SEARCH_BIN_RGB_V1", "420"))
    _SEARCH_BIN_RGB_MAX_YELLOW_RATIO = float(os.environ.get("ATEC_SEARCH_BIN_RGB_MAX_YELLOW_RATIO", "0.25"))
    _SEARCH_DEBUG_PRINT_EVERY = int(os.environ.get("ATEC_SEARCH_DEBUG_PRINT_EVERY", "20"))
    _SEARCH_WAYPOINTS = (
        (-14.0, -14.0),
        (-14.0, -6.0),
        (-10.0, -10.0),
        (-6.0, -14.0),
        (-6.0, -6.0),
    )
    _LEG_JOINT_NAMES = (
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    )
    # b2.py default leg PD 160/5; squat position-hold uses damped gains.
    _SQUAT_LEG_STIFFNESS = float(os.environ.get("ATEC_SQUAT_LEG_STIFFNESS", "130"))
    _SQUAT_LEG_DAMPING = float(os.environ.get("ATEC_SQUAT_LEG_DAMPING", "45"))
    # LOCK low-splay pose for DETECT/GRASP:
    #   hip: outward (FR/RR -, FL/RL +), thigh/calf: lower trunk close to ground.
    _LOCK_SPLAY_HIP_MAG = float(os.environ.get("ATEC_LOCK_SPLAY_HIP_MAG", "0.34"))
    _LOCK_SQUAT_THIGH = float(os.environ.get("ATEC_LOCK_SQUAT_THIGH", "0.78"))
    _LOCK_SQUAT_CALF = float(os.environ.get("ATEC_LOCK_SQUAT_CALF", "-1.45"))
    _ENABLE_ANYGRASP = os.environ.get("ATEC_ENABLE_ANYGRASP", "1").strip().lower() in ("1", "true", "yes", "on")
    _ANYGRASP_CKPT = os.environ.get("ATEC_ANYGRASP_CKPT", "").strip()
    _ANYGRASP_TOPK = int(os.environ.get("ATEC_ANYGRASP_TOPK", "5"))
    _ANYGRASP_PRINT_EVERY = 20
    _PICK_DETECT_CHECK_EVERY = int(os.environ.get("ATEC_PICK_DETECT_CHECK_EVERY", "20"))
    _PICK_DETECT_MIN_STEPS = int(os.environ.get("ATEC_PICK_DETECT_MIN_STEPS", "2"))
    _ENABLE_MOVEIT_PICK = os.environ.get("ATEC_ENABLE_MOVEIT_PICK", "0").strip().lower() in ("1", "true", "yes", "on")
    _MOVEIT_BRIDGE_MODULE = os.environ.get("ATEC_MOVEIT_BRIDGE_MODULE", "demo.piper_moveit_bridge").strip()
    _MOVEIT_BRIDGE_CLASS = os.environ.get("ATEC_MOVEIT_BRIDGE_CLASS", "MoveItBridge").strip()
    _GRASP_TOP_OFFSET_RATIO = float(os.environ.get("ATEC_GRASP_TOP_OFFSET_RATIO", "0.5"))
    _GRASP_TOP_OFFSET_MAX = float(os.environ.get("ATEC_GRASP_TOP_OFFSET_MAX", "0.08"))
    _GRASP_USE_PCA = os.environ.get("ATEC_GRASP_USE_PCA", "0").strip().lower() in ("1", "true", "yes")
    _GRASP_ALIGN_YAW_ENABLE = os.environ.get("ATEC_GRASP_ALIGN_YAW_ENABLE", "1").strip().lower() in ("1", "true", "yes", "on")
    _GRASP_ALIGN_YAW_STEPS = int(os.environ.get("ATEC_GRASP_ALIGN_YAW_STEPS", "10"))
    _GRASP_ALIGN_YAW_GAIN = float(os.environ.get("ATEC_GRASP_ALIGN_YAW_GAIN", "0.7"))
    _GRASP_ALIGN_YAW_STEP = float(os.environ.get("ATEC_GRASP_ALIGN_YAW_STEP", "0.08"))
    _GRASP_ALIGN_YAW_SIGN = float(os.environ.get("ATEC_GRASP_ALIGN_YAW_SIGN", "1.0"))
    _GRASP_ALIGN_YAW_OFFSET = float(os.environ.get("ATEC_GRASP_ALIGN_YAW_OFFSET", "0.0"))
    _GRASP_ALIGN_YAW_TARGET_CLIP = float(os.environ.get("ATEC_GRASP_ALIGN_YAW_TARGET_CLIP", "1.8"))
    _GRASP_ALIGN_YAW_ERR_TOL = float(os.environ.get("ATEC_GRASP_ALIGN_YAW_ERR_TOL", "0.08"))
    _GRASP_WRIST_YAW_EXTRA_MAX = float(os.environ.get("ATEC_GRASP_WRIST_YAW_EXTRA_MAX", "1.2"))
    _ENABLE_DEPTH_IK_PICK = True
    _IK_PRE_STEPS = 14
    _IK_DESCEND_STEPS = 22
    _IK_CLOSE_STEPS = 14
    _IK_LIFT_STEPS = 20
    _IK_ACQUIRE_STEPS = 28
    _IK_ACQUIRE_MOVE_STEPS = 20
    # GRASP approach tuning (env overrides for sim tuning).
    _GRASP_TARGET_FORWARD = float(os.environ.get("ATEC_GRASP_TARGET_FORWARD", "0.12"))#期望前后距离（m）
    _GRASP_TARGET_LATERAL = float(os.environ.get("ATEC_GRASP_TARGET_LATERAL", "0.0"))#期望左右偏移
    _GRASP_TARGET_VERTICAL = float(os.environ.get("ATEC_GRASP_TARGET_VERTICAL", "0.00"))#期望高度
    _GRASP_F_ERR_TOL = float(os.environ.get("ATEC_GRASP_F_ERR_TOL", "0.04"))
    _GRASP_L_ERR_TOL = float(os.environ.get("ATEC_GRASP_L_ERR_TOL", "0.04"))
    _GRASP_V_ERR_TOL = float(os.environ.get("ATEC_GRASP_V_ERR_TOL", "0.04"))
    _GRASP_MIN_APPROACH_STEPS = int(os.environ.get("ATEC_GRASP_MIN_APPROACH_STEPS", "8"))
    _GRASP_APPROACH_TIMEOUT = int(os.environ.get("ATEC_GRASP_APPROACH_TIMEOUT", "45"))
    _GRASP_MULTI_TARGET_ENABLE = os.environ.get("ATEC_GRASP_MULTI_TARGET_ENABLE", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )
    _GRASP_MULTI_TARGET_MAX_CANDIDATES = int(os.environ.get("ATEC_GRASP_MULTI_TARGET_MAX_CANDIDATES", "5"))
    _GRASP_MULTI_TARGET_DEDUP_DIST = float(os.environ.get("ATEC_GRASP_MULTI_TARGET_DEDUP_DIST", "0.08"))
    _GRASP_REACHED_HOLD_STEPS = int(os.environ.get("ATEC_GRASP_REACHED_HOLD_STEPS", "4"))
    _GRASP_TARGET_EMA_ALPHA = float(os.environ.get("ATEC_GRASP_TARGET_EMA_ALPHA", "0.35"))
    _GRASP_J0_GAIN = float(os.environ.get("ATEC_GRASP_J0_GAIN", "0.85"))  # lateral P -> joint1
    _GRASP_J0_LATERAL_SIGN = float(os.environ.get("ATEC_GRASP_J0_LATERAL_SIGN", "1.0"))
    _GRASP_J0_DEADBAND = float(os.environ.get("ATEC_GRASP_J0_DEADBAND", "0.02"))
    #F负责forward V负责高度 ，j1和j2分别是两个自由度。
    _GRASP_P_F_J1 = float(os.environ.get("ATEC_GRASP_P_F_J1", "0.6"))
    _GRASP_P_V_J1 = float(os.environ.get("ATEC_GRASP_P_V_J1", "0.5"))
    _GRASP_P_F_J2 = float(os.environ.get("ATEC_GRASP_P_F_J2", "-0.75"))
    _GRASP_P_V_J2 = float(os.environ.get("ATEC_GRASP_P_V_J2", "-0.6"))
    _GRASP_D_ERR_CLIP = float(os.environ.get("ATEC_GRASP_D_ERR_CLIP", "0.12"))
    _GRASP_D_L_KD = float(os.environ.get("ATEC_GRASP_D_L_KD", "0.22"))
    _GRASP_D_F_KD = float(os.environ.get("ATEC_GRASP_D_F_KD", "0.16"))
    _GRASP_D_V_KD = float(os.environ.get("ATEC_GRASP_D_V_KD", "0.14"))
    _GRASP_JOINT_STEP_MAX = (
        float(os.environ.get("ATEC_GRASP_J0_STEP", "0.14")),
        float(os.environ.get("ATEC_GRASP_J1_STEP", "0.18")),
        float(os.environ.get("ATEC_GRASP_J2_STEP", "0.18")),
    )
    _GRASP_J0_EXTRA_MAX = float(os.environ.get("ATEC_GRASP_J0_EXTRA_MAX", "1.80"))
    # Reach envelope around hold pose for [j0, j1, j2]; larger value allows farther extension.
    _GRASP_JOINT_EXTRA_MAX = float(os.environ.get("ATEC_GRASP_JOINT_EXTRA_MAX", "1.35"))
    _GRASP_CLOSE_PUSH_J1 = float(os.environ.get("ATEC_GRASP_CLOSE_PUSH_J1", "0.035"))
    _GRASP_CLOSE_PUSH_J2 = float(os.environ.get("ATEC_GRASP_CLOSE_PUSH_J2", "-0.022"))
    _GRASP_DEBUG = os.environ.get("ATEC_GRASP_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
    _GRASP_DEBUG_EVERY = int(os.environ.get("ATEC_GRASP_DEBUG_EVERY", "1"))
    _DEPTH_TOUCH_ONLY_SCORE = os.environ.get("ATEC_DEPTH_TOUCH_ONLY_SCORE", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )
    _TOUCH_ACQUIRE_MAX_COUNT = int(os.environ.get("ATEC_TOUCH_ACQUIRE_MAX_COUNT", "45"))
    _DEPTH_CLUSTER_DEBUG = os.environ.get("ATEC_DEPTH_CLUSTER_DEBUG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Force disable direct SEARCH->LOCK shortcut to avoid premature squatting.
    _DIRECT_SQUAT_IK = False
    _GRASP_USE_TASKE_IK = os.environ.get("ATEC_GRASP_USE_TASKE_IK", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )
    _GRASP_IK_CARRY_Z = float(os.environ.get("ATEC_GRASP_IK_CARRY_Z", "0"))
    _GRASP_IK_Z_OFFSET = float(os.environ.get("ATEC_GRASP_IK_Z_OFFSET", "0.09"))
    _GRASP_RECORD = os.environ.get("ATEC_GRASP_RECORD", "0").strip().lower() in ("1", "true", "yes", "on")
    _GRASP_RECORD_DIR = os.environ.get("ATEC_GRASP_RECORD_DIR", "logs/grasp_demos")
    _TASK_B_NUM_OBJECTS = 18
    _ENABLE_GRASPNET = os.environ.get("ATEC_ENABLE_GRASPNET", "0").strip().lower() in ("1", "true", "yes", "on")
    _GRASPNET_BASELINE_ROOT = os.environ.get(
        "ATEC_GRASPNET_BASELINE_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "graspnet-baseline-ref"),
    )
    _GRASPNET_API_ROOT = os.environ.get(
        "ATEC_GRASPNET_API_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "graspnetAPI-ref"),
    )
    _GRASPNET_CKPT = os.environ.get(
        "ATEC_GRASPNET_CKPT",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "third_party",
            "graspnet-baseline-ref",
            "checkpoints",
            "checkpoint-rs.tar",
        ),
    )
    _GRASPNET_NUM_POINT = int(os.environ.get("ATEC_GRASPNET_NUM_POINT", "20000"))
    _GRASPNET_CLUSTER_RADIUS = float(os.environ.get("ATEC_GRASPNET_CLUSTER_RADIUS", "0.12"))
    _GRASPNET_CANDIDATE_POOL = int(os.environ.get("ATEC_GRASPNET_CANDIDATE_POOL", "900"))
    _GRASPNET_MULTIVIEW = os.environ.get("ATEC_GRASPNET_MULTIVIEW", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )
    _GRASPNET_BANANA_MULTIVIEW = os.environ.get("ATEC_GRASPNET_BANANA_MULTIVIEW", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )
    _GRASPNET_VERTICAL_PATH = os.environ.get("ATEC_GRASPNET_VERTICAL_PATH", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )
    _GRASPNET_USE_YELLOW_MASK = os.environ.get("ATEC_GRASPNET_USE_YELLOW_MASK", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )
    _GRASPNET_LIFT_HEIGHT = float(os.environ.get("ATEC_GRASPNET_LIFT_HEIGHT", "0.16"))
    _GRASPNET_MUSTARD_SAFE_PRE_Z = float(os.environ.get("ATEC_GRASPNET_MUSTARD_SAFE_PRE_Z", "1.10"))
    _GRASPNET_MAX_JOINT_DELTA = float(os.environ.get("ATEC_GRASPNET_MAX_JOINT_DELTA", "0.10"))
    _GRASPNET_MUSTARD_OPEN_POS = float(os.environ.get("ATEC_GRASPNET_MUSTARD_OPEN_POS", "0.040"))
    _GRASPNET_VIEW_MOVE_STEPS = int(os.environ.get("ATEC_GRASPNET_VIEW_MOVE_STEPS", "68"))
    _GRASPNET_VIEW_SETTLE_STEPS = int(os.environ.get("ATEC_GRASPNET_VIEW_SETTLE_STEPS", "8"))
    _GRASPNET_DETECT_HANDOFF_STEPS = int(os.environ.get("ATEC_GRASPNET_DETECT_HANDOFF_STEPS", "10"))
    _GRASPNET_USE_DETECT_CENTER_FALLBACK = os.environ.get(
        "ATEC_GRASPNET_USE_DETECT_CENTER_FALLBACK", "1"
    ).strip().lower() in ("1", "true", "yes", "on")
    _GRASPNET_PREFER_LOCKED_CENTER = os.environ.get(
        "ATEC_GRASPNET_PREFER_LOCKED_CENTER", "1"
    ).strip().lower() in ("1", "true", "yes", "on")
    _GRASPNET_FIXED_EXTRINSIC = os.environ.get("ATEC_GRASPNET_FIXED_EXTRINSIC", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )
    _GRASPNET_MUSTARD_TOPDOWN_STRAIGHT = os.environ.get(
        "ATEC_GRASPNET_MUSTARD_TOPDOWN_STRAIGHT", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    _GRASPNET_DISABLE_MUSTARD_TOPDOWN_STRAIGHT = os.environ.get(
        "ATEC_GRASPNET_DISABLE_MUSTARD_TOPDOWN_STRAIGHT", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    _GRASPNET_TARGET_OBJECT = os.environ.get("ATEC_GRASPNET_TARGET_OBJECT", "").strip().lower()
    _GRASPNET_BOX_EXEC_MODE = os.environ.get("ATEC_GRASPNET_BOX_EXEC_MODE", "geometric").strip().lower()
    _GRASPNET_BOX_EE_Z = float(os.environ.get("ATEC_GRASPNET_BOX_EE_Z", "-1"))
    _GRASPNET_BOX_MAX_WIDTH = float(os.environ.get("ATEC_GRASPNET_BOX_MAX_WIDTH", "0.082"))
    _GRASPNET_BOX_BBOX_EXPAND_LEFT = int(os.environ.get("ATEC_GRASPNET_BOX_BBOX_EXPAND_LEFT", "18"))
    _GRASPNET_BOX_BBOX_EXPAND_RIGHT = int(os.environ.get("ATEC_GRASPNET_BOX_BBOX_EXPAND_RIGHT", "150"))
    _GRASPNET_BOX_BBOX_EXPAND_UP = int(os.environ.get("ATEC_GRASPNET_BOX_BBOX_EXPAND_UP", "125"))
    _GRASPNET_BOX_BBOX_EXPAND_DOWN = int(os.environ.get("ATEC_GRASPNET_BOX_BBOX_EXPAND_DOWN", "8"))
    _GRASPNET_BOX_DEPTH_SLOP = float(os.environ.get("ATEC_GRASPNET_BOX_DEPTH_SLOP", "0.075"))
    _GRASPNET_BANANA_PRE_OFFSET = float(os.environ.get("ATEC_GRASPNET_BANANA_PRE_OFFSET", "0.12"))
    _GRASPNET_BANANA_LIFT_HEIGHT = float(os.environ.get("ATEC_GRASPNET_BANANA_LIFT_HEIGHT", "0.18"))
    _GRASPNET_BANANA_EE_Z = float(os.environ.get("ATEC_GRASPNET_BANANA_EE_Z", "-1"))
    _GRASPNET_BANANA_SAFE_PRE_Z = float(os.environ.get("ATEC_GRASPNET_BANANA_SAFE_PRE_Z", "-1"))
    _GRASPNET_BANANA_X_OFFSET = float(os.environ.get("ATEC_GRASPNET_BANANA_X_OFFSET", "0.0"))
    _GRASPNET_BANANA_Y_OFFSET = float(os.environ.get("ATEC_GRASPNET_BANANA_Y_OFFSET", "0.0"))
    _GRASPNET_BANANA_MIN_DOWNNESS = float(os.environ.get("ATEC_GRASPNET_BANANA_MIN_DOWNNESS", "0.50"))
    _GRASPNET_BANANA_TARGET_Z_FRAC = float(os.environ.get("ATEC_GRASPNET_BANANA_TARGET_Z_FRAC", "0.58"))
    # Geometry-only object selection (no model).
    # Set via env: ATEC_TARGET_OBJECT_CLASS=banana|mustard|sugar|any
    _TARGET_OBJECT_CLASS = os.environ.get("ATEC_TARGET_OBJECT_CLASS", "any")
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )

    def calculate_velocity(
        self, target_x, target_y, max_vx, max_vy, max_wz, k_v=0.5, k_w=1.0, *, lock_on_arrive=True
    ):
        """
        target_x, target_y: 目标相对坐标
        max_vx, max_vy, max_wz: 速度上限
        k_v: 线性速度增益
        k_w: 角速度增益
        """
        # 1. 计算距离和角度
        dist = np.sqrt(target_x ** 2 + target_y ** 2)
        angle_to_target = np.arctan2(target_y, target_x)

        # 2. 计算目标速度 (随着距离减小，速度线性下降)
        # 使用 min(k*dist, max_v) 实现限幅
        vx = np.clip(k_v * target_x, -max_vx, max_vx)
        vy = np.clip(k_v * target_y, -max_vy, max_vy)

        # 3. 角速度控制 (假设机器人车头朝向为X轴)
        # 仅当目标距离较远时才大幅旋转，近距离时减小旋转以防震荡
        wz = np.clip(k_w * angle_to_target, -max_wz, max_wz)

        # 4. 到达阈值后连续多帧确认再进入 LOCK，降低单帧噪声误触发。
        if lock_on_arrive and self.status == Status.SEARCH:
            if not hasattr(self, "_lock_arrive_confirm_count"):
                self._lock_arrive_confirm_count = 0
            if dist < self._LOCK_ARRIVE_DIST:
                self._lock_arrive_confirm_count += 1
                if self._lock_arrive_confirm_count >= max(self._LOCK_ARRIVE_CONFIRM_STEPS, 1):
                    self._begin_lock_prezero(
                        f"arrive dist={dist:.2f}<{self._LOCK_ARRIVE_DIST:.2f}"
                        f" confirm={self._lock_arrive_confirm_count}"
                    )
                    return 0.0, 0.0, 0.0
            else:
                self._lock_arrive_confirm_count = 0
        # 与 SEARCH 相同：写死 cmd，供策略 obs 里的 velocity_commands 使用
        self.cmd_max_vy = 0.0
        wz_cmd = float(1.4 * -target_x)
        if dist < self._SEARCH_DECEL_START_DIST:
            t = float(np.clip(dist / self._SEARCH_DECEL_START_DIST, 0.0, 1.0))
            vx_cmd = self._SEARCH_SLOW_VX + (self._SEARCH_CRUISE_VX - self._SEARCH_SLOW_VX) * t
            wz_cmd *= t
        else:
            vx_cmd = self._SEARCH_CRUISE_VX
        self.cmd_max_wz = float(np.clip(wz_cmd, -1.0, 1.0))
        self.cmd_max_vx = vx_cmd
        return vx, vy, wz

    def _begin_lock_prezero(self, reason: str) -> None:
        self.status = Status.LOCK
        self.get_down = False
        self._lock_arrive_confirm_count = 0
        self._lock_commit_confirm_count = 0
        self._lock_prepare_start_idx = self.cur_idx
        self.cmd_max_vx = 0.0
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = 0.0
        self._lock_prezero_settle_count = 0
        print(
            f"[TaskB] -> LOCK pre-zero ({self._LOCK_PRE_ZERO_STEPS} steps, {reason}): "
            "vx/vy/wz=0 hold",
            flush=True,
        )
        print(
            "[TaskB][LOCK] search_cloud_height "
            f"src={getattr(self, '_search_last_cloud_src', None)} "
            f"cloud_zmax={getattr(self, '_search_last_cloud_zmax', None)} "
            f"target_zmax={getattr(self, '_search_last_target_zmax', None)}",
            flush=True,
        )

    def _find_search_target_from_obs(self, obs) -> tuple[np.ndarray | None, float | None]:
        head_depth = self._obs_head_depth(obs)
        target, min_dist = (None, None)
        if head_depth is not None:
            target, min_dist = self.find_target_by_depth(head_depth)
            if target is not None:
                self._search_last_cloud_src = "head"
        if target is None:
            ee_depth = self._obs_depth(obs, "ee_depth")
            if ee_depth is not None:
                target, min_dist = self.find_target_by_depth(ee_depth)
                if target is not None:
                    self._search_last_cloud_src = "ee"
        return target, min_dist

    def _search_near_commit_zone(self) -> bool:
        if self._search_last_seen_dist is None or self._search_last_seen_step is None:
            return False
        return (
            self._search_last_seen_dist <= self._LOCK_LOST_COMMIT_DIST
            and (self.cur_idx - self._search_last_seen_step) <= self._LOCK_LOST_COMMIT_STEPS
        )

    def _search_target_implies_near_loss(
        self,
        target: np.ndarray | None,
        min_dist: float | None,
    ) -> tuple[bool, str]:
        if not self._search_near_commit_zone():
            return False, ""
        # Avoid aggressive false lock-ins on transient target dropouts/noise.
        # Only commit to LOCK when we currently still have a valid target.
        if target is None or min_dist is None:
            return False, ""
        last_dist = float(self._search_last_seen_dist)
        if float(min_dist) > self._LOCK_LOST_COMMIT_DIST:
            return True, (
                f"far_cluster min_dist={float(min_dist):.2f}"
                f">commit={self._LOCK_LOST_COMMIT_DIST:.2f}"
                f" last_dist={last_dist:.2f}"
            )
        if self._search_track_target is not None:
            jump = float(np.linalg.norm(target - self._search_track_target))
            if jump > self._SEARCH_TRACK_MAX_JUMP:
                return True, (
                    f"track_jump={jump:.2f}>{self._SEARCH_TRACK_MAX_JUMP:.2f}"
                    f" last_dist={last_dist:.2f}"
                )
        return False, ""

    def _platform_deploy(self) -> bool:
        """True on competition server (no Isaac bind_env)."""
        return self._env is None

    @staticmethod
    def _numpy_depth_2d(depth) -> np.ndarray | None:
        if depth is None:
            return None
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
        arr = np.squeeze(np.asarray(depth, dtype=np.float64))
        if arr.ndim != 2:
            return None
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _obs_depth(self, obs, key: str) -> np.ndarray | None:
        image = obs.get("image")
        if not image or key not in image:
            return None
        return self._numpy_depth_2d(image[key][0])

    def _obs_head_depth(self, obs) -> np.ndarray | None:
        """Task B platform sends head_depth; Task D/E fallback is video_depth."""
        depth = self._obs_depth(obs, "head_depth")
        if depth is not None:
            return depth
        return self._obs_depth(obs, "video_depth")

    def _obs_head_rgb(self, obs) -> np.ndarray | None:
        rgb = self._obs_rgb(obs, "head_rgb")
        if rgb is not None:
            return rgb
        return self._obs_rgb(obs, "video_rgb")

    def _obs_rgb(self, obs, key: str) -> np.ndarray | None:
        image = obs.get("image")
        if not image or key not in image:
            return None
        arr = image[key][0]
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        else:
            arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[..., :3]
        return None

    def _obs_extero(self, obs) -> np.ndarray | None:
        extero = obs.get("extero")
        if extero is None:
            return None
        return extero.to(self.device).cpu().numpy()[0]

    @staticmethod
    def _rgb_yellow_ratio(rgb: np.ndarray) -> float:
        """Simple yellow ratio from RGB patch, robust enough for bin-vs-trash heuristic."""
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.size == 0 or arr.ndim != 3 or arr.shape[-1] < 3:
            return 0.0
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        bright = (r > 80.0) & (g > 80.0)
        yellow = bright & (b < 130.0) & (r > g * 0.75) & (r < g * 1.35)
        total = int(arr.shape[0] * arr.shape[1])
        if total <= 0:
            return 0.0
        return float(np.count_nonzero(yellow)) / float(total)

    def _search_target_bin_rgb_lidar(self, obs, target: np.ndarray) -> tuple[bool, dict]:
        """Fuse LiDAR dustbin bearing + RGB yellow ratio for bin discrimination."""
        info: dict[str, float | None] = {
            "bin_bearing": None,
            "target_bearing": None,
            "bearing_err": None,
            "yellow_ratio": None,
        }
        if not self._SEARCH_BIN_RGB_ENABLE:
            return False, info
        extero = self._obs_extero(obs)
        head_depth = self._obs_head_depth(obs)
        fused = approach_dustbin(extero, head_depth, self.K)
        bin_bearing = fused.get("bearing")
        if bin_bearing is None:
            return False, info
        target_bearing = float(np.arctan2(float(target[1]), float(target[0])))
        bearing_err = abs(self._wrap_angle(target_bearing - float(bin_bearing)))
        info["bin_bearing"] = float(bin_bearing)
        info["target_bearing"] = target_bearing
        info["bearing_err"] = bearing_err
        if bearing_err > self._SEARCH_BIN_RGB_BEARING_TOL:
            return False, info

        rgb = self._obs_head_rgb(obs)
        if rgb is None:
            # If RGB missing but LiDAR/bin bearing strongly aligns, keep conservative bin-positive.
            return True, info
        h, w, _ = rgb.shape
        fx, cx = float(self.K[0, 0]), float(self.K[0, 2])
        cols: list[int] = []
        half_rad = float(np.radians(self._SEARCH_BIN_RGB_HALF_DEG))
        for u in range(w):
            ang = np.arctan2(-(u - cx), fx)
            if abs(self._wrap_angle(ang - float(bin_bearing))) <= half_rad:
                cols.append(u)
        if len(cols) == 0:
            return False, info
        v0 = max(0, min(int(self._SEARCH_BIN_RGB_V0), h - 1))
        v1 = max(v0 + 1, min(int(self._SEARCH_BIN_RGB_V1), h))
        patch = rgb[v0:v1, cols]
        yellow_ratio = self._rgb_yellow_ratio(patch)
        info["yellow_ratio"] = yellow_ratio
        return bool(yellow_ratio <= self._SEARCH_BIN_RGB_MAX_YELLOW_RATIO), info

    def _begin_search_cycle(self) -> None:
        self.status = Status.SEARCH
        self.get_down = False
        self._lock_arrive_confirm_count = 0
        self._lock_commit_confirm_count = 0
        self.start_get_down_idx = None
        self.start_pick_idx = None
        self.start_stand_idx = None
        self.v_list = [0.0 for _ in range(8)]
        self.cmd_max_vx = 1.0
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = 1.0
        self._last_bin_bearing = None
        self._last_bin_dist = None
        self._ik_phase = None
        self._ik_phase_start = None
        self._ik_target_pos_b = None
        self._ik_lift_pos_b = None
        self._ik_acquire_pos_b = None
        self._ik_no_target_count = 0
        self._last_target_cam = None
        self._ik_pick_active = False
        self._pick_arm_hold_cmd = None
        self._grasp_target_cam = None
        self._grasp_axis_cam = None
        self._grasp_target_cam_ema = None
        self._grasp_candidate_cams: list[np.ndarray] = []
        self._grasp_tried_cams: list[np.ndarray] = []
        self._grasp_reached_hold_count = 0
        self._grasp_prev_l_err = None
        self._grasp_prev_f_err = None
        self._grasp_prev_v_err = None
        self._grasp_wrist_yaw_target = None
        self._lock_prepare_start_idx = None
        self._lock_prezero_settle_count = 0
        self._lock_arrive_confirm_count = 0
        self._lock_commit_confirm_count = 0
        self._squat_start_leg_action = None
        self._search_track_target = None
        self._search_last_seen_dist = None
        self._search_last_seen_step = None
        self._search_memory_age = None
        self._search_last_cloud_zmax = None
        self._search_last_target_zmax = None
        self._search_last_cloud_src = None
        self._grasp_sm = None
        self._grasp_obj_idx = None
        self._grasp_ik_active = False
        self._graspnet_plan = None
        self._graspnet_sm = None
        self._graspnet_scan_phase = None
        self._graspnet_scan_view_idx = 0
        self._graspnet_scan_step = 0
        self._graspnet_scan_plans: list = []
        self._graspnet_selected_view_action = None
        self._graspnet_handoff_action = None
        self._graspnet_handoff_step = 0
        self._t_gripper_camera = None
        self._reset_grasp_record_buffers()
        print("[TaskB] bin reached -> SEARCH (loop)", flush=True)

    def _update_go_bin_cmd(self, obs) -> None:
        fused = approach_dustbin(
            self._obs_extero(obs),
            self._obs_head_depth(obs),
            self.K,
        )
        bearing = fused.get("bearing")
        dist = fused.get("dist")
        self._last_bin_bearing = bearing
        self._last_bin_dist = dist

        if dist is not None and float(dist) < self._BIN_ARRIVE_DIST:
            self._begin_search_cycle()
            return

        if bearing is None:
            self.cmd_max_vx = 0.0
            self.cmd_max_vy = 0.0
            self.cmd_max_wz = 0.0
            return

        bearing_f = float(bearing)
        if abs(bearing_f) > self._GO_BIN_ALIGN_RAD:
            # 先原地对准 LiDAR bearing，不要 vx=1.5 硬冲
            self.cmd_max_vx = 0.0
            self.cmd_max_vy = 0.0
            self.cmd_max_wz = float(
                np.clip(self._GO_BIN_WZ_GAIN * bearing_f, -0.8, 0.8)
            )
            return

        self.cmd_max_vx = self._GO_BIN_VX
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = float(np.clip(0.4 * bearing_f, -0.4, 0.4))

    def _apply_bin_collision_guard(self, obs) -> bool:
        """Near-bin safety guard: back off and turn away."""
        if not self._BIN_GUARD_ENABLE or self.status == Status.GO_BIN:
            return False
        fused = approach_dustbin(
            self._obs_extero(obs),
            self._obs_head_depth(obs),
            self.K,
        )
        bearing = fused.get("bearing")
        dist = fused.get("dist")
        self._last_bin_bearing = bearing
        self._last_bin_dist = dist
        if dist is None or float(dist) > self._BIN_GUARD_DIST:
            return False

        bearing_f = 0.0 if bearing is None else float(bearing)
        self.cmd_max_vx = self._BIN_GUARD_REVERSE_VX
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = float(np.clip(-self._BIN_GUARD_WZ_GAIN * bearing_f, -self._BIN_GUARD_MAX_WZ, self._BIN_GUARD_MAX_WZ))
        self._set_video_hud(
            f"status={self.status.name} phase=bin-guard dist={float(dist):.2f} bearing={bearing_f:.2f}",
            f"cmd_vx={self.cmd_max_vx:.2f} cmd_wz={self.cmd_max_wz:.2f}",
        )
        return True

    def _get_robot_world_xy_yaw(self) -> tuple[float, float, float] | None:
        if self._env is None:
            return None
        try:
            root_pos = self.robot.data.root_pos_w[0].detach().cpu().numpy()
            quat = self.robot.data.root_quat_w[0].detach().cpu().numpy()  # (w, x, y, z)
            w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            return float(root_pos[0]), float(root_pos[1]), yaw
        except Exception:
            return None

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def _update_search_coverage_cmd(self) -> bool:
        if not self._SEARCH_COVERAGE_ENABLE or len(self._SEARCH_WAYPOINTS) == 0:
            return False
        pose = self._get_robot_world_xy_yaw()
        if pose is None:
            return False
        rx, ry, ryaw = pose
        idx = int(self._search_wp_idx) % len(self._SEARCH_WAYPOINTS)
        tx, ty = self._SEARCH_WAYPOINTS[idx]
        dx = float(tx - rx)
        dy = float(ty - ry)
        dist = float(np.hypot(dx, dy))
        if dist < self._SEARCH_WP_REACHED_DIST:
            self._search_wp_idx = (idx + 1) % len(self._SEARCH_WAYPOINTS)
            idx = int(self._search_wp_idx) % len(self._SEARCH_WAYPOINTS)
            tx, ty = self._SEARCH_WAYPOINTS[idx]
            dx = float(tx - rx)
            dy = float(ty - ry)
            dist = float(np.hypot(dx, dy))
        target_yaw = float(np.arctan2(dy, dx))
        yaw_err = self._wrap_angle(target_yaw - ryaw)
        vx = min(self._SEARCH_WP_VX_CAP, max(0.2, 0.5 * dist))
        wz = float(np.clip(0.9 * yaw_err, -self._SEARCH_WP_WZ_CAP, self._SEARCH_WP_WZ_CAP))
        if abs(yaw_err) > 0.65:
            vx = 0.0
        self.cmd_max_vx = float(vx)
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = wz
        self._set_video_hud(
            f"status=SEARCH phase=coverage wp={idx}/{len(self._SEARCH_WAYPOINTS)} dist={dist:.2f}",
            f"cmd_vx={self.cmd_max_vx:.2f} cmd_wz={self.cmd_max_wz:.2f}",
        )
        return True

    def bind_env(self, env) -> None:
        self._env = env
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self.robot = unwrapped.scene.articulations["robot"]
        self.device = str(self.robot.device)
        self._leg_joint_ids = None
        try:
            leg_ids, _ = self.robot.find_joints(list(self._LEG_JOINT_NAMES))
            self._leg_joint_ids = list(leg_ids)
            self._default_leg_stiffness = self.robot.data.joint_stiffness[0, self._leg_joint_ids].clone()
            self._default_leg_damping = self.robot.data.joint_damping[0, self._leg_joint_ids].clone()
        except Exception as exc:
            print(f"[TaskB] bind_env leg PD setup failed: {exc}", flush=True)
        if self._ENABLE_GRASPNET or self._GRASP_USE_TASKE_IK:
            self._setup_task_e_grasp_ik()
        elif self._ENABLE_DEPTH_IK_PICK:
            self._setup_depth_ik_pick()
        self._init_graspnet()
        if self._ENABLE_GRASPNET:
            self._calibrate_graspnet_camera_extrinsic()
        print("[TaskB] bind_env OK — sim access + squat leg PD enabled.", flush=True)

    def _init_graspnet(self) -> None:
        self._graspnet_runner = None
        self._graspnet_ready = False
        if not self._ENABLE_GRASPNET:
            return
        if not _GRASPNET_PICK_AVAILABLE or GraspNetTaskBRunner is None:
            print("[TaskB][GraspNet] demo/graspnet_pick unavailable.", flush=True)
            return
        if not os.path.isfile(self._GRASPNET_CKPT):
            print(f"[TaskB][GraspNet] checkpoint missing: {self._GRASPNET_CKPT}", flush=True)
            return
        try:
            plan_cfg = self._build_grasp_plan_config()
            self._graspnet_runner = GraspNetTaskBRunner(
                baseline_root=self._GRASPNET_BASELINE_ROOT,
                api_root=self._GRASPNET_API_ROOT,
                checkpoint_path=self._GRASPNET_CKPT,
                num_point=self._GRASPNET_NUM_POINT,
                cluster_radius=self._GRASPNET_CLUSTER_RADIUS,
                candidate_pool=self._GRASPNET_CANDIDATE_POOL,
                lift_height=self._GRASPNET_LIFT_HEIGHT,
                plan_cfg=plan_cfg,
            )
            self._graspnet_ready = True
            target = self._graspnet_target_object()
            print(
                f"[TaskB][GraspNet] ready ckpt={self._GRASPNET_CKPT} "
                f"pool={self._GRASPNET_CANDIDATE_POOL} target={target} "
                f"multiview={self._graspnet_use_multiview()} fixed_ext={self._GRASPNET_FIXED_EXTRINSIC}",
                flush=True,
            )
        except Exception as exc:
            self._graspnet_runner = None
            self._graspnet_ready = False
            print(f"[TaskB][GraspNet] init failed: {exc}", flush=True)

    def _setup_task_e_grasp_ik(self) -> None:
        """Task E collect_demos_task_e.py IK: pose CartesianController + grasp SM."""
        if TaskEGraspOnlySM is None or CartesianController is None:
            print("[TaskB][GraspIK] task_e helpers unavailable, fallback to visual GRASP.", flush=True)
            return
        try:
            self.ee_name = self._resolve_ee_body_name()
            self.arm_ids = self._resolve_joint_ids(self.ARM_JOINT_NAME_CANDIDATES)
            all_names = list(getattr(self.robot.data, "joint_names", []))
            gripper_name_candidates = [
                "joint7", "joint8",
                "arm_joint7", "arm_joint8",
                "left_finger_joint", "right_finger_joint",
            ]
            gripper_names = [n for n in gripper_name_candidates if n in all_names]
            if len(gripper_names) >= 2:
                self.gripper_ids, _ = self.robot.find_joints(gripper_names[:2])
                self.gripper_ids = list(self.gripper_ids)
            else:
                self.gripper_ids = [18, 19]
            self.default_joint_pos = self.robot.data.default_joint_pos.clone()
            dev = self.device
            dtype = self.robot.data.joint_pos.dtype
            self.gripper_open_pos = torch.tensor([list(GRIPPER_OPEN_POS)], device=dev, dtype=dtype)
            self.gripper_close_pos = torch.tensor([list(GRIPPER_CLOSE_POS)], device=dev, dtype=dtype)
            num_envs = int(getattr(self._env.unwrapped, "num_envs", self.robot.data.joint_pos.shape[0]))
            max_joint_delta = (
                self._GRASPNET_MAX_JOINT_DELTA
                if self._ENABLE_GRASPNET
                else 0.2
            )
            self.cartesian_ctrl = CartesianController(
                robot=self.robot,
                ee_body_name=self.ee_name,
                arm_joint_names=self.arm_joint_names,
                num_envs=num_envs,
                device=dev,
                command_type="pose",
                lambda_val=0.05,
                max_joint_delta=max_joint_delta,
            )
            print(
                f"[TaskB][GraspIK] Task E IK enabled ee={self.ee_name}, "
                f"arm={self.arm_joint_names}, gripper_ids={self.gripper_ids}",
                flush=True,
            )
        except Exception as exc:
            self.cartesian_ctrl = None
            print(f"[TaskB][GraspIK] init failed: {exc}; fallback to visual GRASP.", flush=True)

    def _setup_depth_ik_pick(self) -> None:
        if not self._ENABLE_DEPTH_IK_PICK:
            return
        if CartesianController is None or subtract_frame_transforms is None:
            print("[TaskB][IK] dependency unavailable, fallback to scripted PICK.", flush=True)
            return
        try:
            self.ee_name = self._resolve_ee_body_name()
            self.arm_ids = self._resolve_joint_ids(self.ARM_JOINT_NAME_CANDIDATES)
            all_names = list(getattr(self.robot.data, "joint_names", []))
            gripper_name_candidates = [
                "joint7", "joint8",
                "arm_joint7", "arm_joint8",
                "left_finger_joint", "right_finger_joint",
            ]
            gripper_names = [n for n in gripper_name_candidates if n in all_names]
            if len(gripper_names) >= 2:
                self.gripper_ids, _ = self.robot.find_joints(gripper_names[:2])
                self.gripper_ids = list(self.gripper_ids)
            else:
                # Fallback: use the last two non-leg joints as gripper.
                self.gripper_ids = [18, 19]
            self.default_joint_pos = self.robot.data.default_joint_pos.clone()
            self.gripper_open_pos = self.default_joint_pos[:, self.gripper_ids].clone()
            close_delta = torch.tensor([[0.03, -0.03]], device=self.device, dtype=torch.float32)
            self.gripper_close_pos = self.gripper_open_pos + close_delta
            num_envs = int(getattr(self._env.unwrapped, "num_envs", self.robot.data.joint_pos.shape[0]))
            self.cartesian_ctrl = CartesianController(
                robot=self.robot,
                ee_body_name=self.ee_name,
                arm_joint_names=self.arm_joint_names,
                num_envs=num_envs,
                device=self.device,
                command_type="position",
                lambda_val=0.08,
                max_joint_delta=0.08,
            )
            print(
                f"[TaskB][IK] enabled ee={self.ee_name}, arm={self.arm_joint_names}, "
                f"gripper_ids={self.gripper_ids}",
                flush=True,
            )
        except Exception as e:
            self.cartesian_ctrl = None
            print(f"[TaskB][IK] init failed: {e}; fallback to scripted PICK.", flush=True)

    def _set_squat_leg_pd(self, enabled: bool) -> None:
        if self._env is None or not hasattr(self, "_leg_joint_ids"):
            return
        if enabled == self._squat_pd_active:
            return
        self._squat_pd_active = enabled
        if enabled:
            self.robot.write_joint_stiffness_to_sim(
                self._SQUAT_LEG_STIFFNESS, joint_ids=self._leg_joint_ids
            )
            self.robot.write_joint_damping_to_sim(
                self._SQUAT_LEG_DAMPING, joint_ids=self._leg_joint_ids
            )
            return
        self.robot.write_joint_stiffness_to_sim(
            self._default_leg_stiffness, joint_ids=self._leg_joint_ids
        )
        self.robot.write_joint_damping_to_sim(
            self._default_leg_damping, joint_ids=self._leg_joint_ids
        )

    def _leg_action_from_proprio(self, obs) -> list[float]:
        if self._env is not None and self._leg_joint_ids is not None:
            q = self.robot.data.joint_pos[0, self._leg_joint_ids].detach().cpu().numpy().astype(np.float64)
            q0 = self.robot.data.default_joint_pos[0, self._leg_joint_ids].detach().cpu().numpy().astype(np.float64)
            return ((q - q0) / self._LEG_ACTION_SCALE).tolist()
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        # Platform path (no bind_env): hold legs by reusing last commanded env action
        # from proprio's actions segment, avoiding absolute/relative joint-pos ambiguity.
        act_start = 12 + 2 * action_dim
        act_end = act_start + action_dim
        if act_end <= int(proprio.shape[-1]):
            actions_all = proprio[0, act_start:act_end].detach().cpu().numpy()
            return actions_all[self.leg_joint_indices].astype(np.float64).tolist()
        # Fallback if observation layout is unexpected.
        joint_pos = proprio[0, 12:12 + action_dim].detach().cpu().numpy()
        leg = joint_pos[self.leg_joint_indices]
        return (leg / self._LEG_ACTION_SCALE).astype(np.float64).tolist()

    def _snapshot_squat_start_leg_action(self, obs=None) -> None:
        if self._env is not None and self._leg_joint_ids is not None:
            q = self.robot.data.joint_pos[0, self._leg_joint_ids].detach().cpu().numpy().astype(np.float64)
            q0 = self.robot.data.default_joint_pos[0, self._leg_joint_ids].detach().cpu().numpy().astype(np.float64)
            self._squat_start_leg_action = ((q - q0) / self._LEG_ACTION_SCALE).tolist()
            return
        if obs is not None:
            self._squat_start_leg_action = self._leg_action_from_proprio(obs)
            return
        self._squat_start_leg_action = [0.0 for _ in range(12)]

    def _lock_squat_target_leg_action(self) -> list[float]:
        hip = self._LOCK_SPLAY_HIP_MAG
        thigh = self._LOCK_SQUAT_THIGH
        calf = self._LOCK_SQUAT_CALF
        return [
            -hip, thigh, calf,  # FR
            +hip, thigh, calf,  # FL
            -hip, thigh, calf,  # RR
            +hip, thigh, calf,  # RL
        ]

    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _leg_motion_progress_alpha(self, elapsed: int, total_steps: int) -> float:
        """Slow early + slow late leg ramp."""
        t = float(elapsed) / max(float(total_steps), 1.0)
        t = float(np.clip(t, 0.0, 1.0))
        mid_time = 0.72
        mid_depth = 0.78
        if t <= mid_time:
            u = self._smoothstep01(t / max(mid_time, 1e-6))
            return mid_depth * u
        u = self._smoothstep01((t - mid_time) / max(1.0 - mid_time, 1e-6))
        return mid_depth + (1.0 - mid_depth) * u

    def _squat_progress_alpha(self, squat_elapsed: int) -> float:
        return self._leg_motion_progress_alpha(squat_elapsed, self._SQUAT_STEPS)

    def _lock_body_speed(self, obs=None) -> tuple[float, float]:
        if self._env is not None:
            v = self.robot.data.root_lin_vel_w[0].detach()
            horiz = float(torch.linalg.vector_norm(v[:2]).cpu())
            vert = abs(float(v[2].cpu()))
            return horiz, vert
        if obs is not None and "proprio" in obs:
            proprio = obs["proprio"]
            if torch.is_tensor(proprio):
                v = proprio[0, :3].detach().cpu().numpy()
            else:
                v = np.asarray(proprio, dtype=np.float64).reshape(-1)[:3]
            horiz = float(np.linalg.norm(v[:2]))
            vert = abs(float(v[2]))
            return horiz, vert
        return 0.0, 0.0

    def _root_height_z(self) -> float | None:
        if self._env is None:
            return None
        try:
            return float(self.robot.data.root_pos_w[0, 2].detach().cpu())
        except Exception:
            return None

    def _lock_prezero_ready_for_squat(self, pre_elapsed: int, obs=None) -> bool:
        if pre_elapsed < self._LOCK_PRE_ZERO_STEPS:
            return False
        horiz, vert = self._lock_body_speed(obs)
        if (
            horiz <= self._LOCK_PRE_ZERO_MAX_HSPEED
            and vert <= self._LOCK_PRE_ZERO_MAX_VSPEED
        ):
            self._lock_prezero_settle_count += 1
        else:
            self._lock_prezero_settle_count = 0
        if self._lock_prezero_settle_count >= self._LOCK_PRE_ZERO_SETTLE_STEPS:
            return True
        if pre_elapsed >= self._LOCK_PRE_ZERO_MAX_STEPS:
            return (
                horiz <= self._LOCK_PRE_ZERO_FORCE_HSPEED
                and vert <= self._LOCK_PRE_ZERO_FORCE_VSPEED
            )
        return False

    def _lock_squat_leg_action(self, squat_elapsed: int) -> list[float]:
        target = np.asarray(self._lock_squat_target_leg_action(), dtype=np.float64)
        if self._squat_start_leg_action is None or len(self._squat_start_leg_action) != 12:
            return target.tolist()
        recenter_steps = max(int(self._LOCK_RECENTER_STEPS), 0)
        if recenter_steps > 0 and squat_elapsed < recenter_steps:
            # Briefly recenter legs to neutral pose before deep squat.
            alpha0 = self._smoothstep01(float(squat_elapsed) / max(float(recenter_steps), 1.0))
            start = np.asarray(self._squat_start_leg_action, dtype=np.float64)
            neutral = np.zeros(12, dtype=np.float64)
            return (start * (1.0 - alpha0) + neutral * alpha0).tolist()
        alpha = self._squat_progress_alpha(max(squat_elapsed - recenter_steps, 0))
        start = np.asarray(self._squat_start_leg_action, dtype=np.float64)
        return (start * (1.0 - alpha) + target * alpha).tolist()

    def _resolve_lock_grasp_policy_path(self, demo_dir: str) -> str | None:
        p = os.path.join(demo_dir, "down.pt")
        if os.path.isfile(p):
            return p
        print(f"[TaskB] lock/grasp policy not found: {p}", flush=True)
        return None

    def _resolve_stand_policy_path(self, demo_dir: str) -> str | None:
        p = os.path.join(demo_dir, "stand.pt")
        if os.path.isfile(p):
            return p
        print(f"[TaskB] stand policy not found: {p}", flush=True)
        return None

    def __init__(self):
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        policy_path = demo_dir + '/policy.pt'
        print(policy_path)
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        self._lock_grasp_policy = None
        self._lock_grasp_policy_path = None
        self._stand_policy = None
        self._stand_policy_path = None
        alt_policy_path = self._resolve_lock_grasp_policy_path(demo_dir)
        if alt_policy_path is not None:
            try:
                self._lock_grasp_policy = torch.jit.load(alt_policy_path, map_location=self.device)
                self._lock_grasp_policy.eval()
                self._lock_grasp_policy_path = alt_policy_path
                print(f"[TaskB] lock/grasp locomotion policy: {alt_policy_path}", flush=True)
            except Exception as e:
                self._lock_grasp_policy = None
                self._lock_grasp_policy_path = None
                print(f"[TaskB] lock/grasp policy load failed: {e}", flush=True)
        if self._lock_grasp_policy is None:
            print("[TaskB] lock/grasp locomotion policy unavailable; fallback to default policy.", flush=True)
        stand_policy_path = self._resolve_stand_policy_path(demo_dir)
        if stand_policy_path is not None:
            try:
                self._stand_policy = torch.jit.load(stand_policy_path, map_location=self.device)
                self._stand_policy.eval()
                self._stand_policy_path = stand_policy_path
                print(f"[TaskB] stand locomotion policy: {stand_policy_path}", flush=True)
            except Exception as e:
                self._stand_policy = None
                self._stand_policy_path = None
                print(f"[TaskB] stand policy load failed: {e}", flush=True)
        if self._stand_policy is None:
            print("[TaskB] stand locomotion policy unavailable; fallback to default policy.", flush=True)

        self.leg_action_dim = 12
        self.arm_action_dim = 8

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5],
            device=self.device, dtype=torch.float32
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0],
            device=self.device, dtype=torch.float32
        ).view(1, -1)

        self.arm_default_action = torch.zeros((1, self.arm_action_dim), device=self.device, dtype=torch.float32)

        self.cmd_max_vx = 1.0
        self.cmd_max_vy = 0.5
        self.cmd_max_wz = 1.0
        self.status = Status.SEARCH
        self.start_stand_idx = None
        self._last_bin_bearing = None
        self._last_bin_dist = None
        # ==========================================
        # 虚拟里程计 (用于在世界坐标系下展示点云移动)
        # ==========================================
        self.vis_x = 0.0
        self.vis_y = 0.0
        self.vis_yaw = 0.0

        # 传感器安装高度 (机器人 base 离地高度)
        self.sensor_height = 0.6

        self.first_render = True
        self.debug_printed = False

        # 2. 根据你的 Isaac Lab 配置重建雷达光束的方向
        channels = 16  # 16线
        num_horizontal_rays = 360  # 360度，每度1个点

        # 俯仰角 (Vertical FOV: -20 到 20 度)
        pitch_angles = np.linspace(np.radians(-20.0), np.radians(20.0), channels)
        # 偏航角 (Horizontal FOV: -180 到 180 度)
        yaw_angles = np.linspace(np.radians(-180.0), np.radians(180.0), num_horizontal_rays)

        # 生成网格
        pitch_grid, yaw_grid = np.meshgrid(pitch_angles, yaw_angles, indexing="ij")
        self.pitch_grid = pitch_grid.flatten()
        self.yaw_grid = yaw_grid.flatten()
        self.inited = False
        self.vis = None
        self.pcd = None
        self.coord_frame = None
        self.key = None
        self.auto = False
        self.v_list = [0 for _ in range(8)]
        self.get_down = False
        self.start_get_down_idx = None
        self.start_pick_idx = None
        self.start_pose = None
        self.cur_idx = 0
        self._platform_obs_logged = False
        self._last_platform_score = 0.0
        # Platform server.py: depth (480, 640), cx=320 cy=240
        self.K = np.array(
            [[458.12, 0, SERVER_DEPTH_W / 2.0], [0, 458.12, SERVER_DEPTH_H / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        self._env = None
        self._squat_pd_active = False
        self._target_object_class = self._normalize_target_class(self._TARGET_OBJECT_CLASS)
        self._last_target_class = None
        self._last_target_score = None
        self.cartesian_ctrl = None
        self.ee_name = None
        self.arm_ids = None
        self.gripper_ids = None
        self.default_joint_pos = None
        self.gripper_open_pos = None
        self.gripper_close_pos = None
        self._ik_phase = None
        self._ik_phase_start = None
        self._ik_target_pos_b = None
        self._ik_lift_pos_b = None
        self._ik_acquire_pos_b = None
        self._ik_no_target_count = 0
        self._last_target_cam = None
        self._ik_pick_active = False
        self._pick_arm_hold_cmd = None
        self._anygrasp = None
        self._anygrasp_ready = False
        self._last_anygrasp_score = None
        self._moveit_bridge = None
        self._moveit_ready = False
        self._grasp_target_cam = None
        self._grasp_axis_cam = None
        self._grasp_approach_inited = False
        self._grasp_target_cam_ema = None
        self._grasp_candidate_cams = []
        self._grasp_tried_cams = []
        self._grasp_reached_hold_count = 0
        self._grasp_prev_l_err = None
        self._grasp_prev_f_err = None
        self._grasp_prev_v_err = None
        self._grasp_close_reason = None
        self._video_hud_lines: list[str] = []
        self._lock_prepare_start_idx = None
        self._lock_prezero_settle_count = 0
        self._lock_arrive_confirm_count = 0
        self._squat_start_leg_action = None
        self._search_track_target = None
        self._search_last_seen_dist = None
        self._search_last_seen_step = None
        self._search_memory_age = None
        self._search_last_cloud_zmax = None
        self._search_last_target_zmax = None
        self._search_last_cloud_src = None
        self._search_no_target_steps = 0
        self._search_wp_idx = 0
        self._grasp_sm = None
        self._grasp_obj_idx = None
        self._grasp_ik_active = False
        self._graspnet_runner = None
        self._graspnet_ready = False
        self._graspnet_plan = None
        self._graspnet_sm = None
        self._graspnet_scan_phase = None
        self._graspnet_scan_view_idx = 0
        self._graspnet_scan_step = 0
        self._graspnet_scan_plans = []
        self._graspnet_selected_view_action = None
        self._graspnet_handoff_action = None
        self._graspnet_handoff_step = 0
        self._t_gripper_camera = None
        self._init_anygrasp()
        self._init_moveit_bridge()

    @staticmethod
    def _normalize_target_class(name: str) -> str:
        s = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        alias = {
            "banana": "banana",
            "mustard": "mustard",
            "mustardbottle": "mustard",
            "sugar": "sugar",
            "sugarbox": "sugar",
            "box": "sugar",
            "any": "any",
            "all": "any",
            "*": "any",
        }
        return alias.get(s, "any")

    def _init_anygrasp(self) -> None:
        if not self._ENABLE_ANYGRASP:
            print("[TaskB][AnyGrasp] disabled.", flush=True)
            return
        if not self._ANYGRASP_CKPT:
            print("[TaskB][AnyGrasp] checkpoint not set (ATEC_ANYGRASP_CKPT).", flush=True)
            return
        try:
            module = None
            cls = None
            for mod_name in ("anygrasp", "gsnet", "graspnetAPI.anygrasp"):
                try:
                    module = importlib.import_module(mod_name)
                    if hasattr(module, "AnyGrasp"):
                        cls = getattr(module, "AnyGrasp")
                        break
                except Exception:
                    continue
            if cls is None:
                raise ImportError("Cannot import AnyGrasp class.")
            try:
                self._anygrasp = cls(checkpoint_path=self._ANYGRASP_CKPT)
            except TypeError:
                try:
                    self._anygrasp = cls(checkpoint=self._ANYGRASP_CKPT)
                except TypeError:
                    self._anygrasp = cls(self._ANYGRASP_CKPT)
            if hasattr(self._anygrasp, "load_net"):
                self._anygrasp.load_net()
            elif hasattr(self._anygrasp, "load"):
                try:
                    self._anygrasp.load()
                except TypeError:
                    self._anygrasp.load(self._ANYGRASP_CKPT)
            self._anygrasp_ready = True
            print(f"[TaskB][AnyGrasp] loaded ckpt={self._ANYGRASP_CKPT}", flush=True)
        except Exception as e:
            self._anygrasp = None
            self._anygrasp_ready = False
            print(f"[TaskB][AnyGrasp] init failed: {e}", flush=True)

    def _init_moveit_bridge(self) -> None:
        if not self._ENABLE_MOVEIT_PICK:
            print("[TaskB][MoveIt] disabled.", flush=True)
            return
        try:
            module = importlib.import_module(self._MOVEIT_BRIDGE_MODULE)
            cls = getattr(module, self._MOVEIT_BRIDGE_CLASS)
            self._moveit_bridge = cls()
            self._moveit_ready = True
            print(
                f"[TaskB][MoveIt] bridge ready {self._MOVEIT_BRIDGE_MODULE}.{self._MOVEIT_BRIDGE_CLASS}",
                flush=True,
            )
        except Exception as e:
            self._moveit_bridge = None
            self._moveit_ready = False
            print(f"[TaskB][MoveIt] bridge init failed: {e}", flush=True)

    @staticmethod
    def _grasp_score(g) -> float:
        for key in ("score", "grasp_score", "confidence"):
            if hasattr(g, key):
                return float(getattr(g, key))
            if isinstance(g, dict) and key in g:
                return float(g[key])
        return 0.0

    @staticmethod
    def _grasp_translation(g) -> np.ndarray | None:
        keys = ("translation", "center", "position")
        for key in keys:
            if hasattr(g, key):
                arr = np.asarray(getattr(g, key), dtype=np.float32).reshape(-1)
                if arr.size >= 3:
                    return arr[:3]
            if isinstance(g, dict) and key in g:
                arr = np.asarray(g[key], dtype=np.float32).reshape(-1)
                if arr.size >= 3:
                    return arr[:3]
        return None

    def _find_target_by_anygrasp(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        if not self._anygrasp_ready or self._anygrasp is None or rgb is None or depth is None:
            return None, None
        h, w = depth.shape[:2]
        K = np.asarray(self.K, dtype=np.float32)
        workspace = np.array([[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.2]], dtype=np.float32)
        out = None
        last_e = None
        for fn_name in ("get_grasp", "inference", "predict", "infer"):
            if not hasattr(self._anygrasp, fn_name):
                continue
            fn = getattr(self._anygrasp, fn_name)
            try:
                out = fn(
                    rgb=rgb,
                    depth=depth,
                    camera_intrinsic=K,
                    workspace_limits=workspace,
                    topk=max(1, int(self._ANYGRASP_TOPK)),
                )
                break
            except TypeError:
                try:
                    out = fn(rgb, depth, K, workspace)
                    break
                except Exception as e:
                    last_e = e
            except Exception as e:
                last_e = e
        if out is None:
            if self.cur_idx % self._ANYGRASP_PRINT_EVERY == 0 and last_e is not None:
                print(f"[TaskB][AnyGrasp] inference failed: {last_e}", flush=True)
            return None, None

        grasps = []
        if isinstance(out, (list, tuple)):
            if len(out) > 0 and isinstance(out[0], (list, tuple)):
                grasps = list(out[0])
            else:
                grasps = list(out)
        elif hasattr(out, "grasps"):
            grasps = list(getattr(out, "grasps"))
        elif hasattr(out, "__iter__"):
            grasps = list(out)
        if len(grasps) == 0:
            return None, None

        best_g = max(grasps, key=self._grasp_score)
        trans = self._grasp_translation(best_g)
        if trans is None:
            return None, None

        # Convert to this codebase camera sign convention (z forward is negative).
        cam_point = np.array([trans[0], trans[1], -abs(trans[2])], dtype=np.float32)
        self._last_anygrasp_score = float(self._grasp_score(best_g))
        dist = float(np.linalg.norm(cam_point))
        return cam_point, dist

    @staticmethod
    def _cluster_shape_features(cur_points: np.ndarray) -> dict | None:
        if cur_points.shape[0] < 12:
            return None
        mins = np.min(cur_points, axis=0)
        maxs = np.max(cur_points, axis=0)
        ext = np.maximum(maxs - mins, 1e-6)  # x,y,z extents

        xy = np.maximum(ext[:2], 1e-6)
        long_xy = float(np.max(xy))
        short_xy = float(np.min(xy))
        ratio_xy = long_xy / short_xy
        z_extent = float(ext[2])
        volume = float(ext[0] * ext[1] * ext[2])

        return {
            "long_xy": long_xy,
            "short_xy": short_xy,
            "ratio_xy": ratio_xy,
            "z_extent": z_extent,
            "volume": volume,
        }

    def _class_score(self, feats: dict, target_cls: str) -> float:
        # Heuristic shape templates in flattened-ground frame.
        ratio = feats["ratio_xy"]
        long_xy = feats["long_xy"]
        short_xy = feats["short_xy"]
        z_ext = feats["z_extent"]
        vol = feats["volume"]

        # Banana: elongated in XY, relatively low profile.
        banana = (
            2.2 * ratio
            + 1.6 * long_xy
            - 1.2 * z_ext
            - 0.7 * short_xy
            - 0.1 * abs(vol - 0.0025)
        )
        # Mustard bottle: upright-ish, compact XY, higher Z.
        mustard = (
            2.6 * z_ext
            - 0.9 * ratio
            - 1.0 * short_xy
            + 0.2 * long_xy
            - 0.1 * abs(vol - 0.0018)
        )
        # Sugar box: moderate XY, less elongated than banana.
        sugar = (
            1.8 * short_xy
            + 0.8 * long_xy
            + 0.6 * z_ext
            - 1.3 * abs(ratio - 1.4)
            - 0.1 * abs(vol - 0.0020)
        )

        table = {"banana": banana, "mustard": mustard, "sugar": sugar}
        if target_cls == "any":
            return max(table.values())
        return table[target_cls]

    def init(self):
        # ==========================================
        # 初始化 Open3D 非阻塞可视化器
        # ==========================================
        if self.inited:
            return
        self.auto = True
        self.start_pick_idx = None
        if os.environ.get("DISABLE_O3D_VIS") is None and not os.environ.get("DISPLAY"):
            os.environ["DISABLE_O3D_VIS"] = "1"
        if os.environ.get("DISABLE_O3D_VIS", "0") == "1":
            self.inited = True
            return
        o3d = _try_import_o3d()
        if o3d is None:
            self.inited = True
            return
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Isaac Lab LiDAR Viewer", width=1024, height=768)

        def space_callback(vis):
            self.key = None
            return False

        def w_key_callback(vis):
            self.key = 'w'
            print('w')
            return False

        def a_key_callback(vis):
            self.key = 'a'
            return False

        def s_key_callback(vis):
            self.key = 's'
            return False

        def d_key_callback(vis):
            self.key = 'd'
            return False

        def i_key_callback(vis):
            self.auto = True
            self.status = Status.SEARCH
            return False

        def k_key_callback(vis):
            self.auto = False
            self.v_list = [0 for _ in range(8)]
            return False

        def n_key_callback(vis):
            self.get_down = True
            return False

        def m_key_callback(vis):
            self.get_down = False
            return False

        def key4_callback(vis):
            self.v_list[0] += 0.2
            return False

        def key6_callback(vis):
            self.v_list[0] -= 0.2
            return False

        def key8_callback(vis):
            self.v_list[1] += 0.2
            return False

        def key2_callback(vis):
            self.v_list[1] -= 0.2
            return False

        def o_key_callback(vis):
            self.v_list[2] += 0.2
            return False

        def p_key_callback(vis):
            self.v_list[2] -= 0.2
            return False

        self.vis.register_key_callback(ord(' '), space_callback)
        self.vis.register_key_callback(ord('W'), w_key_callback)
        self.vis.register_key_callback(ord('A'), a_key_callback)
        self.vis.register_key_callback(ord('S'), s_key_callback)
        self.vis.register_key_callback(ord('D'), d_key_callback)
        self.vis.register_key_callback(ord('I'), i_key_callback)
        self.vis.register_key_callback(ord('K'), k_key_callback)
        self.vis.register_key_callback(ord('N'), n_key_callback)
        self.vis.register_key_callback(ord('M'), m_key_callback)
        self.vis.register_key_callback(ord('G'), key4_callback)
        self.vis.register_key_callback(ord('J'), key6_callback)
        self.vis.register_key_callback(ord('Y'), key8_callback)
        self.vis.register_key_callback(ord('H'), key2_callback)

        self.vis.register_key_callback(ord('O'), o_key_callback)
        self.vis.register_key_callback(ord('P'), p_key_callback)

        # 创建全局 PointCloud 和坐标系几何体
        self.pcd = o3d.geometry.PointCloud()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        # 将几何体添加到渲染器 (此时点云是空的)
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.coord_frame)
        self.cur_idx = 0
        self.inited = True

    def __del__(self):
        """安全销毁窗口，防止退出时崩溃"""
        try:
            self.vis.destroy_window()
        except Exception:
            pass

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        """Platform hook: return {} to use official Task B B2Piper default action config."""
        return {}

    def reset(self, **kwargs):
        del kwargs
        self.cur_idx = 0
        self._platform_obs_logged = False
        self.vis_x = 0.0
        self.vis_y = 0.0
        self.vis_yaw = 0.0
        self.first_render = True
        self.debug_printed = False
        self.status = Status.SEARCH
        self.get_down = False
        self.start_get_down_idx = None
        self.start_pick_idx = None
        self.start_stand_idx = None
        self.v_list = [0.0 for _ in range(8)]
        self.cmd_max_vx = 1.0
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = 1.0
        self._last_bin_bearing = None
        self._last_bin_dist = None
        self._last_target_class = None
        self._last_target_score = None
        self._ik_phase = None
        self._ik_phase_start = None
        self._ik_target_pos_b = None
        self._ik_lift_pos_b = None
        self._ik_acquire_pos_b = None
        self._ik_no_target_count = 0
        self._last_target_cam = None
        self._ik_pick_active = False
        self._pick_arm_hold_cmd = None
        self._grasp_target_cam = None
        self._grasp_axis_cam = None
        self._grasp_approach_inited = False
        self._grasp_target_cam_ema = None
        self._grasp_candidate_cams = []
        self._grasp_tried_cams = []
        self._grasp_reached_hold_count = 0
        self._grasp_prev_l_err = None
        self._grasp_prev_f_err = None
        self._grasp_prev_v_err = None
        self._grasp_close_reason = None
        self._grasp_wrist_yaw_target = None
        self._video_hud_lines = []
        self._lock_prepare_start_idx = None
        self._lock_prezero_settle_count = 0
        self._squat_start_leg_action = None
        self._search_track_target = None
        self._search_last_seen_dist = None
        self._search_last_seen_step = None
        self._search_memory_age = None
        self._search_last_cloud_zmax = None
        self._search_last_target_zmax = None
        self._search_last_cloud_src = None
        self._search_no_target_steps = 0
        self._search_wp_idx = 0
        self._grasp_sm = None
        self._grasp_obj_idx = None
        self._grasp_ik_active = False
        self._graspnet_plan = None
        self._graspnet_sm = None
        self._graspnet_scan_phase = None
        self._graspnet_scan_view_idx = 0
        self._graspnet_scan_step = 0
        self._graspnet_scan_plans = []
        self._graspnet_selected_view_action = None
        self._graspnet_handoff_action = None
        self._graspnet_handoff_step = 0
        self._last_platform_score = 0.0
        self._reset_grasp_record_buffers()

    def _build_grasp_plan_config(self) -> "GraspPlanConfig | None":
        if GraspPlanConfig is None:
            return None
        return GraspPlanConfig(
            target_object=self._graspnet_target_object(),
            candidate_pool=self._GRASPNET_CANDIDATE_POOL,
            mustard_vertical_path=self._GRASPNET_VERTICAL_PATH,
            mustard_topdown_straight=self._GRASPNET_MUSTARD_TOPDOWN_STRAIGHT,
            disable_mustard_topdown_straight=self._GRASPNET_DISABLE_MUSTARD_TOPDOWN_STRAIGHT,
            mustard_safe_pre_z=self._GRASPNET_MUSTARD_SAFE_PRE_Z,
            use_yellow_mask=self._GRASPNET_USE_YELLOW_MASK,
            box_exec_mode="geometric" if self._GRASPNET_BOX_EXEC_MODE != "graspnet" else "graspnet",
            box_max_width=self._GRASPNET_BOX_MAX_WIDTH,
            box_bbox_expand_left=self._GRASPNET_BOX_BBOX_EXPAND_LEFT,
            box_bbox_expand_right=self._GRASPNET_BOX_BBOX_EXPAND_RIGHT,
            box_bbox_expand_up=self._GRASPNET_BOX_BBOX_EXPAND_UP,
            box_bbox_expand_down=self._GRASPNET_BOX_BBOX_EXPAND_DOWN,
            box_depth_slop=self._GRASPNET_BOX_DEPTH_SLOP,
            box_ee_z=self._GRASPNET_BOX_EE_Z,
            banana_pre_offset=self._GRASPNET_BANANA_PRE_OFFSET,
            banana_lift_height=self._GRASPNET_BANANA_LIFT_HEIGHT,
            banana_ee_z=self._GRASPNET_BANANA_EE_Z,
            banana_safe_pre_z=self._GRASPNET_BANANA_SAFE_PRE_Z,
            banana_x_offset=self._GRASPNET_BANANA_X_OFFSET,
            banana_y_offset=self._GRASPNET_BANANA_Y_OFFSET,
            banana_min_downness=self._GRASPNET_BANANA_MIN_DOWNNESS,
            banana_target_z_frac=self._GRASPNET_BANANA_TARGET_Z_FRAC,
        )

    def _graspnet_target_object(self) -> str:
        explicit = self._GRASPNET_TARGET_OBJECT
        if explicit in ("mustard", "box", "banana"):
            return explicit
        cls = self._target_object_class
        if cls == "banana":
            return "banana"
        if cls in ("sugar", "box"):
            return "box"
        if cls == "mustard":
            return "mustard"
        return "mustard"

    def _calibrate_graspnet_camera_extrinsic(self) -> None:
        self._t_gripper_camera = None
        if not self._GRASPNET_FIXED_EXTRINSIC:
            return
        if (
            self._env is None
            or calibrate_gripper_camera_extrinsic is None
            or make_transform is None
            or quat_wxyz_to_matrix is None
        ):
            return
        try:
            cam = self._env.unwrapped.scene["ee_camera"]
            cam_pos = cam.data.pos_w[0].detach().cpu().numpy().astype(np.float64)
            cam_quat = cam.data.quat_w_ros[0].detach().cpu().numpy().astype(np.float64)
            t_world_camera = make_transform(quat_wxyz_to_matrix(cam_quat), cam_pos)
            ee_pos, ee_quat = self._get_ee_body_world_pose()
            t_world_gripper = make_transform(quat_wxyz_to_matrix(ee_quat), ee_pos)
            self._t_gripper_camera = calibrate_gripper_camera_extrinsic(t_world_gripper, t_world_camera)
            print("[TaskB][GraspNet] fixed T_gripper_camera calibrated.", flush=True)
        except Exception as exc:
            self._t_gripper_camera = None
            print(f"[TaskB][GraspNet] extrinsic calibration failed: {exc}", flush=True)
        self._set_squat_leg_pd(False)
        self.cur_idx = 0

    def get_video_overlay_lines(self) -> list[str]:
        return list(self._video_hud_lines)

    def _set_video_hud(self, *lines: str) -> None:
        self._video_hud_lines = [str(line) for line in lines if line]

    def _log_grasp_debug(self, tag: str, **fields) -> None:
        if not self._GRASP_DEBUG:
            return
        parts = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[TaskB][GRASP][{tag}] {parts}", flush=True)

    def _graspnet_debug_view_name(self) -> str:
        phase = str(self._graspnet_scan_phase or "")
        if phase == "scan":
            specs = self._graspnet_view_specs()
            idx = int(self._graspnet_scan_view_idx)
            if 0 <= idx < len(specs):
                return str(specs[idx][0])
            return "scan_done"
        if phase == "handoff":
            return "detect_handoff"
        if phase == "return_to_view":
            return str(self._graspnet_plan.get("view_name", "selected_view")) if self._graspnet_plan else "selected_view"
        if self._graspnet_plan is not None:
            return str(self._graspnet_plan.get("view_name", "single_view"))
        return "single_view"

    def _graspnet_debug_target_yaw(self) -> float | None:
        phase = str(self._graspnet_scan_phase or "")
        if phase == "scan":
            specs = self._graspnet_view_specs()
            idx = int(self._graspnet_scan_view_idx)
            if 0 <= idx < len(specs) and len(specs[idx][1]) > 3:
                return float(specs[idx][1][3])
            return None
        if phase == "handoff":
            if self._graspnet_handoff_action is not None and len(self._graspnet_handoff_action) > 3:
                return float(self._graspnet_handoff_action[3])
            return None
        if phase == "return_to_view":
            if self._graspnet_selected_view_action is not None and len(self._graspnet_selected_view_action) > 3:
                return float(self._graspnet_selected_view_action[3])
            return None
        return None

    def _log_graspnet_step_debug(self, state: str) -> None:
        if not self._GRASP_DEBUG:
            return
        if self._GRASP_DEBUG_EVERY > 1 and (self.cur_idx % self._GRASP_DEBUG_EVERY) != 0:
            return
        arm = [float(self.v_list[i]) if i < len(self.v_list) else 0.0 for i in range(3)]
        wrist_cmd = float(self.v_list[3]) if len(self.v_list) > 3 else None
        self._log_grasp_debug(
            "graspnet-step",
            state=state,
            scan_phase=self._graspnet_scan_phase,
            view_name=self._graspnet_debug_view_name(),
            arm=f"[{arm[0]:+.2f},{arm[1]:+.2f},{arm[2]:+.2f}]",
            wrist_cmd=wrist_cmd,
            target_yaw=self._graspnet_debug_target_yaw(),
        )

    def _live_grasp_target_cam(self, obs) -> np.ndarray | None:
        """Prefer live ee_depth cluster; fall back to locked target."""
        ee_depth = self._obs_depth(obs, "ee_depth")
        if ee_depth is not None:
            target_cam, _ = self._find_target_cam_by_depth(ee_depth)
            if target_cam is not None:
                return target_cam.astype(np.float32)
        if self._grasp_target_cam is not None:
            return self._grasp_target_cam.astype(np.float32)
        return None

    def _clamp_arm_cmd_to_hold(self) -> None:
        if self._pick_arm_hold_cmd is None:
            return
        extra_common = self._GRASP_JOINT_EXTRA_MAX
        extra_j0 = max(self._GRASP_J0_EXTRA_MAX, extra_common)
        # j0 controls lateral reach; allow a wider envelope than j1/j2.
        lo0 = self._pick_arm_hold_cmd[0] - extra_j0
        hi0 = self._pick_arm_hold_cmd[0] + extra_j0
        self.v_list[0] = float(np.clip(self.v_list[0], lo0, hi0))
        for i in (1, 2):
            lo = self._pick_arm_hold_cmd[i] - extra_common
            hi = self._pick_arm_hold_cmd[i] + extra_common
            self.v_list[i] = float(np.clip(self.v_list[i], lo, hi))

    def _clamp_wrist_yaw_to_hold(self) -> None:
        if self._pick_arm_hold_cmd is None:
            return
        extra = self._GRASP_WRIST_YAW_EXTRA_MAX
        lo = self._pick_arm_hold_cmd[3] - extra
        hi = self._pick_arm_hold_cmd[3] + extra
        self.v_list[3] = float(np.clip(self.v_list[3], lo, hi))

    def _snapshot_pick_hold_arm_cmd(self) -> None:
        self._pick_arm_hold_cmd = [float(v) for v in self.v_list]

    def _apply_pick_hold_arm_cmd(
        self,
        *,
        yaw_delta: float = 0.0,
        j1_delta: float = 0.0,
        j2_delta: float = 0.0,
        gripper_close: bool = False,
    ) -> None:
        if self._pick_arm_hold_cmd is None:
            return
        self.v_list = [float(v) for v in self._pick_arm_hold_cmd]
        self.v_list[0] += float(yaw_delta)
        self.v_list[1] += float(j1_delta)
        self.v_list[2] += float(j2_delta)
        if gripper_close:
            self.v_list[6] = -1.0
            self.v_list[7] = -1.0

    def _resolve_joint_ids(self, candidates: tuple[list[str], ...]) -> list[int]:
        for names in candidates:
            try:
                ids, found_names = self.robot.find_joints(names)
                if len(ids) == len(names):
                    if candidates is self.ARM_JOINT_NAME_CANDIDATES:
                        self.arm_joint_names = list(found_names)
                    return list(ids)
            except ValueError:
                continue
        raise ValueError("Cannot resolve required joints.")

    def _resolve_ee_body_name(self) -> str:
        for name in self.EE_BODY_NAME_CANDIDATES:
            try:
                body_ids, _ = self.robot.find_bodies(name)
                if len(body_ids) == 1: return name
            except ValueError:
                continue
        raise ValueError("Cannot resolve EE body.")

    def _current_ee_pose_base(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._env is None or self.ee_name is None or subtract_frame_transforms is None:
            return None, None
        try:
            ee_body_id, _ = self.robot.find_bodies(self.ee_name)
            ee_body_id = int(ee_body_id[0])
            if hasattr(self.robot.data, "root_pose_w") and hasattr(self.robot.data, "body_pose_w"):
                root_pose_w = self.robot.data.root_pose_w
                ee_pose_w = self.robot.data.body_pose_w[:, ee_body_id]
                root_pos_w = root_pose_w[:, :3]
                root_quat_w = root_pose_w[:, 3:]
                ee_pos_w = ee_pose_w[:, :3]
                ee_quat_w = ee_pose_w[:, 3:]
            else:
                root_pos_w = self.robot.data.root_pos_w
                root_quat_w = self.robot.data.root_quat_w
                ee_pos_w = self.robot.data.body_pos_w[:, ee_body_id]
                ee_quat_w = self.robot.data.body_quat_w[:, ee_body_id]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )
            return ee_pos_b, ee_quat_b
        except Exception:
            return None, None

    def _find_target_cam_by_depth(
        self,
        depth: np.ndarray,
        *,
        K: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, float | None]:
        candidates = self._find_target_cams_by_depth(
            depth,
            K=K,
            max_candidates=1,
        )
        if len(candidates) == 0:
            return None, None
        return candidates[0]

    def _find_target_cams_by_depth(
        self,
        depth: np.ndarray,
        *,
        K: np.ndarray | None = None,
        max_candidates: int | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        if depth is None:
            return []
        points = self.depth_to_point_cloud(depth.squeeze(), K=K)
        if points is None or len(points) < 3:
            return []
        ground_mask, plane_model = self.detect_ground_ransac(points, distance_threshold=0.03)
        if plane_model is None or len(ground_mask) != len(points):
            ground_mask = np.zeros(len(points), dtype=bool)
        others = points[~ground_mask]
        if len(others) < 8:
            return []
        others = np.unique(np.round(others * 35.0), axis=0) / 35.0
        labels, n_clusters = self.cluster_euclidean_open3d(others, eps=0.08, min_points=8)
        candidates: list[tuple[np.ndarray, float]] = []
        for label in range(n_clusters):
            cur = others[labels == label]
            if cur.shape[0] < 8:
                continue
            centroid = np.mean(cur, axis=0)
            # Camera depth point uses z<0 in front of the camera.
            forward = float(-centroid[2])
            if forward < 0.1 or forward > 1.2:
                continue
            dist = float(np.linalg.norm(centroid))
            candidates.append((centroid.astype(np.float32), dist))
        if len(candidates) == 0:
            return []
        candidates.sort(key=lambda item: float(item[1]))
        if max_candidates is not None and max_candidates > 0:
            return candidates[: int(max_candidates)]
        return candidates

    def _refresh_grasp_candidates_from_obs(self, obs) -> int:
        ee_depth = self._obs_depth(obs, "ee_depth")
        if ee_depth is None:
            self._grasp_candidate_cams = []
            return 0
        candidates = self._find_target_cams_by_depth(
            ee_depth,
            K=self._get_ee_camera_intrinsic_safe(),
            max_candidates=max(self._GRASP_MULTI_TARGET_MAX_CANDIDATES, 1),
        )
        self._grasp_candidate_cams = [np.asarray(c[0], dtype=np.float32).copy() for c in candidates]
        return len(self._grasp_candidate_cams)

    def _pop_next_grasp_candidate(self, *, obs=None) -> np.ndarray | None:
        if len(self._grasp_candidate_cams) == 0 and obs is not None:
            self._refresh_grasp_candidates_from_obs(obs)
        while len(self._grasp_candidate_cams) > 0:
            cand = np.asarray(self._grasp_candidate_cams.pop(0), dtype=np.float32).reshape(3)
            is_dup = False
            for prev in self._grasp_tried_cams:
                if float(np.linalg.norm(cand - prev)) < self._GRASP_MULTI_TARGET_DEDUP_DIST:
                    is_dup = True
                    break
            if is_dup:
                continue
            self._grasp_tried_cams.append(cand.copy())
            return cand
        return None

    def _switch_to_next_grasp_target(self, obs, reason: str) -> bool:
        if not (self._DEPTH_TOUCH_ONLY_SCORE and self._GRASP_MULTI_TARGET_ENABLE):
            return False
        next_target = self._pop_next_grasp_candidate(obs=obs)
        if next_target is None:
            return False
        self._grasp_target_cam = next_target.copy()
        self._last_target_cam = next_target.copy()
        self._grasp_axis_cam = None
        self._ik_phase = "acquire"
        self._ik_phase_start = self.cur_idx
        self._ik_no_target_count = 0
        self._grasp_target_cam_ema = None
        self._grasp_reached_hold_count = 0
        self._grasp_prev_l_err = None
        self._grasp_prev_f_err = None
        self._grasp_prev_v_err = None
        self._apply_pick_hold_arm_cmd()
        print(
            f"[TaskB][DepthPCA] switch target ({reason}) -> acquire center={self._grasp_target_cam}",
            flush=True,
        )
        return True

    def _get_ee_camera_intrinsic_safe(self) -> np.ndarray:
        if self._env is None:
            return self.K
        try:
            cam = self._env.unwrapped.scene["ee_camera"]
            intrinsic = cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
            if intrinsic.shape == (3, 3):
                return intrinsic
        except Exception:
            pass
        return self.K

    @staticmethod
    def _center_cam_to_graspnet_frame(center_cam: np.ndarray) -> np.ndarray:
        """Convert local depth-to-point-cloud camera frame to GraspNet frame.

        Our local helpers use:
          x = right, y = up, z = -forward
        GraspNet point cloud uses:
          x = right, y = down, z = +forward
        """
        c = np.asarray(center_cam, dtype=np.float64).reshape(3).copy()
        c[1] = -c[1]
        c[2] = -c[2]
        return c

    def _extract_local_object_cloud(
        self, depth: np.ndarray, center_cam: np.ndarray | None
    ) -> np.ndarray | None:
        if depth is None:
            return None
        points = self.depth_to_point_cloud(depth.squeeze())
        if points is None or len(points) < 30:
            return None
        forward = -points[:, 2]
        keep = np.isfinite(forward) & (forward > 0.08) & (forward < 1.1)
        points = points[keep]
        if len(points) < 20:
            return None
        if center_cam is not None:
            dist = np.linalg.norm(points - center_cam.reshape(1, 3), axis=1)
            points = points[dist < 0.20]
            if len(points) < 20:
                return None
        points = np.unique(np.round(points * 80.0), axis=0) / 80.0
        if len(points) < 20:
            return None
        labels, n_clusters = self.cluster_euclidean_open3d(points, eps=0.055, min_points=12)
        if n_clusters <= 0:
            return None
        best_cluster = None
        best_metric = None
        for label in range(n_clusters):
            cur = points[labels == label]
            if len(cur) < 15:
                continue
            centroid = np.mean(cur, axis=0)
            metric = float(np.linalg.norm(centroid - center_cam)) if center_cam is not None else float(np.linalg.norm(centroid))
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_cluster = cur
        return best_cluster

    @staticmethod
    def _estimate_grasp_pose_from_cloud(points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if points is None or len(points) < 12:
            return None, None
        centroid = np.mean(points, axis=0).astype(np.float32)
        centered = points - centroid.reshape(1, 3)
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return centroid, None
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return centroid, None
        axis = axis / n
        if abs(float(axis[1])) > 0.75:
            axis[1] = 0.0
            m = float(np.linalg.norm(axis))
            if m > 1e-6:
                axis = axis / m
        return centroid, axis

    def _build_grasp_pose_from_ee_depth(
        self, ee_depth: np.ndarray, center_cam: np.ndarray | None
    ) -> dict | None:
        local_points = self._extract_local_object_cloud(ee_depth, center_cam)
        centroid, axis = self._estimate_grasp_pose_from_cloud(local_points)
        if centroid is None:
            return None
        # Grab upper half instead of geometric center: center + ratio * object_height * z_axis.
        # In this camera frame, -Y is "up".
        z_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        object_height = 0.0
        grasp_center = centroid.astype(np.float32)
        if local_points is not None and len(local_points) >= 10:
            proj = np.asarray(local_points @ z_axis.reshape(3, 1), dtype=np.float32).reshape(-1)
            if proj.size > 0:
                lo = float(np.percentile(proj, 5.0))
                hi = float(np.percentile(proj, 95.0))
                object_height = max(0.0, hi - lo)
                up_shift = float(
                    np.clip(
                        self._GRASP_TOP_OFFSET_RATIO * object_height,
                        0.0,
                        self._GRASP_TOP_OFFSET_MAX,
                    )
                )
                grasp_center = (centroid + up_shift * z_axis).astype(np.float32)
        return {
            "center_cam": grasp_center,
            "centroid_cam": centroid.astype(np.float32),
            "axis_cam": (axis if axis is not None else np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            "approach_cam": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "z_axis_cam": z_axis,
            "object_height": float(object_height),
            "num_points": int(local_points.shape[0]) if local_points is not None else 0,
        }

    def _estimate_grasp_wrist_yaw_target(self, obs) -> float | None:
        """Estimate desired wrist yaw (v_list[3]) from object major axis in ee camera image plane."""
        ee_depth = self._obs_depth(obs, "ee_depth")
        if ee_depth is None:
            return None
        center_cam = self._grasp_target_cam_ema if self._grasp_target_cam_ema is not None else self._grasp_target_cam
        local_points = self._extract_local_object_cloud(ee_depth, center_cam)
        _, axis = self._estimate_grasp_pose_from_cloud(local_points)
        if axis is None:
            return None
        axis_xy = np.asarray([axis[0], axis[1]], dtype=np.float32)
        norm_xy = float(np.linalg.norm(axis_xy))
        if norm_xy < 1e-5:
            return None
        axis_xy /= norm_xy
        # Image-plane major-axis orientation ("横竖"): atan2(y, x).
        orient_img = float(np.arctan2(axis_xy[1], axis_xy[0]))
        yaw_target = self._GRASP_ALIGN_YAW_SIGN * orient_img + self._GRASP_ALIGN_YAW_OFFSET
        return float(np.clip(yaw_target, -self._GRASP_ALIGN_YAW_TARGET_CLIP, self._GRASP_ALIGN_YAW_TARGET_CLIP))

    def _derive_ik_target_base_from_obs(self, obs) -> torch.Tensor | None:
        ee_pos_b, _ = self._current_ee_pose_base()
        if ee_pos_b is None:
            return None
        cam_point, _ = (None, None)
        ee_depth = self._obs_depth(obs, "ee_depth")
        if ee_depth is not None:
            cam_point, _ = self._find_target_cam_by_depth(ee_depth)
        if cam_point is None:
            return None
        # Heuristic ee-camera -> base delta mapping:
        #   forward ~= -z, lateral ~= -x, vertical ~= -y.
        d_forward = float(np.clip(-cam_point[2] - 0.20, -0.15, 0.30))
        d_lateral = float(np.clip(-cam_point[0], -0.16, 0.16))
        d_vertical = float(np.clip(-cam_point[1], -0.10, 0.08))
        target = ee_pos_b.clone()
        target[:, 0] = target[:, 0] + d_forward
        target[:, 1] = target[:, 1] + d_lateral
        target[:, 2] = torch.clamp(target[:, 2] + d_vertical, 0.03, 0.45)
        return target

    def _compute_ik_pick_arm_action(self, target_pos_b: torch.Tensor, close_gripper: bool) -> torch.Tensor | None:
        if self.cartesian_ctrl is None or target_pos_b is None:
            return None
        try:
            self.cartesian_ctrl.reset()
            arm_jpos_des = self.cartesian_ctrl.compute_base(target_pos_b, None)
            full_target = self.robot.data.joint_pos.clone()
            full_target[:, self.arm_ids] = arm_jpos_des
            if close_gripper:
                full_target[:, self.gripper_ids] = self.gripper_close_pos.repeat(full_target.shape[0], 1)
            else:
                full_target[:, self.gripper_ids] = self.gripper_open_pos.repeat(full_target.shape[0], 1)
            arm_action = (full_target - self.default_joint_pos) / self.ACTION_SCALE
            return arm_action[:, self.arm_joint_indices]
        except Exception:
            return None

    def _grasp_use_task_e_ik(self) -> bool:
        return (
            self._GRASP_USE_TASKE_IK
            and not self._grasp_use_graspnet()
            and self._env is not None
            and self.cartesian_ctrl is not None
            and TaskEGraspOnlySM is not None
            and compute_grasp_quat is not None
        )

    def _grasp_use_graspnet(self) -> bool:
        return (
            self._ENABLE_GRASPNET
            and self._graspnet_ready
            and self._graspnet_runner is not None
            and self._env is not None
            and self.cartesian_ctrl is not None
            and GraspNetWaypointSM is not None
            and make_transform is not None
        )

    def _graspnet_use_multiview(self) -> bool:
        if not self._GRASPNET_MULTIVIEW:
            return False
        target = self._graspnet_target_object()
        if target == "mustard":
            return True
        if target == "banana":
            return bool(self._GRASPNET_BANANA_MULTIVIEW)
        return False

    def _sync_graspnet_plan_config(self) -> None:
        if self._graspnet_runner is None:
            return
        cfg = self._build_grasp_plan_config()
        if cfg is not None:
            self._graspnet_runner.plan_cfg = cfg

    def _get_ee_camera_world_transform(self) -> tuple[np.ndarray, np.ndarray]:
        cam = self._env.unwrapped.scene["ee_camera"]
        intrinsic = cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
        if (
            self._GRASPNET_FIXED_EXTRINSIC
            and self._t_gripper_camera is not None
            and world_camera_from_gripper_extrinsic is not None
            and make_transform is not None
            and quat_wxyz_to_matrix is not None
        ):
            ee_pos, ee_quat = self._get_ee_body_world_pose()
            t_world_gripper = make_transform(quat_wxyz_to_matrix(ee_quat), ee_pos)
            t_world_camera = world_camera_from_gripper_extrinsic(t_world_gripper, self._t_gripper_camera)
        else:
            pos = cam.data.pos_w[0].detach().cpu().numpy().astype(np.float64)
            quat = cam.data.quat_w_ros[0].detach().cpu().numpy().astype(np.float64)
            t_world_camera = make_transform(quat_wxyz_to_matrix(quat), pos)
        return t_world_camera, intrinsic

    def _get_ee_body_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        body_ids, _ = self.robot.find_bodies(self.ee_name)
        body_id = int(body_ids[0])
        pos = self.robot.data.body_pos_w[0, body_id, :3].detach().cpu().numpy().astype(np.float64)
        quat = self.robot.data.body_quat_w[0, body_id, :4].detach().cpu().numpy().astype(np.float64)
        return pos, quat

    def _get_ee_open_axis_world(self) -> np.ndarray:
        pos, quat = self._get_ee_body_world_pose()
        rot = quat_wxyz_to_matrix(quat)
        axis = rot[:3, 1]
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return (axis / norm).astype(np.float64)

    def _graspnet_gripper_tensor(self, cmd: str):
        dev = self.device
        dtype = self.robot.data.joint_pos.dtype
        if cmd == "open":
            if self._graspnet_target_object() == "mustard":
                open_abs = float(self._GRASPNET_MUSTARD_OPEN_POS)
                vals = [open_abs, -open_abs]
            else:
                vals = list(GRIPPER_OPEN_POS)
        else:
            vals = list(GRIPPER_CLOSE_POS)
        return torch.tensor([vals], device=dev, dtype=dtype)

    def _reset_graspnet_scan(self) -> None:
        self._graspnet_scan_phase = None
        self._graspnet_scan_view_idx = 0
        self._graspnet_scan_step = 0
        self._graspnet_scan_plans = []
        self._graspnet_selected_view_action = None
        self._graspnet_handoff_action = None
        self._graspnet_handoff_step = 0

    def _graspnet_view_specs(self) -> list[tuple[str, list[float]]]:
        target = self._graspnet_target_object()
        if target == "banana" and BANANA_VIEW_SPECS:
            return list(BANANA_VIEW_SPECS)
        if target == "mustard" and MUSTARD_VIEW_SPECS:
            return list(MUSTARD_VIEW_SPECS)
        return [("single_view", [-1.0, 0.6, 0.6, 0.0, -0.6, 0.0, 0.0, 0.0])]

    def _apply_graspnet_view_action(self, action_values: list[float]) -> None:
        self.v_list = [float(v) for v in action_values[:8]]
        while len(self.v_list) < 8:
            self.v_list.append(0.0)

    def _plan_graspnet_from_obs(
        self,
        obs,
        *,
        view_name: str,
    ) -> dict:
        rgb = self._obs_rgb(obs, "ee_rgb")
        if rgb is None:
            rgb = self._obs_head_rgb(obs)
        if rgb is None:
            rgb = self._obs_rgb(obs, "ee_rgb")
        depth = self._obs_depth(obs, "ee_depth")
        if rgb is None or depth is None:
            raise RuntimeError("GraspNet requires ee_rgb + ee_depth.")
        t_world_camera, intrinsic = self._get_ee_camera_world_transform()
        center_cam = None
        target_object = self._graspnet_target_object()
        # Keep detect->grasp consistency: for banana especially, prefer the
        # locked DETECT center to avoid plan-time target switching to another
        # yellow object/component.
        if (
            self._GRASPNET_PREFER_LOCKED_CENTER
            and self._grasp_target_cam is not None
            and target_object in ("banana", "box")
        ):
            center_cam = np.asarray(self._grasp_target_cam, dtype=np.float64).reshape(3)
        if center_cam is None and depth is not None:
            center_cam, _ = self._find_target_cam_by_depth(depth, K=intrinsic)
        if (
            center_cam is None
            and self._GRASPNET_USE_DETECT_CENTER_FALLBACK
            and self._grasp_target_cam is not None
        ):
            center_cam = np.asarray(self._grasp_target_cam, dtype=np.float64).reshape(3)
        center_cam_plan = (
            None if center_cam is None else self._center_cam_to_graspnet_frame(center_cam)
        )
        self._sync_graspnet_plan_config()
        plan = self._graspnet_runner.plan_from_rgbd(
            rgb,
            depth,
            intrinsic,
            t_world_camera,
            device=self.device,
            center_cam=center_cam_plan,
            view_open_axis_w=self._get_ee_open_axis_world(),
            view_name=view_name,
            target_object=target_object,
        )
        plan["selection_score"] = float(plan.get("selection_score", plan["best"]["final_score"]))
        if self._GRASP_DEBUG:
            try:
                cc = None if center_cam is None else [float(v) for v in np.asarray(center_cam).reshape(-1)[:3]]
                cc_plan = (
                    None
                    if center_cam_plan is None
                    else [float(v) for v in np.asarray(center_cam_plan).reshape(-1)[:3]]
                )
                print(
                    f"[TaskB][GraspNet] plan view={view_name} target={target_object} "
                    f"center_cam_local={cc} center_cam_graspnet={cc_plan}",
                    flush=True,
                )
            except Exception:
                pass
        return plan

    def _start_graspnet_multiview_scan(self) -> None:
        self._graspnet_scan_phase = "scan"
        self._graspnet_scan_view_idx = 0
        self._graspnet_scan_step = 0
        self._graspnet_scan_plans = []
        self._graspnet_plan = None
        self._graspnet_sm = None

    def _graspnet_multiview_step(self, obs) -> None:
        view_specs = self._graspnet_view_specs()
        if self._graspnet_scan_view_idx >= len(view_specs):
            self._finish_graspnet_multiview_scan(obs)
            return

        view_name, view_action = view_specs[self._graspnet_scan_view_idx]
        self._apply_graspnet_view_action(view_action)
        self._graspnet_scan_step += 1
        move_steps = self._GRASPNET_VIEW_MOVE_STEPS + self._GRASPNET_VIEW_SETTLE_STEPS
        self._set_video_hud(
            f"status=GRASP ik=graspnet scan={view_name}",
            f"step={self._graspnet_scan_step}/{move_steps}",
        )
        if self._graspnet_scan_step < move_steps:
            return

        try:
            plan = self._plan_graspnet_from_obs(obs, view_name=view_name)
            plan["view_action"] = list(view_action)
            self._graspnet_scan_plans.append(plan)
            print(
                f"[TaskB][GraspNet] {view_name} score={plan['selection_score']:.4f} "
                f"src={plan.get('component_source')} "
                f"bbox={plan['selected_component'].get('bbox')}",
                flush=True,
            )
        except Exception as exc:
            print(f"[TaskB][GraspNet] {view_name} failed: {exc}", flush=True)

        self._graspnet_scan_view_idx += 1
        self._graspnet_scan_step = 0
        if self._graspnet_scan_view_idx >= len(view_specs):
            try:
                self._finish_graspnet_multiview_scan(obs)
            except RuntimeError as exc:
                self._reset_graspnet_scan()
                raise exc

    def _finish_graspnet_multiview_scan(self, obs) -> None:
        if not self._graspnet_scan_plans:
            self._graspnet_scan_phase = None
            raise RuntimeError("All multiview GraspNet plans failed.")
        selected = max(self._graspnet_scan_plans, key=lambda p: float(p["selection_score"]))
        self._graspnet_plan = selected
        self._graspnet_selected_view_action = list(selected.get("view_action", self._graspnet_view_specs()[0][1]))
        self._graspnet_scan_phase = "return_to_view"
        self._graspnet_scan_step = 0
        print(
            f"[TaskB][GraspNet] selected view={selected.get('view_name')} "
            f"score={selected['selection_score']:.4f}",
            flush=True,
        )

    def _graspnet_return_to_view_step(self, obs) -> None:
        if self._graspnet_scan_phase == "handoff":
            if self._graspnet_handoff_action is not None:
                self._apply_graspnet_view_action(self._graspnet_handoff_action)
            self._graspnet_handoff_step += 1
            self._set_video_hud(
                "status=GRASP ik=graspnet handoff",
                f"step={self._graspnet_handoff_step}/{self._GRASPNET_DETECT_HANDOFF_STEPS}",
            )
            if self._graspnet_handoff_step < self._GRASPNET_DETECT_HANDOFF_STEPS:
                return
            self._graspnet_scan_phase = None
            if self._graspnet_use_multiview() and len(self._graspnet_view_specs()) > 1:
                self._start_graspnet_multiview_scan()
                self._graspnet_multiview_step(obs)
                return
            self._graspnet_plan = self._plan_graspnet_from_obs(obs, view_name="single_view")
            self._launch_graspnet_execution()
            return
        if self._graspnet_selected_view_action is not None:
            self._apply_graspnet_view_action(self._graspnet_selected_view_action)
        self._graspnet_scan_step += 1
        move_steps = self._GRASPNET_VIEW_MOVE_STEPS
        self._set_video_hud(
            "status=GRASP ik=graspnet return_to_view",
            f"step={self._graspnet_scan_step}/{move_steps}",
        )
        if self._graspnet_scan_step < move_steps:
            return
        self._launch_graspnet_execution()

    def _launch_graspnet_execution(self) -> None:
        if self._graspnet_plan is None:
            raise RuntimeError("GraspNet plan missing at execution launch.")
        ee_start, ee_start_quat = self._get_ee_body_world_pose()
        sm_cls = GraspNetStructuredSM if GraspNetStructuredSM is not None else GraspNetWaypointSM
        self._graspnet_sm = sm_cls(
            self._graspnet_plan,
            ee_start,
            ee_start_quat,
        )
        self._grasp_ik_active = True
        self._graspnet_scan_phase = None
        if self.cartesian_ctrl is not None:
            self.cartesian_ctrl.reset()
        best = self._graspnet_plan["best"]
        pre_w = self._graspnet_plan.get("pregrasp_world")
        grasp_w = self._graspnet_plan.get("ee_grasp_world")
        lift_w = self._graspnet_plan.get("lift_world")
        print(
            f"[TaskB][GraspNet] execute score={best['final_score']:.3f} "
            f"graspnet={best['graspnet_score']:.3f} mode={self._graspnet_plan['execution_mode']} "
            f"pre={pre_w} grasp={grasp_w} lift={lift_w}",
            flush=True,
        )

    def _begin_graspnet_grasp(self, obs) -> None:
        if self._graspnet_handoff_action is not None and self._GRASPNET_DETECT_HANDOFF_STEPS > 0:
            self._graspnet_scan_phase = "handoff"
            self._graspnet_handoff_step = 0
            self._graspnet_return_to_view_step(obs)
            return
        if self._graspnet_use_multiview() and len(self._graspnet_view_specs()) > 1:
            self._start_graspnet_multiview_scan()
            self._graspnet_multiview_step(obs)
            return
        self._graspnet_plan = self._plan_graspnet_from_obs(obs, view_name="single_view")
        self._launch_graspnet_execution()

    def _update_graspnet_grasp_ik(self, obs) -> None:
        if self._graspnet_sm is None:
            try:
                if self._graspnet_scan_phase == "scan":
                    self._log_graspnet_step_debug("SCAN")
                    self._graspnet_multiview_step(obs)
                    return
                if self._graspnet_scan_phase == "handoff":
                    self._log_graspnet_step_debug("HANDOFF")
                    self._graspnet_return_to_view_step(obs)
                    return
                if self._graspnet_scan_phase == "return_to_view":
                    self._log_graspnet_step_debug("RETURN_TO_VIEW")
                    self._graspnet_return_to_view_step(obs)
                    return
                if self._graspnet_plan is None:
                    self._log_graspnet_step_debug("PLAN_START")
                    self._begin_graspnet_grasp(obs)
                    return
            except Exception as exc:
                self._reset_graspnet_scan()
                self._set_video_hud("status=GRASP ik=graspnet", f"plan_failed={exc}")
                if self.cur_idx % 20 == 0:
                    print(f"[TaskB][GraspNet] plan failed: {exc}", flush=True)
                return
        if self._graspnet_sm is None:
            return
        pos_w, quat_w, gripper_cmd = self._graspnet_sm.tick()
        state = self._graspnet_sm.state
        dev = self.device
        dtype = self.robot.data.joint_pos.dtype
        pos_t = torch.tensor(pos_w, dtype=torch.float32, device=dev).reshape(1, 3)
        # Match qxx behavior: use phase quaternion from planner/state-machine,
        # keeping each segment's orientation command fixed.
        quat_t = torch.tensor(quat_w, dtype=torch.float32, device=dev).reshape(1, 4)
        arm_jpos_des = self.cartesian_ctrl.compute(pos_t, quat_t)
        gripper_target = self._graspnet_gripper_tensor(gripper_cmd)
        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = gripper_target
        arm_env = (full_target - self.default_joint_pos) / TASKE_ACTION_SCALE
        arm_cmd = arm_env[0, self.arm_joint_indices].detach().cpu().tolist()
        self.v_list = arm_cmd
        self._set_video_hud(
            f"status=GRASP ik=graspnet state={state} step={self._graspnet_sm._count}",
            f"gripper={gripper_cmd} score={self._graspnet_plan['best']['final_score']:.3f}",
            f"pos=[{float(pos_w[0]):.2f},{float(pos_w[1]):.2f},{float(pos_w[2]):.2f}]",
        )
        self._log_graspnet_step_debug(str(state))
        if self._graspnet_sm.done:
            print("[TaskB][GraspNet] LIFT done -> STAND", flush=True)
            self._finish_grasp_to_stand()

    def _get_object_asset(self, obj_idx: int):
        scene = self._env.unwrapped.scene
        key = f"object_{obj_idx}"
        if hasattr(scene, "rigid_objects") and key in scene.rigid_objects:
            return scene.rigid_objects[key]
        return scene[key]

    def _nearest_grasp_object(self) -> tuple[int, torch.Tensor] | None:
        if self._env is None:
            return None
        robot_pos = self.robot.data.root_pos_w[0]
        rx, ry = float(robot_pos[0].item()), float(robot_pos[1].item())
        best: tuple[int, torch.Tensor, float] | None = None
        for obj_idx in range(1, self._TASK_B_NUM_OBJECTS + 1):
            try:
                pos = self._get_object_asset(obj_idx).data.root_pos_w[0].clone()
            except (AttributeError, KeyError):
                continue
            dx = float(pos[0].item()) - rx
            dy = float(pos[1].item()) - ry
            dist = (dx * dx + dy * dy) ** 0.5
            if best is None or dist < best[2]:
                best = (obj_idx, pos, dist)
        if best is None:
            return None
        return best[0], best[1]

    def _check_platform_score_touch(self, current_score) -> bool:
        """Platform/server touch done: current_score increased during pick."""
        score = float(current_score)
        if not self._DEPTH_TOUCH_ONLY_SCORE:
            self._last_platform_score = score
            return False
        increased = score > self._last_platform_score + 1e-6
        prev = self._last_platform_score
        self._last_platform_score = score
        if not increased:
            return False
        if self.status not in (Status.LOCK, Status.DETECT, Status.GRASP):
            return False
        print(
            f"[TaskB][TouchScore] platform_score +{score - prev:.2f} "
            f"({prev:.2f}->{score:.2f}) status={self.status.name} -> STAND",
            flush=True,
        )
        self._finish_grasp_to_stand()
        return True

    def _reset_grasp_record_buffers(self) -> None:
        self._grasp_rec_qpos = []
        self._grasp_rec_qvel = []
        self._grasp_rec_action = []
        self._grasp_rec_ee_pos = []
        self._grasp_rec_ee_quat = []

    def _grasp_record_step(self, action_row: torch.Tensor) -> None:
        self._grasp_rec_qpos.append(self.robot.data.joint_pos[0].detach().cpu().numpy())
        self._grasp_rec_qvel.append(self.robot.data.joint_vel[0].detach().cpu().numpy())
        self._grasp_rec_action.append(action_row.detach().cpu().numpy())
        self._grasp_rec_ee_pos.append(self.cartesian_ctrl.ee_pos_w[0].detach().cpu().numpy())
        self._grasp_rec_ee_quat.append(self.cartesian_ctrl.ee_quat_w[0].detach().cpu().numpy())

    def _grasp_record_flush(self) -> None:
        if not self._grasp_rec_qpos:
            return
        try:
            import h5py
        except ImportError:
            print("[TaskB][GraspIK] h5py missing — skip demo save.", flush=True)
            return
        os.makedirs(self._GRASP_RECORD_DIR, exist_ok=True)
        traj_path = os.path.join(self._GRASP_RECORD_DIR, "trajectory.hdf5")
        mode = "a" if os.path.isfile(traj_path) else "w"
        with h5py.File(traj_path, mode) as f:
            idx = len(f.keys())
            grp = f.create_group(f"traj_{idx}")
            grp.create_dataset("obs", data=np.stack(self._grasp_rec_qpos), compression="gzip")
            grp.create_dataset("actions", data=np.stack(self._grasp_rec_action), compression="gzip")
            grp.create_dataset("qvel", data=np.stack(self._grasp_rec_qvel), compression="gzip")
            grp.create_dataset("ee_pos", data=np.stack(self._grasp_rec_ee_pos), compression="gzip")
            grp.create_dataset("ee_quat", data=np.stack(self._grasp_rec_ee_quat), compression="gzip")
        print(
            f"[TaskB][GraspIK] saved traj_{idx} ({len(self._grasp_rec_qpos)} steps) -> {traj_path}",
            flush=True,
        )
        self._reset_grasp_record_buffers()

    def _begin_task_e_grasp_sm(self, obj_idx: int) -> None:
        obj = self._get_object_asset(obj_idx)
        obj_quat = obj.data.root_state_w[0, 3:7]
        grasp_quat = compute_grasp_quat(obj_quat, self.device)
        carry_z = self._GRASP_IK_CARRY_Z if self._GRASP_IK_CARRY_Z > 0.0 else 0.0
        self._grasp_sm = TaskEGraspOnlySM(
            grasp_quat,
            self.device,
            carry_z=carry_z if carry_z > 0.0 else None,
            grasp_z_offset=self._GRASP_IK_Z_OFFSET,
        )
        self._grasp_obj_idx = obj_idx
        self._grasp_ik_active = True
        self._ik_pick_active = True
        if self.cartesian_ctrl is not None:
            self.cartesian_ctrl.reset()
        if self._GRASP_RECORD:
            self._reset_grasp_record_buffers()
        print(
            f"[TaskB][GraspIK] object_{obj_idx} -> PRE_GRASP "
            f"(steps={ {s: TASKE_GRASP_STEPS.get(s, '?') for s in GRASP_STATE_ORDER} })",
            flush=True,
        )

    def _finish_grasp_to_stand(self) -> None:
        self.status = Status.STAND
        self.get_down = False
        self.start_stand_idx = self.cur_idx
        self.v_list = [0.0 for _ in range(8)]
        self.cmd_max_vx = 0.0
        self.cmd_max_vy = 0.0
        self.cmd_max_wz = 0.0
        self._ik_phase = None
        self._ik_phase_start = None
        self._ik_target_pos_b = None
        self._ik_lift_pos_b = None
        self._ik_acquire_pos_b = None
        self._ik_no_target_count = 0
        self._last_target_cam = None
        self._ik_pick_active = False
        self._pick_arm_hold_cmd = None
        self._grasp_target_cam = None
        self._grasp_axis_cam = None
        self._grasp_approach_inited = False
        self._grasp_target_cam_ema = None
        self._grasp_candidate_cams = []
        self._grasp_tried_cams = []
        self._grasp_reached_hold_count = 0
        self._grasp_prev_l_err = None
        self._grasp_prev_f_err = None
        self._grasp_prev_v_err = None
        self._grasp_close_reason = None
        self._grasp_wrist_yaw_target = None
        self._grasp_sm = None
        self._grasp_obj_idx = None
        self._grasp_ik_active = False
        self._graspnet_plan = None
        self._graspnet_sm = None
        self._reset_graspnet_scan()

    def _task_e_grasp_arm_action(self) -> list[float] | None:
        if self._grasp_sm is None or self.cartesian_ctrl is None or self._grasp_obj_idx is None:
            return None
        obj_pos = self._get_object_asset(self._grasp_obj_idx).data.root_pos_w[0].clone()
        ee_pos_des, ee_quat_des, gripper_cmd = self._grasp_sm.tick(obj_pos)
        dev = self.device
        dtype = self.robot.data.joint_pos.dtype
        ee_pos = ee_pos_des.unsqueeze(0).to(device=dev, dtype=dtype)
        ee_quat = ee_quat_des.unsqueeze(0).to(device=dev, dtype=dtype)
        arm_jpos_des = self.cartesian_ctrl.compute(ee_pos, ee_quat)
        gripper_target = gripper_tensor(gripper_cmd, dev, dtype)
        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = gripper_target
        arm_env = (full_target - self.default_joint_pos) / TASKE_ACTION_SCALE
        if self._GRASP_RECORD:
            self._grasp_record_step(arm_env[0])
        state = self._grasp_sm.state
        step_total = TASKE_GRASP_STEPS.get(state, "?")
        self._set_video_hud(
            f"status=GRASP ik=taske state={state} step={self._grasp_sm._count}/{step_total}",
            f"obj={self._grasp_obj_idx} gripper={gripper_cmd}",
            f"ee=[{float(ee_pos_des[0]):.2f},{float(ee_pos_des[1]):.2f},{float(ee_pos_des[2]):.2f}]",
        )
        if self._grasp_sm.done:
            print("[TaskB][GraspIK] LIFT done -> STAND", flush=True)
            if self._GRASP_RECORD:
                self._grasp_record_flush()
            self._finish_grasp_to_stand()
        return arm_env[0, self.arm_joint_indices].detach().cpu().tolist()

    def _update_task_e_grasp_ik(self, obs) -> None:
        del obs
        if self._grasp_sm is None:
            nearest = self._nearest_grasp_object()
            if nearest is None:
                self._set_video_hud("status=GRASP ik=taske", "no sim object found")
                return
            self._begin_task_e_grasp_sm(nearest[0])
        arm = self._task_e_grasp_arm_action()
        if arm is not None:
            self.v_list = [float(x) for x in arm]

    def _try_begin_depth_pick(self, obs, *, during_lower: bool = False) -> bool:
        """Start pick once an ee-depth object is visible."""
        if self._ik_pick_active:
            return True
        ee_depth = self._obs_depth(obs, "ee_depth")
        if ee_depth is None:
            return False
        self._refresh_grasp_candidates_from_obs(obs)
        target_cam = self._pop_next_grasp_candidate()
        if target_cam is None:
            return False
        if (
            not self._DEPTH_TOUCH_ONLY_SCORE
            and self._grasp_use_graspnet()
        ):
            self._snapshot_pick_hold_arm_cmd()
            self._grasp_target_cam = target_cam.astype(np.float32)
            if self._pick_arm_hold_cmd is not None:
                self._graspnet_handoff_action = [float(v) for v in self._pick_arm_hold_cmd[:8]]
            else:
                self._graspnet_handoff_action = [float(v) for v in self.v_list[:8]]
            while len(self._graspnet_handoff_action) < 8:
                self._graspnet_handoff_action.append(0.0)
            self._graspnet_handoff_step = 0
            self.status = Status.GRASP
            self._ik_pick_active = True
            where = "lower" if during_lower else "hold"
            print(f"[TaskB][GraspNet] target during {where} -> GRASP", flush=True)
            return True
        if (
            not self._DEPTH_TOUCH_ONLY_SCORE
            and self._grasp_use_task_e_ik()
        ):
            self.status = Status.GRASP
            self._ik_pick_active = True
            where = "lower" if during_lower else "hold"
            print(f"[TaskB][GraspIK] target during {where} -> GRASP (Task E IK)", flush=True)
            return True
        self._snapshot_pick_hold_arm_cmd()
        self._last_target_cam = target_cam.astype(np.float32)
        self._grasp_target_cam = self._last_target_cam.copy()
        self._grasp_axis_cam = None
        self._ik_pick_active = True
        self._ik_no_target_count = 0
        self.status = Status.GRASP
        self._ik_phase = "acquire"
        self._ik_phase_start = self.cur_idx
        self._apply_pick_hold_arm_cmd()
        where = "lower" if during_lower else "hold"
        print(
            f"[TaskB][DepthPCA] depth-cluster target during {where} -> acquire "
            f"center={self._grasp_target_cam} arm={self._pick_arm_hold_cmd}",
            flush=True,
        )
        return True

    def _run_pick_detect_stage(self, obs, pick_elapsed: int) -> None:
        if pick_elapsed < self._PICK_SCRIPTED_PRE_STEPS:
            self._set_video_hud(
                f"status=DETECT phase=pre-lower step={pick_elapsed}/{self._PICK_SCRIPTED_PRE_STEPS}",
                f"arm=[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}]",
            )
            # Stage 1: keep lowering while detecting.
            if pick_elapsed < 10:
                self.v_list[1] += 0.12
            self.v_list[1] += 0.12
            self.v_list[2] -= 0.07
            if pick_elapsed == 0:
                print("[TaskB][PICK_DETECT] scripted pre-lower start", flush=True)
            if (
                pick_elapsed >= self._PICK_DETECT_MIN_STEPS
                and pick_elapsed % self._PICK_DETECT_CHECK_EVERY == 0
            ):
                self._try_begin_depth_pick(obs, during_lower=True)
            return

        if pick_elapsed < (self._PICK_SCRIPTED_PRE_STEPS + self._PICK_SCRIPTED_HOLD_STEPS):
            hold_elapsed = pick_elapsed - self._PICK_SCRIPTED_PRE_STEPS
            self._set_video_hud(
                f"status=DETECT phase=hold-detect step={hold_elapsed}/{self._PICK_SCRIPTED_HOLD_STEPS}",
                f"arm=[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}]",
            )
            if pick_elapsed == self._PICK_SCRIPTED_PRE_STEPS:
                self._snapshot_pick_hold_arm_cmd()
                print(
                    f"[TaskB][PICK_DETECT] scripted bottom hold arm={self._pick_arm_hold_cmd}",
                    flush=True,
                )
            self._apply_pick_hold_arm_cmd()
            if pick_elapsed % self._PICK_DETECT_CHECK_EVERY == 0:
                self._try_begin_depth_pick(obs, during_lower=False)
            return

        # After detect window, force transition to grasp stage.
        self.status = Status.GRASP
        self._run_pick_grasp_stage(obs)

    def _run_pick_grasp_stage(self, obs) -> None:
        if self._DEPTH_TOUCH_ONLY_SCORE:
            self._update_ik_pick_state(obs)
            return
        if self._grasp_use_task_e_ik():
            self._update_task_e_grasp_ik(obs)
            return
        if self._grasp_use_graspnet():
            self._update_graspnet_grasp_ik(obs)
            return
        self._update_ik_pick_state(obs)

    def _update_ik_pick_state(self, obs) -> None:
        # ee_depth local point-cloud center + PCA principal axis.
        if self._pick_arm_hold_cmd is None:
            self._snapshot_pick_hold_arm_cmd()
        if self._ik_phase is None:
            self._ik_phase = "acquire"
            self._ik_phase_start = self.cur_idx
            self._ik_no_target_count = 0
            self._ik_pick_active = True
            self._last_target_cam = None
            self._apply_pick_hold_arm_cmd()
            print("[TaskB][DepthPCA] PICK acquire start (hold lowered pose)", flush=True)
            self._set_video_hud("status=GRASP ik=acquire step=0", "scanning ee_depth...")
            return

        elapsed = self.cur_idx - int(self._ik_phase_start)
        if self._ik_phase == "acquire":
            ee_depth = self._obs_depth(obs, "ee_depth")
            if self._grasp_target_cam is None:
                target_cam = self._pop_next_grasp_candidate(obs=obs)
                if target_cam is None or ee_depth is None:
                    self._ik_no_target_count += 1
                    if self._ik_no_target_count >= self._TOUCH_ACQUIRE_MAX_COUNT:
                        print(
                            "[TaskB][TouchScore] acquire timeout without ee_depth target -> STAND",
                            flush=True,
                        )
                        self._finish_grasp_to_stand()
                        return
                    scan = 0.08 if ((self._ik_no_target_count // 8) % 2 == 0) else -0.08
                    self._apply_pick_hold_arm_cmd(yaw_delta=scan)
                    self._set_video_hud(
                        f"status=GRASP ik=acquire step={elapsed}",
                        f"no_target count={self._ik_no_target_count}",
                    )
                    if self._ik_no_target_count % 15 == 0:
                        print(
                            f"[TaskB][DepthPCA] no target yet (count={self._ik_no_target_count})",
                            flush=True,
                        )
                    return
                self._grasp_target_cam = np.asarray(target_cam, dtype=np.float32).copy()

            # Lock-on policy: once grasp target is set, keep using it.
            # Optional PCA refine (upper-grasp center + bottle axis); off by default.
            if self._GRASP_USE_PCA and self._grasp_axis_cam is None and ee_depth is not None:
                grasp_pose = self._build_grasp_pose_from_ee_depth(ee_depth, self._grasp_target_cam)
                if grasp_pose is not None:
                    self._grasp_target_cam = grasp_pose["center_cam"].astype(np.float32)
                    self._grasp_axis_cam = grasp_pose["axis_cam"].astype(np.float32)
                    print(
                        f"[TaskB][DepthPCA] lock target center={self._grasp_target_cam} "
                        f"centroid={grasp_pose.get('centroid_cam')} h={grasp_pose.get('object_height')} "
                        f"axis={self._grasp_axis_cam}",
                        flush=True,
                    )

            self._last_target_cam = self._grasp_target_cam.copy()
            self._ik_phase = "approach"
            self._ik_phase_start = self.cur_idx
            self._grasp_approach_inited = False
            self._grasp_target_cam_ema = None
            self._grasp_reached_hold_count = 0
            self._grasp_prev_l_err = None
            self._grasp_prev_f_err = None
            self._grasp_prev_v_err = None
            self._apply_pick_hold_arm_cmd()
            print(f"[TaskB][DepthPCA] target acquired and locked center={self._grasp_target_cam}", flush=True)
            return

        if self._ik_phase == "approach":
            if not self._grasp_approach_inited:
                if self._pick_arm_hold_cmd is not None:
                    self.v_list = [float(v) for v in self._pick_arm_hold_cmd]
                self._grasp_approach_inited = True

            target_cam_raw = self._live_grasp_target_cam(obs)
            if target_cam_raw is None:
                self._log_grasp_debug("approach", phase="approach", elapsed=elapsed, note="no_target")
                self._set_video_hud(
                    f"status=GRASP ik=approach step={elapsed}/{self._GRASP_APPROACH_TIMEOUT}",
                    "no_target in ee_depth",
                )
                return
            if self._grasp_target_cam_ema is None:
                self._grasp_target_cam_ema = target_cam_raw.copy()
            else:
                alpha = float(np.clip(self._GRASP_TARGET_EMA_ALPHA, 0.0, 1.0))
                self._grasp_target_cam_ema = (
                    alpha * target_cam_raw + (1.0 - alpha) * self._grasp_target_cam_ema
                ).astype(np.float32)
            target_cam = self._grasp_target_cam_ema

            forward = float(-target_cam[2])
            lateral = float(-target_cam[0])
            vertical = float(-target_cam[1])
            dist_3d = float(np.linalg.norm(target_cam))

            f_err = forward - self._GRASP_TARGET_FORWARD
            l_err = lateral - self._GRASP_TARGET_LATERAL
            v_err = vertical - self._GRASP_TARGET_VERTICAL
            lateral_centered = abs(l_err) < self._GRASP_L_ERR_TOL

            d_clip = self._GRASP_D_ERR_CLIP
            if self._grasp_prev_l_err is None:
                d_l_err = 0.0
                d_f_err = 0.0
                d_v_err = 0.0
            else:
                d_l_err = float(np.clip(l_err - self._grasp_prev_l_err, -d_clip, d_clip))
                d_f_err = float(np.clip(f_err - self._grasp_prev_f_err, -d_clip, d_clip))
                d_v_err = float(np.clip(v_err - self._grasp_prev_v_err, -d_clip, d_clip))

            reached = (
                abs(f_err) < self._GRASP_F_ERR_TOL
                and abs(l_err) < self._GRASP_L_ERR_TOL
                and abs(v_err) < self._GRASP_V_ERR_TOL
            )
            if reached and self._grasp_reached_hold_count == 0:
                self._grasp_reached_hold_count = 1
            elif self._grasp_reached_hold_count > 0:
                self._grasp_reached_hold_count += 1

            hold_freeze = self._grasp_reached_hold_count > 0
            if hold_freeze:
                j0 = 0.0
                j1 = 0.0
                j2 = 0.0
            else:
                axis_yaw_bias = 0.0
                if (
                    self._GRASP_USE_PCA
                    and self._grasp_axis_cam is not None
                    and not lateral_centered
                ):
                    axis_yaw_bias = float(np.clip(0.25 * self._grasp_axis_cam[0], -0.08, 0.08))
                lateral_for_yaw = 0.0 if lateral_centered or abs(l_err) < self._GRASP_J0_DEADBAND else l_err
                d_l_for_j0 = 0.0 if lateral_centered else d_l_err
                j0_max, j1_max, j2_max = self._GRASP_JOINT_STEP_MAX
                j0_lat = self._GRASP_J0_LATERAL_SIGN * (
                    self._GRASP_J0_GAIN * lateral_for_yaw + self._GRASP_D_L_KD * d_l_for_j0
                )
                j0_raw = j0_lat + axis_yaw_bias
                j1_raw = (
                    self._GRASP_P_F_J1 * f_err
                    + self._GRASP_P_V_J1 * v_err
                    + self._GRASP_D_F_KD * d_f_err
                    + self._GRASP_D_V_KD * d_v_err
                )
                j2_raw = (
                    self._GRASP_P_F_J2 * f_err
                    + self._GRASP_P_V_J2 * v_err
                    - self._GRASP_D_F_KD * d_f_err
                    - self._GRASP_D_V_KD * d_v_err
                )
                j0 = float(np.clip(j0_raw, -j0_max, j0_max))
                j1 = float(np.clip(j1_raw, -j1_max, j1_max))
                j2 = float(np.clip(j2_raw, -j2_max, j2_max))
                self.v_list[0] += j0
                self.v_list[1] += j1
                self.v_list[2] += j2
                self._clamp_arm_cmd_to_hold()

            force_close = elapsed >= self._GRASP_APPROACH_TIMEOUT
            reached_hold_done = self._grasp_reached_hold_count >= self._GRASP_REACHED_HOLD_STEPS
            self._grasp_prev_l_err = float(l_err)
            self._grasp_prev_f_err = float(f_err)
            self._grasp_prev_v_err = float(v_err)
            self._set_video_hud(
                f"status=GRASP ik=approach step={elapsed}/{self._GRASP_APPROACH_TIMEOUT}",
                f"forward={forward:.3f} lat={lateral:+.3f} vert={vertical:+.3f} dist3d={dist_3d:.3f}",
                f"f_err={f_err:+.3f} l_err={l_err:+.3f} v_err={v_err:+.3f} tgt={self._GRASP_TARGET_FORWARD:.2f}",
                (
                    f"j0={j0:+.3f} j1={j1:+.3f} j2={j2:+.3f} "
                    f"reached={reached} hold={self._grasp_reached_hold_count}/{self._GRASP_REACHED_HOLD_STEPS} "
                    f"freeze={hold_freeze} timeout={force_close}"
                ),
                f"arm=[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}] "
                f"grip=[{self.v_list[6]:.2f},{self.v_list[7]:.2f}]",
            )
            if elapsed % self._GRASP_DEBUG_EVERY == 0 or force_close:
                self._log_grasp_debug(
                    "approach",
                    step=elapsed,
                    forward=f"{forward:.3f}",
                    lateral=f"{lateral:.3f}",
                    vertical=f"{vertical:.3f}",
                    f_err=f"{f_err:.3f}",
                    l_err=f"{l_err:.3f}",
                    v_err=f"{v_err:.3f}",
                    j0=f"{j0:.3f}",
                    j1=f"{j1:.3f}",
                    j2=f"{j2:.3f}",
                    arm=f"[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}]",
                    grip=f"[{self.v_list[6]:.2f},{self.v_list[7]:.2f}]",
                    reached=reached,
                    hold=f"{self._grasp_reached_hold_count}/{self._GRASP_REACHED_HOLD_STEPS}",
                    freeze=hold_freeze,
                    d_l=f"{d_l_err:.3f}",
                    d_f=f"{d_f_err:.3f}",
                    d_v=f"{d_v_err:.3f}",
                    force_close=force_close,
                )

            if (reached_hold_done and elapsed >= self._GRASP_MIN_APPROACH_STEPS) or force_close:
                if self._DEPTH_TOUCH_ONLY_SCORE:
                    if self._switch_to_next_grasp_target(
                        obs,
                        reason="reached_hold" if reached_hold_done else "approach_timeout",
                    ):
                        return
                    reason = "timeout" if force_close and not reached_hold_done else "reached_hold"
                    print(
                        "[TaskB][DepthPCA] approach done -> touch_score_complete "
                        f"({reason}), skip close/lift/go_bin",
                        flush=True,
                    )
                    self._set_video_hud(
                        "status=GRASP ik=touch_score_complete",
                        f"reason={reason.upper()} dist3d={dist_3d:.3f}",
                        "skip close/lift, return SEARCH after stand",
                    )
                    self._finish_grasp_to_stand()
                    return
                if self._moveit_ready and self._moveit_bridge is not None:
                    try:
                        ok = bool(
                            self._moveit_bridge.plan_and_execute_grasp(
                                {
                                    "center_cam": self._grasp_target_cam.tolist(),
                                    "axis_cam": (
                                        self._grasp_axis_cam.tolist()
                                        if self._grasp_axis_cam is not None
                                        else [1.0, 0.0, 0.0]
                                    ),
                                }
                            )
                        )
                    except Exception as e:
                        ok = False
                        print(f"[TaskB][MoveIt] execution failed: {e}", flush=True)
                    if ok:
                        self._ik_phase = "lift"
                        self._ik_phase_start = self.cur_idx
                        print("[TaskB][MoveIt] grasp executed, start lift", flush=True)
                        return
                self._grasp_wrist_yaw_target = self._estimate_grasp_wrist_yaw_target(obs)
                if self._GRASP_ALIGN_YAW_ENABLE and self._GRASP_ALIGN_YAW_STEPS > 0:
                    self._ik_phase = "align_yaw"
                else:
                    self._ik_phase = "close"
                self._ik_phase_start = self.cur_idx
                reason = "timeout" if force_close and not reached_hold_done else "reached_hold"
                self._grasp_close_reason = reason
                next_phase = self._ik_phase
                print(f"[TaskB][DepthPCA] approach done -> {next_phase} ({reason})", flush=True)
                not_reached = reason == "timeout"
                self._set_video_hud(
                    f"status=GRASP ik={next_phase}-enter reason={reason.upper()}",
                    "NOT REACHED - forced close by timeout" if not_reached else "position OK - closing gripper",
                    f"forward={forward:.3f} f_err={f_err:+.3f} dist3d={dist_3d:.3f}",
                    (
                        f"arm=[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}] "
                        f"wrist_yaw={self.v_list[3]:+.2f} yaw_tgt={self._grasp_wrist_yaw_target}"
                    ),
                )
                self._log_grasp_debug(
                    f"{next_phase}_enter",
                    step=elapsed,
                    forward=f"{forward:.3f}",
                    f_err=f"{f_err:.3f}",
                    arm=f"[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}]",
                )
            return

        if self._ik_phase == "align_yaw":
            if self._grasp_wrist_yaw_target is None:
                self._grasp_wrist_yaw_target = self._estimate_grasp_wrist_yaw_target(obs)
            if elapsed >= self._GRASP_ALIGN_YAW_STEPS:
                self._ik_phase = "close"
                self._ik_phase_start = self.cur_idx
                return
            yaw_tgt = self._grasp_wrist_yaw_target
            j3 = 0.0
            yaw_err = 0.0
            if yaw_tgt is not None:
                yaw_err = float(yaw_tgt - self.v_list[3])
                if abs(yaw_err) > self._GRASP_ALIGN_YAW_ERR_TOL:
                    j3_raw = self._GRASP_ALIGN_YAW_GAIN * yaw_err
                    j3 = float(np.clip(j3_raw, -self._GRASP_ALIGN_YAW_STEP, self._GRASP_ALIGN_YAW_STEP))
                    self.v_list[3] += j3
                    self._clamp_wrist_yaw_to_hold()
            self._set_video_hud(
                f"status=GRASP ik=align_yaw step={elapsed}/{self._GRASP_ALIGN_YAW_STEPS}",
                (
                    f"wrist_yaw={self.v_list[3]:+.3f} yaw_tgt="
                    f"{yaw_tgt:+.3f}" if yaw_tgt is not None else "wrist_yaw_target=none"
                ),
                f"yaw_err={yaw_err:+.3f} j3={j3:+.3f}",
            )
            if elapsed == 0 or elapsed % self._GRASP_DEBUG_EVERY == 0:
                self._log_grasp_debug(
                    "align_yaw",
                    step=elapsed,
                    j3=f"{j3:.3f}",
                    yaw=f"{self.v_list[3]:.3f}",
                    yaw_tgt=f"{yaw_tgt:.3f}" if yaw_tgt is not None else "none",
                    yaw_err=f"{yaw_err:.3f}",
                )
            return

        if self._ik_phase == "close" and elapsed >= self._IK_CLOSE_STEPS:
            self._ik_phase = "lift"
            self._ik_phase_start = self.cur_idx
            return

        if self._ik_phase == "close":
            # Keep current arm pose; do not reset to hold snapshot.
            self.v_list[1] += self._GRASP_CLOSE_PUSH_J1
            self.v_list[2] += self._GRASP_CLOSE_PUSH_J2
            self._clamp_arm_cmd_to_hold()
            self.v_list[6] = -1.0
            self.v_list[7] = -1.0
            reason = self._grasp_close_reason or "?"
            self._set_video_hud(
                f"status=GRASP ik=close step={elapsed}/{self._IK_CLOSE_STEPS} reason={reason.upper()}",
                f"arm=[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}] "
                f"grip=[{self.v_list[6]:.2f},{self.v_list[7]:.2f}]",
            )
            if elapsed == 0 or elapsed % self._GRASP_DEBUG_EVERY == 0:
                self._log_grasp_debug(
                    "close",
                    step=elapsed,
                    arm=f"[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}]",
                    grip=f"[{self.v_list[6]:.2f},{self.v_list[7]:.2f}]",
                )
            return

        if self._ik_phase == "lift" and elapsed >= self._IK_LIFT_STEPS:
            print("[TaskB][DepthPCA] GRASP done -> STAND", flush=True)
            self._finish_grasp_to_stand()
            return

        if self._ik_phase == "lift":
            self.v_list[1] -= 0.10
            self.v_list[2] += 0.08
            self._clamp_arm_cmd_to_hold()
            self.v_list[6] = -1.0
            self.v_list[7] = -1.0
            self._set_video_hud(
                f"status=GRASP ik=lift step={elapsed}/{self._IK_LIFT_STEPS}",
                f"arm=[{self.v_list[0]:.2f},{self.v_list[1]:.2f},{self.v_list[2]:.2f}] "
                f"grip=[{self.v_list[6]:.2f},{self.v_list[7]:.2f}]",
            )

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        b = proprio.shape[0]
        vx_cmd, vy_cmd, yaw_cmd = 0.0, 0.0, 0.0
        if self.auto:
            vx_cmd = self.cmd_max_vx
            yaw_cmd = self.cmd_max_wz
            vy_cmd = self.cmd_max_vy
        else:
            if self.key == 'w':
                vx_cmd = 1
            elif self.key == 's':
                vx_cmd = -1
            if self.key == 'a':
                yaw_cmd = 1.0
            elif self.key == 'd':
                yaw_cmd = -1.0
            if self.key == 'q':
                vy_cmd = self.cmd_max_vy
            elif self.key == 'e':
                vy_cmd = -self.cmd_max_vy

        return torch.tensor([[vx_cmd, vy_cmd, yaw_cmd]], device=proprio.device, dtype=proprio.dtype).repeat(b, 1)

    def _extract_policy_obs(self, obs, action_dim, cmd_override: tuple[float, float, float] | None = None) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        idx = 3
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 6
        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        if cmd_override is None:
            velocity_commands = self._get_velocity_commands(proprio)
        else:
            vx_cmd, vy_cmd, yaw_cmd = cmd_override
            velocity_commands = torch.tensor(
                [[vx_cmd, vy_cmd, yaw_cmd]], device=proprio.device, dtype=proprio.dtype
            ).repeat(proprio.shape[0], 1)

        return torch.cat([
            base_ang_vel * 0.25, projected_gravity, velocity_commands,
            joint_pos_leg, joint_vel_leg * 0.05, actions_train_leg,
        ], dim=-1)

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def find_start_point_optimized(self, head_depth):
        # 1. 计算每一行中相邻列的差值
        # diffs[y, x] = head_depth[y, x+1] - head_depth[y, x]
        # 我们关注的是当前像素比左侧像素小的情况，即 diff < -0.1

        # 获取所有行的 x=1 到 639 的数据与 x=0 到 638 的数据差值
        diffs = head_depth[:, 1:] - head_depth[:, :-1]

        # 2. 寻找满足条件的点 (diff < -0.1)
        # 得到一个布尔矩阵，形状为 (480, 639)
        mask = diffs < -0.05

        # 3. 从底部向上遍历行 (range(479, -1, -1))
        for y in range(479, -1, -1):
            # 找到该行中第一个为 True 的索引
            indices = np.where(mask[y])[0]
            if indices.size > 0:
                # 找到第一个符合条件的 x (注意因为我们用了 [:, 1:]，所以 x 需要 +1)
                x = indices[0] + 1
                return x, y
        return None

    def _detect_ground_ransac_numpy(
        self,
        points: np.ndarray,
        distance_threshold: float = 0.05,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
        n = len(points)
        if n < ransac_n:
            return np.zeros(n, dtype=bool), None
        best_inliers = None
        best_count = 0
        best_plane = None
        rng = np.random.default_rng(0)
        for _ in range(num_iterations):
            idx = rng.choice(n, ransac_n, replace=False)
            p1, p2, p3 = points[idx]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, p1))
            dists = np.abs(points @ normal + d)
            inliers = dists < distance_threshold
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_plane = (float(normal[0]), float(normal[1]), float(normal[2]), d)
        if best_inliers is None or best_plane is None:
            return np.zeros(n, dtype=bool), None
        return best_inliers, best_plane

    def detect_ground_ransac(self, points, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
        """
        使用RANSAC检测地面平面
        points: (N, 3) 点云
        distance_threshold: 点到平面的距离阈值(米)
        ransac_n: 每次采样点数(平面需要3)
        num_iterations: 迭代次数
        返回: (ground_mask, plane_model)
        """
        if points is None or len(points) < ransac_n:
            # Not enough points for plane fitting; treat all points as non-ground.
            return np.zeros(0, dtype=bool), None

        o3d = _try_import_o3d()
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            try:
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold, ransac_n, num_iterations
                )
            except RuntimeError:
                return np.zeros(len(points), dtype=bool), None
            ground_mask = np.zeros(len(points), dtype=bool)
            ground_mask[inliers] = True
            return ground_mask, plane_model

        return self._detect_ground_ransac_numpy(
            points,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

    def depth_to_point_cloud(self, depth_map, K: np.ndarray | None = None):
        height, width = depth_map.shape
        # 1. 创建像素网格
        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        # 2. 将像素坐标转换为相机坐标系下的点
        if K is None:
            K = self.K
        K = np.asarray(K, dtype=np.float64)
        z = -depth_map
        x = -(i - K[0, 2]) * z / K[0, 0]
        y = (j - K[1, 2]) * z / K[1, 1]

        # 3. 堆叠成 (N, 3) 的点云数组
        point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # 过滤掉无效深度值 (例如远端截断值)
        valid_mask = (depth_map > 0.05) & (depth_map < 50.0)
        return point_cloud[valid_mask.reshape(-1)]

    def transform_ground_to_zero(self, points, plane_model):
        """
        points: (N, 3) 原始点云
        plane_model: (a, b, c, d) 平面参数
        """
        if plane_model is None:
            return points, np.eye(3), np.zeros(3)
        a, b, c, d = plane_model
        normal = np.array([a, b, c])

        # 1. 计算旋转矩阵 (Rotation Matrix)
        # 目标是将 normal 旋转到 [0, 0, 1]
        target_normal = np.array([0, 0, 1])

        # 使用罗德里格斯旋转公式或求旋转轴和角
        v = np.cross(normal, target_normal)
        s = np.linalg.norm(v)
        c_val = np.dot(normal, target_normal)

        # 反对称矩阵
        v_skew = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])

        # 旋转矩阵 R
        R = np.eye(3) + v_skew + np.dot(v_skew, v_skew) * ((1 - c_val) / (s ** 2 + 1e-9))

        # 2. 计算平移向量
        # 平面上距离原点最近的点是 P0 = -d * normal
        p0 = -d * normal

        # 3. 构建变换矩阵 T (从世界到地面坐标系的变换)
        # 新坐标 P' = R * (P - p0)
        # 这一步使地面点旋转到水平且 p0 移到原点
        points_transformed = (R @ (points - p0).T).T

        return points_transformed, R, p0

    def _cluster_dbscan_sklearn(
        self,
        points: np.ndarray,
        eps: float = 0.1,
        min_points: int = 10,
    ) -> tuple[np.ndarray, int]:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=eps, min_samples=min_points).fit(points).labels_
        labels = np.asarray(labels, dtype=np.int32)
        n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
        return labels, n_clusters

    def cluster_euclidean_open3d(self, points, eps=0.1, min_points=10, max_points=10000):
        """
        DBSCAN clustering for depth objects. Uses Open3D when available, else sklearn.
        """
        o3d = _try_import_o3d()
        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            cluster_labels = np.array(
                pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
            )
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if self._DEPTH_CLUSTER_DEBUG or self._GRASP_DEBUG:
                print(f"Open3D聚类: 找到 {n_clusters} 个物体", flush=True)
            return cluster_labels, n_clusters

        cluster_labels, n_clusters = self._cluster_dbscan_sklearn(points, eps=eps, min_points=min_points)
        if self._DEPTH_CLUSTER_DEBUG or self._GRASP_DEBUG:
            print(f"sklearn DBSCAN: 找到 {n_clusters} 个物体", flush=True)
        return cluster_labels, n_clusters

    def find_target_by_depth(self, depth):
        if depth is None:
            return None, None
        depth = self._numpy_depth_2d(depth)
        if depth is None:
            return None, None

        points = self.depth_to_point_cloud(depth)
        if points is None or len(points) < 10:
            return None, None

        ground_mask, plane_model = self.detect_ground_ransac(points, distance_threshold=0.03)
        if plane_model is None or len(ground_mask) != len(points):
            ground_mask = np.zeros(len(points), dtype=bool)

        points_flat, R, p0 = self.transform_ground_to_zero(points, plane_model)
        others = points_flat[~ground_mask]
        if len(others) < 5:
            return None, None
        self._search_last_cloud_zmax = float(np.max(np.abs(others[:, 2])))

        # Hard-cap clustering size to keep SEARCH runtime stable over long episodes.
        cluster_points = others
        if self._SEARCH_CLUSTER_MAX_POINTS > 0 and len(cluster_points) > self._SEARCH_CLUSTER_MAX_POINTS:
            stride = max(1, int(np.ceil(len(cluster_points) / float(self._SEARCH_CLUSTER_MAX_POINTS))))
            cluster_points = cluster_points[::stride]

        labels, n_clusters = self.cluster_euclidean_open3d(cluster_points)
        if n_clusters == 0:
            return None, None

        candidates: list[tuple[np.ndarray, float, float]] = []
        for label in range(n_clusters):
            cur_points = cluster_points[labels == label]
            if len(cur_points) == 0:
                continue
            z_max = float(np.max(np.abs(cur_points[:, 2])))
            # Optional legacy geometric bin filter (disabled by default).
            if self._SEARCH_BIN_FILTER_ENABLE and z_max >= self._SEARCH_BIN_FILTER_MIN_HEIGHT:
                continue
            centroid = np.mean(cur_points, axis=0)
            dist_to_origin = np.linalg.norm(centroid)
            candidates.append((centroid, float(dist_to_origin), z_max))

        if len(candidates) == 0:
            return None, None

        # Base choice: globally nearest cluster.
        target, min_dist, target_zmax = min(candidates, key=lambda item: item[1])
        self._search_last_target_zmax = float(target_zmax)

        # Continuity constraint:
        # once close enough (< enable_dist), prefer cluster nearest to last target.
        if self._search_track_target is not None and (
            min_dist < self._SEARCH_TRACK_ENABLE_DIST
            or float(np.linalg.norm(self._search_track_target)) < self._SEARCH_TRACK_ENABLE_DIST
        ):
            cont_target, _, cont_zmax = min(
                candidates,
                key=lambda item: float(np.linalg.norm(item[0] - self._search_track_target)),
            )
            jump = float(np.linalg.norm(cont_target - self._search_track_target))
            if jump <= self._SEARCH_TRACK_MAX_JUMP:
                target = cont_target
                min_dist = float(np.linalg.norm(target))
                self._search_last_target_zmax = float(cont_zmax)
        self._search_track_target = target.copy()

        self._last_target_class = None
        self._last_target_score = None
        return target, min_dist

    def predicts(self, obs, current_score):
        self.init()
        if not self._platform_obs_logged:
            image = obs.get("image") or {}
            keys = sorted(image.keys())
            extero = obs.get("extero")
            print(
                f"[TaskB][Platform] deploy={self._platform_deploy()} "
                f"image_keys={keys} extero={'yes' if extero is not None else 'no'}",
                flush=True,
            )
            self._platform_obs_logged = True
        if self.vis is not None:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.cur_idx += 1
        self._check_platform_score_touch(current_score)

        if self.status == Status.SEARCH:
            if self._DIRECT_SQUAT_IK and self.start_get_down_idx is None:
                self.status = Status.LOCK
                self.get_down = True
                self.start_get_down_idx = self.cur_idx
                self._lock_prepare_start_idx = None
                self._snapshot_squat_start_leg_action(obs)
                self.cmd_max_vx = 0.0
                self.cmd_max_vy = 0.0
                self.cmd_max_wz = 0.0
                print("[TaskB] direct LOCK -> squat for IK pick", flush=True)
            near_commit_zone = self._search_near_commit_zone()
            should_detect = (self.cur_idx % 4 == 0) or near_commit_zone
            if should_detect and not self._DIRECT_SQUAT_IK:
                target, min_dist = self._find_search_target_from_obs(obs)
                memory_target_used = False
                if (
                    target is None
                    and self._search_track_target is not None
                    and self._search_last_seen_step is not None
                ):
                    age = int(self.cur_idx - self._search_last_seen_step)
                    if age <= self._SEARCH_LOST_MEMORY_STEPS:
                        target = np.asarray(self._search_track_target, dtype=np.float64).copy()
                        min_dist = float(np.linalg.norm(target))
                        memory_target_used = True
                        self._search_memory_age = age
                bin_avoid_active = False
                if (
                    target is not None
                    and (not memory_target_used)
                    and self._SEARCH_BIN_AVOID_ENABLE
                ):
                    rgb_lidar_hit, rgb_lidar_info = self._search_target_bin_rgb_lidar(obs, target)
                    if self.cur_idx % max(self._SEARCH_DEBUG_PRINT_EVERY, 1) == 0:
                        print(
                            "[TaskB][SEARCH][bin-check] "
                            f"hit={rgb_lidar_hit} "
                            f"bearing_err={rgb_lidar_info.get('bearing_err')} "
                            f"yellow={rgb_lidar_info.get('yellow_ratio')} "
                            f"bin_bearing={rgb_lidar_info.get('bin_bearing')} "
                            f"target_bearing={rgb_lidar_info.get('target_bearing')}",
                            flush=True,
                        )
                    if not rgb_lidar_hit:
                        rgb_lidar_info = {}
                    else:
                        self._lock_commit_confirm_count = 0
                        self._search_memory_age = None
                        tx = float(target[0])
                        ty = float(target[1])
                        target_bearing = float(np.arctan2(ty, tx))
                        self.cmd_max_vx = self._SEARCH_BIN_AVOID_REVERSE_VX
                        self.cmd_max_vy = 0.0
                        self.cmd_max_wz = float(np.clip(
                            -self._SEARCH_BIN_AVOID_WZ_GAIN * target_bearing,
                            -self._SEARCH_BIN_AVOID_MAX_WZ,
                            self._SEARCH_BIN_AVOID_MAX_WZ,
                        ))
                        self._set_video_hud(
                            "status=SEARCH phase=bin-avoid rgb+lidar",
                            f"rgb_lidar={rgb_lidar_hit} yellow={rgb_lidar_info.get('yellow_ratio')}",
                            f"cmd_vx={self.cmd_max_vx:.2f} cmd_wz={self.cmd_max_wz:.2f}",
                        )
                        print(
                            f"[TaskB][SEARCH] bin-avoid rgb_lidar_hit={rgb_lidar_hit} "
                            f"yellow={rgb_lidar_info.get('yellow_ratio')} "
                            f"bearing_err={rgb_lidar_info.get('bearing_err')} target=({tx:.2f},{ty:.2f}) "
                            f"cmd=({self.cmd_max_vx:.2f},{self.cmd_max_wz:.2f})",
                            flush=True,
                        )
                        bin_avoid_active = True
                # Separate confirmation gate for commit-lock branch to avoid single-frame false locks.
                if (not bin_avoid_active) and (
                    target is not None
                    and min_dist is not None
                    and (not memory_target_used)
                    and float(min_dist) <= self._LOCK_LOST_COMMIT_DIST
                ):
                    self._lock_commit_confirm_count += 1
                else:
                    self._lock_commit_confirm_count = 0
                commit_lock, commit_reason = self._search_target_implies_near_loss(target, min_dist)
                if bin_avoid_active:
                    pass
                elif commit_lock and self._lock_commit_confirm_count >= max(self._LOCK_COMMIT_CONFIRM_STEPS, 1):
                    self._begin_lock_prezero(commit_reason)
                elif target is not None:
                    self._search_no_target_steps = 0
                    if (not memory_target_used) and min_dist is not None:
                        self._search_last_seen_dist = float(min_dist)
                        self._search_last_seen_step = self.cur_idx
                    vx_cap = 1.0
                    wz_cap = 1.0
                    if memory_target_used:
                        vx_cap = self._SEARCH_LOST_MEMORY_VX_CAP
                        wz_cap = self._SEARCH_LOST_MEMORY_WZ_CAP
                    elif near_commit_zone or (
                        min_dist is not None
                        and float(min_dist) <= self._LOCK_LOST_COMMIT_DIST
                    ):
                        vx_cap = self._LOCK_COMMIT_VX_CAP
                        wz_cap = 0.5
                    elif min_dist is not None and float(min_dist) < self._SEARCH_DECEL_START_DIST:
                        t = float(np.clip(float(min_dist) / self._SEARCH_DECEL_START_DIST, 0.0, 1.0))
                        vx_cap = self._SEARCH_SLOW_VX + (1.0 - self._SEARCH_SLOW_VX) * t
                        wz_cap = max(0.4, t)
                    self.calculate_velocity(
                        target[0],
                        target[1],
                        vx_cap,
                        0.5,
                        wz_cap,
                        lock_on_arrive=not memory_target_used,
                    )
                else:
                    # No target in SEARCH: use coverage cruise after short timeout.
                    self._search_no_target_steps += 1
                    if self._search_no_target_steps >= self._SEARCH_NO_TARGET_WP_TIMEOUT_STEPS:
                        if not self._update_search_coverage_cmd():
                            self.cmd_max_vx = 0.0
                            self.cmd_max_vy = 0.0
                            self.cmd_max_wz = self._SEARCH_NO_TARGET_WZ
                    else:
                        self.cmd_max_vx = 0.0
                        self.cmd_max_vy = 0.0
                        self.cmd_max_wz = self._SEARCH_NO_TARGET_WZ
                    self._search_memory_age = None
                if self.cur_idx % max(self._SEARCH_DEBUG_PRINT_EVERY, 1) == 0:
                    print(
                        f"[SEARCH] target={target}, min_dist={min_dist}, "
                        f"last_seen={self._search_last_seen_dist}"
                        f"{'' if self._search_memory_age is None else f' mem_age={self._search_memory_age}'}",
                        flush=True,
                    )
            pre_steps = 0
            if self._lock_prepare_start_idx is not None:
                pre_steps = self.cur_idx - self._lock_prepare_start_idx
            self._set_video_hud(
                f"status=SEARCH cmd_vx={self.cmd_max_vx:.2f} cmd_wz={self.cmd_max_wz:.2f}",
            )

        elif self.status == Status.LOCK:
            if not self.get_down:
                if self._lock_prepare_start_idx is None:
                    self._lock_prepare_start_idx = self.cur_idx
                pre_elapsed = self.cur_idx - self._lock_prepare_start_idx
                # Pre-zero: hold still (no depth clustering / no body yaw correction).
                self.cmd_max_vx = 0.0
                self.cmd_max_vy = 0.0
                self.cmd_max_wz = 0.0
                self._set_video_hud(
                    f"status=LOCK phase=pre-zero step={pre_elapsed}/{self._LOCK_PRE_ZERO_STEPS}",
                    "cmd_vx=0.00 cmd_vy=0.00 cmd_wz=0.00",
                )
                if pre_elapsed >= self._LOCK_PRE_ZERO_STEPS:
                    self._snapshot_squat_start_leg_action(obs)
                    self.get_down = True
                    self.start_get_down_idx = self.cur_idx
                    self._lock_prepare_start_idx = None
                    self._lock_prezero_settle_count = 0
                    print(
                        f"[TaskB] lock pre-zero done -> lock policy "
                        f"(steps={pre_elapsed}/{self._LOCK_PRE_ZERO_STEPS})",
                        flush=True,
                    )
            else:
                squat_elapsed = 0
                if self.start_get_down_idx is not None:
                    squat_elapsed = self.cur_idx - self.start_get_down_idx
                self._set_video_hud(
                    f"status=LOCK phase=policy step={squat_elapsed}/{self._LOCK_POLICY_STEPS}",
                )

        elif self.status == Status.DETECT and self.start_pick_idx is not None:
            pick_elapsed = self.cur_idx - self.start_pick_idx
            self._run_pick_detect_stage(obs, pick_elapsed)
        elif self.status == Status.GRASP and self.start_pick_idx is not None:
            self._run_pick_grasp_stage(obs)

        elif self.status == Status.STAND and self.start_stand_idx is not None:
            stand_elapsed = self.cur_idx - self.start_stand_idx
            self._set_video_hud(
                f"status=STAND step={stand_elapsed}/{self._STAND_STEPS}",
            )
            if self.cur_idx >= self.start_stand_idx + self._STAND_STEPS:
                if self._DEPTH_TOUCH_ONLY_SCORE:
                    self._begin_search_cycle()
                    print("[TaskB] STAND done -> SEARCH (touch-score mode)", flush=True)
                else:
                    self.status = Status.GO_BIN
                    print("[TaskB] STAND done -> GO_BIN (LiDAR yaw + depth range)", flush=True)

        elif self.status == Status.GO_BIN:
            if self.cur_idx % 4 == 0:
                self._update_go_bin_cmd(obs)
                print(
                    f"[GO_BIN] bearing={self._last_bin_bearing} dist={self._last_bin_dist}",
                    flush=True,
                )
            self._set_video_hud(
                f"status=GO_BIN bearing={self._last_bin_bearing} dist={self._last_bin_dist}",
            )

        if self._apply_bin_collision_guard(obs):
            print(
                f"[TaskB] bin guard active: dist={self._last_bin_dist} bearing={self._last_bin_bearing}",
                flush=True,
            )

        proprio = obs["proprio"].to(self.device)

        # ==========================================
        # 策略推理
        # ==========================================
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        use_stand_policy = self._stand_policy is not None and self.status == Status.STAND
        use_lock_grasp_policy = (
            self._lock_grasp_policy is not None
            and (
                self.status == Status.GRASP
                or self.status == Status.DETECT
                or (self.status == Status.LOCK and self.get_down)
            )
        )
        policy_obs = self._extract_policy_obs(
            obs,
            action_dim,
            cmd_override=(0.0, 0.0, 0.0) if (use_lock_grasp_policy or use_stand_policy) else None,
        )

        with torch.inference_mode():
            if use_stand_policy:
                action_train = self._stand_policy(policy_obs)
            elif use_lock_grasp_policy:
                action_train = self._lock_grasp_policy(policy_obs)
            else:
                action_train = self.policy(policy_obs)

        action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        if action_train.ndim == 1: action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        # return {'action': action_env.cpu().numpy().tolist(), 'giveup': False}

        proprio = obs['proprio']
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        action = [0 for _ in range(action_dim)]
        root_z = self._root_height_z()
        if (
            self.status == Status.LOCK
            and self.get_down
            and self.start_get_down_idx is not None
            and (
                self.cur_idx >= self.start_get_down_idx + self._LOCK_POLICY_STEPS
                or (
                    root_z is not None
                    and root_z <= self._LOCK_DETECT_MIN_ROOT_Z
                )
            )
        ):
            self.status = Status.DETECT
            self.start_pick_idx = self.cur_idx
            self._squat_start_leg_action = None
            self._ik_phase = None
            self._ik_phase_start = None
            self._ik_target_pos_b = None
            self._ik_lift_pos_b = None
            self._ik_acquire_pos_b = None
            self._ik_no_target_count = 0
            self._ik_pick_active = False
            self._pick_arm_hold_cmd = None
            self._grasp_target_cam = None
            self._grasp_axis_cam = None
            self.v_list = [0.0 for _ in range(8)]
            print("[TaskB] squat done -> DETECT", flush=True)
        hold_legs_at_current = False
        use_policy_legs = self.status in (Status.SEARCH, Status.STAND, Status.GO_BIN) or (
            self.status == Status.LOCK and not self.get_down
        ) or (
            self.status in (Status.LOCK, Status.DETECT, Status.GRASP)
            and (self.status != Status.LOCK or self.get_down)
            and self._lock_grasp_policy is not None
        )
        self._set_squat_leg_pd(not use_policy_legs)
        if hold_legs_at_current:
            # During DETECT/GRASP, freeze legs at current pose with position control.
            action[:12] = self._leg_action_from_proprio(obs)
        elif use_policy_legs:
            aa = action_env.cpu().numpy().tolist()
            action[:action_dim] = aa[0]
        else:
            self.cmd_max_vx = 0.0
            self.cmd_max_vy = 0.0
            self.cmd_max_wz = 0.0
            squat_elapsed = 0
            if self.start_get_down_idx is not None:
                squat_elapsed = self.cur_idx - self.start_get_down_idx
            squat_pose = self._lock_squat_leg_action(squat_elapsed)
            if (
                self.status == Status.LOCK
                and self.get_down
                and
                self.start_get_down_idx is not None
                and (
                    self.cur_idx >= self.start_get_down_idx + self._LOCK_POLICY_STEPS
                    or (
                        root_z is not None
                        and root_z <= self._LOCK_DETECT_MIN_ROOT_Z
                    )
                )
            ):
                self.status = Status.DETECT
                self.start_pick_idx = self.cur_idx
                self._squat_start_leg_action = None
                self._ik_phase = None
                self._ik_phase_start = None
                self._ik_target_pos_b = None
                self._ik_lift_pos_b = None
                self._ik_acquire_pos_b = None
                self._ik_no_target_count = 0
                self._ik_pick_active = False
                self._pick_arm_hold_cmd = None
                self._grasp_target_cam = None
                self._grasp_axis_cam = None
                self.v_list = [0.0 for _ in range(8)]
                print("[TaskB] squat done -> DETECT", flush=True)
            action[:12] = squat_pose
        if self.status in (Status.DETECT, Status.GRASP):
            action[12:20] = self.v_list
        else:
            action[12:20] = [0.0 for _ in range(8)]
        return {'action': [float(x) for x in action], 'giveup': False}
