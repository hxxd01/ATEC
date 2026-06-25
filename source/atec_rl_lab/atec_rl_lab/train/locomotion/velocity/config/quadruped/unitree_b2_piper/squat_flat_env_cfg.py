"""Backward-compatible aliases for squat transition env config."""

from .squat_env_cfg import UnitreeB2PiperSquatFlatEnvCfg, squat_target_pose_exp

# Keep old class name import paths working.
UnitreeB2PiperFlatSquatEnvCfg = UnitreeB2PiperSquatFlatEnvCfg
