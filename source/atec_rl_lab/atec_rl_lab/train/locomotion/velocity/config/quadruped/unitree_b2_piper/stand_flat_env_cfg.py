"""Backward-compatible aliases for stand transition env config."""

from .stand_env_cfg import UnitreeB2PiperStandFlatEnvCfg, stand_target_pose_exp

# Keep old class name import paths working.
UnitreeB2PiperFlatStandEnvCfg = UnitreeB2PiperStandFlatEnvCfg
