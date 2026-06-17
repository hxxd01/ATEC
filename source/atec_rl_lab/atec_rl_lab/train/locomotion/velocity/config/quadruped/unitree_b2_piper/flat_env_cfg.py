from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.rough_env_cfg import (
    UnitreeB2PiperRoughEnvCfg,
)


@configclass
class UnitreeB2PiperFlatEnvCfg(UnitreeB2PiperRoughEnvCfg):
    """Flat-ground B2Piper locomotion (same rewards/obs as rough, plane terrain, no height scan)."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "UnitreeB2PiperFlatEnvCfg":
            self.disable_zero_weight_rewards()
