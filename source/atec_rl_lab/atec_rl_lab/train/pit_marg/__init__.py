"""Task D pit-crossing MARG locomotion: shared train/play (MargActorCritic + pit env rewards/terminations)."""

from .taskd_pit_marg_runner import play_pit_marg, register_marg_modules, train_pit_marg

__all__ = ["play_pit_marg", "register_marg_modules", "train_pit_marg"]
