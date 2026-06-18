"""MARG-style asymmetric actor-critic with Estimator + Elevation encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP

from atec_rl_lab.train.locomotion.marg.constants import (
    MARG_CRITIC_PRIV_DIM,
    MARG_ELEVATION_OUT_DIM,
    MARG_ESTIMATOR_CONTACT_DIM,
    MARG_ESTIMATOR_OUT_DIM,
    MARG_ESTIMATOR_VEL_DIM,
    MARG_HEIGHT_MAP_DIM,
    MARG_HISTORY_DIM,
    MARG_PROPRIO_DIM,
)


class MargActorCritic(nn.Module):
    """Asymmetric AC: actor uses proprio + estimated state + elevation; critic adds privileged state."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        *,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list | None = None,
        critic_hidden_dims: list | None = None,
        activation: str = "relu",
        init_noise_std: float = 1.0,
        max_noise_std: float = 2.0,
        noise_std_type: str = "scalar",
        proprio_dim: int = MARG_PROPRIO_DIM,
        history_dim: int = MARG_HISTORY_DIM,
        height_map_dim: int = MARG_HEIGHT_MAP_DIM,
        critic_priv_dim: int = MARG_CRITIC_PRIV_DIM,
        estimator_hidden_dims: list | None = None,
        elevation_hidden_dims: list | None = None,
        **kwargs,
    ):
        if kwargs:
            print(f"[MargActorCritic] Ignoring extra kwargs: {list(kwargs)}")
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        if estimator_hidden_dims is None:
            estimator_hidden_dims = [128]
        if elevation_hidden_dims is None:
            elevation_hidden_dims = [128, 64]

        self.obs_groups = obs_groups
        self.proprio_dim = int(proprio_dim)
        self.history_dim = int(history_dim)
        self.height_map_dim = int(height_map_dim)
        self.critic_priv_dim = int(critic_priv_dim)
        self.estimator_out_dim = MARG_ESTIMATOR_OUT_DIM
        self.elevation_out_dim = MARG_ELEVATION_OUT_DIM

        self._validate_obs(obs)

        # Estimator: history -> (v_hat, c_hat)
        self.estimator = MLP(self.history_dim, self.estimator_out_dim, estimator_hidden_dims, activation)
        # Elevation encoder: relative height map -> features
        self.elevation = MLP(self.height_map_dim, self.elevation_out_dim, elevation_hidden_dims, activation)

        actor_in = self.proprio_dim + self.estimator_out_dim + self.elevation_out_dim
        critic_in = self.proprio_dim + self.critic_priv_dim + self.elevation_out_dim

        self.actor = MLP(actor_in, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_in, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(actor_in) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_in) if critic_obs_normalization else nn.Identity()
        )

        self.noise_std_type = noise_std_type
        self.max_noise_std = float(max_noise_std)
        self.min_noise_std = 1e-3
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(
            f"[MargActorCritic] estimator {self.history_dim}->{self.estimator_out_dim}, "
            f"elevation {self.height_map_dim}->{self.elevation_out_dim}, "
            f"actor_in={actor_in}, critic_in={critic_in}, "
            f"noise_std=[{init_noise_std:.2f}, {self.max_noise_std:.2f}]",
            flush=True,
        )
        print(f"[MargActorCritic] actor MLP: {self.actor}", flush=True)
        print(f"[MargActorCritic] critic MLP: {self.critic}", flush=True)

    def _validate_obs(self, obs: dict) -> None:
        expected = {
            "proprio": self.proprio_dim,
            "proprio_history": self.history_dim,
            "height_map": self.height_map_dim,
            "critic_priv": self.critic_priv_dim,
        }
        for key, dim in expected.items():
            if key not in obs:
                raise KeyError(f"MargActorCritic requires observation group '{key}'")
            if obs[key].shape[-1] != dim:
                raise ValueError(
                    f"Observation '{key}' dim {obs[key].shape[-1]} != expected {dim}. "
                    "Check MARG observation terms in env cfg."
                )

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _obs_tensor(self, obs: dict, key: str) -> torch.Tensor:
        return obs[key]

    def _encode_actor(self, obs: dict) -> torch.Tensor:
        proprio = self._obs_tensor(obs, "proprio")
        history = self._obs_tensor(obs, "proprio_history")
        height = self._obs_tensor(obs, "height_map")
        est_out = self.estimator(history)
        elev_out = self.elevation(height)
        return torch.cat([proprio, est_out, elev_out], dim=-1)

    def _encode_critic(self, obs: dict) -> torch.Tensor:
        proprio = self._obs_tensor(obs, "proprio")
        height = self._obs_tensor(obs, "height_map")
        priv = self._obs_tensor(obs, "critic_priv")
        elev_out = self.elevation(height)
        return torch.cat([proprio, priv, elev_out], dim=-1)

    def _clamped_action_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        return torch.clamp(std, min=self.min_noise_std, max=self.max_noise_std)

    def update_distribution(self, actor_obs: torch.Tensor):
        mean = self.actor(actor_obs)
        std = self._clamped_action_std(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.actor_obs_normalizer(self._encode_actor(obs))
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self._encode_actor(obs))
        return self.actor(actor_obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self._encode_critic(obs))
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actor_obs(self, obs):
        return self._encode_actor(obs)

    def get_critic_obs(self, obs):
        return self._encode_critic(obs)

    def estimator_regression_loss(self, obs) -> torch.Tensor:
        """MARG loss_reg: MSE(v_hat, v) + MSE(c_hat, c) using privileged targets."""
        est_out = self.estimator(self._obs_tensor(obs, "proprio_history"))
        targets = self._obs_tensor(obs, "critic_priv")[:, : self.estimator_out_dim]
        v_dim = MARG_ESTIMATOR_VEL_DIM
        c_dim = MARG_ESTIMATOR_CONTACT_DIM
        v_hat = est_out[:, :v_dim]
        c_hat = est_out[:, v_dim : v_dim + c_dim]
        v_tgt = targets[:, :v_dim]
        c_tgt = targets[:, v_dim : v_dim + c_dim]
        loss_v = torch.nn.functional.mse_loss(v_hat, v_tgt)
        loss_c = torch.nn.functional.mse_loss(c_hat, c_tgt)
        return loss_v + loss_c

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
