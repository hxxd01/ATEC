"""MARG-style pit actor-critic: depth CNN replaces elevation MLP (deploy-friendly)."""

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
    MARG_HISTORY_DIM,
    MARG_PROPRIO_DIM,
)
from atec_rl_lab.train.nav.taskd_student_actor_critic import ConvEncoder


class MargDepthPitActorCritic(nn.Module):
    """MARG asymmetric AC: estimator(history) + depth CNN (replaces height-map elevation)."""

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
        depth_dim: int | None = None,
        img_h: int = 24,
        img_w: int = 32,
        depth_channels: int = 1,
        enc_dim: int = 128,
        critic_priv_dim: int = MARG_CRITIC_PRIV_DIM,
        estimator_hidden_dims: list | None = None,
        depth_hidden_dims: list | None = None,
        elevation_out_dim: int = MARG_ELEVATION_OUT_DIM,
        head_depth_only: bool | None = None,
        **kwargs,
    ):
        if kwargs:
            print(f"[MargDepthPitActorCritic] Ignoring extra kwargs: {list(kwargs)}")
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        if estimator_hidden_dims is None:
            estimator_hidden_dims = [128]
        if depth_hidden_dims is None:
            depth_hidden_dims = [128, 64]

        self.obs_groups = obs_groups
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.depth_channels = int(depth_channels)
        self.depth_flat_per_cam = self.depth_channels * self.img_h * self.img_w
        if depth_dim is None and isinstance(obs, dict) and "depth" in obs:
            depth_dim = int(obs["depth"].shape[-1])
        if depth_dim is None:
            depth_dim = self.depth_flat_per_cam if head_depth_only else 2 * self.depth_flat_per_cam
        if head_depth_only is None:
            head_depth_only = int(depth_dim) <= self.depth_flat_per_cam
        self.depth_dim = int(depth_dim)
        self.head_depth_only = bool(head_depth_only)
        self.num_depth_cams = max(1, self.depth_dim // self.depth_flat_per_cam)
        if self.depth_dim != self.num_depth_cams * self.depth_flat_per_cam:
            raise ValueError(
                f"depth_dim {self.depth_dim} is not a multiple of per-cam flat "
                f"{self.depth_flat_per_cam} ({self.depth_channels}x{self.img_h}x{self.img_w})"
            )
        self.proprio_dim = int(proprio_dim)
        self.history_dim = int(history_dim)
        self.critic_priv_dim = int(critic_priv_dim)
        self.estimator_out_dim = MARG_ESTIMATOR_OUT_DIM
        self.elevation_out_dim = int(elevation_out_dim)

        self._validate_obs(obs)

        self.estimator = MLP(self.history_dim, self.estimator_out_dim, estimator_hidden_dims, activation)
        self.head_depth_encoder = ConvEncoder(in_ch=self.depth_channels, out_dim=enc_dim)
        if self.head_depth_only:
            self.ee_depth_encoder = None
            self.depth_fusion = MLP(enc_dim, self.elevation_out_dim, depth_hidden_dims, activation)
        else:
            self.ee_depth_encoder = ConvEncoder(in_ch=self.depth_channels, out_dim=enc_dim)
            self.depth_fusion = MLP(enc_dim + enc_dim, self.elevation_out_dim, depth_hidden_dims, activation)

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

        cam_tag = "head" if self.head_depth_only else "head+ee"
        print(
            f"[MargDepthPitActorCritic] estimator {self.history_dim}->{self.estimator_out_dim}, "
            f"depth CNN {cam_tag} {self.depth_channels}x{self.img_h}x{self.img_w}->{self.elevation_out_dim}, "
            f"actor_in={actor_in}, critic_in={critic_in}, actions={num_actions}",
            flush=True,
        )

    def _validate_obs(self, obs: dict) -> None:
        expected = {
            "proprio": self.proprio_dim,
            "proprio_history": self.history_dim,
            "depth": self.depth_dim,
            "critic_priv": self.critic_priv_dim,
        }
        for key, dim in expected.items():
            if key not in obs:
                raise KeyError(f"MargDepthPitActorCritic requires observation group '{key}'")
            if obs[key].shape[-1] != dim:
                raise ValueError(
                    f"Observation '{key}' dim {obs[key].shape[-1]} != expected {dim}."
                )

    def _reshape_head_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        batch = depth_flat.shape[0]
        if self.head_depth_only:
            flat = depth_flat
        else:
            flat = depth_flat[:, : self.depth_flat_per_cam]
        return flat.reshape(batch, self.depth_channels, self.img_h, self.img_w)

    def _reshape_ee_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        batch = depth_flat.shape[0]
        ee = depth_flat[:, self.depth_flat_per_cam :]
        return ee.reshape(batch, self.depth_channels, self.img_h, self.img_w)

    def _encode_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        head = self._reshape_head_depth(depth_flat)
        head_feat = self.head_depth_encoder(head)
        if self.head_depth_only:
            return self.depth_fusion(head_feat)
        if self.ee_depth_encoder is None:
            raise RuntimeError("ee_depth_encoder missing in dual-camera mode")
        ee = self._reshape_ee_depth(depth_flat)
        feat = torch.cat([head_feat, self.ee_depth_encoder(ee)], dim=-1)
        return self.depth_fusion(feat)

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

    def _encode_actor(self, obs: dict) -> torch.Tensor:
        proprio = obs["proprio"]
        history = obs["proprio_history"]
        depth = obs["depth"]
        est_out = self.estimator(history)
        depth_out = self._encode_depth(depth)
        return torch.cat([proprio, est_out, depth_out], dim=-1)

    def _encode_critic(self, obs: dict) -> torch.Tensor:
        proprio = obs["proprio"]
        depth = obs["depth"]
        priv = obs["critic_priv"]
        depth_out = self._encode_depth(depth)
        return torch.cat([proprio, priv, depth_out], dim=-1)

    def _clamped_action_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = torch.nan_to_num(
            std,
            nan=self.min_noise_std,
            posinf=self.max_noise_std,
            neginf=self.min_noise_std,
        )
        return torch.clamp(std, min=self.min_noise_std, max=self.max_noise_std)

    def update_policy_distribution(self, obs, **kwargs) -> None:
        """Refresh action distribution without sampling (for PPO update)."""
        del kwargs
        actor_obs = self.actor_obs_normalizer(self._encode_actor(obs))
        self.update_distribution(actor_obs)

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
        est_out = self.estimator(obs["proprio_history"])
        targets = obs["critic_priv"][:, : self.estimator_out_dim]
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
