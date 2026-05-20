"""Custom ActorCritic with a CNN front-end for camera-based navigation.

Architecture:
  Camera image [B, C, H, W]  ──► NatureCNN ──► features [B, 256]  ─┐
                                                                      ├─► MLP ──► actions / value
  Proprioception [B, 9]       ──────────────────────────────────────┘

Usage in NavPPORunnerCfg:
    policy = CameraActorCriticCfg(img_flat_dim=3*64*64, img_hw=64, ...)

The class is injected into rsl_rl.runners.on_policy_runner's namespace at
training time so that OnPolicyRunner can find it via eval("ActorCriticWithCNN").
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


# ──────────────────────────────────────────────────────────────────────────────
# CNN encoder
# ──────────────────────────────────────────────────────────────────────────────

class NatureCNN(nn.Module):
    """Nature-DQN style CNN adapted for 64×64 RGB input.

    Architecture (input 3×64×64):
      Conv 8×8 stride 4  → [32, 15, 15]
      Conv 4×4 stride 2  → [64,  6,  6]
      Conv 3×3 stride 1  → [64,  4,  4]
      Flatten            → 1024
      Linear             → feature_dim (256)
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(),
            nn.Flatten(),
        )
        # 64 × 4 × 4 = 1024  (for 64×64 input)
        self.fc = nn.Sequential(nn.Linear(1024, feature_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] float32, pixel values in [0, 255]."""
        return self.fc(self.convnet(x / 255.0))


# ──────────────────────────────────────────────────────────────────────────────
# ActorCritic with CNN
# ──────────────────────────────────────────────────────────────────────────────

def _get_activation(name: str) -> nn.Module:
    mapping = {
        "elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(), "selu": nn.SELU(),
    }
    act = mapping.get(name.lower())
    if act is None:
        raise ValueError(f"Unknown activation: {name}")
    return act


class ActorCriticWithCNN(nn.Module):
    """Drop-in replacement for rsl_rl.modules.ActorCritic that encodes images.

    The observation tensor layout expected in obs["policy"] is:
        [img_flat_dim dims | proprio_dim dims]
        where img_flat_dim = C * H * W  (e.g. 3 * 64 * 64 = 12288)

    All extra kwargs are accepted and ignored for compatibility with
    rsl_rl's ActorCritic instantiation signature.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        *,
        img_flat_dim: int = 12288,       # C * H * W (3 * 64 * 64)
        img_hw: int = 64,
        img_channels: int = 3,
        cnn_feature_dim: int = 256,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(f"[ActorCriticWithCNN] Ignoring extra kwargs: {list(kwargs)}")
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]

        self.obs_groups = obs_groups
        self.img_flat_dim = img_flat_dim
        self.img_shape = (img_channels, img_hw, img_hw)

        # Compute raw obs dim and derived dims
        raw_dim = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        proprio_dim = raw_dim - img_flat_dim
        encoded_dim = cnn_feature_dim + proprio_dim

        # CNN (shared between actor and critic)
        self.cnn = NatureCNN(img_channels, feature_dim=cnn_feature_dim)

        # Actor
        self.actor = MLP(encoded_dim, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalizer = (
            EmpiricalNormalization(encoded_dim) if actor_obs_normalization else nn.Identity()
        )
        print(f"[ActorCriticWithCNN] Actor: CNN({img_channels}×{img_hw}×{img_hw}→{cnn_feature_dim}) "
              f"+ proprio({proprio_dim}) → MLP{actor_hidden_dims} → {num_actions}")

        # Critic (same shape, same CNN)
        raw_critic_dim = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        critic_proprio_dim = raw_critic_dim - img_flat_dim
        critic_encoded_dim = cnn_feature_dim + critic_proprio_dim
        self.critic = MLP(critic_encoded_dim, 1, critic_hidden_dims, activation)
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_encoded_dim) if critic_obs_normalization else nn.Identity()
        )

        # Action noise
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """Split flat obs → [CNN(image), proprio], return encoded [B, cnn+proprio]."""
        img = flat_obs[:, : self.img_flat_dim].view(-1, *self.img_shape)
        proprio = flat_obs[:, self.img_flat_dim :]
        return torch.cat([self.cnn(img), proprio], dim=-1)

    def _get_flat_obs(self, obs: dict, groups: list) -> torch.Tensor:
        return torch.cat([obs[g] for g in groups], dim=-1)

    # ── rsl_rl interface ──────────────────────────────────────────────────────

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

    def update_distribution(self, encoded_obs: torch.Tensor):
        mean = self.actor(encoded_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: dict, **kwargs) -> torch.Tensor:
        encoded = self.actor_obs_normalizer(
            self._encode(self._get_flat_obs(obs, self.obs_groups["policy"]))
        )
        self.update_distribution(encoded)
        return self.distribution.sample()

    def act_inference(self, obs: dict) -> torch.Tensor:
        encoded = self.actor_obs_normalizer(
            self._encode(self._get_flat_obs(obs, self.obs_groups["policy"]))
        )
        return self.actor(encoded)

    def evaluate(self, obs: dict, **kwargs) -> torch.Tensor:
        encoded = self.critic_obs_normalizer(
            self._encode(self._get_flat_obs(obs, self.obs_groups["critic"]))
        )
        return self.critic(encoded)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actor_obs(self, obs: dict) -> torch.Tensor:
        return self._encode(self._get_flat_obs(obs, self.obs_groups["policy"]))

    def get_critic_obs(self, obs: dict) -> torch.Tensor:
        return self._encode(self._get_flat_obs(obs, self.obs_groups["critic"]))

    def update_normalization(self, obs: dict):
        if hasattr(self.actor_obs_normalizer, "update"):
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if hasattr(self.critic_obs_normalizer, "update"):
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
