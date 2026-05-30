"""Task-D recurrent student ActorCritic for direct BC->PPO fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP, Memory


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x).flatten(1)
        return self.proj(y)


class TaskDStudentActorCritic(nn.Module):
    """Recurrent actor-critic that matches BC encoder+GRU structure."""

    is_recurrent = True

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        *,
        img_hw: int = 64,
        img_channels: int = 4,
        proprio_dim: int = 9,
        enc_dim: int = 128,
        fuse_dim: int = 256,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dims: list | None = None,
        critic_hidden_dims: list | None = None,
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(f"[TaskDStudentActorCritic] Ignoring extra kwargs: {list(kwargs)}")
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128]

        self.obs_groups = obs_groups
        self.img_hw = int(img_hw)
        self.img_channels = int(img_channels)
        self.head_flat = self.img_channels * self.img_hw * self.img_hw
        self.ee_flat = self.img_channels * self.img_hw * self.img_hw
        self.proprio_dim = int(proprio_dim)

        self.head_encoder = ConvEncoder(in_ch=self.img_channels, out_dim=enc_dim)
        self.ee_encoder = ConvEncoder(in_ch=self.img_channels, out_dim=enc_dim)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(self.proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(enc_dim + enc_dim + 64, fuse_dim),
            nn.ReLU(inplace=True),
        )
        self._base_obs_dim = self.head_flat + self.ee_flat + self.proprio_dim
        raw_critic_dim = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self._critic_priv_dim = max(0, int(raw_critic_dim - self._base_obs_dim))
        self.critic_priv_mlp = (
            nn.Sequential(
                nn.Linear(self._critic_priv_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
            )
            if self._critic_priv_dim > 0
            else nn.Identity()
        )

        self.memory_a = Memory(
            fuse_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_dim=rnn_hidden_dim
        )
        self.memory_c = Memory(
            fuse_dim + (64 if self._critic_priv_dim > 0 else 0),
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_dim=rnn_hidden_dim,
        )

        # Keep actor head shape BC-compatible: 256 -> 256 -> action_dim
        self.actor = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )
        critic_in_dim = fuse_dim + (64 if self._critic_priv_dim > 0 else 0)
        self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, "elu")
        self.actor_obs_normalizer = (
            EmpiricalNormalization(fuse_dim) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_in_dim) if critic_obs_normalization else nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

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

    def _get_flat_obs(self, obs: dict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[g] for g in groups], dim=-1)

    def _encode_base(self, flat_obs: torch.Tensor) -> torch.Tensor:
        # base layout: [head(C*H*W), ee(C*H*W), proprio(9)]
        lead_shape = flat_obs.shape[:-1]
        x = flat_obs.reshape(-1, flat_obs.shape[-1])
        head = x[:, : self.head_flat].view(-1, self.img_channels, self.img_hw, self.img_hw)
        ee = x[:, self.head_flat : self.head_flat + self.ee_flat].view(
            -1, self.img_channels, self.img_hw, self.img_hw
        )
        proprio = x[:, self.head_flat + self.ee_flat : self.head_flat + self.ee_flat + self.proprio_dim]
        h_feat = self.head_encoder(head)
        e_feat = self.ee_encoder(ee)
        p_feat = self.proprio_mlp(proprio)
        out = self.fuse(torch.cat([h_feat, e_feat, p_feat], dim=-1))
        return out.view(*lead_shape, -1)

    def _encode_actor(self, flat_obs: torch.Tensor) -> torch.Tensor:
        return self._encode_base(flat_obs)

    def _encode_critic(self, flat_obs: torch.Tensor) -> torch.Tensor:
        base = self._encode_base(flat_obs[..., : self._base_obs_dim])
        if self._critic_priv_dim <= 0:
            return base
        priv = flat_obs[..., self._base_obs_dim : self._base_obs_dim + self._critic_priv_dim]
        lead_shape = priv.shape[:-1]
        priv_feat = self.critic_priv_mlp(priv.reshape(-1, priv.shape[-1])).view(*lead_shape, -1)
        return torch.cat([base, priv_feat], dim=-1)

    def update_distribution(self, encoded_obs: torch.Tensor):
        mean = self.actor(encoded_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: dict, masks=None, hidden_state=None) -> torch.Tensor:
        encoded = self._encode_actor(self._get_flat_obs(obs, self.obs_groups["policy"]))
        encoded = self.actor_obs_normalizer(encoded)
        out_mem = self.memory_a(encoded, masks, hidden_state).squeeze(0)
        self.update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: dict) -> torch.Tensor:
        encoded = self._encode_actor(self._get_flat_obs(obs, self.obs_groups["policy"]))
        encoded = self.actor_obs_normalizer(encoded)
        out_mem = self.memory_a(encoded).squeeze(0)
        return self.actor(out_mem)

    def evaluate(self, obs: dict, masks=None, hidden_state=None) -> torch.Tensor:
        encoded = self._encode_critic(self._get_flat_obs(obs, self.obs_groups["critic"]))
        encoded = self.critic_obs_normalizer(encoded)
        out_mem = self.memory_c(encoded, masks, hidden_state).squeeze(0)
        return self.critic(out_mem)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actor_obs(self, obs: dict) -> torch.Tensor:
        return self._encode_actor(self._get_flat_obs(obs, self.obs_groups["policy"]))

    def get_critic_obs(self, obs: dict) -> torch.Tensor:
        return self._encode_critic(self._get_flat_obs(obs, self.obs_groups["critic"]))

    def update_normalization(self, obs: dict):
        if hasattr(self.actor_obs_normalizer, "update"):
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if hasattr(self.critic_obs_normalizer, "update"):
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def get_hidden_states(self):
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def detach_hidden_states(self, dones=None):
        self.memory_a.detach_hidden_state(dones)
        self.memory_c.detach_hidden_state(dones)

