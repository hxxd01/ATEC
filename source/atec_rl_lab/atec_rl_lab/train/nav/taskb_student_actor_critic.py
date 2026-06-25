"""Task-B student actor-critic: recurrent CNN actor + feedforward critic [proprio, priv, scored_mask]."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP, Memory
from rsl_rl.utils import unpad_trajectories

from .taskd_student_actor_critic import ConvEncoder

TASK_B_PROPRIO_DIM = 9
TASK_B_CRITIC_PRIV_DIM = 60  # robot env-local pose(4) + ee env-local xy(2) + trash xyz(54)
TASK_B_SCORED_MASK_DIM = 18


class TaskBStudentActorCritic(nn.Module):
    """Recurrent image actor; critic is a feedforward MLP on privileged state only."""

    is_recurrent = True

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        *,
        img_h: int = 24,
        img_w: int = 32,
        img_hw: int | None = None,
        img_channels: int = 4,
        proprio_dim: int = TASK_B_PROPRIO_DIM,
        priv_dim: int = TASK_B_CRITIC_PRIV_DIM,
        scored_mask_dim: int = TASK_B_SCORED_MASK_DIM,
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
        nav_action_dim: int = 3,
        **kwargs,
    ):
        if kwargs:
            print(f"[TaskBStudentActorCritic] Ignoring extra kwargs: {list(kwargs)}")
        super().__init__()

        if img_hw is not None:
            img_h = img_w = int(img_hw)

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128]

        self.obs_groups = obs_groups
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.img_channels = int(img_channels)
        self.head_flat = self.img_channels * self.img_h * self.img_w
        self.ee_flat = self.img_channels * self.img_h * self.img_w
        self.proprio_dim = int(proprio_dim)
        self.priv_dim = int(priv_dim)
        self.scored_mask_dim = int(scored_mask_dim)
        self.nav_action_dim = int(nav_action_dim)
        self.num_actions = int(num_actions)
        if self.num_actions != self.nav_action_dim:
            raise ValueError(
                f"TaskBStudentActorCritic expects nav_action_dim={self.nav_action_dim}, got {self.num_actions}"
            )

        self._validate_obs(obs)

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
        self.memory_a = Memory(
            fuse_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )
        self.nav_actor = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.nav_action_dim),
        )
        self.actor = self.nav_actor

        critic_in_dim = self.proprio_dim + self.priv_dim + self.scored_mask_dim
        self.critic = MLP(critic_in_dim, 1, critic_hidden_dims, "elu")
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
        print(
            f"[TaskBStudentActorCritic] actor=RNN+CNN fuse_dim={fuse_dim}, "
            f"critic=MLP in={critic_in_dim} ([proprio={self.proprio_dim}, "
            f"priv={self.priv_dim}, scored_mask={self.scored_mask_dim}])",
            flush=True,
        )

    def _validate_obs(self, obs: dict) -> None:
        expected = {
            "proprio": self.proprio_dim,
            "priv": self.priv_dim,
            "scored_mask": self.scored_mask_dim,
        }
        for key, dim in expected.items():
            if key not in obs:
                raise KeyError(f"TaskBStudentActorCritic requires observation group '{key}'")
            if obs[key].shape[-1] != dim:
                raise ValueError(
                    f"Observation '{key}' dim {obs[key].shape[-1]} != expected {dim}."
                )

    def reset(self, dones=None):
        self.memory_a.reset(dones)

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

    def _encode_actor(self, flat_obs: torch.Tensor) -> torch.Tensor:
        lead_shape = flat_obs.shape[:-1]
        x = flat_obs.reshape(-1, flat_obs.shape[-1])
        head = x[:, : self.head_flat].view(-1, self.img_channels, self.img_h, self.img_w)
        ee = x[:, self.head_flat : self.head_flat + self.ee_flat].view(
            -1, self.img_channels, self.img_h, self.img_w
        )
        proprio = x[:, self.head_flat + self.ee_flat : self.head_flat + self.ee_flat + self.proprio_dim]
        h_feat = self.head_encoder(head)
        e_feat = self.ee_encoder(ee)
        p_feat = self.proprio_mlp(proprio)
        out = self.fuse(torch.cat([h_feat, e_feat, p_feat], dim=-1))
        return out.view(*lead_shape, -1)

    def _encode_critic(self, obs: dict) -> torch.Tensor:
        return self._get_flat_obs(obs, self.obs_groups["critic"])

    def update_distribution(self, encoded_obs: torch.Tensor):
        mean = self.nav_actor(encoded_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: dict, masks=None, hidden_states=None) -> torch.Tensor:
        encoded = self._encode_actor(self._get_flat_obs(obs, self.obs_groups["policy"]))
        encoded = self.actor_obs_normalizer(encoded)
        out_mem = self.memory_a(encoded, masks, hidden_states).squeeze(0)
        self.update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: dict) -> torch.Tensor:
        encoded = self._encode_actor(self._get_flat_obs(obs, self.obs_groups["policy"]))
        encoded = self.actor_obs_normalizer(encoded)
        out_mem = self.memory_a(encoded).squeeze(0)
        return self.nav_actor(out_mem)

    def evaluate(self, obs: dict, masks=None, hidden_states=None) -> torch.Tensor:
        del hidden_states
        critic_obs = self.critic_obs_normalizer(self._encode_critic(obs))
        if masks is not None:
            critic_obs = unpad_trajectories(critic_obs, masks)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actor_obs(self, obs: dict) -> torch.Tensor:
        return self._encode_actor(self._get_flat_obs(obs, self.obs_groups["policy"]))

    def get_critic_obs(self, obs: dict) -> torch.Tensor:
        return self._encode_critic(obs)

    def update_normalization(self, obs: dict):
        if hasattr(self.actor_obs_normalizer, "update"):
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if hasattr(self.critic_obs_normalizer, "update"):
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def get_hidden_states(self):
        # rsl_rl storage expects two RNN state tuples; critic is feedforward and ignores the second.
        return self.memory_a.hidden_states, self.memory_a.hidden_states

    def detach_hidden_states(self, dones=None):
        self.memory_a.detach_hidden_states(dones)
