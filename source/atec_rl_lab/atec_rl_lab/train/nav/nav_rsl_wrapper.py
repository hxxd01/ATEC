"""RSL-RL VecEnv wrapper for hierarchical nav training.

RslRlVecEnvWrapper assumes the wrapped env is (almost) a raw Isaac Lab env.
HierarchicalNavEnv exposes 3-dim actions and 45-dim 'policy' observations, so we
override the relevant methods here instead of using the stock wrapper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .hierarchical_env import HierarchicalNavEnv


class NavRslRlVecEnvWrapper(VecEnv):
    """VecEnv bridge between HierarchicalNavEnv and rsl_rl OnPolicyRunner."""

    def __init__(self, env: HierarchicalNavEnv, clip_actions: float | None = 1.0):
        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = env.num_envs
        self.device = env.device
        self.num_actions = int(gym.spaces.flatdim(env.action_space))
        self.max_episode_length = env.max_episode_length

        if clip_actions is not None:
            self.env.action_space = gym.spaces.Box(
                low=-clip_actions, high=clip_actions,
                shape=(self.num_actions,), dtype=np.float32,
            )

        # rsl_rl runner does not call reset before first rollout
        self.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        # rsl_rl may randomize initial episode lengths in base env steps
        self.unwrapped.episode_length_buf.copy_(value * self.env.inner_steps)

    def get_observations(self) -> TensorDict:
        obs_dict, _ = self.env.get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, info = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), info

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        if not self.unwrapped.cfg.is_finite_horizon:
            extras = dict(extras) if extras is not None else {}
            extras["time_outs"] = truncated

        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):
        return self.env.close()
