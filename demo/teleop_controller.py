"""Task D teleop replay controller: external (vx, vy, wz) + JIT locomotion policy.pt."""

from __future__ import annotations

import os

import torch


class TaskDTeleopController:
    """Low-level Task D controller driven by explicit velocity commands (teleop replay)."""

    def __init__(
        self,
        *,
        policy_path: str | None = None,
        device: str | None = None,
        vx_min: float = -4.0,
        vx_max: float = 4.0,
        vy_max: float = 2.0,
        wz_max: float = 1.0,
    ) -> None:
        self.device = device or "cuda"
        self.vx_min = float(vx_min)
        self.vx_max = float(vx_max)
        self.vy_max = float(vy_max)
        self.wz_max = float(wz_max)

        resolved = self._resolve_policy_path(policy_path)
        self.policy = torch.jit.load(resolved, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.train_to_env_action_scale = torch.tensor([0.25, 0.5, 0.5] * 4, device=self.device).view(1, -1)
        self.env_to_train_action_scale = torch.tensor([4.0, 2.0, 2.0] * 4, device=self.device).view(1, -1)
        self.arm_default_action = torch.zeros((1, self.arm_action_dim), device=self.device, dtype=torch.float32)
        self._vel_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
        self.reset()

    @staticmethod
    def _resolve_policy_path(policy_path: str | None) -> str:
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        if policy_path:
            candidate = os.path.abspath(policy_path)
            if os.path.isfile(candidate):
                return candidate
        default = os.path.join(demo_dir, "policy.pt")
        if os.path.isfile(default):
            return default
        raise FileNotFoundError(
            "Task D locomotion policy not found. Pass --ll_policy or place demo/policy.pt next to teleop_controller.py."
        )

    def set_device(self, device: str) -> None:
        self.device = device
        self.policy = self.policy.to(device)
        self.train_to_env_action_scale = self.train_to_env_action_scale.to(device)
        self.env_to_train_action_scale = self.env_to_train_action_scale.to(device)
        self.arm_default_action = self.arm_default_action.to(device)
        self.reset()

    def reset(self, **kwargs) -> None:
        del kwargs
        self._vel_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

    def set_velocity_command(self, vx: float, vy: float, wz: float) -> None:
        self._vel_cmd = torch.tensor([[vx, vy, wz]], device=self.device, dtype=torch.float32)

    def _velocity_commands(self, batch: int) -> torch.Tensor:
        if self._vel_cmd.shape[0] == batch:
            return self._vel_cmd
        return self._vel_cmd.expand(batch, -1)

    def _extract_policy_obs(self, obs: dict, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        idx = 3
        base_ang_vel = proprio[:, idx : idx + 3]
        idx += 6
        projected_gravity = proprio[:, idx : idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx : idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_train_leg = actions_all[:, self.leg_joint_indices] * self.env_to_train_action_scale.to(
            dtype=proprio.dtype
        )
        velocity_commands = self._velocity_commands(proprio.shape[0])

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def predicts(self, obs, current_score):
        del current_score
        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        policy_obs = self._extract_policy_obs(obs, action_dim)
        with torch.inference_mode():
            action_train = self.policy(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        return {
            "action": action_env.detach().cpu().numpy().tolist(),
            "giveup": False,
        }
