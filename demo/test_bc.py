import os

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return self.proj(self.net(x).flatten(1))


class TaskDBCGRUPolicy(nn.Module):
    def __init__(self, proprio_dim: int = 9, cmd_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_encoder = ConvEncoder(in_ch=4, out_dim=128)
        self.ee_encoder = ConvEncoder(in_ch=4, out_dim=128)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + 128 + 64, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cmd_dim),
        )

    def forward_step(
        self,
        head_img: torch.Tensor,   # [B,4,H,W]
        ee_img: torch.Tensor,     # [B,4,H,W]
        proprio: torch.Tensor,    # [B,9]
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_feat = self.head_encoder(head_img)
        e_feat = self.ee_encoder(ee_img)
        p_feat = self.proprio_mlp(proprio)
        fused = self.fuse(torch.cat([h_feat, e_feat, p_feat], dim=-1)).unsqueeze(1)  # [B,1,H]
        out, hidden = self.gru(fused, hidden)
        cmd = self.head(out[:, -1, :])  # [B,3], normalized
        return cmd, hidden


class AlgSolution:
    """Task D eval policy: persistent- GRU BC command + JIT locomotion policy."""

    def __init__(self):
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bc_device = self.device

        self.ll_policy = torch.jit.load(os.path.join(demo_dir, "policy.pt"), map_location=self.device)
        self.ll_policy.eval()

        bc_ckpt_path = os.path.join(demo_dir, "policy_bc_state.pt")
        if not os.path.exists(bc_ckpt_path):
            bc_ckpt_path = os.path.join(os.path.dirname(demo_dir), "logs", "bc", "taskd_run1", "best.pt")
        ckpt = torch.load(bc_ckpt_path, map_location=self.bc_device)
        model_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        stats = ckpt.get("stats", None)
        if stats is None:
            raise RuntimeError("BC checkpoint missing 'stats'.")

        self.bc_policy = TaskDBCGRUPolicy().to(self.bc_device)
        self.bc_policy.load_state_dict(model_sd, strict=False)
        self.bc_policy.eval()

        self.proprio_mean = torch.tensor(stats["proprio_mean"], device=self.bc_device, dtype=torch.float32)
        self.proprio_std = torch.tensor(stats["proprio_std"], device=self.bc_device, dtype=torch.float32)
        self.cmd_mean = torch.tensor(stats["cmd_mean"], device=self.bc_device, dtype=torch.float32)
        self.cmd_std = torch.tensor(stats["cmd_std"], device=self.bc_device, dtype=torch.float32)

        self.image_size = 96
        self.depth_max = 5.0
        self.sim_dt = 0.02
        self.bc_hz = 10.0
        self.bc_hold_steps = max(1, int(round(1.0 / (self.bc_hz * self.sim_dt))))

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.train_to_env_action_scale = torch.tensor([0.25, 0.5, 0.5] * 4, device=self.device).view(1, -1)
        self.env_to_train_action_scale = torch.tensor([4.0, 2.0, 2.0] * 4, device=self.device).view(1, -1)
        self.arm_default_action = torch.zeros((1, self.arm_action_dim), device=self.device, dtype=torch.float32)

        self._bc_hidden = None
        self._bc_cached_cmd = None
        self._bc_step_counter = 0
        self._last_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

    def set_device(self, device: str) -> None:
        # Keep BC and LL on same runtime device for hidden-state consistency.
        self.device = device
        self.bc_device = device
        self.ll_policy = self.ll_policy.to(device)
        self.bc_policy = self.bc_policy.to(device)
        self.proprio_mean = self.proprio_mean.to(device)
        self.proprio_std = self.proprio_std.to(device)
        self.cmd_mean = self.cmd_mean.to(device)
        self.cmd_std = self.cmd_std.to(device)
        self.train_to_env_action_scale = self.train_to_env_action_scale.to(device)
        self.env_to_train_action_scale = self.env_to_train_action_scale.to(device)
        self.arm_default_action = self.arm_default_action.to(device)
        self._bc_hidden = None
        self._bc_cached_cmd = None

    def bind_env(self, env) -> None:
        del env
        return

    def reset(self, **kwargs):
        del kwargs
        self._bc_hidden = None
        self._bc_cached_cmd = None
        self._bc_step_counter = 0
        self._last_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

    def _prep_rgb(self, x: torch.Tensor) -> torch.Tensor:
        t = x.to(self.bc_device)
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] in (3, 4):
            t = t[:3]
        elif t.ndim == 3 and t.shape[-1] in (3, 4):
            t = t[..., :3].permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"Unexpected RGB shape: {tuple(t.shape)}")
        t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
        return torch.clamp(t, 0.0, 1.0)

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        src = x.to(self.bc_device)
        src_is_int = not src.dtype.is_floating_point
        t = src
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        elif t.ndim == 3 and t.shape[-1] == 1:
            t = t[..., 0]
        elif t.ndim != 2:
            raise ValueError(f"Unexpected depth shape: {tuple(t.shape)}")
        t = t.float()
        t = torch.nan_to_num(t, nan=self.depth_max, posinf=self.depth_max, neginf=0.0)
        if src_is_int:
            t = torch.clamp(t / 255.0, 0.0, 1.0)
        elif t.max() > 1.5:
            t = torch.clamp(t, 0.05, self.depth_max)
            t = torch.log1p(t) / torch.log1p(torch.tensor(self.depth_max, device=self.bc_device))
        else:
            t = torch.clamp(t, 0.0, 1.0)
        return t.unsqueeze(0)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        ).squeeze(0)

    def _extract_bc_inputs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio = obs["proprio"].to(self.bc_device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        base_lin_vel = proprio[:, 0:3]
        base_ang_vel = proprio[:, 3:6]
        projected_gravity = proprio[:, 9:12]
        bc_prop = torch.cat([base_lin_vel, base_ang_vel, projected_gravity], dim=-1)

        image_obs = obs.get("image", {})
        if not isinstance(image_obs, dict):
            raise RuntimeError("obs['image'] is missing or not a dict; head/ee cameras are required.")

        required = ("head_rgb", "head_depth", "ee_rgb", "ee_depth")
        missing = [k for k in required if image_obs.get(k) is None]
        if missing:
            raise RuntimeError(
                f"Missing required camera streams: {missing}. "
                "This policy requires head_rgb/head_depth/ee_rgb/ee_depth."
            )

        head_rgb = image_obs["head_rgb"]
        head_depth = image_obs["head_depth"]
        ee_rgb = image_obs["ee_rgb"]
        ee_depth = image_obs["ee_depth"]

        head = torch.cat([self._resize(self._prep_rgb(head_rgb)), self._resize(self._prep_depth(head_depth))], dim=0)
        ee = torch.cat([self._resize(self._prep_rgb(ee_rgb)), self._resize(self._prep_depth(ee_depth))], dim=0)
        return head.unsqueeze(0), ee.unsqueeze(0), bc_prop

    def _predict_velocity_command(self, obs: dict) -> torch.Tensor:
        refresh = self._bc_cached_cmd is None or (self._bc_step_counter % self.bc_hold_steps == 0)
        if refresh:
            head, ee, bc_prop = self._extract_bc_inputs(obs)
            prop_n = (bc_prop - self.proprio_mean) / self.proprio_std
            with torch.inference_mode():
                cmd_norm, self._bc_hidden = self.bc_policy.forward_step(head, ee, prop_n, self._bc_hidden)
            self._bc_hidden = self._bc_hidden.detach()
            self._bc_cached_cmd = cmd_norm * self.cmd_std + self.cmd_mean

        self._bc_step_counter += 1
        cmd = self._bc_cached_cmd.to(self.device)
        self._last_cmd = cmd.detach().clone()
        return cmd

    def _extract_ll_obs(self, obs: dict, action_dim: int, velocity_commands: torch.Tensor) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        idx = 0
        idx += 3
        base_ang_vel = proprio[:, idx : idx + 3]
        idx += 3
        idx += 3
        projected_gravity = proprio[:, idx : idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx : idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]
        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)

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

        velocity_commands = self._predict_velocity_command(obs)
        ll_obs = self._extract_ll_obs(obs, action_dim, velocity_commands)
        with torch.inference_mode():
            action_train = self.ll_policy(ll_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train.to(self.device, dtype=torch.float32), action_dim)
        return {"action": action_env.detach().cpu().numpy().tolist(), "giveup": False}

    def get_video_overlay_lines(self) -> list[str]:
        cmd = self._last_cmd[0] if self._last_cmd is not None else torch.zeros(3, device=self.device)
        h_norm = 0.0
        if self._bc_hidden is not None:
            h_norm = float(self._bc_hidden.norm().item())
        return [
            f"hl_cmd=({float(cmd[0]):+.2f},{float(cmd[1]):+.2f},{float(cmd[2]):+.2f})",
            f"bc_hidden_norm={h_norm:.2f}",
            f"bc_hold={self.bc_hold_steps}",
        ]
