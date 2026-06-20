"""Task D hierarchical nav deploy: student PPO (depth+proprio) + JIT policy.pt legs."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from depth_preprocess import prep_depth

TASK_D_PIT_REF_OX = -4.2


class EmpiricalNormalization(nn.Module):
    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class Memory(nn.Module):
    def __init__(self, input_size, type="gru", num_layers=1, hidden_size=256):
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        if masks is not None:
            if hidden_states is None:
                raise ValueError("Hidden states required in batch mode")
            out, _ = self.rnn(input, hidden_states)
            return out
        out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None, hidden_states=None):
        if dones is None:
            self.hidden_states = hidden_states
        elif self.hidden_states is not None:
            if isinstance(self.hidden_states, tuple):
                for hs in self.hidden_states:
                    hs[..., dones == 1, :] = 0.0
            else:
                self.hidden_states[..., dones == 1, :] = 0.0


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str = "elu"):
        super().__init__()
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), act]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)


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


class TaskDStudentActorCritic(nn.Module):
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
        **_kwargs,
    ):
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128]
        if img_hw is not None:
            img_h = img_w = int(img_hw)

        self.obs_groups = obs_groups
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.img_channels = int(img_channels)
        self.head_flat = self.img_channels * self.img_h * self.img_w
        self.ee_flat = self.img_channels * self.img_h * self.img_w
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

        self.memory_a = Memory(fuse_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        critic_in = fuse_dim + (64 if self._critic_priv_dim > 0 else 0)
        self.memory_c = Memory(critic_in, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.nav_action_dim = 3
        self.leg_action_dim = 12
        self.nav_actor = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.nav_action_dim),
        )
        self.pit_actor = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.leg_action_dim),
        )
        self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, "elu")
        self.actor_obs_normalizer = (
            EmpiricalNormalization(fuse_dim) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_in) if critic_obs_normalization else nn.Identity()
        )
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.nav_action_dim))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.nav_action_dim)))

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

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
        out = self.fuse(torch.cat([self.head_encoder(head), self.ee_encoder(ee), self.proprio_mlp(proprio)], dim=-1))
        return out.view(*lead_shape, -1)

    def act_inference(self, obs: dict) -> torch.Tensor:
        encoded = self._encode_actor(self._get_flat_obs(obs, self.obs_groups["policy"]))
        encoded = self.actor_obs_normalizer(encoded)
        out_mem = self.memory_a(encoded).squeeze(0)
        return self.nav_actor(out_mem)


def _remap_nav_student_state(state: dict) -> dict:
    """Map train/BC keys to deploy module names (legacy actor.* -> nav_actor.*)."""
    out: dict = {}
    for key, val in state.items():
        if key.startswith("nav_actor."):
            out[key] = val
        elif key.startswith("actor.") and not key.startswith("actor_obs"):
            out["nav_actor." + key[len("actor.") :]] = val
        else:
            out[key] = val
    return out


def _load_nav_student_weights(model: nn.Module, state: dict) -> None:
    state = _remap_nav_student_state(state)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    if not any(k.startswith("nav_actor.") for k in filtered):
        raise RuntimeError(
            "nav checkpoint has no nav_actor weights (expected train nav student or BC actor.* keys)"
        )
    for prefix in ("head_encoder.", "ee_encoder.", "memory_a.", "nav_actor."):
        if not any(k.startswith(prefix) for k in filtered):
            raise RuntimeError(f"nav checkpoint missing required weights: {prefix}*")
    missing, _unexpected = model.load_state_dict(filtered, strict=False)
    optional = ("pit_actor.", "critic.", "memory_c.", "std", "log_std", "actor.")
    hard_missing = [k for k in missing if not k.startswith(optional)]
    if hard_missing:
        raise RuntimeError(f"nav checkpoint missing weights: {hard_missing[:8]}")


def _load_deploy_cfg(demo_dir: str) -> dict:
    agent_yaml = os.path.join(demo_dir, "agent.yaml")
    if not os.path.isfile(agent_yaml):
        return {}
    try:
        import yaml

        with open(agent_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("policy", {})
    except Exception:
        return {}


def box_nominal_x_from_env(env, env_idx: int = 0) -> float:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    box = base.scene["box"]
    origins = base.scene.env_origins
    bx = float(box.data.root_pos_w[env_idx, 0].item())
    ox = float(origins[env_idx, 0].item())
    return bx - ox + TASK_D_PIT_REF_OX


def box_in_score_zone(box_nominal_x: float) -> bool:
    return (-0.7 <= box_nominal_x <= 0.7) or (-1.4 <= box_nominal_x <= -0.7)


def robot_horizontal_speed_from_env(env, env_idx: int = 0) -> float:
    """World-frame horizontal root speed (m/s)."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = base.scene["robot"]
    wv = robot.data.root_lin_vel_w[env_idx, :2]
    return float(torch.linalg.vector_norm(wv).item())


class TaskDHierarchicalNavDeploy:
    """High-level nav student + JIT legs (same stack as demo/solution copy 2.py)."""

    NUM_STAGES = 4
    CRITIC_EXTRA_DIM = 17 + NUM_STAGES

    def __init__(
        self,
        *,
        demo_dir: str,
        student_ckpt_path: str,
        ll_policy_path: str | None = None,
        device: str | None = None,
    ):
        self.demo_dir = demo_dir
        self.student_ckpt_path = os.path.abspath(student_ckpt_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy_cfg = _load_deploy_cfg(demo_dir)

        legacy_hw = policy_cfg.get("img_hw")
        if legacy_hw is not None and "img_h" not in policy_cfg:
            self.image_h = int(legacy_hw)
            self.image_w = int(legacy_hw)
        else:
            self.image_h = int(policy_cfg.get("img_h", 24))
            self.image_w = int(policy_cfg.get("img_w", 32))
        self.img_channels = int(policy_cfg.get("img_channels", 1))
        self.depth_only = self.img_channels == 1
        self.depth_max = float(policy_cfg.get("depth_max", os.environ.get("NAV_DEPTH_MAX", "5.0")))

        # Match train_nav_taskd_student.py defaults (was -2..+2, under-scaled forward push).
        self.vx_min = float(os.environ.get("NAV_VX_MIN", policy_cfg.get("vx_min", -4.0)))
        self.vx_max = float(os.environ.get("NAV_VX_MAX", policy_cfg.get("vx_max", 4.0)))
        self.vy_max = float(os.environ.get("NAV_VY_MAX", policy_cfg.get("vy_max", 1.2)))
        self.wz_max = float(os.environ.get("NAV_WZ_MAX", policy_cfg.get("wz_max", 0.6)))
        inner_steps = int(os.environ.get("NAV_INNER_STEPS", "5"))
        self.nav_hold_steps = max(1, inner_steps)

        actor_dim = 2 * self.img_channels * self.image_h * self.image_w + 9
        critic_dim = actor_dim + self.CRITIC_EXTRA_DIM
        obs = {"policy": torch.zeros(1, actor_dim), "critic": torch.zeros(1, critic_dim)}
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        ac_kwargs = {
            "img_h": self.image_h,
            "img_w": self.image_w,
            "img_channels": self.img_channels,
            "proprio_dim": int(policy_cfg.get("proprio_dim", 9)),
            "enc_dim": int(policy_cfg.get("enc_dim", 128)),
            "fuse_dim": int(policy_cfg.get("fuse_dim", 256)),
            "rnn_type": policy_cfg.get("rnn_type", "gru"),
            "rnn_hidden_dim": int(policy_cfg.get("rnn_hidden_dim", 256)),
            "rnn_num_layers": int(policy_cfg.get("rnn_num_layers", 1)),
            "actor_obs_normalization": bool(policy_cfg.get("actor_obs_normalization", True)),
            "critic_obs_normalization": bool(policy_cfg.get("critic_obs_normalization", True)),
            "actor_hidden_dims": policy_cfg.get("actor_hidden_dims", [256]),
            "critic_hidden_dims": policy_cfg.get("critic_hidden_dims", [256, 128]),
            "init_noise_std": float(policy_cfg.get("init_noise_std", 0.6)),
            "noise_std_type": policy_cfg.get("noise_std_type", "scalar"),
        }
        self.nav_policy = TaskDStudentActorCritic(obs, obs_groups, num_actions=3, **ac_kwargs).to(self.device)
        loaded = torch.load(self.student_ckpt_path, map_location=self.device, weights_only=False)
        state = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        _load_nav_student_weights(self.nav_policy, state)
        self.nav_policy.eval()

        ll_path = ll_policy_path or os.path.join(demo_dir, "policy.pt")
        self.ll_policy = torch.jit.load(ll_path, map_location=self.device)
        self.ll_policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.train_to_env_action_scale = torch.tensor([0.25, 0.5, 0.5] * 4, device=self.device).view(1, -1)
        self.env_to_train_action_scale = torch.tensor([4.0, 2.0, 2.0] * 4, device=self.device).view(1, -1)
        self.arm_default_action = torch.zeros((1, self.arm_action_dim), device=self.device, dtype=torch.float32)
        self._nav_step_counter = 0
        self._cached_vel_cmd: torch.Tensor | None = None
        self._last_vel_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

        print(
            f"[TaskDHierarchicalNavDeploy] ckpt={os.path.basename(self.student_ckpt_path)} "
            f"depth={self.image_h}x{self.image_w} hold={self.nav_hold_steps} steps "
            f"vx=[{self.vx_min:.1f},{self.vx_max:.1f}] vy=±{self.vy_max:.1f} wz=±{self.wz_max:.1f}",
            flush=True,
        )

    def set_device(self, device: str) -> None:
        self.device = device
        self.nav_policy = self.nav_policy.to(device)
        self.ll_policy = self.ll_policy.to(device)
        self.train_to_env_action_scale = self.train_to_env_action_scale.to(device)
        self.env_to_train_action_scale = self.env_to_train_action_scale.to(device)
        self.arm_default_action = self.arm_default_action.to(device)
        self.reset()

    def reset(self, **kwargs) -> None:
        del kwargs
        self.nav_policy.reset()
        self._nav_step_counter = 0
        self._cached_vel_cmd = None
        self._last_vel_cmd = torch.zeros((1, 3), device=self.device, dtype=torch.float32)

    def _prep_depth(self, x: torch.Tensor) -> torch.Tensor:
        return prep_depth(
            x.to(self.device),
            image_h=self.image_h,
            image_w=self.image_w,
            depth_max=self.depth_max,
        )

    def _build_actor_obs(self, obs: dict) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio_feat = torch.cat([proprio[:, 0:3], proprio[:, 3:6], proprio[:, 9:12]], dim=-1)
        batch = proprio.shape[0]
        per_cam = self.img_channels * self.image_h * self.image_w

        depth_flat = obs.get("depth")
        if depth_flat is not None:
            flat = depth_flat.to(self.device, dtype=torch.float32)
            if flat.ndim == 1:
                flat = flat.unsqueeze(0)
            if flat.shape[-1] == 2 * per_cam:
                head = flat[:, :per_cam]
                ee = flat[:, per_cam:]
                return torch.cat([head, ee, proprio_feat], dim=-1)
            if flat.shape[-1] == per_cam:
                raise RuntimeError(
                    "nav student needs head+ee depth (1536D = 2×24×32), but obs['depth'] is head-only "
                    f"({flat.shape[-1]}D). For nav→pit play, env must use head+ee cameras "
                    "(check log: obs_depth=head+ee, not head)."
                )
            raise RuntimeError(
                f"unexpected obs['depth'] dim {flat.shape[-1]}; expected {2 * per_cam} (head+ee @ "
                f"{self.image_h}x{self.image_w})"
            )

        image_obs = obs.get("image", {})
        if not isinstance(image_obs, dict):
            raise RuntimeError(
                "nav student needs obs['image'] head/ee depth or obs_manager obs['depth'] "
                "(480x640 platform → prep_depth → policy size)."
            )
        head_depth = image_obs.get("head_depth")
        if head_depth is None:
            head_depth = image_obs.get("video_depth")
        ee_depth = image_obs.get("ee_depth")
        if head_depth is None or ee_depth is None:
            raise RuntimeError("depth_only nav student requires head_depth and ee_depth in obs['image'].")
        head = self._prep_depth(head_depth).reshape(batch, -1)
        ee = self._prep_depth(ee_depth).reshape(batch, -1)
        return torch.cat([head, ee, proprio_feat], dim=-1)

    def _nav_action_to_vel_cmd(self, nav_action: torch.Tensor) -> torch.Tensor:
        a = nav_action.clamp(-1.0, 1.0)
        vx = (a[:, 0] + 1.0) * 0.5 * (self.vx_max - self.vx_min) + self.vx_min
        return torch.stack([vx, a[:, 1] * self.vy_max, a[:, 2] * self.wz_max], dim=-1)

    def _predict_velocity_command(self, obs: dict) -> torch.Tensor:
        refresh = self._cached_vel_cmd is None or (self._nav_step_counter % self.nav_hold_steps == 0)
        if refresh:
            actor_obs = self._build_actor_obs(obs)
            with torch.inference_mode():
                nav_action = self.nav_policy.act_inference({"policy": actor_obs})
            self._cached_vel_cmd = self._nav_action_to_vel_cmd(nav_action)
        self._nav_step_counter += 1
        cmd = self._cached_vel_cmd.to(self.device)
        self._last_vel_cmd = cmd.detach().clone()
        return cmd

    def _extract_ll_obs(self, obs: dict, action_dim: int, velocity_commands: torch.Tensor) -> torch.Tensor:
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
        num_envs = action_train.shape[0]
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = action_train * self.train_to_env_action_scale
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

    def overlay_lines(self) -> list[str]:
        cmd = self._last_vel_cmd[0]
        return [
            f"nav_ckpt={os.path.basename(self.student_ckpt_path)}",
            f"hl_cmd=({float(cmd[0]):+.2f},{float(cmd[1]):+.2f},{float(cmd[2]):+.2f})",
            f"img={self.img_channels}ch@{self.image_h}x{self.image_w}",
            f"nav_hold={self.nav_hold_steps} steps",
        ]
