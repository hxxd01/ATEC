"""Task D platform end-to-end stage wrapper: direct 12D leg actions + multi-stage rewards."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from atec_rl_lab.tasks.task_d.mdp.env_origin import task_d_robot_spawn_world_xy
from atec_rl_lab.train.nav.taskd_teacher_env import (
    TaskDTeacherEnv,
    _build_nominal_waypoints,
)

_LEG_DIM = 12
_TOTAL_ACTION_DIM = 20
_TRAIN_TO_ENV = torch.tensor([0.25, 0.5, 0.5] * 4, dtype=torch.float32)

# E2E task reward tuning (dense weights + no per-step delta clip).
_E2E_REWARD_NO_CLIP = float("inf")
_E2E_W_NAV_DIST = 30.0
_E2E_W_PUSH_BOX_AXIS = 30.0
_E2E_W_APPROACH_BOX = 10.0
_E2E_W_FACE_BOX = 20.0
_E2E_SPARSE_BONUS = (50.0, 60.0, 70.0, 80.0)  # retreat, align_box_y, push_box, return_home
_E2E_R_STEP_PENALTY = 0.0

# Box x ranges for 14-point scoring (nominal coords, same as env_cfg RewardsCfg).
_BOX_SCORE_X_RANGES = ((-0.7, 0.7), (-1.4, -0.7))


# Privileged critic extras (match TaskDStudentEnv):
# robot(3)+box(3)+rel_body(3)+r_vel(2)+b_vel(2)+rel_world(2)+contact(1)+stage_prog(1)+stage_onehot(N)
def e2e_critic_task_dim(num_stages: int) -> int:
    return 17 + int(num_stages)


class TaskDE2EStageEnv(gym.Wrapper):
    """Direct leg-policy wrapper with 4 scripted stages (retreat → align y → push → return)."""

    def __init__(
        self,
        env: gym.Env,
        device: str = "cuda",
        stage_reach_tol: float = 0.1,
        return_still_s: float = 1.0,
        stage_log_interval: int = 100,
        reward_mode: str = "base_only",
    ):
        super().__init__(env)
        self._device = device
        self._stage_reach_tol = float(stage_reach_tol)
        self._return_still_s = float(return_still_s)
        self._stage_log_interval = max(0, int(stage_log_interval))
        self._reward_mode = str(reward_mode).strip().lower()
        if self._reward_mode not in {"base_only", "stage_only", "base_plus_stage"}:
            raise ValueError(
                f"reward_mode={reward_mode!r} invalid; choose from "
                "'base_only' | 'stage_only' | 'base_plus_stage'"
            )
        self._step_count = 0

        # 1 retreat | 2 align box y (hold x) | 3 push box to score | 4 return + hold still
        self.stage_specs = [
            dict(name="retreat", axis="x", sign=-0.7, dist=1.0, push=False, sparse_bonus=_E2E_SPARSE_BONUS[0]),
            dict(
                name="align_box_y",
                axis="y",
                sign=0.0,
                dist=0.0,
                push=False,
                sparse_bonus=_E2E_SPARSE_BONUS[1],
                match_box_y_target=True,
            ),
            dict(
                name="push_box",
                axis="x",
                sign=1.0,
                dist=4.0,
                push=True,
                push_forward_only=True,
                push_forward_dist=4.0,
                sparse_bonus=_E2E_SPARSE_BONUS[2],
                box_score_complete=True,
            ),
            dict(
                name="return_home",
                axis="xy",
                sign=1.0,
                dist=0.0,
                push=False,
                sparse_bonus=_E2E_SPARSE_BONUS[3],
                return_spawn=True,
                wait_still_s=float(return_still_s),
            ),
        ]
        self._num_stages = len(self.stage_specs)
        self._stage_names = [spec["name"] for spec in self.stage_specs]
        self._critic_task_dim = e2e_critic_task_dim(self._num_stages)
        self._teacher = TaskDTeacherEnv.__new__(TaskDTeacherEnv)
        self._teacher.env = env
        self._teacher._device = device
        self._teacher.stage_specs = self.stage_specs
        self._teacher._num_stages = self._num_stages
        self._teacher._stage_names = self._stage_names
        self._teacher._init_reward_scalars(
            _stage_reach_tol=self._stage_reach_tol,
            _w_nav_dist=_E2E_W_NAV_DIST,
            _nav_dist_delta_clip=_E2E_REWARD_NO_CLIP,
            _w_push_box_axis=_E2E_W_PUSH_BOX_AXIS,
            _push_box_axis_delta_clip=_E2E_REWARD_NO_CLIP,
            _w_approach_box=_E2E_W_APPROACH_BOX,
            _approach_box_delta_clip=_E2E_REWARD_NO_CLIP,
            _w_face_box=_E2E_W_FACE_BOX,
            _face_box_delta_clip=_E2E_REWARD_NO_CLIP,
            _push_box_yaw_delta_clip=_E2E_REWARD_NO_CLIP,
            _final_robot_x_delta_clip=_E2E_REWARD_NO_CLIP,
            _w_push_box_yaw=0.0,
            _w_final_robot_x=0.0,
            _final_cross_box_x_bonus=0.0,
            _final_cross_box_x_plus1_bonus=0.0,
            _r_push1_yaw_complete_bonus=0.0,
            _r_step_penalty=_E2E_R_STEP_PENALTY,
        )
        self._teacher._idx_retreat = 0
        self._teacher._idx_sidestep_left = 1
        self._teacher._idx_push_adjust = 2
        self._teacher._idx_push2 = 2
        self._teacher._nav_log_interval = 0
        self._teacher._stage_idx_buf = None
        self._teacher._prev_robot_x = None
        self._teacher._prev_robot_y = None
        self._teacher._prev_box_x = None
        self._teacher._prev_box_y = None
        # Teacher sync/terminations must use e2e stage-reach logic (not 5-stage teacher MDP).
        self._teacher._compute_stage_reached = self._compute_stage_reached

        self._init_teacher_stage_tensors()
        self._teacher._traj_waypoints = _build_nominal_waypoints(self.stage_specs)
        stage_targets = self._teacher._traj_waypoints[1 : self._num_stages + 1]
        stage_starts = self._teacher._traj_waypoints[: self._num_stages]
        self._teacher._stage_start_x = torch.tensor(
            [p[0] for p in stage_starts], device=self._device, dtype=torch.float32
        )
        self._teacher._stage_start_y = torch.tensor(
            [p[1] for p in stage_starts], device=self._device, dtype=torch.float32
        )
        self._teacher._stage_target_x = torch.tensor(
            [p[0] for p in stage_targets], device=self._device, dtype=torch.float32
        )
        self._teacher._stage_target_y = torch.tensor(
            [p[1] for p in stage_targets], device=self._device, dtype=torch.float32
        )
        self._teacher._build_traj_segments()
        self._teacher._bind_env_origins()

        self._stage_idx_buf = None
        self._active_stage_count_buf = None
        self._stage_origin_x_buf = None
        self._stage_origin_y_buf = None
        self._stage_progress_buf = None
        self._step_wait_counter = None
        self._step_wait_armed = None
        self._return_still_counter = None
        self._return_still_armed = None
        self._current_obs = None
        self._t2e = _TRAIN_TO_ENV.to(device)
        self._done_stage_counts: list[int] = [0] * (self._num_stages + 1)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_LEG_DIM,), dtype=np.float32
        )

        print(
            f"[TaskDE2EStage] stages={self._num_stages} reach_tol={self._stage_reach_tol:.2f} "
            f"return_still={self._return_still_s:.1f}s leg_actions={_LEG_DIM} "
            f"reward_mode={self._reward_mode} "
            f"critic_task_dim={self._critic_task_dim} "
            f"task_rew=w_nav={_E2E_W_NAV_DIST:g} w_push={_E2E_W_PUSH_BOX_AXIS:g} "
            f"w_approach={_E2E_W_APPROACH_BOX:g} w_face={_E2E_W_FACE_BOX:g} no_clip sparse={_E2E_SPARSE_BONUS}",
            flush=True,
        )

    def _build_critic_task_priv(self) -> torch.Tensor:
        """Task-specific privileged critic features (same layout as TaskDStudentEnv)."""
        robot = self.env.unwrapped.scene["robot"]
        box = self.env.unwrapped.scene["box"]
        r_vel = robot.data.root_lin_vel_w.to(device=self._device, dtype=torch.float32)[:, :2]
        b_vel = box.data.root_lin_vel_w.to(device=self._device, dtype=torch.float32)[:, :2]
        rx, ry, robot_yaw = self._teacher._robot_pose()
        bx, by, _, box_yaw = self._teacher._box_pose()
        bx_body, by_body = self._teacher._relative_box_body(rx, ry, robot_yaw, bx, by)
        rel_yaw = torch.atan2(torch.sin(box_yaw - robot_yaw), torch.cos(box_yaw - robot_yaw))
        rel_world = torch.cat([bx - rx, by - ry], dim=-1)
        cf = self._teacher._contact_net_forces_w()
        contact_on = (cf.norm(dim=-1).max(dim=1).values > 2.0).to(dtype=torch.float32).unsqueeze(-1)
        stage_oh = self._teacher._stage_onehot(self.num_envs)
        stage_prog = self._stage_progress_buf.unsqueeze(-1)
        return torch.cat(
            [
                torch.cat([rx, ry, robot_yaw], dim=-1),
                torch.cat([bx, by, box_yaw], dim=-1),
                torch.cat([bx_body, by_body, rel_yaw], dim=-1),
                r_vel,
                b_vel,
                rel_world,
                contact_on,
                stage_oh,
                stage_prog,
            ],
            dim=-1,
        )

    def _enrich_obs(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            return obs
        out = dict(obs)
        out["critic_task"] = self._build_critic_task_priv()
        return out

    def _init_teacher_stage_tensors(self) -> None:
        specs = self.stage_specs
        dev = self._device
        t = self._teacher
        t._stage_axis_is_x = torch.tensor([spec["axis"] == "x" for spec in specs], device=dev, dtype=torch.bool)
        t._stage_push = torch.tensor([bool(spec["push"]) for spec in specs], device=dev, dtype=torch.bool)
        t._stage_relative_target = torch.tensor(
            [bool(spec.get("relative_robot_target", False)) for spec in specs], device=dev, dtype=torch.bool
        )
        t._stage_match_box_y_target = torch.tensor(
            [bool(spec.get("match_box_y_target", False)) for spec in specs], device=dev, dtype=torch.bool
        )
        t._stage_match_box_x_tol = torch.full((len(specs),), float("nan"), device=dev, dtype=torch.float32)
        t._stage_match_box_y_tol = torch.full((len(specs),), float("nan"), device=dev, dtype=torch.float32)
        t._stage_approach_y = torch.full((len(specs),), float("nan"), device=dev, dtype=torch.float32)
        t._stage_face_box_tol = torch.full((len(specs),), float("nan"), device=dev, dtype=torch.float32)
        t._stage_track_box_offset_x = torch.full((len(specs),), float("nan"), device=dev, dtype=torch.float32)
        t._stage_sign = torch.tensor([float(spec["sign"]) for spec in specs], device=dev, dtype=torch.float32)
        t._stage_dist = torch.tensor([float(spec["dist"]) for spec in specs], device=dev, dtype=torch.float32)
        t._stage_push_right_dist = torch.full((len(specs),), float("nan"), device=dev, dtype=torch.float32)
        t._stage_sparse_rewards = torch.tensor(
            [float(spec.get("sparse_bonus", 1.0)) for spec in specs], device=dev, dtype=torch.float32
        )
        t._stage_return_spawn = torch.tensor(
            [bool(spec.get("return_spawn", False)) for spec in specs], device=dev, dtype=torch.bool
        )
        t._stage_box_score_complete = torch.tensor(
            [bool(spec.get("box_score_complete", False)) for spec in specs], device=dev, dtype=torch.bool
        )
        t._stage_wait_still_s = torch.tensor(
            [
                float(spec["wait_still_s"]) if spec.get("wait_still_s") is not None else float("nan")
                for spec in specs
            ],
            device=dev,
            dtype=torch.float32,
        )

    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_episode_length(self) -> int:
        return getattr(self.env.unwrapped, "max_episode_length", 60000)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.unwrapped.episode_length_buf

    def _ensure_buffers(self, batch: int) -> None:
        if self._stage_idx_buf is not None and int(self._stage_idx_buf.shape[0]) == int(batch):
            return
        self._teacher._ensure_state_buffers(batch)
        self._stage_idx_buf = self._teacher._stage_idx_buf
        self._active_stage_count_buf = self._teacher._active_stage_count_buf
        self._stage_origin_x_buf = self._teacher._stage_origin_x_buf
        self._stage_origin_y_buf = self._teacher._stage_origin_y_buf
        self._stage_progress_buf = self._teacher._stage_progress_buf
        self._step_wait_counter = self._teacher._step_wait_counter
        self._step_wait_armed = self._teacher._step_wait_armed
        self._return_still_counter = torch.zeros(batch, device=self._device, dtype=torch.long)
        self._return_still_armed = torch.zeros(batch, device=self._device, dtype=torch.bool)

    def _build_env_action(self, leg_action: torch.Tensor) -> torch.Tensor:
        batch = leg_action.shape[0]
        action = torch.zeros(batch, _TOTAL_ACTION_DIM, device=self._device, dtype=torch.float32)
        action[:, :_LEG_DIM] = leg_action * self._t2e
        return action

    def _spawn_target_xy(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        ox = self._teacher._env_origin_x
        oy = self._teacher._env_origin_y
        return task_d_robot_spawn_world_xy(ox, oy)

    def _box_in_score_range(self, bx: torch.Tensor) -> torch.Tensor:
        nx = self._teacher._box_nominal_x(bx)
        in_range = torch.zeros_like(nx, dtype=torch.bool)
        for lo, hi in _BOX_SCORE_X_RANGES:
            in_range |= (nx >= lo) & (nx <= hi)
        return in_range

    def _compute_return_still_reached(
        self,
        valid: torch.Tensor,
        stage_idx: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 4: at spawn (tol) and hold low speed for ``return_still_s``."""
        reached = torch.zeros(valid.shape[0], device=self._device, dtype=torch.bool)
        holding = torch.zeros_like(reached)
        return_mask = valid & self._teacher._stage_return_spawn[stage_idx]
        if not bool(return_mask.any()):
            return reached, holding

        spawn_x, spawn_y = self._spawn_target_xy(int(valid.shape[0]))
        rx1 = rx.squeeze(-1)
        ry1 = ry.squeeze(-1)
        at_spawn = torch.hypot(rx1 - spawn_x, ry1 - spawn_y) <= self._stage_reach_tol

        robot = self.env.unwrapped.scene["robot"]
        speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
        still = speed <= 0.03

        active = return_mask & at_spawn & still
        newly_armed = active & (~self._return_still_armed)
        self._return_still_armed[newly_armed] = True
        self._return_still_counter[newly_armed] = 0
        not_ready = return_mask & (~(at_spawn & still))
        self._return_still_counter[not_ready] = 0
        self._return_still_armed[not_ready] = False

        armed = return_mask & self._return_still_armed
        if bool(armed.any()):
            wait_steps = max(1, int(round(self._return_still_s / self.env.unwrapped.step_dt)))
            self._return_still_counter[armed] += 1
            reached = reached | (armed & (self._return_still_counter >= wait_steps))
            holding = holding | (armed & (self._return_still_counter < wait_steps))
        return reached, holding

    def _stage_target_xy(
        self,
        stage_idx: torch.Tensor,
        valid: torch.Tensor,
        rx: torch.Tensor,
        ry: torch.Tensor,
        bx: torch.Tensor,
        by: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tx, ty = self._teacher._compute_stage_target(stage_idx, valid, rx, ry, bx, by)
        return_mask = valid & self._teacher._stage_return_spawn[stage_idx]
        if bool(return_mask.any()):
            spawn_x, spawn_y = self._spawn_target_xy(int(valid.shape[0]))
            tx = torch.where(return_mask, spawn_x, tx)
            ty = torch.where(return_mask, spawn_y, ty)
        return tx, ty

    def _xy_within_tol(
        self,
        rx: torch.Tensor,
        ry: torch.Tensor,
        tx: torch.Tensor,
        ty: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (rx.squeeze(-1) - tx).abs() <= self._stage_reach_tol
        ) & ((ry.squeeze(-1) - ty).abs() <= self._stage_reach_tol)

    def _compute_stage_reached(
        self, rx, ry, bx, by, bz
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_idx = torch.clamp(self._stage_idx_buf, min=0, max=self._num_stages - 1)
        valid = self._stage_idx_buf < self._active_stage_count_buf
        target_x, target_y = self._stage_target_xy(stage_idx, valid, rx, ry, bx, by)
        dist_to_target = torch.hypot(rx.squeeze(-1) - target_x, ry.squeeze(-1) - target_y)
        xy_ok = self._xy_within_tol(rx, ry, target_x, target_y)

        reached = torch.zeros(valid.shape[0], device=self._device, dtype=torch.bool)

        # Stage 1 & 2: xy within tol.
        nav_mask = valid & (~self._teacher._stage_push[stage_idx]) & (~self._teacher._stage_return_spawn[stage_idx])
        reached = torch.where(nav_mask, xy_ok, reached)

        # Stage 3: box enters 14-point x range.
        push_mask = valid & self._teacher._stage_box_score_complete[stage_idx]
        if bool(push_mask.any()):
            reached = torch.where(push_mask, self._box_in_score_range(bx), reached)

        # Stage 4: at spawn xy tol + hold still.
        still_reached, _ = self._compute_return_still_reached(valid, stage_idx, rx, ry)
        return_mask = valid & self._teacher._stage_return_spawn[stage_idx]
        reached = torch.where(return_mask, still_reached & xy_ok, reached)

        self._stage_progress_buf = torch.clamp(
            self._teacher._stage_dist[stage_idx] - dist_to_target, min=0.0
        )
        return reached, dist_to_target

    def _count_done_stages(self, done_now: torch.Tensor, stage_idx: torch.Tensor) -> None:
        if not bool(done_now.any()):
            return
        idx = torch.clamp(stage_idx[done_now], min=0, max=self._num_stages)
        for i in range(self._num_stages):
            self._done_stage_counts[i] += int((idx == i).sum().item())
        self._done_stage_counts[self._num_stages] += int((idx >= self._num_stages).sum().item())

    def _stage_at_done_episode_log(
        self, done_now: torch.Tensor, stage_idx: torch.Tensor
    ) -> dict[str, float]:
        """Fraction of terminating envs per stage (Episode_Termination_Stage/* for rsl_rl)."""
        total = int(done_now.sum().item())
        if total <= 0:
            return {}
        idx = torch.clamp(stage_idx[done_now], min=0, max=self._num_stages)
        out: dict[str, float] = {}
        for i, name in enumerate(self._stage_names):
            out[f"Episode_Termination_Stage/{name}"] = float((idx == i).sum().item()) / total
        out["Episode_Termination_Stage/finished"] = float((idx >= self._num_stages).sum().item()) / total
        return out

    def _stage_reward_episode_log(self, done_now: torch.Tensor) -> dict[str, float]:
        """Per-stage and total task reward at episode end (Episode_StageReward/*, Episode_Reward/e2e_stage)."""
        if self._teacher._ep_stage_reward_buf is None or not bool(done_now.any()):
            return {}
        ep_len_s = float(getattr(self.env.unwrapped, "max_episode_length_s", 1.0))
        ep_len_s = max(ep_len_s, 1.0e-6)
        ep_rew = self._teacher._ep_stage_reward_buf[done_now]
        out = {
            f"Episode_StageReward/{name}": float(ep_rew[:, i].mean().item())
            for i, name in enumerate(self._stage_names)
        }
        out["Episode_Reward/e2e_stage"] = float(ep_rew.sum(dim=1).mean().item()) / ep_len_s
        return out

    def get_observations(self):
        obs, info = self._current_obs, {}
        if obs is not None:
            obs = self._enrich_obs(obs)
        return obs, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self._step_count = 0
        self._done_stage_counts = [0] * (self._num_stages + 1)
        # Match TaskDTeacherEnv.reset: refresh per-tile origins after sim reset (segment endpoints use these).
        self._teacher._bind_env_origins()
        rx, ry, _ = self._teacher._robot_pose()
        bx, by, _, _ = self._teacher._box_pose()
        self._ensure_buffers(rx.shape[0])
        self._teacher._prev_robot_x, self._teacher._prev_robot_y = rx.clone(), ry.clone()
        self._teacher._prev_box_x, self._teacher._prev_box_y = bx.clone(), by.clone()
        full = torch.ones(rx.shape[0], device=self._device, dtype=torch.bool)
        self._teacher._reset_env_state(full, rx, ry, bx, by)
        self._return_still_counter.zero_()
        self._return_still_armed.zero_()
        bx, by, bz, _ = self._teacher._box_pose()
        self._teacher._sync_nav_stage_idx(rx, ry, bx, by, bz)
        return self._enrich_obs(obs), info

    def step(self, action: torch.Tensor):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        action = action.to(self._device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        self._ensure_buffers(self.num_envs)
        self._step_count += 1
        act_abs_mean = float(action.abs().mean().item())

        rx, ry, _ = self._teacher._robot_pose()
        bx, by, bz, _ = self._teacher._box_pose()
        self._teacher._sync_nav_stage_idx(rx, ry, bx, by, bz)

        env_action = self._build_env_action(action.clamp(-1.0, 1.0))
        env_act_abs_mean = float(env_action[:, :_LEG_DIM].abs().mean().item())
        obs, base_rew, term, trunc, info = self.env.step(env_action)
        self._current_obs = obs

        step_term = term.squeeze(-1).bool() if isinstance(term, torch.Tensor) and term.ndim > 1 else term.bool()
        step_trunc = trunc.squeeze(-1).bool() if isinstance(trunc, torch.Tensor) and trunc.ndim > 1 else trunc.bool()
        done_now = step_term | step_trunc
        stage_idx_before = self._stage_idx_buf.clone()

        rx, ry, _ = self._teacher._robot_pose()
        bx, by, bz, _ = self._teacher._box_pose()
        reached, dist_to_target = self._compute_stage_reached(rx, ry, bx, by, bz)
        stage_rew, _, _ = self._teacher._compute_reward(
            dist_to_target, done_now, reached, rx, ry, bx, by, bz
        )
        valid = self._stage_idx_buf < self._active_stage_count_buf
        stage_rew = stage_rew * valid.to(dtype=stage_rew.dtype)
        self._teacher._accumulate_episode_stage_reward(stage_rew, valid)

        if bool(reached.any()):
            stage_bonus = self._teacher._stage_sparse_bonus(reached)
            stage_rew = stage_rew + stage_bonus
            self._teacher._accumulate_episode_stage_reward(stage_bonus, reached)
            rx1 = rx.squeeze(-1)
            ry1 = ry.squeeze(-1)
            self._stage_origin_x_buf = torch.where(reached, rx1, self._stage_origin_x_buf)
            self._stage_origin_y_buf = torch.where(reached, ry1, self._stage_origin_y_buf)
            self._stage_idx_buf = torch.where(reached, self._stage_idx_buf + 1, self._stage_idx_buf)
            self._teacher._sync_nav_stage_idx(rx, ry, bx, by, bz)

        mission_done = self._stage_idx_buf >= self._num_stages
        if bool(mission_done.any()):
            step_term = step_term | mission_done

        ep_done = done_now | mission_done
        log_stage_idx = torch.where(mission_done, self._stage_idx_buf, stage_idx_before)

        if self._reward_mode == "stage_only":
            total_rew = stage_rew
        elif self._reward_mode == "base_plus_stage":
            total_rew = base_rew + stage_rew
        else:
            total_rew = base_rew
        episode_log_patch: dict[str, float] = {}
        done_count = 0
        done_retreat_rew = float("nan")
        if bool(ep_done.any()):
            done_count = int(ep_done.sum().item())
            if self._teacher._ep_stage_reward_buf is not None:
                done_retreat_rew = float(self._teacher._ep_stage_reward_buf[ep_done][:, 0].mean().item())
            self._count_done_stages(ep_done, log_stage_idx)
            episode_log_patch.update(self._stage_reward_episode_log(ep_done))
            episode_log_patch.update(self._stage_at_done_episode_log(ep_done, log_stage_idx))

        self._teacher._prev_robot_x, self._teacher._prev_robot_y = rx.clone(), ry.clone()
        self._teacher._prev_box_x, self._teacher._prev_box_y = bx.clone(), by.clone()
        self._teacher._prev_box_com_z = bz.squeeze(-1)
        self._teacher._reset_env_state(ep_done, rx, ry, bx, by)

        if self._stage_log_interval > 0 and self._step_count % self._stage_log_interval == 0:
            idx0 = int(self._stage_idx_buf[0].item())
            name = self._stage_names[idx0] if idx0 < self._num_stages else "done"
            denom = max(1, sum(self._done_stage_counts))
            stage_at_done = " ".join(
                f"{self._stage_names[i]}={self._done_stage_counts[i] / denom:.2f}"
                for i in range(self._num_stages)
            )
            if self._done_stage_counts[self._num_stages] > 0:
                stage_at_done += f" finished={self._done_stage_counts[self._num_stages] / denom:.2f}"
            print(
                f"[TaskDE2EStage] step={self._step_count} stage={idx0 + 1}/{self._num_stages}({name}) "
                f"rew={float(total_rew.mean().item()):+.3f} base={float(base_rew.mean().item()):+.3f} "
                f"stage={float(stage_rew.mean().item()):+.3f} "
                f"|a|={act_abs_mean:.3f} |a_env|={env_act_abs_mean:.3f} "
                f"done_n={done_count} done_retreat={done_retreat_rew:+.3f} "
                f"stage_at_done[{stage_at_done}]",
                flush=True,
            )

        info = dict(info) if isinstance(info, dict) else {}
        if episode_log_patch:
            log = dict(info.get("log", {}))
            log.update(episode_log_patch)
            info["log"] = log
        info["e2e_stage"] = self._stage_names[
            min(int(self._stage_idx_buf[0].item()), self._num_stages - 1)
        ]
        info["e2e_stage_idx"] = int(self._stage_idx_buf[0].item())
        return self._enrich_obs(obs), total_rew, step_term, step_trunc, info


class TaskDE2ERslVecEnvWrapper(VecEnv):
    """RSL-RL bridge for TaskDE2EStageEnv (12D actions + wrapper obs incl. critic_task)."""

    def __init__(self, env: TaskDE2EStageEnv, clip_actions: float | None = 1.0):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_actions = int(gym.spaces.flatdim(env.action_space))
        self.max_episode_length = env.max_episode_length

        if clip_actions is not None:
            self.env.action_space = gym.spaces.Box(
                low=-clip_actions,
                high=clip_actions,
                shape=(self.num_actions,),
                dtype=np.float32,
            )

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
        self.unwrapped.episode_length_buf.copy_(value)

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
        out = dict(extras) if isinstance(extras, dict) else {}
        if not self.unwrapped.cfg.is_finite_horizon:
            out["time_outs"] = truncated
        # RSL-RL appends extras['log'] every step and averages → dilutes episode metrics to ~0.
        # Only publish Isaac + task episode stats on done steps via extras['episode'].
        if bool(dones.any()) and "log" in out:
            out["episode"] = dict(out["log"])
        out.pop("log", None)
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, out

    def close(self):
        return self.env.close()


__all__ = ["TaskDE2EStageEnv", "TaskDE2ERslVecEnvWrapper", "e2e_critic_task_dim"]
