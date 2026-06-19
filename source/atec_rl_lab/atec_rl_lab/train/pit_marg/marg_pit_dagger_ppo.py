"""DAgger-style PPO for depth student with frozen MARG height-map teacher."""

from __future__ import annotations

import torch
import torch.nn as nn

from atec_rl_lab.train.locomotion.marg.marg_actor_critic import MargActorCritic
from atec_rl_lab.train.locomotion.marg.marg_ppo import MargPPO


class MargPitDaggerPPO(MargPPO):
    """MargPPO + BC loss on teacher leg actions; optional beta-mixed rollouts."""

    def __init__(
        self,
        policy,
        teacher: MargActorCritic | None = None,
        dagger_coef: float = 1.0,
        dagger_beta: float = 1.0,
        dagger_beta_end: float = 0.0,
        dagger_beta_decay_iters: int = 4000,
        **kwargs,
    ):
        self.teacher = teacher
        self.dagger_coef = float(dagger_coef)
        self.dagger_beta_start = float(dagger_beta)
        self.dagger_beta = float(dagger_beta)
        self.dagger_beta_end = float(dagger_beta_end)
        self.dagger_beta_decay_iters = max(1, int(dagger_beta_decay_iters))
        self._dagger_updates = 0
        self._last_teacher_rollout_frac = 0.0
        super().__init__(policy, **kwargs)
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def set_dagger_iteration(self, iteration: int) -> None:
        """Linearly decay rollout mixing beta from start -> end."""
        frac = min(1.0, max(0.0, float(iteration) / float(self.dagger_beta_decay_iters)))
        self.dagger_beta = self.dagger_beta_start + frac * (self.dagger_beta_end - self.dagger_beta_start)

    def _teacher_obs(self, obs) -> dict[str, torch.Tensor]:
        return {
            "proprio": obs["proprio"],
            "proprio_history": obs["proprio_history"],
            "height_map": obs["height_map"],
        }

    @torch.no_grad()
    def _teacher_actions(self, obs) -> torch.Tensor:
        if self.teacher is None:
            raise RuntimeError("MargPitDaggerPPO requires a loaded teacher.")
        return self.teacher.act_inference(self._teacher_obs(obs))

    def _record_transition(self, obs, exec_actions: torch.Tensor) -> None:
        """Store rollout transition; distribution must already be updated."""
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = exec_actions.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(exec_actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs

    def act(self, obs, **kwargs):
        del kwargs
        self.set_dagger_iteration(self._dagger_updates)
        if self.teacher is None or self.dagger_beta <= 0.0:
            return super().act(obs)

        teacher_actions = self._teacher_actions(obs)
        self.policy.update_policy_distribution(obs)

        if self.dagger_beta >= 1.0:
            exec_actions = teacher_actions
            teacher_frac = 1.0
        else:
            student_actions = self.policy.distribution.sample()
            mix = torch.rand(student_actions.shape[0], device=self.device) < self.dagger_beta
            exec_actions = student_actions.clone()
            if mix.any():
                exec_actions[mix] = teacher_actions[mix]
            teacher_frac = float(mix.float().mean().item())

        self._last_teacher_rollout_frac = teacher_frac
        self._record_transition(obs, exec_actions)
        return exec_actions

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_reg_loss = 0.0
        mean_dagger_loss = 0.0
        valid_steps = 0
        # Teacher-heavy rollouts: skip PPO surrogate (log_prob of teacher actions under student explodes).
        use_surrogate = self.dagger_beta < 0.5

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            self.policy.update_policy_distribution(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if use_surrogate and self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if use_surrogate:
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                ratio = torch.clamp(ratio, 0.0, 10.0)
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            else:
                surrogate_loss = torch.zeros((), device=self.device)

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            reg_loss = self.policy.estimator_regression_loss(obs_batch)
            dagger_loss = torch.tensor(0.0, device=self.device)
            if self.teacher is not None and self.dagger_coef > 0.0:
                teacher_actions = self._teacher_actions(obs_batch)
                dagger_loss = torch.nn.functional.mse_loss(mu_batch, teacher_actions)

            entropy_term = entropy_batch.mean()
            if not torch.isfinite(entropy_term):
                entropy_term = torch.zeros((), device=self.device)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_term
                + self.reg_loss_coef * reg_loss
                + self.dagger_coef * dagger_loss
            )

            if not torch.isfinite(loss):
                print(
                    "[MargPitDaggerPPO] skip update step: non-finite loss "
                    f"(surrogate={surrogate_loss.item()}, value={value_loss.item()}, "
                    f"reg={reg_loss.item()}, dagger={dagger_loss.item()})",
                    flush=True,
                )
                continue

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_term.item()
            mean_reg_loss += reg_loss.item()
            mean_dagger_loss += dagger_loss.item()
            valid_steps += 1

        denom = max(1, valid_steps)
        mean_value_loss /= denom
        mean_surrogate_loss /= denom
        mean_entropy /= denom
        mean_reg_loss /= denom
        mean_dagger_loss /= denom
        self.storage.clear()
        self._dagger_updates += 1

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator_reg": mean_reg_loss,
            "dagger_bc": mean_dagger_loss,
            "dagger_beta": float(self.dagger_beta),
            "teacher_rollout_frac": float(self._last_teacher_rollout_frac),
        }
