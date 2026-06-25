"""Task-B student actor-critic: same asymmetric RNN critic stack as TaskDStudentActorCritic."""

from __future__ import annotations

from .taskd_student_actor_critic import TaskDStudentActorCritic


class TaskBStudentActorCritic(TaskDStudentActorCritic):
    """Nav-only Task B student (3D actions); critic = CNN fuse + priv MLP + GRU (Task D layout)."""

    pass
