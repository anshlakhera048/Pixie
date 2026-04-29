"""Execution Engine — state machine for multi-step task execution.

Tracks progress through a plan, stores intermediate results, and
provides step-by-step iteration for the agent loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from utils.logger import log_event

log = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class StepResult:
    """Result of a single execution step."""

    step_index: int
    description: str
    status: TaskStatus
    result: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class ExecutionState:
    """Complete state of a multi-step execution."""

    plan: list[str]
    current_index: int = 0
    status: TaskStatus = TaskStatus.PENDING
    steps: list[StepResult] = field(default_factory=list)
    started_at: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.started_at == 0.0:
            return 0.0
        return time.time() - self.started_at


class ExecutionEngine:
    """State machine that drives plan execution.

    Usage:
        engine = ExecutionEngine()
        engine.start_task(["step 1", "step 2", "step 3"])
        while not engine.is_complete():
            step = engine.next_step()
            result = ... # agent executes step
            engine.update_result(result)
    """

    def __init__(self) -> None:
        self._state: ExecutionState | None = None
        self._interrupted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start_task(self, plan: list[str]) -> None:
        """Initialize execution with a plan."""
        if not plan:
            raise ValueError("Cannot start task with empty plan")

        self._interrupted = False
        self._state = ExecutionState(
            plan=plan,
            status=TaskStatus.IN_PROGRESS,
            started_at=time.time(),
            steps=[
                StepResult(step_index=i, description=desc, status=TaskStatus.PENDING)
                for i, desc in enumerate(plan)
            ],
        )
        log_event(log, logging.INFO, "task_started", total_steps=len(plan))

    def next_step(self) -> str | None:
        """Return the description of the next step to execute, or None if done."""
        if self._state is None or self.is_complete():
            return None

        if self._interrupted:
            self._finalize(TaskStatus.INTERRUPTED)
            return None

        idx = self._state.current_index
        if idx >= len(self._state.steps):
            self._finalize(TaskStatus.COMPLETED)
            return None

        step = self._state.steps[idx]
        step.status = TaskStatus.IN_PROGRESS
        step.started_at = time.time()

        log_event(
            log, logging.INFO, "step_started",
            step=idx + 1, total=len(self._state.plan), description=step.description,
        )
        return step.description

    def update_result(self, result: str, success: bool = True) -> None:
        """Record the result of the current step and advance."""
        if self._state is None:
            return

        idx = self._state.current_index
        if idx >= len(self._state.steps):
            return

        step = self._state.steps[idx]
        step.result = result
        step.finished_at = time.time()
        step.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED

        log_event(
            log, logging.INFO, "step_completed",
            step=idx + 1, success=success,
        )

        if not success:
            self._finalize(TaskStatus.FAILED)
            return

        self._state.current_index += 1

        # Check if all steps done
        if self._state.current_index >= len(self._state.steps):
            self._finalize(TaskStatus.COMPLETED)

    def is_complete(self) -> bool:
        """True if execution has finished (success, failure, or interrupt)."""
        if self._state is None:
            return True
        return self._state.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.INTERRUPTED,
        )

    def interrupt(self) -> None:
        """Signal that execution should stop at the next opportunity."""
        self._interrupted = True
        log_event(log, logging.WARNING, "execution_interrupted")

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of current execution state."""
        if self._state is None:
            return {"status": "idle", "message": "No active task"}

        completed = sum(1 for s in self._state.steps if s.status == TaskStatus.COMPLETED)
        total = len(self._state.steps)

        return {
            "status": self._state.status.value,
            "progress": f"{completed}/{total}",
            "current_step": (
                self._state.steps[self._state.current_index].description
                if self._state.current_index < total
                else None
            ),
            "elapsed_seconds": round(self._state.elapsed, 1),
            "results": [
                {"step": s.step_index + 1, "desc": s.description, "status": s.status.value}
                for s in self._state.steps
            ],
        }

    def get_results(self) -> list[str]:
        """Return all completed step results."""
        if self._state is None:
            return []
        return [
            f"Step {s.step_index + 1}: {s.result}"
            for s in self._state.steps
            if s.status == TaskStatus.COMPLETED
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _finalize(self, status: TaskStatus) -> None:
        """Mark the overall task as finished."""
        if self._state is not None:
            self._state.status = status
            log_event(
                log, logging.INFO, "task_finalized",
                status=status.value,
                elapsed=round(self._state.elapsed, 2),
            )
