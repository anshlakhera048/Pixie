"""Task scheduler — asyncio-based background task scheduling.

Provides delayed and recurring task execution that integrates
with the async runtime without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from uuid import uuid4

log = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A task scheduled for future execution."""

    id: str = field(default_factory=lambda: uuid4().hex[:8])
    name: str = ""
    delay_seconds: float = 0.0
    repeat_interval: float | None = None  # None = one-shot; >0 = recurring
    callback_name: str = ""  # For serialization/reload
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    next_run: float = 0.0
    active: bool = True


class TaskScheduler:
    """Asyncio-based background task scheduler.

    Schedules tasks with delays and optional recurrence.
    Tasks are executed as asyncio tasks within the event loop.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._handles: dict[str, asyncio.Task] = {}
        self._callbacks: dict[str, Callable[..., Coroutine]] = {}
        self._running = False
        self._loop_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_callback(self, name: str, fn: Callable[..., Coroutine]) -> None:
        """Register a named async callback that can be scheduled."""
        self._callbacks[name] = fn
        log.debug("Registered scheduler callback: %s", name)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        name: str,
        callback_name: str,
        delay_seconds: float,
        repeat_interval: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> str:
        """Schedule a task for future execution.

        Args:
            name: Human-readable task name.
            callback_name: Name of registered callback to invoke.
            delay_seconds: Seconds until first execution.
            repeat_interval: If set, re-run every N seconds after first.
            args: Positional args for callback.
            kwargs: Keyword args for callback.

        Returns:
            Task ID.
        """
        if callback_name not in self._callbacks:
            raise ValueError(f"Unknown callback: {callback_name}")

        task = ScheduledTask(
            name=name,
            delay_seconds=delay_seconds,
            repeat_interval=repeat_interval,
            callback_name=callback_name,
            args=args,
            kwargs=kwargs or {},
            next_run=time.time() + delay_seconds,
        )
        self._tasks[task.id] = task
        log.info("Scheduled task %s (%s) in %.1fs", task.id, name, delay_seconds)

        # If already running, start handling immediately
        if self._running:
            self._schedule_handle(task)

        return task.id

    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.active = False
        handle = self._handles.pop(task_id, None)
        if handle and not handle.done():
            handle.cancel()
        del self._tasks[task_id]
        log.info("Cancelled task %s", task_id)
        return True

    def list_tasks(self) -> list[dict[str, Any]]:
        """Return list of active scheduled tasks."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "next_run": t.next_run,
                "repeat": t.repeat_interval,
                "active": t.active,
            }
            for t in self._tasks.values()
            if t.active
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        self._running = True
        # Schedule handles for any pre-loaded tasks
        for task in self._tasks.values():
            if task.active:
                self._schedule_handle(task)
        log.info("TaskScheduler started")

    async def stop(self) -> None:
        """Stop the scheduler and cancel pending handles."""
        self._running = False
        for handle in self._handles.values():
            if not handle.done():
                handle.cancel()
        self._handles.clear()
        log.info("TaskScheduler stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_handle(self, task: ScheduledTask) -> None:
        """Create an asyncio task to execute at the scheduled time."""
        delay = max(0, task.next_run - time.time())
        handle = asyncio.ensure_future(self._run_after_delay(task, delay))
        self._handles[task.id] = handle

    async def _run_after_delay(self, task: ScheduledTask, delay: float) -> None:
        """Wait for delay then execute the task callback."""
        try:
            await asyncio.sleep(delay)

            if not task.active or not self._running:
                return

            callback = self._callbacks.get(task.callback_name)
            if callback is None:
                log.error("Callback %s not found for task %s", task.callback_name, task.id)
                return

            log.info("Executing scheduled task: %s", task.name)
            await callback(*task.args, **task.kwargs)

            # Handle recurrence
            if task.repeat_interval and task.active and self._running:
                task.next_run = time.time() + task.repeat_interval
                self._schedule_handle(task)
            else:
                # One-shot task complete
                task.active = False
                self._handles.pop(task.id, None)
                self._tasks.pop(task.id, None)

        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Scheduled task %s failed", task.id)
            task.active = False
