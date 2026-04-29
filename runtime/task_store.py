"""Persistent task store — saves/reloads scheduled tasks to disk as JSON.

Ensures scheduled tasks survive restarts by persisting them to
.pixie_data/scheduled_tasks.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from runtime.scheduler import ScheduledTask

log = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path(".pixie_data") / "scheduled_tasks.json"


class TaskStore:
    """JSON-based persistent store for scheduled tasks."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_STORE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def save(self, tasks: list[ScheduledTask]) -> None:
        """Persist active tasks to disk."""
        data = [self._serialize(t) for t in tasks if t.active]
        try:
            self._path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            log.debug("Saved %d tasks to %s", len(data), self._path)
        except Exception:
            log.warning("Failed to save tasks", exc_info=True)

    def load(self) -> list[ScheduledTask]:
        """Load tasks from disk. Returns empty list if file doesn't exist."""
        if not self._path.exists():
            return []

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            tasks = [self._deserialize(d) for d in data if d]
            # Filter out tasks whose next_run is way in the past (stale)
            now = time.time()
            valid = []
            for t in tasks:
                if t.repeat_interval or t.next_run > now - 60:
                    valid.append(t)
                else:
                    log.debug("Discarding stale task: %s", t.name)
            log.info("Loaded %d tasks from store", len(valid))
            return valid
        except Exception:
            log.warning("Failed to load task store", exc_info=True)
            return []

    def clear(self) -> None:
        """Remove the task store file."""
        if self._path.exists():
            self._path.unlink()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(task: ScheduledTask) -> dict[str, Any]:
        return {
            "id": task.id,
            "name": task.name,
            "delay_seconds": task.delay_seconds,
            "repeat_interval": task.repeat_interval,
            "callback_name": task.callback_name,
            "args": list(task.args),
            "kwargs": task.kwargs,
            "created_at": task.created_at,
            "next_run": task.next_run,
        }

    @staticmethod
    def _deserialize(data: dict[str, Any]) -> ScheduledTask:
        return ScheduledTask(
            id=data.get("id", ""),
            name=data.get("name", ""),
            delay_seconds=data.get("delay_seconds", 0),
            repeat_interval=data.get("repeat_interval"),
            callback_name=data.get("callback_name", ""),
            args=tuple(data.get("args", ())),
            kwargs=data.get("kwargs", {}),
            created_at=data.get("created_at", time.time()),
            next_run=data.get("next_run", time.time()),
            active=True,
        )
