"""Interaction store — learning from past interactions.

Records user inputs, chosen actions, outcomes, and success/failure
to enable Pixie to improve tool selection and response quality.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STORE_PATH = Path(".pixie_data") / "interactions.jsonl"
MAX_ENTRIES = 500  # Keep store bounded


class InteractionEntry:
    """A single interaction record."""

    __slots__ = ("timestamp", "user_input", "action", "tool_used", "outcome", "success")

    def __init__(
        self,
        user_input: str,
        action: str,
        tool_used: str,
        outcome: str,
        success: bool,
        timestamp: float | None = None,
    ) -> None:
        self.timestamp = timestamp or time.time()
        self.user_input = user_input
        self.action = action
        self.tool_used = tool_used
        self.outcome = outcome
        self.success = success

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "action": self.action,
            "tool_used": self.tool_used,
            "outcome": self.outcome,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionEntry:
        return cls(
            user_input=data.get("user_input", ""),
            action=data.get("action", ""),
            tool_used=data.get("tool_used", ""),
            outcome=data.get("outcome", ""),
            success=data.get("success", False),
            timestamp=data.get("timestamp"),
        )


class InteractionStore:
    """Append-only store of past interactions for learning.

    Uses JSONL format for efficient append and bounded memory.
    Supports keyword-based similarity search for finding relevant past actions.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or STORE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[InteractionEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add(self, entry: InteractionEntry) -> None:
        """Add a new interaction record."""
        self._entries.append(entry)
        self._append_to_disk(entry)

        # Truncate in-memory if too large
        if len(self._entries) > MAX_ENTRIES:
            self._entries = self._entries[-MAX_ENTRIES:]

    def query_similar(self, user_input: str, top_k: int = 3) -> list[InteractionEntry]:
        """Find past interactions similar to the given input.

        Uses simple keyword overlap scoring (no ML required).
        """
        if not self._entries:
            return []

        input_words = set(user_input.lower().split())
        if not input_words:
            return []

        scored: list[tuple[float, InteractionEntry]] = []
        for entry in self._entries:
            entry_words = set(entry.user_input.lower().split())
            if not entry_words:
                continue
            overlap = len(input_words & entry_words)
            score = overlap / max(len(input_words), 1)
            if score > 0.2:  # Minimum relevance threshold
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def get_successful_actions(self, top_k: int = 5) -> list[dict[str, Any]]:
        """Return recently successful actions for context injection."""
        successful = [e for e in reversed(self._entries) if e.success]
        results = []
        for entry in successful[:top_k]:
            results.append({
                "input": entry.user_input[:80],
                "tool": entry.tool_used,
                "action": entry.action[:80],
            })
        return results

    def get_stats(self) -> dict[str, Any]:
        """Return basic interaction statistics."""
        total = len(self._entries)
        successes = sum(1 for e in self._entries if e.success)
        return {
            "total_interactions": total,
            "success_rate": successes / max(total, 1),
            "total_successes": successes,
            "total_failures": total - successes,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from JSONL file."""
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-MAX_ENTRIES:]:  # Only keep recent entries
                if line.strip():
                    data = json.loads(line)
                    self._entries.append(InteractionEntry.from_dict(data))
            log.debug("Loaded %d interactions from store", len(self._entries))
        except Exception:
            log.warning("Failed to load interaction store", exc_info=True)

    def _append_to_disk(self, entry: InteractionEntry) -> None:
        """Append a single entry to the JSONL file."""
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception:
            log.warning("Failed to append interaction", exc_info=True)
