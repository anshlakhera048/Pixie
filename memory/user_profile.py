"""User profile — persistent user preferences and usage patterns.

Stores user preferences, frequently used commands, and behavioral
patterns to personalize Pixie's responses over time.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROFILE_PATH = Path(".pixie_data") / "user_profile.json"


class UserProfile:
    """Persistent user profile for personalization.

    Tracks preferences, command frequency, and interaction patterns.
    All data stored as a flat JSON file for fast read/write.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or PROFILE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {
            "preferences": {},
            "command_frequency": {},
            "tool_frequency": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load profile from disk. No-op if file doesn't exist."""
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            loaded = json.loads(raw)
            # Merge with defaults to handle schema evolution
            for key in self._data:
                if key in loaded:
                    self._data[key] = loaded[key]
            log.debug("Loaded user profile from %s", self._path)
        except Exception:
            log.warning("Failed to load user profile", exc_info=True)

    def save(self) -> None:
        """Persist profile to disk."""
        self._data["updated_at"] = time.time()
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2), encoding="utf-8"
            )
        except Exception:
            log.warning("Failed to save user profile", exc_info=True)

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def update_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        self._data["preferences"][key] = value
        self.save()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference value."""
        return self._data["preferences"].get(key, default)

    def get_all_preferences(self) -> dict[str, Any]:
        """Return all stored preferences."""
        return dict(self._data["preferences"])

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def record_command(self, command: str) -> None:
        """Track frequency of a user command/input pattern."""
        freq = self._data["command_frequency"]
        freq[command] = freq.get(command, 0) + 1
        # Periodic save (every 5 commands)
        total = sum(freq.values())
        if total % 5 == 0:
            self.save()

    def record_tool_use(self, tool_name: str) -> None:
        """Track which tools are used most frequently."""
        freq = self._data["tool_frequency"]
        freq[tool_name] = freq.get(tool_name, 0) + 1

    def get_frequent_tools(self, top_n: int = 5) -> list[str]:
        """Return the most frequently used tools."""
        freq = self._data["tool_frequency"]
        sorted_tools = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_tools[:top_n]]

    def get_frequent_commands(self, top_n: int = 5) -> list[str]:
        """Return the most frequently used commands."""
        freq = self._data["command_frequency"]
        sorted_cmds = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [cmd for cmd, _ in sorted_cmds[:top_n]]

    # ------------------------------------------------------------------
    # Profile summary (for context injection)
    # ------------------------------------------------------------------

    def get_profile_summary(self) -> str:
        """Return a short text summary of the user profile for LLM context."""
        parts: list[str] = []

        prefs = self._data["preferences"]
        if prefs:
            pref_lines = [f"  • {k}: {v}" for k, v in list(prefs.items())[:8]]
            parts.append("User Preferences:\n" + "\n".join(pref_lines))

        frequent_tools = self.get_frequent_tools(3)
        if frequent_tools:
            parts.append(f"Frequently used tools: {', '.join(frequent_tools)}")

        if not parts:
            return ""

        return "\n".join(parts)
