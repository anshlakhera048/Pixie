"""Notifications — console and optional TTS alerts.

Lightweight notification system that prints to console and
optionally speaks alerts via TTS when voice mode is active.
"""

from __future__ import annotations

import logging
from typing import Callable, Coroutine

log = logging.getLogger(__name__)


class Notifier:
    """Sends notifications to the user via console and optional TTS."""

    def __init__(self) -> None:
        self._tts_fn: Callable[[str], Coroutine] | None = None

    def set_tts(self, tts_fn: Callable[[str], Coroutine] | None) -> None:
        """Register an async TTS function for spoken alerts."""
        self._tts_fn = tts_fn

    async def notify(self, message: str, *, speak: bool = True) -> None:
        """Send a notification.

        Args:
            message: The notification text.
            speak: If True and TTS is available, also speak the message.
        """
        # Console notification with visual indicator
        print(f"\n  🔔 [notification] {message}\n")

        # TTS alert if available and requested
        if speak and self._tts_fn:
            try:
                await self._tts_fn(message)
            except Exception:
                log.debug("TTS notification failed", exc_info=True)

    def notify_sync(self, message: str) -> None:
        """Synchronous console-only notification."""
        print(f"\n  🔔 [notification] {message}\n")
