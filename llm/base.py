from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a list of messages and return the raw assistant response text."""

    async def achat(self, messages: list[dict[str, str]]) -> str:
        """Async version of chat. Default falls back to sync."""
        return self.chat(messages)

    async def stream_generate(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream partial tokens. Default simulates chunking from full response."""
        full = await self.achat(messages)
        # Simulate streaming by yielding word-by-word
        words = full.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word
