"""Context manager — assembles the prompt sent to the LLM.

Combines the system prompt, recent conversation history, and retrieved
vector-memory context into a single message list that respects token
budget constraints.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from memory.memory import BaseVectorMemory, ConversationMemory

log = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 5  # last N user/assistant pairs kept in prompt
MAX_RETRIEVED_CHUNKS = 3

# Patterns that indicate the user is asking about screen content
_SCREEN_PATTERNS = re.compile(
    r"\b(what'?s on ?(my )?screen|screen|this page|this button|"
    r"what do you see|what'?s visible|read ?(my )?screen|"
    r"look at ?(my )?screen|on my monitor|what'?s displayed)\b",
    re.IGNORECASE,
)


class ContextManager:
    """Builds the LLM message list from conversation + vector memory."""

    def __init__(
        self,
        system_prompt: str,
        conversation: ConversationMemory,
        vector_memory: BaseVectorMemory | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._conversation = conversation
        self._vector_memory = vector_memory
        self._screen_context: str | None = None
        self._personalization_context: str | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_screen_context(self, text: str | None) -> None:
        """Inject screen context for the next build_messages call."""
        self._screen_context = text

    def set_personalization_context(self, text: str | None) -> None:
        """Inject personalization context (preferences, past actions, tool bias)."""
        self._personalization_context = text

    @staticmethod
    def is_screen_query(user_input: str) -> bool:
        """Return True if the user input appears to be asking about the screen."""
        return bool(_SCREEN_PATTERNS.search(user_input))

    def build_messages(self, user_input: str) -> list[dict[str, str]]:
        """Return the full message list ready for the LLM.

        Order:
          1. System prompt (with injected relevant context)
          2. Trimmed conversation history (last N messages)
        """
        relevant_context = self._retrieve_context(user_input)
        system_content = self._system_prompt

        if relevant_context:
            context_block = "\n\n# RELEVANT CONTEXT FROM MEMORY\n" + "\n---\n".join(relevant_context)
            system_content = system_content + context_block

        # Inject screen context if available
        if self._screen_context:
            system_content = system_content + "\n\n# SCREEN CONTEXT\n" + self._screen_context
            self._screen_context = None  # consume once

        # Inject personalization context if available
        if self._personalization_context:
            system_content = system_content + "\n\n# USER PERSONALIZATION\n" + self._personalization_context

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # Trim history to the most recent turns
        history = self._conversation.get_history()
        trimmed = self._trim_history(history)

        for entry in trimmed:
            role = entry["role"]
            if role == "tool":
                role = "user"
            messages.append({"role": role, "content": entry["content"]})

        return messages

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> list[str]:
        """Search vector memory for chunks relevant to the query."""
        if self._vector_memory is None:
            return []

        try:
            results = self._vector_memory.search(query, top_k=MAX_RETRIEVED_CHUNKS)
            return [r["text"] for r in results if r.get("text")]
        except Exception:
            log.warning("Vector memory search failed", exc_info=True)
            return []

    @staticmethod
    def _trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Keep the last MAX_HISTORY_TURNS interactions.

        An "interaction" is counted per message.  We keep the tail of the
        history so the most recent exchanges are always visible to the LLM.
        """
        # Each user message + assistant reply + possible tool message = ~2-3 entries
        # We keep 2 * MAX_HISTORY_TURNS entries to cover user+assistant pairs
        max_entries = MAX_HISTORY_TURNS * 2
        if len(history) <= max_entries:
            return history
        return history[-max_entries:]
