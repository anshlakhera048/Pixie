from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ConversationMemory:
    """Simple in-memory conversation history.

    Stores a flat list of ``{"role": ..., "content": ...}`` dicts.
    Roles: ``user``, ``assistant``, ``tool``.
    """

    def __init__(self) -> None:
        self._history: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[dict[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


# ---------------------------------------------------------------------------
# Vector memory interface (Phase 3 preparation)
# ---------------------------------------------------------------------------


class BaseVectorMemory(ABC):
    """Abstract interface for vector-based memory stores (RAG-ready)."""

    @abstractmethod
    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a text chunk with optional metadata."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant entries for a query."""


class VectorMemory(BaseVectorMemory):
    """In-memory placeholder vector store.

    Performs naive substring matching as a stand-in until a real vector
    backend (FAISS, ChromaDB, etc.) is integrated in Phase 3.
    """

    def __init__(self) -> None:
        self._store: list[dict[str, Any]] = []

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._store.append({"text": text, "metadata": metadata or {}})

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Naive relevance: return entries containing query terms."""
        query_lower = query.lower()
        scored: list[tuple[int, dict[str, Any]]] = []
        for entry in self._store:
            text_lower = entry["text"].lower()
            # Simple overlap scoring by word match count
            score = sum(1 for word in query_lower.split() if word in text_lower)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
