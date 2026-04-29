"""Lightweight TTL cache with LRU eviction.

Provides async-safe caching for LLM responses and tool results.
No external dependencies — pure Python with threading locks.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any


class Cache:
    """Thread-safe LRU cache with per-entry TTL.

    Args:
        max_size: Maximum number of entries before LRU eviction.
        default_ttl: Default time-to-live in seconds (0 = no expiry).
    """

    def __init__(self, max_size: int = 256, default_ttl: float = 300.0) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value by key. Returns None on miss or expiry."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value with optional per-entry TTL."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._store:
                # Update existing
                self._store[key] = _CacheEntry(value, effective_ttl)
                self._store.move_to_end(key)
            else:
                # Evict LRU if at capacity
                if len(self._store) >= self._max_size:
                    self._store.popitem(last=False)
                self._store[key] = _CacheEntry(value, effective_ttl)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries (including possibly expired)."""
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self.hit_rate, 3),
                "default_ttl_seconds": self._default_ttl,
            }

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0


class _CacheEntry:
    """Single cache entry with expiration."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float) -> None:
        self.value = value
        self.expires_at = time.time() + ttl if ttl > 0 else float("inf")

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


# ---------------------------------------------------------------------------
# Key generation helpers
# ---------------------------------------------------------------------------


def make_cache_key(*parts: Any) -> str:
    """Generate a deterministic cache key from arbitrary arguments."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Global cache instances
# ---------------------------------------------------------------------------

_llm_cache: Cache | None = None
_tool_cache: Cache | None = None


def get_llm_cache() -> Cache:
    """LLM response cache (larger TTL, moderate size)."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = Cache(max_size=512, default_ttl=600.0)
    return _llm_cache


def get_tool_cache() -> Cache:
    """Tool result cache (shorter TTL, smaller size)."""
    global _tool_cache
    if _tool_cache is None:
        _tool_cache = Cache(max_size=128, default_ttl=120.0)
    return _tool_cache
