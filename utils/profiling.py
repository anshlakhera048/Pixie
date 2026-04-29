"""Profiling utilities — timing decorators and component timing.

Lightweight profiling that integrates with the metrics system.
No external dependencies.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import threading
from collections import defaultdict
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Timing storage
# ---------------------------------------------------------------------------

class _TimingStore:
    """Thread-safe storage for component timing data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._max_samples = 500

    def record(self, label: str, ms: float) -> None:
        """Record a timing sample."""
        with self._lock:
            samples = self._timings[label]
            samples.append(ms)
            if len(samples) > self._max_samples:
                # Keep recent half
                self._timings[label] = samples[-self._max_samples // 2:]

    def get_stats(self, label: str) -> dict[str, Any]:
        """Get timing statistics for a label."""
        with self._lock:
            samples = self._timings.get(label, [])
            if not samples:
                return {"count": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}
            return {
                "count": len(samples),
                "avg_ms": round(sum(samples) / len(samples), 2),
                "min_ms": round(min(samples), 2),
                "max_ms": round(max(samples), 2),
                "last_ms": round(samples[-1], 2),
            }

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Get timing stats for all labels."""
        with self._lock:
            labels = list(self._timings.keys())
        return {label: self.get_stats(label) for label in labels}

    def reset(self) -> None:
        """Clear all timings."""
        with self._lock:
            self._timings.clear()


# Global timing store
_timing_store = _TimingStore()


def get_timing_store() -> _TimingStore:
    """Return the global timing store."""
    return _timing_store


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def timed(label: str | None = None) -> Callable:
    """Decorator that records execution time of a sync function.

    Usage:
        @timed("llm.chat")
        def call_llm(messages):
            ...
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        effective_label = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _timing_store.record(effective_label, elapsed_ms)

        return wrapper
    return decorator


def async_timed(label: str | None = None) -> Callable:
    """Decorator that records execution time of an async function.

    Usage:
        @async_timed("llm.achat")
        async def call_llm(messages):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        effective_label = label or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _timing_store.record(effective_label, elapsed_ms)

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Context manager for inline timing
# ---------------------------------------------------------------------------

class timer:
    """Context manager for measuring a code block.

    Usage:
        with timer("my_operation") as t:
            do_something()
        print(t.elapsed_ms)
    """

    def __init__(self, label: str, *, record: bool = True) -> None:
        self.label = label
        self._record = record
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self._record:
            _timing_store.record(self.label, self.elapsed_ms)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def profile_report() -> str:
    """Generate a human-readable profile report of all timed components."""
    stats = _timing_store.all_stats()
    if not stats:
        return "  No profiling data collected.\n"

    lines = [
        f"\n  {'='*55}",
        f"  COMPONENT TIMING PROFILE",
        f"  {'='*55}",
        f"  {'Component':<30} {'Count':>6} {'Avg':>8} {'Min':>8} {'Max':>8}",
        f"  {'-'*55}",
    ]

    # Sort by total time (avg * count) descending
    sorted_stats = sorted(
        stats.items(),
        key=lambda x: x[1]["avg_ms"] * x[1]["count"],
        reverse=True,
    )

    for label, s in sorted_stats:
        lines.append(
            f"  {label:<30} {s['count']:>6} {s['avg_ms']:>7.1f} {s['min_ms']:>7.1f} {s['max_ms']:>7.1f}"
        )

    lines.append(f"  {'='*55}\n")
    return "\n".join(lines)
