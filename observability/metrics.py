"""Observability metrics — request latency, tool usage, error rates.

Lightweight in-memory metrics collection. No external dependencies.
Exposes data as JSON for the /metrics endpoint.
Includes latency percentiles (p50, p90, p95, p99).
"""

from __future__ import annotations

import bisect
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _LatencyBucket:
    """Accumulates latency samples with percentile support."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    _samples: list[float] = field(default_factory=list)
    # Cap samples to prevent unbounded memory (reservoir sampling-like trim)
    _max_samples: int = 2000

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)
        # Insert sorted for efficient percentile calculation
        bisect.insort(self._samples, ms)
        # Trim oldest entries if over cap (keeps most recent distribution)
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples:]

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0-100) from stored samples."""
        if not self._samples:
            return 0.0
        idx = int(len(self._samples) * p / 100.0)
        idx = min(idx, len(self._samples) - 1)
        return self._samples[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else 0,
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.percentile(50), 2),
            "p90_ms": round(self.percentile(90), 2),
            "p95_ms": round(self.percentile(95), 2),
            "p99_ms": round(self.percentile(99), 2),
            "total_ms": round(self.total_ms, 2),
        }


class Metrics:
    """Thread-safe in-memory metrics collector.

    Tracks:
    - Request latency per endpoint
    - Tool usage frequency
    - Tool success/failure rates
    - Error counts
    - Active session count
    - Concurrent task gauge
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Latency per endpoint
        self._latencies: dict[str, _LatencyBucket] = defaultdict(_LatencyBucket)

        # Counters
        self._request_count: int = 0
        self._error_count: int = 0
        self._tool_calls: dict[str, int] = defaultdict(int)
        self._tool_successes: dict[str, int] = defaultdict(int)
        self._tool_failures: dict[str, int] = defaultdict(int)

        # Gauges
        self._active_tasks: int = 0
        self._peak_concurrent: int = 0
        self._active_sessions: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_request(self, endpoint: str, latency_ms: float) -> None:
        """Record a completed request."""
        with self._lock:
            self._request_count += 1
            self._latencies[endpoint].record(latency_ms)

    def record_error(self, endpoint: str) -> None:
        """Record a request error."""
        with self._lock:
            self._error_count += 1

    def record_tool_call(self, tool_name: str, success: bool) -> None:
        """Record a tool execution."""
        with self._lock:
            self._tool_calls[tool_name] += 1
            if success:
                self._tool_successes[tool_name] += 1
            else:
                self._tool_failures[tool_name] += 1

    def task_started(self) -> None:
        """Increment active task gauge."""
        with self._lock:
            self._active_tasks += 1
            self._peak_concurrent = max(self._peak_concurrent, self._active_tasks)

    def task_completed(self) -> None:
        """Decrement active task gauge."""
        with self._lock:
            self._active_tasks = max(0, self._active_tasks - 1)

    def set_active_sessions(self, count: int) -> None:
        """Update active session gauge."""
        with self._lock:
            self._active_sessions = count

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics snapshot."""
        with self._lock:
            uptime = time.time() - self._start_time

            # Tool stats
            tool_stats = {}
            for name in self._tool_calls:
                calls = self._tool_calls[name]
                successes = self._tool_successes.get(name, 0)
                failures = self._tool_failures.get(name, 0)
                tool_stats[name] = {
                    "calls": calls,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": round(successes / calls, 3) if calls else 0,
                }

            data: dict[str, Any] = {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self._request_count,
                "total_errors": self._error_count,
                "error_rate": round(
                    self._error_count / self._request_count, 3
                ) if self._request_count else 0,
                "active_tasks": self._active_tasks,
                "peak_concurrent_tasks": self._peak_concurrent,
                "active_sessions": self._active_sessions,
                "latencies": {
                    endpoint: bucket.to_dict()
                    for endpoint, bucket in self._latencies.items()
                },
                "tools": tool_stats,
            }

        # Append cache stats (outside lock — caches have their own locks)
        try:
            from cache.cache import get_llm_cache, get_tool_cache
            data["cache"] = {
                "llm": get_llm_cache().stats(),
                "tools": get_tool_cache().stats(),
            }
        except Exception:
            pass

        # Append profiling data
        try:
            from utils.profiling import get_timing_store
            data["profiling"] = get_timing_store().all_stats()
        except Exception:
            pass

        return data

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._start_time = time.time()
            self._request_count = 0
            self._error_count = 0
            self._latencies.clear()
            self._tool_calls.clear()
            self._tool_successes.clear()
            self._tool_failures.clear()
            self._active_tasks = 0
            self._peak_concurrent = 0
            self._active_sessions = 0


# Global metrics singleton
_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """Return the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics
