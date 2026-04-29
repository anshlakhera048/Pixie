"""Lightweight circuit breaker for resilience.

Tracks consecutive failures for a named component and temporarily blocks
calls when the failure threshold is exceeded. Automatically recovers
after a cooldown period.

States:
  CLOSED  — normal operation, requests pass through
  OPEN    — too many failures, requests are blocked
  HALF_OPEN — recovery window, one probe request allowed
"""

from __future__ import annotations

import time
import threading
from enum import Enum
from typing import Any


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected by an open circuit breaker."""

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """Per-component circuit breaker.

    Usage:
        cb = CircuitBreaker("llm", failure_threshold=5, recovery_seconds=30)

        if not cb.allow_request():
            raise CircuitOpenError(...)

        try:
            result = await do_call()
            cb.record_success()
        except Exception:
            cb.record_failure()
            raise
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._evaluate_state()

    def _evaluate_state(self) -> CircuitState:
        """Determine current state (must be called under lock)."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._opened_at
            if elapsed >= self._recovery_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def allow_request(self) -> bool:
        """Return True if the request should proceed."""
        with self._lock:
            state = self._evaluate_state()
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.HALF_OPEN:
                return True  # Allow probe request
            return False

    def record_success(self) -> None:
        """Record a successful call — resets failure count."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call — may trip the breaker."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()

    def time_until_recovery(self) -> float:
        """Seconds remaining until the breaker may close (0 if closed)."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0
            elapsed = time.time() - self._opened_at
            remaining = self._recovery_seconds - elapsed
            return max(0.0, remaining)

    def reset(self) -> None:
        """Manually reset the breaker (for testing/admin)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def snapshot(self) -> dict[str, Any]:
        """Return breaker status as a dict."""
        with self._lock:
            state = self._evaluate_state()
            return {
                "name": self.name,
                "state": state.value,
                "failure_count": self._failure_count,
                "threshold": self._failure_threshold,
                "recovery_seconds": self._recovery_seconds,
            }


# ---------------------------------------------------------------------------
# Registry of named circuit breakers
# ---------------------------------------------------------------------------

_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_seconds: float = 30.0,
) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    with _registry_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name, failure_threshold, recovery_seconds
            )
        return _breakers[name]


def all_breakers() -> dict[str, CircuitBreaker]:
    """Return all registered breakers."""
    with _registry_lock:
        return dict(_breakers)
