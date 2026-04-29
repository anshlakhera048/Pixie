"""Timeout + retry wrapper for async calls.

Provides a decorator/utility for wrapping LLM and tool calls with:
- Configurable timeout per call
- Retry with exponential backoff (max 2 retries by default)
- Circuit breaker integration
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, TypeVar

from utils.circuit_breaker import CircuitBreaker, CircuitOpenError, get_breaker

log = logging.getLogger(__name__)

T = TypeVar("T")


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    timeout: float = 30.0,
    max_retries: int = 2,
    breaker: CircuitBreaker | None = None,
    label: str = "call",
    **kwargs: Any,
) -> T:
    """Execute an async function with timeout, retry, and circuit breaker.

    Args:
        fn: Async callable to execute.
        timeout: Per-attempt timeout in seconds.
        max_retries: Maximum retry attempts (total calls = 1 + max_retries).
        breaker: Optional circuit breaker to check/update.
        label: Label for logging.

    Returns:
        The result of fn(*args, **kwargs).

    Raises:
        CircuitOpenError: If the circuit breaker is open.
        asyncio.TimeoutError: If all attempts time out.
        Exception: The last exception after exhausting retries.
    """
    if breaker and not breaker.allow_request():
        raise CircuitOpenError(breaker.name, breaker.time_until_recovery())

    last_exc: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            result = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
            if breaker:
                breaker.record_success()
            return result
        except asyncio.TimeoutError:
            last_exc = asyncio.TimeoutError(  # type: ignore[assignment]
                f"{label} timed out (attempt {attempt + 1}/{1 + max_retries})"
            )
            log.warning("%s timeout (attempt %d/%d)", label, attempt + 1, 1 + max_retries)
        except asyncio.CancelledError:
            raise  # Never retry cancellations
        except Exception as exc:
            last_exc = exc
            log.warning(
                "%s failed (attempt %d/%d): %s",
                label, attempt + 1, 1 + max_retries, exc,
            )

        # Record failure for circuit breaker
        if breaker:
            breaker.record_failure()

        # Backoff before retry (skip on last attempt)
        if attempt < max_retries:
            backoff = min(2 ** attempt * 0.5, 5.0)
            await asyncio.sleep(backoff)

    # All attempts exhausted
    raise last_exc  # type: ignore[misc]


def retriable(
    timeout: float = 30.0,
    max_retries: int = 2,
    breaker_name: str | None = None,
) -> Callable:
    """Decorator version of with_retry for async functions.

    Usage:
        @retriable(timeout=15, max_retries=2, breaker_name="llm")
        async def call_llm(prompt: str) -> str:
            ...
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        breaker = get_breaker(breaker_name) if breaker_name else None

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await with_retry(
                fn, *args,
                timeout=timeout,
                max_retries=max_retries,
                breaker=breaker,
                label=fn.__qualname__,
                **kwargs,
            )

        return wrapper

    return decorator
