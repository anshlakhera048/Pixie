"""Structured logging for Pixie.

Emits JSON-formatted log lines for observability without external dependencies.
Uses QueueHandler/QueueListener to ensure logging never blocks the async event loop.

Supports dual output:
  - stderr (always)
  - JSONL file (configurable via PIXIE_LOG_FILE env var)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any

# Module-level listener reference for cleanup
_log_listener: logging.handlers.QueueListener | None = None


class StructuredFormatter(logging.Formatter):
    """Formats log records as single-line JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach extra structured fields if present
        if hasattr(record, "data"):
            entry["data"] = record.data  # type: ignore[attr-defined]
        if hasattr(record, "request_id"):
            entry["request_id"] = record.request_id  # type: ignore[attr-defined]
        if hasattr(record, "latency_ms"):
            entry["latency_ms"] = record.latency_ms  # type: ignore[attr-defined]
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        return json.dumps(entry, default=str)


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure root logger with async-safe queue-based logging.

    Outputs to stderr always. If log_file is provided (or PIXIE_LOG_FILE env),
    also writes JSONL to a rotating file.

    Uses QueueHandler on the root logger so that log calls never block.
    A QueueListener in a background thread drains the queue and writes
    to the actual handlers.
    """
    global _log_listener

    # Stop existing listener if re-initializing
    if _log_listener is not None:
        _log_listener.stop()
        _log_listener = None

    handlers: list[logging.Handler] = []

    # Stderr handler (always present)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(StructuredFormatter())
    handlers.append(stderr_handler)

    # File handler (optional)
    file_path = log_file or os.environ.get("PIXIE_LOG_FILE", "")
    if file_path:
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(StructuredFormatter())
        handlers.append(file_handler)

    # Queue-based non-blocking handler
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=2048)
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(queue_handler)

    # Start listener thread that drains queue → handlers
    _log_listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    _log_listener.start()


def shutdown_logging() -> None:
    """Cleanly stop the log listener. Call on application exit."""
    global _log_listener
    if _log_listener is not None:
        _log_listener.stop()
        _log_listener = None


def log_event(logger: logging.Logger, level: int, message: str, **data: Any) -> None:
    """Emit a structured log event with arbitrary key-value data."""
    extra = {"data": data} if data else {}
    logger.log(level, message, extra=extra)


def log_request(
    logger: logging.Logger,
    *,
    request_id: str,
    method: str,
    path: str,
    status: int,
    latency_ms: float,
    client: str = "",
    error: str | None = None,
) -> None:
    """Emit a structured request log line with tracing fields."""
    extra: dict[str, Any] = {
        "data": {
            "method": method,
            "path": path,
            "status": status,
            "client": client,
        },
        "request_id": request_id,
        "latency_ms": round(latency_ms, 2),
    }
    if error:
        extra["data"]["error"] = error
    level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
    logger.log(level, f"{method} {path} {status} {latency_ms:.1f}ms", extra=extra)
