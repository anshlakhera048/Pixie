"""Production settings — centralized configuration with validation.

Loads from environment variables with sensible defaults.
Validates required fields on startup.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import FrozenSet


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    return int(val) if val else default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    return float(val) if val else default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def _parse_api_keys() -> FrozenSet[str]:
    """Parse comma-separated API keys from env.

    If no keys are set, generates a random one and prints it (dev mode).
    """
    raw = os.environ.get("PIXIE_API_KEYS", "").strip()
    if raw:
        keys = frozenset(k.strip() for k in raw.split(",") if k.strip())
        if keys:
            return keys
    return frozenset()


@dataclass(frozen=True)
class Settings:
    """Immutable production settings. Loaded once at startup."""

    # --- Security ---
    api_keys: FrozenSet[str] = field(default_factory=_parse_api_keys)
    auth_enabled: bool = field(default_factory=lambda: _env_bool("PIXIE_AUTH_ENABLED", True))

    # --- Server ---
    host: str = field(default_factory=lambda: _env("PIXIE_API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PIXIE_API_PORT", 8000))

    # --- Concurrency ---
    max_concurrent_tasks: int = field(
        default_factory=lambda: _env_int("PIXIE_MAX_CONCURRENT", 10)
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: _env_float("PIXIE_REQUEST_TIMEOUT", 60.0)
    )

    # --- Rate limiting (per API key) ---
    rate_limit_rpm: int = field(
        default_factory=lambda: _env_int("PIXIE_RATE_LIMIT_RPM", 60)
    )

    # --- Retry / resilience ---
    llm_timeout_seconds: float = field(
        default_factory=lambda: _env_float("PIXIE_LLM_TIMEOUT", 30.0)
    )
    tool_timeout_seconds: float = field(
        default_factory=lambda: _env_float("PIXIE_TOOL_TIMEOUT", 15.0)
    )
    max_retries: int = field(
        default_factory=lambda: _env_int("PIXIE_MAX_RETRIES", 2)
    )

    # --- Circuit breaker ---
    cb_failure_threshold: int = field(
        default_factory=lambda: _env_int("PIXIE_CB_FAILURE_THRESHOLD", 5)
    )
    cb_recovery_seconds: float = field(
        default_factory=lambda: _env_float("PIXIE_CB_RECOVERY_SECONDS", 30.0)
    )

    # --- Logging ---
    log_level: str = field(default_factory=lambda: _env("PIXIE_LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: _env("PIXIE_LOG_FILE", "logs/pixie.jsonl"))

    # --- Mock mode ---
    mock: bool = field(default_factory=lambda: _env_bool("PIXIE_MOCK", False))

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []
        if self.auth_enabled and not self.api_keys:
            errors.append(
                "PIXIE_API_KEYS not set but auth is enabled. "
                "Set PIXIE_API_KEYS=<key1>,<key2> or PIXIE_AUTH_ENABLED=0 to disable."
            )
        if self.max_concurrent_tasks < 1:
            errors.append("PIXIE_MAX_CONCURRENT must be >= 1")
        if self.request_timeout_seconds <= 0:
            errors.append("PIXIE_REQUEST_TIMEOUT must be > 0")
        if self.rate_limit_rpm < 1:
            errors.append("PIXIE_RATE_LIMIT_RPM must be >= 1")
        return errors


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the global Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (for testing)."""
    global _settings
    _settings = None
