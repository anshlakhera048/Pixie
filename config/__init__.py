"""Pixie configuration layer.

Centralizes runtime settings for voice, STT, TTS, and general behavior.
All values can be overridden via environment variables prefixed with PIXIE_.

This package also contains config.settings for production deployment settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


@dataclass
class VoiceConfig:
    """Voice subsystem configuration."""

    enabled: bool = field(default_factory=lambda: _env_bool("PIXIE_VOICE_ENABLED", False))

    # STT settings
    stt_backend: str = field(
        default_factory=lambda: os.environ.get("PIXIE_STT_BACKEND", "faster-whisper")
    )
    stt_model: str = field(
        default_factory=lambda: os.environ.get("PIXIE_STT_MODEL", "base.en")
    )
    stt_device: str = field(
        default_factory=lambda: os.environ.get("PIXIE_STT_DEVICE", "cpu")
    )
    stt_language: str = field(
        default_factory=lambda: os.environ.get("PIXIE_STT_LANGUAGE", "en")
    )

    # TTS settings
    tts_backend: str = field(
        default_factory=lambda: os.environ.get("PIXIE_TTS_BACKEND", "coqui")
    )
    tts_model: str = field(
        default_factory=lambda: os.environ.get(
            "PIXIE_TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"
        )
    )
    tts_speaker: str = field(
        default_factory=lambda: os.environ.get("PIXIE_TTS_SPEAKER", "")
    )

    # Wake word settings
    wake_word: str = field(
        default_factory=lambda: os.environ.get("PIXIE_WAKE_WORD", "hey pixie")
    )
    wake_word_backend: str = field(
        default_factory=lambda: os.environ.get("PIXIE_WAKE_WORD_BACKEND", "keyword")
    )

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30  # ms per audio chunk for VAD/wake detection
    silence_threshold_ms: int = 800  # ms of silence before end-of-utterance
    audio_buffer_max_seconds: int = 30  # max recording length


@dataclass
class APIConfig:
    """API server configuration."""

    enabled: bool = field(default_factory=lambda: _env_bool("PIXIE_API_ENABLED", False))
    host: str = field(default_factory=lambda: os.environ.get("PIXIE_API_HOST", "127.0.0.1"))
    port: int = field(
        default_factory=lambda: int(os.environ.get("PIXIE_API_PORT", "8000"))
    )
    max_concurrent: int = field(
        default_factory=lambda: int(os.environ.get("PIXIE_MAX_CONCURRENT", "10"))
    )
    request_timeout: float = field(
        default_factory=lambda: float(os.environ.get("PIXIE_REQUEST_TIMEOUT", "60"))
    )
    rate_limit: int = field(
        default_factory=lambda: int(os.environ.get("PIXIE_RATE_LIMIT", "60"))
    )


@dataclass
class PixieConfig:
    """Top-level application configuration."""

    mock: bool = field(default_factory=lambda: _env_bool("PIXIE_MOCK", False))
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log_level: str = field(
        default_factory=lambda: os.environ.get("PIXIE_LOG_LEVEL", "INFO")
    )


# Singleton instance for the application
_config: PixieConfig | None = None


def get_config() -> PixieConfig:
    """Return the application config singleton."""
    global _config
    if _config is None:
        _config = PixieConfig()
    return _config


def reset_config() -> None:
    """Reset config (useful for testing)."""
    global _config
    _config = None
