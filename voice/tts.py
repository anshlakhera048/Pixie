"""Text-to-Speech — streaming audio synthesis via Coqui TTS.

Generates speech from text and plays it through the system audio output.
Supports streaming playback (starts playing before full synthesis is done)
and can be interrupted mid-utterance.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from typing import AsyncGenerator

import numpy as np

from config import VoiceConfig

log = logging.getLogger(__name__)

# Output audio config
OUTPUT_SAMPLE_RATE = 22050  # Coqui TTS default output rate
PLAYBACK_CHUNK_SIZE = 4096  # samples per playback chunk


class TextToSpeech:
    """Streaming text-to-speech engine.

    Uses Coqui TTS for offline speech synthesis with a female voice.
    Audio is generated and played in chunks so the user hears output
    before the full response is synthesized.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._tts = None
        self._interrupted = False
        self._is_speaking = False
        self._playback_stream = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load the TTS model on first use."""
        if self._tts is not None:
            return

        if self._config.tts_backend == "coqui":
            try:
                from TTS.api import TTS

                self._tts = TTS(model_name=self._config.tts_model, progress_bar=False)
                log.info("Loaded Coqui TTS model: %s", self._config.tts_model)
            except ImportError:
                raise RuntimeError(
                    "Coqui TTS is not installed. Install it with: pip install TTS"
                )
        else:
            raise RuntimeError(f"Unknown TTS backend: {self._config.tts_backend}")

    def interrupt(self) -> None:
        """Stop playback immediately."""
        self._interrupted = True
        self._stop_playback()

    def _stop_playback(self) -> None:
        """Halt any active audio output."""
        if self._playback_stream is not None:
            try:
                self._playback_stream.stop()
                self._playback_stream.close()
            except Exception:
                pass
            self._playback_stream = None

    # ------------------------------------------------------------------
    # Streaming speech
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Synthesize and play a single text string."""
        self._ensure_model()
        self._interrupted = False
        self._is_speaking = True

        try:
            loop = asyncio.get_running_loop()
            # Synthesize in executor (blocking call)
            audio = await loop.run_in_executor(None, self._synthesize_sync, text)
            if audio is not None and not self._interrupted:
                await self._play_audio(audio)
        finally:
            self._is_speaking = False

    async def speak_stream(self, text_stream: AsyncGenerator[str, None]) -> None:
        """Synthesize and play text as it streams in.

        Accumulates text into sentence-sized chunks, synthesizes each chunk,
        and plays audio progressively.  This minimizes time-to-first-audio.
        """
        self._ensure_model()
        self._interrupted = False
        self._is_speaking = True

        try:
            buffer = ""
            sentence_delimiters = {".", "!", "?", "\n", ";"}

            async for chunk in text_stream:
                if self._interrupted:
                    return

                buffer += chunk

                # Check if we have a complete sentence to synthesize
                flush = False
                for delim in sentence_delimiters:
                    if delim in buffer:
                        flush = True
                        break

                if flush and len(buffer) > 10:
                    # Synthesize and play what we have
                    text_to_speak = buffer.strip()
                    buffer = ""

                    if text_to_speak:
                        loop = asyncio.get_running_loop()
                        audio = await loop.run_in_executor(
                            None, self._synthesize_sync, text_to_speak
                        )
                        if audio is not None and not self._interrupted:
                            await self._play_audio(audio)

            # Flush remaining buffer
            if buffer.strip() and not self._interrupted:
                loop = asyncio.get_running_loop()
                audio = await loop.run_in_executor(
                    None, self._synthesize_sync, buffer.strip()
                )
                if audio is not None and not self._interrupted:
                    await self._play_audio(audio)
        finally:
            self._is_speaking = False
            self._stop_playback()

    # ------------------------------------------------------------------
    # Audio synthesis
    # ------------------------------------------------------------------

    def _synthesize_sync(self, text: str) -> np.ndarray | None:
        """Synchronous TTS synthesis (runs in thread pool)."""
        if not text or self._tts is None:
            return None

        try:
            # Coqui TTS returns a list of float samples
            wav = self._tts.tts(
                text=text,
                speaker=self._config.tts_speaker or None,
            )
            return np.array(wav, dtype=np.float32)
        except Exception as exc:
            log.error("TTS synthesis failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Audio playback
    # ------------------------------------------------------------------

    async def _play_audio(self, audio: np.ndarray) -> None:
        """Play audio array through speakers without blocking the event loop."""
        import sounddevice as sd

        loop = asyncio.get_running_loop()

        # Play in chunks so we can check for interrupts
        total_samples = len(audio)
        offset = 0

        try:
            self._playback_stream = sd.OutputStream(
                samplerate=OUTPUT_SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            self._playback_stream.start()

            while offset < total_samples and not self._interrupted:
                end = min(offset + PLAYBACK_CHUNK_SIZE, total_samples)
                chunk = audio[offset:end]

                # Write to stream (non-blocking via executor)
                await loop.run_in_executor(
                    None, self._playback_stream.write, chunk.reshape(-1, 1)
                )
                offset = end

                # Yield control to allow interrupt checks
                await asyncio.sleep(0)

        except Exception as exc:
            log.error("Audio playback error: %s", exc)
        finally:
            self._stop_playback()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
