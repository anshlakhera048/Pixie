"""Speech-to-Text — streaming transcription via faster-whisper.

Captures microphone input in a background thread and yields partial
transcripts as they become available.  Designed to integrate with the
async runtime without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import AsyncGenerator

import numpy as np

from config import VoiceConfig

log = logging.getLogger(__name__)

# Silence detection constants
ENERGY_THRESHOLD = 0.01  # RMS energy below this = silence


class SpeechToText:
    """Streaming speech-to-text engine.

    Uses faster-whisper for local transcription with partial result streaming.
    Audio is captured via sounddevice in a background thread and fed to the
    transcriber through an asyncio queue.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None
        self._stream = None
        self._audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=100)
        self._is_listening = False
        self._interrupted = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy-load the whisper model on first use."""
        if self._model is not None:
            return

        if self._config.stt_backend == "faster-whisper":
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self._config.stt_model,
                    device=self._config.stt_device,
                    compute_type="int8",
                )
                log.info(
                    "Loaded faster-whisper model: %s on %s",
                    self._config.stt_model,
                    self._config.stt_device,
                )
            except ImportError:
                raise RuntimeError(
                    "faster-whisper is not installed. "
                    "Install it with: pip install faster-whisper"
                )
        else:
            raise RuntimeError(f"Unknown STT backend: {self._config.stt_backend}")

    def interrupt(self) -> None:
        """Signal to stop listening immediately."""
        self._interrupted = True
        # Push sentinel to unblock queue
        try:
            self._audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback — runs in audio thread, pushes to queue."""
        if status:
            log.warning("Audio input status: %s", status)
        # Copy the data since the buffer is reused
        audio_chunk = indata[:, 0].copy().astype(np.float32)
        try:
            self._audio_queue.put_nowait(audio_chunk)
        except asyncio.QueueFull:
            pass  # Drop oldest frame rather than blocking audio thread

    def _start_audio_stream(self) -> None:
        """Open the microphone input stream."""
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype="float32",
            blocksize=int(self._config.sample_rate * self._config.chunk_duration_ms / 1000),
            callback=self._audio_callback,
        )
        self._stream.start()

    def _stop_audio_stream(self) -> None:
        """Close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    async def stream_transcribe(self) -> AsyncGenerator[str, None]:
        """Capture audio and yield partial transcripts.

        Yields text fragments as the user speaks.  Returns when silence
        is detected after speech, or on interrupt.

        The caller should iterate this generator to get incremental text,
        then use the accumulated result as the full utterance.
        """
        self._ensure_model()
        self._interrupted = False
        self._is_listening = True

        # Drain any stale audio from the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._start_audio_stream()
        log.info("STT: listening...")

        try:
            audio_buffer: list[np.ndarray] = []
            silence_start: float | None = None
            has_speech = False
            max_samples = self._config.sample_rate * self._config.audio_buffer_max_seconds
            total_samples = 0

            while not self._interrupted:
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if chunk is None:  # Sentinel for interrupt
                    break

                audio_buffer.append(chunk)
                total_samples += len(chunk)

                # Check energy level
                rms = np.sqrt(np.mean(chunk ** 2))

                if rms > ENERGY_THRESHOLD:
                    has_speech = True
                    silence_start = None
                elif has_speech:
                    # Track silence duration
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) * 1000 > self._config.silence_threshold_ms:
                        # End of utterance detected
                        break

                # Safety: cap recording length
                if total_samples >= max_samples:
                    break

                # Periodic partial transcription (every ~0.5s of audio)
                chunk_threshold = int(self._config.sample_rate * 0.5)
                if total_samples > 0 and total_samples % chunk_threshold < len(chunk):
                    partial = await self._transcribe_buffer(audio_buffer)
                    if partial:
                        yield partial

            # Final transcription of complete utterance
            if audio_buffer and has_speech:
                final = await self._transcribe_buffer(audio_buffer)
                if final:
                    yield final

        finally:
            self._stop_audio_stream()
            self._is_listening = False
            log.info("STT: stopped listening")

    async def _transcribe_buffer(self, chunks: list[np.ndarray]) -> str:
        """Run transcription on accumulated audio buffer."""
        if not chunks:
            return ""

        audio = np.concatenate(chunks)
        loop = asyncio.get_running_loop()

        # Run whisper in executor to avoid blocking event loop
        text = await loop.run_in_executor(None, self._transcribe_sync, audio)
        return text.strip()

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription call (runs in thread pool)."""
        if self._model is None:
            return ""

        segments, _info = self._model.transcribe(
            audio,
            language=self._config.stt_language,
            beam_size=1,  # Fast, low-latency decoding
            vad_filter=True,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        return " ".join(text_parts)

    @property
    def is_listening(self) -> bool:
        return self._is_listening
