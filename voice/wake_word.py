"""Wake word detection — listens for activation phrase.

Continuously monitors the microphone for the wake word (default: "hey pixie").
Uses a lightweight keyword-spotting approach:
  1. Primary: Porcupine (if pvporcupine is installed + access key available)
  2. Fallback: Energy-based detection + fast whisper transcription of short segments

The fallback approach captures small audio windows when energy is detected,
transcribes them with whisper, and checks for the wake phrase.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from config import VoiceConfig

log = logging.getLogger(__name__)

# Energy threshold for the fallback detector to consider a segment worth transcribing
WAKE_ENERGY_THRESHOLD = 0.015
# Duration of audio window for wake word detection (seconds)
WAKE_WINDOW_SECONDS = 2.0


class WakeWordDetector:
    """Listens for the activation phrase to trigger the STT pipeline.

    Runs continuously in the background.  When the wake word is detected,
    the `listen()` coroutine returns, signalling the caller to begin
    full speech recognition.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._interrupted = False
        self._stream = None
        self._audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=50)
        self._whisper_model = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def listen(self) -> bool:
        """Block until wake word is detected or interrupted.

        Returns True if wake word was detected, False if interrupted.
        """
        self._interrupted = False

        backend = self._config.wake_word_backend

        if backend == "porcupine":
            return await self._listen_porcupine()

        # Default: keyword spotting via energy + whisper
        return await self._listen_keyword()

    def interrupt(self) -> None:
        """Stop wake word detection."""
        self._interrupted = True
        try:
            self._audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # Keyword fallback (energy + whisper transcription)
    # ------------------------------------------------------------------

    def _ensure_whisper(self) -> None:
        """Lazy-load a tiny whisper model for wake word detection."""
        if self._whisper_model is not None:
            return

        try:
            from faster_whisper import WhisperModel

            # Use the smallest model for speed
            self._whisper_model = WhisperModel(
                "tiny.en", device=self._config.stt_device, compute_type="int8"
            )
            log.info("Wake word detector: loaded tiny.en whisper model")
        except ImportError:
            raise RuntimeError(
                "faster-whisper is required for wake word detection. "
                "Install it with: pip install faster-whisper"
            )

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback for wake word detection."""
        if status:
            log.debug("Wake audio status: %s", status)
        chunk = indata[:, 0].copy().astype(np.float32)
        try:
            self._audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    async def _listen_keyword(self) -> bool:
        """Energy-gated keyword detection using whisper transcription."""
        import sounddevice as sd

        self._ensure_whisper()

        chunk_samples = int(self._config.sample_rate * self._config.chunk_duration_ms / 1000)
        window_samples = int(self._config.sample_rate * WAKE_WINDOW_SECONDS)

        # Drain queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype="float32",
            blocksize=chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()
        log.info("Wake word detector: listening for '%s'...", self._config.wake_word)

        try:
            audio_buffer: list[np.ndarray] = []
            buffer_samples = 0
            energy_detected_at: float | None = None

            while not self._interrupted:
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if chunk is None:
                    return False

                rms = np.sqrt(np.mean(chunk ** 2))

                if rms > WAKE_ENERGY_THRESHOLD:
                    if energy_detected_at is None:
                        energy_detected_at = time.time()
                    audio_buffer.append(chunk)
                    buffer_samples += len(chunk)
                elif energy_detected_at is not None:
                    audio_buffer.append(chunk)
                    buffer_samples += len(chunk)

                    # Check if we have enough silence after speech
                    silence_duration = time.time() - energy_detected_at
                    if buffer_samples >= window_samples or silence_duration > WAKE_WINDOW_SECONDS:
                        # Transcribe the window
                        detected = await self._check_wake_word(audio_buffer)
                        if detected:
                            return True

                        # Reset buffer
                        audio_buffer.clear()
                        buffer_samples = 0
                        energy_detected_at = None

                # Prevent buffer from growing too large
                if buffer_samples > window_samples * 2:
                    audio_buffer.clear()
                    buffer_samples = 0
                    energy_detected_at = None

        finally:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

        return False

    async def _check_wake_word(self, chunks: list[np.ndarray]) -> bool:
        """Transcribe audio and check if it contains the wake phrase."""
        if not chunks:
            return False

        audio = np.concatenate(chunks)
        loop = asyncio.get_running_loop()

        text = await loop.run_in_executor(None, self._transcribe_short, audio)
        text_lower = text.lower().strip()

        wake_phrase = self._config.wake_word.lower()
        # Fuzzy match: check if wake phrase words appear in transcription
        wake_words = wake_phrase.split()
        if all(w in text_lower for w in wake_words):
            log.info("Wake word detected: '%s' in '%s'", wake_phrase, text_lower)
            return True

        return False

    def _transcribe_short(self, audio: np.ndarray) -> str:
        """Quick transcription of a short audio segment."""
        if self._whisper_model is None:
            return ""

        segments, _ = self._whisper_model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(seg.text for seg in segments)

    # ------------------------------------------------------------------
    # Porcupine backend (premium, if available)
    # ------------------------------------------------------------------

    async def _listen_porcupine(self) -> bool:
        """Use Porcupine for wake word detection (requires access key)."""
        try:
            import pvporcupine
            import sounddevice as sd
        except ImportError:
            log.warning("pvporcupine not available, falling back to keyword detection")
            return await self._listen_keyword()

        import os

        access_key = os.environ.get("PORCUPINE_ACCESS_KEY")
        if not access_key:
            log.warning("PORCUPINE_ACCESS_KEY not set, falling back to keyword detection")
            return await self._listen_keyword()

        try:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=["hey google"],  # Closest built-in; custom requires .ppn file
                sensitivities=[0.7],
            )
        except Exception as exc:
            log.warning("Porcupine init failed (%s), falling back", exc)
            return await self._listen_keyword()

        frame_length = porcupine.frame_length
        audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=50)

        def callback(indata, frames, time_info, status):
            chunk = (indata[:, 0] * 32767).astype(np.int16)
            try:
                audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

        stream = sd.InputStream(
            samplerate=porcupine.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_length,
            callback=callback,
        )
        stream.start()

        try:
            while not self._interrupted:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if chunk is None:
                    return False

                keyword_index = porcupine.process(chunk)
                if keyword_index >= 0:
                    log.info("Porcupine wake word detected")
                    return True
        finally:
            stream.stop()
            stream.close()
            porcupine.delete()

        return False
