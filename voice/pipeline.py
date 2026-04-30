"""Voice pipeline — coordinates wake word, STT, agent, and TTS.

Orchestrates the full voice interaction loop:
  1. Wait for wake word
  2. Capture and transcribe speech (STT)
  3. Send transcript to agent
  4. Stream agent response through TTS
  5. Handle interruption at any stage

This module ties together the voice components without modifying
the core async runtime — it operates as a parallel mode alongside
the text CLI.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from config import VoiceConfig
from voice.stt import SpeechToText
from voice.tts import TextToSpeech
from voice.wake_word import WakeWordDetector

log = logging.getLogger(__name__)


class VoicePipeline:
    """Full voice interaction pipeline.

    Manages the lifecycle of wake-word → STT → agent → TTS with
    interrupt-at-any-point semantics.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._stt = SpeechToText(config)
        self._tts = TextToSpeech(config)
        self._wake = WakeWordDetector(config)
        self._running = False
        self._interrupted = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_speaking(self) -> bool:
        return self._tts.is_speaking

    @property
    def is_listening(self) -> bool:
        return self._stt.is_listening

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Activate the voice pipeline."""
        self._running = True
        self._interrupted = False
        log.info("Voice pipeline started")

    def stop(self) -> None:
        """Deactivate the voice pipeline."""
        self._running = False
        self.interrupt()
        log.info("Voice pipeline stopped")

    def interrupt(self) -> None:
        """Interrupt all voice components immediately.

        Stops TTS playback, STT recording, and wake word detection.
        Used when user speaks during Pixie's response (barge-in).
        """
        self._interrupted = True
        self._tts.interrupt()
        self._stt.interrupt()
        self._wake.interrupt()

    def reset_interrupt(self) -> None:
        """Clear interrupt state for next interaction cycle."""
        self._interrupted = False

    # ------------------------------------------------------------------
    # Main voice loop
    # ------------------------------------------------------------------

    async def run_loop(self, agent_callback) -> None:
        """Run the continuous voice interaction loop.

        Args:
            agent_callback: An async callable that takes user text and returns
                           an AsyncGenerator yielding response chunks.
                           Signature: async def cb(text) -> AsyncGenerator[str, None]
        """
        self._running = True
        log.info("Voice loop started — say '%s' to activate", self._config.wake_word)

        try:
            while self._running:
                self.reset_interrupt()

                # Phase 1: Wait for wake word
                print("\n  [listening for wake word...]\n", flush=True)
                detected = await self._wake.listen()
                if not detected or not self._running:
                    continue

                # Phase 2: Acknowledge and start STT
                print("  [wake word detected — listening...]\n", flush=True)

                # Phase 3: Capture user speech
                transcript = await self._capture_speech()
                if not transcript or not self._running:
                    continue

                print(f"  you> {transcript}\n", flush=True)

                # Phase 4: Run agent and stream response through TTS
                print("  pixie> ", end="", flush=True)
                await self._respond(transcript, agent_callback)
                print("\n", flush=True)

        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            log.info("Voice loop ended")

    # ------------------------------------------------------------------
    # Single interaction (without wake word)
    # ------------------------------------------------------------------

    async def single_interaction(self, agent_callback) -> str | None:
        """Run one voice interaction cycle: listen → process → speak.

        Returns the transcript or None if interrupted/nothing detected.
        """
        self.reset_interrupt()

        transcript = await self._capture_speech()
        if not transcript:
            return None

        await self._respond(transcript, agent_callback)
        return transcript

    # ------------------------------------------------------------------
    # Internal phases
    # ------------------------------------------------------------------

    async def _capture_speech(self) -> str:
        """Capture and transcribe user speech.

        Returns the full transcription text.
        """
        parts: list[str] = []
        async for partial in self._stt.stream_transcribe():
            if self._interrupted:
                break
            parts.append(partial)

        # The last partial is the most complete transcription
        return parts[-1] if parts else ""

    async def _respond(self, text: str, agent_callback) -> None:
        """Send text to agent and stream response to TTS.

        Handles barge-in: if the user speaks while TTS is playing,
        the pipeline interrupts TTS and switches to listening.
        """
        # Create a monitoring task that watches for user interruption
        interrupt_task = asyncio.create_task(self._monitor_barge_in())

        try:
            response_stream = agent_callback(text)

            # Tee the stream: print to console AND feed to TTS
            tee_stream = self._tee_print_stream(response_stream)
            await self._tts.speak_stream(tee_stream)
        except asyncio.CancelledError:
            pass
        finally:
            interrupt_task.cancel()
            try:
                await interrupt_task
            except asyncio.CancelledError:
                pass

    async def _tee_print_stream(
        self, stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Yield chunks from stream while also printing them."""
        async for chunk in stream:
            if self._interrupted:
                return
            print(chunk, end="", flush=True)
            yield chunk

    async def _monitor_barge_in(self) -> None:
        """Monitor microphone for barge-in (user speaking during TTS).

        If energy is detected on the mic while TTS is active, trigger
        interrupt to stop playback and prepare for new input.
        """
        try:
            import sounddevice as sd
        except ImportError:
            return  # Can't monitor without sounddevice

        # Small queue for energy monitoring
        energy_queue: asyncio.Queue[float] = asyncio.Queue(maxsize=10)
        chunk_samples = int(self._config.sample_rate * 0.05)  # 50ms chunks

        def callback(indata, frames, time_info, status):
            import numpy as np
            rms = float(np.sqrt(np.mean(indata ** 2)))
            try:
                energy_queue.put_nowait(rms)
            except asyncio.QueueFull:
                pass

        stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=callback,
        )
        stream.start()

        # High threshold to avoid feedback from speaker
        barge_in_threshold = 0.04
        consecutive_frames = 0
        required_frames = 3  # Need 3 consecutive loud frames (~150ms)

        try:
            while not self._interrupted:
                try:
                    rms = await asyncio.wait_for(energy_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if self._tts.is_speaking and rms > barge_in_threshold:
                    consecutive_frames += 1
                    if consecutive_frames >= required_frames:
                        log.info("Barge-in detected — interrupting TTS")
                        self.interrupt()
                        return
                else:
                    consecutive_frames = 0
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop()
            stream.close()
