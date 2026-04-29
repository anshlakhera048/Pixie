"""Async runtime — event-loop-based execution driver for Pixie.

Provides an asynchronous wrapper around the agent, handling:
- Task queuing via asyncio.Queue
- Streaming output delivery
- Interrupt/cancellation support
- Voice mode integration
- Graceful lifecycle management
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Callable

from config import VoiceConfig, get_config
from core.agent import Agent
from utils.logger import log_event

log = logging.getLogger(__name__)


class AsyncRuntime:
    """Asynchronous execution runtime for the Pixie agent.

    Wraps the agent in an event loop, processes tasks from a queue,
    and supports streaming output, interruption, and voice mode.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._task_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running: bool = False
        self._current_task: asyncio.Task[Any] | None = None
        self._interrupt_event = asyncio.Event()
        self._voice_pipeline = None
        self._voice_task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def execution_engine(self):
        """Expose the agent's execution engine for status queries."""
        return self._agent.execution_engine

    @property
    def voice_active(self) -> bool:
        """Whether voice mode is currently running."""
        return self._voice_pipeline is not None and self._voice_pipeline.is_running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the runtime loop."""
        self._running = True
        self._interrupt_event.clear()
        log_event(log, logging.INFO, "runtime_started")

    async def stop(self) -> None:
        """Gracefully shut down the runtime."""
        self._running = False
        await self.stop_voice()
        self.interrupt()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        log_event(log, logging.INFO, "runtime_stopped")

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    async def submit(self, user_input: str) -> str:
        """Submit a task and await the full result."""
        self._interrupt_event.clear()
        self._current_task = asyncio.current_task()
        try:
            result = await self._agent.arun(user_input)
            return result
        except asyncio.CancelledError:
            return "[interrupted]"
        finally:
            self._current_task = None

    async def submit_streaming(self, user_input: str) -> AsyncGenerator[str, None]:
        """Submit a task and yield partial results as they arrive."""
        self._interrupt_event.clear()
        self._current_task = asyncio.current_task()
        try:
            async for chunk in self._agent.arun_streaming(user_input):
                if self._interrupt_event.is_set():
                    yield "\n[interrupted]"
                    return
                yield chunk
        except asyncio.CancelledError:
            yield "\n[interrupted]"
        finally:
            self._current_task = None

    # ------------------------------------------------------------------
    # Interrupt
    # ------------------------------------------------------------------

    def interrupt(self) -> None:
        """Signal the agent to stop at the next safe point."""
        self._interrupt_event.set()
        self._agent.interrupt()
        if self._voice_pipeline:
            self._voice_pipeline.interrupt()
        log_event(log, logging.WARNING, "interrupt_signalled")

    @property
    def is_interrupted(self) -> bool:
        return self._interrupt_event.is_set()

    # ------------------------------------------------------------------
    # Voice mode
    # ------------------------------------------------------------------

    async def start_voice(self, config: VoiceConfig | None = None) -> None:
        """Activate voice interaction mode.

        Starts the voice pipeline (wake word → STT → agent → TTS)
        as a background task alongside the text CLI.
        """
        if self._voice_pipeline is not None and self._voice_pipeline.is_running:
            log.info("Voice mode already active")
            return

        from voice.pipeline import VoicePipeline

        voice_config = config or get_config().voice
        self._voice_pipeline = VoicePipeline(voice_config)
        self._voice_pipeline.start()

        # Agent callback: receives text, returns streaming response
        async def agent_streaming_callback(text: str) -> AsyncGenerator[str, None]:
            async for chunk in self._agent.arun_streaming(text):
                if self._interrupt_event.is_set():
                    return
                yield chunk

        # Run voice loop as a background task
        self._voice_task = asyncio.create_task(
            self._voice_pipeline.run_loop(agent_streaming_callback)
        )
        log_event(log, logging.INFO, "voice_mode_started")

    async def stop_voice(self) -> None:
        """Deactivate voice interaction mode."""
        if self._voice_pipeline is not None:
            self._voice_pipeline.stop()
            self._voice_pipeline = None

        if self._voice_task is not None:
            self._voice_task.cancel()
            try:
                await self._voice_task
            except asyncio.CancelledError:
                pass
            self._voice_task = None

        log_event(log, logging.INFO, "voice_mode_stopped")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current execution status."""
        engine = self._agent.execution_engine
        status = engine.get_status()
        status["runtime_running"] = self._running
        status["interrupted"] = self.is_interrupted
        status["voice_active"] = self.voice_active
        if self._voice_pipeline:
            status["voice_speaking"] = self._voice_pipeline.is_speaking
            status["voice_listening"] = self._voice_pipeline.is_listening
        return status
