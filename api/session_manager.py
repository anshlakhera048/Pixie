"""Session manager — isolated per-session state for concurrent API access.

Each session gets its own Orchestrator (memory, agent, context)
so multiple clients can interact independently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from core.orchestrator import Orchestrator
from runtime.async_runtime import AsyncRuntime

log = logging.getLogger(__name__)

# Sessions expire after 30 minutes of inactivity
SESSION_TTL_SECONDS = 1800
MAX_SESSIONS = 50


@dataclass
class Session:
    """An isolated user session with its own runtime."""

    id: str
    orchestrator: Orchestrator
    runtime: AsyncRuntime
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_active = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS


class SessionManager:
    """Manages isolated sessions for concurrent API access.

    Thread-safe via asyncio.Lock. Each session has its own
    orchestrator and async runtime with independent memory.
    """

    def __init__(self, *, mock: bool = False) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._mock = mock

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> Session | None:
        """Retrieve an existing session by ID. Returns None if not found/expired."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                await self._destroy_session(session)
                return None
            session.touch()
            return session

    async def create_session(self) -> Session:
        """Create a new isolated session."""
        async with self._lock:
            # Enforce max sessions — evict oldest expired first
            await self._evict_expired()
            if len(self._sessions) >= MAX_SESSIONS:
                await self._evict_oldest()

            session_id = uuid4().hex[:16]
            orchestrator = Orchestrator(mock=self._mock)
            runtime = AsyncRuntime(orchestrator.agent)
            await runtime.start()

            session = Session(
                id=session_id,
                orchestrator=orchestrator,
                runtime=runtime,
            )
            self._sessions[session_id] = session
            log.info("Created session %s (total: %d)", session_id, len(self._sessions))
            return session

    async def get_or_create_session(self, session_id: str | None) -> Session:
        """Get existing session or create a new one."""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session
        return await self.create_session()

    async def destroy_session(self, session_id: str) -> bool:
        """Explicitly destroy a session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            await self._destroy_session(session)
            return True

    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)

    async def shutdown(self) -> None:
        """Stop all sessions (for graceful server shutdown)."""
        async with self._lock:
            for session in list(self._sessions.values()):
                await self._destroy_session(session)
            self._sessions.clear()
        log.info("All sessions destroyed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _destroy_session(self, session: Session) -> None:
        """Stop runtime and remove session."""
        try:
            await session.runtime.stop()
        except Exception:
            log.debug("Error stopping session runtime", exc_info=True)
        self._sessions.pop(session.id, None)
        log.debug("Destroyed session %s", session.id)

    async def _evict_expired(self) -> None:
        """Remove all expired sessions."""
        expired = [s for s in self._sessions.values() if s.is_expired]
        for session in expired:
            await self._destroy_session(session)

    async def _evict_oldest(self) -> None:
        """Remove the oldest session to make room."""
        if not self._sessions:
            return
        oldest = min(self._sessions.values(), key=lambda s: s.last_active)
        await self._destroy_session(oldest)
