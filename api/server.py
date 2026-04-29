"""Pixie API server — FastAPI-based HTTP layer.

Wraps the existing async runtime/orchestrator system with a REST API.
Supports concurrent sessions, request tracing, rate limiting,
background task scheduling, API key auth, and circuit breakers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from api.session_manager import SessionManager
from config import get_config
from config.settings import get_settings
from observability.metrics import get_metrics
from runtime.scheduler import TaskScheduler
from utils.circuit_breaker import all_breakers, get_breaker
from utils.logger import log_request

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from production settings)
# ---------------------------------------------------------------------------

_settings = get_settings()
MAX_CONCURRENT_TASKS = _settings.max_concurrent_tasks
REQUEST_TIMEOUT_SECONDS = _settings.request_timeout_seconds

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


def _extract_api_key(request: Request) -> str | None:
    """Extract API key from Authorization header (Bearer <key>)."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


def _verify_api_key(request: Request) -> str:
    """Verify the API key and return it (or raise 401/403).

    Returns the verified key (used as rate-limit identity).
    If auth is disabled, returns a fixed placeholder.
    """
    if not _settings.auth_enabled:
        return "__noauth__"

    key = _extract_api_key(request)
    if not key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide Authorization: Bearer <key>",
        )
    if key not in _settings.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ---------------------------------------------------------------------------
# Rate limiting (per API key, sliding window)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Per-identity sliding window rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, identity: str) -> bool:
        """Return True if request is allowed for this identity."""
        now = time.time()
        async with self._lock:
            timestamps = self._requests.get(identity, [])
            timestamps = [t for t in timestamps if now - t < self._window]
            if len(timestamps) >= self._max:
                self._requests[identity] = timestamps
                return False
            timestamps.append(now)
            self._requests[identity] = timestamps
            return True


# ---------------------------------------------------------------------------
# Concurrency semaphore
# ---------------------------------------------------------------------------

_task_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _task_semaphore
    if _task_semaphore is None:
        _task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    return _task_semaphore


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str | None = Field(None, description="Session ID. Omit to create new session.")
    message: str = Field(..., min_length=1, max_length=4096)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    request_id: str


class ExecuteRequest(BaseModel):
    session_id: str | None = None
    task: str = Field(..., min_length=1, max_length=4096)


class ExecuteResponse(BaseModel):
    session_id: str
    status: str
    result: str | None = None
    request_id: str


class ScheduleRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    callback_name: str
    delay_seconds: float = Field(gt=0)
    repeat_interval: float | None = None
    args: list[str] = Field(default_factory=list)


class ScheduleResponse(BaseModel):
    task_id: str
    name: str
    request_id: str


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_session_manager: SessionManager | None = None
_scheduler: TaskScheduler | None = None
_rate_limiter: _RateLimiter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown of shared resources."""
    global _session_manager, _scheduler, _rate_limiter

    # Ensure structured logging is initialized (covers uvicorn direct launch)
    import logging as _logging
    from utils.logger import setup_logging, shutdown_logging

    setup_logging(
        level=getattr(_logging, _settings.log_level.upper(), _logging.INFO),
        log_file=_settings.log_file,
    )

    config = get_config()
    mock = config.mock or (
        not os.environ.get("OPENROUTER_API_KEY") and os.environ.get("PIXIE_MOCK", "0") == "1"
    )

    _session_manager = SessionManager(mock=mock)
    _scheduler = TaskScheduler()
    _rate_limiter = _RateLimiter(
        max_requests=_settings.rate_limit_rpm, window_seconds=60.0
    )

    # Initialize circuit breakers
    get_breaker("llm", _settings.cb_failure_threshold, _settings.cb_recovery_seconds)
    get_breaker("tools", _settings.cb_failure_threshold, _settings.cb_recovery_seconds)

    # Register default scheduler callbacks
    async def _log_callback(message: str) -> None:
        log.info("[scheduled] %s", message)

    _scheduler.register_callback("log", _log_callback)
    await _scheduler.start()

    log.info(
        "Pixie API started (mock=%s, max_concurrent=%d, auth=%s)",
        mock, MAX_CONCURRENT_TASKS, _settings.auth_enabled,
    )
    yield

    # Shutdown
    await _scheduler.stop()
    await _session_manager.shutdown()
    log.info("Pixie API shut down")
    shutdown_logging()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pixie API",
    description="AI assistant with async execution, tools, and learning",
    version="0.9.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: auth + request tracing + rate limiting + metrics
# ---------------------------------------------------------------------------

# Paths exempt from authentication (health/docs)
_AUTH_EXEMPT_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


@app.middleware("http")
async def trace_and_metrics_middleware(request: Request, call_next) -> Response:
    """Auth, request_id, rate limiting, latency measurement, structured logging."""
    request_id = request.headers.get("X-Request-ID", uuid4().hex[:12])
    request.state.request_id = request_id

    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"

    # --- Authentication ---
    if path not in _AUTH_EXEMPT_PATHS:
        try:
            api_key_identity = _verify_api_key(request)
        except HTTPException as auth_err:
            return JSONResponse(
                status_code=auth_err.status_code,
                content={"detail": auth_err.detail, "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
        request.state.api_key = api_key_identity
    else:
        request.state.api_key = "__health__"
        api_key_identity = "__health__"

    # --- Rate limiting (per API key) ---
    if path not in _AUTH_EXEMPT_PATHS and _rate_limiter:
        if not await _rate_limiter.check(api_key_identity):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "request_id": request_id},
                headers={"X-Request-ID": request_id, "Retry-After": "60"},
            )

    # --- Execute request + metrics ---
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics = get_metrics()
        metrics.record_request(path, elapsed_ms)
        if _session_manager:
            metrics.set_active_sessions(_session_manager.active_count)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = f"{elapsed_ms:.1f}"

        # Structured request log
        log_request(
            log,
            request_id=request_id,
            method=request.method,
            path=path,
            status=status_code,
            latency_ms=elapsed_ms,
            client=client_ip,
        )
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        get_metrics().record_error(path)
        log_request(
            log,
            request_id=request_id,
            method=request.method,
            path=path,
            status=500,
            latency_ms=elapsed_ms,
            client=client_ip,
            error=str(exc),
        )
        raise


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Send a message and get a response. Creates session if needed."""
    request_id = request.state.request_id
    metrics = get_metrics()

    session = await _session_manager.get_or_create_session(body.session_id)

    sem = _get_semaphore()
    if not sem._value:
        raise HTTPException(503, "Server at capacity. Try again shortly.")

    async with sem:
        metrics.task_started()
        try:
            result = await asyncio.wait_for(
                _execute_chat(session, body.message),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            metrics.record_error("/chat")
            raise HTTPException(504, "Request timed out")
        except Exception as exc:
            metrics.record_error("/chat")
            log.exception("Chat error for session %s", session.id)
            raise HTTPException(500, f"Internal error: {exc}")
        finally:
            metrics.task_completed()

    return ChatResponse(
        session_id=session.id,
        response=result,
        request_id=request_id,
    )


async def _execute_chat(session, message: str) -> str:
    """Run chat through the session's runtime."""
    chunks: list[str] = []
    async for chunk in session.runtime.submit_streaming(message):
        chunks.append(chunk)
    return "".join(chunks)


@app.post("/execute", response_model=ExecuteResponse)
async def execute_task(request: Request, body: ExecuteRequest):
    """Execute a complex task via the execution engine."""
    request_id = request.state.request_id
    metrics = get_metrics()

    session = await _session_manager.get_or_create_session(body.session_id)

    sem = _get_semaphore()
    if not sem._value:
        raise HTTPException(503, "Server at capacity. Try again shortly.")

    async with sem:
        metrics.task_started()
        try:
            result = await asyncio.wait_for(
                session.runtime.submit(body.task),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            metrics.record_error("/execute")
            raise HTTPException(504, "Task timed out")
        except Exception as exc:
            metrics.record_error("/execute")
            log.exception("Execute error for session %s", session.id)
            raise HTTPException(500, f"Internal error: {exc}")
        finally:
            metrics.task_completed()

    return ExecuteResponse(
        session_id=session.id,
        status="completed",
        result=result,
        request_id=request_id,
    )


@app.get("/status")
async def get_status(request: Request, session_id: str | None = None):
    """Get current system or session status."""
    request_id = request.state.request_id

    if session_id:
        session = await _session_manager.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        status = session.runtime.get_status()
    else:
        status = {
            "server": "running",
            "active_sessions": _session_manager.active_count,
            "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
        }

    status["request_id"] = request_id
    return status


@app.get("/tools")
async def list_tools(request: Request, session_id: str | None = None):
    """List available tools."""
    request_id = request.state.request_id

    if session_id:
        session = await _session_manager.get_session(session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        descriptions = session.orchestrator.get_tool_descriptions()
    else:
        # Create a temporary session to get tool list
        session = await _session_manager.create_session()
        descriptions = session.orchestrator.get_tool_descriptions()
        await _session_manager.destroy_session(session.id)

    return {"tools": descriptions, "request_id": request_id}


@app.get("/metrics")
async def metrics_endpoint():
    """Expose observability metrics with latency percentiles and circuit breaker state."""
    data = get_metrics().snapshot()
    # Append circuit breaker status
    breakers = all_breakers()
    data["circuit_breakers"] = {
        name: cb.snapshot() for name, cb in breakers.items()
    }
    return data


# ---------------------------------------------------------------------------
# Scheduler endpoints
# ---------------------------------------------------------------------------


@app.post("/schedule", response_model=ScheduleResponse)
async def schedule_task(request: Request, body: ScheduleRequest):
    """Schedule a background task."""
    request_id = request.state.request_id

    if body.callback_name not in _scheduler._callbacks:
        raise HTTPException(400, f"Unknown callback: {body.callback_name}")

    task_id = _scheduler.schedule(
        name=body.name,
        callback_name=body.callback_name,
        delay_seconds=body.delay_seconds,
        repeat_interval=body.repeat_interval,
        args=tuple(body.args),
    )

    return ScheduleResponse(task_id=task_id, name=body.name, request_id=request_id)


@app.get("/schedule")
async def list_scheduled(request: Request):
    """List active scheduled tasks."""
    return {"tasks": _scheduler.list_tasks(), "request_id": request.state.request_id}


@app.delete("/schedule/{task_id}")
async def cancel_scheduled(request: Request, task_id: str):
    """Cancel a scheduled task."""
    if not _scheduler.cancel(task_id):
        raise HTTPException(404, "Task not found")
    return {"cancelled": task_id, "request_id": request.state.request_id}


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------


@app.post("/sessions")
async def create_session(request: Request):
    """Create a new isolated session."""
    session = await _session_manager.create_session()
    return {"session_id": session.id, "request_id": request.state.request_id}


@app.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Destroy a session."""
    if not await _session_manager.destroy_session(session_id):
        raise HTTPException(404, "Session not found")
    return {"destroyed": session_id, "request_id": request.state.request_id}


# ---------------------------------------------------------------------------
# Health check (enhanced)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Detailed health check with dependency status."""
    metrics = get_metrics()
    uptime = round(time.time() - metrics._start_time, 1)

    # Check component health
    breakers = all_breakers()
    deps: dict[str, Any] = {}

    # LLM health (based on circuit breaker state)
    llm_cb = breakers.get("llm")
    if llm_cb:
        deps["llm"] = {
            "status": "healthy" if llm_cb.state.value != "open" else "degraded",
            "circuit": llm_cb.snapshot(),
        }
    else:
        deps["llm"] = {"status": "healthy"}

    # Session manager
    deps["sessions"] = {
        "status": "healthy" if _session_manager else "unavailable",
        "active": _session_manager.active_count if _session_manager else 0,
    }

    # Scheduler
    deps["scheduler"] = {
        "status": "healthy" if _scheduler else "unavailable",
        "tasks": len(_scheduler.list_tasks()) if _scheduler else 0,
    }

    # Memory/tools circuit breaker
    tools_cb = breakers.get("tools")
    if tools_cb:
        deps["tools"] = {
            "status": "healthy" if tools_cb.state.value != "open" else "degraded",
            "circuit": tools_cb.snapshot(),
        }

    # Overall status
    degraded = any(
        d.get("status") == "degraded" for d in deps.values()
    )
    overall = "degraded" if degraded else "healthy"

    return {
        "status": overall,
        "uptime_seconds": uptime,
        "dependencies": deps,
    }
