# Pixie — Deep-Dive Technical Documentation

## 1. Project Purpose

### Why This Was Built

Pixie was built to demonstrate a complete, production-grade AI agent system — not a wrapper around an LLM API, but a full orchestration platform with structured reasoning, multi-modal interaction, and operational reliability.

### Real-World Problem Mapping

Most LLM integrations are stateless request-response wrappers. They lack:

- **Structured tool execution** — reliable, typed interaction with external systems
- **Memory and learning** — context retention across sessions, user preference adaptation
- **Operational safety** — timeouts, circuit breakers, rate limiting, observability
- **Multi-modal access** — voice, vision, API, CLI from a single agent core

Pixie addresses these gaps as a reference implementation of how to build an agent system that could operate in production environments — with all the engineering concerns (reliability, security, observability, testability) that production demands.

---

## 2. System Design

### End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INTERFACE LAYER                          │
│  CLI (main.py) │ API (FastAPI) │ Voice │ Web UI              │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     RUNTIME LAYER                             │
│  AsyncRuntime: streaming, interrupt, concurrency control      │
│  Scheduler: recurring tasks │ TaskStore: persistence          │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      AGENT LAYER                              │
│  Orchestrator → Agent (Think→Act→Observe loop)               │
│  Planner → ExecutionEngine (multi-step state machine)        │
│  ContextManager → builds LLM prompt with memory + RAG        │
│  Reflector → post-task evaluation                            │
└───────┬──────────────┬──────────────┬───────────────────────┘
        │              │              │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│  TOOL LAYER  │ │ LLM LAYER│ │MEMORY LAYER│
│  Registry    │ │ OpenRouter│ │ Conv Memory│
│  File, Sys,  │ │ Mock mode │ │ FAISS RAG  │
│  Web, Vision │ │ Caching   │ │ Profile    │
└──────────────┘ └──────────┘ └────────────┘
```

### Data Flow (Single Request)

1. **Input arrives** — CLI stdin, HTTP POST, or voice transcription
2. **Runtime wraps** — creates async task, sets up interrupt handling
3. **Orchestrator routes** — checks if planning is needed (complexity heuristic)
4. **Context assembled** — system prompt + conversation history + vector memory retrieval + user profile + tool descriptions
5. **LLM called** — sends assembled messages, receives JSON response
6. **Response parsed** — extracts `{thought, action, args, final_answer}`
7. **Tool executed** (if action specified) — async dispatch with 30s timeout
8. **Feedback injected** — tool result added to memory, loop continues
9. **Final answer emitted** — streamed to interface
10. **Post-task** — reflection scoring, interaction stored, profile updated, trace finalized

### Key Component Interactions

| From | To | Mechanism |
|------|-----|-----------|
| Interface → Runtime | `submit_streaming()` async generator |
| Runtime → Agent | `arun_streaming()` / `arun()` |
| Agent → LLM | `achat(messages)` with cache check |
| Agent → Tools | `ToolRegistry.acall(name, **args)` |
| Agent → Memory | `ConversationMemory.add()` |
| Agent → Trace | `ExecutionTrace.add_step()` |
| API → Auth | Middleware extracts Bearer token |
| API → Rate Limiter | Per-key sliding window check |

---

## 3. Tech Stack Justification

| Technology | Why Chosen | Trade-off vs Alternative |
|-----------|-----------|--------------------------|
| **Python 3.13** | Native async/await, rich ML ecosystem, rapid prototyping | Slower than Go/Rust for raw throughput; acceptable for I/O-bound agent work |
| **FastAPI** | Async-native, auto OpenAPI docs, Pydantic validation, middleware support | Heavier than Starlette alone; justified by dev productivity |
| **httpx** | Async HTTP client, connection pooling, streaming support | Could use aiohttp; httpx has cleaner API and sync/async dual interface |
| **FAISS** | Sub-millisecond vector search, no server needed, battle-tested | No persistence built-in; acceptable for local agent memory |
| **sentence-transformers** | High-quality embeddings, local inference, no API cost | Large model download; justified by avoiding per-query embedding API costs |
| **faster-whisper** | CTranslate2 optimized Whisper, 4x faster than original | Requires C++ runtime; acceptable since voice is optional |
| **Coqui TTS** | Local neural TTS, no API dependency, multiple voices | Large models; acceptable as optional install |
| **asyncio** | stdlib, no dependency, first-class in Python 3.13 | Single-threaded; sufficient for I/O-bound agent workloads |
| **Docker** | Reproducible deployment, isolation, health checks | Adds complexity for local dev; mitigated by mock mode |
| **OpenRouter** | Multi-model gateway (GPT-4, Claude, Llama), single API key | External dependency; mitigated by mock mode and caching |

---

## 4. Core Modules Breakdown

### `core/agent.py` — Agent

**Responsibility**: Implements the Think→Act→Observe reasoning loop.

**Key classes/functions**:
- `Agent` — main class, holds LLM, tools, memory, planner, trace
- `_async_agent_loop()` — core iteration loop (max 10 iterations)
- `_async_agent_loop_streaming()` — streaming variant
- `_async_execute_tool()` — tool dispatch with 30s timeout
- `_post_task_learn()` — reflection and interaction recording

**Design patterns**:
- Strategy (LLM is injected as interface `BaseLLM`)
- Observer (trace recording at each step)
- Template Method (sync/async/streaming variants share logic)

---

### `core/orchestrator.py` — Orchestrator

**Responsibility**: Wires all components together, provides high-level API.

**Key functions**:
- `__init__()` — creates LLM, tools, memory, agent
- `chat()` / `reset()` — public interface
- Tool registration (file, system, web, vision)

**Design pattern**: Facade — single entry point hiding internal complexity.

---

### `core/planner.py` — Planner

**Responsibility**: Detects complex requests and decomposes into steps.

**Key functions**:
- `is_complex(input)` — heuristic keyword detection
- `create_plan(input)` — LLM-assisted step decomposition

**Design pattern**: Strategy (complexity detection is swappable).

---

### `core/execution_engine.py` — ExecutionEngine

**Responsibility**: State machine for multi-step plan execution.

**Key functions**:
- `start_task(steps)` — initializes step queue
- `next_step()` / `update_result()` — state transitions
- `is_complete()` — termination check

**Design pattern**: State Machine (PENDING → IN_PROGRESS → COMPLETED).

---

### `core/trace.py` — Execution Tracing

**Responsibility**: Records reasoning steps for explainability.

**Key classes**:
- `ExecutionTrace` — holds full interaction trace
- `TraceStep` — single iteration (thought, action, result, timing)
- `format_explain()` / `format_trace()` — human-readable output

**Design pattern**: Event Sourcing (append-only trace of decisions).

---

### `llm/openrouter.py` — LLM Provider

**Responsibility**: Handles LLM API communication with caching.

**Key functions**:
- `chat()` / `achat()` — sync/async with LRU cache
- `_live_chat()` / `_async_live_chat()` — actual API calls
- `_mock_chat()` — deterministic mock for testing
- `stream_generate()` — SSE streaming (not cached)

**Design patterns**:
- Proxy (cache layer wraps actual calls)
- Null Object (mock mode as alternative implementation)

---

### `tools/registry.py` — Tool Registry

**Responsibility**: Central registry for all tools with caching and profiling.

**Key classes**:
- `ToolRegistry` — stores `ToolSpec` objects
- `call()` / `acall()` — dispatch with cache check + timing
- `record_tool_result()` — success/failure tracking

**Design patterns**:
- Registry (name → callable mapping)
- Decorator (profiling timer wraps execution)

---

### `memory/memory.py` — Memory System

**Responsibility**: Conversation history and vector-based RAG retrieval.

**Key classes**:
- `ConversationMemory` — sliding window message history
- `VectorMemory` — FAISS-backed semantic search

**Design pattern**: Repository (stores and retrieves by semantic similarity).

---

### `api/server.py` — API Server

**Responsibility**: HTTP interface with full production middleware.

**Key components**:
- `trace_and_metrics_middleware` — auth, rate limiting, tracing, metrics
- Session management via `SessionManager`
- Circuit breaker integration for LLM/tools

**Design patterns**:
- Middleware Chain (request pipeline)
- Circuit Breaker (fault isolation)

---

### `runtime/async_runtime.py` — Async Runtime

**Responsibility**: Manages async execution, streaming, and interrupts.

**Key classes**:
- `AsyncRuntime` — wraps agent for async CLI/API consumption
- `submit_streaming()` — async generator with interrupt support

**Design pattern**: Adapter (bridges sync agent interface to async consumers).

---

### `cache/cache.py` — Caching Layer

**Responsibility**: LRU+TTL caching for LLM and tool results.

**Key classes**:
- `Cache` — OrderedDict-based LRU with per-entry TTL
- `make_cache_key()` — SHA256-based deterministic key generation
- `get_llm_cache()` / `get_tool_cache()` — singletons

**Design patterns**:
- Singleton (shared cache instances)
- Decorator (transparent caching at call site)

---

## 5. Critical Workflows

### Workflow 1: Chat Request via API

```
POST /chat {message: "What time is it?"}
  → Middleware: verify API key, check rate limit, generate request ID
  → SessionManager: get or create session
  → Orchestrator.agent.arun(message)
    → ContextManager.build_messages()
    → LLM.achat(messages) → cache miss → OpenRouter API call
    → parse_llm_response() → {action: "get_current_time", args: {}}
    → ToolRegistry.acall("get_current_time") → "2026-04-30 16:37:00"
    → Memory.add("tool", result)
    → LLM.achat(updated_messages) → {action: "none", final_answer: "It's 4:37 PM"}
  → Return {response: "It's 4:37 PM", session_id: "..."}
  → Middleware: record latency, emit metrics
```

### Workflow 2: Multi-Step Planning

```
User: "First list files, then summarize the Python ones"
  → Planner.is_complex() → True (detected "first...then")
  → Planner.create_plan() → ["List files in current directory", "Summarize Python files"]
  → ExecutionEngine.start_task(steps)
  → Loop:
    → Step 1: Agent runs tool loop → list_files → returns file list
    → ExecutionEngine.update_result(file_list)
    → Step 2: Agent runs tool loop → reads files → generates summary
    → ExecutionEngine.update_result(summary)
  → Combined results returned to user
```

### Workflow 3: Voice Interaction

```
VoicePipeline.start()
  → WakeWordDetector.listen() → detects "hey pixie"
  → SpeechToText.transcribe(audio_buffer) → "what's the weather"
  → Agent.arun(transcript) → agent response
  → TextToSpeech.speak(response) → audio output
  → Return to wake word listening
```

---

## 6. Performance Considerations

### Latency-Sensitive Areas

| Component | Typical Latency | Optimization |
|-----------|----------------|--------------|
| LLM API call | 500ms–3000ms | LRU cache (600s TTL), avoids redundant calls |
| FAISS vector search | <1ms | In-memory index, pre-computed embeddings |
| Tool execution | 10ms–5000ms | 30s timeout, async execution |
| JSON parsing | <1ms | Three-strategy fallback (direct, fence-strip, brace-extract) |
| Context assembly | <5ms | Pre-built system prompt, sliding window memory |

### Bottlenecks

1. **LLM latency** — dominates total response time. Mitigated by caching.
2. **Embedding computation** — sentence-transformers inference on first use. Mitigated by lazy loading.
3. **Tool iteration count** — multi-tool queries require sequential LLM calls. Max 10 iterations prevents runaway.

### Optimizations Applied

- `@timed` / `@async_timed` decorators for profiling hot paths
- LRU cache with 512 entries for LLM, 128 for tools
- `asyncio.wait_for` timeout prevents tool hangs
- Streaming output reduces perceived latency (first token arrives faster)
- Connection pooling via httpx for LLM API calls

---

## 7. Scalability Design

### Current Architecture: Single-Instance

Pixie is designed as a single-instance system. This is intentional — it serves as a personal agent, not a multi-tenant platform.

### Horizontal Scaling Path (If Needed)

| Layer | Current | Scaled |
|-------|---------|--------|
| API | Single uvicorn worker | Multiple workers + load balancer |
| Memory | In-process dict/FAISS | Redis/PostgreSQL pgvector |
| Cache | In-process LRU | Redis with TTL |
| Sessions | In-memory dict | Redis-backed session store |
| Tasks | In-memory scheduler | Celery/Redis queue |
| Metrics | In-memory counters | Prometheus + Grafana |

### Stateless vs Stateful Decisions

| Component | State | Justification |
|-----------|-------|---------------|
| API handlers | Stateless | Session ID maps to external store |
| Agent loop | Stateful (per-request) | Conversation context needed within a turn |
| Cache | Stateful (in-memory) | Could be externalized to Redis |
| Metrics | Stateful (in-memory) | Acceptable loss on restart for single-instance |
| Scheduler | Stateful (persisted) | TaskStore serializes to disk on shutdown |

### Caching Strategy

- **LLM cache**: 512 entries, 600s TTL. Key = SHA256(model + messages). Prevents duplicate API calls for identical conversation states.
- **Tool cache**: 128 entries, 120s TTL. Only caches safe (read-only, deterministic) tools: `get_system_info`, `list_files`, `search_files`.

---

## 8. Failure Handling

### Edge Cases

| Scenario | Handling |
|----------|----------|
| LLM returns non-JSON text | Three-stage parser: strip fences → direct parse → brace extraction |
| LLM returns JSON missing required keys | Validation rejects, retry with correction prompt |
| Tool doesn't exist | Return error feedback to agent, let it choose another tool |
| Tool raises exception | Catch, record failure metric, feed error back for alternative approach |
| Tool hangs indefinitely | `asyncio.wait_for(timeout=30)` cancels and returns timeout message |
| Max iterations (10) reached | Return partial results if last tool succeeded, else suggest simplification |
| Circuit breaker open | Fail fast with cached response or error, auto-recover after 30s |
| Network timeout to OpenRouter | httpx timeout (30s), circuit breaker trips after 5 consecutive failures |
| Concurrent request overload | Semaphore (default 10) queues excess requests |
| Rate limit exceeded | 429 response with retry-after guidance |

### Retry Mechanisms

- **LLM parse failure**: 1 automatic retry with self-correction prompt injected
- **Tool failure**: No automatic retry — agent receives error and decides next action
- **HTTP failures**: Configurable max retries (default 2) via `utils/retry.py`
- **Circuit breaker**: After threshold failures, blocks requests for recovery period

### Error Propagation

```
Tool exception → caught in _async_execute_tool()
  → feedback = {status: "failure", result: error_msg}
  → added to conversation memory
  → agent sees error in next iteration
  → agent can try alternative tool or answer without it
```

---

## 9. Testing Strategy

### Current Coverage

| Layer | Type | Implementation |
|-------|------|----------------|
| Agent reasoning | Evaluation suite | `evaluation/test_suite.py` — 5 test cases verifying tool selection, keyword presence, latency |
| Multi-step flows | Scenario tests | `evaluation/scenarios.py` — 3 scenarios (continuity, tool+followup, error recovery) |
| API throughput | Load testing | `evaluation/load_test.py` — concurrent user simulation with p50/p95/p99 metrics |
| Component timing | Profiling | `utils/profiling.py` — per-component latency tracking |

### Testing Approach

- **Mock mode** (`PIXIE_MOCK=1`) enables full system testing without external API calls
- Mock LLM returns deterministic JSON with correct schema
- All evaluation runs in mock mode for reproducibility

### Gaps

- No unit test files (pytest-based) for individual functions
- No integration tests for voice pipeline (hardware-dependent)
- No contract tests for API response schemas
- No mutation testing or coverage measurement

### Recommended Additions

```bash
# Run evaluation suite
python main.py  # then type: /test

# Load test (requires API running)
python main.py --api  # in terminal 1
python main.py        # in terminal 2, then: /load_test
```

---

## 10. Deployment Model

### Docker Deployment

```dockerfile
# Single-stage build
FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/logs
RUN useradd --create-home pixie && chown -R pixie:pixie /app
USER pixie
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import httpx; r = httpx.get('http://127.0.0.1:8000/health'); assert r.status_code == 200"
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

- Single service with full environment configuration
- Named volume for logs persistence
- Health check with 30s interval
- `unless-stopped` restart policy

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
PIXIE_MOCK=1 python main.py
```

### CI/CD (Not Yet Implemented)

Recommended pipeline:
1. Lint (ruff/flake8)
2. Type check (mypy)
3. Run evaluation suite in mock mode
4. Build Docker image
5. Run health check against container
6. Push to registry

---

## 11. Interview-Focused Q&A

### "Explain this project in 1 minute"

Pixie is a modular AI agent platform. It receives natural language input through CLI, API, voice, or web UI. A structured reasoning loop sends the input to an LLM (via OpenRouter) which returns JSON with a thought process, an action to take, and arguments. If an action is specified, the system executes the corresponding tool (file operations, system queries, web fetches, screen capture), feeds the result back, and loops until the LLM produces a final answer. The system includes memory (vector search for context retrieval), caching (LRU+TTL to avoid redundant API calls), production middleware (auth, rate limiting, circuit breakers), and full observability (metrics, tracing, profiling). It's deployed via Docker with health checks and structured logging.

### "Biggest challenges faced"

1. **Reliable JSON parsing from LLMs** — Models don't always produce valid JSON. Solved with a three-strategy parser (direct parse, fence stripping, brace-depth extraction) plus a retry mechanism with self-correction prompts.

2. **Async architecture with interrupts** — Streaming responses while supporting mid-execution cancellation required careful use of `asyncio.Event` and `asyncio.wait_for` for timeout control without deadlocking the event loop.

3. **Balancing cache freshness with latency** — LLM responses should be cached to reduce cost/latency, but context changes make stale caches dangerous. Solved with SHA256 keys that include the full message history (so any context change invalidates the cache) plus short TTLs.

4. **Tool timeout handling** — Tools that hang (network issues, file system locks) must not block the agent loop. `asyncio.wait_for` with 30s timeout provides hard cancellation, and the circuit breaker prevents repeated calls to failing tools.

### "What would you improve?"

1. **Persistent memory** — Replace in-memory FAISS with SQLite-backed vector store for cross-session persistence
2. **Better planning** — Replace keyword heuristic with LLM-based complexity classification
3. **Plugin system** — Allow third-party tool registration without modifying core code
4. **Streaming refinement** — Stream intermediate tool results, not just the final answer
5. **Proper test suite** — Add pytest unit tests with coverage measurement
6. **Model routing** — Select cheaper/faster models for simple queries, expensive models for complex ones

### "How would you scale this to 1M users?"

1. **Stateless API** — Move sessions to Redis, remove in-memory state
2. **Horizontal scaling** — Multiple API workers behind a load balancer (nginx/traefik)
3. **Shared cache** — Replace in-memory LRU with Redis (same TTL semantics)
4. **Vector memory** — Migrate FAISS to PostgreSQL pgvector or Pinecone for multi-tenant isolation
5. **Task queue** — Replace in-process scheduler with Celery + Redis for distributed task execution
6. **Metrics** — Export to Prometheus, visualize with Grafana, alert on error rates
7. **Rate limiting** — Move to distributed rate limiter (Redis sliding window)
8. **Multi-model** — Route simple queries to fast/cheap models, complex to GPT-4/Claude

### "Design trade-offs you made"

| Trade-off | Chose | Over | Why |
|-----------|-------|------|-----|
| JSON-constrained output | Reliability | Naturalness | Deterministic tool dispatch requires parseable output |
| Single-instance | Simplicity | Scalability | Personal agent doesn't need distributed systems |
| OpenRouter | Multi-model flexibility | Direct API | One key, many models, easy switching |
| In-memory everything | Zero infrastructure | Durability | Acceptable for single-user; persistence is additive |
| 30s tool timeout | Safety | Completeness | Better to fail fast than hang indefinitely |
| LRU cache (not distributed) | No dependencies | Sharing | Single instance means no cache coherence needed |

---

## 12. Possible Extensions

| Extension | Value | Complexity |
|-----------|-------|-----------|
| **WebSocket streaming** | Real-time token delivery to web UI | Medium — FastAPI supports WebSocket natively |
| **Plugin marketplace** | Third-party tools without core changes | Medium — needs manifest format, sandboxing |
| **Multi-agent delegation** | Spawn sub-agents for complex sub-tasks | High — needs coordination protocol, result aggregation |
| **Fine-tuned model** | Better JSON compliance, lower latency | High — needs training data, hosting |
| **Persistent memory (pgvector)** | Cross-session knowledge retention | Low — swap FAISS for pgvector client |
| **Voice wake word training** | Custom wake word beyond keyword matching | Medium — needs training pipeline |
| **MCP (Model Context Protocol)** | Standard tool interface for LLM agents | Medium — implement MCP server/client spec |
| **Automated testing CI** | Run eval suite on every push | Low — GitHub Actions + mock mode |
| **Rate-adaptive model routing** | Use cheap models when rate-limited | Medium — needs routing logic + model performance profiles |
| **Conversation export** | Export chat history as markdown/JSON | Low — serialize ConversationMemory |
