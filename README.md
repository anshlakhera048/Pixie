# Pixie

**A modular AI agent platform with structured reasoning, tool execution, and multi-modal interaction.**

Pixie is not a chatbot. It is an orchestrated agent system that reasons over problems, selects and executes tools, maintains memory across sessions, and exposes itself through CLI, voice, vision, and API interfaces.

---

## Overview

Pixie solves the gap between standalone LLM wrappers and production agent systems. It provides:

- **Structured reasoning** via a Think-Act-Observe loop with JSON-constrained output
- **Reliable tool execution** with timeout protection, circuit breakers, and fallback handling
- **Persistent intelligence** through vector memory (FAISS), user profiling, and interaction learning
- **Multiple interfaces** — CLI, REST API, voice (wake-word activated), and web UI
- **Production readiness** — auth, rate limiting, metrics, structured logging, Docker deployment

The system is designed for local-first development with cloud LLM backends (OpenRouter), full async execution, and zero paid infrastructure dependencies.

---

## Features

### Core Intelligence

- ReAct-style agent loop (structured JSON reasoning)
- Multi-step planning with execution engine state machine
- Tool registry with dynamic dispatch (sync/async)
- Post-task reflection and self-evaluation
- LLM response caching (LRU + TTL)

### Interaction

- Interactive CLI with streaming output
- RESTful API (FastAPI) with session management
- Voice mode (Whisper STT + Coqui TTS + wake word detection)
- Minimal web UI (FastAPI + vanilla HTML/JS)

### Memory and Learning

- Conversation memory with sliding window
- FAISS-backed vector memory for RAG retrieval
- User profile tracking (preferences, tool usage patterns)
- Interaction store with similarity-based recall

### Vision and Automation

- Screen capture (mss)
- OCR text extraction (Tesseract)
- Workflow engine with reusable step sequences
- Task scheduler (one-shot and recurring)

### System Capabilities

- Fully async runtime (asyncio + httpx)
- Streaming response generation
- Interrupt/cancel support mid-execution
- Concurrent task limiting (semaphore-based)

### Observability and Evaluation

- Structured JSONL logging
- In-memory metrics with percentile tracking (p50/p95/p99)
- Execution tracing with explainability commands
- Automated evaluation test suite
- Load testing (concurrent users simulation)
- Component profiling with timing decorators

---

## Architecture

```
                    +----------------+
                    |   Interfaces   |
                    |  CLI / API /   |
                    |  Voice / UI    |
                    +-------+--------+
                            |
                    +-------v--------+
                    | Async Runtime  |
                    | (streaming,    |
                    |  interrupts)   |
                    +-------+--------+
                            |
                    +-------v--------+
                    |     Agent      |
                    | Think-Act-     |
                    | Observe Loop   |
                    +---+---+---+----+
                        |   |   |
            +-----------+   |   +-----------+
            |               |               |
    +-------v-----+ +------v------+ +------v------+
    | Tool Registry| |   Planner   | |   Memory    |
    | (file, sys,  | | (complexity | | (conv, RAG, |
    |  web, vision)| |  detection) | |  profile)   |
    +--------------+ +-------------+ +-------------+
            |
    +-------v--------+
    |   LLM Provider  |
    | (OpenRouter /   |
    |   Mock mode)    |
    +-----------------+
```

### Component Breakdown

| Component | Location | Responsibility |
|-----------|----------|----------------|
| Runtime | `runtime/` | Async execution, streaming, interrupt handling |
| Agent | `core/agent.py` | Reasoning loop, tool dispatch, trace recording |
| Planner | `core/planner.py` | Complexity detection, step decomposition |
| Execution Engine | `core/execution_engine.py` | Multi-step state machine (PENDING/IN_PROGRESS/COMPLETED) |
| Memory | `memory/` | Conversation history, FAISS vectors, user profiles |
| Tools | `tools/` | File ops, system info, web search, vision |
| LLM | `llm/` | OpenRouter integration, mock mode, caching |
| Voice | `voice/` | STT (Whisper), TTS (Coqui), wake word |
| Vision | `vision/` | Screen capture, OCR extraction |
| API | `api/` | FastAPI server, auth, sessions, rate limiting |
| Config | `config/` | Settings, environment parsing, validation |
| Observability | `observability/` | Metrics, structured logging |
| Cache | `cache/` | LRU+TTL caching for LLM and tool results |
| Evaluation | `evaluation/` | Test suites, scenario testing, load testing |

---

## Agent Execution Flow

```
User Input
    |
    v
[Complexity Check] --> Simple: single-turn loop
    |                   Complex: decompose into plan
    v
+-- Agent Loop (max 10 iterations) --+
|                                     |
|  1. Build context (system prompt    |
|     + history + RAG + profile)      |
|  2. LLM call --> JSON response      |
|  3. Parse {thought, action, args,   |
|     final_answer}                   |
|  4. If action != "none":            |
|       Execute tool --> feedback      |
|       Add to memory --> loop again  |
|  5. If action == "none":            |
|       Return final_answer           |
|                                     |
+-------------------------------------+
    |
    v
[Post-task reflection]
    |
    v
[Store interaction + update profile]
```

Key behaviors:

- **Tool selection**: The LLM decides which tool to call based on tool descriptions injected into the system prompt. Output is constrained to valid JSON.
- **Planning**: Requests with multiple actions or sequential keywords trigger the planner, which decomposes into steps executed via the execution engine.
- **Feedback loop**: Tool results are fed back as context for the next iteration, allowing the agent to reason over results before answering.
- **Fallback**: On tool timeout (30s), parse failures, or max iterations, the agent provides graceful degradation with partial results when available.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.13 |
| HTTP/API | FastAPI, uvicorn, httpx |
| Async | asyncio (native) |
| Vector Search | FAISS (faiss-cpu) |
| Embeddings | sentence-transformers |
| STT | faster-whisper |
| TTS | Coqui TTS |
| Audio I/O | sounddevice, numpy |
| Vision | mss, Pillow, pytesseract |
| Deployment | Docker, docker-compose |
| Testing | pytest |

---

## Project Structure

```
pixie/
├── main.py                 # CLI entry point
├── api/                    # FastAPI server, auth, sessions, middleware
├── cache/                  # LRU+TTL cache for LLM and tool results
├── config/                 # Settings, env parsing, validation
├── core/                   # Agent, planner, execution engine, orchestrator, trace
├── demo/                   # Demo scenarios for showcasing
├── evaluation/             # Test suite, scenario tests, load testing
├── llm/                    # LLM provider (OpenRouter), base class, mock mode
├── memory/                 # Conversation memory, FAISS vectors, user profile
├── observability/          # Metrics, structured JSONL logging
├── prompts/                # System prompt template
├── runtime/                # Async runtime, streaming, interrupt support
├── tools/                  # Tool registry + built-in tools (file, system, web, vision)
├── ui/                     # Minimal web UI (FastAPI + HTML)
├── utils/                  # Parser, logger, profiling decorators
├── vision/                 # Screen capture, OCR
├── voice/                  # STT, TTS, wake word detection
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup and Installation

### Local Setup

```bash
git clone https://github.com/youruser/pixie.git
cd pixie
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -r requirements.txt
```

Voice and vision dependencies are optional. Install only if needed:

```bash
# Voice mode
pip install faster-whisper TTS sounddevice numpy

# Vision mode
pip install mss Pillow pytesseract
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes (or use mock) | LLM provider API key |
| `PIXIE_MOCK` | No | Set to `1` for mock mode (no LLM calls) |
| `PIXIE_API_KEYS` | For API mode | Comma-separated bearer tokens |
| `PIXIE_AUTH_ENABLED` | No | `true`/`false` (default: `true`) |
| `PIXIE_RATE_LIMIT_RPM` | No | Requests per minute per key (default: `60`) |
| `PIXIE_MAX_CONCURRENT` | No | Max concurrent tasks (default: `10`) |
| `PIXIE_LOG_LEVEL` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `PIXIE_LOG_FILE` | No | Path to JSONL log file |
| `PIXIE_LLM_TIMEOUT` | No | LLM call timeout in seconds (default: `30`) |
| `PIXIE_CB_FAILURE_THRESHOLD` | No | Circuit breaker trip threshold (default: `5`) |

Create a `.env` file or export directly:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export PIXIE_API_KEYS="my-secret-key"
```

---

## Running Pixie

### CLI Mode

```bash
python main.py
```

### Mock Mode (no API key required)

```bash
PIXIE_MOCK=1 python main.py
```

### API Mode

```bash
python main.py --api
# or directly:
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Web UI

```bash
python -m ui.app
# Opens on http://localhost:8501
```

### Docker

```bash
# Set env vars in .env file or shell
docker compose up --build
```

---

## API Usage

All endpoints require `Authorization: Bearer <api_key>` when auth is enabled.

### Send a message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it?", "session_id": "optional-id"}'
```

### Execute a complex task

```bash
curl -X POST http://localhost:8000/execute \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"task": "List all Python files then summarize them"}'
```

### Health check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics \
  -H "Authorization: Bearer my-secret-key"
```

### List tools

```bash
curl http://localhost:8000/tools \
  -H "Authorization: Bearer my-secret-key"
```

### Session management

```bash
# Create session
curl -X POST http://localhost:8000/sessions \
  -H "Authorization: Bearer my-secret-key"

# Destroy session
curl -X DELETE http://localhost:8000/sessions/{session_id} \
  -H "Authorization: Bearer my-secret-key"
```

### Scheduling

```bash
# Schedule a task
curl -X POST http://localhost:8000/schedule \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "daily-check", "task": "Check system status", "repeat_seconds": 86400}'

# List scheduled tasks
curl http://localhost:8000/schedule \
  -H "Authorization: Bearer my-secret-key"
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/exit` | Quit Pixie |
| `/reset` | Clear conversation history |
| `/tools` | List registered tools with descriptions |
| `/stop` | Interrupt current execution |
| `/status` | Show current task state |
| `/status_full` | Detailed system metrics (memory, cache, tools, profiling) |
| `/voice` | Switch to voice mode (wake-word activated) |
| `/text` | Switch back to text mode |
| `/save_workflow` | Save a reusable multi-step workflow |
| `/run_workflow` | Execute a saved workflow |
| `/workflows` | List saved workflows |
| `/schedule` | List scheduled tasks |
| `/feedback good/bad` | Rate the last response (feeds into learning) |
| `/profile` | Show your user profile and interaction stats |
| `/explain_last` | Explain reasoning behind the last decision |
| `/trace` | Full execution trace (input, plan, steps, tool calls, results) |
| `/demo <name>` | Run a demo scenario |
| `/test` | Run evaluation test suite |
| `/load_test` | Run load test against the API |
| `/perf` | Show component timing profile |

---

## Demo Scenarios

Run with `/demo <name>` in CLI mode:

| Name | Description | Steps |
|------|-------------|-------|
| `hello` | Basic conversation flow | 2 |
| `tools` | Tool selection and execution | 3 |
| `planning` | Multi-step plan decomposition | 1 |
| `memory` | Context retention across turns | 3 |
| `error_recovery` | Graceful handling of failures | 2 |
| `workflow` | Structured task output | 2 |
| `full_demo` | End-to-end capability showcase | 5 |

```bash
# List all demos
/demo list

# Run a specific demo
/demo full_demo
```

---

## Evaluation and Testing

### Test Suite

Automated evaluation of agent responses against expected outputs:

```bash
# From CLI
/test

# Programmatic
python -c "import asyncio; from evaluation.test_suite import run_default_suite; print(asyncio.run(run_default_suite()).summary())"
```

Evaluates: tool selection accuracy, response relevance, latency thresholds.

### Load Testing

Simulates concurrent API users:

```bash
# From CLI (requires API server running)
/load_test

# Reports: p50, p95, p99 latency, throughput (rps), error rate
```

### Component Profiling

```bash
/perf
# Shows: avg/p95 latency per component (llm.chat, tool.*, etc.)
```

---

## Observability

### Logging

- Structured JSONL format written to `logs/pixie.jsonl`
- Each entry includes timestamp, level, event name, and structured data
- Configurable via `PIXIE_LOG_LEVEL` and `PIXIE_LOG_FILE`

### Metrics

Available via `/metrics` endpoint and `/status_full` CLI command:

- Request counts and error rates
- Latency percentiles (p50, p95, p99) per endpoint
- Cache hit rates (LLM and tool caches)
- Tool success rates
- Active sessions and concurrent tasks
- Circuit breaker state

### Tracing

Per-interaction execution traces:

- `/explain_last` — concise reasoning explanation
- `/trace` — full step-by-step trace with timing

Traces capture: iteration count, thought process, tool selections, arguments, results, and per-step duration.

---

## Security

| Mechanism | Implementation |
|-----------|----------------|
| Authentication | Bearer token API keys (`PIXIE_API_KEYS`) |
| Rate Limiting | Sliding window per key (default 60 req/min) |
| Circuit Breaker | Trips after N consecutive failures, auto-recovers |
| Timeouts | LLM (30s), tools (30s), requests (60s) |
| Concurrency Control | Semaphore limits concurrent task execution |
| Non-root Container | Docker runs as unprivileged `pixie` user |
| Input Validation | Pydantic models on all API inputs |

Auth can be disabled for local development: `PIXIE_AUTH_ENABLED=false`

---

## Design Decisions and Trade-offs

| Decision | Rationale |
|----------|-----------|
| JSON-constrained LLM output | Enables reliable parsing and structured tool dispatch. Trades natural language flexibility for deterministic control flow. |
| Async-first runtime | Allows concurrent tool execution, non-blocking I/O, and interrupt support without threads. |
| OpenRouter as LLM backend | Single API for multiple models (GPT-4, Claude, Llama). No vendor lock-in. Mock mode for offline development. |
| LRU+TTL caching | Reduces redundant LLM calls for repeated queries. Short TTL (10min LLM, 2min tools) prevents stale data. |
| Local FAISS over hosted vector DB | Zero infrastructure cost. Adequate for single-user agent memory. Trades scalability for simplicity. |
| In-memory metrics over Prometheus | No external dependencies. Sufficient for single-instance deployment. Export-ready via `/metrics` endpoint. |
| FastAPI over Flask | Native async support, automatic OpenAPI docs, Pydantic validation. |

---

## Failure Modes

| Failure | Handling |
|---------|----------|
| LLM returns invalid JSON | Retry with correction prompt (max 1 retry), then graceful error message |
| Tool raises exception | Catch, record failure, feed error back to agent for alternative approach |
| Tool timeout (>30s) | `asyncio.wait_for` cancellation, fallback message |
| Max iterations reached | Return partial results if available, suggest breaking request into parts |
| LLM API unavailable | Circuit breaker trips after 5 failures, returns cached or error response |
| Parse failure | Structured retry with self-correction prompt injected into context |

---

## Limitations

- **LLM dependency**: Core intelligence requires an external API (OpenRouter). Mock mode is functional but not intelligent.
- **Single-instance**: No distributed agent coordination. Memory is process-local.
- **Planning heuristics**: Complexity detection uses keyword matching. May under- or over-plan.
- **Voice accuracy**: Depends on ambient noise and microphone quality. No speaker diarization.
- **OCR precision**: Tesseract accuracy varies with screen resolution and font rendering.
- **No fine-tuning**: Relies entirely on prompt engineering and in-context learning.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: OPENROUTER_API_KEY not set` | No LLM key configured | Set `OPENROUTER_API_KEY` or run with `PIXIE_MOCK=1` |
| `ModuleNotFoundError: faiss` | Optional RAG dependency | `pip install faiss-cpu` (or ignore — falls back to placeholder memory) |
| `ModuleNotFoundError: faster_whisper` | Voice deps not installed | `pip install faster-whisper TTS sounddevice numpy` |
| `401 Unauthorized` on API | Auth enabled without valid key | Set `PIXIE_API_KEYS=your-key` or `PIXIE_AUTH_ENABLED=0` |
| `429 Too Many Requests` | Rate limit exceeded | Wait 60s or increase `PIXIE_RATE_LIMIT_RPM` |
| Agent loops without answering | LLM returning malformed JSON | Check model compatibility; simpler models may not follow JSON schema |
| Docker health check failing | App not ready within 10s | Increase `start_period` in `docker-compose.yml` or check logs |
| Voice mode not activating | Missing audio device | Verify microphone access and `sounddevice` installation |

---

## Future Improvements

- Multi-agent coordination with task delegation
- Persistent vector memory (SQLite or PostgreSQL pgvector)
- Improved planning via tree-of-thought or iterative refinement
- Plugin system for third-party tool registration
- WebSocket streaming for the web UI
- Model routing (select model based on task complexity)

---

## Summary

Pixie is a production-grade AI agent platform demonstrating:

- End-to-end system design from LLM integration to deployment
- Async architecture with streaming, concurrency control, and fault tolerance
- Multi-modal interaction (CLI, API, voice, vision, web UI)
- Observability engineering (metrics, tracing, evaluation, load testing)
- Security hardening (auth, rate limiting, circuit breakers, timeouts)

---

## License

MIT
