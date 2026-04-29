"""Pixie — CLI-based AI assistant with async runtime and voice support."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from config import get_config
from core.orchestrator import Orchestrator
from core.workflow import Workflow, WorkflowStep, WorkflowStore
from runtime.async_runtime import AsyncRuntime
from runtime.scheduler import TaskScheduler
from runtime.task_store import TaskStore
from utils.logger import setup_logging, shutdown_logging
from utils.notifications import Notifier

BANNER = r"""
  ____  _      _
 |  _ \(_)_  _(_) ___
 | |_) | \ \/ / |/ _ \
 |  __/| |>  <| |  __/
 |_|   |_/_/\_\_|\___|

 Commands:
   /exit           — quit Pixie
   /reset          — clear conversation history
   /tools          — list available tools
   /stop           — interrupt current execution
   /status         — show current task state
   /status_full    — detailed system metrics
   /explain_last   — explain last decision
   /trace          — full execution trace
   /demo <name>    — run a demo scenario
   /voice          — switch to voice mode
   /text           — switch back to text mode
   /save_workflow  — save a reusable workflow
   /run_workflow   — execute a saved workflow
   /workflows      — list saved workflows
   /schedule       — list scheduled tasks
   /feedback good  — mark last response as helpful
   /feedback bad   — mark last response as unhelpful
   /profile        — show your user profile
   /test           — run evaluation test suite
   /load_test      — run load test against API
   /perf           — show component timing profile
   /help           — show this help
"""


def _is_mock_mode() -> bool:
    """Determine if we should run in mock mode."""
    return not os.environ.get("OPENROUTER_API_KEY") and os.environ.get("PIXIE_MOCK", "0") == "1"


async def _async_input(prompt: str) -> str:
    """Non-blocking input that doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def async_main() -> None:
    config = get_config()
    from config.settings import get_settings
    settings = get_settings()
    setup_logging(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        log_file=settings.log_file,
    )
    log = logging.getLogger(__name__)

    print(BANNER)

    mock = _is_mock_mode() or config.mock
    if mock:
        print("  [running in mock mode — set OPENROUTER_API_KEY for live LLM]\n")

    try:
        orchestrator = Orchestrator(mock=mock)
    except RuntimeError as exc:
        print(f"\n  ERROR: {exc}")
        print("  Set OPENROUTER_API_KEY or run with PIXIE_MOCK=1 for local testing.\n")
        sys.exit(1)

    # Create the async runtime wrapping the agent
    runtime = AsyncRuntime(orchestrator.agent)
    await runtime.start()

    # Initialize scheduler and notifier
    scheduler = TaskScheduler()
    task_store = TaskStore()
    notifier = Notifier()
    workflow_store = WorkflowStore()

    # Register a notification callback for the scheduler
    async def _notify_callback(message: str) -> None:
        await notifier.notify(message)

    scheduler.register_callback("notify", _notify_callback)

    # Load persisted tasks and start scheduler
    for task in task_store.load():
        if task.callback_name in ("notify",):
            scheduler._tasks[task.id] = task
    await scheduler.start()

    # Auto-start voice if configured
    if config.voice.enabled:
        await _start_voice_mode(runtime)

    while True:
        try:
            user_input = (await _async_input("you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Command handling
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/exit", "/quit"):
                print("Goodbye!")
                break
            elif cmd == "/reset":
                orchestrator.reset()
                print("[conversation cleared]\n")
                continue
            elif cmd == "/tools":
                print("\nAvailable tools:")
                print(orchestrator.get_tool_descriptions())
                print()
                continue
            elif cmd == "/stop":
                runtime.interrupt()
                print("[execution interrupted]\n")
                continue
            elif cmd == "/status":
                status = runtime.get_status()
                _print_status(status)
                continue
            elif cmd == "/voice":
                await _start_voice_mode(runtime)
                continue
            elif cmd == "/text":
                await _stop_voice_mode(runtime)
                continue
            elif cmd == "/save_workflow":
                await _save_workflow_interactive(workflow_store)
                continue
            elif cmd == "/run_workflow":
                await _run_workflow_interactive(workflow_store, runtime, notifier)
                continue
            elif cmd == "/workflows":
                _list_workflows(workflow_store)
                continue
            elif cmd == "/schedule":
                _list_scheduled(scheduler)
                continue
            elif cmd == "/feedback":
                _handle_feedback(user_input, orchestrator)
                continue
            elif cmd == "/profile":
                _show_profile(orchestrator)
                continue
            elif cmd == "/test":
                await _run_eval_tests()
                continue
            elif cmd == "/load_test":
                await _run_load_test()
                continue
            elif cmd == "/perf":
                _show_perf_profile()
                continue
            elif cmd == "/explain_last":
                _show_explain_last()
                continue
            elif cmd == "/trace":
                _show_trace()
                continue
            elif cmd == "/demo":
                await _run_demo(user_input, runtime)
                continue
            elif cmd == "/status_full":
                _show_status_full(orchestrator)
                continue
            elif cmd == "/help":
                print(BANNER)
                continue
            else:
                print(f"Unknown command: {user_input}\n")
                continue

        # Legacy command support
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            orchestrator.reset()
            print("[conversation cleared]\n")
            continue

        # Stream the response
        try:
            print("\npixie> ", end="", flush=True)
            async for chunk in runtime.submit_streaming(user_input):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as exc:
            log.exception("Unhandled error during chat")
            print(f"\n[error] Something went wrong: {exc}\n")

    # Persist scheduled tasks and stop scheduler
    active_tasks = [t for t in scheduler._tasks.values() if t.active]
    task_store.save(active_tasks)
    await scheduler.stop()

    await runtime.stop()
    shutdown_logging()


async def _start_voice_mode(runtime: AsyncRuntime) -> None:
    """Activate voice mode with dependency check."""
    try:
        await runtime.start_voice()
        print("  [voice mode activated — say 'hey pixie' to interact]\n")
    except RuntimeError as exc:
        print(f"  [voice mode unavailable: {exc}]\n")
    except ImportError as exc:
        print(f"  [voice dependencies missing: {exc}]\n")
        print("  Install with: pip install faster-whisper TTS sounddevice numpy\n")


async def _stop_voice_mode(runtime: AsyncRuntime) -> None:
    """Deactivate voice mode."""
    await runtime.stop_voice()
    print("  [switched to text mode]\n")


# ------------------------------------------------------------------
# Workflow commands
# ------------------------------------------------------------------


async def _save_workflow_interactive(store: WorkflowStore) -> None:
    """Interactive workflow creation."""
    print("\n  --- Save Workflow ---")
    name = input("  Workflow name: ").strip()
    if not name:
        print("  [cancelled]\n")
        return

    description = input("  Description (optional): ").strip()
    print("  Enter steps (one per line, empty line to finish):")

    steps: list[WorkflowStep] = []
    i = 1
    while True:
        step_text = input(f"    Step {i}: ").strip()
        if not step_text:
            break
        steps.append(WorkflowStep(instruction=step_text))
        i += 1

    if not steps:
        print("  [no steps provided, cancelled]\n")
        return

    workflow = Workflow(name=name, description=description, steps=steps)
    path = store.save(workflow)
    print(f"  [saved workflow '{name}' with {len(steps)} steps]\n")


async def _run_workflow_interactive(
    store: WorkflowStore, runtime: AsyncRuntime, notifier: Notifier
) -> None:
    """Run a saved workflow."""
    workflows = store.list_workflows()
    if not workflows:
        print("  [no saved workflows — use /save_workflow first]\n")
        return

    print("\n  Available workflows:")
    for i, name in enumerate(workflows, 1):
        print(f"    {i}. {name}")

    choice = input("  Select (number or name): ").strip()

    # Resolve by number or name
    workflow_name = None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(workflows):
            workflow_name = workflows[idx]
    except ValueError:
        workflow_name = choice

    if not workflow_name:
        print("  [invalid selection]\n")
        return

    workflow = store.load(workflow_name)
    if not workflow:
        print(f"  [workflow '{workflow_name}' not found]\n")
        return

    print(f"\n  Running workflow '{workflow.name}' ({len(workflow.steps)} steps)...\n")

    # Execute each step through the runtime
    async def _agent_fn(instruction: str) -> str:
        chunks = []
        async for chunk in runtime.submit_streaming(instruction):
            chunks.append(chunk)
        return "".join(chunks)

    results = await workflow.execute(_agent_fn)

    for r in results:
        status_icon = "✓" if r["status"] == "completed" else "✗"
        print(f"  {status_icon} Step {r['step']}: {r['instruction']}")
        # Show brief result
        result_preview = r["result"][:200]
        if len(r["result"]) > 200:
            result_preview += "..."
        print(f"    → {result_preview}\n")

    await notifier.notify(f"Workflow '{workflow.name}' completed")


def _list_workflows(store: WorkflowStore) -> None:
    """Print saved workflows."""
    workflows = store.list_workflows()
    if not workflows:
        print("\n  No saved workflows.\n")
        return
    print("\n  Saved workflows:")
    for name in workflows:
        print(f"    • {name}")
    print()


def _list_scheduled(scheduler: TaskScheduler) -> None:
    """Print scheduled tasks."""
    tasks = scheduler.list_tasks()
    if not tasks:
        print("\n  No scheduled tasks.\n")
        return
    print("\n  Scheduled tasks:")
    for t in tasks:
        repeat = f" (every {t['repeat']}s)" if t.get("repeat") else " (one-shot)"
        print(f"    • [{t['id']}] {t['name']}{repeat}")
    print()


def _print_status(status: dict) -> None:
    """Pretty-print execution status."""
    print(f"\n  Status: {status.get('status', 'unknown')}")
    if status.get("progress"):
        print(f"  Progress: {status['progress']}")
    if status.get("current_step"):
        print(f"  Current: {status['current_step']}")
    if status.get("elapsed_seconds"):
        print(f"  Elapsed: {status['elapsed_seconds']}s")
    if status.get("interrupted"):
        print("  [interrupted]")
    if status.get("voice_active"):
        print("  Voice: active", end="")
        if status.get("voice_speaking"):
            print(" (speaking)", end="")
        if status.get("voice_listening"):
            print(" (listening)", end="")
        print()
    print()


# ------------------------------------------------------------------
# Feedback and profile commands
# ------------------------------------------------------------------


def _handle_feedback(user_input: str, orchestrator: Orchestrator) -> None:
    """Handle /feedback good|bad command."""
    parts = user_input.lower().split()
    if len(parts) < 2 or parts[1] not in ("good", "bad"):
        print("  Usage: /feedback good  or  /feedback bad\n")
        return

    positive = parts[1] == "good"
    orchestrator.agent.record_feedback(positive)
    if positive:
        print("  [feedback recorded: positive ✓]\n")
    else:
        print("  [feedback recorded: negative — will adjust]\n")


def _show_profile(orchestrator: Orchestrator) -> None:
    """Display the current user profile."""
    profile = orchestrator.agent.user_profile
    summary = profile.get_profile_summary()

    print("\n  --- User Profile ---")
    if summary:
        for line in summary.split("\n"):
            print(f"  {line}")
    else:
        print("  No preferences recorded yet.")

    # Show interaction stats
    store = orchestrator.agent.interaction_store
    stats = store.get_stats()
    if stats["total_interactions"] > 0:
        print(f"\n  Interactions: {stats['total_interactions']}")
        print(f"  Success rate: {stats['success_rate']:.0%}")

    # Show tool stats
    tool_stats = orchestrator.agent._tools.get_tool_stats()
    used_tools = {k: v for k, v in tool_stats.items() if v["total"] > 0}
    if used_tools:
        print("\n  Tool usage:")
        for name, s in sorted(used_tools.items(), key=lambda x: x[1]["total"], reverse=True)[:5]:
            print(f"    • {name}: {s['total']} calls ({s['rate']:.0%} success)")
    print()


# ------------------------------------------------------------------
# Evaluation and performance commands
# ------------------------------------------------------------------


async def _run_eval_tests() -> None:
    """Run the default evaluation test suite."""
    from evaluation.test_suite import run_default_suite
    from evaluation.scenarios import run_default_scenarios

    print("\n  Running evaluation test suite...")
    suite_result = await run_default_suite()
    print(f"\n  {suite_result.summary()}")

    print("\n  Running scenario tests...")
    scenario_results = await run_default_scenarios()
    for sr in scenario_results:
        status = "PASS" if sr.passed else "FAIL"
        print(f"    [{status}] {sr.scenario_name} ({sr.passed_steps}/{sr.total_steps} steps)")
    print()


async def _run_load_test() -> None:
    """Run a quick load test against the local API."""
    from evaluation.load_test import quick_load_test

    print("\n  Running load test (5 users × 3 requests)...")
    print("  Make sure the API server is running (--api flag).\n")
    try:
        result = await quick_load_test()
        print(f"  {result.summary()}\n")
    except Exception as exc:
        print(f"  [load test failed: {exc}]\n")


def _show_perf_profile() -> None:
    """Display component timing profile."""
    from utils.profiling import profile_report

    report = profile_report()
    if not report:
        print("\n  No profiling data collected yet.\n")
        return
    print(f"\n{report}")


# ------------------------------------------------------------------
# Phase 11: Explainability, Demo, Status commands
# ------------------------------------------------------------------


def _show_explain_last() -> None:
    """Show explanation of the last agent decision."""
    from core.trace import get_last_trace

    trace = get_last_trace()
    if trace is None:
        print("\n  No recent interaction to explain.\n")
        return
    print(f"\n{trace.format_explain()}\n")


def _show_trace() -> None:
    """Show full execution trace of last interaction."""
    from core.trace import get_last_trace

    trace = get_last_trace()
    if trace is None:
        print("\n  No recent trace available.\n")
        return
    print(f"\n{trace.format_trace()}\n")


async def _run_demo(user_input: str, runtime: "AsyncRuntime") -> None:
    """Run a demo scenario."""
    from demo.scenarios import SCENARIOS, run_demo, list_demos

    parts = user_input.split(maxsplit=1)
    name = parts[1].strip() if len(parts) > 1 else ""

    if not name or name == "list":
        demos = list_demos()
        print("\n  Available demos:")
        for d in demos:
            print(f"    {d['name']:15s} — {d['title']} ({d['steps']} steps)")
        print(f"\n  Usage: /demo <name>\n")
        return

    if name not in SCENARIOS:
        print(f"\n  Unknown demo '{name}'. Use /demo list to see options.\n")
        return

    scenario = SCENARIOS[name]
    print(f"\n  ╔══ Demo: {scenario.title} ══╗")
    print(f"  ║ {scenario.description}")
    print(f"  ╚{'═' * (len(scenario.title) + 12)}╝\n")

    async def agent_fn(msg: str) -> str:
        chunks: list[str] = []
        async for chunk in runtime.submit_streaming(msg):
            chunks.append(chunk)
        return "".join(chunks)

    def on_step(step, response):
        print(f"  you> {step.user_says}")
        preview = response[:300]
        if len(response) > 300:
            preview += "..."
        print(f"  pixie> {preview}\n")

    await run_demo(name, agent_fn, on_step=on_step)
    print("  [demo complete]\n")


def _show_status_full(orchestrator: "Orchestrator") -> None:
    """Show full system status including memory, cache, tools, tasks."""
    import sys as _sys
    from cache.cache import get_llm_cache, get_tool_cache
    from observability.metrics import get_metrics

    metrics = get_metrics().snapshot()

    print("\n  ┌─ System Status ────────────────────────────")
    print(f"  │ Uptime: {metrics.get('uptime_seconds', 0):.0f}s")
    print(f"  │ Requests: {metrics.get('total_requests', 0)}")
    print(f"  │ Errors: {metrics.get('total_errors', 0)}")
    print(f"  │ Active tasks: {metrics.get('active_tasks', 0)}")

    # Memory usage
    try:
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        # Windows — use approximate via sys
        import os as _os
        try:
            import psutil
            mem_mb = psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
        except ImportError:
            mem_mb = _sys.getsizeof(orchestrator) / (1024 * 1024)
    print(f"  │ Memory: ~{mem_mb:.1f} MB")

    # Cache stats
    llm_cache = get_llm_cache().stats()
    tool_cache = get_tool_cache().stats()
    print("  ├─ Cache ──────────────────────────────────")
    print(f"  │ LLM:  {llm_cache['size']}/{llm_cache['max_size']} entries, hit rate {llm_cache['hit_rate']:.0%}")
    print(f"  │ Tool: {tool_cache['size']}/{tool_cache['max_size']} entries, hit rate {tool_cache['hit_rate']:.0%}")

    # Tool success rates
    tool_stats = orchestrator.agent._tools.get_tool_stats()
    used_tools = {k: v for k, v in tool_stats.items() if v["total"] > 0}
    if used_tools:
        print("  ├─ Tools ─────────────────────────────────")
        for name, s in sorted(used_tools.items(), key=lambda x: x[1]["total"], reverse=True):
            print(f"  │ {name}: {s['total']} calls, {s['rate']:.0%} success")

    # Profiling summary
    from utils.profiling import get_timing_store
    timings = get_timing_store().all_stats()
    if timings:
        print("  ├─ Performance ────────────────────────────")
        for label, stats in sorted(timings.items()):
            print(f"  │ {label}: avg {stats['avg_ms']:.0f}ms, p95 {stats['p95_ms']:.0f}ms ({stats['count']} calls)")

    print("  └─────────────────────────────────────────────\n")


def main() -> None:
    """Entry point — runs CLI or API server based on args/config."""
    # Check for --api flag or PIXIE_API_ENABLED env
    api_mode = "--api" in sys.argv or get_config().api.enabled

    if api_mode:
        _run_api_server()
    else:
        try:
            asyncio.run(async_main())
        except KeyboardInterrupt:
            print("\nGoodbye!")


def _run_api_server() -> None:
    """Start the FastAPI server via uvicorn."""
    try:
        import uvicorn
    except ImportError:
        print("  ERROR: uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    from config.settings import get_settings
    settings = get_settings()

    # Validate settings on startup
    errors = settings.validate()
    if errors:
        for e in errors:
            print(f"  CONFIG ERROR: {e}")
        sys.exit(1)

    config = get_config()
    setup_logging(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        log_file=settings.log_file,
    )

    print(f"  Starting Pixie API server on {settings.host}:{settings.port}")
    print(f"  Auth: {'enabled' if settings.auth_enabled else 'disabled'}")
    print(f"  Rate limit: {settings.rate_limit_rpm} req/min per key")
    print(f"  Docs: http://{settings.host}:{settings.port}/docs\n")

    uvicorn.run(
        "api.server:app",
        host=settings.host,
        port=settings.port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
