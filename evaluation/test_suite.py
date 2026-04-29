"""Evaluation test suite for Pixie.

Runs test cases against the agent and measures:
- Correct tool selection
- Response quality (keyword matching)
- Latency per test
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.orchestrator import Orchestrator
from runtime.async_runtime import AsyncRuntime

log = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single evaluation test case."""

    name: str
    input: str
    expected_tool: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    max_latency_ms: float = 5000.0  # Maximum acceptable latency


@dataclass
class TestResult:
    """Result of running a single test case."""

    name: str
    passed: bool
    latency_ms: float
    tool_correct: bool | None = None  # None if no tool expected
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    response: str = ""
    error: str | None = None


@dataclass
class SuiteResult:
    """Aggregated results from a full test suite run."""

    results: list[TestResult]
    total_ms: float

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def summary(self) -> str:
        lines = [
            f"\n  {'='*50}",
            f"  EVALUATION RESULTS",
            f"  {'='*50}",
            f"  Total: {self.total}  |  Passed: {self.passed}  |  Failed: {self.failed}",
            f"  Pass rate: {self.pass_rate:.1%}",
            f"  Avg latency: {self.avg_latency_ms:.1f} ms",
            f"  Total time: {self.total_ms:.0f} ms",
            f"  {'-'*50}",
        ]
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  {icon} {r.name} ({r.latency_ms:.0f}ms)")
            if not r.passed:
                if r.error:
                    lines.append(f"      Error: {r.error}")
                if r.keywords_missing:
                    lines.append(f"      Missing keywords: {r.keywords_missing}")
                if r.tool_correct is False:
                    lines.append(f"      Wrong tool used")
        lines.append(f"  {'='*50}\n")
        return "\n".join(lines)


class Evaluator:
    """Runs test cases against a Pixie agent instance."""

    def __init__(self, *, mock: bool = True) -> None:
        self._mock = mock
        self._orchestrator: Orchestrator | None = None
        self._runtime: AsyncRuntime | None = None

    async def _ensure_runtime(self) -> None:
        """Lazy-init orchestrator and runtime."""
        if self._runtime is None:
            self._orchestrator = Orchestrator(mock=self._mock)
            self._runtime = AsyncRuntime(self._orchestrator.agent)
            await self._runtime.start()

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._runtime:
            await self._runtime.stop()
            self._runtime = None
            self._orchestrator = None

    async def run_test(self, test_case: TestCase) -> TestResult:
        """Execute a single test case and evaluate the result."""
        await self._ensure_runtime()

        start = time.perf_counter()
        try:
            chunks: list[str] = []
            async for chunk in self._runtime.submit_streaming(test_case.input):
                chunks.append(chunk)
            response = "".join(chunks)
            latency_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return TestResult(
                name=test_case.name,
                passed=False,
                latency_ms=latency_ms,
                error=str(exc),
            )

        # Evaluate tool usage
        tool_correct: bool | None = None
        if test_case.expected_tool is not None:
            # Check if the expected tool was invoked by inspecting agent internals
            last_tools = self._orchestrator.agent._last_tool_results
            used_tools = [t.get("tool") for t in last_tools] if last_tools else []
            tool_correct = test_case.expected_tool in used_tools

        # Evaluate keyword matching
        response_lower = response.lower()
        keywords_found = [k for k in test_case.expected_keywords if k.lower() in response_lower]
        keywords_missing = [k for k in test_case.expected_keywords if k.lower() not in response_lower]

        # Determine pass/fail
        passed = True
        if tool_correct is False:
            passed = False
        if keywords_missing:
            passed = False
        if latency_ms > test_case.max_latency_ms:
            passed = False

        return TestResult(
            name=test_case.name,
            passed=passed,
            latency_ms=latency_ms,
            tool_correct=tool_correct,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            response=response[:500],
        )

    async def run_all(self, test_cases: list[TestCase]) -> SuiteResult:
        """Run all test cases sequentially and return aggregated results."""
        results: list[TestResult] = []
        suite_start = time.perf_counter()

        for tc in test_cases:
            result = await self.run_test(tc)
            results.append(result)
            # Reset conversation between tests
            if self._orchestrator:
                self._orchestrator.reset()

        total_ms = (time.perf_counter() - suite_start) * 1000
        await self.shutdown()
        return SuiteResult(results=results, total_ms=total_ms)


# ---------------------------------------------------------------------------
# Default test cases
# ---------------------------------------------------------------------------

DEFAULT_TESTS: list[TestCase] = [
    TestCase(
        name="basic_greeting",
        input="Hello, how are you?",
        expected_tool=None,
        expected_keywords=["mock", "understood"],
    ),
    TestCase(
        name="list_files_tool",
        input="List the files in the current directory",
        expected_tool="list_files",
        expected_keywords=["found"],
    ),
    TestCase(
        name="system_info",
        input="What operating system am I running?",
        expected_tool=None,
        expected_keywords=["mock"],
    ),
    TestCase(
        name="read_file_request",
        input="Read the file called notes.txt",
        expected_tool=None,
        expected_keywords=["file", "path"],
    ),
    TestCase(
        name="general_question",
        input="What is the capital of France?",
        expected_tool=None,
        expected_keywords=["mock", "understood"],
    ),
]


async def run_default_suite(*, mock: bool = True) -> SuiteResult:
    """Run the default evaluation suite."""
    evaluator = Evaluator(mock=mock)
    return await evaluator.run_all(DEFAULT_TESTS)
