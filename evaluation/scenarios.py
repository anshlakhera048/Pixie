"""Scenario-based evaluation — multi-step tasks and workflow tests.

Tests that Pixie can handle complex multi-step interactions and
workflow execution correctly.
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
class ScenarioStep:
    """One step in a multi-step scenario."""

    input: str
    expected_keywords: list[str] = field(default_factory=list)
    expect_tool: str | None = None
    description: str = ""


@dataclass
class Scenario:
    """Multi-step evaluation scenario."""

    name: str
    description: str
    steps: list[ScenarioStep]
    max_total_ms: float = 30000.0  # Overall scenario timeout


@dataclass
class StepResult:
    """Result of a single scenario step."""

    step_index: int
    input: str
    response: str
    latency_ms: float
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    passed: bool = True
    error: str | None = None


@dataclass
class ScenarioResult:
    """Result of running a full scenario."""

    name: str
    passed: bool
    total_ms: float
    step_results: list[StepResult] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> str:
        icon = "✓" if self.passed else "✗"
        lines = [f"  {icon} {self.name} ({self.total_ms:.0f}ms total)"]
        for sr in self.step_results:
            step_icon = "✓" if sr.passed else "✗"
            lines.append(f"      {step_icon} Step {sr.step_index + 1}: {sr.input[:50]}... ({sr.latency_ms:.0f}ms)")
            if sr.keywords_missing:
                lines.append(f"          Missing: {sr.keywords_missing}")
            if sr.error:
                lines.append(f"          Error: {sr.error}")
        return "\n".join(lines)


class ScenarioRunner:
    """Executes multi-step scenarios against Pixie."""

    def __init__(self, *, mock: bool = True) -> None:
        self._mock = mock

    async def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """Run a single multi-step scenario (maintains conversation state)."""
        orchestrator = Orchestrator(mock=self._mock)
        runtime = AsyncRuntime(orchestrator.agent)
        await runtime.start()

        step_results: list[StepResult] = []
        scenario_start = time.perf_counter()
        all_passed = True

        try:
            for i, step in enumerate(scenario.steps):
                step_start = time.perf_counter()
                try:
                    chunks: list[str] = []
                    async for chunk in runtime.submit_streaming(step.input):
                        chunks.append(chunk)
                    response = "".join(chunks)
                    latency_ms = (time.perf_counter() - step_start) * 1000
                except Exception as exc:
                    latency_ms = (time.perf_counter() - step_start) * 1000
                    step_results.append(StepResult(
                        step_index=i,
                        input=step.input,
                        response="",
                        latency_ms=latency_ms,
                        passed=False,
                        error=str(exc),
                    ))
                    all_passed = False
                    continue

                # Evaluate keywords
                response_lower = response.lower()
                found = [k for k in step.expected_keywords if k.lower() in response_lower]
                missing = [k for k in step.expected_keywords if k.lower() not in response_lower]

                step_passed = len(missing) == 0
                if not step_passed:
                    all_passed = False

                step_results.append(StepResult(
                    step_index=i,
                    input=step.input,
                    response=response[:300],
                    latency_ms=latency_ms,
                    keywords_found=found,
                    keywords_missing=missing,
                    passed=step_passed,
                ))

        finally:
            await runtime.stop()

        total_ms = (time.perf_counter() - scenario_start) * 1000

        # Check total time budget
        if total_ms > scenario.max_total_ms:
            all_passed = False

        return ScenarioResult(
            name=scenario.name,
            passed=all_passed,
            total_ms=total_ms,
            step_results=step_results,
        )

    async def run_all(self, scenarios: list[Scenario]) -> list[ScenarioResult]:
        """Run all scenarios and return results."""
        results: list[ScenarioResult] = []
        for scenario in scenarios:
            result = await self.run_scenario(scenario)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Default scenarios
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS: list[Scenario] = [
    Scenario(
        name="conversation_continuity",
        description="Test that the agent maintains conversation context across turns",
        steps=[
            ScenarioStep(
                input="Hello, my name is Alice",
                expected_keywords=["mock", "understood"],
                description="Introduction",
            ),
            ScenarioStep(
                input="What did I just tell you?",
                expected_keywords=["mock"],
                description="Context recall",
            ),
        ],
    ),
    Scenario(
        name="tool_then_followup",
        description="Test tool usage followed by a follow-up question",
        steps=[
            ScenarioStep(
                input="List the files in this directory",
                expect_tool="list_files",
                expected_keywords=["found"],
                description="Tool invocation",
            ),
            ScenarioStep(
                input="Tell me more about what you found",
                expected_keywords=["mock"],
                description="Follow-up on tool results",
            ),
        ],
    ),
    Scenario(
        name="error_recovery",
        description="Test graceful handling of ambiguous requests",
        steps=[
            ScenarioStep(
                input="Do the thing",
                expected_keywords=["mock"],
                description="Ambiguous request",
            ),
            ScenarioStep(
                input="I mean, list all files please",
                expected_keywords=["found"],
                description="Clarified request",
            ),
        ],
    ),
]


async def run_default_scenarios(*, mock: bool = True) -> list[ScenarioResult]:
    """Run the default scenario suite."""
    runner = ScenarioRunner(mock=mock)
    return await runner.run_all(DEFAULT_SCENARIOS)
