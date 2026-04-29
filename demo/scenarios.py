"""Deterministic demo scenarios for showcasing Pixie capabilities.

Each scenario is a scripted conversation that demonstrates a specific feature.
Runs in mock mode for reliable, repeatable output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class DemoStep:
    """A single step in a demo scenario."""

    user_says: str
    description: str = ""


@dataclass
class DemoScenario:
    """A complete demo scenario with metadata."""

    name: str
    title: str
    description: str
    steps: list[DemoStep]
    tags: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Built-in demo scenarios
# ------------------------------------------------------------------

SCENARIOS: dict[str, DemoScenario] = {}


def _register(scenario: DemoScenario) -> DemoScenario:
    SCENARIOS[scenario.name] = scenario
    return scenario


_register(DemoScenario(
    name="hello",
    title="Basic Conversation",
    description="Simple greeting and follow-up to show natural conversation flow.",
    steps=[
        DemoStep("Hi Pixie, what can you do?", "Introduction"),
        DemoStep("Tell me a fun fact about computers", "Knowledge query"),
    ],
    tags=["basic", "conversation"],
))

_register(DemoScenario(
    name="tools",
    title="Tool Usage",
    description="Demonstrates Pixie selecting and using tools to answer questions.",
    steps=[
        DemoStep("What's the current time?", "Time tool invocation"),
        DemoStep("List the files in the current directory", "File listing tool"),
        DemoStep("What operating system am I using?", "System info tool"),
    ],
    tags=["tools", "system"],
))

_register(DemoScenario(
    name="planning",
    title="Multi-Step Planning",
    description="Shows Pixie breaking a complex request into planned steps.",
    steps=[
        DemoStep(
            "First check what files are in the current directory, then tell me what OS I'm on",
            "Multi-step plan execution",
        ),
    ],
    tags=["planning", "multi-step"],
))

_register(DemoScenario(
    name="memory",
    title="Conversation Memory",
    description="Demonstrates context retention across multiple turns.",
    steps=[
        DemoStep("My favorite color is blue", "Store preference"),
        DemoStep("What's my favorite color?", "Recall from conversation"),
        DemoStep("Suggest a desktop wallpaper based on my preference", "Use recalled context"),
    ],
    tags=["memory", "context"],
))

_register(DemoScenario(
    name="error_recovery",
    title="Error Recovery",
    description="Shows graceful handling when things go wrong.",
    steps=[
        DemoStep("Use the nonexistent_tool to do something", "Unknown tool fallback"),
        DemoStep("What happened with that last request?", "Self-awareness"),
    ],
    tags=["resilience", "errors"],
))

_register(DemoScenario(
    name="workflow",
    title="Task Workflow",
    description="Demonstrates a structured task with clear reasoning.",
    steps=[
        DemoStep("Explain how to set up a Python virtual environment step by step", "Structured output"),
        DemoStep("Now how do I install packages in it?", "Follow-up continuation"),
    ],
    tags=["workflow", "tutorial"],
))

_register(DemoScenario(
    name="full_demo",
    title="Full Capabilities Demo",
    description="End-to-end showcase of Pixie's main features in sequence.",
    steps=[
        DemoStep("Hello! Give me a quick overview of your capabilities.", "Introduction"),
        DemoStep("What time is it?", "Tool usage"),
        DemoStep("Remember that I prefer concise answers", "Preference setting"),
        DemoStep("First list files here, then tell me the OS version", "Planning"),
        DemoStep("Based on everything we've discussed, summarize our conversation", "Memory + synthesis"),
    ],
    tags=["full", "showcase"],
))


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


async def run_demo(
    scenario_name: str,
    agent_fn: Callable[[str], Awaitable[str]],
    on_step: Callable[[DemoStep, str], None] | None = None,
) -> list[dict[str, str]]:
    """Run a demo scenario through the agent.

    Args:
        scenario_name: Name of the scenario to run.
        agent_fn: Async function that takes user input and returns agent response.
        on_step: Optional callback called after each step with (step, response).

    Returns:
        List of {user, response, description} dicts.
    """
    if scenario_name not in SCENARIOS:
        raise KeyError(
            f"Unknown demo: '{scenario_name}'. Available: {', '.join(SCENARIOS.keys())}"
        )

    scenario = SCENARIOS[scenario_name]
    results: list[dict[str, str]] = []

    for step in scenario.steps:
        response = await agent_fn(step.user_says)
        results.append({
            "user": step.user_says,
            "response": response,
            "description": step.description,
        })
        if on_step:
            on_step(step, response)

    return results


def list_demos() -> list[dict[str, str]]:
    """Return list of available demos with metadata."""
    return [
        {
            "name": s.name,
            "title": s.title,
            "description": s.description,
            "steps": str(len(s.steps)),
        }
        for s in SCENARIOS.values()
    ]
