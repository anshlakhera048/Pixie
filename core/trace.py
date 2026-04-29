"""Execution trace recorder for explainability.

Captures reasoning steps, tool calls, and decisions made during agent execution.
Lightweight — stores only the last interaction trace for /explain_last and /trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    """A single step in the agent's execution."""

    iteration: int
    thought: str
    action: str
    args: dict[str, Any]
    tool_result: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


@dataclass
class ExecutionTrace:
    """Full trace of a single user interaction."""

    user_input: str = ""
    plan: list[str] | None = None
    steps: list[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: float = 0.0
    started_at: float = field(default_factory=time.time)

    def add_step(
        self,
        iteration: int,
        thought: str,
        action: str,
        args: dict[str, Any],
        tool_result: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        self.steps.append(
            TraceStep(
                iteration=iteration,
                thought=thought,
                action=action,
                args=args,
                tool_result=tool_result,
                duration_ms=duration_ms,
            )
        )

    def finalize(self, final_answer: str) -> None:
        self.final_answer = final_answer
        self.total_duration_ms = (time.time() - self.started_at) * 1000

    # ------------------------------------------------------------------
    # Formatted outputs
    # ------------------------------------------------------------------

    def format_explain(self) -> str:
        """Format a concise explanation of the last decision (for /explain_last)."""
        if not self.steps and not self.final_answer:
            return "  No recent interaction to explain."

        lines = [f"  Input: {self.user_input}"]

        if self.plan:
            lines.append(f"  Plan: {len(self.plan)} steps")
            for i, s in enumerate(self.plan, 1):
                lines.append(f"    {i}. {s}")

        lines.append("")
        lines.append("  Reasoning:")
        for step in self.steps:
            lines.append(f"    [{step.iteration}] Thought: {step.thought}")
            if step.action != "none":
                lines.append(f"        Action: {step.action}({_fmt_args(step.args)})")
                if step.tool_result:
                    status = step.tool_result.get("status", "?")
                    result_preview = str(step.tool_result.get("result", ""))[:120]
                    lines.append(f"        Result: [{status}] {result_preview}")

        if self.final_answer:
            answer_preview = self.final_answer[:200]
            if len(self.final_answer) > 200:
                answer_preview += "..."
            lines.append(f"\n  Final answer: {answer_preview}")

        lines.append(f"  Duration: {self.total_duration_ms:.0f}ms")
        return "\n".join(lines)

    def format_trace(self) -> str:
        """Format a full execution trace (for /trace)."""
        if not self.steps and not self.final_answer:
            return "  No recent trace available."

        lines = [
            "  ┌─ Execution Trace ─────────────────────────",
            f"  │ Input: {self.user_input}",
            f"  │ Duration: {self.total_duration_ms:.0f}ms",
        ]

        if self.plan:
            lines.append("  │")
            lines.append(f"  │ Plan ({len(self.plan)} steps):")
            for i, s in enumerate(self.plan, 1):
                lines.append(f"  │   {i}. {s}")

        lines.append("  │")
        lines.append(f"  │ Iterations: {len(self.steps)}")

        for step in self.steps:
            lines.append("  ├───────────────────────────────────────────")
            lines.append(f"  │ Step {step.iteration} ({step.duration_ms:.0f}ms)")
            lines.append(f"  │   Thought: {step.thought}")
            lines.append(f"  │   Action:  {step.action}")
            if step.action != "none" and step.args:
                lines.append(f"  │   Args:    {_fmt_args(step.args)}")
            if step.tool_result:
                status = step.tool_result.get("status", "?")
                result_str = str(step.tool_result.get("result", ""))
                # Truncate long results
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                lines.append(f"  │   Result:  [{status}] {result_str}")

        lines.append("  ├───────────────────────────────────────────")
        if self.final_answer:
            answer = self.final_answer
            if len(answer) > 300:
                answer = answer[:300] + "..."
            lines.append(f"  │ Answer: {answer}")
        lines.append("  └─────────────────────────────────────────────")

        return "\n".join(lines)


def _fmt_args(args: dict[str, Any]) -> str:
    """Format args dict concisely."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:60] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


# ------------------------------------------------------------------
# Singleton trace store — holds last trace
# ------------------------------------------------------------------

_last_trace: ExecutionTrace | None = None


def get_last_trace() -> ExecutionTrace | None:
    """Return the last recorded execution trace."""
    return _last_trace


def set_last_trace(trace: ExecutionTrace) -> None:
    """Store the current trace as the last trace."""
    global _last_trace
    _last_trace = trace


def new_trace(user_input: str, plan: list[str] | None = None) -> ExecutionTrace:
    """Create a new trace and register it as the current trace."""
    trace = ExecutionTrace(user_input=user_input, plan=plan)
    set_last_trace(trace)
    return trace
