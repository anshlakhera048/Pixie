"""Reflection layer — evaluates agent performance after each task.

Provides lightweight self-assessment of response quality, tool
effectiveness, and interaction outcomes without heavy ML.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class Reflector:
    """Evaluates agent interactions for continuous improvement.

    Uses heuristic scoring with optional LLM-based assessment
    for complex cases. Designed to be fast and non-blocking.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm  # Optional: for LLM-based reflection

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(
        self,
        user_input: str,
        response: str,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate an interaction and return assessment.

        Args:
            user_input: What the user asked.
            response: The final response given.
            tool_results: List of tool execution results (name, status, result).

        Returns:
            Dict with: success_score (0-1), success (bool), notes, improvements.
        """
        score = 0.0
        notes: list[str] = []
        improvements: list[str] = []

        # --- Heuristic scoring ---

        # 1. Response quality
        if response and not response.startswith("["):
            score += 0.3
        elif not response:
            notes.append("Empty response")
            improvements.append("Ensure a meaningful response is always generated")

        # 2. Response length (not too short, not error-like)
        if len(response) > 20:
            score += 0.1
        if "error" in response.lower() or "failed" in response.lower():
            score -= 0.1
            notes.append("Response contains error indicators")

        # 3. Tool execution success
        if tool_results:
            successes = sum(1 for t in tool_results if t.get("status") == "success")
            failures = sum(1 for t in tool_results if t.get("status") == "failure")
            total = successes + failures

            if total > 0:
                tool_rate = successes / total
                score += 0.3 * tool_rate
                if failures > 0:
                    failed_tools = [t["tool_name"] for t in tool_results if t.get("status") == "failure"]
                    notes.append(f"Tool failures: {', '.join(failed_tools)}")
                    improvements.append("Consider alternative tools for failed operations")
                if successes > 0:
                    notes.append(f"{successes}/{total} tools succeeded")
        else:
            # No tools used — direct answer
            score += 0.2
            notes.append("Direct answer (no tools)")

        # 4. Input-response relevance (basic keyword overlap)
        input_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        if input_words:
            relevance = len(input_words & response_words) / len(input_words)
            score += 0.1 * min(relevance, 1.0)

        # Clamp score
        score = max(0.0, min(1.0, score))

        return {
            "success_score": round(score, 2),
            "success": score >= 0.4,
            "notes": notes,
            "improvements": improvements,
            "tool_results_count": len(tool_results) if tool_results else 0,
        }

    async def aevaluate(
        self,
        user_input: str,
        response: str,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Async version of evaluate. Uses LLM if available for complex cases."""
        # Start with heuristic evaluation
        result = self.evaluate(user_input, response, tool_results)

        # For low-confidence results with LLM available, do deeper analysis
        if self._llm and 0.3 <= result["success_score"] <= 0.6:
            try:
                llm_notes = await self._llm_reflect(user_input, response)
                if llm_notes:
                    result["notes"].append(f"LLM reflection: {llm_notes}")
            except Exception:
                log.debug("LLM reflection failed", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _llm_reflect(self, user_input: str, response: str) -> str:
        """Optional LLM-based reflection for ambiguous cases."""
        if not self._llm:
            return ""

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are evaluating an AI assistant's response quality. "
                    "Respond in ONE sentence: was the response helpful and appropriate? "
                    "If not, what could be improved?"
                ),
            },
            {
                "role": "user",
                "content": f"User asked: {user_input[:200]}\nAssistant replied: {response[:300]}",
            },
        ]
        try:
            result = await self._llm.achat(prompt)
            return result.strip()[:200]
        except Exception:
            return ""
