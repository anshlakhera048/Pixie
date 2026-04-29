"""Planner — decomposes complex queries into sequential steps.

When the user's request involves multiple actions (e.g. "download a file
and summarise it"), the planner asks the LLM to produce a step-by-step
plan that the agent can execute one step at a time.
"""

from __future__ import annotations

import json
import logging
import re

from llm.base import BaseLLM
from utils.parser import parse_llm_response

log = logging.getLogger(__name__)

# Keywords that suggest a multi-step request
_COMPLEXITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\band\b.*\b(?:then|also|next|after)\b", re.IGNORECASE),
    re.compile(r"\bthen\b", re.IGNORECASE),
    re.compile(r"\bfirst\b.*\bthen\b", re.IGNORECASE),
    re.compile(r"\bstep\s*\d", re.IGNORECASE),
]

# If ≥2 distinct action verbs appear, treat as complex
_ACTION_VERBS = {
    "list", "read", "write", "search", "fetch", "download",
    "open", "check", "get", "find", "show", "create", "delete",
    "summarize", "summarise", "analyze", "analyse", "compare",
}

PLAN_SYSTEM_PROMPT = """You are a task planner. Given a user request, break it down into a numbered list of simple, concrete steps.

Rules:
- Each step must be a single action.
- Return ONLY a JSON array of strings, e.g. ["Step 1 description", "Step 2 description"].
- No extra text outside the JSON array.
- Maximum 5 steps.
- If the task is already simple (single action), return a single-element array.
"""


class Planner:
    """Breaks complex user requests into ordered sub-tasks."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def is_complex(self, user_input: str) -> bool:
        """Heuristic check: does this query need multi-step planning?"""
        text = user_input.lower()

        # Check explicit pattern matches
        for pattern in _COMPLEXITY_PATTERNS:
            if pattern.search(text):
                return True

        # Count distinct action verbs
        words = set(re.findall(r"\b\w+\b", text))
        verb_count = len(words & _ACTION_VERBS)
        if verb_count >= 2:
            return True

        return False

    def create_plan(self, user_input: str) -> list[str]:
        """Ask the LLM to decompose the request into steps.

        Returns a list of step descriptions.  If planning fails, returns
        a single-element list with the original input so the agent can
        still proceed.
        """
        messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        try:
            raw = self._llm.chat(messages)
            steps = self._parse_plan(raw)
            if steps:
                log.info("Plan created with %d steps", len(steps))
                return steps
        except Exception:
            log.warning("Planning failed; falling back to single-step", exc_info=True)

        return [user_input]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan(raw: str) -> list[str]:
        """Extract a JSON list of strings from the LLM response."""
        text = raw.strip()

        # Strip code fences
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list) and all(isinstance(s, str) for s in data):
                return data[:5]
        except json.JSONDecodeError:
            pass

        # Fallback: extract JSON array from surrounding text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list) and all(isinstance(s, str) for s in data):
                    return data[:5]
            except json.JSONDecodeError:
                pass

        return []
