from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_KEYS = {"thought", "action", "args", "final_answer"}

# Regex to find a JSON object in surrounding text
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_response(raw: str) -> tuple[dict[str, Any], str | None]:
    """Parse an LLM response string into a validated dict.

    Returns ``(parsed_dict, None)`` on success, or ``({}, error_message)`` on
    failure.  This function never raises.
    """
    text = raw.strip()

    # Strategy 1: Strip markdown code-fence wrappers
    if text.startswith("```"):
        text = _strip_code_fences(text)

    # Strategy 2: Direct JSON parse
    data = _try_parse(text)

    # Strategy 3: Extract JSON substring from surrounding text
    if data is None:
        data = _extract_json_object(raw)

    if data is None:
        return {}, f"Could not extract valid JSON from response: {raw[:200]}"

    if not isinstance(data, dict):
        return {}, "Response is not a JSON object"

    missing = REQUIRED_KEYS - data.keys()
    if missing:
        return {}, f"Missing keys: {missing}"

    # Normalise types
    if not isinstance(data.get("args"), dict):
        data["args"] = {}

    return data, None


def safe_parse(text: str) -> dict[str, Any]:
    """High-level safe parse — returns parsed dict or raises ValueError."""
    result, error = parse_llm_response(text)
    if error:
        raise ValueError(error)
    return result


def _try_parse(text: str) -> dict[str, Any] | None:
    """Attempt JSON parse, return None on failure."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the outermost JSON object in arbitrary text using brace matching."""
    start = text.find("{")
    if start == -1:
        return None

    # Find matching closing brace using depth counting
    depth = 0
    in_string = False
    escape_next = False
    end = -1

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            if in_string:
                escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    candidate = text[start : end + 1]
    return _try_parse(candidate)


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing ```json ... ``` wrappers."""
    lines = text.splitlines()
    # Drop first line if it starts with ```
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    # Drop last line if it is ```
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
