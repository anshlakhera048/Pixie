"""Workflow engine — reusable multi-step task sequences.

Workflows are defined as ordered lists of steps that the agent executes
sequentially.  They can be saved/loaded as JSON for reuse.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

WORKFLOWS_DIR = Path(".pixie_data") / "workflows"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    instruction: str  # natural language instruction for the agent
    tool_hint: str = ""  # optional: tool the agent should prefer
    result: str = ""
    status: str = "pending"  # pending | running | completed | failed


@dataclass
class Workflow:
    """A reusable multi-step workflow definition."""

    name: str
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, agent_fn) -> list[dict[str, Any]]:
        """Execute all steps in sequence via agent_fn.

        Args:
            agent_fn: Async callable that takes a string instruction
                      and returns the agent's response string.

        Returns:
            List of step results.
        """
        results = []
        for i, step in enumerate(self.steps):
            step.status = "running"
            log.info("Workflow '%s' step %d: %s", self.name, i + 1, step.instruction)

            try:
                response = await agent_fn(step.instruction)
                step.result = response
                step.status = "completed"
                results.append({
                    "step": i + 1,
                    "instruction": step.instruction,
                    "result": response,
                    "status": "completed",
                })
            except Exception as exc:
                step.result = str(exc)
                step.status = "failed"
                results.append({
                    "step": i + 1,
                    "instruction": step.instruction,
                    "result": str(exc),
                    "status": "failed",
                })
                log.error("Workflow step %d failed: %s", i + 1, exc)
                break  # Stop on failure

        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "steps": [
                {"instruction": s.instruction, "tool_hint": s.tool_hint}
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workflow:
        steps = [
            WorkflowStep(
                instruction=s.get("instruction", ""),
                tool_hint=s.get("tool_hint", ""),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            steps=steps,
            created_at=data.get("created_at", time.time()),
        )


# ------------------------------------------------------------------
# Workflow persistence
# ------------------------------------------------------------------


class WorkflowStore:
    """Save and load workflows from disk."""

    def __init__(self, directory: Path | None = None) -> None:
        self._dir = directory or WORKFLOWS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, workflow: Workflow) -> Path:
        """Save a workflow to disk. Returns the file path."""
        filename = _safe_filename(workflow.name) + ".json"
        path = self._dir / filename
        path.write_text(
            json.dumps(workflow.to_dict(), indent=2), encoding="utf-8"
        )
        log.info("Saved workflow '%s' to %s", workflow.name, path)
        return path

    def load(self, name: str) -> Workflow | None:
        """Load a workflow by name."""
        filename = _safe_filename(name) + ".json"
        path = self._dir / filename
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Workflow.from_dict(data)
        except Exception:
            log.warning("Failed to load workflow '%s'", name, exc_info=True)
            return None

    def list_workflows(self) -> list[str]:
        """Return names of all saved workflows."""
        names = []
        for p in self._dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                names.append(data.get("name", p.stem))
            except Exception:
                names.append(p.stem)
        return sorted(names)

    def delete(self, name: str) -> bool:
        """Delete a workflow by name."""
        filename = _safe_filename(name) + ".json"
        path = self._dir / filename
        if path.exists():
            path.unlink()
            return True
        return False


def _safe_filename(name: str) -> str:
    """Convert a workflow name to a safe filename."""
    # Only keep alphanumeric, spaces, hyphens, underscores
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
    return safe.strip().replace(" ", "_")[:64] or "workflow"
