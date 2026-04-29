from __future__ import annotations

import logging
from pathlib import Path

from core.agent import Agent
from llm.base import BaseLLM
from llm.openrouter import OpenRouterLLM
from memory.memory import BaseVectorMemory, ConversationMemory
from tools.file_tools import register_file_tools
from tools.registry import ToolRegistry
from tools.system_tools import register_system_tools
from tools.vision_tools import register_vision_tools
from tools.web_tools import register_web_tools

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _create_vector_memory() -> BaseVectorMemory | None:
    """Attempt to create FAISS vector memory; fall back to placeholder."""
    try:
        from memory.vector_store import FAISSVectorMemory
        return FAISSVectorMemory()
    except ImportError:
        log.warning(
            "sentence-transformers or faiss-cpu not installed; "
            "using placeholder vector memory"
        )
        from memory.memory import VectorMemory
        return VectorMemory()
    except Exception:
        log.warning("Failed to initialize FAISS vector memory; disabling", exc_info=True)
        return None


class Orchestrator:
    """Wires up all components and exposes a high-level chat interface."""

    def __init__(self, llm: BaseLLM | None = None, *, mock: bool = False) -> None:
        self._tool_registry = ToolRegistry()
        self._memory = ConversationMemory()
        self._llm = llm or OpenRouterLLM(mock=mock)
        self._vector_memory = _create_vector_memory()
        self._system_prompt = self._load_system_prompt()

        # Register built-in tools
        register_file_tools(self._tool_registry)
        register_system_tools(self._tool_registry)
        register_web_tools(self._tool_registry)
        register_vision_tools(self._tool_registry)

        # Inject available tool descriptions into the system prompt
        self._system_prompt = self._system_prompt.replace(
            "{{TOOL_DESCRIPTIONS}}", self._tool_registry.get_tool_descriptions()
        )

        self._agent = Agent(
            llm=self._llm,
            tool_registry=self._tool_registry,
            memory=self._memory,
            system_prompt=self._system_prompt,
            vector_memory=self._vector_memory,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def agent(self) -> Agent:
        """Expose the agent for the async runtime."""
        return self._agent

    def chat(self, user_input: str) -> str:
        """Send a user message and get the agent's final answer."""
        return self._agent.run(user_input)

    def reset(self) -> None:
        """Clear conversation history (vector memory persists)."""
        self._memory.clear()

    def get_tool_descriptions(self) -> str:
        """Return formatted tool descriptions (for /tools command)."""
        return self._tool_registry.get_tool_descriptions()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_system_prompt() -> str:
        prompt_path = PROMPTS_DIR / "system_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt not found at {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
