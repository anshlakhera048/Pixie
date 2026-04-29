from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

from utils.profiling import timer as prof_timer

log = logging.getLogger(__name__)

ToolFunction = Callable[..., Any]


class ToolSpec:
    """Metadata wrapper around a registered tool function."""

    __slots__ = ("name", "description", "parameters", "fn")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, str],
        fn: ToolFunction,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters  # param_name → description
        self.fn = fn


class ToolRegistry:
    """Central registry for tools the agent may invoke."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._stats: dict[str, dict[str, int]] = {}  # tool_name → {successes, failures}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, str],
        fn: ToolFunction,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = ToolSpec(name, description, parameters, fn)
        self._stats[name] = {"successes": 0, "failures": 0}
        log.info("Registered tool: %s", name)

    def has(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Tool success scoring
    # ------------------------------------------------------------------

    def record_tool_result(self, tool_name: str, success: bool) -> None:
        """Record a tool execution outcome for scoring."""
        if tool_name not in self._stats:
            self._stats[tool_name] = {"successes": 0, "failures": 0}
        if success:
            self._stats[tool_name]["successes"] += 1
        else:
            self._stats[tool_name]["failures"] += 1

    def get_success_rate(self, tool_name: str) -> float:
        """Return success rate for a tool (0.0 to 1.0)."""
        stats = self._stats.get(tool_name)
        if not stats:
            return 0.5  # Unknown tool gets neutral score
        total = stats["successes"] + stats["failures"]
        if total == 0:
            return 0.5
        return stats["successes"] / total

    def get_tool_stats(self) -> dict[str, dict[str, Any]]:
        """Return success/failure stats for all tools."""
        result = {}
        for name, stats in self._stats.items():
            total = stats["successes"] + stats["failures"]
            result[name] = {
                "successes": stats["successes"],
                "failures": stats["failures"],
                "total": total,
                "rate": stats["successes"] / total if total > 0 else 0.5,
            }
        return result

    def get_tool_bias_summary(self) -> str:
        """Return a short text summary of tool reliability for context injection."""
        lines: list[str] = []
        for name, stats in self._stats.items():
            total = stats["successes"] + stats["failures"]
            if total < 3:
                continue  # Not enough data
            rate = stats["successes"] / total
            if rate >= 0.8:
                lines.append(f"  • {name}: highly reliable ({rate:.0%} success)")
            elif rate < 0.5:
                lines.append(f"  • {name}: unreliable ({rate:.0%} success)")
        if not lines:
            return ""
        return "Tool reliability:\n" + "\n".join(lines)

    # Tools that are safe to cache (read-only, deterministic for short periods)
    _CACHEABLE_TOOLS = frozenset({"get_system_info", "list_files", "search_files"})

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")

        # Check cache for safe tools
        if name in self._CACHEABLE_TOOLS:
            from cache.cache import get_tool_cache, make_cache_key
            cache_key = make_cache_key("tool", name, kwargs)
            cached = get_tool_cache().get(cache_key)
            if cached is not None:
                return cached

        with prof_timer(f"tool.{name}"):
            result = self._tools[name].fn(**kwargs)

        # Cache safe tool results
        if name in self._CACHEABLE_TOOLS:
            from cache.cache import get_tool_cache, make_cache_key
            cache_key = make_cache_key("tool", name, kwargs)
            get_tool_cache().set(cache_key, result)

        return result

    async def acall(self, name: str, **kwargs: Any) -> Any:
        """Async tool call. Uses native async if the tool is a coroutine function,
        otherwise wraps the sync call in an executor to avoid blocking the loop."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")

        # Check cache for safe tools
        if name in self._CACHEABLE_TOOLS:
            from cache.cache import get_tool_cache, make_cache_key
            cache_key = make_cache_key("tool", name, kwargs)
            cached = get_tool_cache().get(cache_key)
            if cached is not None:
                return cached

        fn = self._tools[name].fn
        with prof_timer(f"tool.{name}"):
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                # Run sync tools in the default executor (thread pool)
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: fn(**kwargs))

        # Cache safe tool results
        if name in self._CACHEABLE_TOOLS:
            from cache.cache import get_tool_cache, make_cache_key
            cache_key = make_cache_key("tool", name, kwargs)
            get_tool_cache().set(cache_key, result)

        return result

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """Return formatted tool descriptions for injection into system prompt."""
        return self.describe_all()

    def describe_all(self) -> str:
        """Return a human-readable description block for the system prompt.

        Format per tool:
          - tool_name(param1, param2): Description
            Parameters:
              param1: description
              param2: description
        """
        if not self._tools:
            return "No tools available."

        lines: list[str] = []
        for spec in self._tools.values():
            param_names = ", ".join(spec.parameters.keys())
            lines.append(f"- {spec.name}({param_names}): {spec.description}")
            if spec.parameters:
                lines.append("    Parameters:")
                for pname, pdesc in spec.parameters.items():
                    lines.append(f"      {pname}: {pdesc}")
        return "\n".join(lines)
