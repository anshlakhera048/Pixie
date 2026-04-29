from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from core.context_manager import ContextManager
from core.execution_engine import ExecutionEngine
from core.planner import Planner
from core.reflection import Reflector
from core.trace import ExecutionTrace, new_trace
from llm.base import BaseLLM
from memory.interaction_store import InteractionEntry, InteractionStore
from memory.memory import BaseVectorMemory, ConversationMemory
from memory.user_profile import UserProfile
from tools.registry import ToolRegistry
from utils.logger import log_event
from utils.parser import parse_llm_response

log = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10
MAX_PARSE_RETRIES = 1


class Agent:
    """Core reasoning agent.

    Runs the think → act → observe loop until the LLM produces a final answer
    or the iteration budget is exhausted.  Supports multi-step planning and
    RAG-style context retrieval via vector memory.

    Phase 4: Also supports async execution, streaming output, execution engine
    state tracking, and interruption.

    Phase 7: Integrates reflection, interaction learning, tool scoring,
    and user profile personalization.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        memory: ConversationMemory,
        system_prompt: str,
        vector_memory: BaseVectorMemory | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tool_registry
        self._memory = memory
        self._system_prompt = system_prompt
        self._vector_memory = vector_memory
        self._context_mgr = ContextManager(system_prompt, memory, vector_memory)
        self._planner = Planner(llm)
        self._execution_engine = ExecutionEngine()
        self._interrupted = False

        # Phase 7: Learning components
        self._reflector = Reflector(llm)
        self._interaction_store = InteractionStore()
        self._user_profile = UserProfile()
        self._last_tool_results: list[dict[str, Any]] = []
        self._last_user_input: str = ""
        self._current_trace: ExecutionTrace | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def execution_engine(self) -> ExecutionEngine:
        return self._execution_engine

    @property
    def user_profile(self) -> UserProfile:
        return self._user_profile

    @property
    def interaction_store(self) -> InteractionStore:
        return self._interaction_store

    # ------------------------------------------------------------------
    # Feedback (explicit user feedback)
    # ------------------------------------------------------------------

    def record_feedback(self, positive: bool) -> None:
        """Record explicit user feedback on the last interaction."""
        if not self._last_user_input:
            return
        # Update the last interaction entry if exists
        if self._interaction_store._entries:
            last = self._interaction_store._entries[-1]
            last.success = positive
        # Track feedback preference
        if positive:
            self._user_profile.record_command(self._last_user_input[:50])

    # ------------------------------------------------------------------
    # Interrupt support
    # ------------------------------------------------------------------

    def interrupt(self) -> None:
        """Signal the agent to stop at the next safe point."""
        self._interrupted = True
        self._execution_engine.interrupt()

    def _check_interrupt(self) -> bool:
        """Check and return whether interrupt has been signalled."""
        return self._interrupted

    def _reset_interrupt(self) -> None:
        """Clear interrupt flag for a new run."""
        self._interrupted = False

    # ------------------------------------------------------------------
    # Public — Sync (preserved for backward compatibility)
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Process a single user turn and return the final answer."""
        self._reset_interrupt()
        log_event(log, logging.INFO, "user_input", content=user_input)

        # Store user input in vector memory for future retrieval
        if self._vector_memory:
            self._vector_memory.add(user_input, metadata={"role": "user"})

        # Check if multi-step planning is needed
        if self._planner.is_complex(user_input):
            return self._run_planned(user_input)

        return self._run_single(user_input)

    # ------------------------------------------------------------------
    # Public — Async
    # ------------------------------------------------------------------

    async def arun(self, user_input: str) -> str:
        """Async version of run(). Uses async LLM calls."""
        self._reset_interrupt()
        log_event(log, logging.INFO, "user_input_async", content=user_input)

        if self._vector_memory:
            self._vector_memory.add(user_input, metadata={"role": "user"})

        if self._planner.is_complex(user_input):
            return await self._arun_planned(user_input)

        return await self._arun_single(user_input)

    async def arun_streaming(self, user_input: str) -> AsyncGenerator[str, None]:
        """Async streaming execution — yields chunks as they arrive.

        For multi-step plans, yields step markers + per-step streaming.
        For single-turn, streams the final LLM response.
        """
        self._reset_interrupt()
        log_event(log, logging.INFO, "user_input_streaming", content=user_input)

        if self._vector_memory:
            self._vector_memory.add(user_input, metadata={"role": "user"})

        if self._planner.is_complex(user_input):
            async for chunk in self._arun_planned_streaming(user_input):
                yield chunk
        else:
            async for chunk in self._arun_single_streaming(user_input):
                yield chunk

    # ------------------------------------------------------------------
    # Single-turn execution (sync — original flow)
    # ------------------------------------------------------------------

    def _run_single(self, user_input: str) -> str:
        """Standard single-turn agent loop."""
        self._memory.add("user", user_input)
        answer = self._agent_loop()

        if self._vector_memory and answer:
            self._vector_memory.add(answer, metadata={"role": "assistant"})

        return answer

    # ------------------------------------------------------------------
    # Multi-step planned execution (sync — original flow)
    # ------------------------------------------------------------------

    def _run_planned(self, user_input: str) -> str:
        """Plan, then execute each step sequentially using execution engine."""
        log_event(log, logging.INFO, "planning_triggered", input=user_input)

        steps = self._planner.create_plan(user_input)
        log_event(log, logging.INFO, "plan_created", steps=steps)

        if len(steps) <= 1:
            return self._run_single(user_input)

        # Initialize execution engine
        self._execution_engine.start_task(steps)

        plan_text = "Plan:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        self._memory.add("user", f"{user_input}\n\n[I will follow this plan]\n{plan_text}")

        while not self._execution_engine.is_complete():
            if self._check_interrupt():
                break

            step_desc = self._execution_engine.next_step()
            if step_desc is None:
                break

            self._memory.add("user", f"[Step]: {step_desc}")
            step_answer = self._agent_loop()
            self._execution_engine.update_result(step_answer)

        results = self._execution_engine.get_results()
        combined = "\n".join(results) if results else "Execution was interrupted."

        if self._vector_memory:
            self._vector_memory.add(combined, metadata={"role": "assistant", "type": "plan_result"})

        return combined

    # ------------------------------------------------------------------
    # Single-turn async execution
    # ------------------------------------------------------------------

    async def _arun_single(self, user_input: str) -> str:
        """Async single-turn agent loop."""
        self._memory.add("user", user_input)
        self._last_user_input = user_input
        self._last_tool_results = []
        self._current_trace = new_trace(user_input)

        # Inject personalization context
        self._inject_personalization(user_input)

        answer = await self._async_agent_loop()

        if self._vector_memory and answer:
            self._vector_memory.add(answer, metadata={"role": "assistant"})

        # Phase 7: Post-task reflection and learning
        await self._post_task_learn(user_input, answer)

        return answer

    async def _arun_single_streaming(self, user_input: str) -> AsyncGenerator[str, None]:
        """Streaming single-turn: run tool loop, then stream the final answer."""
        self._memory.add("user", user_input)
        self._last_user_input = user_input
        self._last_tool_results = []
        self._current_trace = new_trace(user_input)

        # Inject personalization context
        self._inject_personalization(user_input)

        # Run the agent loop iterations that need tool calls
        # Then stream the final LLM response
        collected_answer = []
        async for chunk in self._async_agent_loop_streaming():
            collected_answer.append(chunk)
            yield chunk

        # Phase 7: Post-task reflection and learning
        full_answer = "".join(collected_answer)
        await self._post_task_learn(user_input, full_answer)

    # ------------------------------------------------------------------
    # Multi-step async execution
    # ------------------------------------------------------------------

    async def _arun_planned(self, user_input: str) -> str:
        """Async plan, then execute each step."""
        log_event(log, logging.INFO, "planning_triggered_async", input=user_input)

        steps = self._planner.create_plan(user_input)
        log_event(log, logging.INFO, "plan_created", steps=steps)

        if len(steps) <= 1:
            return await self._arun_single(user_input)

        self._current_trace = new_trace(user_input, plan=steps)
        self._execution_engine.start_task(steps)

        plan_text = "Plan:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        self._memory.add("user", f"{user_input}\n\n[I will follow this plan]\n{plan_text}")

        while not self._execution_engine.is_complete():
            if self._check_interrupt():
                break

            step_desc = self._execution_engine.next_step()
            if step_desc is None:
                break

            self._memory.add("user", f"[Step]: {step_desc}")
            step_answer = await self._async_agent_loop()
            self._execution_engine.update_result(step_answer)

            # Yield control so interrupt can be processed
            await asyncio.sleep(0)

        results = self._execution_engine.get_results()
        combined = "\n".join(results) if results else "Execution was interrupted."

        if self._vector_memory:
            self._vector_memory.add(combined, metadata={"role": "assistant", "type": "plan_result"})

        return combined

    async def _arun_planned_streaming(self, user_input: str) -> AsyncGenerator[str, None]:
        """Streaming multi-step: yield step markers and results progressively."""
        log_event(log, logging.INFO, "planning_triggered_streaming", input=user_input)

        steps = self._planner.create_plan(user_input)

        if len(steps) <= 1:
            async for chunk in self._arun_single_streaming(user_input):
                yield chunk
            return

        self._execution_engine.start_task(steps)

        plan_text = "Plan:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        self._memory.add("user", f"{user_input}\n\n[I will follow this plan]\n{plan_text}")

        yield f"[Planning: {len(steps)} steps]\n"

        step_num = 0
        while not self._execution_engine.is_complete():
            if self._check_interrupt():
                yield "\n[interrupted]"
                break

            step_desc = self._execution_engine.next_step()
            if step_desc is None:
                break

            step_num += 1
            yield f"\n[Step {step_num}/{len(steps)}]: {step_desc}\n"

            self._memory.add("user", f"[Step]: {step_desc}")
            step_answer = await self._async_agent_loop()
            self._execution_engine.update_result(step_answer)

            yield f"{step_answer}\n"
            await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Core agent loop — Sync (preserved)
    # ------------------------------------------------------------------

    def _agent_loop(self) -> str:
        """Run the think → act → observe loop until termination."""
        parse_failures = 0

        for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
            if self._check_interrupt():
                return "[interrupted]"

            log_event(log, logging.INFO, "agent_iteration", iteration=iteration)

            messages = self._context_mgr.build_messages("")
            raw_response = self._llm.chat(messages)
            log_event(log, logging.DEBUG, "llm_raw_output", response=raw_response[:500])

            parsed, error = parse_llm_response(raw_response)
            if error:
                parse_failures += 1
                log_event(log, logging.WARNING, "parse_error", error=error, attempt=parse_failures)

                if parse_failures > MAX_PARSE_RETRIES:
                    return "I encountered repeated issues understanding the response. Please try again."

                self._memory.add(
                    "assistant",
                    json.dumps({
                        "thought": "My previous response was malformed. I must respond with valid JSON only.",
                        "action": "none",
                        "args": {},
                        "final_answer": "",
                    }),
                )
                continue

            parse_failures = 0

            thought = parsed.get("thought", "")
            action = parsed.get("action", "none")
            args = parsed.get("args", {})
            final_answer = parsed.get("final_answer", "")

            log_event(log, logging.INFO, "agent_decision", thought=thought, action=action)

            if action == "none":
                self._memory.add("assistant", raw_response)
                if final_answer:
                    return final_answer
                return "I don't have enough information to provide an answer. Could you rephrase?"

            # Tool call path
            tool_feedback = self._execute_tool(action, args)
            self._memory.add("assistant", raw_response)

            feedback_json = json.dumps(tool_feedback)
            self._memory.add("tool", feedback_json)
            log_event(
                log, logging.INFO, "tool_executed",
                tool=action, status=tool_feedback["status"],
            )

        log_event(log, logging.WARNING, "max_iterations_reached")
        return "I was unable to produce an answer within the allowed reasoning steps."

    # ------------------------------------------------------------------
    # Core agent loop — Async
    # ------------------------------------------------------------------

    async def _async_agent_loop(self) -> str:
        """Async think → act → observe loop."""
        parse_failures = 0

        for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
            if self._check_interrupt():
                return "[interrupted]"

            log_event(log, logging.INFO, "agent_iteration_async", iteration=iteration)
            step_start = __import__("time").time()

            messages = self._context_mgr.build_messages("")
            raw_response = await self._llm.achat(messages)
            log_event(log, logging.DEBUG, "llm_raw_output", response=raw_response[:500])

            parsed, error = parse_llm_response(raw_response)
            if error:
                parse_failures += 1
                log_event(log, logging.WARNING, "parse_error", error=error, attempt=parse_failures)

                if parse_failures > MAX_PARSE_RETRIES:
                    return "I encountered repeated issues understanding the response. Please try again."

                self._memory.add(
                    "assistant",
                    json.dumps({
                        "thought": "My previous response was malformed. I must respond with valid JSON only.",
                        "action": "none",
                        "args": {},
                        "final_answer": "",
                    }),
                )
                continue

            parse_failures = 0

            thought = parsed.get("thought", "")
            action = parsed.get("action", "none")
            args = parsed.get("args", {})
            final_answer = parsed.get("final_answer", "")

            log_event(log, logging.INFO, "agent_decision", thought=thought, action=action)

            if action == "none":
                self._memory.add("assistant", raw_response)
                duration_ms = (__import__("time").time() - step_start) * 1000
                if self._current_trace:
                    self._current_trace.add_step(iteration, thought, action, args, duration_ms=duration_ms)
                    self._current_trace.finalize(final_answer or "")
                if final_answer:
                    return final_answer
                return "I don't have enough information to provide an answer. Could you rephrase?"

            # Async tool execution
            tool_feedback = await self._async_execute_tool(action, args)
            self._memory.add("assistant", raw_response)

            feedback_json = json.dumps(tool_feedback)
            self._memory.add("tool", feedback_json)

            duration_ms = (__import__("time").time() - step_start) * 1000
            if self._current_trace:
                self._current_trace.add_step(iteration, thought, action, args, tool_result=tool_feedback, duration_ms=duration_ms)

            log_event(
                log, logging.INFO, "tool_executed_async",
                tool=action, status=tool_feedback["status"],
            )

            # Yield control between iterations
            await asyncio.sleep(0)

        log_event(log, logging.WARNING, "max_iterations_reached")
        # Provide a fallback with partial context if available
        if self._last_tool_results:
            last_result = self._last_tool_results[-1]
            if last_result.get("status") == "success":
                partial = str(last_result.get("result", ""))[:200]
                return f"I ran into my reasoning limit, but here's what I found: {partial}"
        return "I was unable to produce a complete answer within the allowed reasoning steps. Try breaking your request into smaller parts."

    async def _async_agent_loop_streaming(self) -> AsyncGenerator[str, None]:
        """Streaming agent loop: handles tool iterations, then streams the final answer."""
        parse_failures = 0

        for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
            if self._check_interrupt():
                yield "[interrupted]"
                return

            log_event(log, logging.INFO, "agent_iteration_stream", iteration=iteration)
            step_start = __import__("time").time()
            messages = self._context_mgr.build_messages("")

            # For tool-use iterations, use full response (non-streaming)
            # Only stream the final answer iteration
            raw_response = await self._llm.achat(messages)
            parsed, error = parse_llm_response(raw_response)

            if error:
                parse_failures += 1
                if parse_failures > MAX_PARSE_RETRIES:
                    yield "I encountered repeated issues understanding the response. Please try again."
                    return
                self._memory.add(
                    "assistant",
                    json.dumps({
                        "thought": "My previous response was malformed. I must respond with valid JSON only.",
                        "action": "none",
                        "args": {},
                        "final_answer": "",
                    }),
                )
                continue

            parse_failures = 0
            thought = parsed.get("thought", "")
            action = parsed.get("action", "none")
            args = parsed.get("args", {})
            final_answer = parsed.get("final_answer", "")

            if action == "none":
                self._memory.add("assistant", raw_response)
                duration_ms = (__import__("time").time() - step_start) * 1000
                if self._current_trace:
                    self._current_trace.add_step(iteration, thought, action, args, duration_ms=duration_ms)
                    self._current_trace.finalize(final_answer or "")
                if final_answer:
                    yield final_answer
                else:
                    yield "I don't have enough information to provide an answer. Could you rephrase?"
                return

            # Tool call — non-streaming
            tool_feedback = await self._async_execute_tool(action, args)
            self._memory.add("assistant", raw_response)
            feedback_json = json.dumps(tool_feedback)
            self._memory.add("tool", feedback_json)

            duration_ms = (__import__("time").time() - step_start) * 1000
            if self._current_trace:
                self._current_trace.add_step(iteration, thought, action, args, tool_result=tool_feedback, duration_ms=duration_ms)

            await asyncio.sleep(0)

        yield "I was unable to produce an answer within the allowed reasoning steps."

    # ------------------------------------------------------------------
    # Tool execution — Sync
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return structured feedback."""
        if not self._tools.has(name):
            feedback = {
                "tool_name": name,
                "status": "failure",
                "result": f"Unknown tool '{name}'. Available tools: {', '.join(self._tools.list_tools())}",
            }
            self._last_tool_results.append(feedback)
            return feedback
        try:
            result = self._tools.call(name, **args)
            feedback = {
                "tool_name": name,
                "status": "success",
                "result": str(result),
            }
            self._tools.record_tool_result(name, True)
            self._user_profile.record_tool_use(name)
            self._last_tool_results.append(feedback)
            return feedback
        except Exception as exc:
            log.exception("Tool '%s' raised an exception", name)
            feedback = {
                "tool_name": name,
                "status": "failure",
                "result": f"Error executing tool: {exc}",
            }
            self._tools.record_tool_result(name, False)
            self._last_tool_results.append(feedback)
            return feedback

    # ------------------------------------------------------------------
    # Tool execution — Async
    # ------------------------------------------------------------------

    async def _async_execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool asynchronously with timeout protection."""
        if not self._tools.has(name):
            feedback = {
                "tool_name": name,
                "status": "failure",
                "result": f"Unknown tool '{name}'. Available tools: {', '.join(self._tools.list_tools())}",
            }
            self._last_tool_results.append(feedback)
            return feedback
        try:
            result = await asyncio.wait_for(
                self._tools.acall(name, **args),
                timeout=30.0,
            )
            feedback = {
                "tool_name": name,
                "status": "success",
                "result": str(result),
            }
            self._tools.record_tool_result(name, True)
            self._user_profile.record_tool_use(name)
            self._last_tool_results.append(feedback)
            return feedback
        except asyncio.TimeoutError:
            log.warning("Tool '%s' timed out after 30s", name)
            feedback = {
                "tool_name": name,
                "status": "failure",
                "result": f"Tool '{name}' timed out. Try a simpler request or a different approach.",
            }
            self._tools.record_tool_result(name, False)
            self._last_tool_results.append(feedback)
            return feedback
        except Exception as exc:
            log.exception("Tool '%s' raised an exception (async)", name)
            feedback = {
                "tool_name": name,
                "status": "failure",
                "result": f"Tool error: {exc}. I'll try to answer without it.",
            }
            self._tools.record_tool_result(name, False)
            self._last_tool_results.append(feedback)
            return feedback

    # ------------------------------------------------------------------
    # Phase 7: Learning and personalization
    # ------------------------------------------------------------------

    def _inject_personalization(self, user_input: str) -> None:
        """Build and inject personalization context before LLM call."""
        parts: list[str] = []

        # User profile summary
        profile_summary = self._user_profile.get_profile_summary()
        if profile_summary:
            parts.append(profile_summary)

        # Tool bias/reliability info
        tool_bias = self._tools.get_tool_bias_summary()
        if tool_bias:
            parts.append(tool_bias)

        # Relevant past successful actions
        similar = self._interaction_store.query_similar(user_input, top_k=2)
        if similar:
            past_lines = []
            for entry in similar:
                if entry.success:
                    past_lines.append(
                        f"  • \"{entry.user_input[:60]}\" → used {entry.tool_used or 'direct answer'}"
                    )
            if past_lines:
                parts.append("Successful past actions:\n" + "\n".join(past_lines))

        if parts:
            self._context_mgr.set_personalization_context("\n\n".join(parts))
        else:
            self._context_mgr.set_personalization_context(None)

    async def _post_task_learn(self, user_input: str, response: str) -> None:
        """Run reflection and update learning stores after a task completes."""
        try:
            # Evaluate the interaction
            evaluation = await self._reflector.aevaluate(
                user_input, response, self._last_tool_results
            )

            # Determine primary tool used
            tools_used = [t["tool_name"] for t in self._last_tool_results if t.get("tool_name")]
            primary_tool = tools_used[0] if tools_used else ""

            # Record interaction
            entry = InteractionEntry(
                user_input=user_input[:200],
                action=response[:100],
                tool_used=primary_tool,
                outcome="success" if evaluation["success"] else "partial",
                success=evaluation["success"],
            )
            self._interaction_store.add(entry)

            # Update user profile with detected patterns
            if evaluation["success"] and primary_tool:
                self._user_profile.record_tool_use(primary_tool)

            log_event(
                log, logging.DEBUG, "reflection_complete",
                score=evaluation["success_score"],
                success=evaluation["success"],
            )

        except Exception:
            log.debug("Post-task learning failed", exc_info=True)
