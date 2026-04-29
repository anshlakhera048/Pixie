from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncGenerator

import httpx

from cache.cache import get_llm_cache, make_cache_key
from llm.base import BaseLLM
from utils.profiling import timed, async_timed

log = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mistral-7b-instruct"
REQUEST_TIMEOUT = 10  # seconds
STREAM_TIMEOUT = 30  # longer timeout for streaming
MAX_RETRIES = 2


class OpenRouterLLM(BaseLLM):
    """OpenRouter-backed LLM.

    Requires explicit mode selection:
    - If ``OPENROUTER_API_KEY`` env var is set → live API mode.
    - If ``mock=True`` is passed → mock mode for local development.
    - Otherwise raises a configuration error.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, mock: bool = False) -> None:
        self._model = model
        self._mock = mock
        self._api_key: str | None = os.environ.get("OPENROUTER_API_KEY")

        if self._mock:
            log.info("OpenRouter: running in explicit mock mode")
        elif self._api_key:
            log.info("OpenRouter: using live API with model %s", self._model)
        else:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Set it to use the live API, or pass mock=True for local development."
            )

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    @timed("llm.chat")
    def chat(self, messages: list[dict[str, str]]) -> str:
        # Check cache
        cache_key = make_cache_key("llm", self._model, messages)
        cached = get_llm_cache().get(cache_key)
        if cached is not None:
            return cached

        if self._mock:
            result = self._mock_chat(messages)
        else:
            result = self._live_chat(messages)

        get_llm_cache().set(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Live implementation with retry
    # ------------------------------------------------------------------

    def _live_chat(self, messages: list[dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
        }

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES
            try:
                response = httpx.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()

                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as exc:
                    log.error("Unexpected API response structure: %s", data)
                    raise RuntimeError("Failed to parse OpenRouter response") from exc

            except httpx.TimeoutException as exc:
                last_error = exc
                log.warning("Request timed out (attempt %d/%d)", attempt, MAX_RETRIES + 1)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                # Don't retry on client errors (4xx) except 429
                if exc.response.status_code != 429 and 400 <= exc.response.status_code < 500:
                    raise
                log.warning(
                    "HTTP %d (attempt %d/%d)", exc.response.status_code, attempt, MAX_RETRIES + 1
                )
            except httpx.RequestError as exc:
                last_error = exc
                log.warning("Request error (attempt %d/%d): %s", attempt, MAX_RETRIES + 1, exc)

        raise RuntimeError(f"OpenRouter API failed after {MAX_RETRIES + 1} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Mock implementation (explicit opt-in only)
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_chat(messages: list[dict[str, str]]) -> str:
        """Return a deterministic mock response for local testing."""
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"].lower()
                break

        # If we got a tool result back, produce a final answer
        if last_user_msg.startswith("{"):
            return json.dumps(
                {
                    "thought": "I received the tool result. I will summarise it for the user.",
                    "action": "none",
                    "args": {},
                    "final_answer": f"Here is what I found:\n{last_user_msg}",
                }
            )

        # If the user asks to list files, call the tool
        if "list" in last_user_msg and "file" in last_user_msg:
            return json.dumps(
                {
                    "thought": "The user wants to list files. I will use the list_files tool.",
                    "action": "list_files",
                    "args": {"path": "."},
                    "final_answer": "",
                }
            )

        # If the user asks to read a file
        if "read" in last_user_msg and "file" in last_user_msg:
            return json.dumps(
                {
                    "thought": "The user wants to read a file. I need to know which one.",
                    "action": "none",
                    "args": {},
                    "final_answer": "Which file would you like me to read? Please provide the path.",
                }
            )

        # Default: return a direct final answer
        return json.dumps(
            {
                "thought": "The user asked a general question. I will answer directly.",
                "action": "none",
                "args": {},
                "final_answer": f"[mock] I understood your message: '{last_user_msg}'. In production this would be answered by the LLM.",
            }
        )

    # ------------------------------------------------------------------
    # Async implementation
    # ------------------------------------------------------------------

    @async_timed("llm.achat")
    async def achat(self, messages: list[dict[str, str]]) -> str:
        """Async chat — uses httpx.AsyncClient for non-blocking I/O."""
        # Check cache
        cache_key = make_cache_key("llm", self._model, messages)
        cached = get_llm_cache().get(cache_key)
        if cached is not None:
            return cached

        if self._mock:
            result = self._mock_chat(messages)
        else:
            result = await self._async_live_chat(messages)

        get_llm_cache().set(cache_key, result)
        return result

    async def _async_live_chat(self, messages: list[dict[str, str]]) -> str:
        """Non-blocking HTTP request to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
        }

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        OPENROUTER_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    data = response.json()

                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as exc:
                    log.error("Unexpected API response structure: %s", data)
                    raise RuntimeError("Failed to parse OpenRouter response") from exc

            except httpx.TimeoutException as exc:
                last_error = exc
                log.warning("Async request timed out (attempt %d/%d)", attempt, MAX_RETRIES + 1)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code != 429 and 400 <= exc.response.status_code < 500:
                    raise
                log.warning(
                    "HTTP %d (attempt %d/%d)", exc.response.status_code, attempt, MAX_RETRIES + 1
                )
            except httpx.RequestError as exc:
                last_error = exc
                log.warning("Async request error (attempt %d/%d): %s", attempt, MAX_RETRIES + 1, exc)

        raise RuntimeError(f"OpenRouter API failed after {MAX_RETRIES + 1} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Streaming implementation
    # ------------------------------------------------------------------

    async def stream_generate(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from OpenRouter using SSE.

        Falls back to simulated chunking if streaming is unavailable or in mock mode.
        """
        if self._mock:
            full = self._mock_chat(messages)
            # Simulate streaming by yielding word-by-word
            words = full.split(" ")
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word
            return

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=STREAM_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as exc:
            log.warning("Streaming failed, falling back to full response: %s", exc)
            # Fallback: get full response and simulate streaming
            full = await self.achat(messages)
            words = full.split(" ")
            for i, word in enumerate(words):
                yield word if i == 0 else " " + word
