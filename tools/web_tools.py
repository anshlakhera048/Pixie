"""Web interaction tools."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

from tools.registry import ToolRegistry

log = logging.getLogger(__name__)

MAX_RESPONSE_SIZE = 100 * 1024  # 100 KB cap on fetched content
REQUEST_TIMEOUT = 10
ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url: str) -> str | None:
    """Return an error message if the URL is invalid or unsafe, else None."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return f"Error: invalid URL '{url}'."

    if parsed.scheme not in ALLOWED_SCHEMES:
        return f"Error: URL scheme '{parsed.scheme}' is not allowed. Use http or https."
    if not parsed.netloc:
        return f"Error: URL '{url}' has no host."
    return None


# ------------------------------------------------------------------
# Sync implementations (preserved for backward compatibility)
# ------------------------------------------------------------------


def open_url(url: str) -> str:
    """Fetch URL headers and return status + content-type (HEAD request)."""
    error = _validate_url(url)
    if error:
        return error

    try:
        response = httpx.head(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        content_type = response.headers.get("content-type", "unknown")
        return f"Status: {response.status_code}, Content-Type: {content_type}, URL: {response.url}"
    except httpx.TimeoutException:
        return f"Error: request to '{url}' timed out."
    except httpx.RequestError as exc:
        return f"Error: could not reach '{url}': {exc}"


def fetch_page_text(url: str) -> str:
    """Fetch a URL and return the response body as plain text (truncated)."""
    error = _validate_url(url)
    if error:
        return error

    try:
        response = httpx.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        response.raise_for_status()

        # Truncate to prevent memory issues
        text = response.text[:MAX_RESPONSE_SIZE]
        if len(response.text) > MAX_RESPONSE_SIZE:
            text += "\n\n[... truncated ...]"
        return text
    except httpx.TimeoutException:
        return f"Error: request to '{url}' timed out."
    except httpx.HTTPStatusError as exc:
        return f"Error: HTTP {exc.response.status_code} from '{url}'."
    except httpx.RequestError as exc:
        return f"Error: could not reach '{url}': {exc}"


# ------------------------------------------------------------------
# Async implementations (native non-blocking)
# ------------------------------------------------------------------


async def async_open_url(url: str) -> str:
    """Async HEAD request to check URL status."""
    error = _validate_url(url)
    if error:
        return error

    try:
        async with httpx.AsyncClient() as client:
            response = await client.head(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
            content_type = response.headers.get("content-type", "unknown")
            return f"Status: {response.status_code}, Content-Type: {content_type}, URL: {response.url}"
    except httpx.TimeoutException:
        return f"Error: request to '{url}' timed out."
    except httpx.RequestError as exc:
        return f"Error: could not reach '{url}': {exc}"


async def async_fetch_page_text(url: str) -> str:
    """Async GET request to fetch page content."""
    error = _validate_url(url)
    if error:
        return error

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
            response.raise_for_status()

            text = response.text[:MAX_RESPONSE_SIZE]
            if len(response.text) > MAX_RESPONSE_SIZE:
                text += "\n\n[... truncated ...]"
            return text
    except httpx.TimeoutException:
        return f"Error: request to '{url}' timed out."
    except httpx.HTTPStatusError as exc:
        return f"Error: HTTP {exc.response.status_code} from '{url}'."
    except httpx.RequestError as exc:
        return f"Error: could not reach '{url}': {exc}"


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


def register_web_tools(registry: ToolRegistry) -> None:
    """Register web interaction tools (async versions for non-blocking execution)."""
    registry.register(
        name="open_url",
        description="Check a URL by fetching headers (HEAD request). Returns status code and content type. Example: {\"url\": \"https://example.com\"}",
        parameters={"url": "string — the URL to check (must be http or https)"},
        fn=async_open_url,
    )
    registry.register(
        name="fetch_page_text",
        description="Fetch a URL and return the page content as plain text. Example: {\"url\": \"https://example.com\"}",
        parameters={"url": "string — the URL to fetch (must be http or https)"},
        fn=async_fetch_page_text,
    )
