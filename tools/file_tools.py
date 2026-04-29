from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from tools.registry import ToolRegistry

# All file operations are sandboxed to this directory
BASE_DIR = Path.cwd().resolve()
MAX_FILE_SIZE = 50 * 1024  # 50 KB


def _validate_path(path: str) -> tuple[Path, str | None]:
    """Resolve path and validate it is inside BASE_DIR.

    Returns (resolved_path, None) on success or (Path(), error_message) on failure.
    """
    try:
        target = (BASE_DIR / path).resolve()
    except (OSError, ValueError) as exc:
        return Path(), f"Error: invalid path '{path}': {exc}"

    # Prevent directory traversal
    if not str(target).startswith(str(BASE_DIR)):
        return Path(), f"Error: access denied — path '{path}' is outside the allowed directory."

    return target, None


def list_files(path: str = ".") -> list[str]:
    """List files and directories at the given path (sandboxed)."""
    target, error = _validate_path(path)
    if error:
        return [error]

    if not target.exists():
        return [f"Error: path '{path}' does not exist."]
    if not target.is_dir():
        return [f"Error: '{path}' is not a directory."]

    entries: list[str] = []
    try:
        for entry in sorted(target.iterdir()):
            suffix = "/" if entry.is_dir() else ""
            entries.append(f"{entry.name}{suffix}")
    except PermissionError:
        return [f"Error: permission denied for '{path}'."]
    return entries


def read_file(path: str) -> str:
    """Read and return the contents of a file (sandboxed, capped at 50 KB)."""
    target, error = _validate_path(path)
    if error:
        return error

    if not target.exists():
        return f"Error: file '{path}' does not exist."
    if not target.is_file():
        return f"Error: '{path}' is not a file."

    try:
        size = target.stat().st_size
        if size > MAX_FILE_SIZE:
            return f"Error: file is too large ({size} bytes). Max allowed is {MAX_FILE_SIZE} bytes."
        return target.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: permission denied for '{path}'."


def search_files(query: str, path: str = ".") -> list[str]:
    """Search for files whose names match a glob query (sandboxed)."""
    target, error = _validate_path(path)
    if error:
        return [error]

    if not target.exists():
        return [f"Error: path '{path}' does not exist."]
    if not target.is_dir():
        return [f"Error: '{path}' is not a directory."]

    matches: list[str] = []
    try:
        for root, dirs, files in os.walk(target):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in files:
                if fnmatch.fnmatch(filename.lower(), query.lower()):
                    rel = os.path.relpath(os.path.join(root, filename), BASE_DIR)
                    matches.append(rel)
    except PermissionError:
        return [f"Error: permission denied while searching '{path}'."]

    if not matches:
        return [f"No files matching '{query}' found in '{path}'."]
    return matches[:50]  # Cap results


def register_file_tools(registry: ToolRegistry) -> None:
    """Register all file-related tools."""
    registry.register(
        name="list_files",
        description="List files and directories at the given path. Example: {\"path\": \".\"}",
        parameters={"path": "string — directory path (default: current directory)"},
        fn=list_files,
    )
    registry.register(
        name="read_file",
        description="Read and return the text contents of a file. Example: {\"path\": \"notes.txt\"}",
        parameters={"path": "string — file path to read"},
        fn=read_file,
    )
    registry.register(
        name="search_files",
        description="Search for files whose names match a glob pattern. Example: {\"query\": \"*.py\", \"path\": \".\"}",
        parameters={
            "query": "string — glob pattern to match filenames (e.g. '*.py')",
            "path": "string — directory to search in (default: current directory)",
        },
        fn=search_files,
    )
