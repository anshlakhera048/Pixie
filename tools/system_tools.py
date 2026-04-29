"""System information tools."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

from tools.registry import ToolRegistry


def get_system_info() -> dict[str, str]:
    """Return basic system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "cpu_count": str(os.cpu_count() or "unknown"),
        "cwd": os.getcwd(),
    }


def list_processes(limit: int = 20) -> list[str]:
    """List running processes (top N by name). Cross-platform."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = result.stdout.strip().splitlines()
            # Each line: "name.exe","PID","Session Name","Session#","Mem Usage"
            processes = []
            for line in lines[:limit]:
                parts = line.split(",")
                if parts:
                    processes.append(parts[0].strip('"'))
            return processes
        else:
            result = subprocess.run(
                ["ps", "-eo", "comm", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = result.stdout.strip().splitlines()
            return lines[:limit]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return [f"Error listing processes: {exc}"]


def register_system_tools(registry: ToolRegistry) -> None:
    """Register system-related tools."""
    registry.register(
        name="get_system_info",
        description="Get basic system information (OS, Python version, architecture, etc.). Example: {} (no args needed)",
        parameters={},
        fn=get_system_info,
    )
    registry.register(
        name="list_processes",
        description="List currently running processes (top N). Example: {\"limit\": 10}",
        parameters={"limit": "integer — max number of processes to return (default: 20)"},
        fn=list_processes,
    )
