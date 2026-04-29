"""Screen capture — fast cross-platform screenshot acquisition via mss.

Provides efficient screen capture without blocking the event loop.
Supports full-screen and region-based capture.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


class ScreenCapture:
    """Fast cross-platform screen capture using mss.

    Captures are returned as raw PIL Images for downstream processing.
    All heavy operations run in the thread pool to avoid blocking asyncio.
    """

    def __init__(self) -> None:
        self._mss = None

    def _ensure_mss(self):
        """Lazy-init mss instance (not thread-safe; create per-call if needed)."""
        try:
            import mss
            return mss.mss()
        except ImportError:
            raise RuntimeError(
                "mss is not installed. Install it with: pip install mss"
            )

    # ------------------------------------------------------------------
    # Synchronous capture (run in executor)
    # ------------------------------------------------------------------

    def _capture_sync(self, monitor: int = 0) -> Any:
        """Capture full screen. Returns a PIL Image.

        Args:
            monitor: Monitor index (0 = all monitors combined, 1 = primary).
        """
        import mss
        from PIL import Image

        with mss.mss() as sct:
            # monitor 0 = all screens combined; 1 = primary
            mon = sct.monitors[min(monitor, len(sct.monitors) - 1)]
            screenshot = sct.grab(mon)
            # Convert to PIL Image (RGB)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            return img

    def _capture_region_sync(self, x: int, y: int, w: int, h: int) -> Any:
        """Capture a specific screen region. Returns a PIL Image."""
        import mss
        from PIL import Image

        region = {"left": x, "top": y, "width": w, "height": h}
        with mss.mss() as sct:
            screenshot = sct.grab(region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            return img

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def capture(self, monitor: int = 1) -> Any:
        """Capture the full screen asynchronously. Returns PIL Image."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._capture_sync, monitor)

    async def capture_region(self, x: int, y: int, w: int, h: int) -> Any:
        """Capture a screen region asynchronously. Returns PIL Image."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._capture_region_sync, x, y, w, h)

    # ------------------------------------------------------------------
    # Sync interface (for tool calls via executor)
    # ------------------------------------------------------------------

    def capture_sync(self, monitor: int = 1) -> Any:
        """Synchronous full-screen capture."""
        return self._capture_sync(monitor)

    def capture_region_sync(self, x: int, y: int, w: int, h: int) -> Any:
        """Synchronous region capture."""
        return self._capture_region_sync(x, y, w, h)
