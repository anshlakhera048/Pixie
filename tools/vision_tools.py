"""Vision tools — screen capture and OCR tools for the agent.

Registers screen-related tools into the tool registry so the agent
can capture and analyze what's on screen.
"""

from __future__ import annotations

import logging

from tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# Module-level singletons (lazy-initialized)
_screen_capture = None
_vision_processor = None


def _get_capture():
    global _screen_capture
    if _screen_capture is None:
        from vision.screen_capture import ScreenCapture
        _screen_capture = ScreenCapture()
    return _screen_capture


def _get_processor():
    global _vision_processor
    if _vision_processor is None:
        from vision.vision_processor import VisionProcessor
        _vision_processor = VisionProcessor()
    return _vision_processor


# ------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------


async def get_screen_text() -> str:
    """Capture the screen and extract all visible text via OCR."""
    try:
        capture = _get_capture()
        processor = _get_processor()
        image = await capture.capture(monitor=1)
        text = await processor.extract_text(image)
        return text
    except RuntimeError as exc:
        return f"[screen capture unavailable: {exc}]"
    except Exception as exc:
        log.error("get_screen_text failed: %s", exc)
        return f"[error capturing screen: {exc}]"


async def describe_screen() -> str:
    """Capture the screen and produce a description of what's visible."""
    try:
        capture = _get_capture()
        processor = _get_processor()
        image = await capture.capture(monitor=1)
        desc = await processor.describe_screen(image)
        return desc
    except RuntimeError as exc:
        return f"[screen capture unavailable: {exc}]"
    except Exception as exc:
        log.error("describe_screen failed: %s", exc)
        return f"[error describing screen: {exc}]"


async def capture_region(x: str, y: str, width: str, height: str) -> str:
    """Capture a specific screen region and extract text from it.

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        width: Width of the region in pixels.
        height: Height of the region in pixels.
    """
    try:
        ix, iy, iw, ih = int(x), int(y), int(width), int(height)
    except (ValueError, TypeError):
        return "[error: x, y, width, height must be integers]"

    if iw <= 0 or ih <= 0:
        return "[error: width and height must be positive]"

    try:
        capture = _get_capture()
        processor = _get_processor()
        image = await capture.capture_region(ix, iy, iw, ih)
        text = await processor.extract_text(image)
        return text
    except RuntimeError as exc:
        return f"[screen capture unavailable: {exc}]"
    except Exception as exc:
        log.error("capture_region failed: %s", exc)
        return f"[error capturing region: {exc}]"


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


def register_vision_tools(registry: ToolRegistry) -> None:
    """Register all vision/screen tools into the tool registry."""
    registry.register(
        name="get_screen_text",
        description="Capture the screen and extract all visible text using OCR.",
        parameters={},
        fn=get_screen_text,
    )
    registry.register(
        name="describe_screen",
        description="Capture the screen and describe what's visible (resolution, colors, text).",
        parameters={},
        fn=describe_screen,
    )
    registry.register(
        name="capture_region",
        description="Capture a specific region of the screen and extract text from it.",
        parameters={
            "x": "Left coordinate (pixels)",
            "y": "Top coordinate (pixels)",
            "width": "Width of region (pixels)",
            "height": "Height of region (pixels)",
        },
        fn=capture_region,
    )
    log.info("Vision tools registered")
