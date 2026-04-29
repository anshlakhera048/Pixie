"""Vision processor — OCR and image analysis.

Extracts text from screen captures via pytesseract and provides
basic image description capabilities.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

# Max characters to return from OCR to avoid overloading context
MAX_OCR_CHARS = 4000


class VisionProcessor:
    """OCR and basic vision analysis for screen images.

    Uses pytesseract for text extraction.  All heavy operations run
    in the thread pool executor.
    """

    def __init__(self) -> None:
        self._tesseract_available: bool | None = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def _check_tesseract(self) -> bool:
        """Check if pytesseract + tesseract binary are available."""
        if self._tesseract_available is not None:
            return self._tesseract_available

        try:
            import pytesseract
            # Quick validation that the binary exists
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except ImportError:
            log.warning("pytesseract not installed")
            self._tesseract_available = False
        except Exception:
            log.warning("Tesseract binary not found or not configured")
            self._tesseract_available = False

        return self._tesseract_available

    # ------------------------------------------------------------------
    # Synchronous implementations (for executor)
    # ------------------------------------------------------------------

    def _extract_text_sync(self, image: Any) -> str:
        """Extract text from PIL Image using OCR."""
        if not self._check_tesseract():
            return self._fallback_describe(image)

        import pytesseract

        try:
            text = pytesseract.image_to_string(image)
            text = text.strip()
            if len(text) > MAX_OCR_CHARS:
                text = text[:MAX_OCR_CHARS] + "\n\n[... truncated ...]"
            return text if text else "[no text detected on screen]"
        except Exception as exc:
            log.error("OCR failed: %s", exc)
            return f"[OCR error: {exc}]"

    def _describe_screen_sync(self, image: Any) -> str:
        """Produce a textual description of the screen content.

        Combines OCR text with basic image metadata.
        """
        from PIL import Image

        width, height = image.size
        description_parts = [
            f"Screen resolution: {width}x{height}",
        ]

        # Get dominant colors for basic visual context
        try:
            small = image.resize((50, 50))
            colors = small.getcolors(maxcolors=2500)
            if colors:
                colors.sort(key=lambda c: c[0], reverse=True)
                top_colors = colors[:3]
                color_desc = ", ".join(
                    f"RGB({c[1][0]},{c[1][1]},{c[1][2]})" for c in top_colors
                )
                description_parts.append(f"Dominant colors: {color_desc}")
        except Exception:
            pass

        # OCR text
        text = self._extract_text_sync(image)
        if text and not text.startswith("["):
            description_parts.append(f"\nVisible text:\n{text}")
        elif text.startswith("["):
            description_parts.append(f"\n{text}")

        return "\n".join(description_parts)

    @staticmethod
    def _fallback_describe(image: Any) -> str:
        """Basic description when OCR is unavailable."""
        try:
            width, height = image.size
            return (
                f"[OCR unavailable] Screen capture: {width}x{height} pixels. "
                "Install pytesseract and Tesseract for text extraction."
            )
        except Exception:
            return "[unable to process image]"

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def extract_text(self, image: Any) -> str:
        """Extract text from image asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._extract_text_sync, image)

    async def describe_screen(self, image: Any) -> str:
        """Describe screen content asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._describe_screen_sync, image)

    # ------------------------------------------------------------------
    # Sync interface (for tool registry)
    # ------------------------------------------------------------------

    def extract_text_sync(self, image: Any) -> str:
        """Synchronous text extraction."""
        return self._extract_text_sync(image)

    def describe_screen_sync(self, image: Any) -> str:
        """Synchronous screen description."""
        return self._describe_screen_sync(image)
