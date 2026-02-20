"""Image quality signal — pre-VLM image analysis."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_image_quality(
    image_bytes: bytes | None,
) -> tuple[float, bool, str]:
    """Analyze image quality and return a confidence signal.

    Computes: resolution, contrast, sharpness, DPI.
    Uses PIL (Pillow) for analysis.

    Returns (score, available, reasoning).
    """
    if image_bytes is None:
        return 0.0, False, "No image provided"

    try:
        import io

        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        scores: dict[str, float] = {}
        notes: list[str] = []

        # Resolution score
        width, height = img.size
        pixels = width * height
        if pixels >= 2_000_000:
            scores["resolution"] = 1.0
        elif pixels >= 500_000:
            scores["resolution"] = 0.7
        elif pixels >= 100_000:
            scores["resolution"] = 0.4
            notes.append("low resolution")
        else:
            scores["resolution"] = 0.2
            notes.append("very low resolution")

        # Contrast score (std dev of grayscale pixel values)
        gray = img.convert("L")
        pixels_arr = np.array(gray, dtype=np.float64)
        std_dev = float(np.std(pixels_arr))
        if std_dev >= 50:
            scores["contrast"] = 1.0
        elif std_dev >= 30:
            scores["contrast"] = 0.7
        elif std_dev >= 15:
            scores["contrast"] = 0.4
            notes.append("low contrast")
        else:
            scores["contrast"] = 0.2
            notes.append("very low contrast")

        # Blur detection (variance of Laplacian-like edge filter)
        blur_score = _estimate_blur(pixels_arr)
        scores["sharpness"] = blur_score
        if blur_score < 0.4:
            notes.append("blurry")

        # DPI from metadata
        dpi = _get_dpi(img)
        if dpi:
            if dpi >= 300:
                scores["dpi"] = 1.0
            elif dpi >= 150:
                scores["dpi"] = 0.7
            else:
                scores["dpi"] = 0.4
                notes.append(f"low DPI ({dpi})")
        else:
            scores["dpi"] = 0.7  # Unknown DPI, assume acceptable

        overall = sum(scores.values()) / len(scores)
        reasoning = f"Quality scores: {scores}"
        if notes:
            reasoning += f"; issues: {', '.join(notes)}"

        return overall, True, reasoning

    except Exception as e:
        logger.warning("Image quality analysis failed: %s", e)
        return 0.5, False, f"Analysis failed: {e}"


def _estimate_blur(pixels: Any) -> float:
    """Estimate image sharpness using variance of a simple edge filter."""
    import numpy as np

    if pixels.ndim != 2 or pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return 0.5

    # Compute Laplacian (second-order differences)
    laplacian = (
        pixels[:-2, 1:-1] + pixels[2:, 1:-1]
        + pixels[1:-1, :-2] + pixels[1:-1, 2:]
        - 4 * pixels[1:-1, 1:-1]
    )
    variance = float(np.var(laplacian))

    if variance >= 500:
        return 1.0
    elif variance >= 100:
        return 0.7
    elif variance >= 20:
        return 0.4
    else:
        return 0.2


def _get_dpi(img: Any) -> int | None:
    """Extract DPI from image metadata."""
    try:
        info = img.info
        dpi = info.get("dpi")
        if dpi and isinstance(dpi, tuple):
            return int(dpi[0])
    except Exception:
        pass
    return None
