"""Image preprocessing — composable transforms applied before VLM calls."""

from __future__ import annotations

import io
import logging
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from doc2md.types import ImageQuality, PreprocessStep

logger = logging.getLogger(__name__)

# Registry of preprocessing functions
_PREPROCESS_REGISTRY: dict[str, Callable[..., bytes]] = {}


def _register(name: str) -> Callable:
    """Decorator to register an image preprocessing function."""
    def decorator(fn: Callable[..., bytes]) -> Callable[..., bytes]:
        _PREPROCESS_REGISTRY[name] = fn
        return fn
    return decorator


def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Individual transforms ──


@_register("deskew")
def deskew(image_bytes: bytes, **kwargs: Any) -> bytes:
    """Deskew an image by detecting dominant edge angle.

    Uses a simple projection-profile approach with PIL.
    For small skew angles only (±15 degrees).
    """
    img = _bytes_to_pil(image_bytes).convert("L")
    arr = np.array(img, dtype=np.float64)

    # Try a range of small angles and find the one that maximizes
    # the variance of row sums (= best horizontal alignment).
    best_angle = 0.0
    best_score = -1.0

    for angle_10x in range(-150, 151, 5):  # -15.0 to +15.0 in 0.5 steps
        angle = angle_10x / 10.0
        rotated = Image.fromarray(arr).rotate(angle, fillcolor=255)
        row_sums = np.sum(np.array(rotated), axis=1)
        score = float(np.var(row_sums))
        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.5:
        return image_bytes  # No significant skew

    img_color = _bytes_to_pil(image_bytes)
    corrected = img_color.rotate(best_angle, expand=True, fillcolor=(255, 255, 255))
    return _pil_to_bytes(corrected)


@_register("enhance_contrast")
def enhance_contrast(image_bytes: bytes, factor: float = 1.5, **kwargs: Any) -> bytes:
    """Enhance image contrast by the given factor."""
    img = _bytes_to_pil(image_bytes)
    enhancer = ImageEnhance.Contrast(img)
    enhanced = enhancer.enhance(factor)
    return _pil_to_bytes(enhanced)


@_register("binarize")
def binarize(image_bytes: bytes, threshold: int = 128, **kwargs: Any) -> bytes:
    """Convert image to binary (black and white) using a threshold."""
    img = _bytes_to_pil(image_bytes).convert("L")
    binary = img.point(lambda p: 255 if p > threshold else 0)
    return _pil_to_bytes(binary.convert("RGB"))


@_register("resize")
def resize(image_bytes: bytes, max_dimension: int = 2048, **kwargs: Any) -> bytes:
    """Resize image so its longest side is at most max_dimension."""
    img = _bytes_to_pil(image_bytes)
    w, h = img.size
    if max(w, h) <= max_dimension:
        return image_bytes

    scale = max_dimension / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    resized = img.resize(new_size, Image.LANCZOS)
    return _pil_to_bytes(resized)


@_register("denoise")
def denoise(image_bytes: bytes, strength: int = 10, **kwargs: Any) -> bytes:
    """Denoise image using OpenCV's fastNlMeansDenoising.

    Falls back to a PIL median filter if OpenCV is not installed.
    """
    try:
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        denoised = cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
        _, buf = cv2.imencode(".png", denoised)
        return buf.tobytes()
    except ImportError:
        logger.debug("OpenCV not available, using PIL median filter for denoising")
        img = _bytes_to_pil(image_bytes)
        filtered = img.filter(ImageFilter.MedianFilter(size=3))
        return _pil_to_bytes(filtered)


@_register("crop_margins")
def crop_margins(image_bytes: bytes, padding: int = 10, **kwargs: Any) -> bytes:
    """Crop white margins from an image."""
    img = _bytes_to_pil(image_bytes).convert("RGB")
    gray = img.convert("L")
    arr = np.array(gray)

    # Find bounding box of non-white content
    mask = arr < 250
    if not mask.any():
        return image_bytes  # All white, nothing to crop

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding
    h, w = arr.shape
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)

    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
    return _pil_to_bytes(cropped)


@_register("upscale")
def upscale(image_bytes: bytes, factor: float = 2.0, **kwargs: Any) -> bytes:
    """Upscale image by a given factor."""
    img = _bytes_to_pil(image_bytes)
    w, h = img.size
    new_size = (int(w * factor), int(h * factor))
    upscaled = img.resize(new_size, Image.LANCZOS)
    return _pil_to_bytes(upscaled)


@_register("sharpen")
def sharpen(image_bytes: bytes, **kwargs: Any) -> bytes:
    """Sharpen an image using PIL."""
    img = _bytes_to_pil(image_bytes)
    sharpened = img.filter(ImageFilter.SHARPEN)
    return _pil_to_bytes(sharpened)


def compute_quality(image_bytes: bytes) -> ImageQuality:
    """Compute image quality metrics. Returns an ImageQuality model."""
    img = _bytes_to_pil(image_bytes)
    w, h = img.size
    pixels = w * h

    # Resolution/DPI
    dpi = 300
    try:
        dpi_info = img.info.get("dpi")
        if dpi_info and isinstance(dpi_info, tuple):
            dpi = int(dpi_info[0])
    except Exception:
        pass

    # Resolution score
    if pixels >= 2_000_000:
        res_score = 1.0
    elif pixels >= 500_000:
        res_score = 0.7
    elif pixels >= 100_000:
        res_score = 0.4
    else:
        res_score = 0.2

    # Contrast (std dev of grayscale)
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float64)
    std_dev = float(np.std(arr))
    if std_dev >= 50:
        contrast = 1.0
    elif std_dev >= 30:
        contrast = 0.7
    elif std_dev >= 15:
        contrast = 0.4
    else:
        contrast = 0.2

    # Blur (Laplacian variance)
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        laplacian = (
            arr[:-2, 1:-1] + arr[2:, 1:-1]
            + arr[1:-1, :-2] + arr[1:-1, 2:]
            - 4 * arr[1:-1, 1:-1]
        )
        variance = float(np.var(laplacian))
        if variance >= 500:
            blur = 1.0
        elif variance >= 100:
            blur = 0.7
        elif variance >= 20:
            blur = 0.4
        else:
            blur = 0.2
    else:
        blur = 0.5

    # Noise (inverse of SNR estimate — high noise = low score)
    noise = 1.0 - min(1.0, float(np.mean(np.abs(np.diff(arr, axis=0)))) / 50.0)
    noise = max(0.2, noise)

    overall = (res_score + contrast + blur + noise) / 4.0

    return ImageQuality(
        blur_score=blur,
        contrast_score=contrast,
        resolution_dpi=dpi,
        noise_score=noise,
        overall=overall,
    )


# ── Orchestrator ──


def run_preprocessing(
    image_bytes: bytes,
    steps: list[PreprocessStep],
) -> tuple[bytes, ImageQuality]:
    """Run a sequence of preprocessing steps on an image.

    Returns (processed_image_bytes, quality_metrics).
    """
    current = image_bytes
    for step in steps:
        fn = _PREPROCESS_REGISTRY.get(step.name)
        if fn is None:
            logger.warning("Unknown preprocessing step '%s', skipping", step.name)
            continue
        try:
            current = fn(current, **step.params)
        except Exception as e:
            logger.warning("Preprocessing step '%s' failed: %s, skipping", step.name, e)

    quality = compute_quality(current)
    return current, quality
