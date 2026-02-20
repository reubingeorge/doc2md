"""Image loading and conversion utilities."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

_SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def load_image(path: str | Path) -> bytes:
    """Load an image file and return raw bytes."""
    path = Path(path)
    _validate_path(path)
    return path.read_bytes()


def image_to_base64(image_bytes: bytes) -> str:
    """Encode raw image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("ascii")


def pdf_to_images(path: str | Path, dpi: int = 200) -> list[bytes]:
    """Convert a PDF file to a list of PNG image byte arrays (one per page)."""
    from pdf2image import convert_from_path

    path = Path(path)
    _validate_path(path)

    pil_images = convert_from_path(str(path), dpi=dpi)
    result: list[bytes] = []
    for img in pil_images:
        result.append(_pil_to_png_bytes(img))
    return result


def is_pdf(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _validate_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_symlink():
        raise ValueError(f"Symlinks not allowed: {path}")
    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    size = path.stat().st_size
    if size > _MAX_IMAGE_SIZE_BYTES:
        raise ValueError(f"File too large ({size} bytes, max {_MAX_IMAGE_SIZE_BYTES})")
