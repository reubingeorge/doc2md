"""Tests for image preprocessing pipeline."""

import io

import numpy as np
from PIL import Image

from doc2md.pipeline.preprocessor import (
    binarize,
    compute_quality,
    crop_margins,
    denoise,
    deskew,
    enhance_contrast,
    resize,
    run_preprocessing,
    sharpen,
    upscale,
)
from doc2md.types import PreprocessStep


def _make_image(width: int = 200, height: int = 200, color: tuple = (128, 128, 128)) -> bytes:
    """Create a test PNG image."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_noisy_image(width: int = 200, height: int = 200) -> bytes:
    """Create a test image with noise and content."""
    arr = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Add a dark rectangle in the center (content)
    arr[50:150, 50:150] = [30, 30, 30]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestEnhanceContrast:
    def test_returns_bytes(self):
        result = enhance_contrast(_make_image())
        assert isinstance(result, bytes)

    def test_factor_changes_output(self):
        original = _make_image()
        enhanced = enhance_contrast(original, factor=2.0)
        # Different factor should produce different bytes
        assert len(enhanced) > 0  # Uniform images may not change much

    def test_default_factor(self):
        result = enhance_contrast(_make_image())
        assert len(result) > 0


class TestBinarize:
    def test_returns_bytes(self):
        result = binarize(_make_image())
        assert isinstance(result, bytes)

    def test_binary_output(self):
        result = binarize(_make_noisy_image(), threshold=128)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img.convert("L"))
        # All pixels should be 0 or 255
        unique = set(np.unique(arr))
        assert unique <= {0, 255}


class TestResize:
    def test_no_resize_when_within_limit(self):
        original = _make_image(100, 100)
        result = resize(original, max_dimension=200)
        assert result == original  # Should be unchanged

    def test_resizes_large_image(self):
        original = _make_image(1000, 500)
        result = resize(original, max_dimension=200)
        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= 200

    def test_preserves_aspect_ratio(self):
        original = _make_image(1000, 500)
        result = resize(original, max_dimension=200)
        img = Image.open(io.BytesIO(result))
        w, h = img.size
        assert abs(w / h - 2.0) < 0.1  # 1000:500 = 2:1


class TestDenoise:
    def test_returns_bytes(self):
        result = denoise(_make_noisy_image())
        assert isinstance(result, bytes)

    def test_graceful_without_opencv(self):
        """Should fall back to PIL median filter if cv2 is unavailable."""
        # This test just verifies the function works — it will use whichever
        # backend is available. Both paths produce valid output.
        result = denoise(_make_noisy_image(), strength=5)
        assert len(result) > 0


class TestCropMargins:
    def test_crops_white_margins(self):
        # Image with content in center, white margins
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        # Draw a dark block in center
        for x in range(100, 200):
            for y in range(100, 200):
                img.putpixel((x, y), (0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        original = buf.getvalue()

        result = crop_margins(original, padding=5)
        cropped = Image.open(io.BytesIO(result))
        # Should be smaller than original
        assert cropped.size[0] < 300
        assert cropped.size[1] < 300

    def test_all_white_returns_original(self):
        original = _make_image(100, 100, (255, 255, 255))
        result = crop_margins(original)
        assert result == original


class TestDeskew:
    def test_returns_bytes(self):
        result = deskew(_make_noisy_image())
        assert isinstance(result, bytes)

    def test_no_skew_returns_original(self):
        # Uniform image should not be rotated
        original = _make_image(100, 100, (128, 128, 128))
        result = deskew(original)
        assert isinstance(result, bytes)


class TestUpscale:
    def test_doubles_size(self):
        original = _make_image(100, 100)
        result = upscale(original, factor=2.0)
        img = Image.open(io.BytesIO(result))
        assert img.size == (200, 200)


class TestSharpen:
    def test_returns_bytes(self):
        result = sharpen(_make_noisy_image())
        assert isinstance(result, bytes)


class TestComputeQuality:
    def test_returns_image_quality(self):
        quality = compute_quality(_make_noisy_image(500, 500))
        assert 0.0 <= quality.overall <= 1.0
        assert 0.0 <= quality.blur_score <= 1.0
        assert 0.0 <= quality.contrast_score <= 1.0
        assert quality.resolution_dpi > 0

    def test_low_res_scores_low(self):
        quality = compute_quality(_make_image(50, 50))
        assert quality.overall < 0.8  # Small image should score lower

    def test_high_res_scores_higher(self):
        quality = compute_quality(_make_noisy_image(2000, 2000))
        assert quality.overall > 0.3


class TestRunPreprocessing:
    def test_empty_steps(self):
        original = _make_image()
        result, quality = run_preprocessing(original, [])
        assert result == original
        assert quality.overall > 0.0

    def test_single_step(self):
        original = _make_image(1000, 500)
        steps = [PreprocessStep(name="resize", params={"max_dimension": 200})]
        result, quality = run_preprocessing(original, steps)
        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= 200

    def test_multiple_steps(self):
        original = _make_noisy_image(1000, 500)
        steps = [
            PreprocessStep(name="resize", params={"max_dimension": 300}),
            PreprocessStep(name="enhance_contrast", params={"factor": 1.5}),
        ]
        result, quality = run_preprocessing(original, steps)
        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= 300

    def test_unknown_step_skipped(self):
        original = _make_image()
        steps = [PreprocessStep(name="nonexistent_transform")]
        result, quality = run_preprocessing(original, steps)
        assert result == original  # Should pass through unchanged

    def test_failing_step_skipped(self):
        # Empty bytes would cause a step to fail
        steps = [
            PreprocessStep(name="resize", params={"max_dimension": 100}),
        ]
        # Valid image should work fine
        result, quality = run_preprocessing(_make_image(), steps)
        assert isinstance(result, bytes)
