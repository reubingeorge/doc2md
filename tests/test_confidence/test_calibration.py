"""Tests for calibration curves."""

from doc2md.confidence.calibration import _manual_calibrate, calibrate


class TestCalibrate:
    def test_none_method_passthrough(self):
        assert calibrate(0.8, method="none") == 0.8

    def test_manual_with_curve(self):
        curve = [[0.5, 0.4], [0.7, 0.65], [0.9, 0.85]]
        result = calibrate(0.6, method="manual", manual_curve=curve)
        # Linear interpolation between (0.5, 0.4) and (0.7, 0.65)
        # t = (0.6 - 0.5) / (0.7 - 0.5) = 0.5
        # result = 0.4 + 0.5 * (0.65 - 0.4) = 0.525
        assert abs(result - 0.525) < 0.01

    def test_manual_without_curve_passthrough(self):
        assert calibrate(0.8, method="manual") == 0.8

    def test_platt_falls_back_to_manual(self):
        # Without trained model, falls back to manual
        curve = [[0.5, 0.4], [0.9, 0.85]]
        result = calibrate(0.7, method="platt_scaling", manual_curve=curve)
        assert result != 0.7  # Should be calibrated


class TestManualCalibrate:
    def test_exact_point(self):
        curve = [[0.5, 0.4], [0.8, 0.75]]
        assert abs(_manual_calibrate(0.5, curve) - 0.4) < 0.001

    def test_below_range(self):
        curve = [[0.3, 0.2], [0.8, 0.7]]
        assert _manual_calibrate(0.1, curve) == 0.2  # Clamped to first point

    def test_above_range(self):
        curve = [[0.3, 0.2], [0.8, 0.7]]
        assert _manual_calibrate(0.95, curve) == 0.7  # Clamped to last point

    def test_empty_curve_passthrough(self):
        assert _manual_calibrate(0.75, []) == 0.75

    def test_single_point(self):
        # Below and at single point
        assert _manual_calibrate(0.3, [[0.5, 0.4]]) == 0.4
        assert _manual_calibrate(0.5, [[0.5, 0.4]]) == 0.4
        assert _manual_calibrate(0.9, [[0.5, 0.4]]) == 0.4

    def test_interpolation_midpoint(self):
        curve = [[0.0, 0.0], [1.0, 0.8]]
        # At midpoint: t = 0.5, result = 0 + 0.5 * 0.8 = 0.4
        assert abs(_manual_calibrate(0.5, curve) - 0.4) < 0.001
