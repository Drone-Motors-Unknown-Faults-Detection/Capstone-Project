import unittest

import numpy as np

from experiments.health.calibration import HealthIndexCalibrator


class HealthCalibrationTests(unittest.TestCase):
    def test_fit_is_monotone_and_bounded(self):
        calibrator = HealthIndexCalibrator.fit(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        health = calibrator.transform(np.array([0.1, 0.3, 0.5, 2.0]))
        self.assertTrue(np.all((health >= 0.0) & (health <= 1.0)))
        self.assertTrue(np.all(np.diff(health) <= 0.0))
        self.assertEqual(float(health[-1]), 0.0)

    def test_rejection_line_maps_to_warning_health_and_twice_line_to_zero(self):
        """q10 → 1.0，q95（拒絕線）→ 0.5，2×拒絕線 → 0.0，兩段線性。"""
        scores = np.linspace(0.0, 1.0, 101)
        calibrator = HealthIndexCalibrator.fit(scores, confidence=0.95, lower_quantile=0.10)
        self.assertAlmostEqual(calibrator.healthy_anchor, 0.10)
        self.assertAlmostEqual(calibrator.warning_anchor, 0.95)
        self.assertAlmostEqual(calibrator.critical_anchor, 1.90)
        a, b, c = calibrator.healthy_anchor, calibrator.warning_anchor, calibrator.critical_anchor
        probes = np.array([0.0, a, (a + b) / 2, b, (b + c) / 2, c, 5.0])
        np.testing.assert_allclose(calibrator.transform(probes), [1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0])

    def test_healthy_population_lands_mostly_above_warning_health(self):
        """cal 自身回代：confidence 分位以下的分數都在 warning_health 以上。"""
        rng = np.random.default_rng(42)
        scores = rng.gamma(4.0, 0.2, size=500)
        calibrator = HealthIndexCalibrator.fit(scores)
        health = calibrator.transform(scores)
        self.assertGreaterEqual(float(np.mean(health >= 0.5)), 0.95)
        self.assertGreaterEqual(float(np.mean(health >= 1.0)), 0.10)

    def test_calibration_parameters_are_serializable(self):
        calibrator = HealthIndexCalibrator.fit(np.array([0.1, 0.2, 0.3]))
        restored = HealthIndexCalibrator.from_dict(calibrator.to_dict())
        self.assertEqual(restored, calibrator)
        np.testing.assert_allclose(calibrator.transform([0.2, 0.4]), restored.transform([0.2, 0.4]))

    def test_degenerate_calibration_keeps_anchor_order(self):
        calibrator = HealthIndexCalibrator.fit(np.zeros(5))
        self.assertLess(calibrator.healthy_anchor, calibrator.warning_anchor)
        self.assertLess(calibrator.warning_anchor, calibrator.critical_anchor)
        self.assertEqual(float(calibrator.transform(0.0)), 1.0)

    def test_invalid_or_empty_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            HealthIndexCalibrator.fit(np.array([]))
        with self.assertRaises(ValueError):
            HealthIndexCalibrator.fit(np.array([0.1, np.nan]))
        with self.assertRaises(ValueError):
            HealthIndexCalibrator.fit(np.array([0.1, 0.2]), critical_multiple=1.0)
        with self.assertRaises(ValueError):
            HealthIndexCalibrator(0.95, 0.1, 0.5, 0.4, 1.0, 3)
