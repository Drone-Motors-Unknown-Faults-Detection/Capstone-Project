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

    def test_calibration_parameters_are_serializable(self):
        calibrator = HealthIndexCalibrator.fit(np.array([0.1, 0.2, 0.3]))
        restored = HealthIndexCalibrator.from_dict(calibrator.to_dict())
        np.testing.assert_allclose(calibrator.transform([0.2]), restored.transform([0.2]))

    def test_invalid_or_empty_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            HealthIndexCalibrator.fit(np.array([]))
        with self.assertRaises(ValueError):
            HealthIndexCalibrator.fit(np.array([0.1, np.nan]))

