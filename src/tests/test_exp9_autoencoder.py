from __future__ import annotations

import unittest

import numpy as np

from core.detectors import ALL_DETECTORS, MLPAutoencoderDet


def _health_data(seed: int = 5):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(0.0, 0.3, size=(60, 6))
    X_cal = rng.normal(0.0, 0.3, size=(20, 6))
    X_fault = rng.normal(5.0, 0.3, size=(15, 6))
    return X_train, X_cal, X_fault


class MLPAutoencoderDetTests(unittest.TestCase):
    def test_registered_in_all_detectors(self):
        self.assertIn(MLPAutoencoderDet, ALL_DETECTORS)
        self.assertEqual(MLPAutoencoderDet.name, "mlp_autoencoder")

    def test_seed_reproducible(self):
        X_train, X_cal, X_fault = _health_data()
        first = MLPAutoencoderDet(confidence=0.95, seed=11).fit(X_train, X_cal)
        second = MLPAutoencoderDet(confidence=0.95, seed=11).fit(X_train, X_cal)
        np.testing.assert_allclose(first.score(X_fault), second.score(X_fault))
        self.assertEqual(first.threshold, second.threshold)

    def test_unknown_data_never_influences_fit_or_threshold(self):
        X_train, X_cal, X_fault = _health_data()
        detector = MLPAutoencoderDet(confidence=0.95, seed=11).fit(X_train, X_cal)
        threshold_before = detector.threshold
        detector.score(X_fault)  # scoring unknown data must not mutate the fitted model
        self.assertEqual(detector.threshold, threshold_before)

    def test_fault_scores_exceed_healthy_scores(self):
        X_train, X_cal, X_fault = _health_data()
        detector = MLPAutoencoderDet(confidence=0.95, seed=11).fit(X_train, X_cal)
        healthy_scores = detector.score(X_cal)
        fault_scores = detector.score(X_fault)
        self.assertLess(np.median(healthy_scores), np.median(fault_scores))


if __name__ == "__main__":
    unittest.main()
