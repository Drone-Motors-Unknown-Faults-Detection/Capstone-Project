import unittest

import numpy as np

from experiments.health.index import CalibratedHealthIndex


class HealthIndexTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X_train = rng.normal(0.0, 0.4, size=(60, 4))
        self.X_cal = rng.normal(0.0, 0.4, size=(20, 4))
        self.y_train = np.zeros(60, dtype=int)
        self.y_cal = np.zeros(20, dtype=int)
        self.X_probe = np.vstack(
            [
                self.X_cal[:2],
                np.full((2, 4), 3.0),
            ]
        )

    def test_health_index_is_bounded_and_unknown_is_explicit(self):
        model = CalibratedHealthIndex.fit(
            self.X_train,
            self.y_train,
            self.X_cal,
            self.y_cal,
            openset_method="mahalanobis",
        )
        results = model.predict(self.X_probe, condition="T1/8000rpm")
        self.assertEqual(len(results), 4)
        self.assertTrue(all(0.0 <= item.health_index <= 1.0 for item in results))
        self.assertTrue(all(item.estimated_rul is None for item in results))
        self.assertEqual(results[-1].fault_type, "unknown")
        self.assertTrue(results[-1].is_unknown_fault)
        self.assertEqual(results[-1].to_dict("binary"), {"is_fault": 1})

    def test_knn_uses_same_output_contract(self):
        model = CalibratedHealthIndex.fit(
            self.X_train,
            self.y_train,
            self.X_cal,
            self.y_cal,
            openset_method="knn",
            knn_neighbors=3,
        )
        result = model.predict_one(self.X_probe[0], condition="T2/10000rpm")
        self.assertEqual(result.openset_method, "knn")
        self.assertAlmostEqual(result.health_index + result.degradation_score, 1.0)
        self.assertEqual(result.trend, "insufficient_history")

    def test_calibration_cannot_use_unknown_label(self):
        with self.assertRaises(ValueError):
            CalibratedHealthIndex.fit(
                self.X_train,
                self.y_train,
                self.X_cal,
                np.ones(len(self.X_cal), dtype=int),
            )

    def test_fit_is_reproducible(self):
        first = CalibratedHealthIndex.fit(self.X_train, self.y_train, self.X_cal, self.y_cal)
        second = CalibratedHealthIndex.fit(self.X_train, self.y_train, self.X_cal, self.y_cal)
        first_scores = [row.openset_score for row in first.predict(self.X_probe, condition="x")]
        second_scores = [row.openset_score for row in second.predict(self.X_probe, condition="x")]
        np.testing.assert_allclose(first_scores, second_scores)
