from __future__ import annotations

import argparse
import unittest

import numpy as np

from core.monitor import OpenSetMonitor
from core.mahalanobis import MahalanobisOpenSetDetector
from core.openset import KNNOpenSetDetector, canonical_openset_method, create_openset_detector
from core.runner import add_openset_args
from experiments.compare_openset import run as run_comparison


def detector_data():
    X_train = np.array(
        [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [4.0, 4.0], [4.2, 4.0], [4.0, 4.2]]
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_cal = np.array([[0.1, 0.1], [0.2, 0.1], [4.1, 4.1], [4.2, 4.1]])
    y_cal = np.array([0, 0, 1, 1])
    return X_train, y_train, X_cal, y_cal


def synthetic_pools(seed: int = 7):
    rng = np.random.default_rng(seed)
    return {
        "8screws": rng.normal(0.0, 0.3, size=(60, 4)),
        "7screws": rng.normal(3.0, 0.3, size=(35, 4)),
        "6screws": rng.normal(-3.0, 0.3, size=(35, 4)),
    }


class KNNDetectorTests(unittest.TestCase):
    def test_fit_score_and_prediction(self):
        X_train, y_train, X_cal, y_cal = detector_data()
        detector = KNNOpenSetDetector(confidence=0.95, n_neighbors=2)
        detector.fit(X_train, y_train, X_cal, y_cal)
        scores = detector.score_samples(np.array([[0.1, 0.1], [10.0, 10.0]]))
        predictions = detector.predict_known_class(np.array([[0.1, 0.1], [10.0, 10.0]]))
        self.assertLessEqual(scores[0], 1.0)
        self.assertGreater(scores[1], 1.0)
        self.assertEqual(predictions.tolist(), [0, -1])

    def test_threshold_boundary_is_known_and_larger_is_unknown(self):
        detector = KNNOpenSetDetector(confidence=0.95, n_neighbors=1).fit(
            np.array([[0.0]]), np.array([0]), np.array([[2.0]]), np.array([0])
        )
        self.assertAlmostEqual(detector.score_samples(np.array([[2.0]]))[0], 1.0)
        self.assertEqual(detector.predict_known_class(np.array([[2.0]]))[0], 0)
        self.assertEqual(detector.predict_known_class(np.array([[2.000001]]))[0], -1)

    def test_empty_scoring_and_single_training_sample(self):
        detector = KNNOpenSetDetector(n_neighbors=5).fit(
            np.array([[0.0, 0.0]]),
            np.array([0]),
            np.array([[1.0, 1.0]]),
            np.array([0]),
        )
        self.assertEqual(detector.class_summaries()[0]["effective_neighbors"], 1)
        self.assertEqual(detector.score_samples(np.empty((0, 2))).shape, (0,))

    def test_invalid_inputs_are_clear(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            KNNOpenSetDetector().fit(
                np.empty((0, 2)), np.empty(0), np.ones((1, 2)), np.array([0])
            )
        with self.assertRaisesRegex(ValueError, "no samples for class"):
            KNNOpenSetDetector().fit(
                np.array([[0.0], [2.0]]),
                np.array([0, 1]),
                np.array([[0.5]]),
                np.array([0]),
            )


class CompatibilityTests(unittest.TestCase):
    def test_public_method_aliases_are_canonicalized(self) -> None:
        self.assertEqual(canonical_openset_method("k-nn"), "knn")
        self.assertEqual(canonical_openset_method("k_nn"), "knn")
        self.assertEqual(canonical_openset_method("maha"), "mahalanobis")

    def test_mahalanobis_factory_is_regression_equivalent(self):
        X_train, y_train, X_cal, y_cal = detector_data()
        direct = MahalanobisOpenSetDetector(method="ledoit_wolf", confidence=0.9)
        explicit = create_openset_detector(
            "mahalanobis", confidence=0.9, mahalanobis_method="ledoit_wolf"
        )
        direct.fit(X_train, y_train, X_cal, y_cal)
        explicit.fit(X_train, y_train, X_cal, y_cal)
        query = np.array([[0.1, 0.1], [4.1, 4.1], [8.0, 8.0]])
        np.testing.assert_allclose(direct.score_samples(query), explicit.score_samples(query))
        np.testing.assert_array_equal(
            direct.predict_known_class(query), explicit.predict_known_class(query)
        )

    def test_default_monitor_is_explicit_mahalanobis(self):
        pools = synthetic_pools()
        default = OpenSetMonitor(pools, seed=42)
        explicit = OpenSetMonitor(pools, seed=42, openset_method="mahalanobis")
        default.fit_initial()
        explicit.fit_initial()
        query = pools["8screws"][:8]
        np.testing.assert_allclose(default.score(query), explicit.score(query))

    def test_singular_covariance_remains_supported(self):
        detector = create_openset_detector("mahalanobis")
        detector.fit(
            np.ones((6, 3)), np.zeros(6), np.ones((2, 3)), np.zeros(2)
        )
        self.assertTrue(np.isfinite(detector.score_samples(np.ones((1, 3)))).all())

    def test_factory_rejects_invalid_method(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Open Set method"):
            create_openset_detector("energy")


class IntegrationTests(unittest.TestCase):
    def test_cli_can_select_both_methods(self):
        parser = argparse.ArgumentParser()
        add_openset_args(parser)
        self.assertEqual(
            parser.parse_args(["--openset-method", "mahalanobis"]).openset_method,
            "mahalanobis",
        )
        args = parser.parse_args(["--openset-method", "knn", "--knn-neighbors", "3"])
        self.assertEqual((args.openset_method, args.knn_neighbors), ("knn", 3))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--openset-method", "invalid"])

    def test_monitor_knn_is_reproducible(self):
        pools = synthetic_pools()
        first = OpenSetMonitor(pools, seed=19, openset_method="knn")
        second = OpenSetMonitor(pools, seed=19, openset_method="knn")
        first.fit_initial()
        second.fit_initial()
        np.testing.assert_allclose(first.score(pools["7screws"]), second.score(pools["7screws"]))
        self.assertEqual(first.summary(), second.summary())

    def test_fair_comparison_emits_required_metrics(self):
        result = run_comparison(synthetic_pools(), seed=42)
        self.assertEqual([r["openset_method"] for r in result["results"]], ["mahalanobis", "knn"])
        required = {
            "auroc",
            "aupr_unknown_positive",
            "fpr_at_95_tpr",
            "open_set_accuracy",
            "known_class_accuracy",
            "unknown_recall",
            "unknown_precision",
            "f1_unknown",
            "threshold",
        }
        for row in result["results"]:
            self.assertTrue(required.issubset(row))
            self.assertEqual(row["threshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
