from __future__ import annotations

import argparse
import unittest

import numpy as np

from core.monitor import OpenSetMonitor
from core.mahalanobis import MahalanobisOpenSetDetector
from core.openset import (KNNOpenSetDetector, OpenSetConfigError, canonical_openset_method, create_openset_detector,
                          resolve_openset_config)
from core.runner import add_openset_args
from experiments.compare_openset import run as run_comparison
from experiments.exp1_cold_start import run as run_exp1


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


class ResolvedConfigTests(unittest.TestCase):
    """CLI、run() 與 detector 共用同一份 resolved config。"""

    def test_default_is_mahalanobis_ledoit_wolf(self):
        resolved = resolve_openset_config()
        self.assertEqual(resolved["version"], "openset_config_v1")
        self.assertEqual(resolved["method_id"], "mahalanobis/ledoit_wolf")
        self.assertEqual(resolved["effective"], {"confidence": 0.95, "threshold": 1.0, "mahalanobis_method": "ledoit_wolf"})
        self.assertEqual([item["parameter"] for item in resolved["ignored"]], ["knn_neighbors"])

    def test_knn_records_covariance_option_as_ignored(self):
        resolved = resolve_openset_config("k-nn", mahalanobis_method="oas", knn_neighbors=3)
        self.assertEqual((resolved["method_id"], resolved["openset_method"], resolved["requested_method"]),
                         ("knn/k3", "knn", "k-nn"))
        self.assertEqual(resolved["effective"], {"confidence": 0.95, "threshold": 1.0, "knn_neighbors": 3})
        self.assertEqual(resolved["ignored"][0]["parameter"], "mahalanobis_method")
        self.assertEqual(resolved["ignored"][0]["value"], "oas")

    def test_covariance_option_does_not_change_knn_scores(self):
        X_train, y_train, X_cal, y_cal = detector_data()
        query = np.array([[0.1, 0.1], [2.0, 2.0], [10.0, 10.0]])
        scores = [
            create_openset_detector("knn", mahalanobis_method=method, knn_neighbors=2)
            .fit(X_train, y_train, X_cal, y_cal).score_samples(query)
            for method in ("ledoit_wolf", "oas", "mcd")
        ]
        np.testing.assert_array_equal(scores[0], scores[1])
        np.testing.assert_array_equal(scores[0], scores[2])

    def test_invalid_values_have_fixed_errors(self):
        cases = (
            ({"openset_method": "energy"}, "^openset_method: "),
            ({"confidence": 1.0}, "^confidence: "),
            ({"confidence": float("nan")}, "^confidence: "),
            ({"confidence": True}, "^confidence: "),
            ({"confidence": "0.9"}, "^confidence: "),
            ({"knn_neighbors": 0}, "^knn_neighbors: "),
            ({"knn_neighbors": 5.7}, "^knn_neighbors: "),
            ({"knn_neighbors": True}, "^knn_neighbors: "),
            ({"mahalanobis_method": "shrunk"}, "^mahalanobis_method: "),
        )
        for kwargs, pattern in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(OpenSetConfigError, pattern):
                    resolve_openset_config(**kwargs)
                with self.assertRaisesRegex(OpenSetConfigError, pattern):
                    create_openset_detector(**kwargs)
                monitor_kwargs = {"method" if k == "mahalanobis_method" else k: v for k, v in kwargs.items()}
                with self.assertRaisesRegex(OpenSetConfigError, pattern):
                    OpenSetMonitor(synthetic_pools(), **monitor_kwargs)

    def test_dimension_mismatch_has_fixed_error_for_both_detectors(self):
        X_train, y_train, X_cal, y_cal = detector_data()
        for method in ("mahalanobis", "knn"):
            with self.subTest(method=method):
                detector = create_openset_detector(method, knn_neighbors=2).fit(X_train, y_train, X_cal, y_cal)
                with self.assertRaisesRegex(ValueError, "X has 3 features; expected 2"):
                    detector.score_samples(np.zeros((1, 3)))

    def test_alias_no_longer_mislabels_monitor_score_type(self):
        monitor = OpenSetMonitor(synthetic_pools(), seed=3, openset_method="maha")
        monitor.fit_initial()
        summary = monitor.summary()
        self.assertEqual(summary["openset_method"], "mahalanobis")
        self.assertEqual(summary["score_type"], "normalized_class_mahalanobis")
        self.assertEqual(summary["resolved_config"]["requested_method"], "maha")

    def test_cli_and_run_record_the_same_resolved_config(self):
        parser = argparse.ArgumentParser()
        add_openset_args(parser)
        pools = synthetic_pools()
        for argv in ([], ["--openset-method", "knn", "--knn-neighbors", "3", "--confidence", "0.9"],
                     ["--method", "oas"]):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                expected = resolve_openset_config(args.openset_method, confidence=args.confidence,
                                                  mahalanobis_method=args.method, knn_neighbors=args.knn_neighbors)
                result = run_exp1(pools, seed=42, confidence=args.confidence, method=args.method,
                                  openset_method=args.openset_method, knn_neighbors=args.knn_neighbors)
                self.assertEqual(result["model"]["resolved_config"], expected)

    def test_resolved_config_does_not_change_scores(self):
        pools = synthetic_pools()
        monitor = OpenSetMonitor(pools, seed=11)
        monitor.fit_initial()
        scaled = monitor.scaler.transform(pools["7screws"])
        direct = MahalanobisOpenSetDetector(method="ledoit_wolf", confidence=0.95)
        split = monitor.splits["8screws"]
        direct.fit(monitor.scaler.transform(pools["8screws"][split.train]), np.zeros(len(split.train)),
                   monitor.scaler.transform(pools["8screws"][split.cal]), np.zeros(len(split.cal)))
        np.testing.assert_array_equal(monitor.detector.score_samples(scaled), direct.score_samples(scaled))


class FprAtTprTests(unittest.TestCase):
    """#235：三支 runner 共用的 ROC 定義。"""

    def test_separable_scores_give_zero(self):
        from core.openset import fpr_at_tpr

        self.assertEqual(fpr_at_tpr(np.array([0.2, 0.4]), np.array([2.0, 3.0, 4.0, 5.0])), 0.0)

    def test_ties_at_threshold_count_as_false_positive(self):
        from core.openset import fpr_at_tpr

        # 需要接受分數 2.0 的 unknown 才達 TPR 100%，同分的 known 也被一起判為 unknown
        self.assertEqual(fpr_at_tpr(np.array([2.0, 0.1]), np.array([2.0, 3.0]), tpr=1.0), 0.5)

    def test_runners_share_definition(self):
        from core.openset import fpr_at_tpr
        from experiments import compare_openset, exp6_formal_benchmark

        known, unknown = np.array([0.5, 1.5, 2.5]), np.array([1.0, 2.0, 3.0, 4.0])
        expected = fpr_at_tpr(known, unknown)
        self.assertEqual(exp6_formal_benchmark._fpr_at_tpr95(known, unknown), expected)
        y = np.r_[np.zeros(3), np.ones(4)]
        self.assertEqual(compare_openset._fpr_at_tpr95(y, np.r_[known, unknown]), expected)


if __name__ == "__main__":
    unittest.main()
