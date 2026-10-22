"""OpenSetMonitor：legacy 合併分布的摘要列（#234、#266）、classify 對 legacy 回傳最近已知類（#266）、
add_classes 與逐一 add_class 相同、單執行緒擬合與一次算分數＋類別數字相同（#275）、seed 傳到 MCD（#273）。"""
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from core.mahalanobis import LEGACY_POOLED_LABEL, MahalanobisOpenSetDetector
from core.monitor import OpenSetMonitor


def _pools():
    rng = np.random.default_rng(3)
    return {"8screws": rng.normal(0, .3, (80, 6)), "7screws": rng.normal(3, .3, (60, 6)),
            "6screws": rng.normal(-3, .3, (60, 6))}


def _fitted(method="ledoit_wolf", openset_method="mahalanobis", seed=42):
    monitor = OpenSetMonitor(_pools(), seed=seed, method=method, openset_method=openset_method)
    monitor.fit_initial()
    monitor.add_class("7screws")
    return monitor


class SummaryTests(unittest.TestCase):
    def test_legacy_summary_reports_pooled_counts(self):
        monitor = _fitted("legacy")
        (row,) = monitor.summary()["classes"]
        self.assertEqual(row["config"], "pooled_known")
        self.assertEqual(row["label"], LEGACY_POOLED_LABEL)
        expected = sum(len(monitor.splits[c].train) for c in monitor.known)
        self.assertEqual(row["n_train"], expected)

    def test_legacy_summary_records_training_distance_threshold_source(self):
        summary = _fitted("legacy").summary()
        self.assertEqual(summary["score_type"], "normalized_pooled_mahalanobis")
        self.assertEqual(summary["threshold_strategy"], "known_train_pooled_distance_quantile_0.95")
        self.assertIn("train", summary["calibration_source"])
        self.assertNotIn("calibration split", summary["calibration_source"])

    def test_classwise_summary_keeps_one_row_per_class(self):
        monitor = _fitted()
        summary = monitor.summary()
        self.assertEqual([r["config"] for r in summary["classes"]], ["8screws", "7screws"])
        self.assertEqual(summary["classes"][1]["n_train"], len(monitor.splits["7screws"].train))
        self.assertEqual(summary["threshold_strategy"], "known_calibration_class_quantile_0.95")
        self.assertEqual(summary["calibration_source"], "known-only 20% calibration split")


class LegacyClassifyTests(unittest.TestCase):
    def test_legacy_classify_returns_known_names_for_rows_inside_threshold(self):
        monitor = _fitted("legacy")
        for config in monitor.known:
            X = monitor.holdout(config)
            scores = monitor.score(X)
            labels = monitor.classify(X)
            inside = [label for label, score in zip(labels, scores) if score <= 1.0]
            self.assertGreater(len(inside), 0)
            self.assertEqual(set(inside), {config})
            self.assertTrue(all(label is None for label, score in zip(labels, scores) if score > 1.0))


class AddClassesTests(unittest.TestCase):
    def test_add_classes_matches_sequential_add_class_bitwise(self):
        for method in ("mahalanobis", "knn"):
            with self.subTest(method=method):
                a = OpenSetMonitor(_pools(), seed=42, openset_method=method)
                a.fit_initial()
                a.add_class("7screws")
                a.add_class("6screws")
                b = OpenSetMonitor(_pools(), seed=42, openset_method=method)
                b.fit_initial()
                self.assertEqual(b.add_classes(["7screws", "6screws"]), [1, 2])
                query = np.vstack(list(_pools().values()))
                np.testing.assert_array_equal(a.score(query), b.score(query))
                self.assertEqual(a.classify(query), b.classify(query))
                self.assertEqual(a.summary(), b.summary())
                for config in a.splits:
                    np.testing.assert_array_equal(a.splits[config].train, b.splits[config].train)

    def test_empty_add_classes_keeps_model(self):
        monitor = _fitted()
        detector = monitor.detector
        self.assertEqual(monitor.add_classes([]), [])
        self.assertIs(monitor.detector, detector)


class SinglePassAndThreadTests(unittest.TestCase):
    def test_score_and_classify_equals_separate_calls(self):
        query = np.vstack(list(_pools().values()))
        for method, openset_method in (("ledoit_wolf", "mahalanobis"), ("legacy", "mahalanobis"), ("ledoit_wolf", "knn")):
            with self.subTest(method=method, openset_method=openset_method):
                monitor = _fitted(method, openset_method)
                scores, labels = monitor.score_and_classify(query)
                np.testing.assert_array_equal(scores, monitor.score(query))
                self.assertEqual(labels, monitor.classify(query))

    def test_single_thread_refit_matches_unlimited_fit_bitwise(self):
        monitor = _fitted()
        X_tr = np.vstack([monitor._pool(c)[monitor.splits[c].train] for c in monitor.known])
        y_tr = np.concatenate([np.full(len(monitor.splits[c].train), label) for c, label in monitor.known.items()])
        X_ca = np.vstack([monitor._pool(c)[monitor.splits[c].cal] for c in monitor.known])
        y_ca = np.concatenate([np.full(len(monitor.splits[c].cal), label) for c, label in monitor.known.items()])
        with threadpool_limits(limits=None):
            reference = MahalanobisOpenSetDetector(method="ledoit_wolf", confidence=0.95).fit(
                monitor.scaler.transform(X_tr), y_tr, monitor.scaler.transform(X_ca), y_ca)
        query = monitor.scaler.transform(np.vstack(list(_pools().values())))
        np.testing.assert_array_equal(monitor.detector.score_samples(query), reference.score_samples(query))
        np.testing.assert_array_equal(monitor.detector.predict_known_class(query), reference.predict_known_class(query))


class SeedTests(unittest.TestCase):
    def test_seed_reaches_mcd_detector(self):
        monitor = OpenSetMonitor(_pools(), seed=123, method="mcd")
        monitor.fit_initial()
        self.assertEqual(monitor.detector.random_state, 123)
        self.assertEqual(monitor.detector.pca_.random_state, 123)


if __name__ == "__main__":
    unittest.main()
