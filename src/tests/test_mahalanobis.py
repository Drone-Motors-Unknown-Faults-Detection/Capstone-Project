"""MahalanobisOpenSetDetector：legacy 合併分布的標籤與最近類別判讀（#266）、seed 轉發（#273）、
距離計算改寫後數字逐位元相同（#275）。"""
import unittest

import numpy as np

from core.mahalanobis import (
    LEGACY_POOLED_LABEL,
    UNKNOWN_LABEL,
    MahalanobisOpenSetDetector,
    _distances,
)
from core.openset import create_detector_from_config, create_openset_detector, resolve_openset_config


def _data(seed: int = 3, n: int = 40, dim: int = 6):
    rng = np.random.default_rng(seed)
    centers = [0.0, 4.0, 8.0]
    X_train = np.vstack([rng.normal(c, 0.3, (n, dim)) for c in centers])
    y_train = np.repeat(np.arange(len(centers)), n)
    X_cal = np.vstack([rng.normal(c, 0.3, (n // 3, dim)) for c in centers])
    y_cal = np.repeat(np.arange(len(centers)), n // 3)
    return X_train, y_train, X_cal, y_cal


class DistancesTests(unittest.TestCase):
    def test_matches_three_operand_einsum_bitwise(self):
        """單筆（exp2 逐筆）到整池的各種列數，都要與改前的三運算元 einsum 逐位元相同。"""
        rng = np.random.default_rng(1)
        for dim in (12, 40, 105):
            location = rng.normal(size=dim)
            A = rng.normal(size=(dim, dim))
            precision = A @ A.T + np.eye(dim)
            for n in (1, 2, 7, 50, 300):
                with self.subTest(dim=dim, n=n):
                    X = rng.normal(size=(n, dim))
                    delta = X - location
                    reference = np.sqrt(np.maximum(
                        np.einsum("ij,jk,ik->i", delta, precision, delta, optimize=True), 0.0))
                    np.testing.assert_array_equal(_distances(X, location, precision), reference)


class LegacyLabelTests(unittest.TestCase):
    def test_pooled_label_is_distinct_from_unknown_sentinel(self):
        detector = MahalanobisOpenSetDetector(method="legacy").fit(*_data())
        self.assertNotEqual(LEGACY_POOLED_LABEL, UNKNOWN_LABEL)
        self.assertEqual([d.label for d in detector.distributions_], [LEGACY_POOLED_LABEL])
        self.assertEqual(detector.class_summaries()[0]["label"], LEGACY_POOLED_LABEL)

    def test_known_rows_get_nearest_class_centroid_and_far_rows_stay_unknown(self):
        X_train, y_train, X_cal, y_cal = _data()
        detector = MahalanobisOpenSetDetector(method="legacy").fit(X_train, y_train, X_cal, y_cal)
        labels = detector.predict_known_class(X_train)
        scores = detector.score_samples(X_train)
        inside = scores <= 1.0
        self.assertGreater(inside.sum(), 0)
        np.testing.assert_array_equal(labels[inside], y_train[inside])
        self.assertTrue(np.all(labels[~inside] == UNKNOWN_LABEL))
        far = np.full((3, X_train.shape[1]), 60.0)
        self.assertTrue(np.all(detector.predict_known_class(far) == UNKNOWN_LABEL))

    def test_legacy_scores_unchanged_by_labelling(self):
        X_train, y_train, X_cal, y_cal = _data()
        detector = MahalanobisOpenSetDetector(method="legacy").fit(X_train, y_train, X_cal, y_cal)
        pooled = detector.distributions_[0]
        reference = _distances(X_cal, pooled.location, pooled.precision) / pooled.threshold
        np.testing.assert_array_equal(detector.score_samples(X_cal), reference)


class ScoreAndPredictTests(unittest.TestCase):
    def test_single_pass_equals_separate_calls(self):
        X_train, y_train, X_cal, y_cal = _data()
        query = np.vstack([X_cal, np.full((2, X_cal.shape[1]), 30.0)])
        for method in ("ledoit_wolf", "legacy", "oas"):
            with self.subTest(method=method):
                detector = MahalanobisOpenSetDetector(method=method).fit(X_train, y_train, X_cal, y_cal)
                scores, labels = detector.score_and_predict(query)
                np.testing.assert_array_equal(scores, detector.score_samples(query))
                np.testing.assert_array_equal(labels, detector.predict_known_class(query))


class SeedForwardingTests(unittest.TestCase):
    def test_factory_forwards_random_state_to_mcd(self):
        resolved = resolve_openset_config("mahalanobis", mahalanobis_method="mcd")
        self.assertEqual(create_detector_from_config(resolved, random_state=2026).random_state, 2026)
        self.assertEqual(create_openset_detector("mahalanobis", mahalanobis_method="mcd", random_state=7).random_state, 7)
        self.assertEqual(create_detector_from_config(resolved).random_state, 42)

    def test_mcd_with_same_seed_reproduces_and_pca_takes_seed(self):
        X_train, y_train, X_cal, y_cal = _data(n=60, dim=5)
        a = MahalanobisOpenSetDetector(method="mcd", random_state=5).fit(X_train, y_train, X_cal, y_cal)
        b = MahalanobisOpenSetDetector(method="mcd", random_state=5).fit(X_train, y_train, X_cal, y_cal)
        np.testing.assert_array_equal(a.score_samples(X_cal), b.score_samples(X_cal))
        self.assertEqual(a.pca_.random_state, 5)


if __name__ == "__main__":
    unittest.main()
