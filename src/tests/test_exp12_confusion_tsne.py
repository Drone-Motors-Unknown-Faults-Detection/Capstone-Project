from __future__ import annotations

import unittest

import numpy as np

from experiments.exp12_confusion_tsne import (
    _binary_confusion,
    _openset_confusion,
    _tsne_one_dataset,
)


def synthetic_ancestor_pools(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = {
        "8screws": 0.0, "1screw": 2.0, "2screws": 4.0, "3screws": 6.0, "4screws": 8.0,
        "5screws": 10.0, "6screws": 12.0, "3_14screws": 14.0,
    }
    return {name: rng.normal(center, 0.2, size=(30, 5)) for name, center in centers.items()}


class BinaryConfusionTests(unittest.TestCase):
    def test_shape_and_row_sums_match_sample_counts(self):
        pools = synthetic_ancestor_pools()
        all_pools = {"T1/8000": pools}
        result = _binary_confusion(all_pools, seed=42, confidence=0.95)
        matrix = np.array(result["matrix"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(result["labels"], ["healthy", "fault"])
        # healthy holdout 列總和＝healthy holdout 樣本數；fault 列總和＝全部故障樣本數
        n_fault = sum(len(v) for k, v in pools.items() if k != "8screws")
        self.assertEqual(matrix[1].sum(), n_fault)
        self.assertGreater(matrix[0].sum(), 0)


class OpensetConfusionTests(unittest.TestCase):
    def test_shape_is_six_by_six_with_canonical_one_screw_label(self):
        pools = synthetic_ancestor_pools()
        all_pools = {"T1/8000": pools}
        result = _openset_confusion(all_pools, seed=42, confidence=0.95, method="ledoit_wolf")
        matrix = np.array(result["matrix"])
        self.assertEqual(matrix.shape, (6, 6))
        self.assertEqual(result["labels"], ["8screws", "1screw", "2screws", "3screws", "4screws", "unknown"])

    def test_1screws_directory_name_merges_into_canonical_label(self):
        pools = synthetic_ancestor_pools()
        pools["1screws"] = pools.pop("1screw")
        all_pools = {"T1/8000": pools}
        result = _openset_confusion(all_pools, seed=42, confidence=0.95, method="ledoit_wolf")
        self.assertEqual(result["labels"], ["8screws", "1screw", "2screws", "3screws", "4screws", "unknown"])

    def test_legacy_matrix_has_lower_diagonal_concentration_than_ledoit_wolf(self):
        pools = synthetic_ancestor_pools()
        all_pools = {"T1/8000": pools}
        lw = np.array(_openset_confusion(all_pools, seed=42, confidence=0.95, method="ledoit_wolf")["matrix"])
        legacy = np.array(_openset_confusion(all_pools, seed=42, confidence=0.95, method="legacy")["matrix"])
        self.assertGreaterEqual(np.trace(lw), np.trace(legacy))


class TsneOneDatasetTests(unittest.TestCase):
    def test_embedding_shape_and_no_nan(self):
        pools = synthetic_ancestor_pools()
        result = _tsne_one_dataset(pools, seed=42, confidence=0.95, perplexity=5, max_per_class=10)
        n_expected = sum(min(len(v), 10) for v in pools.values())
        self.assertEqual(result["embedding"].shape, (n_expected, 2))
        self.assertFalse(np.isnan(result["embedding"]).any())
        self.assertEqual(len(result["config_labels"]), n_expected)
        self.assertEqual(len(result["verdicts"]), n_expected)
        self.assertTrue(set(result["verdicts"]).issubset({"known_correct", "known_wrong", "unknown"}))


if __name__ == "__main__":
    unittest.main()
