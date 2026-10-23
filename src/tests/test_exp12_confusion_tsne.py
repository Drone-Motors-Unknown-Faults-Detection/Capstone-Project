from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp12_confusion_tsne import (
    _binary_confusion,
    _openset_confusion,
    _tsne_one_dataset,
)


def synthetic_ancestor_pools(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = {
        "8screws": 0.0, "1screws": 2.0, "2screws": 4.0, "3screws": 6.0, "4screws": 8.0,
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
        # fault 列總和＝全部故障樣本數；healthy holdout 列至少有樣本
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
        self.assertEqual(result["labels"], ["8screws", "1screws", "2screws", "3screws", "4screws", "unknown"])
        # 1screws 真值列是該類的 holdout（30 筆的 20%）；標籤跟池 key 對得上才會非零
        self.assertEqual(matrix[1].sum(), 6)

    def test_legacy_matrix_has_lower_diagonal_concentration_than_ledoit_wolf(self):
        pools = synthetic_ancestor_pools()
        all_pools = {"T1/8000": pools}
        lw = np.array(_openset_confusion(all_pools, seed=42, confidence=0.95, method="ledoit_wolf")["matrix"])
        legacy = np.array(_openset_confusion(all_pools, seed=42, confidence=0.95, method="legacy")["matrix"])
        self.assertGreater(np.trace(lw), np.trace(legacy))


class SharedMonitorTests(unittest.TestCase):
    """#275：每個工況的 LW 祖先 monitor 只建一次；給定 monitor 與自行建立的結果相同。"""

    def test_prebuilt_monitor_gives_same_matrix_and_is_not_rebuilt(self):
        import experiments.exp12_confusion_tsne as mod

        pools = synthetic_ancestor_pools()
        all_pools = {"T1/8000": pools}
        monitor = mod.build_ancestor_monitor(pools, seed=42, confidence=0.95, method="ledoit_wolf")
        expected = _openset_confusion(all_pools, seed=42, confidence=0.95, method="ledoit_wolf")
        with patch.object(mod, "build_ancestor_monitor", side_effect=AssertionError("rebuilt")):
            shared = _openset_confusion(all_pools, seed=42, confidence=0.95, method="ledoit_wolf",
                                        monitors={"T1/8000": monitor})
            tsne = _tsne_one_dataset(pools, seed=42, confidence=0.95, perplexity=5.0, max_per_class=20,
                                     monitor=monitor)
        self.assertEqual(shared, expected)
        self.assertEqual(len(tsne["verdicts"]), len(tsne["config_labels"]))

    def test_run_builds_ledoit_wolf_monitor_once_per_dataset(self):
        import experiments.exp12_confusion_tsne as mod

        pools = synthetic_ancestor_pools()
        calls = []
        original = mod.build_ancestor_monitor

        def counting(pools_, **kwargs):
            calls.append(kwargs.get("method"))
            return original(pools_, **kwargs)

        from types import SimpleNamespace
        with patch.object(mod, "build_ancestor_monitor", side_effect=counting):
            mod._run(data_root="unused", seed=42, confidence=0.95, perplexity=5.0,
                     tsne_max_per_class=20, source_run=SimpleNamespace(load_all=lambda: {"T1/8000": pools}))
        self.assertEqual(sorted(calls), ["ledoit_wolf", "legacy"])


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
