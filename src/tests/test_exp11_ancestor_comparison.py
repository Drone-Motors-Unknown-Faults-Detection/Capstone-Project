from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp11_ancestor_comparison import (
    _ancestor_known_configs,
    _evaluate_protocol,
    _paired_wilcoxon,
    build_ancestor_monitor,
    run,
)


def synthetic_ancestor_pools(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = {
        "8screws": 0.0, "1screws": 2.0, "2screws": 4.0, "3screws": 6.0, "4screws": 8.0,
        "5screws": 10.0, "6screws": 12.0, "3_14screws": 14.0,
    }
    return {name: rng.normal(center, 0.2, size=(40, 5)) for name, center in centers.items()}


class AncestorKnownConfigsTests(unittest.TestCase):
    def test_known_configs_intersect_with_pool_in_protocol_order(self):
        pools = synthetic_ancestor_pools()
        known = _ancestor_known_configs(pools)
        self.assertEqual(known, ["8screws", "1screws", "2screws", "3screws", "4screws"])
        pools.pop("2screws")
        self.assertEqual(_ancestor_known_configs(pools), ["8screws", "1screws", "3screws", "4screws"])

    def test_loaded_pools_use_canonical_key_so_1screw_directory_is_known(self):
        """core.data 把 1screw 目錄讀成 1screws，本實驗只認正規名（#271）。"""
        import tempfile
        from pathlib import Path

        import pandas as pd

        from core.data import load_pools
        from core.feature_schema import SCHEMAS

        with tempfile.TemporaryDirectory() as directory:
            rng = np.random.default_rng(1)
            for name in ("8screws", "1screw"):
                target = Path(directory) / name
                target.mkdir()
                pd.DataFrame(rng.normal(size=(12, 105)), columns=SCHEMAS["historical_clean_v1"]).to_csv(
                    target / "T1_Group_feature_data_clean.csv", index=False)
            pools = load_pools(directory)
        self.assertEqual(_ancestor_known_configs(pools), ["8screws", "1screws"])


class BuildAncestorMonitorTests(unittest.TestCase):
    def test_monitor_only_learns_ancestor_known_configs(self):
        pools = synthetic_ancestor_pools()
        monitor = build_ancestor_monitor(pools, seed=42, confidence=0.95, method="ledoit_wolf")
        self.assertEqual(set(monitor.known), {"8screws", "1screws", "2screws", "3screws", "4screws"})
        # 未知配置從未被 add_class 納入已知
        self.assertNotIn("5screws", monitor.known)
        self.assertNotIn("6screws", monitor.known)
        self.assertNotIn("3_14screws", monitor.known)

    def test_legacy_method_is_selectable(self):
        pools = synthetic_ancestor_pools()
        monitor = build_ancestor_monitor(pools, seed=42, confidence=0.95, method="legacy")
        self.assertEqual(monitor.method, "legacy")

    def test_build_matches_sequential_add_class(self):
        pools = synthetic_ancestor_pools()
        from core.monitor import OpenSetMonitor

        reference = OpenSetMonitor(pools, seed=42, confidence=0.95)
        reference.fit_initial()
        for config in ["1screws", "2screws", "3screws", "4screws"]:
            reference.add_class(config)
        monitor = build_ancestor_monitor(pools, seed=42, confidence=0.95)
        query = pools["5screws"]
        np.testing.assert_array_equal(monitor.score(query), reference.score(query))
        self.assertEqual(monitor.summary(), reference.summary())

    def test_legacy_classify_returns_known_names(self):
        """#266：legacy 對 score≤1 的已知 holdout 回最近類別，不再全判 unknown。"""
        pools = synthetic_ancestor_pools()
        monitor = build_ancestor_monitor(pools, seed=42, confidence=0.95, method="legacy")
        metrics = _evaluate_protocol(monitor, ["5screws", "6screws", "3_14screws"], pools)
        self.assertGreater(metrics["balanced_accuracy"], 1 / 6)
        for config in monitor.known:
            labels = monitor.classify(monitor.holdout(config))
            self.assertIn(config, labels)


class EvaluateProtocolTests(unittest.TestCase):
    def test_metrics_keys_present_and_in_range(self):
        pools = synthetic_ancestor_pools()
        monitor = build_ancestor_monitor(pools, seed=42, confidence=0.95, method="ledoit_wolf")
        unknown = ["5screws", "6screws", "3_14screws"]
        metrics = _evaluate_protocol(monitor, unknown, pools)
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "unknown_f1", "n"]:
            self.assertIn(key, metrics)
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "unknown_f1"]:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)


class RunIntegrationTests(unittest.TestCase):
    def test_run_structure_with_actual_fixture_source(self):
        import pandas as pd
        from core.feature_schema import SCHEMAS

        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config, values in synthetic_ancestor_pools().items():
                target = root / "Step-1/myfeature/T1/8000rpm" / config
                target.mkdir(parents=True)
                values = np.hstack([values, rng.normal(0, .2, (len(values), 100))])
                pd.DataFrame(values, columns=SCHEMAS["historical_clean_v1"]).to_csv(
                    target / "T1_Group_feature_data_clean.csv", index=False)
            result = run(data_root=root, seeds=(1, 2), methods=("legacy", "ledoit_wolf"))
        self.assertEqual(result["provenance"]["source"]["status"], "VERIFIED")

        self.assertEqual(set(result["summary"]), {"A", "B"})
        self.assertEqual(set(result["summary"]["A"]), {"legacy", "ledoit_wolf"})
        self.assertIsNotNone(result["wilcoxon"])
        self.assertIn("protocol_A_macro_f1", result["wilcoxon"])
        self.assertEqual(result["wilcoxon"]["protocol_A_macro_f1"]["n_pairs"], 2)


class EmptyInputTests(unittest.TestCase):
    """沒有資料或沒有配對時報錯／回 None，不算 NaN、不捏造 p=1（#301 F-08）。"""

    def test_empty_data_root_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):  # 來源綁定（#213）或 _run 都會先擋下
                run(data_root=directory, seeds=(1,))

    def test_no_pairs_reports_none(self):
        rows = [{"protocol": protocol, "dataset": "T1/8000", "seed": 1, "method": "legacy", "macro_f1": 0.5}
                for protocol in ("A", "B")]
        result = _paired_wilcoxon(rows, ("legacy", "ledoit_wolf"))
        for protocol in ("A", "B"):
            self.assertEqual(result[f"protocol_{protocol}_macro_f1"],
                             {"n_pairs": 0, "mean_diff_pp": None, "statistic": None, "p_value": None})


if __name__ == "__main__":
    unittest.main()
