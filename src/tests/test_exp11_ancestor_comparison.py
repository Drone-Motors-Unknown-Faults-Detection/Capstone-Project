from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp11_ancestor_comparison import (
    _ancestor_known_configs,
    _evaluate_protocol,
    build_ancestor_monitor,
    run,
)


def synthetic_ancestor_pools(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = {
        "8screws": 0.0, "1screw": 2.0, "2screws": 4.0, "3screws": 6.0, "4screws": 8.0,
        "5screws": 10.0, "6screws": 12.0, "3_14screws": 14.0,
    }
    return {name: rng.normal(center, 0.2, size=(40, 5)) for name, center in centers.items()}


class AncestorKnownConfigsTests(unittest.TestCase):
    def test_known_configs_intersect_with_pool_and_avoid_duplicate_one_screw(self):
        pools = synthetic_ancestor_pools()
        known = _ancestor_known_configs(pools)
        self.assertEqual(known, ["8screws", "1screw", "2screws", "3screws", "4screws"])

    def test_1screws_variant_also_recognized(self):
        pools = synthetic_ancestor_pools()
        pools["1screws"] = pools.pop("1screw")
        known = _ancestor_known_configs(pools)
        self.assertIn("1screws", known)
        self.assertNotIn("1screw", known)


class BuildAncestorMonitorTests(unittest.TestCase):
    def test_monitor_only_learns_ancestor_known_configs(self):
        pools = synthetic_ancestor_pools()
        monitor = build_ancestor_monitor(pools, seed=42, confidence=0.95, method="ledoit_wolf")
        self.assertEqual(set(monitor.known), {"8screws", "1screw", "2screws", "3screws", "4screws"})
        # 未知配置從未被 add_class 納入已知
        self.assertNotIn("5screws", monitor.known)
        self.assertNotIn("6screws", monitor.known)
        self.assertNotIn("3_14screws", monitor.known)

    def test_legacy_method_is_selectable(self):
        pools = synthetic_ancestor_pools()
        monitor = build_ancestor_monitor(pools, seed=42, confidence=0.95, method="legacy")
        self.assertEqual(monitor.method, "legacy")


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


if __name__ == "__main__":
    unittest.main()
