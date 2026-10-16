from __future__ import annotations

import unittest

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
    def test_run_structure_with_monkeypatched_discovery(self):
        import experiments.exp11_ancestor_comparison as mod

        pools = synthetic_ancestor_pools()
        fake_dataset = [{"motor": "T1", "rpm": "8000rpm", "path": "unused"}]

        original_discover = mod.discover_datasets
        original_load = mod.load_pools
        mod.discover_datasets = lambda data_root: fake_dataset
        mod.load_pools = lambda path: pools
        try:
            result = run(data_root="unused", seeds=(1, 2), methods=("legacy", "ledoit_wolf"))
        finally:
            mod.discover_datasets = original_discover
            mod.load_pools = original_load

        self.assertEqual(set(result["summary"]), {"A", "B"})
        self.assertEqual(set(result["summary"]["A"]), {"legacy", "ledoit_wolf"})
        self.assertIsNotNone(result["wilcoxon"])
        self.assertIn("protocol_A_macro_f1", result["wilcoxon"])
        self.assertEqual(result["wilcoxon"]["protocol_A_macro_f1"]["n_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
