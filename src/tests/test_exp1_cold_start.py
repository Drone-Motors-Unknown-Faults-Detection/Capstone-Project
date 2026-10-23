"""實驗一 run()：只有健康池時平均偵測率／AUROC 記 None，不寫 NaN（#271）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.data import HEALTHY
from core.runner import save_json
from experiments.exp1_cold_start import run


def _pools(with_fault: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    pools = {HEALTHY: rng.normal(0.0, 0.3, size=(60, 6))}
    if with_fault:
        pools["7screws"] = rng.normal(4.0, 0.3, size=(30, 6))
    return pools


class ColdStartEmptyFaultTests(unittest.TestCase):
    def test_healthy_only_pools_record_none_and_serialise(self):
        result = run(_pools(with_fault=False), seed=42)
        self.assertEqual([row["kind"] for row in result["rows"]], ["healthy-holdout"])
        self.assertIsNone(result["macro_detect_rate"])
        self.assertIsNone(result["macro_auroc"])
        with tempfile.TemporaryDirectory() as directory:
            save_json(Path(directory) / "summary.json", result)

    def test_fault_pools_keep_finite_macro_metrics(self):
        result = run(_pools(with_fault=True), seed=42)
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(np.isfinite(result["macro_detect_rate"]))
        self.assertTrue(np.isfinite(result["macro_auroc"]))


if __name__ == "__main__":
    unittest.main()
