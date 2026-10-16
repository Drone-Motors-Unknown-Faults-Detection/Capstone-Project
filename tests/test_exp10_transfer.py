from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp10_transfer import _draw_support, run
from core.feature_schema import SCHEMAS


def _fixture(root: Path) -> Path:
    rng = np.random.default_rng(3)
    columns = SCHEMAS["adapter_generated_v1"]
    datasets = {"T1": 0.0, "T2": 2.0, "T3": -2.0}
    for motor, center in datasets.items():
        directory = root / "Step-1" / "myfeature" / motor / "8000rpm"
        for config, offset in (("8screws", 0.0), ("7screws", 8.0)):
            target = directory / config
            target.mkdir(parents=True)
            n = 40 if config == "8screws" else 20
            frame = pd.DataFrame(
                rng.normal(center + offset, 0.2, size=(n, 105)), columns=columns
            )
            frame.to_csv(target / f"{motor}_Group_feature_data_clean.csv", index=False)
    return root


class DrawSupportTests(unittest.TestCase):
    def test_support_and_eval_indices_do_not_overlap(self):
        pool = np.arange(30.0).reshape(30, 1)
        rng = np.random.default_rng(1)
        support, eval_pool = _draw_support(pool, 8, rng)
        self.assertEqual(len(support), 8)
        self.assertEqual(len(eval_pool), 22)
        self.assertEqual(len(set(support.flatten()) & set(eval_pool.flatten())), 0)


class Exp10TransferTests(unittest.TestCase):
    def test_run_returns_three_strategies_with_valid_auroc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            result = run(str(root), seed=42, n_values=(5, 10))

        self.assertEqual(set(result["summary"]), {"direct", "adapted", "scratch"})
        self.assertEqual([r["n"] for r in result["summary"]["direct"]], [0])
        self.assertEqual([r["n"] for r in result["summary"]["adapted"]], [5, 10])
        self.assertEqual([r["n"] for r in result["summary"]["scratch"]], [5, 10])

        for rows in result["summary"].values():
            for row in rows:
                self.assertGreaterEqual(row["auroc_mean"], 0.0)
                self.assertLessEqual(row["auroc_mean"], 1.0)

        # 每組 src/dst/n/strategy 都要有對應列，且 src != dst
        for row in result["rows"]:
            self.assertNotEqual(row["src"], row["dst"])

    def test_target_support_never_appears_in_eval_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            result = run(str(root), seed=7, n_values=(5,))

        adapted_rows = [r for r in result["rows"] if r["strategy"] == "adapted" and r["n"] == 5]
        scratch_rows = [r for r in result["rows"] if r["strategy"] == "scratch" and r["n"] == 5]
        self.assertTrue(adapted_rows)
        self.assertTrue(scratch_rows)


if __name__ == "__main__":
    unittest.main()
