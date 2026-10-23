from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import exp10_transfer
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

        # 遷移列的來源與目標工況必不相同
        for row in result["rows"]:
            self.assertNotEqual(row["src"], row["dst"])

    def test_figure_label_counts_actual_pairs(self):
        """三個工況是 6 組 src→dst，圖表不能寫死 72 組（#305 G-03）。"""
        figures = []
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            argv = ["exp10_transfer", "--data-root", str(root), "--n-values", "5,10"]
            with patch("sys.argv", argv), \
                    patch.object(exp10_transfer, "save_plot", side_effect=lambda fig, *_: figures.append(fig)):
                exp10_transfer.main()
        self.assertEqual(figures[0].axes[0].get_ylabel(), "AUROC（跨 6 組 src→dst 配對平均）")

    def test_target_support_never_appears_in_eval_metrics(self):
        draws: list[tuple[np.ndarray, np.ndarray]] = []
        healthy_lengths: list[int] = []

        def recording_draw(pool, n, rng):
            support, eval_pool = _draw_support(pool, n, rng)
            draws.append((support, eval_pool))
            return support, eval_pool

        def recording_metrics(healthy_scores, fault_scores):
            healthy_lengths.append(len(healthy_scores))
            return original_metrics(healthy_scores, fault_scores)

        original_metrics = exp10_transfer._metrics
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(exp10_transfer, "_draw_support", side_effect=recording_draw), \
                patch.object(exp10_transfer, "_metrics", side_effect=recording_metrics):
            root = _fixture(Path(directory))
            run(str(root), seed=7, n_values=(5,))

        # 3 個工況 × 2 個 target，每對抽一次 support
        self.assertEqual(len(draws), 6)
        for support, eval_pool in draws:
            self.assertEqual(len(support) + len(eval_pool), 40)
            self.assertFalse({row.tobytes() for row in support} & {row.tobytes() for row in eval_pool})
        # 每對依序算 direct、adapted、scratch；後兩者的健康分母只有 eval 的 35 列
        self.assertEqual(healthy_lengths, [40, 35, 35] * 6)


class NValueValidationTests(unittest.TestCase):
    """N 必須小於最小目標健康池，否則評估集是空的（#271 S5）；少於兩個工況直接報錯。"""

    def test_n_at_or_above_smallest_healthy_pool_is_rejected_with_pool_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            for bad in (40, 41, 0, -1):
                with self.subTest(n=bad), self.assertRaisesRegex(ValueError, "40 筆"):
                    run(str(root), seed=42, n_values=(5, bad))
            with self.assertRaises(ValueError):
                run(str(root), seed=42, n_values=())

    def test_n_below_scratch_minimum_is_rejected(self):
        """N < 5 時 scratch 的 60/20/20 切不出 train／cal，跟 add_observed 同下限（#301 F-07）。"""
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            for bad in (1, 2, 3, 4):
                with self.subTest(n=bad), self.assertRaisesRegex(ValueError, "N 必須是 5 到 39"):
                    run(str(root), seed=42, n_values=(bad,))

    def test_single_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            for motor in ("T2", "T3"):
                shutil.rmtree(root / "Step-1" / "myfeature" / motor)
            with self.assertRaisesRegex(ValueError, "至少要兩個工況"):
                run(str(root), seed=42, n_values=(5,))


if __name__ == "__main__":
    unittest.main()
