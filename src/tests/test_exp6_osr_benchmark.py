"""實驗六 OSR 基準（exp6_osr_benchmark）：七種偵測器 × 每個工況的列、彙總排序、決定論與 CLI 輸出（#276 K04）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.detectors import ALL_DETECTORS
from core.feature_schema import SCHEMAS
from experiments import exp6_osr_benchmark

ROW_KEYS = ("dataset", "method", "auroc", "fpr_at_tpr95", "healthy_fp", "fault_detect")


def _fixture(root: Path) -> Path:
    """兩個工況，各 60 筆健康、30 筆故障；LOF 的 n_neighbors=20 需要 train 超過 20 筆。"""
    rng = np.random.default_rng(11)
    columns = SCHEMAS["historical_clean_v1"]
    for motor, center in (("T1", 0.0), ("T2", 3.0)):
        directory = root / "Step-1" / "myfeature" / motor / "8000rpm"
        for config, offset, n in (("8screws", 0.0, 60), ("7screws", 6.0, 30)):
            target = directory / config
            target.mkdir(parents=True)
            frame = pd.DataFrame(rng.normal(center + offset, 0.2, size=(n, 105)), columns=columns)
            frame.to_csv(target / f"{motor}_Group_feature_data_clean.csv", index=False)
    return root


class Exp6OsrBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = _fixture(Path(cls.temp.name))
        cls.result = exp6_osr_benchmark.run(str(cls.root), seed=42, confidence=0.95)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_one_row_per_dataset_and_detector(self):
        rows = self.result["rows"]
        self.assertEqual(self.result["n_datasets"], 2)
        self.assertEqual(len(rows), 2 * len(ALL_DETECTORS))
        self.assertEqual({row["dataset"] for row in rows}, {"T1/8000", "T2/8000"})
        self.assertEqual({row["method"] for row in rows}, {det.name for det in ALL_DETECTORS})
        for row in rows:
            self.assertEqual(tuple(row), ROW_KEYS)
            for key in ROW_KEYS[2:]:
                self.assertTrue(0.0 <= row[key] <= 1.0, (row["method"], key, row[key]))
                self.assertEqual(row[key], round(row[key], 4))

    def test_separable_fixture_is_detected(self):
        # 故障池離健康池 6 個單位（std 0.2），每種方法都應該幾乎全數偵測到
        for row in self.result["rows"]:
            self.assertGreaterEqual(row["auroc"], 0.99, row)
            self.assertGreaterEqual(row["fault_detect"], 0.9, row)

    def test_summary_is_mean_std_over_datasets_sorted_by_auroc(self):
        summary = self.result["summary"]
        self.assertEqual([item["method"] for item in summary].count("maha_ledoit_wolf"), 1)
        self.assertEqual(len(summary), len(ALL_DETECTORS))
        for item in summary:
            subset = [row for row in self.result["rows"] if row["method"] == item["method"]]
            self.assertEqual(len(subset), 2)
            for metric in ("auroc", "fpr_at_tpr95", "healthy_fp", "fault_detect"):
                self.assertEqual(item[f"{metric}_mean"], round(float(np.mean([r[metric] for r in subset])), 4))
            for metric in ("auroc", "healthy_fp"):
                self.assertEqual(item[f"{metric}_std"], round(float(np.std([r[metric] for r in subset])), 4))
        order = [(-item["auroc_mean"], item["fpr_at_tpr95_mean"]) for item in summary]
        self.assertEqual(order, sorted(order))

    def test_same_seed_reproduces_every_row(self):
        again = exp6_osr_benchmark.run(str(self.root), seed=42, confidence=0.95)
        self.assertEqual(again["rows"], self.result["rows"])
        self.assertEqual(again["summary"], self.result["summary"])

    def test_metrics_come_from_shared_openset_helper(self):
        """auroc／fpr_at_tpr95 走 core.openset.openset_metrics，與其他 runner 同一定義。"""
        with patch.object(exp6_osr_benchmark, "openset_metrics",
                          wraps=exp6_osr_benchmark.openset_metrics) as shared:
            exp6_osr_benchmark.run(str(self.root), seed=42, confidence=0.95)
        self.assertEqual(shared.call_count, 2 * len(ALL_DETECTORS))

    def test_main_writes_results_summary_and_figure(self):
        # conftest 已把 cwd 換到暫存目錄，logs/ 與 output/ 不會寫進 repo
        with patch("sys.argv", ["exp6_osr_benchmark", "--data-root", str(self.root), "--seed", "42"]):
            exp6_osr_benchmark.main()
        runs = sorted(Path("output/exp6_osr_benchmark").iterdir())
        self.assertEqual(len(runs), 1)
        output = runs[0]
        self.assertTrue((output / "osr_benchmark.png").is_file())
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["rows"], self.result["rows"])
        table = pd.read_csv(output / "results.csv")
        self.assertEqual(list(table.columns), list(ROW_KEYS))
        self.assertEqual(len(table), len(self.result["rows"]))


class OsrBenchmarkInputTests(unittest.TestCase):
    def test_empty_data_root_is_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                exp6_osr_benchmark.run(Path(directory), seed=1)


if __name__ == "__main__":
    unittest.main()
