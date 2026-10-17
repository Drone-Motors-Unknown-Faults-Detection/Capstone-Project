"""效能量測工具（#48）的輸出結構與門檻判定；不檢查秒數。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from loguru import logger

from tests.profiling_evidence import MATRIX_LOADS, THRESHOLDS, decide, main, profile, write_synthetic_dataset


def _pipeline(load: float, fit: float, score: float) -> dict:
    stats = {"load": {"median_seconds": load}, "fit": {"median_seconds": fit}, "score": {"median_seconds": score}}
    total = load + fit + score
    return {"stats": stats, "share_of_run": {k: v["median_seconds"] / total for k, v in stats.items()}}


def _knn(one: float, ten: float) -> list[dict]:
    return [{"known_classes": 1, "stats": {"median_seconds": one}},
            {"known_classes": 10, "stats": {"median_seconds": ten}}]


class DecisionTests(unittest.TestCase):
    def test_cache_by_share_or_matrix_total(self):
        self.assertFalse(decide({"m": _pipeline(0.01, 0.05, 0.05)}, _knn(0.1, 0.1))["csv_cache"]["worth_it"])
        self.assertTrue(decide({"m": _pipeline(0.06, 0.03, 0.02)}, _knn(0.1, 0.1))["csv_cache"]["worth_it"])
        big = THRESHOLDS["cache_matrix_load_seconds"] / MATRIX_LOADS + 0.001
        decision = decide({"m": _pipeline(big, 10.0, 10.0)}, _knn(0.1, 0.1))["csv_cache"]
        self.assertTrue(decision["worth_it"])
        self.assertGreaterEqual(decision["estimated_matrix_load_seconds"], THRESHOLDS["cache_matrix_load_seconds"])

    def test_knn_needs_both_ratio_and_absolute_time(self):
        pipelines = {"m": _pipeline(0.01, 0.05, 0.05)}
        self.assertFalse(decide(pipelines, _knn(0.01, 0.2))["knn_vectorization"]["worth_it"])
        self.assertFalse(decide(pipelines, _knn(0.5, 0.9))["knn_vectorization"]["worth_it"])
        self.assertTrue(decide(pipelines, _knn(0.1, 0.6))["knn_vectorization"]["worth_it"])


class ProfileTests(unittest.TestCase):
    def test_profile_keeps_every_sample_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = write_synthetic_dataset(Path(directory), healthy_rows=40, fault_rows=12)
            result = profile(dataset, methods=("knn",), repeats=2, warmup=1, class_counts=(1, 3))
        pipeline = result["pipelines"]["knn"]
        self.assertEqual((len(pipeline["samples"]), len(pipeline["warmup"])), (2, 1))
        self.assertEqual(pipeline["rows_loaded"], 40 + 9 * 12)
        self.assertAlmostEqual(sum(pipeline["share_of_run"].values()), 1.0)
        self.assertEqual([row["known_classes"] for row in result["knn_score_by_classes"]], [1, 3])
        self.assertEqual(result["os_page_cache"], "not_controlled")
        self.assertIn("worth_it", result["decision"]["csv_cache"])

    def test_invalid_repeats_rejected(self):
        with self.assertRaises(ValueError):
            profile(Path("unused"), repeats=0)

    def test_main_writes_setup_run_output(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as data:
            dataset = write_synthetic_dataset(Path(data), healthy_rows=40, fault_rows=12)
            try:
                os.chdir(workspace)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(["--data-root", str(dataset), "--repeats", "1", "--warmup", "0"]), 0)
                (output,) = Path("output/profiling_evidence").iterdir()
                saved = json.loads((output / "profile.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["data_source"], "data_root")
                self.assertTrue((output / "environment.json").is_file())
                self.assertEqual(len(list(Path("logs/profiling_evidence").glob("*.log"))), 1)
            finally:
                logger.remove()
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
