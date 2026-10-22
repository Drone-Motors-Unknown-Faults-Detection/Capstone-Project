import contextlib
import io
import json
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from loguru import logger

from experiments import health_index_benchmark
from experiments.health.severity import RelativeSeverityPolicy
from experiments.health_index_benchmark import _health_distribution, main, run
from core.feature_schema import SCHEMAS


class HealthDistributionTests(unittest.TestCase):
    def test_fractions_follow_severity_policy_thresholds(self):
        values = np.array([0.05, 0.25, 0.65, 0.85, 0.95])
        default = _health_distribution(values)
        self.assertEqual(default["healthy_fraction"], 0.6)
        self.assertEqual(default["critical_fraction"], 0.2)
        policy = RelativeSeverityPolicy(healthy_min=0.9, early_warning_min=0.5, degraded_min=0.3)
        with patch.object(health_index_benchmark, "DEFAULT_SEVERITY_POLICY", policy):
            moved = _health_distribution(values)
        self.assertEqual(moved["healthy_fraction"], 0.2)
        self.assertEqual(moved["critical_fraction"], 0.4)


class HealthBenchmarkTests(unittest.TestCase):
    def _fixture_root(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        base = root / "Step-1" / "myfeature" / "T1" / "8000rpm"
        rng = np.random.default_rng(7)
        columns = SCHEMAS["historical_clean_v1"]
        for config, offset in (("8screws", 0.0), ("1screws", 2.0)):
            directory = base / config
            directory.mkdir(parents=True)
            frame = pd.DataFrame(rng.normal(offset, 0.2, size=(40, 105)), columns=columns)
            frame.to_csv(directory / "sample_Group_feature_data_clean.csv", index=False)
        return temporary, root

    def test_smoke_root_runs_without_unknown_calibration_leakage(self):
        temporary, root = self._fixture_root()
        try:
            result = run(root, seed=42, openset_method="mahalanobis", require_nine=False)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["formal_condition_count"], 1)
            self.assertFalse(result["unknown_calibration_leakage"])
            row = result["rows"][0]
            self.assertIn("health_gap_known_minus_unknown", row)
            self.assertFalse(row["rul_available"])
            self.assertLess(row["calibration_healthy_anchor"], row["calibration_warning_anchor"])
            self.assertAlmostEqual(row["calibration_critical_anchor"], 2.0 * row["calibration_warning_anchor"])
            # cal q95 is the rejection line, so the known holdout mostly stays healthy
            self.assertGreaterEqual(row["known_healthy_fraction"], 0.85)
            self.assertEqual(row["unknown_health_mean"], 0.0)
        finally:
            temporary.cleanup()

    def test_knn_smoke_root_uses_same_contract(self):
        temporary, root = self._fixture_root()
        try:
            result = run(root, seed=123, openset_method="knn", require_nine=False)
            self.assertEqual(result["method"], "knn")
            self.assertEqual(result["rows"][0]["method"], "knn")
            self.assertEqual(result["resolved_config"]["method_id"], "knn/k5")
        finally:
            temporary.cleanup()

    def test_cli_uses_setup_run_and_saves_same_result_as_api(self):
        """CLI 經 setup_run 寫 logs/output，同 seed 的 result.json 與 API 回傳相同。"""
        temporary, root = self._fixture_root()
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as workspace:
                try:
                    os.chdir(workspace)
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        self.assertEqual(main(["--data-root", str(root), "--seed", "42", "--allow-partial"]), 0)
                    self.assertEqual(len(list(Path("logs/exp8_health_index_benchmark").glob("*.log"))), 1)
                    (output_dir,) = Path("output/exp8_health_index_benchmark").iterdir()
                    saved = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
                    table = pd.read_csv(output_dir / "rows.csv")
                    summary = json.loads(stdout.getvalue())
                    self.assertEqual(summary["output_dir"], str(output_dir))
                    self.assertFalse(Path("reports").exists())
                finally:
                    logger.remove()
                    os.chdir(previous)
            expected = run(root, seed=42, require_nine=False)
            self.assertEqual(saved, json.loads(json.dumps(expected)))
            self.assertEqual(len(table), len(expected["rows"]))
        finally:
            temporary.cleanup()
