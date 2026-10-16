import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.health_index_benchmark import run
from core.feature_schema import SCHEMAS


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
        finally:
            temporary.cleanup()

    def test_knn_smoke_root_uses_same_contract(self):
        temporary, root = self._fixture_root()
        try:
            result = run(root, seed=123, openset_method="knn", require_nine=False)
            self.assertEqual(result["method"], "knn")
            self.assertEqual(result["rows"][0]["method"], "knn")
        finally:
            temporary.cleanup()
