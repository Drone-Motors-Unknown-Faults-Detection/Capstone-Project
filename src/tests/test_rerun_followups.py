"""Aero 重跑後發現的輸出問題（#252、#253、#256）；只用 fixture，不讀正式 data。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.feature_schema import SCHEMAS
from experiments import aggregate_exp6, exp2_scale_growth, exp6_matrix, navigation_data_contract


def _dataset(root: Path) -> dict:
    base = root / "Step-1/myfeature/T1/8000rpm"
    rng = np.random.default_rng(1)
    for config, center in (("8screws", 0.0), ("7screws", 3.0)):
        (base / config).mkdir(parents=True)
        pd.DataFrame(rng.normal(center, 0.1, (20, 105)), columns=SCHEMAS["historical_clean_v1"]).to_csv(
            base / config / "T1_Group_feature_data_clean.csv", index=False)
    return {"motor": "T1", "rpm": "8000rpm", "path": base}


class NavigationSourcePathTests(unittest.TestCase):
    def test_source_paths_are_relative_to_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            result = navigation_data_contract.run(_dataset(Path(directory)))
        paths = [s["path"] for sources in result["sources"].values() for s in sources]
        self.assertEqual(paths[0].split("/")[0], "Step-1")
        self.assertTrue(all(not Path(p).is_absolute() and directory not in p for p in paths))


class MatrixDefaultTests(unittest.TestCase):
    def test_default_output_root_is_new_timestamp_dir(self):
        captured = {}

        def fake(data_root, output_root, **kwargs):
            captured["output_root"] = output_root
            return {"status": "completed", "completed_runs": 0, "expected_runs": 0, "failed_runs": 0, "missing_runs": 0}

        with tempfile.TemporaryDirectory() as directory:
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                with patch.object(exp6_matrix, "run_matrix", side_effect=fake):
                    exp6_matrix.main([])
            finally:
                os.chdir(cwd)
        self.assertRegex(captured["output_root"], r"^output/exp6_formal_matrix/\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

    def test_aggregate_requires_manifest(self):
        with self.assertRaises(SystemExit):
            aggregate_exp6.main([])


class StagesCsvIntegerTests(unittest.TestCase):
    def test_missing_stage_keeps_integer_columns(self):
        stages = [{"config": "7screws", "discovered": True, "matched": "7screws", "matched_is_config": True,
                   "samples_to_candidate": 215, "cluster_size": 31,
                   "cluster_purity": 1.0, "own_holdout_acc": 1.0, "healthy_holdout_acc": 1.0, "known_after": 2},
                  {"config": "6screws", "discovered": False, "matched_is_config": False}]
        result = {"stages": stages, "n_stages": 2, "n_learned": 1, "n_matched_config": 1,
                  "final_known": ["8screws"], "model": {}}
        with tempfile.TemporaryDirectory() as directory:
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                with patch.object(exp2_scale_growth, "resolve_dataset", return_value=({"motor": "T1", "rpm": "8000rpm"}, {})), \
                        patch.object(exp2_scale_growth, "run", return_value=result), \
                        patch("sys.argv", ["exp2"]):
                    exp2_scale_growth.main()
                (run_dir,) = Path("output/exp2_scale_growth").iterdir()
                text = (run_dir / "stages.csv").read_text(encoding="utf-8")
                json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            finally:
                os.chdir(cwd)
        self.assertIn(",215,", text)
        self.assertNotIn("215.0", text)


if __name__ == "__main__":
    unittest.main()
