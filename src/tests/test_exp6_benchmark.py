"""Regression tests for the formal exp6 entry point."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.openset import create_openset_detector
from core.feature_schema import SCHEMAS
from experiments.exp6_formal_benchmark import _fpr_at_tpr95, run


def _fixture(root: Path) -> Path:
    directory = root / "Step-1" / "myfeature" / "T1" / "8000rpm"
    rng = np.random.default_rng(7)
    columns = SCHEMAS["historical_clean_v1"]
    for config, center in (("8screws", 0.0), ("7screws", 5.0)):
        target = directory / config
        target.mkdir(parents=True)
        frame = pd.DataFrame(rng.normal(center, 0.1, size=(30, 105)), columns=columns)
        frame.to_csv(target / "T1_Group_feature_data_clean.csv", index=False)
    return root


class Exp6BenchmarkTests(unittest.TestCase):
    def test_missing_formal_data_fails_instead_of_nan_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run(Path(directory), require_nine=True)

    def test_fixture_run_records_shared_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            result = run(root, seed=42, openset_method="knn", require_nine=False)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["method"], "knn")
            self.assertEqual(result["dataset_root"], "formal_features")
            self.assertFalse(Path(result["dataset_root"]).is_absolute())
            self.assertEqual(result["checkpoint_identifier"], None)
            self.assertEqual(result["config"]["normalization"], "RobustScaler fit on known training split only")
            self.assertEqual(result["rows"][0]["distance_metric"], "euclidean")
            self.assertTrue(result["rows"][0]["reference_bank_samples"])
            self.assertEqual(result["polarmap_base_method"], "mahalanobis")
            row = result["rows"][0]
            self.assertEqual(row["calibration_source"], "known training/calibration only")
            self.assertEqual(row["score_direction"], "higher_is_unknown")
            self.assertTrue(np.isfinite(row["auroc"]))

    def test_exp6_calls_shared_factory_and_canonicalizes_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            with patch(
                "experiments.exp6_formal_benchmark.create_openset_detector",
                wraps=create_openset_detector,
            ) as factory:
                result = run(root, seed=42, openset_method="k-nn", require_nine=False)
            self.assertTrue(factory.called)
            self.assertEqual(factory.call_args.args[0], "knn")
            self.assertEqual(result["method"], "knn")
            self.assertEqual(result["requested_method"], "k-nn")
            self.assertEqual(result["rows"][0]["requested_method"], "k-nn")

    def test_formal_benchmark_rejects_legacy_training_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "training-distance threshold"):
                run(
                    root,
                    seed=42,
                    openset_method="mahalanobis",
                    mahalanobis_method="legacy",
                    require_nine=False,
                )

    def test_fpr_metric_is_separate_from_normalized_detector_threshold(self) -> None:
        fpr = _fpr_at_tpr95(np.array([0.2, 0.4]), np.array([2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(fpr, 0.0)


if __name__ == "__main__":
    unittest.main()
