"""Regression tests for the formal exp6 entry point."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.openset import create_openset_detector
from core.feature_schema import SCHEMAS
from experiments.exp6_formal_benchmark import _fpr_at_tpr95, run, _condition_metrics


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
            self.assertEqual(result["dataset_root"], root.name)
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

    def test_distance_metric_names_actual_covariance_estimator(self) -> None:
        """oas／mcd 的 distance_metric 跟著實際估計器，不寫死 ledoit_wolf（#301 F-09）。"""
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            for covariance in ("ledoit_wolf", "oas", "mcd"):
                with self.subTest(covariance=covariance):
                    result = run(root, seed=42, openset_method="mahalanobis", mahalanobis_method=covariance,
                                 require_nine=False)
                    row = result["rows"][0]
                    self.assertEqual(row["mahalanobis_method"], covariance)
                    self.assertEqual(row["distance_metric"], f"{covariance}_mahalanobis")

    def test_api_environment_uses_actual_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory) / "actual_features")
            with patch("sys.argv", ["api", "--data-root", "missing_fixture_root"]):
                result = run(root, openset_method="knn", require_nine=False)
            self.assertEqual(result["environment"]["data"]["dataset_fingerprint"],
                             result["source"]["dataset_fingerprint"])
            self.assertEqual(result["environment"]["data"]["status"], "VERIFIED")

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
            self.assertEqual(result["resolved_config"]["method_id"], "knn/k5")
            self.assertEqual(result["resolved_config"]["requested_method"], "k-nn")
            self.assertEqual(result["method"], "knn")
            self.assertEqual(result["requested_method"], "k-nn")
            self.assertEqual(result["rows"][0]["requested_method"], "k-nn")

    def test_seed_reaches_detector_random_state(self) -> None:
        """MCD 的亂數跟著 --seed，不固定 42（#285 R2-12）。"""
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            with patch("experiments.exp6_formal_benchmark.create_openset_detector",
                       wraps=create_openset_detector) as factory:
                run(root, seed=7, require_nine=False)
            self.assertEqual(factory.call_args.kwargs["random_state"], 7)

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

    def test_api_run_leaves_logger_and_disk_alone(self) -> None:
        """run() 不呼叫 setup_run：呼叫端的 loguru sink 不被移除、不開 output/ 目錄（#269 C02）。"""
        from loguru import logger

        from experiments import exp6_formal_benchmark as formal

        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory) / "data")
            sink = io.StringIO()
            handle = logger.add(sink, format="{message}")
            try:
                with patch.object(formal, "capture_source", wraps=formal.capture_source) as capture:
                    result = run(root, seed=42, require_nine=False)
                logger.info("still here")
            finally:
                logger.remove(handle)
            self.assertIn("still here", sink.getvalue())
            self.assertFalse(Path("output").exists())
            self.assertFalse(Path("logs").exists())
            self.assertFalse(Path(".lineage_private").exists())
            # 執行前後各一次，指紋不再另外雜湊第三次（F3）
            self.assertEqual(capture.call_count, 2)
            self.assertIsNone(result["provenance_record_id"])
            self.assertEqual(result["environment"]["data"]["dataset_fingerprint"], result["dataset_fingerprint"])
            self.assertEqual(result["environment"]["data"]["status"], "VERIFIED")

    def test_evidence_dir_receives_provenance_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory) / "data")
            evidence = Path(directory) / "runs" / "T1_8000rpm_seed42_knn"
            result = run(root, seed=42, openset_method="knn", require_nine=False, evidence_dir=evidence)
            self.assertEqual(result["provenance_record_id"], "runs/T1_8000rpm_seed42_knn")
            record = json.loads((evidence / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "COMPLETED")
            self.assertEqual(record["source"]["dataset_fingerprint"], result["dataset_fingerprint"])
            environment = json.loads((evidence / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(environment, result["environment"])
            self.assertNotIn(directory, json.dumps(result).replace("\\\\", "\\"))
            # 只有呼叫端說剛寫過（CLI 的 setup_run）才沿用；矩陣重跑同一個 run 目錄要重新收集並覆寫（#285 R2-11）
            (evidence / "environment.json").write_text('{"marker": true}', encoding="utf-8")
            again = run(root, seed=42, openset_method="knn", require_nine=False, evidence_dir=evidence)
            self.assertEqual(again["environment"], environment)
            self.assertEqual(json.loads((evidence / "environment.json").read_text(encoding="utf-8")), environment)
            (evidence / "environment.json").write_text('{"marker": true}', encoding="utf-8")
            reused = run(root, seed=42, openset_method="knn", require_nine=False, evidence_dir=evidence,
                         reuse_environment=True)
            self.assertEqual(reused["environment"], {"marker": True})

    def test_fpr_metric_is_separate_from_normalized_detector_threshold(self) -> None:
        fpr = _fpr_at_tpr95(np.array([0.2, 0.4]), np.array([2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(fpr, 0.0)


class ConditionMetricsTests(unittest.TestCase):
    """_condition_metrics：純函數，unknown 為正類，分數 > 1 才判 unknown。"""

    def test_threshold_tie_counts_as_known(self) -> None:
        metrics = _condition_metrics(np.array([0.5, 1.0]), np.array([1.0, 2.0]))
        self.assertEqual(metrics["known_accuracy"], 1.0)
        self.assertEqual(metrics["unknown_recall"], 0.5)
        self.assertEqual(metrics["unknown_precision"], 1.0)
        self.assertEqual(metrics["open_set_accuracy"], 0.75)

    def test_perfect_separation(self) -> None:
        metrics = _condition_metrics(np.array([0.1, 0.2, 0.3]), np.array([3.0, 4.0]))
        self.assertEqual((metrics["auroc"], metrics["aupr_unknown_positive"], metrics["unknown_f1"]), (1.0, 1.0, 1.0))
        self.assertEqual(metrics["fpr_at_tpr95"], 0.0)
        self.assertEqual((metrics["known_score_median"], metrics["unknown_score_median"]), (0.2, 3.5))

    def test_all_scores_tied_gives_chance_auroc_and_zero_f1(self) -> None:
        metrics = _condition_metrics(np.full(4, 0.5), np.full(3, 0.5))
        self.assertEqual(metrics["auroc"], 0.5)
        self.assertEqual((metrics["unknown_recall"], metrics["unknown_f1"]), (0.0, 0.0))

    def test_empty_batches_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _condition_metrics(np.array([0.5]), np.array([]))
        with self.assertRaises(ValueError):
            _condition_metrics(np.array([]), np.array([2.0]))


if __name__ == "__main__":
    unittest.main()
