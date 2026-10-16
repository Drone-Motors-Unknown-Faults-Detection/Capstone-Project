"""實驗六彙總的來源一致性（#43）；fixture 加上封存矩陣的回歸。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from loguru import logger

from experiments.aggregate_exp6 import aggregate, main, write_aggregate
from tests.exp6_fixture import write_matrix

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output" / "exp6_formal_matrix"


class AggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest_path = write_matrix(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def _run_dir(self, run_id: str) -> Path:
        return self.manifest_path.parent / "runs" / run_id

    def test_aggregate_computes_macro_and_paired_difference(self) -> None:
        result = aggregate(self.manifest_path)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_runs"], 4)
        self.assertEqual(result["macro_summary"][1]["method"], "knn")
        auroc = next(row for row in result["paired_differences"] if row["metric"] == "auroc")
        self.assertAlmostEqual(auroc["mean_difference"], 0.05)

    def test_writer_emits_reproducible_artifacts(self) -> None:
        first, second = self.root / "a", self.root / "b"
        write_aggregate(self.manifest_path, first)
        write_aggregate(self.manifest_path, second)
        for name in ("aggregate.json", "condition_summary.csv", "macro_summary.csv", "paired_differences.csv", "aggregate.md"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_manifest_order_does_not_change_output(self) -> None:
        expected = aggregate(self.manifest_path)
        manifest = self._manifest()
        manifest["runs"].reverse()
        self._save(manifest)
        self.assertEqual(aggregate(self.manifest_path), expected)

    def test_missing_csv_or_log_rejects_the_matrix(self) -> None:
        for name in ("results.csv", "run.log"):
            with self.subTest(name=name):
                target = self._run_dir("T1_6000rpm_seed42_knn") / name
                payload = target.read_bytes()
                target.unlink()
                with self.assertRaisesRegex(ValueError, f"T1_6000rpm_seed42_knn: .*missing {name}"):
                    aggregate(self.manifest_path)
                target.write_bytes(payload)

    def test_duplicate_and_missing_runs_are_rejected(self) -> None:
        manifest = self._manifest()
        manifest["runs"][3] = dict(manifest["runs"][2])
        self._save(manifest)
        with self.assertRaisesRegex(ValueError, "duplicate run_id: T1_6000rpm_seed123_mahalanobis"):
            aggregate(self.manifest_path)
        manifest["runs"] = manifest["runs"][:3]
        manifest["expected_runs"] = 3
        self._save(manifest)
        with self.assertRaisesRegex(ValueError, "missing runs: T1/6000rpm seed123 knn"):
            aggregate(self.manifest_path)

    def test_wrong_seed_method_or_condition_is_rejected(self) -> None:
        for field, value, fragment in (("seed", 7, "run_id does not match"), ("method", "knn", "run_id does not match"),
                                       ("condition", "T1/8000rpm", "condition='T1/8000rpm'")):
            with self.subTest(field=field):
                write_matrix(self.root)
                manifest = self._manifest()
                manifest["runs"][0][field] = value
                self._save(manifest)
                with self.assertRaisesRegex(ValueError, fragment):
                    aggregate(self.manifest_path)

    def test_manifest_summary_fingerprint_mismatch_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["runs"][0]["dataset_fingerprint"] = "other"
        self._save(manifest)
        with self.assertRaisesRegex(ValueError, "dataset_fingerprint does not match"):
            aggregate(self.manifest_path)

    def test_fingerprint_mismatch_across_runs_is_rejected(self) -> None:
        manifest = self._manifest()
        entry = manifest["runs"][0]
        entry["dataset_fingerprint"] = "other"
        summary_path = self._run_dir(entry["run_id"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["dataset_fingerprint"] = "other"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        self._save(manifest)
        with self.assertRaisesRegex(ValueError, "dataset fingerprint mismatch across runs"):
            aggregate(self.manifest_path)
        with self.assertRaisesRegex(ValueError, "dataset fingerprint mismatch across runs"):
            aggregate(self.manifest_path, allow_partial=True)

    def test_csv_value_difference_is_traceable(self) -> None:
        csv_path = self._run_dir("T1_6000rpm_seed42_knn") / "results.csv"
        frame = pd.read_csv(csv_path)
        frame.loc[0, "unknown_f1"] = 0.5
        frame.to_csv(csv_path, index=False)
        with self.assertRaisesRegex(ValueError, r"results.csv unknown_f1: summary=0.75 csv=0.5"):
            aggregate(self.manifest_path)

    def test_windows_summary_reference_is_accepted(self) -> None:
        manifest = self._manifest()
        for entry in manifest["runs"]:
            entry["summary"] = entry["summary"].replace("/", "\\")
        self._save(manifest)
        self.assertEqual(aggregate(self.manifest_path)["status"], "completed")

    def test_allow_partial_reports_rejected_runs(self) -> None:
        (self._run_dir("T1_6000rpm_seed42_knn") / "run.log").unlink()
        result = write_aggregate(self.manifest_path, allow_partial=True)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["accepted_runs"], 3)
        self.assertEqual(result["rejected_runs"], [{"run_id": "T1_6000rpm_seed42_knn", "reasons": ["missing run.log"]}])
        auroc = next(row for row in result["paired_differences"] if row["metric"] == "auroc")
        self.assertEqual(auroc["n_pairs"], 1)
        partial_dir = self.manifest_path.parent / "aggregate_partial"
        self.assertIn("missing run.log", (partial_dir / "aggregate.md").read_text(encoding="utf-8"))
        self.assertFalse((self.manifest_path.parent / "aggregate").exists())

    def test_main_uses_setup_run(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as workspace:
            try:
                os.chdir(workspace)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(["--manifest", str(self.manifest_path), "--output-root", "agg"]), 0)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "completed")
                self.assertEqual(len(list(Path("logs/exp6_aggregate").glob("*.log"))), 1)
                self.assertTrue(Path("agg/aggregate.json").is_file())
            finally:
                logger.remove()
                os.chdir(previous)


@unittest.skipUnless((ARCHIVE / "matrix_manifest.json").is_file(), "封存矩陣不在此 checkout")
class ArchivedMatrixRegression(unittest.TestCase):
    def test_archived_54_runs_pass_and_reproduce_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = write_aggregate(ARCHIVE / "matrix_manifest.json", directory)
            self.assertEqual((result["status"], result["expected_runs"]), ("completed", 54))
            for name in ("aggregate.json", "condition_summary.csv", "macro_summary.csv", "paired_differences.csv"):
                self.assertEqual((Path(directory) / name).read_bytes(), (ARCHIVE / "aggregate" / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
