"""實驗六矩陣的完成判定規範（#43）；只用 fixture，不讀正式 data。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments import exp6_matrix
from experiments.exp6_matrix import _condition_result, _is_complete, expected_run_matrix, run_matrix, validate_run
from tests.exp6_fixture import condition_row, write_run

EXPECTED = {"method": "knn", "seed": 42, "motor": "T1", "rpm": "6000rpm", "condition": "T1/6000rpm"}


class MatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "runs" / "T1_6000rpm_seed42_knn"
        write_run(self.run_dir, condition_row("T1", "6000rpm", 42, "knn"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _summary(self) -> dict:
        return json.loads((self.run_dir / "summary.json").read_text(encoding="utf-8"))

    def _write_summary(self, summary: dict) -> None:
        (self.run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def test_matrix_requires_all_nine_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expected 9"):
                expected_run_matrix(Path(directory), seeds=(42,), methods=("mahalanobis",))

    def test_complete_fixture_passes(self) -> None:
        self.assertEqual(validate_run(self.run_dir, EXPECTED, "abc"), [])
        self.assertTrue(_is_complete(self.run_dir / "summary.json", EXPECTED, "abc"))

    def test_missing_csv_or_log_is_rejected(self) -> None:
        for name, reason in (("results.csv", "missing results.csv"), ("run.log", "missing run.log")):
            with self.subTest(name=name):
                payload = (self.run_dir / name).read_bytes()
                (self.run_dir / name).unlink()
                self.assertIn(reason, validate_run(self.run_dir, EXPECTED, "abc"))
                self.assertFalse(_is_complete(self.run_dir / "summary.json", EXPECTED, "abc"))
                (self.run_dir / name).write_bytes(payload)

    def test_log_without_completed_status_is_rejected(self) -> None:
        log = self.run_dir / "run.log"
        log.write_text(log.read_text(encoding="utf-8").replace("status=completed", "status=failed"), encoding="utf-8")
        self.assertIn("run.log lacks 'status=completed'", validate_run(self.run_dir, EXPECTED, "abc"))

    def test_csv_metric_difference_names_the_column(self) -> None:
        frame = pd.read_csv(self.run_dir / "results.csv")
        frame.loc[0, "auroc"] = 0.123
        frame.to_csv(self.run_dir / "results.csv", index=False)
        reasons = validate_run(self.run_dir, EXPECTED, "abc")
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("results.csv auroc: summary=0.9"), reasons)

    def test_csv_with_two_rows_is_rejected(self) -> None:
        frame = pd.read_csv(self.run_dir / "results.csv")
        pd.concat([frame, frame]).to_csv(self.run_dir / "results.csv", index=False)
        self.assertIn("results.csv must hold exactly one row, found 2", validate_run(self.run_dir, EXPECTED, "abc"))

    def test_wrong_identity_is_rejected(self) -> None:
        for expected, fragment in (
            (dict(EXPECTED, seed=123), "summary seed=42, expected 123"),
            (dict(EXPECTED, method="mahalanobis"), "summary method='knn'"),
            (dict(EXPECTED, motor="T2"), "summary row motor='T1'"),
            (dict(EXPECTED, condition="T1/8000rpm"), "run.log lacks 'condition=T1/8000rpm'"),
        ):
            with self.subTest(fragment=fragment):
                reasons = validate_run(self.run_dir, expected, "abc")
                self.assertTrue(any(fragment in reason for reason in reasons), reasons)

    def test_fingerprint_mismatch_and_failed_row_are_rejected(self) -> None:
        self.assertIn("summary dataset_fingerprint does not match the expected source",
                      validate_run(self.run_dir, EXPECTED, "other"))
        summary = self._summary()
        summary["rows"][0]["status"] = "failed"
        self._write_summary(summary)
        self.assertIn("summary row status='failed'", validate_run(self.run_dir, EXPECTED, "abc"))

    def test_summary_with_two_rows_is_rejected(self) -> None:
        summary = self._summary()
        summary["rows"] = summary["rows"] * 2
        self._write_summary(summary)
        self.assertIn("summary must hold exactly one row, found 2", validate_run(self.run_dir, EXPECTED, "abc"))


class RunMatrixResumeTests(unittest.TestCase):
    """run_matrix 續跑：完整格沿用並補回 manifest；缺件的格重算。"""

    def _run(self, output: Path, calls: list) -> dict:
        entry = {"run_id": "T1_6000rpm_seed42_knn", "condition": "T1/6000rpm", "motor": "T1", "rpm": "6000rpm",
                 "seed": 42, "method": "knn", "status": "pending", "log_path": "runs/T1_6000rpm_seed42_knn/run.log"}

        def fake_run(data_path, *, seed, openset_method, require_nine, pool_cache=None):
            calls.append((seed, openset_method))
            return {"status": "completed", "method": openset_method, "seed": seed, "dataset_fingerprint": "fp",
                    "commit_sha": "fixture", "rows": [condition_row("T1", "6000rpm", seed, openset_method)]}

        with patch.object(exp6_matrix, "expected_run_matrix", return_value=[dict(entry)]), \
                patch.object(exp6_matrix, "dataset_fingerprint", return_value="fp"), \
                patch.object(exp6_matrix, "run", fake_run):
            return run_matrix("data", output, seeds=(42,), methods=("knn",))

    def test_resume_fills_manifest_and_reruns_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            calls: list = []
            first = self._run(output, calls)
            self.assertEqual((first["status"], len(calls)), ("completed", 1))
            self.assertEqual(first["runs"][0]["summary"], "runs/T1_6000rpm_seed42_knn/summary.json")
            second = self._run(output, calls)
            self.assertEqual(len(calls), 1)
            entry = second["runs"][0]
            self.assertTrue(entry["resumed"])
            self.assertEqual((entry["dataset_fingerprint"], entry["summary"]),
                             ("fp", "runs/T1_6000rpm_seed42_knn/summary.json"))
            (output / "runs" / "T1_6000rpm_seed42_knn" / "results.csv").unlink()
            third = self._run(output, calls)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("resumed", third["runs"][0])
            self.assertTrue((output / "runs" / "T1_6000rpm_seed42_knn" / "results.csv").is_file())


class ConditionResultTests(unittest.TestCase):
    """_condition_result：只留目標工況那一列。"""

    ENTRY = {"motor": "T1", "rpm": "6000rpm", "condition": "T1/6000rpm"}

    def _result(self, *rows: dict) -> dict:
        return {"status": "completed", "formal_condition_count": len(rows), "rows": list(rows)}

    def test_keeps_only_the_target_row_without_mutating_input(self) -> None:
        target = condition_row("T1", "6000rpm", 42, "knn")
        other = condition_row("T3", "6000rpm", 42, "knn")
        result = self._result(other, target)
        picked = _condition_result(result, self.ENTRY)
        self.assertEqual((picked["rows"], picked["formal_condition_count"]), ([target], 1))
        self.assertEqual((result["rows"], result["formal_condition_count"]), ([other, target], 2))

    def test_missing_or_duplicate_rows_raise(self) -> None:
        target = condition_row("T1", "6000rpm", 42, "knn")
        for rows, found in (((), 0), ((target, target), 2)):
            with self.subTest(found=found):
                with self.assertRaisesRegex(ValueError, f"expected one row for T1/6000rpm, found {found}"):
                    _condition_result(self._result(*rows), self.ENTRY)


if __name__ == "__main__":
    unittest.main()
