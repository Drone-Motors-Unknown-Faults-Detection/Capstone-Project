"""實驗六矩陣的完成判定規範；只用 fixture，不讀正式 data。"""

from __future__ import annotations

import json
import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments import exp6_matrix
from experiments.exp6_matrix import _condition_result, _is_complete, expected_run_matrix, run_matrix, validate_run
from tests.exp6_fixture import condition_row, producer_result, source_fixture, write_run, write_verified_matrix

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

    def test_malformed_contract_shapes_are_rejected_explicitly(self):
        from core.provenance import CONTENT_VERSION, canonical_sha

        path = write_verified_matrix(Path(self.tmp.name) / "verified")
        original = json.loads(path.read_bytes())["contract"]
        for field in ("source", "files", "execution", "git", "code", "configs", "keys", "identity", "duplicate"):
            with self.subTest(field=field):
                contract = copy.deepcopy(original)
                if field == "source":
                    contract["source"] = None
                elif field == "execution":
                    contract["execution"] = None
                elif field in {"git", "code"}:
                    contract["execution"]["git" if field == "git" else "code_sha256"] = None
                elif field == "configs":
                    contract["method_configs"] = ["knn"]
                elif field == "keys":
                    contract["run_keys"] = [None]
                else:
                    source = contract["source"]
                    if field == "files":
                        source["files"] = {"not": "a_list"}
                    elif field == "identity":
                        source["files"][0]["source_id"] = "../private_source"
                    else:
                        source["files"].append(dict(source["files"][0]))
                    source["dataset_fingerprint"] = canonical_sha(source["files"])
                    source["data_version"] = CONTENT_VERSION + ":" + source["dataset_fingerprint"]
                with self.assertRaises(ValueError):
                    exp6_matrix.validate_contract(contract)

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
    """新契約完成格驗bytes，拒絕缺件／混版覆寫；legacy仍唯讀。"""

    def _run(self, output: Path, calls: list) -> dict:
        entry = {"run_id": "T1_6000rpm_seed42_knn", "condition": "T1/6000rpm", "motor": "T1", "rpm": "6000rpm",
                 "seed": 42, "method": "knn", "status": "pending", "log_path": "runs/T1_6000rpm_seed42_knn/run.log"}

        def fake_run(data_path, *, seed, openset_method, **kwargs):
            calls.append((seed, openset_method))
            self.assertEqual(Path(kwargs["evidence_dir"]), output / "runs" / "T1_6000rpm_seed42_knn")
            return producer_result(data_path, condition_row("T1", "6000rpm", seed, openset_method))

        data = source_fixture(output.parent / "features")
        with patch.object(exp6_matrix, "expected_run_matrix", return_value=[dict(entry)]), \
                patch.object(exp6_matrix, "run", fake_run):
            return run_matrix(data, output, seeds=(42,), methods=("knn",))

    def test_legacy_resume_is_rejected_before_any_write(self) -> None:
        from tests.exp6_fixture import write_matrix
        from tests.test_exp6_benchmark import _fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _fixture(root / "features")
            manifest = write_matrix(root)
            original = {p.relative_to(manifest.parent).as_posix(): p.read_bytes()
                        for p in manifest.parent.rglob("*") if p.is_file()}
            entry = dict(EXPECTED, run_id="T1_6000rpm_seed42_knn", status="pending")
            with patch.object(exp6_matrix, "expected_run_matrix", return_value=[entry]), \
                    patch.object(exp6_matrix, "run") as benchmark:
                with self.assertRaises(ValueError):
                    run_matrix(data, manifest.parent, seeds=(42,), methods=("knn",))
            benchmark.assert_not_called()
            current = {p.relative_to(manifest.parent).as_posix(): p.read_bytes()
                       for p in manifest.parent.rglob("*") if p.is_file()}
            self.assertEqual(current, original)

    def test_nonempty_root_without_manifest_is_not_repurposed(self) -> None:
        from tests.test_exp6_benchmark import _fixture

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _fixture(root / "features")
            output = root / "archive"
            output.mkdir()
            (output / "evidence.txt").write_bytes(b"preserved failure")
            entry = dict(EXPECTED, run_id="T1_6000rpm_seed42_knn", status="pending")
            with patch.object(exp6_matrix, "expected_run_matrix", return_value=[entry]), \
                    patch.object(exp6_matrix, "run") as benchmark:
                with self.assertRaises(ValueError):
                    run_matrix(data, output, seeds=(42,), methods=("knn",))
            benchmark.assert_not_called()
            self.assertEqual([p.name for p in output.iterdir()], ["evidence.txt"])
            self.assertEqual((output / "evidence.txt").read_bytes(), b"preserved failure")

    def test_resume_fills_manifest_and_reruns_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix"
            calls: list = []
            first = self._run(output, calls)
            self.assertEqual((first["status"], len(calls)), ("completed", 1))
            self.assertEqual(first["runs"][0]["summary"], "runs/T1_6000rpm_seed42_knn/summary.json")
            second = self._run(output, calls)
            self.assertEqual(len(calls), 1)
            entry = second["runs"][0]
            self.assertTrue(entry["resumed"])
            self.assertEqual(entry["dataset_fingerprint"], first["contract"]["source"]["dataset_fingerprint"])
            self.assertEqual(entry["summary"], "runs/T1_6000rpm_seed42_knn/summary.json")
            (output / "runs" / "T1_6000rpm_seed42_knn" / "results.csv").unlink()
            before = (output / "matrix_manifest.json").read_bytes()
            with self.assertRaises(ValueError):
                self._run(output, calls)
            self.assertEqual(len(calls), 1)
            self.assertEqual((output / "matrix_manifest.json").read_bytes(), before)
            self.assertFalse((output / "runs" / "T1_6000rpm_seed42_knn" / "results.csv").exists())

    def test_mixed_contract_or_tampered_completed_run_never_changes_archive(self):
        from core.provenance import canonical_sha

        for tamper in ("schema", "data_version", "program", "config", "summary", "csv", "log"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = write_verified_matrix(root, seeds=(42,), methods=("knn",))
                manifest = json.loads(path.read_bytes())
                if tamper == "schema":
                    manifest["schema_version"] = 3
                elif tamper == "data_version":
                    manifest["data_version"] = "unverified_version"
                elif tamper == "program":
                    manifest["contract"]["execution"]["git"]["head"] = "0" * 40
                    manifest["contract_sha256"] = canonical_sha(manifest["contract"])
                elif tamper == "config":
                    manifest["contract"]["method_configs"]["knn"]["resolved_config"]["effective"]["knn_neighbors"] = 7
                    manifest["contract_sha256"] = canonical_sha(manifest["contract"])
                else:
                    name = {"summary": "summary.json", "csv": "results.csv", "log": "run.log"}[tamper]
                    target = path.parent / "runs" / manifest["runs"][0]["run_id"] / name
                    target.write_bytes(target.read_bytes() + b"\n")
                if tamper in {"schema", "data_version", "program", "config"}:
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                before = {p.relative_to(path.parent).as_posix(): p.read_bytes()
                          for p in path.parent.rglob("*") if p.is_file()}
                with self.assertRaises(ValueError):
                    self._run(path.parent, [])
                after = {p.relative_to(path.parent).as_posix(): p.read_bytes()
                         for p in path.parent.rglob("*") if p.is_file()}
                self.assertEqual(after, before)

    def test_same_stat_changed_source_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            self._run(output, [])
            csv = next((root / "features").rglob("*_clean.csv"))
            stat = csv.stat()
            content = csv.read_bytes()
            index = next(i for i in range(content.index(b"\n") + 1, len(content)) if chr(content[i]).isdigit())
            csv.write_bytes(content[:index] + (b"1" if content[index:index+1] != b"1" else b"2") + content[index+1:])
            os.utime(csv, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            before = (output / "matrix_manifest.json").read_bytes()
            # 不讓fixture helper重新寫回來源，直接檢查固定契約。
            with patch.object(exp6_matrix, "expected_run_matrix", return_value=[
                    dict(EXPECTED, run_id="T1_6000rpm_seed42_knn", status="pending")]):
                with self.assertRaises(ValueError):
                    run_matrix(root / "features", output, seeds=(42,), methods=("knn",))
            self.assertEqual((output / "matrix_manifest.json").read_bytes(), before)

    def test_source_change_between_cells_never_completes_or_rebases_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = source_fixture(root / "features")
            entries = [dict(EXPECTED, seed=seed, run_id=f"T1_6000rpm_seed{seed}_knn", status="pending")
                       for seed in (42, 123)]
            def compute(data_path, *, seed, openset_method, **kwargs):
                result = producer_result(data_path, condition_row("T1", "6000rpm", seed, openset_method))
                csv = next(data_path.rglob("*_clean.csv"))
                csv.write_bytes(csv.read_bytes() + b"\n")
                return result
            with patch.object(exp6_matrix, "expected_run_matrix", return_value=copy.deepcopy(entries)), \
                    patch.object(exp6_matrix, "run", side_effect=compute) as callback:
                result = run_matrix(data, root / "matrix", seeds=(42, 123), methods=("knn",))
            self.assertEqual(callback.call_count, 1)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["source_verification_status"], "INCOMPLETE_SOURCE_CHANGED")
            self.assertEqual(result["failed_runs"], 1)
            self.assertNotEqual(result["contract"]["source"]["dataset_fingerprint"],
                                exp6_matrix.capture_source(data)[0]["dataset_fingerprint"])

    def test_rehashed_wrong_summary_config_or_execution_is_rejected(self):
        from core.provenance import file_sha

        for field in ("config", "execution", "source"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = write_verified_matrix(root, seeds=(42,), methods=("knn",))
                manifest = json.loads(path.read_bytes())
                entry = manifest["runs"][0]
                summary_path = path.parent / entry["summary"]
                summary = json.loads(summary_path.read_bytes())
                if field == "config":
                    summary["resolved_config"]["effective"]["knn_neighbors"] = 7
                elif field == "execution":
                    summary["matrix_provenance"]["execution"]["git"]["head"] = "0" * 40
                else:
                    summary["source"]["data_version"] = "wrong"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                entry["artifact_sha256"]["summary.json"] = file_sha(summary_path)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                before = path.read_bytes(), summary_path.read_bytes()
                with self.assertRaises(ValueError):
                    self._run(path.parent, [])
                self.assertEqual((path.read_bytes(), summary_path.read_bytes()), before)

    def test_failure_retries_keep_attempt_and_redact_exception_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = source_fixture(root / "features")
            entry = dict(EXPECTED, run_id="T1_6000rpm_seed42_knn", status="pending")
            with patch.object(exp6_matrix, "expected_run_matrix", return_value=[entry]), \
                    patch.object(exp6_matrix, "run", side_effect=RuntimeError("私人路徑與token不可公開")):
                failed = run_matrix(data, root / "matrix", seeds=(42,), methods=("knn",))
            run_dir = root / "matrix/runs" / entry["run_id"]
            original_log = (run_dir / "run.log").read_bytes()
            self.assertEqual(failed["failed_runs"], 1)
            self.assertNotIn("私人路徑", json.dumps(failed, ensure_ascii=False))
            self.assertNotIn("token", original_log.decode("utf-8"))
            completed = self._run(root / "matrix", [])
            self.assertEqual(completed["status"], "completed")
            history = completed["runs"][0]["attempt_history"][0]
            self.assertEqual(history["error_type"], "RuntimeError")
            self.assertEqual((run_dir / history["attempt_id"] / "run.log").read_bytes(), original_log)

    def test_identical_source_copy_can_resume_without_public_absolute_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matrix"
            first = self._run(output, [])
            alternate = root / "copy_features"
            shutil.copytree(root / "features", alternate)
            entry = dict(EXPECTED, run_id="T1_6000rpm_seed42_knn", status="pending")
            with patch.object(exp6_matrix, "expected_run_matrix", return_value=[entry]), \
                    patch.object(exp6_matrix, "run") as callback:
                resumed = run_matrix(alternate, output, seeds=(42,), methods=("knn",))
            callback.assert_not_called()
            self.assertEqual(first["contract_sha256"], resumed["contract_sha256"])
            self.assertEqual(resumed["data_root"], alternate.name)
            self.assertNotIn(str(root), json.dumps(resumed))
            self.assertEqual(resumed["contract"]["source"]["raw_session_independence"], "UNKNOWN")


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
