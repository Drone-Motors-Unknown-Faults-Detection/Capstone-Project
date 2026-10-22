"""實驗六彙總的來源一致性；fixture 加上封存矩陣的回歸。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from loguru import logger

from experiments.aggregate_exp6 import aggregate, main, write_aggregate
from experiments import aggregate_exp6 as module
from tests.exp6_fixture import write_matrix, write_verified_matrix

ROOT = Path(__file__).resolve().parents[2]
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

    def test_actual_parent_bytes_and_legacy_status_are_recorded(self):
        from core.provenance import file_sha

        result = aggregate(self.manifest_path)
        proof = result["provenance"]
        self.assertEqual(proof["verification_status"], "LEGACY_UNVERIFIED")
        self.assertEqual(proof["manifest_sha256"], file_sha(self.manifest_path))
        self.assertEqual(len(proof["parent_runs"]), 4)
        for parent in proof["parent_runs"]:
            for name, checksum in parent["artifact_sha256"].items():
                self.assertEqual(checksum, file_sha(self._run_dir(parent["run_id"]) / name))

    def test_legacy_absolute_root_is_not_copied_to_public_summary(self):
        manifest = self._manifest()
        manifest["data_root"] = str(self.root / "私人來源")
        self._save(manifest)
        result = aggregate(self.manifest_path)
        self.assertEqual(result["dataset_root"], "LEGACY_PRIVATE_ROOT")
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_existing_aggregate_destination_is_not_overwritten(self):
        destination = self.root / "sealed"
        write_aggregate(self.manifest_path, destination)
        original = {p.name: p.read_bytes() for p in destination.iterdir()}
        with self.assertRaises(ValueError):
            write_aggregate(self.manifest_path, destination)
        self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()}, original)

    def test_manifest_change_during_read_is_rejected(self):
        validator = module.validate_run
        changed = []
        def validate(*args, **kwargs):
            result = validator(*args, **kwargs)
            if not changed:
                changed.append(True)
                self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b"\n")
            return result
        with patch.object(module, "validate_run", side_effect=validate), self.assertRaises(ValueError):
            aggregate(self.manifest_path)

    def test_new_contract_mixed_program_or_config_is_fatal_even_in_partial(self):
        from core.provenance import file_sha

        for partial in (False, True):
            for field in ("program", "config", "data_version"):
                with self.subTest(partial=partial, field=field):
                    root = self.root / f"strict_{partial}_{field}"
                    path = write_verified_matrix(root)
                    manifest = json.loads(path.read_bytes())
                    entry = manifest["runs"][0]
                    target = path.parent / entry["summary"]
                    summary = json.loads(target.read_bytes())
                    if field == "program":
                        summary["matrix_provenance"]["execution"]["git"]["head"] = "0" * 40
                    elif field == "config":
                        summary["config"]["threshold_strategy"] = "other_calibration"
                    else:
                        summary["matrix_provenance"]["data_version"] = "other_version"
                    target.write_text(json.dumps(summary), encoding="utf-8")
                    entry["artifact_sha256"]["summary.json"] = file_sha(target)
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        aggregate(path, allow_partial=partial)

    def test_verified_methods_keep_same_scientific_aggregate_and_actual_parent_hashes(self):
        from core.provenance import file_sha

        path = write_verified_matrix(self.root / "verified")
        current, legacy = aggregate(path), aggregate(self.manifest_path)
        for key in ("condition_summary", "macro_summary", "paired_differences", "methods", "seeds"):
            self.assertEqual(current[key], legacy[key])
        proof = current["provenance"]
        self.assertEqual(proof["verification_status"], "VERIFIED")
        self.assertEqual(proof["manifest_source_id"], "sha256:" + file_sha(path))
        self.assertEqual(proof["manifest_sha256"], file_sha(path))
        self.assertEqual(proof["environment"]["data"]["status"], "UNKNOWN")
        self.assertEqual(proof["environment"]["data"]["reason"], "ARTIFACT_INPUTS_ONLY")
        self.assertNotIn(str(self.root), json.dumps(proof))

    def test_unknown_schema_fingerprint_git_or_lock_is_never_verified(self):
        from core.provenance import canonical_sha

        for partial in (False, True):
            for field in ("schema", "fingerprint", "git", "lock", "code"):
                with self.subTest(partial=partial, field=field):
                    path = write_verified_matrix(self.root / f"unknown_{partial}_{field}")
                    manifest = json.loads(path.read_bytes())
                    if field == "schema":
                        manifest["schema_version"] = 99
                    elif field == "fingerprint":
                        manifest["fingerprint_version"] = "unknown_schema"
                    elif field == "git":
                        manifest["contract"]["execution"]["git"]["status"] = "UNKNOWN"
                    elif field == "lock":
                        manifest["contract"]["execution"]["constraints_sha256"] = None
                    else:
                        manifest["contract"]["execution"]["code_sha256"].pop("src/core/feature_schema.py")
                    manifest["contract_sha256"] = canonical_sha(manifest["contract"])
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    original = path.read_bytes()
                    with self.assertRaises(ValueError):
                        aggregate(path, allow_partial=partial)
                    self.assertEqual(path.read_bytes(), original)

    def test_verified_missing_or_damaged_cell_only_allows_explicit_partial(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                path = write_verified_matrix(self.root / f"partial_{missing}")
                manifest = json.loads(path.read_bytes())
                entry = manifest["runs"][0]
                csv = path.parent / "runs" / entry["run_id"] / "results.csv"
                if missing:
                    csv.unlink()
                else:
                    csv.write_bytes(csv.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    aggregate(path)
                result = aggregate(path, allow_partial=True)
                self.assertEqual((result["status"], result["accepted_runs"]), ("PARTIAL", 3))
                self.assertEqual(result["provenance"]["verification_status"], "VERIFIED")
                self.assertEqual(len(result["provenance"]["parent_runs"]), 3)
                self.assertEqual(result["rejected_runs"][0]["run_id"], entry["run_id"])

    def test_verified_absent_manifest_entry_is_partial_not_complete(self):
        path = write_verified_matrix(self.root / "missing_entry")
        manifest = json.loads(path.read_bytes())
        manifest["runs"].pop()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            aggregate(path)
        result = aggregate(path, allow_partial=True)
        self.assertEqual((result["status"], result["accepted_runs"]), ("PARTIAL", 3))
        self.assertEqual(result["expected_runs"], 4)

    def test_empty_pair_intersection_is_rejected_instead_of_nan(self):
        for run_id in ("T1_6000rpm_seed42_mahalanobis", "T1_6000rpm_seed123_knn"):
            (self._run_dir(run_id) / "run.log").unlink()
        with self.assertRaisesRegex(ValueError, "共同工況"):
            aggregate(self.manifest_path, allow_partial=True)

    def test_declared_completed_count_cannot_inflate_observed_rows(self):
        manifest = self._manifest()
        manifest["completed_runs"] = 54
        self._save(manifest)
        with self.assertRaises(ValueError):
            aggregate(self.manifest_path)
        result = aggregate(self.manifest_path, allow_partial=True)
        self.assertEqual((result["status"], result["completed_runs"], result["accepted_runs"]), ("PARTIAL", 4, 4))

    def test_each_parent_role_change_during_read_is_fatal(self):
        for name in ("summary.json", "results.csv", "run.log"):
            for partial in (False, True):
                with self.subTest(name=name, partial=partial):
                    write_matrix(self.root)
                    validator = module.validate_run
                    changed = []
                    def validate(run_dir, *args, **kwargs):
                        reasons = validator(run_dir, *args, **kwargs)
                        if not changed:
                            changed.append(True)
                            target = run_dir / name
                            target.write_bytes(target.read_bytes() + b"\n")
                        return reasons
                    with patch.object(module, "validate_run", side_effect=validate), self.assertRaises(ValueError):
                        aggregate(self.manifest_path, allow_partial=partial)

    def test_parent_change_during_write_leaves_failed_not_completed(self):
        original = pd.DataFrame.to_csv
        changed = []
        def write(frame, *args, **kwargs):
            value = original(frame, *args, **kwargs)
            if not changed:
                changed.append(True)
                self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b"\n")
            return value
        destination = self.root / "failed_write"
        with patch.object(pd.DataFrame, "to_csv", write), self.assertRaises(ValueError):
            write_aggregate(self.manifest_path, destination)
        result = json.loads((destination / "aggregate.json").read_bytes())
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["provenance"]["verification_status"], "INCOMPLETE_PARENT_CHANGED")

    def test_writer_emits_reproducible_artifacts(self) -> None:
        first, second = self.root / "a", self.root / "b"
        write_aggregate(self.manifest_path, first)
        write_aggregate(self.manifest_path, second)
        for name in ("aggregate.json", "condition_summary.csv", "macro_summary.csv", "paired_differences.csv", "aggregate.md"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_manifest_order_does_not_change_output(self) -> None:
        expected = aggregate(self.manifest_path)
        old_proof = expected.pop("provenance")
        manifest = self._manifest()
        manifest["runs"].reverse()
        self._save(manifest)
        actual = aggregate(self.manifest_path)
        new_proof = actual.pop("provenance")
        self.assertEqual(actual, expected)
        self.assertNotEqual(old_proof["manifest_sha256"], new_proof["manifest_sha256"])
        self.assertEqual(old_proof["parent_runs"], new_proof["parent_runs"])

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
        from core.provenance import file_sha

        before = {p.relative_to(ARCHIVE).as_posix(): file_sha(p) for p in ARCHIVE.rglob("*") if p.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            result = write_aggregate(ARCHIVE / "matrix_manifest.json", directory)
            self.assertEqual((result["status"], result["expected_runs"]), ("completed", 54))
            self.assertEqual(result["provenance"]["verification_status"], "LEGACY_UNVERIFIED")
            for name in ("aggregate.json", "condition_summary.csv", "macro_summary.csv", "paired_differences.csv"):
                current = (Path(directory) / name).read_bytes()
                original = (ARCHIVE / "aggregate" / name).read_bytes()
                if name == "aggregate.json":
                    value = json.loads(current)
                    value.pop("provenance")
                    value["schema_version"] = 1
                    self.assertEqual(value, json.loads(original))
                else:
                    self.assertEqual(current, original, name)
        after = {p.relative_to(ARCHIVE).as_posix(): file_sha(p) for p in ARCHIVE.rglob("*") if p.is_file()}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
