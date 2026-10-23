import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger

from experiments import health_index_matrix
from experiments.health_index_matrix import FORMAL_CONDITION_COUNT, RESULT_SCHEMA_VERSION, expected_run_ids, main


FINGERPRINT = "fp"


def _fake_run(data_root, *, seed, openset_method, require_nine, pool_cache=None):
    rows = [{"condition": f"c{index}", "seed": seed, "method": openset_method} for index in range(FORMAL_CONDITION_COUNT)]
    return {"schema_version": RESULT_SCHEMA_VERSION, "status": "completed", "seed": seed, "method": openset_method, "dataset_fingerprint": FINGERPRINT,
            "formal_condition_count": FORMAL_CONDITION_COUNT, "rows": rows}


class HealthMatrixTests(unittest.TestCase):
    def test_default_matrix_has_three_seeds_and_two_methods(self):
        run_ids = expected_run_ids()
        self.assertEqual(len(run_ids), 6)
        self.assertIn("seed_42_mahalanobis", run_ids)
        self.assertIn("seed_2026_knn", run_ids)

    def test_method_aliases_are_canonicalized(self):
        self.assertEqual(expected_run_ids([7], ["maha", "k-nn"]), ["seed_7_mahalanobis", "seed_7_knn"])

    def test_invalid_method_is_rejected(self):
        with self.assertRaises(ValueError):
            expected_run_ids([1], ["invalid"])

    def _main(self, argv):
        stdout = io.StringIO()
        with patch.object(health_index_matrix, "run", _fake_run), \
                patch.object(health_index_matrix, "dataset_fingerprint", return_value=FINGERPRINT), \
                contextlib.redirect_stdout(stdout):
            code = main(argv)
        return code, json.loads(stdout.getvalue())

    def test_cli_default_writes_setup_run_output_not_reports(self):
        """不給 --output-root 時寫 output/exp8_health_index_matrix/{ts}/，並有 logs。"""
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as workspace:
            try:
                os.chdir(workspace)
                code, summary = self._main(["--seed", "42", "--method", "mahalanobis", "--method", "knn"])
                self.assertEqual(code, 0)
                (output_dir,) = Path("output/exp8_health_index_matrix").iterdir()
                self.assertEqual(Path(summary["output_root"]).resolve(), output_dir.resolve())
                manifest = json.loads((output_dir / "matrix_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["completed_runs"], 2)
                self.assertTrue((output_dir / "seed_42" / "knn.csv").is_file())
                self.assertTrue((output_dir / "environment.json").is_file())
                (log_file,) = Path("logs/exp8_health_index_matrix").glob("*.log")
                self.assertIn("seed_42_knn：完成 9 列", log_file.read_text(encoding="utf-8"))
                self.assertFalse(Path("reports").exists())
            finally:
                logger.remove()
                os.chdir(previous)

    def test_cli_explicit_output_root_resumes_there(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as workspace:
            try:
                os.chdir(workspace)
                argv = ["--seed", "7", "--method", "knn", "--output-root", "resume_root"]
                self.assertEqual(self._main(argv)[0], 0)
                code, summary = self._main(argv)
                self.assertEqual(code, 0)
                self.assertEqual(summary["output_root"], "resume_root")
                manifest = json.loads(Path("resume_root/matrix_manifest.json").read_text(encoding="utf-8"))
                self.assertTrue(manifest["runs"][0]["resumed"])
                self.assertEqual(manifest["dataset_fingerprint"], FINGERPRINT)
                self.assertEqual(manifest["runs"][0]["dataset_fingerprint"], FINGERPRINT)
            finally:
                logger.remove()
                os.chdir(previous)

    def test_resume_reruns_when_dataset_fingerprint_differs(self):
        """既有 JSON 的指紋跟現在的資料不同就重跑，不沿用（#269 C06）。"""
        calls: list = []

        def counting_run(*args, **kwargs):
            calls.append(kwargs["seed"])
            return _fake_run(*args, **kwargs)

        with tempfile.TemporaryDirectory() as workspace:
            output = Path(workspace) / "matrix"
            with patch.object(health_index_matrix, "run", counting_run), \
                    patch.object(health_index_matrix, "dataset_fingerprint", return_value=FINGERPRINT):
                health_index_matrix.run_matrix("data", output, seeds=(42,), methods=("knn",))
                health_index_matrix.run_matrix("data", output, seeds=(42,), methods=("knn",))
            self.assertEqual(calls, [42])
            with patch.object(health_index_matrix, "run", counting_run), \
                    patch.object(health_index_matrix, "dataset_fingerprint", return_value="other"):
                manifest = health_index_matrix.run_matrix("data", output, seeds=(42,), methods=("knn",))
            self.assertEqual(calls, [42, 42])
            self.assertNotIn("resumed", manifest["runs"][0])
            self.assertEqual(manifest["runs"][0]["status"], "failed")
            self.assertIn("dataset_fingerprint changed", manifest["runs"][0]["error"])

    def test_resume_reruns_results_from_older_schema(self):
        """舊映射（schema_version 1）的結果不沿用，manifest 記下目前版本（#301 F-01）。"""
        calls: list = []

        def counting_run(*args, **kwargs):
            calls.append(kwargs["seed"])
            return _fake_run(*args, **kwargs)

        self.assertEqual(RESULT_SCHEMA_VERSION, 2)
        with tempfile.TemporaryDirectory() as workspace:
            output = Path(workspace) / "matrix"
            old = _fake_run("data", seed=42, openset_method="knn", require_nine=True)
            old["schema_version"] = 1
            (output / "seed_42").mkdir(parents=True)
            (output / "seed_42" / "knn.json").write_text(json.dumps(old), encoding="utf-8")
            with patch.object(health_index_matrix, "run", counting_run), \
                    patch.object(health_index_matrix, "dataset_fingerprint", return_value=FINGERPRINT):
                manifest = health_index_matrix.run_matrix("data", output, seeds=(42,), methods=("knn",))
            self.assertEqual(calls, [42])
            self.assertNotIn("resumed", manifest["runs"][0])
            self.assertEqual(manifest["result_schema_version"], RESULT_SCHEMA_VERSION)
            cached = json.loads((output / "seed_42" / "knn.json").read_text(encoding="utf-8"))
            self.assertEqual(cached["schema_version"], RESULT_SCHEMA_VERSION)

    def test_run_rejects_result_with_unexpected_schema(self):
        def stale_run(*args, **kwargs):
            return {**_fake_run(*args, **kwargs), "schema_version": 1}

        with tempfile.TemporaryDirectory() as workspace:
            with patch.object(health_index_matrix, "run", stale_run), \
                    patch.object(health_index_matrix, "dataset_fingerprint", return_value=FINGERPRINT):
                manifest = health_index_matrix.run_matrix("data", Path(workspace), seeds=(42,), methods=("knn",))
        self.assertEqual(manifest["runs"][0]["status"], "failed")
        self.assertIn("schema_version", manifest["runs"][0]["error"])

    def test_run_matrix_has_no_default_output_root(self):
        with self.assertRaises(TypeError):
            health_index_matrix.run_matrix("data")
