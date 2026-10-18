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
from experiments.health_index_matrix import FORMAL_CONDITION_COUNT, expected_run_ids, main


def _fake_run(data_root, *, seed, openset_method, require_nine, pool_cache=None):
    rows = [{"condition": f"c{index}", "seed": seed, "method": openset_method} for index in range(FORMAL_CONDITION_COUNT)]
    return {"status": "completed", "seed": seed, "method": openset_method,
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
        with patch.object(health_index_matrix, "run", _fake_run), contextlib.redirect_stdout(stdout):
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
            finally:
                logger.remove()
                os.chdir(previous)

    def test_run_matrix_has_no_default_output_root(self):
        with self.assertRaises(TypeError):
            health_index_matrix.run_matrix("data")
