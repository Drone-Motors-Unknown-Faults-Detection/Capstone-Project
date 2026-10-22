"""exp6_formal_benchmark／exp6_matrix 對基準 commit（BASE）的回歸規範；規範見 docs/refactoring_contract.md。

現行程式與基準 commit 的同名函數在同一份 fixture 上配對執行。允許新增欄位（例如
resolved_config）或有預設值的 keyword-only 參數（例如 pool_cache），
既有欄位、參數與數值不得改變。
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.feature_schema import SCHEMAS
from experiments import exp6_formal_benchmark as formal
from experiments import exp6_matrix as matrix
from tests.exp6_fixture import condition_row

ROOT = Path(__file__).resolve().parents[2]
BASE = "39d39931e9573d6b70b3822420b085aa496f52da"  # exp6 orchestration 基準 commit
VOLATILE = {"inference_seconds", "commit_sha", "python", "dataset_fingerprint"}
# 刻意改變、另外驗證的欄位：dataset_root 由寫死的 formal_features 改記實際資料根名稱（#248）
INTENDED = {"dataset_root"}


def base_module(relative: str, name: str):
    source = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / relative)
    sys.modules[name] = module
    exec(compile(source, f"{BASE[:7]}/{relative}", "exec"), module.__dict__)
    return module


def write_dataset(root: Path, *, configs=(("8screws", 0.0), ("7screws", 0.15), ("5screws", 0.3))) -> Path:
    rng = np.random.default_rng(11)
    for motor, rows in (("T1", 40), ("T3", 33)):  # 兩個工況：split 的亂數消耗順序才有意義
        dataset = root / "Step-1" / "myfeature" / motor / "8000rpm"
        for config, center in configs:
            target = dataset / config
            target.mkdir(parents=True)
            frame = pd.DataFrame(rng.normal(center, 0.5, size=(rows, 105)), columns=SCHEMAS["historical_clean_v1"])
            frame.to_csv(target / f"{motor}_Group_feature_data_clean.csv", index=False)
    return root


def assert_signature_extends(test: unittest.TestCase, old, new) -> None:
    old_params = inspect.signature(old).parameters
    new_params = inspect.signature(new).parameters
    for name, parameter in old_params.items():
        test.assertIn(name, new_params, name)
        test.assertEqual((new_params[name].kind, new_params[name].default), (parameter.kind, parameter.default), name)
    for name in set(new_params) - set(old_params):
        test.assertEqual(new_params[name].kind, inspect.Parameter.KEYWORD_ONLY, name)
        test.assertIsNot(new_params[name].default, inspect.Parameter.empty, name)


class FormalBenchmarkContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old = base_module("src/experiments/exp6_formal_benchmark.py", "_contract_base_formal")

    def _traced(self, module, root: Path, **kwargs):
        splits, scores = [], []
        real_split = module.make_split
        real_factory = module.create_openset_detector

        def split(n, rng):
            result = real_split(n, rng)
            splits.append((n, result.train.tolist(), result.cal.tolist(), result.holdout.tolist()))
            return result

        def factory(*args, **options):
            detector = real_factory(*args, **options)
            original = detector.score_samples

            def score(samples):
                values = original(samples)
                scores.append((np.array(samples, copy=True), np.array(values, copy=True)))
                return values

            detector.score_samples = score
            return detector

        with patch.object(module, "make_split", side_effect=split), \
                patch.object(module, "create_openset_detector", side_effect=factory):
            result = module._run_benchmark(root, require_nine=False, **kwargs)
        return result, splits, scores

    def test_signatures_only_grow(self) -> None:
        for name in ("run", "_run_benchmark", "main"):
            with self.subTest(name=name):
                assert_signature_extends(self, getattr(self.old, name), getattr(formal, name))

    def test_rows_rng_order_and_detector_inputs_match_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_dataset(Path(directory))
            for method in ("mahalanobis", "knn"):
                with self.subTest(method=method):
                    old, old_splits, old_scores = self._traced(self.old, root, seed=42, openset_method=method)
                    new, new_splits, new_scores = self._traced(formal, root, seed=42, openset_method=method)
                    self.assertEqual(new_splits, old_splits)
                    self.assertEqual(len(new_scores), len(old_scores))
                    for (a_in, a_out), (b_in, b_out) in zip(old_scores, new_scores):
                        np.testing.assert_array_equal(a_in, b_in)
                        np.testing.assert_array_equal(a_out, b_out)
                    self.assertTrue(set(old).issubset(new), set(old) - set(new))
                    for key in set(old) - VOLATILE - INTENDED - {"rows"}:
                        self.assertEqual(new[key], old[key], key)
                    self.assertEqual(new["dataset_root"], root.name)
                    self.assertEqual(len(new["rows"]), len(old["rows"]))
                    for a, b in zip(old["rows"], new["rows"]):
                        self.assertTrue(set(a).issubset(b), set(a) - set(b))
                        for key in set(a) - VOLATILE:
                            self.assertEqual(b[key], a[key], key)

    def test_errors_match_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases = {
                "legacy": (write_dataset(Path(directory) / "legacy"), {"mahalanobis_method": "legacy"}),
                "no_unknown": (write_dataset(Path(directory) / "healthy_only", configs=(("8screws", 0.0),)), {}),
            }
            for label, (root, kwargs) in cases.items():
                with self.subTest(case=label):
                    errors = []
                    for module in (self.old, formal):
                        with self.assertRaises(Exception) as caught:
                            module._run_benchmark(root, require_nine=False, **kwargs)
                        errors.append((type(caught.exception), str(caught.exception)))
                    self.assertEqual(errors[1], errors[0])


class MatrixContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old = base_module("src/experiments/exp6_matrix.py", "_contract_base_matrix")

    @staticmethod
    def _entries():
        rows = []
        for seed in (42, 123):
            for method in ("mahalanobis", "knn"):
                run_id = matrix.run_key("T1", "6000rpm", seed, method)
                rows.append({"run_id": run_id, "condition": "T1/6000rpm", "motor": "T1", "rpm": "6000rpm",
                             "seed": seed, "method": method, "status": "pending",
                             "log_path": f"runs/{run_id}/run.log"})
        return rows

    def _run(self, module, output: Path, fail: set[str] = frozenset(), resume: bool = True) -> tuple[dict, list]:
        calls = []

        def fake_run(data_path, *, seed, openset_method, require_nine, **_):
            run_id = matrix.run_key("T1", "6000rpm", seed, openset_method)
            calls.append(run_id)
            if run_id in fail:
                raise RuntimeError("注入中斷")
            row = condition_row("T1", "6000rpm", seed, openset_method)
            other = dict(row, motor="T3", rpm="8000rpm", dataset="T3/8000")
            return {"status": "completed", "method": openset_method, "seed": seed, "dataset_fingerprint": "fp",
                    "commit_sha": "fixture", "formal_condition_count": 2, "rows": [other, row]}

        with patch.object(module, "expected_run_matrix", side_effect=lambda *a, **k: self._entries()), \
                patch.object(module, "dataset_fingerprint", return_value="fp"), \
                patch.object(module, "run", side_effect=fake_run):
            manifest = module.run_matrix("data", output, seeds=(42, 123), methods=("mahalanobis", "knn"),
                                         resume=resume)
        return manifest, calls

    @staticmethod
    def _stable(manifest: dict) -> dict:
        drop = {"started_at_utc", "finished_at_utc", "duration_seconds"}
        stable = {k: v for k, v in manifest.items() if k not in drop}
        stable["runs"] = [{k: v for k, v in run.items() if k not in drop} for run in manifest["runs"]]
        return stable

    def test_signatures_only_grow(self) -> None:
        for name in ("run_matrix", "main", "expected_run_matrix", "validate_run"):
            with self.subTest(name=name):
                assert_signature_extends(self, getattr(self.old, name), getattr(matrix, name))

    def test_manifest_and_artifacts_match_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / "old", Path(directory) / "new"]
            manifests = [self._run(module, root)[0] for module, root in zip((self.old, matrix), roots)]
            self.assertEqual(self._stable(manifests[1]), self._stable(manifests[0]))
            for entry in manifests[0]["runs"]:
                for name in ("summary.json", "results.csv"):
                    old_bytes = (roots[0] / "runs" / entry["run_id"] / name).read_bytes()
                    new_bytes = (roots[1] / "runs" / entry["run_id"] / name).read_bytes()
                    self.assertEqual(new_bytes, old_bytes, f"{entry['run_id']}/{name}")

    def test_interrupted_run_is_recorded_and_only_it_reruns(self) -> None:
        failing = matrix.run_key("T1", "6000rpm", 42, "knn")
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for module, root in ((self.old, Path(directory) / "old"), (matrix, Path(directory) / "new")):
                first, _ = self._run(module, root, fail={failing})
                second, calls = self._run(module, root)
                results.append((self._stable(first), self._stable(second), calls))
            self.assertEqual(results[1], results[0])
            first, second, calls = results[1]
            self.assertEqual((first["status"], first["failed_runs"]), ("incomplete", 1))
            failed = next(run for run in first["runs"] if run["run_id"] == failing)
            self.assertEqual(failed["error"], "RuntimeError: 注入中斷")
            self.assertEqual(calls, [failing])
            self.assertEqual(second["status"], "completed")

    def test_condition_row_errors_match_base(self) -> None:
        def zero_rows(data_path, *, seed, openset_method, require_nine, **_):
            return {"status": "completed", "method": openset_method, "seed": seed, "dataset_fingerprint": "fp",
                    "commit_sha": "fixture", "rows": []}

        with tempfile.TemporaryDirectory() as directory:
            errors = []
            for module, root in ((self.old, Path(directory) / "old"), (matrix, Path(directory) / "new")):
                with patch.object(module, "expected_run_matrix", side_effect=lambda *a, **k: self._entries()[:1]), \
                        patch.object(module, "dataset_fingerprint", return_value="fp"), \
                        patch.object(module, "run", side_effect=zero_rows):
                    manifest = module.run_matrix("data", root, seeds=(42,), methods=("mahalanobis",))
                errors.append(manifest["runs"][0]["error"])
                log = (root / "runs" / manifest["runs"][0]["run_id"] / "run.log").read_text(encoding="utf-8")
                self.assertIn("status=failed", log)
            self.assertEqual(errors[1], errors[0])
            self.assertEqual(errors[0], "ValueError: expected one row for T1/6000rpm, found 0")


if __name__ == "__main__":
    unittest.main()
