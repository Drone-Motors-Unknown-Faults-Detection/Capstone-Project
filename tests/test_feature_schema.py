"""正式格式契約fixture；完全不讀ignored data。"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from loguru import logger

from core.data import load_pools, make_split, run
from core.feature_schema import (SCHEMAS, CONFIGURATIONS, FeatureSchemaError,
                                 validate_pool_coverage)
from core.formal_data import FEATURE_NAMES
from experiments.exp6_formal_benchmark import run as formal_run

ROOT = Path(__file__).resolve().parents[1]


def write(root, config="8screws", *, header=None, rows=None, suffix="fixture"):
    folder = root / config
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{suffix}_Group_feature_data_clean.csv"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(SCHEMAS["historical_clean_v1"] if header is None else header)
    writer.writerows(np.ones((6, 105)) if rows is None else rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(stream.getvalue())
    return path


class FeatureSchemaTests(unittest.TestCase):
    def test_registry_matches_committed_contract_and_adapter(self):
        spec = json.loads((ROOT / "docs/formal_feature_schema.json").read_text(encoding="utf-8"))
        self.assertEqual({k: list(v) for k, v in SCHEMAS.items()}, spec["schemas"])
        self.assertEqual(list(SCHEMAS["adapter_generated_v1"]), FEATURE_NAMES)
        self.assertEqual(spec["runtime_source"], "core/feature_schema.py:SCHEMAS")
        self.assertEqual(spec["operation_manual"], "docs/formal_feature_schema_contract.md")

    def test_both_exact_versions_bom_numeric_values_and_split_unchanged(self):
        values = np.random.default_rng(42).normal(size=(32, 105))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version, columns in SCHEMAS.items():
                with self.subTest(version=version):
                    path = write(root, header=columns, rows=values)
                    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
                    expected = pd.read_csv(path).select_dtypes("number").to_numpy(dtype=float)
                    audit = {}
                    actual = load_pools(root, audit=audit)["8screws"]
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(audit["files"][0]["schema_version"], version)
                    self.assertEqual(audit["rejected_rows"], 0)
                    self.assertEqual(audit["coverage"]["status"], "INCOMPLETE")
                    for name in ("train", "cal", "holdout"):
                        np.testing.assert_array_equal(
                            getattr(make_split(len(actual), np.random.default_rng(42)), name),
                            getattr(make_split(len(expected), np.random.default_rng(42)), name))

    def test_all_header_rejections_and_counts(self):
        columns = list(SCHEMAS["historical_clean_v1"])
        swapped = columns.copy()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        duplicate = columns.copy()
        duplicate[1] = duplicate[0]
        mixed = columns.copy()
        mixed[12] = SCHEMAS["adapter_generated_v1"][12]
        variants = [(columns[:-1], "ordered_header_mismatch"),
                    (columns + ["extra"], "ordered_header_mismatch"),
                    (swapped, "ordered_header_mismatch"), (duplicate, "duplicate_header"),
                    (mixed, "ordered_header_mismatch"),
                    ([f"f{i}" for i in range(105)], "ordered_header_mismatch")]
        for header, reason in variants:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = write(root, header=header)
                before = path.read_bytes()
                audit = {}
                with self.assertRaisesRegex(FeatureSchemaError, reason):
                    load_pools(root, audit=audit)
                self.assertEqual(audit["status"], "REJECTED")
                self.assertEqual(audit["rejected_rows"], 6)
                self.assertEqual(audit["discarded_rows"], 0)
                self.assertEqual(before, path.read_bytes())
                self.assertNotIn(directory, str(audit))

    def test_non_numeric_and_non_finite_never_silently_drop(self):
        for value, reason in (("bad", "non_numeric"), ("NaN", "non_finite"),
                              ("inf", "non_finite"), ("-inf", "non_finite"), ("", "non_finite")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                rows = [["1"] * 105 for _ in range(3)]
                rows[1][19] = value
                root = Path(directory)
                write(root, rows=rows)
                audit = {}
                with self.assertRaisesRegex(FeatureSchemaError, reason):
                    load_pools(root, audit=audit)
                self.assertEqual(audit["rejection"]["rejected_row_indices"], [1])
                self.assertEqual(audit["rejected_rows"], 1)

    def test_empty_file_header_only_blank_row_ragged_and_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rows, reason in (([], "empty_configuration"), ([[]], "row_width_mismatch"),
                                  ([[1] * 104], "row_width_mismatch")):
                path = write(root, rows=rows)
                with self.assertRaisesRegex(FeatureSchemaError, reason):
                    load_pools(root)
            for payload, reason in ((b"", "empty_file"), (b'"unclosed', "unreadable_csv"),
                                     (b"\xff\xff", "unreadable_csv")):
                path.write_bytes(payload)
                with self.assertRaisesRegex(FeatureSchemaError, reason):
                    load_pools(root)

    def test_wrong_label_empty_class_and_missing_healthy(self):
        for config, reason in (("8screw", "unknown_configuration"), ("7screws", "missing_healthy")):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                write(Path(directory), config)
                with self.assertRaisesRegex(FeatureSchemaError, reason):
                    load_pools(directory)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root)
            (root / "7screws").mkdir()
            with self.assertRaisesRegex(FeatureSchemaError, "empty_configuration"):
                load_pools(root)

    def test_alias_preserved_but_double_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root)
            write(root, "1screw")
            loaded = load_pools(root)
            self.assertIn("1screw", loaded)
            self.assertNotIn("1screws", loaded)
            write(root, "1screws")
            with self.assertRaisesRegex(FeatureSchemaError, "ambiguous_configuration_alias"):
                load_pools(root)

    def test_healthy_only_valid_but_not_complete_matrix_and_full_alias_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root)
            self.assertEqual(list(load_pools(root)), ["8screws"])
            with self.assertRaisesRegex(FeatureSchemaError, "incomplete_matrix_coverage"):
                load_pools(root, require_complete=True)
            for name in CONFIGURATIONS[1:]:
                write(root, "1screw" if name == "1screws" else name)
            pools = load_pools(root, require_complete=True)
            self.assertEqual(validate_pool_coverage(pools)["status"], "VERIFIED")

    def test_multiple_files_stable_order_and_auxiliary_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, rows=np.ones((3, 105)), suffix="b")
            write(root, rows=np.zeros((4, 105)), suffix="a")
            (root / "notes").mkdir()
            first, second = {}, {}
            a, b = load_pools(root, audit=first), load_pools(root, audit=second)
            np.testing.assert_array_equal(a["8screws"], np.vstack([np.zeros((4, 105)), np.ones((3, 105))]))
            np.testing.assert_array_equal(a["8screws"], b["8screws"])
            self.assertEqual(first, second)

    def test_formal_complete_coverage_rejects_before_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "Step-1/myfeature/T1/8000rpm"
            write(dataset)
            ds = [{"motor": f"T{m}", "rpm": rpm, "path": dataset}
                  for m in (1, 2, 3) for rpm in ("6000rpm", "8000rpm", "11000rpm")]
            with patch("experiments.exp6_formal_benchmark.discover_datasets", return_value=ds), \
                    patch("experiments.exp6_formal_benchmark.create_openset_detector") as factory:
                with self.assertRaisesRegex(FeatureSchemaError, "incomplete_matrix_coverage"):
                    formal_run(root, require_nine=True)
                factory.assert_not_called()

    def test_run_and_cli_failure_are_public_and_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, header=[f"f{i}" for i in range(105)])
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                result = run(root)
                output = next((root / "output/formal_schema_audit").glob("*/schema_audit.json"))
                self.assertNotIn(directory, output.read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "REJECTED")
            finally:
                logger.remove()
                os.chdir(old_cwd)
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
            child = subprocess.run([sys.executable, "-m", "core.data", "--dataset", directory],
                                   cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(child.returncode, 1, child.stderr)
            self.assertIn("ordered_header_mismatch", child.stderr)
            self.assertNotIn(directory, child.stderr)

    def test_cli_healthy_only_and_complete_policy_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root)
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
            for complete, expected in ((False, 0), (True, 1)):
                with self.subTest(complete=complete):
                    command = [sys.executable, "-m", "core.data", "--dataset", directory]
                    if complete:
                        command.append("--require-complete")
                    result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                            text=True, encoding="utf-8", timeout=60)
                    self.assertEqual(result.returncode, expected, result.stderr)

    def test_unreadable_dataset_is_rejected_without_private_exception_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = {}
            with patch("core.data._load_pools", side_effect=OSError("私人路徑token-secret")):
                with self.assertRaisesRegex(FeatureSchemaError, "unreadable_dataset") as caught:
                    load_pools(directory, audit=audit)
            self.assertEqual(audit["status"], "REJECTED")
            self.assertNotIn("token-secret", str(caught.exception))
            self.assertNotIn(directory, str(audit))


if __name__ == "__main__":
    unittest.main()
