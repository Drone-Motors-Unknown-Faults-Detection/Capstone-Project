"""內容指紋、portable/private與原科學計算不變的fixture。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from datetime import datetime

import numpy as np
from loguru import logger

from core.provenance import (CONTENT_VERSION, MATERIALIZATION_VERSION, ProvenanceError,
                             capture_source, file_sha, run, write_private_context)
from core import formal_data
from core.logger import setup_run
from core.openset import create_openset_detector
from experiments import exp6_formal_benchmark as formal
from tests.test_exp6_benchmark import _fixture
from tests.test_formal_materialization_safety import stage_source

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "06189bda0b3e83f620ccd08679fcf4bee935dc0a"


def old_module():
    source = subprocess.check_output(["git", "show", f"{BASELINE}:experiments/exp6_formal_benchmark.py"], cwd=ROOT)
    module = types.ModuleType("_provenance_fixed_baseline")
    module.__file__ = str(ROOT / "experiments/exp6_formal_benchmark.py")
    sys.modules[module.__name__] = module
    exec(compile(source, "fixed_baseline/exp6_formal_benchmark.py", "exec"), module.__dict__)
    return module


def manifest(root, version=MATERIALIZATION_VERSION):
    public, _ = capture_source(root)
    legacy = version != MATERIALIZATION_VERSION
    records = [{"output": str(root / item["source_id"]) if legacy else item["source_id"],
                "source_sha256" if legacy else "output_sha256": item["sha256"]} for item in public["files"]]
    result = {"schema_version": version, "files": records, "archive_sha256": {}, "formal_contract": {}}
    if version is None:
        result.pop("schema_version")
    path = root / "formal_materialization_manifest.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


class ProvenanceTests(unittest.TestCase):
    def test_same_stat_different_bytes_and_explicit_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            path = next(root.rglob("*_clean.csv"))
            before = path.stat()
            first = formal.dataset_fingerprint(root)
            old = formal.dataset_fingerprint(root, version="legacy_fingerprint_v1")
            payload = path.read_bytes()
            changed = payload.replace(b"0", b"1", 1)
            self.assertEqual(len(changed), len(payload))
            path.write_bytes(changed)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertEqual(path.stat().st_size, before.st_size)
            self.assertNotEqual(first, formal.dataset_fingerprint(root))
            self.assertEqual(old, formal.dataset_fingerprint(root, version="legacy_fingerprint_v1"))
            with self.assertRaisesRegex(ValueError, "未知fingerprint"):
                formal.dataset_fingerprint(root, version="fake")

    def test_copy_root_same_identity_and_duplicate_content_is_not_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _fixture(root / "one")
            shutil.copytree(source, root / "two")
            self.assertEqual(capture_source(source)[0], capture_source(root / "two")[0])
            path = next(source.rglob("*_clean.csv"))
            path.with_name("copy_Group_feature_data_clean.csv").write_bytes(path.read_bytes())
            public, private = capture_source(source)
            self.assertEqual(len(public["content_aliases"]), 1)
            self.assertEqual(public["raw_session_independence"], "UNKNOWN")
            self.assertNotIn(directory, json.dumps(public))
            # Windows TEMP 可能為 8.3 別名；兩側都以實際解析根比較。
            self.assertTrue(Path(private["resolved_data_root"]).is_relative_to(root.resolve(strict=True)))

    def test_legacy_manifests_are_readonly_and_portable(self):
        for version in (None, 1, "formal_materialization_v1", "formal_materialization_v2", MATERIALIZATION_VERSION):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = _fixture(Path(directory))
                path = manifest(root, version)
                before = path.read_bytes()
                public, _ = capture_source(root)
                self.assertEqual(public["manifest_validation"], "VERIFIED_OUTPUT_BYTES_ONLY")
                self.assertEqual(path.read_bytes(), before)
                self.assertNotIn(directory, json.dumps(public))

    def test_windows_absolute_legacy_manifest_relocates_without_leaking_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            path = manifest(root, 1)
            value = json.loads(path.read_text())
            for row in value["files"]:
                relative = Path(row["output"]).relative_to(root).as_posix().replace("/", "\\")
                row["output"] = "D:\\private-user\\" + relative
            path.write_text(json.dumps(value))
            public, _ = capture_source(root)
            self.assertNotIn("private-user", json.dumps(public))
            self.assertEqual(public["manifest_schema"], 1)

    def test_corrupt_unknown_missing_duplicate_and_tampered_manifest_rejected(self):
        variants = [lambda v: v.update(schema_version="future_99"),
                    lambda v: v.update(schema_version=[]),
                    lambda v: v.update(schema_version={}),
                    lambda v: v.update(schema_version=True),
                    lambda v: v.update(files=[]),
                    lambda v: v["files"].append(v["files"][0]),
                    lambda v: v["files"][0].update(output_sha256="0" * 64),
                    lambda v: v["files"][0].pop("output_sha256"),
                    lambda v: v["files"][0].update(output="../outside.csv")]
        for mutation in variants:
            with self.subTest(mutation=variants.index(mutation)), tempfile.TemporaryDirectory() as directory:
                root = _fixture(Path(directory))
                path = manifest(root)
                value = json.loads(path.read_text())
                mutation(value)
                path.write_text(json.dumps(value))
                with self.assertRaises(ProvenanceError):
                    capture_source(root)
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            path = root / "formal_materialization_manifest.json"
            for payload in ("[0]", "{", "{}"):
                path.write_text(payload)
                with self.assertRaises(ProvenanceError):
                    capture_source(root)

    def test_empty_and_unreadable_sources_fail_without_private_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                capture_source(directory)
            with patch.object(Path, "open", side_effect=OSError("private-token-path")):
                with self.assertRaises(ProvenanceError) as caught:
                    file_sha(Path(directory) / "hidden")
            self.assertNotIn("private-token", str(caught.exception))

    def test_private_sidecar_is_atomic_and_git_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".lineage_private"
            identity = write_private_context({"resolved_data_root": "private-path"}, directory=root)
            self.assertEqual(json.loads((root / f"{identity}.json").read_text())["resolved_data_root"], "private-path")
            with patch("core.provenance.os.replace", side_effect=OSError("fixture")):
                with self.assertRaises(OSError):
                    write_private_context({}, directory=root)
            self.assertEqual(len(list(root.iterdir())), 1)
            result = subprocess.run(["git", "check-ignore", "--no-index", ".lineage_private/example.json"], cwd=ROOT,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0)

    def test_materialized_manifest_v3_has_no_absolute_roots_and_same_feature_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_source(root / "source")
            result = formal_data.materialize(root / "source", root / "out", stages=("2",))
            self.assertEqual(result["schema_version"], MATERIALIZATION_VERSION)
            self.assertNotIn(directory, json.dumps(result))
            self.assertNotIn("source_root", result)
            public, _ = capture_source(root / "out")
            self.assertEqual(public["manifest_validation"], "VERIFIED_OUTPUT_BYTES_ONLY")
            for row in result["files"]:
                self.assertEqual(row["output_sha256"], hashlib.sha256((root / "out" / row["output"]).read_bytes()).hexdigest())

    def test_formal_scientific_rows_equal_fixed_baseline_and_environment_present(self):
        old = old_module()
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            for method in ("mahalanobis", "knn"):
                with self.subTest(method=method):
                    traces = [[], []]

                    def factory(trace):
                        def build(*args, **kwargs):
                            detector = create_openset_detector(*args, **kwargs)
                            original = detector.score_samples

                            def score(samples):
                                values = original(samples)
                                trace.append((samples.copy(), values.copy()))
                                return values

                            detector.score_samples = score
                            return detector
                        return build

                    with patch.object(old, "create_openset_detector", side_effect=factory(traces[0])):
                        expected = old.run(root, openset_method=method, seed=42, require_nine=False)
                    with patch.object(formal, "create_openset_detector", side_effect=factory(traces[1])):
                        actual = formal.run(root, openset_method=method, seed=42, require_nine=False)
                    self.assertEqual(len(traces[0]), len(traces[1]))
                    self.assertGreater(len(traces[0]), 0)
                    for before, after in zip(traces[0], traces[1]):
                        np.testing.assert_array_equal(before[0], after[0])
                        np.testing.assert_array_equal(before[1], after[1])
                    for a, b in zip(actual["rows"], expected["rows"]):
                        keys = set(a) - {"inference_seconds"}
                        self.assertEqual({k: a[k] for k in keys}, {k: b[k] for k in keys})
                    self.assertEqual(actual["fingerprint_version"], CONTENT_VERSION)
                    self.assertEqual(actual["environment"]["python"]["version"].split(".")[:2], ["3", "10"])
                    self.assertNotIn(directory, json.dumps(actual).replace("\\\\", "\\"))

    def test_formal_rejects_source_mutation_and_records_public_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _fixture(Path(directory))
            before = capture_source(root)
            changed = dict(before[0], dataset_fingerprint="a" * 64)
            with patch.object(formal, "capture_source", side_effect=[before, before, (changed, before[1])]):
                with self.assertRaisesRegex(ValueError, "來源內容"):
                    formal.run(root, require_nine=False)
            output = root / "failure_evidence"
            output.mkdir()
            paths = types.SimpleNamespace(output_dir=output, program="fixture", timestamp="fixed")
            with patch.object(formal, "setup_run", return_value=(logger, paths)), \
                    patch.object(formal, "capture_source", side_effect=ProvenanceError("private-path-secret")):
                with self.assertRaises(ProvenanceError):
                    formal.run(root, require_nine=False)
            record = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "FAILED")
            self.assertEqual(record["error_type"], "ProvenanceError")
            self.assertNotIn("private-path-secret", str(record))

    def test_unique_run_does_not_overwrite_same_second_environment_or_log(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with patch("core.logger.datetime") as clock:
                    clock.now.return_value = datetime(2026, 10, 9, 14, 0, 0)
                    _, first = setup_run("fixture_unique", unique=True)
                    marker = first.output_dir / "preserve.json"
                    marker.write_text('{"保留":true}', encoding="utf-8")
                    _, second = setup_run("fixture_unique", unique=True)
                    _, third = setup_run("fixture_unique", unique=True)
                self.assertEqual(len({first.timestamp, second.timestamp, third.timestamp}), 3)
                self.assertEqual(marker.read_text(encoding="utf-8"), '{"保留":true}')
                self.assertTrue(all(p.output_dir.joinpath("environment.json").is_file() for p in (first, second, third)))
            finally:
                logger.remove()
                os.chdir(previous)

    def test_source_run_and_main_save_public_records_without_git(self):
        from core import provenance
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = _fixture(workspace / "data")
            previous = Path.cwd()
            try:
                os.chdir(workspace)
                first = run(source)
                self.assertEqual(first["status"], "VERIFIED")
                with patch("sys.argv", ["provenance", "--data-root", str(source)]):
                    provenance.main()
                public_paths = list(workspace.glob("output/source_provenance/*/source.json"))
                self.assertEqual(len(public_paths), 2)
                for path in public_paths:
                    public = json.loads(path.read_text(encoding="utf-8"))
                    self.assertNotIn(directory, json.dumps(public).replace("\\\\", "\\"))
                    environment = json.loads(path.with_name("environment.json").read_text(encoding="utf-8"))
                    self.assertEqual(environment["git"]["status"], "UNKNOWN")
                self.assertEqual(len(list(workspace.glob(".lineage_private/*.json"))), 2)
            finally:
                logger.remove()
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
