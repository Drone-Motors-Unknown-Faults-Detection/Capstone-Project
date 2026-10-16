"""archive 路徑、通道形狀與寫入中斷；不讀正式 ZIP／data。"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

import numpy as np
import pandas as pd
from loguru import logger

from core import formal_data as formal

BASE = "0282209959183bf7b8654115b175a94aafa26f35"


def zip_bytes(entries):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return stream.getvalue()


def channel_entries(config="8screws", *, unequal_windows=False, unequal_rows=False):
    rng = np.random.default_rng(42)
    result = []
    for channel in ("Current", "X", "Y", "Z", "Delta_T"):
        width = 2 if unequal_windows and channel == "Y" else 3
        length = 127 if unequal_rows and channel == "Z" else 128
        frame = pd.DataFrame(rng.normal(size=(length, width)) + 1)
        result.append((f"{config}/T2_{channel}_data.csv", frame.to_csv(index=False).encode("utf-8")))
    return result


def stage_source(root, entries=None):
    root.mkdir()
    condition = zip_bytes(channel_entries() if entries is None else entries)
    csv = zip_bytes([("T2/8000rpm.zip", condition)])
    archive = root / "階段2.zip"
    archive.write_bytes(zip_bytes([("csv.zip", csv)]))
    return archive


@lru_cache(None)
def old_module():
    module = types.ModuleType("_formal_materialization_baseline")
    sys.modules[module.__name__] = module
    source = subprocess.check_output(["git", "show", f"{BASE}:core/formal_data.py"])
    exec(compile(source, f"{BASE}/core/formal_data.py", "exec"), module.__dict__)
    return module


@contextmanager
def cwd(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        logger.remove()
        os.chdir(previous)


class MaterializationSafetyTests(unittest.TestCase):
    def test_member_paths_reject_cross_platform_traversal(self):
        invalid = ("../8screws/T2_X_data.csv", "a/../../8screws/T2_X_data.csv", "/8screws/T2_X_data.csv",
                   "C:/8screws/T2_X_data.csv", "C:8screws/T2_X_data.csv", "//host/share/data.csv",
                   "\\\\host\\share\\data.csv", "8screws\\T2_X_data.csv", "a/./file", "a//file")
        for name in invalid:
            with self.subTest(name=name), self.assertRaises(formal.FormalDataError):
                formal._archive_path(name)

    def test_bad_member_causes_no_output_even_after_valid_member(self):
        for name in ("../7screws/T2_X_data.csv", "/8screws/T2_X_data.csv", "C:/8screws/T2_X_data.csv",
                     "//host/share/T2_X_data.csv", "../../outside/T2_X_data.csv"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = stage_source(root / "source", [*channel_entries(), (name, b"0\n1\n")])
                before = source.read_bytes()
                with self.assertRaises(formal.FormalDataError):
                    formal.materialize(root / "source", root / "output", stages=["2"])
                self.assertFalse((root / "output").exists())
                self.assertFalse((root / "outside").exists())
                self.assertEqual(source.read_bytes(), before)

    def test_archive_symlink_and_duplicate_member_rejected(self):
        info = ZipInfo("8screws/T2_X_data.csv")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        for entries in ([(info, b"../../outside")], [("same", b"a"), ("same", b"b")]):
            with self.subTest(case=str(entries)), ZipFile(io.BytesIO(zip_bytes(entries))) as archive:
                with self.assertRaises(formal.FormalDataError):
                    formal._validate_archive(archive)

    def test_safe_relative_member_and_unknown_config(self):
        self.assertEqual(formal._safe_relative_member("myfeature/T1/8000rpm/8screws/a.csv", "myfeature"),
                         Path("T1/8000rpm/8screws/a.csv"))
        for config in ("..", "../output_alias", "unknown", "C:"):
            with self.subTest(config=config), self.assertRaises(formal.FormalDataError):
                formal._validate_condition("T2", "8000rpm", config)
        formal._validate_condition("T2", "8000rpm", "1screw")

    def test_string_prefix_alias_is_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            alias = Path(directory) / "output_alias" / "payload.csv"
            with self.assertRaises(formal.FormalDataError):
                formal._write_bytes(alias, b"payload", force=False, output_root=root)
            self.assertFalse(alias.parent.exists())

    def test_existing_output_symlink_rejected_without_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "output"
            out.mkdir()
            outside = root / "outside"
            outside.mkdir()
            try:
                (out / "Step-2").symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"此平台未提供 symlink 權限：{type(exc).__name__}")
            stage_source(root / "source")
            with self.assertRaises(formal.FormalDataError):
                formal.materialize(root / "source", out, stages=["2"])
            self.assertEqual(list(outside.iterdir()), [])

    def test_same_root_symlink_is_also_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").mkdir()
            try:
                (root / "link").symlink_to(root / "real", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"此平台未提供 symlink 權限：{type(exc).__name__}")
            with self.assertRaises(formal.FormalDataError):
                formal._write_bytes(root / "link" / "file", b"bytes", force=False, output_root=root)
            self.assertEqual(list((root / "real").iterdir()), [])

    def test_caller_declared_root_alias_resolves_consistently(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            real = parent / "real"
            real.mkdir()
            alias = parent / "alias"
            try:
                alias.symlink_to(real, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"此平台未提供 symlink 權限：{type(exc).__name__}")
            formal._write_bytes(alias / "legal.csv", b"legal", force=False, output_root=alias)
            self.assertEqual((real / "legal.csv").read_bytes(), b"legal")

    @unittest.skipUnless(os.name == "nt", "Windows 8.3 路徑專用回歸")
    def test_windows_short_root_alias_is_not_mixed_with_resolved_root(self):
        import ctypes
        with tempfile.TemporaryDirectory(prefix="lineage_long_root_") as directory:
            output = ctypes.create_unicode_buffer(32768)
            result = ctypes.windll.kernel32.GetShortPathNameW(str(Path(directory)), output, len(output))
            if not result or result >= len(output) or output.value == str(Path(directory)):
                self.skipTest("此檔案系統未提供不同的 8.3 alias")
            root = Path(output.value)
            formal._write_bytes(root / "keep.csv", b"content", force=False, output_root=root)
            self.assertEqual((Path(directory) / "keep.csv").read_bytes(), b"content")

    def test_unequal_windows_and_sample_lengths_rejected_without_truncation(self):
        for options in ({"unequal_windows": True}, {"unequal_rows": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stage_source(root / "source", channel_entries(**options))
                with self.assertRaisesRegex(formal.FormalDataError, "原始shape=.*丟棄數=0.*UNKNOWN"):
                    formal.materialize(root / "source", root / "output", stages=["2"])
                self.assertFalse((root / "output").exists())

    def test_duplicate_channel_alias_rejected(self):
        entries = channel_entries()
        entries.append(("8screws/T2_Acceleration_X_data.csv", entries[1][1]))
        with ZipFile(io.BytesIO(zip_bytes(entries))) as archive, tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(formal.FormalDataError, "重複通道"):
                formal._convert_condition(archive, "fixture", Path(directory), "T2", "8000rpm",
                                          force=False, selected_conditions=None)

    def test_legal_equal_channels_match_fixed_baseline_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = stage_source(root / "source")
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            old = old_module().materialize(root / "source", root / "old", stages=["2"])
            new = formal.materialize(root / "source", root / "new", stages=["2"])
            old_payload = Path(old["files"][0]["output"]).read_bytes()
            new_payload = (root / "new" / new["files"][0]["output"]).read_bytes()
            self.assertEqual(new_payload, old_payload)
            self.assertEqual(new["archive_sha256"]["2"], checksum)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), checksum)
            record = new["files"][0]
            self.assertTrue(all(shape == [128, 3] for shape in record["channel_shapes"].values()))
            self.assertTrue(all(value == 0 for value in record["discarded_windows"].values()))
            self.assertEqual(record["alignment_status"], "UNKNOWN")
            self.assertEqual(new["schema_version"], "formal_materialization_v3")

    def test_clean_feature_copy_preserves_bytes_and_rejects_nested_attack(self):
        payload = pd.DataFrame(np.ones((3, 105)), columns=formal.FEATURE_NAMES).to_csv(index=False).encode()
        for bad in (False, True):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                entries = [("myfeature/T1/8000rpm/8screws/T1_Group_feature_data_clean.csv", payload)]
                if bad:
                    entries.append(("myfeature/../../outside.csv", payload))
                (source / "階段1.zip").write_bytes(zip_bytes([("myfeature.zip", zip_bytes(entries))]))
                if bad:
                    with self.assertRaises(formal.FormalDataError):
                        formal.materialize(source, root / "output", stages=["1"])
                    self.assertFalse((root / "output").exists())
                else:
                    manifest = formal.materialize(source, root / "output", stages=["1"])
                    self.assertEqual((root / "output" / manifest["files"][0]["output"]).read_bytes(), payload)

    def test_interruption_cleans_temporary_file_and_keeps_existing_target(self):
        for exception in (OSError("磁碟失敗fixture"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / "keep.csv"
                target.write_bytes(b"old")
                with patch.object(formal.os, "replace", side_effect=exception):
                    with self.assertRaises(type(exception)):
                        formal._write_bytes(target, b"new", force=True, output_root=root)
                self.assertEqual(target.read_bytes(), b"old")
                self.assertEqual(list(root.iterdir()), [target])

    def test_invalid_destination_preflight_does_not_write_valid_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "out"
            with self.assertRaises(formal.FormalDataError):
                formal._commit_writes([(root / "good.csv", b"good"), (root.parent / "out_alias/file", b"bad")], root, force=False)
            self.assertFalse(root.exists())

    def test_source_mutation_is_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_source(root / "source")
            with patch.object(formal, "_sha256_file", side_effect=["a" * 64, "b" * 64]):
                with self.assertRaisesRegex(formal.FormalDataError, "SHA 改變"):
                    formal.materialize(root / "source", root / "out", stages=["2"])
            self.assertFalse((root / "out").exists())

    def test_run_and_main_save_public_summary_without_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_source(root / "source")
            with cwd(root):
                formal.run(root / "source", root / "first", stages=["2"])
                self.assertEqual(formal.main(["--source-root", str(root / "source"), "--output-root", str(root / "second"), "--stage", "2"]), 0)
                summaries = list(Path("output/formal_materialization").glob("*/summary.json"))
                self.assertTrue(summaries)
                for path in summaries:
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn(str(root), text)
                    self.assertEqual(json.loads(text)["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
