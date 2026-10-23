"""core.data.PoolCache：以內容 SHA 為鍵的 load_pools 快取。"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core import data as data_module
from core.data import PoolCache, load_pools
from core.feature_schema import SCHEMAS, FeatureSchemaError
from experiments.exp6_formal_benchmark import _run_benchmark


def write_condition(root: Path, seed: int = 7) -> Path:
    dataset = root / "Step-1" / "myfeature" / "T1" / "8000rpm"
    rng = np.random.default_rng(seed)
    for config, center in (("8screws", 0.0), ("7screws", 5.0)):
        target = dataset / config
        target.mkdir(parents=True)
        frame = pd.DataFrame(rng.normal(center, 0.1, size=(30, 105)).round(6), columns=SCHEMAS["historical_clean_v1"])
        frame.to_csv(target / "T1_Group_feature_data_clean.csv", index=False)
    return dataset


class PoolCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = write_condition(self.root / "a")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _csv(self) -> Path:
        return self.dataset / "8screws" / "T1_Group_feature_data_clean.csv"

    def test_hit_returns_same_pools_and_audit_as_load_pools(self) -> None:
        cache = PoolCache()
        expected_audit, first_audit, second_audit = {}, {}, {}
        expected = load_pools(self.dataset, audit=expected_audit)
        first = cache.load(self.dataset, audit=first_audit)
        second = cache.load(self.dataset, audit=second_audit)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        for pools in (first, second):
            self.assertEqual(sorted(pools), sorted(expected))
            for name in expected:
                np.testing.assert_array_equal(pools[name], expected[name])
        self.assertEqual(first_audit, expected_audit)
        self.assertEqual(second_audit, expected_audit)

    def test_alias_directory_is_keyed_canonically_through_cache(self) -> None:
        """PoolCache 回傳的 key 與 audit 跟 load_pools 一樣是正規名（#271 C33）。"""
        dataset = write_condition(self.root / "b", seed=11)
        (dataset / "7screws").rename(dataset / "1screw")
        cache = PoolCache()
        audit = {}
        pools = cache.load(dataset, audit=audit)
        self.assertEqual(sorted(pools), ["1screws", "8screws"])
        self.assertEqual(audit["configuration_dirs"]["1screws"], "1screw")
        self.assertEqual(sorted(cache.load(dataset)), ["1screws", "8screws"])

    def test_returned_arrays_do_not_alias_the_cache(self) -> None:
        cache = PoolCache()
        first = cache.load(self.dataset)
        original = first["8screws"].copy()
        first["8screws"][:] = 0.0
        np.testing.assert_array_equal(cache.load(self.dataset)["8screws"], original)

    def test_same_size_and_mtime_but_different_bytes_misses(self) -> None:
        cache = PoolCache()
        before = cache.load(self.dataset)["8screws"]
        path = self._csv()
        stat = path.stat()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        first_cell, rest = lines[1].split(",", 1)
        digit = "1" if first_cell[-1] != "1" else "2"
        lines[1] = first_cell[:-1] + digit + "," + rest
        path.write_text("".join(lines), encoding="utf-8")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual((path.stat().st_size, path.stat().st_mtime_ns), (stat.st_size, stat.st_mtime_ns))
        after = cache.load(self.dataset)["8screws"]
        self.assertEqual((cache.hits, cache.misses), (0, 2))
        self.assertNotEqual(before[0, 0], after[0, 0])

    def test_copy_with_same_content_hits_and_matches(self) -> None:
        cache = PoolCache()
        copy_root = self.root / "copy" / "8000rpm"
        shutil.copytree(self.dataset, copy_root)
        original = cache.load(self.dataset)
        copied = cache.load(copy_root)
        self.assertEqual(cache.hits, 1)
        np.testing.assert_array_equal(original["7screws"], copied["7screws"])

    def test_require_complete_is_part_of_the_key(self) -> None:
        cache = PoolCache()
        cache.load(self.dataset)
        with self.assertRaises(FeatureSchemaError):
            cache.load(self.dataset, require_complete=True)
        self.assertEqual(cache.misses, 2)

    def test_failures_are_not_cached(self) -> None:
        cache = PoolCache()
        path = self._csv()
        good = path.read_bytes()
        path.write_text("bad,header\n1,2\n", encoding="utf-8")
        for _ in range(2):
            with self.assertRaises(FeatureSchemaError):
                cache.load(self.dataset)
        path.write_bytes(good)
        self.assertIn("8screws", cache.load(self.dataset))
        self.assertEqual((cache.hits, cache.misses), (0, 3))

    def test_content_changed_during_load_is_not_stored(self) -> None:
        cache = PoolCache()
        real = data_module.load_pools

        def load_then_modify(*args, **kwargs):
            pools = real(*args, **kwargs)
            frame = pd.read_csv(self._csv())
            frame.iloc[0, 0] = 99.0
            frame.to_csv(self._csv(), index=False)
            return pools

        with patch.object(data_module, "load_pools", side_effect=load_then_modify):
            cache.load(self.dataset)
        self.assertEqual(cache.load(self.dataset)["8screws"][0, 0], 99.0)
        self.assertEqual(cache.hits, 0)


class CachedBenchmarkTests(unittest.TestCase):
    def test_formal_benchmark_rows_are_identical_with_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_condition(root)
            cache = PoolCache()
            for method in ("mahalanobis", "knn"):
                with self.subTest(method=method):
                    plain = _run_benchmark(root, seed=42, openset_method=method, require_nine=False)
                    for _ in range(2):
                        cached = _run_benchmark(root, seed=42, openset_method=method, require_nine=False,
                                                pool_cache=cache)
                        for a, b in zip(plain["rows"], cached["rows"]):
                            keys = set(a) - {"inference_seconds", "peak_memory_mb"}
                            self.assertEqual({k: a[k] for k in keys}, {k: b[k] for k in keys})
            self.assertEqual((cache.misses, cache.hits), (1, 3))


if __name__ == "__main__":
    unittest.main()
