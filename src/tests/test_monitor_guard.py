"""未擬合防護與固定舊版逐欄回歸。"""
import hashlib
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from core.monitor import OpenSetMonitor

BASE = "4f783c676849a27185cd28d2410b4e76639b7357"
ATOL = RTOL = 1e-12
METHODS = ("mahalanobis", "knn")


def pools():
    rng = np.random.default_rng(7)
    return {"8screws": rng.normal(0, .3, (80, 105)),
            "7screws": rng.normal(3, .3, (80, 105))}


def paired_baseline():
    source = subprocess.check_output(["git", "show", f"{BASE}:core/monitor.py"])
    namespace = {"__name__": "fixed_baseline_monitor"}
    exec(compile(source, f"{BASE}:core/monitor.py", "exec"), namespace)
    old_class = namespace["OpenSetMonitor"]
    data = pools()
    sample_ids = [f"synthetic/{c}/{i}" for c in data for i in range(8)]
    query = np.vstack([x[:8] for x in data.values()])
    records = []
    for method in METHODS:
        for seed in (42, 123):
            old = old_class(data, seed=seed, openset_method=method)
            new = OpenSetMonitor(data, seed=seed, openset_method=method)
            old.fit_initial()
            new.fit_initial()
            for stage in ("healthy", "confirmed_fault"):
                if stage == "confirmed_fault":
                    old.add_class("7screws")
                    new.add_class("7screws")
                a, b = old.score(query), new.score(query)
                x, y = old.project(query), new.project(query)
                np.testing.assert_allclose(a, b, atol=ATOL, rtol=RTOL)
                np.testing.assert_allclose(x, y, atol=ATOL, rtol=RTOL)
                assert old.classify(query) == new.classify(query)
                assert old.summary() == new.summary()
                for config in old.splits:
                    for role in ("train", "cal", "holdout"):
                        np.testing.assert_array_equal(getattr(old.splits[config], role),
                                                      getattr(new.splits[config], role))
                records.append({"method": method, "seed": seed, "stage": stage,
                                "sample_ids": sample_ids, "score_before": a.tolist(), "score_after": b.tolist(),
                                "class_before": old.classify(query), "class_after": new.classify(query),
                                "pca_before": x.tolist(), "pca_after": y.tolist(),
                                "max_score_diff": float(np.max(np.abs(a-b))),
                                "max_pca_diff": float(np.max(np.abs(x-y)))})
    return {"baseline": BASE, "source_sha256": hashlib.sha256(source).hexdigest(),
            "fixture_seed": 7, "feature_dim": 105, "atol": ATOL, "rtol": RTOL,
            "records": records, "claim": "工程等價；不計準確率改善"}


class MonitorGuardTests(unittest.TestCase):
    def assert_guard(self, monitor):
        for name in ("score", "classify", "project", "summary"):
            with self.subTest(method=monitor.openset_method, entry=name):
                with self.assertRaisesRegex(RuntimeError, r"fit_initial\(\)"):
                    getattr(monitor, name)(*[np.zeros(105)] if name != "summary" else [])

    def test_score_before_fit(self):
        for method in METHODS:
            with self.subTest(method=method), self.assertRaisesRegex(RuntimeError, r"fit_initial\(\)"):
                OpenSetMonitor(pools(), openset_method=method).score(np.zeros(105))

    def test_classify_before_fit(self):
        for method in METHODS:
            with self.subTest(method=method), self.assertRaisesRegex(RuntimeError, r"fit_initial\(\)"):
                OpenSetMonitor(pools(), openset_method=method).classify(np.zeros(105))

    def test_project_before_fit(self):
        for method in METHODS:
            with self.subTest(method=method), self.assertRaisesRegex(RuntimeError, r"fit_initial\(\)"):
                OpenSetMonitor(pools(), openset_method=method).project(np.zeros(105))

    def test_summary_before_fit(self):
        for method in METHODS:
            with self.subTest(method=method), self.assertRaisesRegex(RuntimeError, r"fit_initial\(\)"):
                OpenSetMonitor(pools(), openset_method=method).summary()

    def test_successful_fit_matches_fixed_baseline(self):
        self.assertEqual(len(paired_baseline()["records"]), 8)

    def test_failed_initial_fit(self):
        for method in METHODS:
            monitor = OpenSetMonitor(pools(), openset_method=method)
            with patch("core.monitor.create_openset_detector", side_effect=RuntimeError("注入擬合失敗")):
                with self.assertRaisesRegex(RuntimeError, "注入擬合失敗"):
                    monitor.fit_initial()
            self.assert_guard(monitor)
            monitor.fit_initial()
            self.assertEqual(monitor.project(np.zeros(105)).shape, (1, 2))

    def test_failed_reinitialization_blocks_stale_inference(self):
        for method in METHODS:
            monitor = OpenSetMonitor(pools(), openset_method=method)
            monitor.fit_initial()
            with self.assertRaises(KeyError):
                monitor.fit_initial("missing")
            self.assert_guard(monitor)

    def test_failed_add_refit_blocks_partial_model(self):
        for method in METHODS:
            monitor = OpenSetMonitor(pools(), openset_method=method)
            monitor.fit_initial()
            with patch("core.monitor.create_openset_detector", side_effect=RuntimeError("注入擬合失敗")):
                with self.assertRaisesRegex(RuntimeError, "注入擬合失敗"):
                    monitor.add_class("7screws")
            self.assert_guard(monitor)

    def test_invalid_registration_keeps_valid_model(self):
        for method in METHODS:
            monitor = OpenSetMonitor(pools(), openset_method=method)
            monitor.fit_initial()
            before = monitor.score(np.zeros(105))
            with self.assertRaises(KeyError):
                monitor.add_class("missing")
            with self.assertRaises(ValueError):
                monitor.add_class("8screws")
            np.testing.assert_array_equal(before, monitor.score(np.zeros(105)))

    def test_holdout_and_configuration_reads_keep_existing_contract(self):
        monitor = OpenSetMonitor(pools(), seed=123)
        self.assertEqual(monitor.seed, 123)
        self.assertEqual(monitor.method, "ledoit_wolf")
        with self.assertRaises(KeyError):
            monitor.holdout("8screws")
        monitor.fit_initial()
        np.testing.assert_array_equal(monitor.holdout("8screws"),
                                      monitor.pools["8screws"][monitor.splits["8screws"].holdout])

    def test_explicit_add_class_keeps_existing_fit_semantics(self):
        for method in METHODS:
            monitor = OpenSetMonitor(pools(), openset_method=method)
            monitor.add_class("8screws")
            self.assertEqual(monitor.summary()["n_known"], 1)
