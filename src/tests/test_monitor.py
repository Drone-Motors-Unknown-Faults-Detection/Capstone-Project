"""OpenSetMonitor.summary()：legacy 的合併分布（標籤 -1）不能對不到 splits（#234）。"""
import unittest

import numpy as np

from core.monitor import OpenSetMonitor


def _pools():
    rng = np.random.default_rng(3)
    return {"8screws": rng.normal(0, .3, (80, 6)), "7screws": rng.normal(3, .3, (60, 6))}


class SummaryTests(unittest.TestCase):
    def test_legacy_summary_reports_pooled_counts(self):
        monitor = OpenSetMonitor(_pools(), seed=42, method="legacy")
        monitor.fit_initial()
        monitor.add_class("7screws")
        (row,) = monitor.summary()["classes"]
        self.assertEqual(row["config"], "pooled_known")
        self.assertEqual(row["label"], -1)
        expected = sum(len(monitor.splits[c].train) for c in monitor.known)
        self.assertEqual(row["n_train"], expected)

    def test_classwise_summary_keeps_one_row_per_class(self):
        monitor = OpenSetMonitor(_pools(), seed=42)
        monitor.fit_initial()
        monitor.add_class("7screws")
        rows = monitor.summary()["classes"]
        self.assertEqual([r["config"] for r in rows], ["8screws", "7screws"])
        self.assertEqual(rows[1]["n_train"], len(monitor.splits["7screws"].train))


if __name__ == "__main__":
    unittest.main()
