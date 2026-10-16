"""配對紀錄不改sampler狀態，且保留非唯一列ID。"""
from types import SimpleNamespace
import unittest

import numpy as np

from core.data import CycleSampler
from experiments.navigation_regression import tick_with_row


class PairLedgerTests(unittest.TestCase):
    def test_recording_preserves_sequence_and_duplicate_candidates(self):
        pool = np.array([[1., 2.], [1., 2.], [3., 4.]])
        samplers = [CycleSampler(pool, np.arange(3), np.random.default_rng(42)) for _ in range(2)]
        demo = SimpleNamespace(pools={"source": pool}, samplers={"source": samplers[0]})
        for _ in range(12):
            x, row = tick_with_row(demo, lambda: samplers[0].draw())
            np.testing.assert_array_equal(x, samplers[1].draw())
            self.assertEqual(row["finite_row_indices"], [0, 1] if x[0] == 1 else [2])
            self.assertEqual(len(row["feature_sha256"]), 64)

    def test_failure_restores_original_draw(self):
        pool = np.array([[1., 2.]])
        sampler = CycleSampler(pool, np.array([0]), np.random.default_rng(42))
        demo = SimpleNamespace(pools={"source": pool}, samplers={"source": sampler})
        old = sampler.draw
        with self.assertRaisesRegex(ValueError, "注入失敗"):
            tick_with_row(demo, lambda: (_ for _ in ()).throw(ValueError("注入失敗")))
        self.assertEqual(sampler.draw, old)
