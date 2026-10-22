"""實驗二：密度階梯開關（adaptive_density）只改 HDBSCAN 參數，其他流程不變。"""
import unittest
from unittest.mock import patch

import numpy as np

from experiments import exp2_scale_growth as exp2


def _pools() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    return {
        "8screws": rng.normal(0.0, 0.3, (120, 6)),
        "7screws": rng.normal(6.0, 0.3, (80, 6)),
    }


def _attempted_params(adaptive_density: bool) -> tuple[list[tuple[int, int]], bool]:
    session = exp2.ScaleGrowthSession(_pools(), seed=42, adaptive_density=adaptive_density)
    session.quarantine_X = list(np.random.default_rng(9).normal(6.0, 0.3, (120, 6)))
    session.quarantine_truth = ["7screws"] * 120
    session.quarantine_t = list(range(120))
    calls = []

    class NoClusterHDBSCAN:
        """每一階都只回雜訊 -1，階梯會一路試到底。"""

        def __init__(self, min_cluster_size, min_samples):
            calls.append((min_cluster_size, min_samples))

        def fit_predict(self, X):
            return np.full(len(X), -1)

    with patch.object(exp2.hdbscan, "HDBSCAN", NoClusterHDBSCAN):
        found = session._try_cluster()
    return calls, found


class DensityLadderTests(unittest.TestCase):
    def test_adaptive_density_tries_full_ladder(self):
        calls, found = _attempted_params(True)
        self.assertEqual(calls, [(25, 3), (15, 3), (10, 2)])
        self.assertFalse(found)

    def test_fixed_density_only_tries_strictest_step(self):
        calls, found = _attempted_params(False)
        self.assertEqual(calls, [(25, 3)])
        self.assertFalse(found)

    def test_run_records_density_setting(self):
        for adaptive in (True, False):
            with self.subTest(adaptive=adaptive):
                result = exp2.run(_pools(), seed=42, max_ticks_per_stage=5, adaptive_density=adaptive)
                self.assertIs(result["adaptive_density"], adaptive)

    def test_cli_flag_turns_ladder_off(self):
        captured = {}

        def fake_run(pools, **kwargs):
            captured.update(kwargs)
            return {"stages": [], "final_known": [], "model": {}}

        with patch.object(exp2, "resolve_dataset", return_value=({"motor": "T1", "rpm": "8000rpm"}, _pools())), \
                patch.object(exp2, "run", side_effect=fake_run), \
                patch("sys.argv", ["exp2", "--fixed-density"]):
            exp2.main()
        self.assertIs(captured["adaptive_density"], False)


if __name__ == "__main__":
    unittest.main()
