"""實驗五 (b)：三種策略的擬合資料都不得含評估 holdout（鐵則 6）。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.data import HEALTHY, make_split
from core.runner import save_json
from experiments import exp5_cross_condition as exp5


def _pools() -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(3)
    pools = {}
    for i, key in enumerate(["T1/8000", "T2/8000", "T1/10000"]):
        # 每列第一欄放唯一編號，之後可以回推是哪一列進了擬合
        healthy = rng.normal(0, 0.3, (60, 6))
        healthy[:, 0] = np.arange(60) + 1000 * i
        fault = rng.normal(3, 0.3, (30, 6))
        fault[:, 0] = -1 - np.arange(30) - 1000 * i
        pools[key] = {HEALTHY: healthy, "7screws": fault}
    return pools


class StrategyHoldoutTests(unittest.TestCase):
    def test_eval_holdout_rows_never_enter_any_strategy_fit(self):
        pools = _pools()
        seed = 42
        fitted: list[np.ndarray] = []
        original = exp5._fit_healthy_monitor

        def recording_fit(fit_pools, *args, **kwargs):
            fitted.append(np.vstack(list(fit_pools.values())))
            return original(fit_pools, *args, **kwargs)

        with patch.object(exp5, "_fit_healthy_monitor", side_effect=recording_fit):
            result = exp5.run_strategy(pools, seed=seed)

        # 照 run_strategy 的順序重建同一份 outer split
        rng = np.random.default_rng(seed + 7)
        holdout_ids = set()
        for key in sorted(pools):
            split = make_split(len(pools[key][HEALTHY]), rng)
            holdout_ids |= set(pools[key][HEALTHY][split.holdout, 0].tolist())

        # per_condition 3 + global 1 + per_rpm 2
        self.assertEqual(len(fitted), 6)
        for rows in fitted:
            self.assertFalse(holdout_ids & set(rows[:, 0].tolist()))
        self.assertEqual(set(result["strategies"]["per_condition"]), set(pools))


class EmptyAggregateTests(unittest.TestCase):
    """少工況時空集合記 None、沒有工況直接報錯，summary.json 不會出現 NaN（#271）。"""

    def test_single_dataset_matrix_has_none_for_offdiagonal_stats(self):
        pools = {"T1/8000": _pools()["T1/8000"]}
        result = exp5.run_matrix(pools, seed=42)
        summary = result["summary"]
        self.assertIsNotNone(summary["diag_auroc_mean"])
        for key in ("offdiag_auroc_mean", "offdiag_auroc_min",
                    "same_rpm_cross_motor_auroc_mean", "same_motor_cross_rpm_auroc_mean"):
            self.assertIsNone(summary[key], key)
        with tempfile.TemporaryDirectory() as directory:
            save_json(Path(directory) / "summary.json", result)

    def test_two_datasets_different_rpm_and_motor_keep_finite_means(self):
        pools = {k: v for k, v in _pools().items() if k in ("T1/8000", "T2/8000")}
        summary = exp5.run_matrix(pools, seed=42)["summary"]
        self.assertIsNotNone(summary["same_rpm_cross_motor_auroc_mean"])
        self.assertIsNone(summary["same_motor_cross_rpm_auroc_mean"])

    def test_no_datasets_is_an_error_not_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                exp5.run(directory, seed=42)


if __name__ == "__main__":
    unittest.main()
