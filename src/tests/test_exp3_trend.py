"""實驗三趨勢（exp3_trend）：重複次數必須 ≥ 1，否則 alarm_rate 除以零（#285 R2-05）。"""
import unittest

import numpy as np

from experiments.exp3_trend import iter_run, run


def _pools() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {"8screws": rng.normal(0.0, 0.3, (120, 6)),
            **{c: rng.normal(shift, 0.3, (60, 6)) for c, shift in (("7screws", 3.0), ("6screws", 6.0),
                                                                   ("5screws", 9.0), ("4screws", 12.0))}}


class TrialCountTests(unittest.TestCase):
    def test_zero_trials_rejected_before_fitting(self):
        for n_trials in (0, -1):
            with self.subTest(n_trials=n_trials), self.assertRaisesRegex(ValueError, "n_trials"):
                run(_pools(), n_trials=n_trials, seed=1)
            with self.subTest(n_trials=n_trials, entry="iter_run"), self.assertRaisesRegex(ValueError, "n_trials"):
                next(iter_run(_pools(), n_trials=n_trials, seed=1))

    def test_one_trial_runs(self):
        result = run(_pools(), n_trials=1, seed=1)
        self.assertEqual({row["trials"] for row in result["summary"].values()}, {1})


if __name__ == "__main__":
    unittest.main()
