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


def _session_with_quarantine(rows: np.ndarray, truth: str, cluster: int) -> exp2.ScaleGrowthSession:
    """隔離區放入 rows，前 cluster 筆設成候選群。"""
    session = exp2.ScaleGrowthSession(_pools(), seed=42)
    session.quarantine_X = list(rows)
    session.quarantine_truth = [truth] * len(rows)
    session.quarantine_t = list(range(len(rows)))
    session.candidate = {
        "indices": list(range(cluster)), "size": cluster, "majority": truth,
        "majority_display": truth, "purity": 1.0,
    }
    return session


class AdoptTests(unittest.TestCase):
    def test_learned_prunes_residual_now_explained(self):
        rows = np.random.default_rng(11).normal(6.0, 0.3, (40, 6))
        session = _session_with_quarantine(rows, "7screws", cluster=30)
        out = session.adopt()
        self.assertEqual(out["action"], "learned")
        self.assertIn("discovered-1", session.monitor.known)
        # 10 筆殘餘：被新類接受的移出，仍超線的留下（小校準集的 95% 門檻會留幾筆）
        self.assertGreater(out["pruned"], 0)
        self.assertEqual(out["pruned"] + len(session.quarantine_X), 10)
        if session.quarantine_X:
            left = session.monitor.classify(np.vstack(session.quarantine_X))
            self.assertTrue(all(label is None for label in left))

    def test_pruning_keeps_rows_new_model_rejects(self):
        rng = np.random.default_rng(12)
        rows = np.vstack([rng.normal(6.0, 0.3, (30, 6)), rng.normal(-6.0, 0.3, (8, 6))])
        session = _session_with_quarantine(rows, "7screws", cluster=30)
        out = session.adopt()
        self.assertEqual(out["pruned"], 0)
        self.assertEqual(len(session.quarantine_X), 8)

    def test_cluster_inside_known_boundary_is_rejected(self):
        rows = np.random.default_rng(13).normal(0.0, 0.3, (30, 6))
        session = _session_with_quarantine(rows, "8screws", cluster=30)
        out = session.adopt()
        self.assertEqual(out["action"], "rejected_boundary")
        self.assertLessEqual(out["median_score"], out["known_cal_max"])
        self.assertEqual(list(session.monitor.known), ["8screws"])
        self.assertEqual(session.quarantine_X, [])


class InsufficientTests(unittest.TestCase):
    def test_insufficient_cluster_leaves_quarantine(self):
        rows = np.random.default_rng(14).normal(6.0, 0.3, (10, 6))
        session = _session_with_quarantine(rows, "7screws", cluster=3)
        out = session.adopt()
        self.assertEqual(out["action"], "insufficient")
        self.assertEqual(len(session.quarantine_X), 7)
        self.assertEqual(list(session.monitor.known), ["8screws"])


class EvalSplitTests(unittest.TestCase):
    def test_stream_and_eval_are_disjoint_and_cover_pool(self):
        parts = exp2.split_stream_eval(_pools(), ["7screws"], seed=42)["7screws"]
        self.assertEqual(len(parts["eval"]), 16)
        self.assertEqual(len(parts["stream"]), 64)
        self.assertEqual(set(parts["eval"]) & set(parts["stream"]), set())
        self.assertEqual(set(parts["eval"]) | set(parts["stream"]), set(range(80)))

    def test_split_is_deterministic(self):
        a = exp2.split_stream_eval(_pools(), ["7screws"], seed=42)["7screws"]["eval"]
        b = exp2.split_stream_eval(_pools(), ["7screws"], seed=42)["7screws"]["eval"]
        np.testing.assert_array_equal(a, b)

    def test_run_reports_discovery_and_external_eval(self):
        result = exp2.run(_pools(), seed=42, max_ticks_per_stage=300)
        stage = result["stages"][0]
        self.assertTrue(stage["discovered"])
        self.assertTrue(stage["matched_is_config"])
        self.assertEqual(stage["eval_n"], 16)
        self.assertGreater(stage["own_holdout_n"], 0)
        self.assertTrue(result["discovery"]["7screws"]["discovered"])
        self.assertEqual(result["discovery"]["7screws"]["delay"], stage["samples_to_candidate"])
        self.assertEqual(result["final_eval"]["7screws"]["n"], 16)
        self.assertAlmostEqual(result["final_eval"]["7screws"]["known_rate"], stage["eval_known_rate"])
        self.assertEqual(result["rejections"], [])

    def test_run_records_rejected_candidates(self):
        real_adopt = exp2.ScaleGrowthSession.adopt
        calls = []

        def reject_first(session):
            if not calls:
                calls.append(1)
                session.candidate = None
                return {"action": "rejected_boundary", "size": 30, "matched": "7screws",
                        "median_score": 0.9, "known_cal_max": 1.2}
            return real_adopt(session)

        with patch.object(exp2.ScaleGrowthSession, "adopt", reject_first):
            result = exp2.run(_pools(), seed=42, max_ticks_per_stage=300)
        self.assertEqual(len(result["rejections"]), 1)
        self.assertEqual(result["rejections"][0]["action"], "rejected_boundary")
        self.assertEqual(result["rejections"][0]["config"], "7screws")


if __name__ == "__main__":
    unittest.main()
