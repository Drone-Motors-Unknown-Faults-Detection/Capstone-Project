"""實驗二：密度階梯開關、adopt() 與隔離區規則（#268：清理套進入規則、不收重複列、平手決定論、min_cluster_size 驗證、待確認不加列）。"""
import unittest
from unittest.mock import patch

import numpy as np

from experiments import exp2_scale_growth as exp2


def _pools() -> dict[str, np.ndarray]:
    """7screws 300 列：隔離區不收重複列後，重播只剩 240 筆不同的列，要夠多才聚得出候選。"""
    rng = np.random.default_rng(5)
    return {
        "8screws": rng.normal(0.0, 0.3, (120, 6)),
        "7screws": rng.normal(6.0, 0.3, (300, 6)),
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

        with patch("core.runner._dataset_info", return_value={"motor": "T1", "rpm": "8000rpm", "path": "unused"}), \
                patch("core.runner.load_pool_source", return_value=(_pools(), None)), \
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
        self.assertEqual(len(parts["eval"]), 60)
        self.assertEqual(len(parts["stream"]), 240)
        self.assertEqual(set(parts["eval"]) & set(parts["stream"]), set())
        self.assertEqual(set(parts["eval"]) | set(parts["stream"]), set(range(300)))

    def test_split_is_deterministic(self):
        a = exp2.split_stream_eval(_pools(), ["7screws"], seed=42)["7screws"]["eval"]
        b = exp2.split_stream_eval(_pools(), ["7screws"], seed=42)["7screws"]["eval"]
        np.testing.assert_array_equal(a, b)

    def test_run_reports_discovery_and_external_eval(self):
        result = exp2.run(_pools(), seed=42, max_ticks_per_stage=300)
        stage = result["stages"][0]
        self.assertTrue(stage["discovered"])
        self.assertTrue(stage["matched_is_config"])
        self.assertEqual(stage["eval_n"], 60)
        self.assertGreater(stage["own_holdout_n"], 0)
        self.assertTrue(result["discovery"]["7screws"]["discovered"])
        self.assertEqual(result["discovery"]["7screws"]["delay"], stage["samples_to_candidate"])
        self.assertEqual(result["final_eval"]["7screws"]["n"], 60)
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


class MatchedRateAndHealthyMixTests(unittest.TestCase):
    """#243 判進對應新類比例；#240 健康混入。"""

    def test_matched_rate_bounded_by_known_rate(self):
        result = exp2.run(_pools(), seed=42, max_ticks_per_stage=300)
        ev = result["final_eval"]["7screws"]
        self.assertLessEqual(ev["matched_rate"], ev["known_rate"])
        self.assertEqual(result["stages"][0]["eval_matched_rate"], ev["matched_rate"])

    def test_breakdown_counts_only_classes_matched_to_config(self):
        labels = ["discovered-1", "discovered-2", None, "8screws"]
        ev = exp2._eval_breakdown(labels, "7screws", {"discovered-1": "7screws", "discovered-2": "6screws"})
        self.assertEqual(ev["known_rate"], 0.5)
        self.assertEqual(ev["matched_rate"], 0.25)

    def test_zero_mix_matches_default(self):
        a = exp2.run(_pools(), seed=42, max_ticks_per_stage=300)
        b = exp2.run(_pools(), seed=42, max_ticks_per_stage=300, healthy_mix=0.0)
        self.assertEqual(a["stages"], b["stages"])
        self.assertEqual(a["stages"][0]["healthy_seen"], 0)

    def test_healthy_mix_feeds_healthy_windows_without_quarantine(self):
        result = exp2.run(_pools(), seed=42, max_ticks_per_stage=300, healthy_mix=0.3)
        stage = result["stages"][0]
        self.assertGreater(stage["healthy_seen"], 0)
        self.assertEqual(stage["healthy_quarantined"], 0)
        self.assertEqual(result["healthy_mix"], 0.3)

    def test_sequence_alias_is_canonicalised_and_unknown_config_rejected(self):
        """--sequence 1screw 對到池 key 1screws；不在池裡或健康配置直接 ValueError（#285 R2-03）。"""
        pools = {**_pools(), "1screws": np.random.default_rng(6).normal(9.0, 0.3, (60, 6))}
        alias = exp2.run(pools, sequence=["1screw"], seed=42, max_ticks_per_stage=3)
        canonical = exp2.run(pools, sequence=["1screws"], seed=42, max_ticks_per_stage=3)
        self.assertEqual(alias["stages"], canonical["stages"])
        for sequence in (["9screws"], ["8screws"]):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                exp2.run(pools, sequence=sequence, seed=42, max_ticks_per_stage=3)

    def test_mix_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            exp2.run(_pools(), seed=42, max_ticks_per_stage=5, healthy_mix=1.5)


if __name__ == "__main__":
    unittest.main()


class QuarantineRuleTests(unittest.TestCase):
    """#268：隔離區清理與進入規則一致、重複列不入隔離區、待確認期間不加列。"""

    def test_prune_drops_rows_at_or_below_quarantine_margin(self):
        rows = np.random.default_rng(21).normal(6.0, 0.3, (40, 6))
        session = _session_with_quarantine(rows, "7screws", cluster=30)
        out = session.adopt()
        self.assertEqual(out["action"], "learned")
        if session.quarantine_X:
            scores, labels = session.monitor.score_and_classify(np.vstack(session.quarantine_X))
            self.assertTrue(all(label is None for label in labels))
            self.assertTrue(np.all(scores > session.quarantine_margin))

    def test_prune_uses_margin_not_detection_line(self):
        session = exp2.ScaleGrowthSession(_pools(), seed=42)
        rows = np.random.default_rng(22).normal(6.0, 0.3, (12, 6))
        session.quarantine_X, session.quarantine_truth, session.quarantine_t = list(rows), ["7screws"] * 12, list(range(12))
        scores = np.array([0.5, 1.5, 2.0, 2.5, 3.0, 0.9, 1.1, 1.9, 2.01, 4.0, 1.0, 2.2])
        labels = [None] * 12
        labels[0] = "8screws"
        with patch.object(session.monitor, "score_and_classify", return_value=(scores, labels)):
            pruned = session._prune_explained()
        self.assertEqual(pruned, 7)
        self.assertEqual(session.quarantine_t, [3, 4, 8, 9, 11])

    def test_exact_duplicate_rows_are_not_stored_twice(self):
        session = exp2.ScaleGrowthSession(_pools(), seed=42)
        row = np.full(6, 6.0)
        first = session.process(row, "7screws")
        second = session.process(row.copy(), "7screws")
        self.assertEqual(first["quarantine"], 1)
        self.assertEqual(second["quarantine"], 1)
        self.assertEqual(session.quarantine_duplicates, 1)
        self.assertEqual(session.quarantine_truth, ["7screws"])
        session.process(row + 0.01, "7screws")
        self.assertEqual(len(session.quarantine_X), 2)

    def test_pending_candidate_blocks_new_quarantine_rows(self):
        session = exp2.ScaleGrowthSession(_pools(), seed=42)
        session.candidate = {"indices": [], "size": 0, "majority": "7screws", "majority_display": "7screws",
                             "purity": 1.0}
        out = session.process(np.full(6, 6.0), "7screws")
        self.assertIsNone(out["label"])
        self.assertEqual(out["quarantine"], 0)
        self.assertEqual(session.quarantine_X, [])

    def test_run_reports_duplicate_count_and_learned_class_has_no_duplicate_rows(self):
        result = exp2.run(_pools(), seed=42, max_ticks_per_stage=300)
        self.assertEqual(result["quarantine_duplicates"], 0)  # 240 筆不同的列在 225 tick 內成群，還沒循環到重抽
        rng = np.random.default_rng(5)
        small = {"8screws": rng.normal(0.0, 0.3, (120, 6)), "7screws": rng.normal(6.0, 0.3, (40, 6))}
        session = exp2.ScaleGrowthSession(small, seed=42)
        for _ in range(3):
            for row in small["7screws"]:
                session.process(row, "7screws")
        self.assertEqual(len(session.quarantine_X), 40)
        self.assertEqual(session.quarantine_duplicates, 80)
        self.assertEqual(len(np.unique(np.vstack(session.quarantine_X), axis=0)), 40)


class MajorityAndValidationTests(unittest.TestCase):
    def test_majority_tie_breaks_by_name(self):
        self.assertEqual(exp2._majority(["b", "a", "b", "a"]), "a")
        self.assertEqual(exp2._majority(["b", "a", "b"]), "b")
        ev = exp2._eval_breakdown(["discovered-2", "discovered-1"], "7screws", {})
        self.assertEqual(ev["top_class"], "discovered-1")

    def test_min_cluster_size_below_five_is_rejected(self):
        with self.assertRaises(ValueError):
            exp2.ScaleGrowthSession(_pools(), seed=42, min_cluster_size=4)
        exp2.ScaleGrowthSession(_pools(), seed=42, min_cluster_size=5)


class ResultProvenanceTests(unittest.TestCase):
    """#273：結果欄位取自 monitor.summary()，別名呼叫與 model 區塊一致。"""

    def test_alias_openset_method_is_canonical_in_result(self):
        result = exp2.run(_pools(), seed=42, max_ticks_per_stage=5, openset_method="k-nn", knn_neighbors=3)
        self.assertEqual(result["openset_method"], "knn")
        self.assertEqual(result["knn_neighbors"], 3)
        self.assertEqual(result["model"]["openset_method"], "knn")
        default = exp2.run(_pools(), seed=42, max_ticks_per_stage=5)
        self.assertEqual(default["openset_method"], "mahalanobis")
        self.assertIsNone(default["knn_neighbors"])
        self.assertEqual(default["calibration_source"], "known-only 20% calibration split")
        self.assertEqual(default["split"], {"train": 0.6, "calibration": 0.2, "holdout": 0.2})
