import unittest

from experiments.health.evaluation import (
    assert_group_disjoint,
    assert_temporal_order,
    evaluate_event_metrics,
    evaluate_open_set,
    evaluate_results,
    evaluate_trajectory,
)
from experiments.health.schema import HealthMonitoringResult


def _result(score: float, alarm: str = "normal") -> HealthMonitoringResult:
    health = max(0.0, min(1.0, 1.0 - score / 3.0))
    return HealthMonitoringResult(
        is_fault=score > 1.0,
        health_index=health,
        degradation_score=1.0 - health,
        severity_stage="healthy" if health >= 0.8 else "degraded",
        trend="worsening" if score > 1.0 else "stable",
        degradation_rate=0.01 if score > 1.0 else 0.0,
        fault_type="unknown" if score > 1.0 else "uncertain",
        fault_type_confidence=None,
        openset_method="mahalanobis",
        openset_score=score,
        is_unknown_fault=score > 1.0,
        prediction_confidence=0.5,
        uncertainty=0.5,
        estimated_rul=None,
        rul_available=False,
        timestamp=None,
        condition="T1/8000rpm",
        data_quality="valid",
        alarm_state=alarm,
    )


class HealthEvaluationTests(unittest.TestCase):
    def test_open_set_metrics_use_held_out_labels(self):
        metrics = evaluate_open_set(
            [_result(0.2), _result(1.2), _result(2.0)],
            [0, 1, 1],
        )
        self.assertEqual(metrics["score_threshold"], 1.0)
        self.assertGreater(metrics["unknown_recall"], 0.0)

    def test_missing_event_labels_are_explicitly_unavailable(self):
        result = evaluate_results([_result(0.2), _result(1.2)])
        self.assertEqual(result["open_set"]["status"], "unavailable_no_unknown_labels")
        self.assertEqual(result["events"]["event_metrics_status"], "unavailable_no_event_labels")
        self.assertEqual(result["trajectory"]["ground_truth_status"], "descriptive_only_no_event_labels")

    def test_event_rate_requires_aligned_timestamps(self):
        metrics = evaluate_event_metrics(
            [_result(0.2, "warning"), _result(0.2, "normal")],
            [0, 1],
            [0, 3600],
        )
        self.assertEqual(metrics["event_metrics_status"], "computed")
        self.assertEqual(metrics["false_alarms_per_hour"], 1.0)

    def test_group_and_time_leakage_guards(self):
        assert_group_disjoint(["m1"], ["m2"])
        assert_temporal_order([0, 1], [2, 3])
        with self.assertRaises(ValueError):
            assert_group_disjoint(["m1"], ["m1"])
        with self.assertRaises(ValueError):
            assert_temporal_order([0, 3], [2, 4])

    def test_trajectory_metrics_report_alarm_transitions(self):
        metrics = evaluate_trajectory([_result(0.2), _result(1.2, "warning"), _result(1.4, "critical")])
        self.assertEqual(metrics["alarm_transition_count"], 2)
        self.assertGreater(metrics["alarm_fraction"], 0.0)
