"""Leakage-aware metrics for health and Open Set monitoring outputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiments.health.schema import HealthMonitoringResult


def _binary(values: Iterable[bool | int], *, name: str) -> np.ndarray:
    array = np.asarray([int(bool(value)) for value in values], dtype=int)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return array


def evaluate_open_set(
    results: Sequence[HealthMonitoringResult],
    unknown_labels: Iterable[bool | int],
) -> dict:
    """Evaluate held-out unknown labels after all fitting is complete."""

    if not results:
        raise ValueError("results must contain at least one item")
    labels = _binary(unknown_labels, name="unknown_labels")
    if len(labels) != len(results):
        raise ValueError("unknown_labels must align one-to-one with results")
    scores = np.asarray([item.openset_score for item in results], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("results contain non-finite Open Set scores")
    predictions = (scores > 1.0).astype(int)
    output = {
        "n_windows": int(len(results)),
        "open_set_accuracy": float(accuracy_score(labels, predictions)),
        "unknown_precision": float(precision_score(labels, predictions, zero_division=0)),
        "unknown_recall": float(recall_score(labels, predictions, zero_division=0)),
        "unknown_f1": float(f1_score(labels, predictions, zero_division=0)),
        "score_threshold": 1.0,
        "score_direction": "higher_is_unknown",
    }
    if len(np.unique(labels)) == 2:
        output["auroc"] = float(roc_auc_score(labels, scores))
        output["aupr_unknown_positive"] = float(average_precision_score(labels, scores))
    else:
        output["auroc"] = None
        output["aupr_unknown_positive"] = None
        output["ranking_metrics_status"] = "unavailable_single_ground_truth_class"
    return output


def evaluate_trajectory(results: Sequence[HealthMonitoringResult]) -> dict:
    """Report descriptive trajectory coverage without claiming ground truth."""

    if not results:
        raise ValueError("results must contain at least one item")
    trends = [item.trend for item in results]
    alarms = [item.alarm_state for item in results]
    rates = np.asarray(
        [item.degradation_rate for item in results if item.degradation_rate is not None],
        dtype=float,
    )
    transitions = sum(left != right for left, right in zip(alarms, alarms[1:]))
    return {
        "n_windows": int(len(results)),
        "history_coverage": float(sum(item.trend != "insufficient_history" for item in results) / len(results)),
        "worsening_fraction": float(sum(trend == "worsening" for trend in trends) / len(results)),
        "recovering_fraction": float(sum(trend == "recovering" for trend in trends) / len(results)),
        "alarm_fraction": float(sum(alarm != "normal" for alarm in alarms) / len(results)),
        "alarm_transition_count": int(transitions),
        "flapping_transition_rate": float(transitions / max(1, len(results) - 1)),
        "mean_degradation_rate": float(np.mean(rates)) if len(rates) else None,
        "ground_truth_status": "descriptive_only_no_event_labels",
    }


def _time_value(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def evaluate_event_metrics(
    results: Sequence[HealthMonitoringResult],
    event_labels: Iterable[bool | int] | None = None,
    timestamps: Sequence[object] | None = None,
) -> dict:
    """Calculate event metrics only when event labels are explicitly supplied."""

    if event_labels is None:
        return {
            "event_metrics_status": "unavailable_no_event_labels",
            "event_recall": None,
            "false_alarms_per_hour": None,
            "detection_delay_seconds": None,
        }
    labels = _binary(event_labels, name="event_labels")
    if len(labels) != len(results):
        raise ValueError("event_labels must align one-to-one with results")
    alarm_predictions = np.asarray([int(item.alarm_state != "normal") for item in results])
    true_events = labels == 1
    event_recall = float(alarm_predictions[true_events].mean()) if true_events.any() else None
    false_alarm_count = int(((alarm_predictions == 1) & (labels == 0)).sum())
    false_alarms_per_hour = None
    if timestamps is not None:
        if len(timestamps) != len(results):
            raise ValueError("timestamps must align one-to-one with results")
        values = [_time_value(item) for item in timestamps]
        duration_hours = max(0.0, (max(values) - min(values)) / 3600.0)
        if duration_hours > 0:
            false_alarms_per_hour = float(false_alarm_count / duration_hours)
    return {
        "event_metrics_status": "computed",
        "event_recall": event_recall,
        "false_alarms_per_hour": false_alarms_per_hour,
        "detection_delay_seconds": None,
        "false_alarm_count": false_alarm_count,
    }


def evaluate_results(
    results: Sequence[HealthMonitoringResult],
    *,
    unknown_labels: Iterable[bool | int] | None = None,
    event_labels: Iterable[bool | int] | None = None,
    timestamps: Sequence[object] | None = None,
) -> dict:
    """Combine available metrics and preserve unavailable-state explanations."""

    output = {"trajectory": evaluate_trajectory(results), "events": evaluate_event_metrics(results, event_labels, timestamps)}
    if unknown_labels is None:
        output["open_set"] = {"status": "unavailable_no_unknown_labels"}
    else:
        output["open_set"] = evaluate_open_set(results, unknown_labels)
    return output


def assert_group_disjoint(train_groups: Iterable[object], test_groups: Iterable[object]) -> None:
    overlap = set(train_groups).intersection(test_groups)
    if overlap:
        raise ValueError(f"group leakage detected: {sorted(overlap, key=str)}")


def assert_temporal_order(train_timestamps: Sequence[object], test_timestamps: Sequence[object]) -> None:
    if not train_timestamps or not test_timestamps:
        raise ValueError("both train_timestamps and test_timestamps are required")
    if max(_time_value(item) for item in train_timestamps) >= min(_time_value(item) for item in test_timestamps):
        raise ValueError("temporal leakage detected: test starts before train ends")
