"""Stable output schema for binary, health and full monitoring modes.

The schema deliberately describes a *relative* health state.  It does not imply
physical damage percentage or remaining useful life unless a future data
contract explicitly provides those labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


SeverityStage = Literal["healthy", "early_warning", "degraded", "critical"]
Trend = Literal["stable", "worsening", "recovering", "insufficient_history"]
DataQuality = Literal["valid", "invalid", "insufficient"]
AlarmState = Literal["normal", "warning", "critical"]
SeverityBasis = Literal["relative_calibrated", "supervised_ordinal"]
ChangePointState = Literal["none", "candidate", "confirmed"]

_SEVERITY_STAGES = {"healthy", "early_warning", "degraded", "critical"}
_TRENDS = {"stable", "worsening", "recovering", "insufficient_history"}
_DATA_QUALITY = {"valid", "invalid", "insufficient"}
_ALARM_STATES = {"normal", "warning", "critical"}
_SEVERITY_BASES = {"relative_calibrated", "supervised_ordinal"}
_CHANGE_POINT_STATES = {"none", "candidate", "confirmed"}


def _bounded(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class HealthMonitoringResult:
    """One-window or one-update health result.

    ``is_fault`` is intentionally retained as a boolean so callers can emit
    the historical ``0/1`` representation without losing the richer fields.
    ``estimated_rul`` stays ``None`` by default because the current formal data
    has no motor-level run-to-failure endpoint.
    """

    is_fault: bool
    health_index: float
    degradation_score: float
    severity_stage: SeverityStage
    trend: Trend
    degradation_rate: float | None
    fault_type: str
    fault_type_confidence: float | None
    openset_method: str
    openset_score: float
    is_unknown_fault: bool
    prediction_confidence: float
    uncertainty: float
    estimated_rul: float | None
    rul_available: bool
    timestamp: str | None
    condition: str
    data_quality: DataQuality
    alarm_state: AlarmState = "normal"
    alarm_reason: str | None = None
    severity_basis: SeverityBasis = "relative_calibrated"
    raw_health_index: float | None = None
    smoothed_health_index: float | None = None
    change_point_state: ChangePointState = "none"

    def __post_init__(self) -> None:
        health = _bounded(self.health_index, "health_index")
        degradation = _bounded(self.degradation_score, "degradation_score")
        if abs(degradation - (1.0 - health)) > 1e-9:
            raise ValueError("degradation_score must equal 1 - health_index")
        if self.severity_stage not in _SEVERITY_STAGES:
            raise ValueError(f"unsupported severity_stage: {self.severity_stage!r}")
        if self.trend not in _TRENDS:
            raise ValueError(f"unsupported trend: {self.trend!r}")
        if self.data_quality not in _DATA_QUALITY:
            raise ValueError(f"unsupported data_quality: {self.data_quality!r}")
        if self.alarm_state not in _ALARM_STATES:
            raise ValueError(f"unsupported alarm_state: {self.alarm_state!r}")
        if self.severity_basis not in _SEVERITY_BASES:
            raise ValueError(f"unsupported severity_basis: {self.severity_basis!r}")
        if self.raw_health_index is not None:
            _bounded(self.raw_health_index, "raw_health_index")
        if self.smoothed_health_index is not None:
            _bounded(self.smoothed_health_index, "smoothed_health_index")
        if self.change_point_state not in _CHANGE_POINT_STATES:
            raise ValueError(f"unsupported change_point_state: {self.change_point_state!r}")
        if self.fault_type_confidence is not None:
            _bounded(self.fault_type_confidence, "fault_type_confidence")
        _bounded(self.prediction_confidence, "prediction_confidence")
        _bounded(self.uncertainty, "uncertainty")
        if abs(float(self.prediction_confidence) + float(self.uncertainty) - 1.0) > 1e-9:
            raise ValueError("prediction_confidence + uncertainty must equal 1")
        if not self.openset_method:
            raise ValueError("openset_method must be non-empty")
        if not self.condition:
            raise ValueError("condition must be non-empty")
        if self.is_unknown_fault and self.fault_type != "unknown":
            raise ValueError("unknown Open Set predictions must use fault_type='unknown'")
        if self.rul_available and self.estimated_rul is None:
            raise ValueError("rul_available=True requires estimated_rul")
        if not self.rul_available and self.estimated_rul is not None:
            raise ValueError("estimated_rul must be None when rul_available=False")

    def to_dict(self, mode: str = "full") -> dict:
        """Serialize as ``binary``, ``health`` or ``full`` output.

        ``binary`` intentionally contains exactly the backwards-compatible
        field.  ``health`` contains the monitoring contract but not optional
        diagnosis/RUL detail; ``full`` contains every schema field.
        """
        if mode not in {"binary", "health", "full"}:
            raise ValueError("mode must be binary, health or full")
        if mode == "binary":
            return {"is_fault": int(self.is_fault)}
        payload = asdict(self)
        payload["is_fault"] = int(self.is_fault)
        if mode == "health":
            return {
                key: payload[key]
                for key in (
                    "is_fault",
                    "health_index",
                    "degradation_score",
                    "severity_stage",
                    "trend",
                    "degradation_rate",
                    "openset_method",
                    "openset_score",
                    "is_unknown_fault",
                    "prediction_confidence",
                    "uncertainty",
                    "timestamp",
                    "condition",
                    "data_quality",
                    "alarm_state",
                    "alarm_reason",
                    "severity_basis",
                    "raw_health_index",
                    "smoothed_health_index",
                    "change_point_state",
                )
            }
        return payload
