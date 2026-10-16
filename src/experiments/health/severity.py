"""Severity-stage policy for the current Level-A data contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.health.schema import SeverityStage


@dataclass(frozen=True)
class RelativeSeverityPolicy:
    """Map relative health to presentation stages without fake labels.

    The current formal data has no ordinal severity annotation.  Therefore the
    policy is explicitly relative to the known calibration population and its
    metadata must accompany any report that displays these stages.
    """

    healthy_min: float = 0.8
    early_warning_min: float = 0.5
    degraded_min: float = 0.2
    basis: str = "relative_calibrated"

    def __post_init__(self) -> None:
        for name in ("healthy_min", "early_warning_min", "degraded_min"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not self.healthy_min > self.early_warning_min > self.degraded_min:
            raise ValueError("severity thresholds must descend healthy > warning > degraded")
        if self.basis != "relative_calibrated":
            raise ValueError("only relative_calibrated is supported without ordinal labels")

    def classify(self, health_index: float) -> SeverityStage:
        value = float(health_index)
        if not 0.0 <= value <= 1.0:
            raise ValueError("health_index must be in [0, 1]")
        if value >= self.healthy_min:
            return "healthy"
        if value >= self.early_warning_min:
            return "early_warning"
        if value >= self.degraded_min:
            return "degraded"
        return "critical"

    def metadata(self) -> dict:
        return {
            **asdict(self),
            "interpretation": "relative presentation stage; not physical damage or RUL",
            "ordinal_labels_available": False,
        }


DEFAULT_SEVERITY_POLICY = RelativeSeverityPolicy()
