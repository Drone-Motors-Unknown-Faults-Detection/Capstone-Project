"""Calibration-only mapping from Open Set score to relative health index."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HealthIndexCalibrator:
    """Piecewise monotone calibration using known calibration scores only.

    Scores below the lower healthy quantile map to 1.0.  Scores at or above
    the configured upper quantile map to 0.0.  The result is a *relative*
    health index, not a physical damage percentage or RUL estimate.
    """

    confidence: float
    lower_quantile: float
    healthy_anchor: float
    critical_anchor: float
    n_calibration: int
    score_direction: str = "higher_is_worse"

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        if not 0.0 <= self.lower_quantile < self.confidence:
            raise ValueError("lower_quantile must be >=0 and below confidence")
        if self.critical_anchor <= self.healthy_anchor:
            raise ValueError("critical_anchor must exceed healthy_anchor")
        if self.n_calibration < 1:
            raise ValueError("n_calibration must be positive")
        if self.score_direction != "higher_is_worse":
            raise ValueError("unsupported score direction")

    @classmethod
    def fit(
        cls,
        calibration_scores: np.ndarray,
        *,
        confidence: float = 0.95,
        lower_quantile: float = 0.10,
    ) -> "HealthIndexCalibrator":
        scores = np.asarray(calibration_scores, dtype=float).reshape(-1)
        if scores.size == 0:
            raise ValueError("calibration_scores must contain at least one value")
        if not np.isfinite(scores).all():
            raise ValueError("calibration_scores must be finite")
        if np.any(scores < 0):
            raise ValueError("calibration_scores must be non-negative")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        if not 0.0 <= lower_quantile < confidence:
            raise ValueError("lower_quantile must be >=0 and below confidence")
        healthy = float(np.quantile(scores, lower_quantile))
        critical = float(np.quantile(scores, confidence))
        if critical <= healthy:
            critical = healthy + max(np.finfo(float).eps, abs(healthy) * 1e-12)
        return cls(confidence, lower_quantile, healthy, critical, int(scores.size))

    def transform(self, scores: np.ndarray | float) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("scores must be finite")
        scaled = (values - self.healthy_anchor) / (self.critical_anchor - self.healthy_anchor)
        return np.clip(1.0 - scaled, 0.0, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "HealthIndexCalibrator":
        return cls(**payload)
