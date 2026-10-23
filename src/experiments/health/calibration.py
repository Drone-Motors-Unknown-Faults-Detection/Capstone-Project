"""Calibration-only mapping from Open Set score to relative health index."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HealthIndexCalibrator:
    """Piecewise monotone calibration using known calibration scores only.

    Three anchors on the score axis: ``healthy_anchor`` (calibration lower
    quantile) maps to 1.0, ``warning_anchor`` (calibration ``confidence``
    quantile, i.e. the Open Set rejection line, normalised score 1.0) maps to
    ``warning_health``, and ``critical_anchor`` (``critical_multiple`` times
    the rejection line, the same position as the exp2 quarantine line) maps
    to 0.0.  Both segments are linear and the ends are clipped, so
    ``health >= warning_health`` means the score has not crossed the rejection
    line.  The result is a *relative* health index, not a physical damage
    percentage or RUL estimate.
    """

    confidence: float
    lower_quantile: float
    healthy_anchor: float
    warning_anchor: float
    critical_anchor: float
    n_calibration: int
    warning_health: float = 0.5
    critical_multiple: float = 2.0
    score_direction: str = "higher_is_worse"

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        if not 0.0 <= self.lower_quantile < self.confidence:
            raise ValueError("lower_quantile must be >=0 and below confidence")
        if not self.healthy_anchor < self.warning_anchor < self.critical_anchor:
            raise ValueError("anchors must satisfy healthy < warning < critical")
        if not 0.0 < self.warning_health < 1.0:
            raise ValueError("warning_health must be strictly between zero and one")
        if self.critical_multiple <= 1.0:
            raise ValueError("critical_multiple must exceed one")
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
        warning_health: float = 0.5,
        critical_multiple: float = 2.0,
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
        if critical_multiple <= 1.0:
            raise ValueError("critical_multiple must exceed one")
        healthy = float(np.quantile(scores, lower_quantile))
        warning = float(np.quantile(scores, confidence))
        # Degenerate calibration (all scores equal, or all zero) still needs a
        # strictly increasing anchor sequence.
        if warning <= healthy:
            warning = healthy + max(np.finfo(float).eps, abs(healthy) * 1e-12)
        critical = float(critical_multiple * warning)
        if critical <= warning:
            critical = warning + max(np.finfo(float).eps, abs(warning) * 1e-12)
        return cls(
            confidence=confidence,
            lower_quantile=lower_quantile,
            healthy_anchor=healthy,
            warning_anchor=warning,
            critical_anchor=critical,
            n_calibration=int(scores.size),
            warning_health=float(warning_health),
            critical_multiple=float(critical_multiple),
        )

    def transform(self, scores: np.ndarray | float) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("scores must be finite")
        inside = 1.0 - (1.0 - self.warning_health) * (
            (values - self.healthy_anchor) / (self.warning_anchor - self.healthy_anchor)
        )
        beyond = self.warning_health - self.warning_health * (
            (values - self.warning_anchor) / (self.critical_anchor - self.warning_anchor)
        )
        return np.clip(np.where(values <= self.warning_anchor, inside, beyond), 0.0, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "HealthIndexCalibrator":
        return cls(**payload)
