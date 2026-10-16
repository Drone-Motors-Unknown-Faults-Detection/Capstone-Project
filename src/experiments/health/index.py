"""Calibrated relative health index built on the shared Open Set detector.

The class in this module is deliberately a thin adapter around
``core.openset.create_openset_detector``.  It keeps the historical detector
threshold (normalised score ``> 1`` means unknown) and adds a monotone health
view calibrated *only* on the known calibration split.  The resulting value is
relative to the calibration population; it is not a physical damage fraction,
failure probability, or remaining-useful-life prediction.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import numpy as np
from sklearn.preprocessing import RobustScaler

from core.openset import OpenSetDetector, canonical_openset_method, create_openset_detector
from experiments.health.calibration import HealthIndexCalibrator
from experiments.health.diagnosis import DiagnosisResolver
from experiments.health.schema import HealthMonitoringResult
from experiments.health.severity import DEFAULT_SEVERITY_POLICY, RelativeSeverityPolicy


def _matrix(value: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if not allow_empty and matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def relative_severity(health_index: float, policy: RelativeSeverityPolicy = DEFAULT_SEVERITY_POLICY) -> str:
    """Map a relative health value to a policy stage.

    These thresholds are presentation policy, not supervised severity labels.
    They are kept in one place so a later ordinal data contract can replace
    them without changing the detector or calibration code.
    """

    return policy.classify(health_index)


def _decision_margin_confidence(openset_score: float) -> float:
    """Return a bounded distance-from-threshold confidence proxy.

    This is explicitly a detector-margin proxy, not a calibrated class
    probability.  A score exactly at the Open Set threshold has zero
    confidence; confidence increases as the score moves away from 1.
    """

    distance = abs(float(openset_score) - 1.0)
    return float(distance / (1.0 + distance))


class CalibratedHealthIndex:
    """Fit a known-only Open Set detector and expose relative health output."""

    def __init__(
        self,
        *,
        scaler: RobustScaler,
        detector: OpenSetDetector,
        calibrator: HealthIndexCalibrator,
        openset_method: str,
        mahalanobis_method: str,
        confidence: float,
        knn_neighbors: int,
        label_to_fault_type: Mapping[int, str] | None = None,
    ) -> None:
        self.scaler = scaler
        self.detector = detector
        self.calibrator = calibrator
        self.openset_method = canonical_openset_method(openset_method)
        self.mahalanobis_method = str(mahalanobis_method)
        self.confidence = float(confidence)
        self.knn_neighbors = int(knn_neighbors)
        self.label_to_fault_type = {int(k): str(v) for k, v in (label_to_fault_type or {}).items()}
        self.diagnosis = DiagnosisResolver(self.label_to_fault_type)

    @classmethod
    def fit(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
        *,
        openset_method: str = "mahalanobis",
        mahalanobis_method: str = "ledoit_wolf",
        confidence: float = 0.95,
        knn_neighbors: int = 5,
        lower_quantile: float = 0.10,
        label_to_fault_type: Mapping[int, str] | None = None,
    ) -> "CalibratedHealthIndex":
        """Fit using known train/calibration arrays only.

        ``y_calibration`` may contain only labels present in ``y_train``.  This
        prevents an unknown test pool from being accidentally used to set the
        health scale or Open Set threshold.
        """

        X_train = _matrix(X_train, "X_train")
        X_calibration = _matrix(X_calibration, "X_calibration")
        if X_train.shape[1] != X_calibration.shape[1]:
            raise ValueError("training and calibration feature dimensions must match")
        y_train = np.asarray(y_train)
        y_calibration = np.asarray(y_calibration)
        if y_train.ndim != 1 or len(y_train) != len(X_train):
            raise ValueError("y_train must have one label per training sample")
        if y_calibration.ndim != 1 or len(y_calibration) != len(X_calibration):
            raise ValueError("y_calibration must have one label per calibration sample")
        known_labels = set(np.unique(y_train).tolist())
        calibration_labels = set(np.unique(y_calibration).tolist())
        if not calibration_labels.issubset(known_labels):
            raise ValueError("calibration labels must be a subset of known training labels")
        canonical_method = canonical_openset_method(openset_method)
        scaler = RobustScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_calibration_scaled = scaler.transform(X_calibration)
        detector = create_openset_detector(
            canonical_method,
            confidence=confidence,
            mahalanobis_method=mahalanobis_method,  # type: ignore[arg-type]
            knn_neighbors=knn_neighbors,
        ).fit(X_train_scaled, y_train, X_calibration_scaled, y_calibration)
        calibration_scores = np.asarray(detector.score_samples(X_calibration_scaled), dtype=float)
        calibrator = HealthIndexCalibrator.fit(
            calibration_scores,
            confidence=confidence,
            lower_quantile=lower_quantile,
        )
        return cls(
            scaler=scaler,
            detector=detector,
            calibrator=calibrator,
            openset_method=canonical_method,
            mahalanobis_method=mahalanobis_method,
            confidence=confidence,
            knn_neighbors=knn_neighbors,
            label_to_fault_type=label_to_fault_type,
        )

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        matrix = _matrix(features, "features", allow_empty=True)
        expected = int(getattr(self.scaler, "n_features_in_", matrix.shape[1]))
        if matrix.shape[1] != expected:
            raise ValueError(f"features has {matrix.shape[1]} columns; expected {expected}")
        return matrix

    def predict(
        self,
        features: np.ndarray,
        *,
        timestamp: str | None = None,
        condition: str,
    ) -> list[HealthMonitoringResult]:
        """Return one immutable result per input window."""

        matrix = self._validate_features(features)
        if len(matrix) == 0:
            return []
        scaled = self.scaler.transform(matrix)
        scores = np.asarray(self.detector.score_samples(scaled), dtype=float)
        labels = np.asarray(self.detector.predict_known_class(scaled), dtype=int)
        health_values = np.asarray(self.calibrator.transform(scores), dtype=float)
        results: list[HealthMonitoringResult] = []
        for score, label, health in zip(scores, labels, health_values, strict=True):
            unknown = int(label) < 0 or float(score) > 1.0
            health_value = float(health)
            confidence = _decision_margin_confidence(float(score))
            diagnosis = self.diagnosis.resolve(int(label), float(score), confidence)
            results.append(
                HealthMonitoringResult(
                    is_fault=unknown,
                    health_index=health_value,
                    degradation_score=1.0 - health_value,
                    severity_stage=relative_severity(health_value),
                    trend="insufficient_history",
                    degradation_rate=None,
                    fault_type=diagnosis.fault_type,
                    fault_type_confidence=diagnosis.confidence,
                    openset_method=self.openset_method,
                    openset_score=float(score),
                    is_unknown_fault=diagnosis.is_unknown,
                    prediction_confidence=confidence,
                    uncertainty=1.0 - confidence,
                    estimated_rul=None,
                    rul_available=False,
                    timestamp=timestamp,
                    condition=condition,
                    data_quality="valid",
                    raw_health_index=health_value,
                    smoothed_health_index=health_value,
                )
            )
        return results

    def predict_one(
        self,
        features: np.ndarray,
        *,
        timestamp: str | None = None,
        condition: str,
    ) -> HealthMonitoringResult:
        results = self.predict(np.atleast_2d(features), timestamp=timestamp, condition=condition)
        if len(results) != 1:
            raise ValueError("predict_one requires exactly one feature window")
        return results[0]

    def metadata(self) -> dict:
        """Return portable provenance without serialising fitted model objects."""

        return {
            "openset_method": self.openset_method,
            "mahalanobis_method": self.mahalanobis_method
            if self.openset_method == "mahalanobis"
            else None,
            "knn_neighbors": self.knn_neighbors if self.openset_method == "knn" else None,
            "health_mapping": "known calibration lower quantile to confidence quantile",
            "score_direction": "higher_is_worse",
            "unknown_threshold": 1.0,
            "rul_available": False,
            "calibration": asdict(self.calibrator),
            "label_to_fault_type": dict(self.label_to_fault_type),
            "confidence_proxy": "distance from Open Set threshold; not probability",
        }
