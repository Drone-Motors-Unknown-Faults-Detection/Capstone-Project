"""Mahalanobis open-set detectors with reproducible calibration.

複製來源：scripts/openset_mahalanobis.py（本 repo git 歷史 commit dda8910；
論文版程式碼現存於 Ancestor repo）。
本專案不依賴論文版程式碼，需要的零件以「複製 + 註記來源」方式帶入；
此檔內容除本段註記外與來源一致。

The legacy detector models all known classes as one Gaussian.  The improved
detectors model every known class separately and divide its distance by a
class-specific calibration quantile.  A normalized score above one is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.covariance import LedoitWolf, MinCovDet, OAS
from sklearn.decomposition import PCA


Method = Literal["legacy", "ledoit_wolf", "oas", "mcd"]


@dataclass(frozen=True)
class ClassDistribution:
    label: int
    location: np.ndarray
    precision: np.ndarray
    threshold: float


def _distances(X: np.ndarray, location: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = np.asarray(X, dtype=float) - location
    squared = np.einsum("ij,jk,ik->i", delta, precision, delta, optimize=True)
    return np.sqrt(np.maximum(squared, 0.0))


class MahalanobisOpenSetDetector:
    """Fit legacy or class-conditional Mahalanobis open-set scores.

    Parameters
    ----------
    method:
        ``legacy`` reproduces the global empirical covariance with a small
        diagonal ridge.  Other methods use one covariance model per class.
    confidence:
        Known-only calibration quantile.  Unknown test samples never influence
        this value.
    mcd_variance:
        MCD requires samples > dimensions.  It therefore uses a global PCA
        learned only from training data before robust covariance estimation.
    """

    def __init__(
        self,
        method: Method = "ledoit_wolf",
        confidence: float = 0.95,
        ridge: float = 1e-6,
        mcd_variance: float = 0.95,
        random_state: int = 42,
    ) -> None:
        if method not in {"legacy", "ledoit_wolf", "oas", "mcd"}:
            raise ValueError(f"Unsupported method: {method}")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        self.method = method
        self.confidence = confidence
        self.ridge = ridge
        self.mcd_variance = mcd_variance
        self.random_state = random_state
        self.pca_: PCA | None = None
        self.distributions_: list[ClassDistribution] = []

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.pca_.transform(X) if self.pca_ is not None else X

    def _fit_estimator(self, X: np.ndarray):
        if self.method == "ledoit_wolf":
            return LedoitWolf(assume_centered=False).fit(X)
        if self.method == "oas":
            return OAS(assume_centered=False).fit(X)
        if self.method == "mcd":
            return MinCovDet(random_state=self.random_state).fit(X)
        raise RuntimeError("legacy does not use a sklearn covariance estimator")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> "MahalanobisOpenSetDetector":
        X_train = np.asarray(X_train, dtype=float)
        X_calibration = np.asarray(X_calibration, dtype=float)
        y_train = np.asarray(y_train)
        y_calibration = np.asarray(y_calibration)
        if not np.isfinite(X_train).all() or not np.isfinite(X_calibration).all():
            raise ValueError("Mahalanobis inputs must be finite")

        if self.method == "mcd":
            smallest_class = min(np.sum(y_train == label) for label in np.unique(y_train))
            max_components = max(1, min(X_train.shape[1], smallest_class - 2))
            self.pca_ = PCA(
                n_components=min(self.mcd_variance, max_components)
                if isinstance(self.mcd_variance, int)
                else self.mcd_variance,
                svd_solver="full",
                random_state=self.random_state,
            )
            self.pca_.fit(X_train)
            if self.pca_.n_components_ > max_components:
                self.pca_ = PCA(n_components=max_components, svd_solver="full")
                self.pca_.fit(X_train)

        train_t = self._transform(X_train)
        cal_t = self._transform(X_calibration)
        self.distributions_ = []

        if self.method == "legacy":
            location = train_t.mean(axis=0)
            covariance = np.cov(train_t, rowvar=False)
            precision = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * self.ridge)
            # Preserve the notebook baseline exactly: its threshold is measured
            # on the same training features used to estimate mean/covariance.
            threshold = float(np.quantile(_distances(train_t, location, precision), self.confidence))
            self.distributions_.append(ClassDistribution(-1, location, precision, threshold))
            return self

        for label in np.unique(y_train):
            estimator = self._fit_estimator(train_t[y_train == label])
            calibration = cal_t[y_calibration == label]
            if calibration.size == 0:
                raise ValueError(f"Calibration split has no samples for class {label}")
            distances = _distances(calibration, estimator.location_, estimator.precision_)
            threshold = float(np.quantile(distances, self.confidence))
            self.distributions_.append(
                ClassDistribution(int(label), estimator.location_, estimator.precision_, threshold)
            )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return normalized open-set scores; values above one are unknown."""
        if not self.distributions_:
            raise RuntimeError("fit must be called before score_samples")
        X_t = self._transform(X)
        normalized = [
            _distances(X_t, item.location, item.precision) / max(item.threshold, np.finfo(float).eps)
            for item in self.distributions_
        ]
        return np.min(np.column_stack(normalized), axis=1)

    def predict_known_class(self, X: np.ndarray) -> np.ndarray:
        """Return nearest known class, or -1 when the normalized score exceeds one."""
        X_t = self._transform(X)
        normalized = np.column_stack(
            [
                _distances(X_t, item.location, item.precision)
                / max(item.threshold, np.finfo(float).eps)
                for item in self.distributions_
            ]
        )
        indices = np.argmin(normalized, axis=1)
        labels = np.array([self.distributions_[index].label for index in indices], dtype=int)
        labels[np.min(normalized, axis=1) > 1.0] = -1
        return labels

    def class_summaries(self) -> list[dict]:
        """Return calibration metadata for the common Open Set detector API."""
        return [
            {"label": int(item.label), "threshold": float(item.threshold)}
            for item in self.distributions_
        ]
