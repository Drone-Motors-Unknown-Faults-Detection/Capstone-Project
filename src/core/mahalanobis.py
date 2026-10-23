"""Mahalanobis open-set detectors with reproducible calibration.

複製來源：scripts/openset_mahalanobis.py（本 repo git 歷史 commit dda8910；
論文版程式碼現存於原始論文 repo）。
本專案不依賴論文版程式碼，需要的零件以「複製 + 註記來源」方式帶入。

The legacy detector models all known classes as one Gaussian.  The improved
detectors model every known class separately and divide its distance by a
class-specific calibration quantile.  A normalized score above one is unknown.

Legacy 的合併分布標籤是 ``LEGACY_POOLED_LABEL``（-2），與 ``predict_known_class``
的未知哨兵 -1 分開；score ≤ 1 的列取縮放空間中歐氏距離最近的類別 train 中心
（原始論文 Step 4 的 ``LABEL_ORDER[closest_known_cluster]``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.covariance import LedoitWolf, MinCovDet, OAS
from sklearn.decomposition import PCA


Method = Literal["legacy", "ledoit_wolf", "oas", "mcd"]
UNKNOWN_LABEL = -1
LEGACY_POOLED_LABEL = -2


@dataclass(frozen=True)
class ClassDistribution:
    label: int
    location: np.ndarray
    precision: np.ndarray
    threshold: float


def _distances(X: np.ndarray, location: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = np.asarray(X, dtype=float) - location
    # 拆成兩個兩運算元 einsum：兩運算元不跑路徑搜尋，數字與三運算元 optimize=True 的路徑逐位元相同
    # （delta @ precision 會差 1 ulp，不採用）
    weighted = np.einsum("ij,jk->ik", delta, precision, optimize=True)
    squared = np.einsum("ij,ij->i", weighted, delta)
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
        self.legacy_centroids_: dict[int, np.ndarray] = {}  # legacy：各已知類 train 中心（轉換後空間）

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        expected = getattr(self, "n_features_in_", None)
        if expected is not None and (X.ndim != 2 or X.shape[1] != expected):
            actual = X.shape[1] if X.ndim == 2 else f"shape {X.shape}"
            raise ValueError(f"X has {actual} features; expected {expected}")
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
        if X_train.ndim != 2 or X_calibration.ndim != 2 or X_train.shape[1] != X_calibration.shape[1]:
            raise ValueError("training and calibration feature dimensions must match")
        self.n_features_in_ = None

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
        self.n_features_in_ = int(X_train.shape[1])
        self.distributions_ = []
        self.legacy_centroids_ = {}

        if self.method == "legacy":
            location = train_t.mean(axis=0)
            covariance = np.cov(train_t, rowvar=False)
            precision = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * self.ridge)
            # Preserve the notebook baseline exactly: its threshold is measured
            # on the same training features used to estimate mean/covariance.
            threshold = float(np.quantile(_distances(train_t, location, precision), self.confidence))
            self.distributions_.append(ClassDistribution(LEGACY_POOLED_LABEL, location, precision, threshold))
            self.legacy_centroids_ = {
                int(label): train_t[y_train == label].mean(axis=0) for label in np.unique(y_train)
            }
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

    def _normalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """每類正規化分數矩陣與轉換後特徵；兩個推論方法共用，距離只算一次。"""
        if not self.distributions_:
            raise RuntimeError("fit must be called before scoring")
        X_t = self._transform(X)
        normalized = np.column_stack(
            [
                _distances(X_t, item.location, item.precision) / max(item.threshold, np.finfo(float).eps)
                for item in self.distributions_
            ]
        )
        return normalized, X_t

    def _labels_from(self, normalized: np.ndarray, X_t: np.ndarray) -> np.ndarray:
        if self.method == "legacy":
            # 合併分布只決定「已知／未知」；已知列取縮放空間中歐氏距離最近的類別 train 中心
            class_labels = sorted(self.legacy_centroids_)
            class_distances = np.column_stack(
                [np.linalg.norm(X_t - self.legacy_centroids_[label], axis=1) for label in class_labels]
            )
            labels = np.array(class_labels, dtype=int)[np.argmin(class_distances, axis=1)]
        else:
            indices = np.argmin(normalized, axis=1)
            labels = np.array([self.distributions_[index].label for index in indices], dtype=int)
        labels[np.min(normalized, axis=1) > 1.0] = UNKNOWN_LABEL
        return labels

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return normalized open-set scores; values above one are unknown."""
        normalized, _ = self._normalized(X)
        return np.min(normalized, axis=1)

    def predict_known_class(self, X: np.ndarray) -> np.ndarray:
        """Return nearest known class, or -1 when the normalized score exceeds one."""
        normalized, X_t = self._normalized(X)
        return self._labels_from(normalized, X_t)

    def score_and_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """一次算距離，同時回傳正規化分數與已知類標籤（與分開呼叫結果相同）。"""
        normalized, X_t = self._normalized(X)
        return np.min(normalized, axis=1), self._labels_from(normalized, X_t)

    def class_summaries(self) -> list[dict]:
        """Return calibration metadata for the common Open Set detector API."""
        return [
            {"label": int(item.label), "threshold": float(item.threshold)}
            for item in self.distributions_
        ]
