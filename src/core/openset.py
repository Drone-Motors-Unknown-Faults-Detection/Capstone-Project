"""可切換的 Open Set detector 介面與建立工廠。

目前支援兩個原理不同的方法：

``mahalanobis``
    使用 :class:`core.mahalanobis.MahalanobisOpenSetDetector`。逐類估計
    共變異數，以校準距離分位數正規化；分數大於 1 判為未知。

``knn``
    在相同的 RobustScaler 特徵空間中，計算樣本到每個已知類別訓練集的
    k 個最近鄰平均歐氏距離。每類以獨立 calibration split 的距離分位數
    正規化；最小正規化分數大於 1 判為未知。

兩者都只用 known train/calibration 資料擬合與選 threshold，unknown/test
資料不參與校準。分數方向一致：越大越可能是 unknown。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from core.mahalanobis import MahalanobisOpenSetDetector, Method as MahalanobisMethod


OpenSetMethod = Literal["mahalanobis", "knn"]
SUPPORTED_OPENSET_METHODS: tuple[OpenSetMethod, ...] = ("mahalanobis", "knn")
OPENSET_METHOD_ALIASES = {
    "mahalanobis": "mahalanobis",
    "maha": "mahalanobis",
    "knn": "knn",
    "k-nn": "knn",
    "k_nn": "knn",
}


RESOLVED_CONFIG_VERSION = "openset_config_v1"
MAHALANOBIS_METHODS = ("legacy", "ledoit_wolf", "oas", "mcd")


class OpenSetConfigError(ValueError):
    """Open Set 參數不合法；訊息以參數名開頭。"""


def canonical_openset_method(value: str) -> OpenSetMethod:
    """Normalize public method names to the two formal factory methods."""
    canonical = OPENSET_METHOD_ALIASES.get(str(value).strip().lower())
    if canonical is None:
        supported = ", ".join(SUPPORTED_OPENSET_METHODS)
        raise ValueError(f"Unsupported Open Set method: {value!r}; choose from {supported}")
    return canonical  # type: ignore[return-value]


def resolve_openset_config(
    openset_method: str = "mahalanobis",
    *,
    confidence: float = 0.95,
    mahalanobis_method: str = "ledoit_wolf",
    knn_neighbors: int = 5,
) -> dict:
    """把 Open Set 參數解析成唯一的 resolved config；非法值丟 OpenSetConfigError。

    CLI、程式 run() 與 Web 都經這裡，所以同一組參數得到同一份設定。
    """
    try:
        canonical = canonical_openset_method(openset_method)
    except ValueError as exc:
        raise OpenSetConfigError(f"openset_method: {exc}") from None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float, np.integer, np.floating)):
        raise OpenSetConfigError(f"confidence: expected a number in (0, 1), got {confidence!r}")
    if not np.isfinite(confidence) or not 0.0 < float(confidence) < 1.0:
        raise OpenSetConfigError(f"confidence: expected a number in (0, 1), got {confidence!r}")
    if isinstance(knn_neighbors, bool) or not isinstance(knn_neighbors, (int, np.integer)) or knn_neighbors < 1:
        raise OpenSetConfigError(f"knn_neighbors: expected an integer >= 1, got {knn_neighbors!r}")
    if mahalanobis_method not in MAHALANOBIS_METHODS:
        raise OpenSetConfigError(
            f"mahalanobis_method: expected one of {', '.join(MAHALANOBIS_METHODS)}, got {mahalanobis_method!r}")

    effective: dict = {"confidence": float(confidence), "threshold": 1.0}
    if canonical == "mahalanobis":
        effective["mahalanobis_method"] = mahalanobis_method
        method_id = f"mahalanobis/{mahalanobis_method}"
        ignored = [{"parameter": "knn_neighbors", "value": int(knn_neighbors),
                    "reason": "only used by openset_method=knn"}]
    else:
        effective["knn_neighbors"] = int(knn_neighbors)
        method_id = f"knn/k{int(knn_neighbors)}"
        ignored = [{"parameter": "mahalanobis_method", "value": mahalanobis_method,
                    "reason": "covariance estimator is only used by openset_method=mahalanobis"}]
    return {
        "version": RESOLVED_CONFIG_VERSION,
        "method_id": method_id,
        "openset_method": canonical,
        "requested_method": str(openset_method),
        "effective": effective,
        "ignored": ignored,
    }


@runtime_checkable
class OpenSetDetector(Protocol):
    """Open Set detector 的最小共用介面。"""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> "OpenSetDetector": ...

    def score_samples(self, X: np.ndarray) -> np.ndarray: ...

    def predict_known_class(self, X: np.ndarray) -> np.ndarray: ...

    def class_summaries(self) -> list[dict]: ...


@dataclass(frozen=True)
class KNNClassModel:
    label: int
    neighbors: NearestNeighbors
    threshold: float
    n_train: int
    n_neighbors: int


def _as_feature_matrix(X: np.ndarray, *, name: str, allow_empty: bool = False) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if not allow_empty and X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} must contain only finite values")
    return X


class KNNOpenSetDetector:
    """逐類 k-NN 距離拒絕法。

    原始 class score 是樣本到該類 train split 的 ``k`` 個最近鄰之平均
    Euclidean distance。每類 threshold 是該類 calibration split 原始分數的
    ``confidence`` 分位數。輸出的 Open Set score 是所有類別
    ``raw_distance / class_threshold`` 的最小值，因此 ``score > 1`` 為 unknown。
    """

    def __init__(self, confidence: float = 0.95, n_neighbors: int = 5) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least one")
        self.confidence = confidence
        self.n_neighbors = int(n_neighbors)
        self.models_: list[KNNClassModel] = []
        self.n_features_in_: int | None = None

    @staticmethod
    def _mean_neighbor_distance(model: KNNClassModel, X: np.ndarray) -> np.ndarray:
        distances, _ = model.neighbors.kneighbors(X, return_distance=True)
        return distances.mean(axis=1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> "KNNOpenSetDetector":
        X_train = _as_feature_matrix(X_train, name="X_train")
        X_calibration = _as_feature_matrix(X_calibration, name="X_calibration")
        y_train = np.asarray(y_train)
        y_calibration = np.asarray(y_calibration)
        if y_train.ndim != 1 or len(y_train) != len(X_train):
            raise ValueError("y_train must have one label per training sample")
        if y_calibration.ndim != 1 or len(y_calibration) != len(X_calibration):
            raise ValueError("y_calibration must have one label per calibration sample")
        if X_train.shape[1] != X_calibration.shape[1]:
            raise ValueError("training and calibration feature dimensions must match")

        self.n_features_in_ = X_train.shape[1]
        self.models_ = []
        for raw_label in np.unique(y_train):
            label = int(raw_label)
            train_class = X_train[y_train == raw_label]
            calibration = X_calibration[y_calibration == raw_label]
            if len(calibration) == 0:
                raise ValueError(f"Calibration split has no samples for class {label}")
            effective_k = min(self.n_neighbors, len(train_class))
            neighbors = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
            neighbors.fit(train_class)
            provisional = KNNClassModel(label, neighbors, 1.0, len(train_class), effective_k)
            distances = self._mean_neighbor_distance(provisional, calibration)
            threshold = float(np.quantile(distances, self.confidence))
            self.models_.append(
                KNNClassModel(label, neighbors, threshold, len(train_class), effective_k)
            )
        return self

    def _normalized_scores(self, X: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("fit must be called before scoring")
        X = _as_feature_matrix(X, name="X", allow_empty=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_in_}"
            )
        if len(X) == 0:
            return np.empty((0, len(self.models_)), dtype=float)
        eps = np.finfo(float).eps
        return np.column_stack(
            [
                self._mean_neighbor_distance(model, X) / max(model.threshold, eps)
                for model in self.models_
            ]
        )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """回傳正規化分數；越大越像 unknown，嚴格大於 1 判 unknown。"""
        normalized = self._normalized_scores(X)
        if len(normalized) == 0:
            return np.empty(0, dtype=float)
        return np.min(normalized, axis=1)

    def predict_known_class(self, X: np.ndarray) -> np.ndarray:
        normalized = self._normalized_scores(X)
        if len(normalized) == 0:
            return np.empty(0, dtype=int)
        indices = np.argmin(normalized, axis=1)
        labels = np.array([self.models_[index].label for index in indices], dtype=int)
        labels[np.min(normalized, axis=1) > 1.0] = -1
        return labels

    def class_summaries(self) -> list[dict]:
        return [
            {
                "label": model.label,
                "threshold": model.threshold,
                "n_reference": model.n_train,
                "effective_neighbors": model.n_neighbors,
            }
            for model in self.models_
        ]


def create_openset_detector(
    openset_method: str = "mahalanobis",
    *,
    confidence: float = 0.95,
    mahalanobis_method: MahalanobisMethod = "ledoit_wolf",
    knn_neighbors: int = 5,
) -> OpenSetDetector:
    """建立 detector；所有實作的正規化 unknown threshold 都固定為 1。"""
    resolved = resolve_openset_config(openset_method, confidence=confidence,
                                      mahalanobis_method=mahalanobis_method, knn_neighbors=knn_neighbors)
    return create_detector_from_config(resolved)


def create_detector_from_config(resolved: dict) -> OpenSetDetector:
    """依 resolve_openset_config() 的結果建立 detector。"""
    openset_method = resolved["openset_method"]
    effective = resolved["effective"]
    if openset_method == "mahalanobis":
        return MahalanobisOpenSetDetector(
            method=effective["mahalanobis_method"],
            confidence=effective["confidence"],
        )
    if openset_method == "knn":
        return KNNOpenSetDetector(confidence=effective["confidence"], n_neighbors=effective["knn_neighbors"])
    raise AssertionError(f"unhandled canonical Open Set method: {openset_method}")
