"""健康-only 冷啟動設定下的異常偵測器基準（實驗六用）。

統一介面：fit(健康訓練集, 健康校準集) → score(X) 回傳正規化分數（> 1 = 未知）。
正規化方式一致：各法的原始異常分數以「校準集第 confidence 百分位」為 1.0，
與主線 Mahalanobis 偵測器同一套校準哲學，比較才公平。
所有方法共用 RobustScaler（以健康訓練集擬合）。僅使用 scikit-learn，不引入深度學習依賴。
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler

from core.mahalanobis import MahalanobisOpenSetDetector


class _Base:
    name = "base"

    def __init__(self, confidence: float = 0.95, seed: int = 42):
        self.confidence = confidence
        self.seed = seed
        self.scaler: RobustScaler | None = None
        self.threshold: float | None = None

    # 子類實作：擬合原始模型 / 回傳「越大越異常」的原始分數
    def _fit_raw(self, X: np.ndarray) -> None: ...
    def _raw_score(self, X: np.ndarray) -> np.ndarray: ...

    def fit(self, X_train_raw: np.ndarray, X_cal_raw: np.ndarray) -> "_Base":
        self.scaler = RobustScaler().fit(X_train_raw)
        self._fit_raw(self.scaler.transform(X_train_raw))
        cal = self._raw_score(self.scaler.transform(X_cal_raw))
        self.threshold = float(np.quantile(cal, self.confidence))
        if self.threshold <= 0:  # 分數可能為負（如 decision_function 取負），平移正規化
            self._shift = -float(cal.min()) + 1e-9
            self.threshold = float(np.quantile(cal + self._shift, self.confidence))
        else:
            self._shift = 0.0
        return self

    def score(self, X_raw: np.ndarray) -> np.ndarray:
        raw = self._raw_score(self.scaler.transform(np.atleast_2d(X_raw))) + self._shift
        return raw / max(self.threshold, np.finfo(float).eps)


class MahalanobisLW(_Base):
    """主線方法：逐類 Ledoit–Wolf 馬氏（健康-only 時即單類）。"""

    name = "maha_ledoit_wolf"

    def fit(self, X_train_raw, X_cal_raw):
        self.scaler = RobustScaler().fit(X_train_raw)
        self._det = MahalanobisOpenSetDetector(method="ledoit_wolf", confidence=self.confidence)
        n_tr, n_ca = len(X_train_raw), len(X_cal_raw)
        self._det.fit(self.scaler.transform(X_train_raw), np.zeros(n_tr),
                      self.scaler.transform(X_cal_raw), np.zeros(n_ca))
        return self

    def score(self, X_raw):
        return self._det.score_samples(self.scaler.transform(np.atleast_2d(X_raw)))


class MahalanobisLegacy(MahalanobisLW):
    """論文版對照：全域經驗共變異數 + pinv + 訓練集自身 95 百分位。"""

    name = "maha_legacy"

    def fit(self, X_train_raw, X_cal_raw):
        self.scaler = RobustScaler().fit(X_train_raw)
        self._det = MahalanobisOpenSetDetector(method="legacy", confidence=self.confidence)
        n_tr, n_ca = len(X_train_raw), len(X_cal_raw)
        self._det.fit(self.scaler.transform(X_train_raw), np.zeros(n_tr),
                      self.scaler.transform(X_cal_raw), np.zeros(n_ca))
        return self


class OneClassSVMDet(_Base):
    name = "ocsvm"

    def _fit_raw(self, X):
        from sklearn.svm import OneClassSVM

        self._m = OneClassSVM(nu=0.05, gamma="scale").fit(X)

    def _raw_score(self, X):
        return -self._m.decision_function(X)


class IsolationForestDet(_Base):
    name = "iforest"

    def _fit_raw(self, X):
        from sklearn.ensemble import IsolationForest

        self._m = IsolationForest(n_estimators=200, random_state=self.seed).fit(X)

    def _raw_score(self, X):
        return -self._m.score_samples(X)


class LOFDet(_Base):
    name = "lof"

    def _fit_raw(self, X):
        from sklearn.neighbors import LocalOutlierFactor

        self._m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X)

    def _raw_score(self, X):
        return -self._m.decision_function(X)


class KNNDistanceDet(_Base):
    name = "knn_dist"

    def _fit_raw(self, X):
        from sklearn.neighbors import NearestNeighbors

        self._m = NearestNeighbors(n_neighbors=5).fit(X)

    def _raw_score(self, X):
        dist, _ = self._m.kneighbors(X)
        return dist.mean(axis=1)


class PCAReconDet(_Base):
    """PCA 重建誤差（線性 autoencoder 的等價物）。"""

    name = "pca_recon"

    def _fit_raw(self, X):
        from sklearn.decomposition import PCA

        self._m = PCA(n_components=0.95, svd_solver="full", random_state=self.seed).fit(X)

    def _raw_score(self, X):
        Z = self._m.inverse_transform(self._m.transform(X))
        return np.linalg.norm(X - Z, axis=1)


class MLPAutoencoderDet(_Base):
    """非線性 autoencoder 重建誤差（瓶頸 MLP，對照 PCAReconDet 的線性版本）。"""

    name = "mlp_autoencoder"

    def _fit_raw(self, X):
        from sklearn.neural_network import MLPRegressor

        self._m = MLPRegressor(
            hidden_layer_sizes=(64, 16, 64),
            activation="relu",
            random_state=self.seed,
            max_iter=2000,
            early_stopping=True,
        ).fit(X, X)

    def _raw_score(self, X):
        return np.linalg.norm(X - self._m.predict(X), axis=1)


ALL_DETECTORS = [
    MahalanobisLW, MahalanobisLegacy, OneClassSVMDet,
    IsolationForestDet, LOFDet, KNNDistanceDet, PCAReconDet, MLPAutoencoderDet,
]
