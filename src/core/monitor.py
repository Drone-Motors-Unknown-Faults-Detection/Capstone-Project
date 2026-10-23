"""OpenSetMonitor — 從健康基準冷啟動、可逐類擴張的開放集監測器。

階段 0 只以 8screws 擬合單一類別；之後每確認一種新故障呼叫
add_class()，整組重新擬合。Open Set detector 可選既有逐類 Mahalanobis 或
逐類 k-NN 距離；兩者皆以 calibration split 正規化，分數 > 1 = 未知。
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from core.data import HEALTHY, Split, make_split
from core.mahalanobis import LEGACY_POOLED_LABEL
from core.openset import OpenSetDetector, create_openset_detector, resolve_openset_config


class OpenSetMonitor:
    def __init__(
        self,
        pools: dict[str, np.ndarray],
        seed: int = 42,
        confidence: float = 0.95,
        method: str = "ledoit_wolf",
        openset_method: str = "mahalanobis",
        knn_neighbors: int = 5,
    ) -> None:
        self.resolved_config = resolve_openset_config(
            openset_method, confidence=confidence, mahalanobis_method=method, knn_neighbors=knn_neighbors)
        self.pools = pools
        self.confidence = confidence
        self.openset_method = self.resolved_config["openset_method"]
        self.method = method
        self.knn_neighbors = knn_neighbors
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.known: dict[str, int] = {}
        self.splits: dict[str, Split] = {}
        self.observed: dict[str, np.ndarray] = {}
        self.scaler: RobustScaler | None = None
        self.detector: OpenSetDetector | None = None
        self.pca: PCA | None = None
        self.centroids: dict[str, tuple[float, float]] = {}
        self._label_to_config: dict[int, str] = {}
        self._fitted = False

    # -- 擬合 ------------------------------------------------------------------

    def fit_initial(self, healthy: str = HEALTHY) -> None:
        """階段 0：只認識健康。"""
        self._fitted = False
        self.known = {}
        self.splits = {}
        self.observed = {}
        self._register(healthy)
        self._refit()

    def add_class(self, config: str) -> int:
        """把一種已確認的故障納入已知，重新擬合整組分布（量尺擴張）。"""
        label = self._register(config)
        self._refit()
        return label

    def add_classes(self, configs: list[str]) -> list[int]:
        """一次註冊多個已知配置、只重擬合一次；結果與逐一 add_class 相同（切分 rng 順序不變）。"""
        labels = [self._register(config) for config in configs]
        if labels:
            self._refit()
        return labels

    def add_observed(self, name: str, samples: np.ndarray) -> int:
        """只用這批已到達樣本註冊新類並重擬合，不讀該配置的完整資料池。"""
        rows = np.atleast_2d(np.asarray(samples, dtype=float))
        if len(rows) < 5:
            raise ValueError("發現樣本少於 5 筆，無法切出訓練與校準")
        if name in self.known:
            raise ValueError(f"{name} 已是已知類別")
        label = len(self.known)
        self.known[name] = label
        self.observed[name] = rows
        self.splits[name] = make_split(len(rows), self.rng)
        self._refit()
        return label

    def _pool(self, config: str) -> np.ndarray:
        if config in self.observed:
            return self.observed[config]
        return self.pools[config]

    def _register(self, config: str) -> int:
        if config not in self.pools:
            raise KeyError(f"資料池沒有配置 {config}")
        if config in self.known:
            raise ValueError(f"{config} 已是已知類別")
        label = len(self.known)
        self.known[config] = label
        self.splits[config] = make_split(len(self.pools[config]), self.rng)
        return label

    def _refit(self) -> None:
        # 105 維小矩陣在多執行緒 BLAS 下反而慢數十倍；單執行緒的數字逐位元相同
        with threadpool_limits(limits=1):
            self._refit_single_thread()

    def _refit_single_thread(self) -> None:
        self._fitted = False
        X_tr, y_tr, X_ca, y_ca = [], [], [], []
        for config, label in self.known.items():
            pool, sp = self._pool(config), self.splits[config]
            X_tr.append(pool[sp.train])
            y_tr.append(np.full(len(sp.train), label))
            X_ca.append(pool[sp.cal])
            y_ca.append(np.full(len(sp.cal), label))
        X_tr = np.vstack(X_tr)
        y_tr = np.concatenate(y_tr)
        X_ca = np.vstack(X_ca)
        y_ca = np.concatenate(y_ca)

        self.scaler = RobustScaler().fit(X_tr)
        X_tr_s = self.scaler.transform(X_tr)
        effective = self.resolved_config["effective"]
        self.detector = create_openset_detector(
            self.openset_method,
            confidence=effective["confidence"],
            mahalanobis_method=self.method,
            knn_neighbors=self.knn_neighbors,
            random_state=self.seed,  # MCD 與其 PCA 的亂數來源跟著 seed 走（鐵則 7）
        ).fit(X_tr_s, y_tr, self.scaler.transform(X_ca), y_ca)

        self.pca = PCA(n_components=2, random_state=0).fit(X_tr_s)
        self.centroids = {
            config: tuple(
                float(v)
                for v in self.pca.transform(X_tr_s[y_tr == label].mean(axis=0, keepdims=True))[0]
            )
            for config, label in self.known.items()
        }
        self._label_to_config = {label: config for config, label in self.known.items()}
        self._fitted = True

    # -- 推論 ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("模型尚未完整擬合，請先成功呼叫 fit_initial()。")

    def score(self, X_raw: np.ndarray) -> np.ndarray:
        """正規化開集分數；> 1 視為未知。"""
        self._require_fitted()
        return self.detector.score_samples(self.scaler.transform(np.atleast_2d(X_raw)))

    def _configs(self, labels: np.ndarray) -> list[str | None]:
        return [self._label_to_config.get(int(label)) if label >= 0 else None for label in labels]

    def classify(self, X_raw: np.ndarray) -> list[str | None]:
        """回傳最近的已知配置名稱；未知回傳 None。"""
        self._require_fitted()
        return self._configs(self.detector.predict_known_class(self.scaler.transform(np.atleast_2d(X_raw))))

    def score_and_classify(self, X_raw: np.ndarray) -> tuple[np.ndarray, list[str | None]]:
        """同時回傳分數與類別；距離只算一次，結果與 score()＋classify() 相同。"""
        self._require_fitted()
        scores, labels = self.detector.score_and_predict(self.scaler.transform(np.atleast_2d(X_raw)))
        return scores, self._configs(labels)

    def project(self, X_raw: np.ndarray) -> np.ndarray:
        """投影到已知資料擬合出的 PCA 平面（僅供視覺化）。"""
        self._require_fitted()
        return self.pca.transform(self.scaler.transform(np.atleast_2d(X_raw)))

    def holdout(self, config: str) -> np.ndarray:
        """已知配置的保留集（未參與擬合與校準）。"""
        return self._pool(config)[self.splits[config].holdout]

    def calibration(self, config: str) -> np.ndarray:
        """已知配置的校準集（定門檻用，未參與擬合）。"""
        return self._pool(config)[self.splits[config].cal]

    def summary(self) -> dict:
        """目前擬合狀態的可序列化摘要（保存用：逐類閾值與樣本數）。"""
        self._require_fitted()
        legacy = self.openset_method == "mahalanobis" and self.method == "legacy"
        classes = []
        for item in self.detector.class_summaries():
            label = int(item["label"])
            if label == LEGACY_POOLED_LABEL:  # legacy：全部已知類合併成單一分布
                config = "pooled_known"
                splits = [self.splits[c] for c in self.known]
            else:
                config = self._label_to_config[label]
                splits = [self.splits[config]]
            row = {
                "config": config,
                "label": label,
                "n_train": sum(len(sp.train) for sp in splits),
                "n_cal": sum(len(sp.cal) for sp in splits),
                "n_holdout": sum(len(sp.holdout) for sp in splits),
                "threshold": float(item["threshold"]),
                "centroid_pca": list(self.centroids.get(config, ())),
            }
            row.update({k: v for k, v in item.items() if k not in {"label", "threshold"}})
            classes.append(row)
        if legacy:  # 門檻量在估分布的同一批 train 距離上，不讀校準集；已知列再取縮放空間中最近的類別 train 中心
            score_type = "normalized_pooled_mahalanobis"
            threshold_strategy = f"known_train_pooled_distance_quantile_{self.confidence:.4g}"
            calibration_source = "known-only 60% train split (legacy: threshold on training distances)"
        else:
            score_type = ("normalized_class_mahalanobis" if self.openset_method == "mahalanobis"
                          else "normalized_class_knn_mean_euclidean")
            threshold_strategy = f"known_calibration_class_quantile_{self.confidence:.4g}"
            calibration_source = "known-only 20% calibration split"
        return {
            "resolved_config": self.resolved_config,
            "openset_method": self.openset_method,
            "method": self.method,
            "score_type": score_type,
            "threshold": 1.0,
            "threshold_strategy": threshold_strategy,
            "calibration_source": calibration_source,
            "confidence": self.confidence,
            "knn_neighbors": self.knn_neighbors if self.openset_method == "knn" else None,
            "random_seed": self.seed,
            "split": {"train": 0.6, "calibration": 0.2, "holdout": 0.2},
            "n_known": len(self.known),
            "classes": classes,
        }
