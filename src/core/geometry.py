"""極座標健康地圖（實驗四核心）：方向 = 故障類型、半徑 = 嚴重度。

幾何全部定義在**健康類的白化空間**：以健康類 Ledoit–Wolf precision P_H 的
Cholesky 分解取白化矩陣 W = chol(P_H)^T，對任意樣本（已標準化空間）x：

    z(x) = W (x − μ_H)            白化座標（健康中心為原點）
    r(x) = ‖z(x)‖                 半徑 = 到健康分布的馬氏距離（嚴重度的原始量）
    u_c  = z(μ_c) / ‖z(μ_c)‖      已知故障類 c 的「射線」方向
    cos_c(x) = ⟨z/‖z‖, u_c⟩       方向相似度

三分判定（疊在現行「已知/未知」之上）：
    - same_ray：max_c cos_c ≥ τ_c（該類 holdout cos 的低百分位校準）
      → 疑似已知故障 c 的惡化/變體；嚴重度 severity = r·cos_c / ‖z(μ_c)‖
      （= 沿射線投影長相對類中心半徑；類中心處 ≈ 1、健康 ≈ 0、更嚴重 > 1）
    - new_direction：所有 cos 皆低 → 走新類發現（隔離區）

幾何基準固定為馬氏距離：不論開集偵測採用 Mahalanobis 或 k-NN（monitor 的
--openset-method），PolarMap 一律在 monitor 已知類的 train/calibration split
上另行擬合 Ledoit–Wolf Mahalanobis，讓半徑與射線方向在方法比較實驗間可比。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.data import HEALTHY
from core.mahalanobis import MahalanobisOpenSetDetector


@dataclass(frozen=True)
class Ray:
    config: str
    direction: np.ndarray
    radius: float
    tau: float


class PolarMap:
    """由已擬合的 OpenSetMonitor 建立極座標幾何（基準固定為 Mahalanobis）。

    monitor 需已 fit（任意已知類數）；healthy 之外的每個已知類各成一條射線。
    僅健康一類時沒有射線：所有異常樣本判 new_direction、severity 為 None。
    """

    polarmap_base_method = "mahalanobis"

    def __init__(self, monitor, tau_percentile: float = 5.0, healthy: str = HEALTHY):
        if monitor.scaler is None or not monitor.known:
            raise RuntimeError("OpenSetMonitor must be fitted before constructing PolarMap")
        self.healthy = healthy
        self.tau_percentile = float(tau_percentile)
        self._scaler = monitor.scaler

        X_train, y_train, X_calibration, y_calibration = [], [], [], []
        for config, label in monitor.known.items():
            pool = monitor.pools[config]
            split = monitor.splits[config]
            X_train.append(pool[split.train])
            y_train.append(np.full(len(split.train), label, dtype=int))
            X_calibration.append(pool[split.cal])
            y_calibration.append(np.full(len(split.cal), label, dtype=int))
        X_train_scaled = self._scaler.transform(np.vstack(X_train))
        X_calibration_scaled = self._scaler.transform(np.vstack(X_calibration))
        self._base_detector = MahalanobisOpenSetDetector(
            method="ledoit_wolf", confidence=monitor.confidence, random_state=monitor.seed
        ).fit(
            X_train_scaled,
            np.concatenate(y_train),
            X_calibration_scaled,
            np.concatenate(y_calibration),
        )
        distributions = {
            monitor._label_to_config[int(item.label)]: item
            for item in self._base_detector.distributions_
        }
        if healthy not in distributions:
            raise ValueError(f"healthy configuration {healthy!r} is not known")
        health = distributions[healthy]
        self._mu_h = health.location
        self._W = np.linalg.cholesky(health.precision).T
        self.rays: list[Ray] = []
        for config, distribution in distributions.items():
            if config == healthy:
                continue
            vector = self._whiten_scaled(distribution.location.reshape(1, -1))[0]
            radius = float(np.linalg.norm(vector))
            if radius <= np.finfo(float).eps:
                continue
            direction = vector / radius
            cos_own = self._cosines(monitor.holdout(config), [direction])[:, 0]
            tau = float(np.percentile(cos_own, self.tau_percentile))
            self.rays.append(Ray(config, direction, radius, tau))

    def _whiten_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X_scaled) - self._mu_h) @ self._W.T

    def whiten_raw(self, X_raw: np.ndarray) -> np.ndarray:
        return self._whiten_scaled(self._scaler.transform(np.atleast_2d(X_raw)))

    def radius(self, X_raw: np.ndarray) -> np.ndarray:
        """Raw Mahalanobis radius from the health distribution."""
        return np.linalg.norm(self.whiten_raw(X_raw), axis=1)

    def _cosines(self, X_raw: np.ndarray, directions: list[np.ndarray]) -> np.ndarray:
        whitened = self.whiten_raw(X_raw)
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        unit = whitened / np.maximum(norms, np.finfo(float).eps)
        return unit @ np.column_stack(directions)

    def cosines(self, X_raw: np.ndarray) -> np.ndarray:
        if not self.rays:
            return np.zeros((len(np.atleast_2d(X_raw)), 0))
        return self._cosines(X_raw, [ray.direction for ray in self.rays])

    def analyze(self, X_raw: np.ndarray) -> list[dict]:
        """逐樣本極座標判讀（不含「是否異常」——那由 monitor 的開集分數決定）。

        回傳欄位：radius、best_ray（config 或 None）、best_cos、
        verdict（"same_ray"/"new_direction"）、severity（無射線時為 None）。
        """
        X_raw = np.atleast_2d(X_raw)
        radii = self.radius(X_raw)
        if not self.rays:
            return [
                {
                    "radius": float(radius),
                    "best_ray": None,
                    "best_cos": None,
                    "verdict": "new_direction",
                    "severity": None,
                }
                for radius in radii
            ]
        cos = self.cosines(X_raw)
        output = []
        for index, radius in enumerate(radii):
            ray_index = int(np.argmax(cos[index]))
            ray = self.rays[ray_index]
            best_cos = float(cos[index, ray_index])
            same_ray = best_cos >= ray.tau
            output.append(
                {
                    "radius": float(radius),
                    "best_ray": ray.config,
                    "best_cos": round(best_cos, 4),
                    "verdict": "same_ray" if same_ray else "new_direction",
                    "severity": round(float(radius * best_cos / ray.radius), 4),
                }
            )
        return output

    def ray_cos_matrix(self) -> tuple[list[str], np.ndarray]:
        """射線兩兩之間的 cos（星狀假設的直接量測）。"""
        names = [ray.config for ray in self.rays]
        if not self.rays:
            return names, np.empty((0, 0))
        directions = np.column_stack([ray.direction for ray in self.rays])
        return names, directions.T @ directions

    def summary(self) -> dict:
        return {
            "polarmap_base_method": self.polarmap_base_method,
            "tau_percentile": self.tau_percentile,
            "rays": [
                {"config": ray.config, "radius": round(ray.radius, 6), "tau": round(ray.tau, 6)}
                for ray in self.rays
            ],
        }
