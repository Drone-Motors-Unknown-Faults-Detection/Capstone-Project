"""Regression tests for PolarMap's fixed Mahalanobis geometry."""

from __future__ import annotations

import unittest

import numpy as np

from core.geometry import PolarMap
from core.monitor import OpenSetMonitor


def _pools(seed: int = 9) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "8screws": rng.normal(0.0, 0.4, size=(60, 8)),
        "7screws": rng.normal(2.0, 0.4, size=(60, 8)),
    }


class PolarMapTests(unittest.TestCase):
    def test_switching_openset_detector_does_not_change_mahalanobis_geometry(self) -> None:
        pools = _pools()
        maha = OpenSetMonitor(pools, seed=42, openset_method="mahalanobis")
        maha.fit_initial()
        maha.add_class("7screws")
        knn = OpenSetMonitor(pools, seed=42, openset_method="knn")
        knn.fit_initial()
        knn.add_class("7screws")
        map_maha = PolarMap(maha)
        map_knn = PolarMap(knn)
        probe = pools["7screws"][:7]
        np.testing.assert_allclose(map_maha.radius(probe), map_knn.radius(probe), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(map_maha.cosines(probe), map_knn.cosines(probe), rtol=1e-10, atol=1e-10)
        self.assertEqual(map_maha.summary()["polarmap_base_method"], "mahalanobis")
        self.assertEqual(map_knn.summary()["polarmap_base_method"], "mahalanobis")

    def test_ledoit_wolf_monitor_detector_is_reused_and_other_methods_refit_identically(self) -> None:
        """同 seed 的 LW 與 OAS monitor 切分與 scaler 相同：沿用的 detector 與另擬合的 LW 射線逐位元相同。"""
        pools = _pools()
        lw = OpenSetMonitor(pools, seed=42)
        lw.fit_initial()
        lw.add_class("7screws")
        reused = PolarMap(lw)
        self.assertIs(reused._base_detector, lw.detector)
        oas = OpenSetMonitor(pools, seed=42, method="oas")
        oas.fit_initial()
        oas.add_class("7screws")
        refit = PolarMap(oas)
        self.assertIsNot(refit._base_detector, oas.detector)
        self.assertEqual(refit._base_detector.method, "ledoit_wolf")
        probe = pools["7screws"][:7]
        np.testing.assert_array_equal(reused.radius(probe), refit.radius(probe))
        np.testing.assert_array_equal(reused.cosines(probe), refit.cosines(probe))
        self.assertEqual([(r.config, r.radius, r.tau) for r in reused.rays],
                         [(r.config, r.radius, r.tau) for r in refit.rays])

    def test_health_only_map_has_no_fault_ray(self) -> None:
        monitor = OpenSetMonitor({"8screws": _pools()["8screws"]}, seed=42)
        monitor.fit_initial()
        geometry = PolarMap(monitor)
        self.assertEqual(geometry.rays, [])
        self.assertEqual(geometry.analyze(np.zeros((1, 8)))[0]["verdict"], "new_direction")


if __name__ == "__main__":
    unittest.main()
