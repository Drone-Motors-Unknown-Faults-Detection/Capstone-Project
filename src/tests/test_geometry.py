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

    def test_health_only_map_has_no_fault_ray(self) -> None:
        monitor = OpenSetMonitor({"8screws": _pools()["8screws"]}, seed=42)
        monitor.fit_initial()
        geometry = PolarMap(monitor)
        self.assertEqual(geometry.rays, [])
        self.assertEqual(geometry.analyze(np.zeros((1, 8)))[0]["verdict"], "new_direction")


if __name__ == "__main__":
    unittest.main()
