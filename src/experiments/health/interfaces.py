"""Protocols for window prediction and per-session sequence monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from experiments.health.schema import HealthMonitoringResult


class WindowHealthPredictor(Protocol):
    def predict_window(
        self,
        features: np.ndarray,
        *,
        timestamp: str | None = None,
        condition: str,
    ) -> HealthMonitoringResult: ...


class SessionHealthMonitor(Protocol):
    def update(
        self,
        features: np.ndarray,
        *,
        motor_id: str | None,
        session_id: str | None,
        timestamp: str | None,
        condition: str,
    ) -> HealthMonitoringResult: ...

    def reset_session(self, motor_id: str, session_id: str) -> None: ...

    def history(self, motor_id: str, session_id: str) -> Sequence[HealthMonitoringResult]: ...

    def save_calibration(self, path: Path) -> None: ...

    @classmethod
    def load_calibration(cls, path: Path) -> "SessionHealthMonitor": ...
