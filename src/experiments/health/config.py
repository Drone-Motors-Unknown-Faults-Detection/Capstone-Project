"""Explicit configuration for health output and future stream monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OutputMode = Literal["binary", "health", "full"]


@dataclass(frozen=True)
class HealthMonitorConfig:
    """No hidden global history: identity and output policy are explicit."""

    output_mode: OutputMode = "binary"
    openset_method: Literal["mahalanobis", "knn"] = "mahalanobis"
    mahalanobis_method: Literal["legacy", "ledoit_wolf", "oas", "mcd"] = "ledoit_wolf"
    confidence: float = 0.95
    knn_neighbors: int = 5
    enable_history: bool = False
    require_motor_session_identity: bool = True
    rul_enabled: bool = False

    def __post_init__(self) -> None:
        if self.output_mode not in {"binary", "health", "full"}:
            raise ValueError("output_mode must be binary, health or full")
        if self.openset_method not in {"mahalanobis", "knn"}:
            raise ValueError("openset_method must be mahalanobis or knn")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.knn_neighbors < 1:
            raise ValueError("knn_neighbors must be at least one")
        if self.rul_enabled:
            raise ValueError("RUL is disabled until run-to-failure data is registered")
