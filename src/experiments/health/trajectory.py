"""Per-session smoothing, trend and alarm state for health results."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from experiments.health.index import relative_severity
from experiments.health.schema import HealthMonitoringResult


@dataclass(frozen=True)
class TrajectoryConfig:
    """Explicit policy for temporal monitoring.

    Thresholds operate on the *relative* health index.  They are not physical
    severity labels because the current formal data has no run-to-failure or
    ordinal damage annotation.
    """

    history_window: int = 120
    min_history: int = 5
    ewma_alpha: float = 0.2
    stable_slope: float = 0.005
    warning_enter: float = 0.5
    warning_clear: float = 0.6
    critical_enter: float = 0.2
    critical_clear: float = 0.3
    enter_persistence: int = 3
    clear_persistence: int = 3
    change_point_delta: float = 0.15
    change_point_persistence: int = 2

    def __post_init__(self) -> None:
        if self.history_window < 2:
            raise ValueError("history_window must be at least 2")
        if self.min_history < 2 or self.min_history > self.history_window:
            raise ValueError("min_history must be in [2, history_window]")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.stable_slope < 0.0:
            raise ValueError("stable_slope must be non-negative")
        for name in ("warning_enter", "warning_clear", "critical_enter", "critical_clear"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not self.critical_enter < self.warning_enter <= self.warning_clear:
            raise ValueError("alarm thresholds must satisfy critical_enter < warning_enter <= warning_clear")
        if not self.critical_enter < self.critical_clear <= 1.0:
            raise ValueError("critical_clear must exceed critical_enter")
        if self.enter_persistence < 1 or self.clear_persistence < 1:
            raise ValueError("persistence values must be positive")
        if not 0.0 < self.change_point_delta <= 1.0:
            raise ValueError("change_point_delta must be in (0, 1]")
        if self.change_point_persistence < 1:
            raise ValueError("change_point_persistence must be positive")


@dataclass
class _SessionState:
    results: deque[HealthMonitoringResult]
    smoothed: deque[float]
    alarm_state: str = "normal"
    candidate_streak: int = 0
    clear_streak: int = 0
    change_point_streak: int = 0


class SessionTrajectoryMonitor:
    """Keep independent temporal state for each ``(motor_id, session_id)``.

    ``predictor`` must expose ``predict_one(features, timestamp=..., condition=...)``
    and return a :class:`HealthMonitoringResult`.  State is instance-local and
    never shared across monitors.
    """

    def __init__(
        self,
        predictor: Callable[..., HealthMonitoringResult] | object,
        *,
        config: TrajectoryConfig | None = None,
        require_identity: bool = True,
    ) -> None:
        self.predictor = predictor
        self.config = config or TrajectoryConfig()
        self.require_identity = bool(require_identity)
        self._sessions: dict[tuple[str, str], _SessionState] = {}

    def _predict(
        self,
        features: np.ndarray,
        *,
        timestamp: str | None,
        condition: str,
    ) -> HealthMonitoringResult:
        fn = getattr(self.predictor, "predict_one", self.predictor)
        result = fn(features, timestamp=timestamp, condition=condition)
        if not isinstance(result, HealthMonitoringResult):
            raise TypeError("predictor must return HealthMonitoringResult")
        return result

    def update(
        self,
        features: np.ndarray,
        *,
        motor_id: str | None,
        session_id: str | None,
        timestamp: str | None,
        condition: str,
    ) -> HealthMonitoringResult:
        raw = self._predict(features, timestamp=timestamp, condition=condition)
        if not motor_id or not session_id:
            if self.require_identity:
                return replace(
                    raw,
                    trend="insufficient_history",
                    degradation_rate=None,
                    data_quality="insufficient",
                    alarm_state="normal",
                    alarm_reason="motor_id and session_id are required for temporal state",
                )
            return raw
        return self.observe(raw, motor_id=motor_id, session_id=session_id)

    def observe(
        self,
        result: HealthMonitoringResult,
        *,
        motor_id: str,
        session_id: str,
    ) -> HealthMonitoringResult:
        if not motor_id or not session_id:
            raise ValueError("motor_id and session_id must be non-empty")
        key = (str(motor_id), str(session_id))
        state = self._sessions.setdefault(
            key,
            _SessionState(
                results=deque(maxlen=self.config.history_window),
                smoothed=deque(maxlen=self.config.history_window),
            ),
        )
        prior_smoothed = state.smoothed[-1] if state.smoothed else None
        # A short rolling median suppresses one-window spikes before EWMA.
        recent_raw = [item.health_index for item in list(state.results)[-2:]]
        recent_raw.append(float(result.health_index))
        robust_health = float(np.median(recent_raw))
        smoothed = robust_health if prior_smoothed is None else (
            self.config.ewma_alpha * robust_health
            + (1.0 - self.config.ewma_alpha) * prior_smoothed
        )
        state.results.append(result)
        state.smoothed.append(smoothed)

        if prior_smoothed is not None and prior_smoothed - robust_health >= self.config.change_point_delta:
            state.change_point_streak += 1
        else:
            state.change_point_streak = 0
        if state.change_point_streak >= self.config.change_point_persistence:
            change_point_state = "confirmed"
        elif state.change_point_streak > 0:
            change_point_state = "candidate"
        else:
            change_point_state = "none"

        if len(state.smoothed) < self.config.min_history:
            trend = "insufficient_history"
            rate = None
        else:
            values = np.asarray(state.smoothed, dtype=float)
            x = np.arange(len(values), dtype=float)
            slope = float(np.polyfit(x, values, 1)[0])
            rate = float(-slope)
            if rate > self.config.stable_slope:
                trend = "worsening"
            elif rate < -self.config.stable_slope:
                trend = "recovering"
            else:
                trend = "stable"

        candidate = "critical" if smoothed < self.config.critical_enter else (
            "warning" if smoothed < self.config.warning_enter else "normal"
        )
        if state.alarm_state == "critical":
            # Critical alarms clear in two stages: persistent recovery above
            # critical_clear first downgrades to warning, then warning_clear
            # clears the warning.  This prevents a single healthy spike from
            # hiding an active critical episode.
            if smoothed >= self.config.critical_clear:
                state.clear_streak += 1
                if state.clear_streak >= self.config.clear_persistence:
                    state.alarm_state = "normal" if smoothed >= self.config.warning_clear else "warning"
                    state.clear_streak = 0
            else:
                state.clear_streak = 0
            state.candidate_streak = 0
        elif state.alarm_state == "warning":
            if smoothed >= self.config.warning_clear:
                state.clear_streak += 1
                if state.clear_streak >= self.config.clear_persistence:
                    state.alarm_state = "normal"
                    state.clear_streak = 0
            else:
                state.clear_streak = 0
            state.candidate_streak = 0
        elif candidate != "normal":
            state.clear_streak = 0
            state.candidate_streak += 1
            if state.candidate_streak >= self.config.enter_persistence:
                state.alarm_state = candidate
                state.candidate_streak = 0
        else:
            state.candidate_streak = 0
            state.clear_streak = 0

        alarm_reason = None
        if state.alarm_state != "normal":
            alarm_reason = (
                f"relative health below {state.alarm_state} threshold; "
                f"trend={trend}"
            )
        return replace(
            result,
            health_index=smoothed,
            degradation_score=1.0 - smoothed,
            severity_stage=relative_severity(smoothed),
            trend=trend,
            degradation_rate=rate,
            alarm_state=state.alarm_state,  # type: ignore[arg-type]
            alarm_reason=alarm_reason,
            data_quality="valid",
            raw_health_index=result.raw_health_index if result.raw_health_index is not None else result.health_index,
            smoothed_health_index=smoothed,
            change_point_state=change_point_state,  # type: ignore[arg-type]
        )

    def history(self, motor_id: str, session_id: str) -> tuple[HealthMonitoringResult, ...]:
        state = self._sessions.get((str(motor_id), str(session_id)))
        return tuple(state.results) if state else ()

    def reset_session(self, motor_id: str, session_id: str) -> None:
        self._sessions.pop((str(motor_id), str(session_id)), None)

    def session_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._sessions))
