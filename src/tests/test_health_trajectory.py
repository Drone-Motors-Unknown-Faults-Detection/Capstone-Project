import unittest

import numpy as np

from experiments.health.schema import HealthMonitoringResult
from experiments.health.trajectory import SessionTrajectoryMonitor, TrajectoryConfig


def _result(health: float, timestamp: str) -> HealthMonitoringResult:
    return HealthMonitoringResult(
        is_fault=health < 0.5,
        health_index=health,
        degradation_score=1.0 - health,
        severity_stage="healthy" if health >= 0.8 else "early_warning",
        trend="insufficient_history",
        degradation_rate=None,
        fault_type="known",
        fault_type_confidence=0.8,
        openset_method="mahalanobis",
        openset_score=1.0 - health,
        is_unknown_fault=False,
        prediction_confidence=0.8,
        uncertainty=0.2,
        estimated_rul=None,
        rul_available=False,
        timestamp=timestamp,
        condition="T1/8000rpm",
        data_quality="valid",
    )


class SessionTrajectoryTests(unittest.TestCase):
    def test_missing_identity_does_not_create_shared_history(self):
        def predictor(features, **kwargs):
            return _result(0.4, kwargs["timestamp"])
        monitor = SessionTrajectoryMonitor(predictor)
        result = monitor.update(
            np.zeros(3),
            motor_id=None,
            session_id=None,
            timestamp="t0",
            condition="T1/8000rpm",
        )
        self.assertEqual(result.data_quality, "insufficient")
        self.assertEqual(result.trend, "insufficient_history")
        self.assertEqual(monitor.session_keys(), ())

    def test_sessions_are_isolated_and_worsening_trend_is_detected(self):
        def predictor(features, **kwargs):
            return _result(float(features[0]), kwargs["timestamp"])
        monitor = SessionTrajectoryMonitor(
            predictor,
            config=TrajectoryConfig(
                min_history=3,
                ewma_alpha=0.8,
                enter_persistence=2,
                clear_persistence=2,
            ),
        )
        first_session = []
        for idx, health in enumerate((0.9, 0.7, 0.4, 0.2, 0.1, 0.05)):
            first_session.append(
                monitor.update(
                    np.array([health]),
                    motor_id="motor-a",
                    session_id="session-1",
                    timestamp=f"t{idx}",
                    condition="T1/8000rpm",
                )
            )
        self.assertEqual(first_session[-1].trend, "worsening")
        self.assertIn(first_session[-1].alarm_state, {"warning", "critical"})
        self.assertEqual(first_session[-1].smoothed_health_index, first_session[-1].health_index)
        self.assertEqual(len(monitor.history("motor-a", "session-1")), 6)
        other = monitor.update(
            np.array([0.9]),
            motor_id="motor-a",
            session_id="session-2",
            timestamp="t0",
            condition="T1/8000rpm",
        )
        self.assertEqual(other.trend, "insufficient_history")
        self.assertEqual(len(monitor.history("motor-a", "session-2")), 1)

    def test_change_point_requires_persistent_drop(self):
        def predictor(features, **kwargs):
            return _result(float(features[0]), kwargs["timestamp"])
        monitor = SessionTrajectoryMonitor(
            predictor,
            config=TrajectoryConfig(
                min_history=2,
                ewma_alpha=1.0,
                change_point_delta=0.2,
                change_point_persistence=2,
            ),
        )
        states = []
        for idx, health in enumerate((0.9, 0.9, 0.5, 0.1, 0.1)):
            result = monitor.update(
                np.array([health]),
                motor_id="m",
                session_id="change",
                timestamp=str(idx),
                condition="x",
            )
            states.append(result.change_point_state)
        self.assertEqual(states[3], "candidate")
        self.assertEqual(states[4], "confirmed")

    def test_alarm_hysteresis_requires_persistent_clear(self):
        def predictor(features, **kwargs):
            return _result(float(features[0]), kwargs["timestamp"])
        monitor = SessionTrajectoryMonitor(
            predictor,
            config=TrajectoryConfig(
                min_history=2,
                ewma_alpha=0.8,
                enter_persistence=2,
                clear_persistence=2,
            ),
        )
        for idx, health in enumerate((0.1, 0.1, 0.4, 0.4)):
            result = monitor.update(
                np.array([health]),
                motor_id="m",
                session_id="s",
                timestamp=str(idx),
                condition="x",
            )
        self.assertEqual(result.alarm_state, "critical")
        for idx, health in enumerate((0.9, 0.9, 0.9), start=4):
            result = monitor.update(
                np.array([health]),
                motor_id="m",
                session_id="s",
                timestamp=str(idx),
                condition="x",
            )
        self.assertEqual(result.alarm_state, "normal")
