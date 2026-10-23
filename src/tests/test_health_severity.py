import unittest

from experiments.health.calibration import HealthIndexCalibrator
from experiments.health.severity import DEFAULT_SEVERITY_POLICY, RelativeSeverityPolicy
from experiments.health.trajectory import TrajectoryConfig


class RelativeSeverityTests(unittest.TestCase):
    def test_policy_is_monotone_and_explicitly_relative(self):
        policy = RelativeSeverityPolicy()
        self.assertEqual(policy.classify(0.95), "healthy")
        self.assertEqual(policy.classify(0.5), "healthy")
        self.assertEqual(policy.classify(0.45), "early_warning")
        self.assertEqual(policy.classify(0.3), "degraded")
        self.assertEqual(policy.classify(0.05), "critical")
        metadata = policy.metadata()
        self.assertEqual(metadata["basis"], "relative_calibrated")
        self.assertFalse(metadata["ordinal_labels_available"])

    def test_defaults_line_up_with_calibrator_and_alarm_policy(self):
        """healthy 界線＝拒絕線映到的 health；degraded 界線＝critical 告警進入線。"""
        calibrator = HealthIndexCalibrator.fit([0.1, 0.5, 0.9])
        self.assertEqual(DEFAULT_SEVERITY_POLICY.healthy_min, calibrator.warning_health)
        self.assertEqual(DEFAULT_SEVERITY_POLICY.healthy_min, TrajectoryConfig().warning_enter)
        self.assertEqual(DEFAULT_SEVERITY_POLICY.degraded_min, TrajectoryConfig().critical_enter)
        self.assertEqual(DEFAULT_SEVERITY_POLICY.classify(calibrator.transform(calibrator.warning_anchor)), "healthy")
        self.assertEqual(DEFAULT_SEVERITY_POLICY.classify(calibrator.transform(calibrator.critical_anchor)), "critical")

    def test_invalid_order_is_rejected(self):
        with self.assertRaises(ValueError):
            RelativeSeverityPolicy(healthy_min=0.5, early_warning_min=0.8)

    def test_out_of_range_health_is_rejected(self):
        with self.assertRaises(ValueError):
            RelativeSeverityPolicy().classify(1.1)
