import unittest

from experiments.health.severity import RelativeSeverityPolicy


class RelativeSeverityTests(unittest.TestCase):
    def test_policy_is_monotone_and_explicitly_relative(self):
        policy = RelativeSeverityPolicy()
        self.assertEqual(policy.classify(0.95), "healthy")
        self.assertEqual(policy.classify(0.65), "early_warning")
        self.assertEqual(policy.classify(0.35), "degraded")
        self.assertEqual(policy.classify(0.05), "critical")
        metadata = policy.metadata()
        self.assertEqual(metadata["basis"], "relative_calibrated")
        self.assertFalse(metadata["ordinal_labels_available"])

    def test_invalid_order_is_rejected(self):
        with self.assertRaises(ValueError):
            RelativeSeverityPolicy(healthy_min=0.5, early_warning_min=0.8)

    def test_out_of_range_health_is_rejected(self):
        with self.assertRaises(ValueError):
            RelativeSeverityPolicy().classify(1.1)
