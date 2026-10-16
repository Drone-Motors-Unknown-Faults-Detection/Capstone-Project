import unittest

from experiments.health.diagnosis import DiagnosisResolver


class DiagnosisResolverTests(unittest.TestCase):
    def test_unknown_score_has_priority_over_any_mapping(self):
        resolver = DiagnosisResolver({0: "bearing_wear"})
        decision = resolver.resolve(0, 1.01, 0.9)
        self.assertEqual(decision.fault_type, "unknown")
        self.assertTrue(decision.is_unknown)
        self.assertIsNone(decision.confidence)

    def test_missing_taxonomy_is_uncertain_not_a_fake_class(self):
        decision = DiagnosisResolver().resolve(0, 0.4, 0.8)
        self.assertEqual(decision.fault_type, "uncertain")
        self.assertFalse(decision.is_unknown)
        self.assertIsNone(decision.confidence)

    def test_versioned_mapping_can_name_a_supported_class(self):
        decision = DiagnosisResolver({2: "bearing_wear"}).resolve(2, 0.4, 0.8)
        self.assertEqual(decision.fault_type, "bearing_wear")
        self.assertEqual(decision.confidence, 0.8)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            DiagnosisResolver().resolve(0, -0.1, 0.8)
        with self.assertRaises(ValueError):
            DiagnosisResolver().resolve(0, 0.2, 1.1)
