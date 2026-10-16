import unittest

from experiments.health.config import HealthMonitorConfig
from experiments.health.schema import HealthMonitoringResult


def _result(**overrides):
    payload = {
        "is_fault": True,
        "health_index": 0.42,
        "degradation_score": 0.58,
        "severity_stage": "degraded",
        "trend": "worsening",
        "degradation_rate": 0.013,
        "fault_type": "uncertain",
        "fault_type_confidence": None,
        "openset_method": "mahalanobis",
        "openset_score": 2.81,
        "is_unknown_fault": False,
        "prediction_confidence": 0.81,
        "uncertainty": 0.19,
        "estimated_rul": None,
        "rul_available": False,
        "timestamp": "2026-09-20T00:00:00Z",
        "condition": "T1/8000rpm",
        "data_quality": "valid",
    }
    payload.update(overrides)
    return HealthMonitoringResult(**payload)


class HealthSchemaTests(unittest.TestCase):
    def test_binary_mode_preserves_legacy_shape(self):
        self.assertEqual(_result().to_dict("binary"), {"is_fault": 1})

    def test_health_mode_contains_relative_contract(self):
        payload = _result().to_dict("health")
        self.assertEqual(payload["is_fault"], 1)
        self.assertEqual(payload["degradation_score"], 0.58)
        self.assertNotIn("estimated_rul", payload)

    def test_full_mode_contains_rul_disabled_explicitly(self):
        payload = _result().to_dict("full")
        self.assertIsNone(payload["estimated_rul"])
        self.assertFalse(payload["rul_available"])

    def test_unknown_cannot_claim_known_fault_type(self):
        with self.assertRaises(ValueError):
            _result(is_unknown_fault=True, fault_type="bearing_wear")

    def test_health_and_degradation_must_be_complements(self):
        with self.assertRaises(ValueError):
            _result(degradation_score=0.5)

    def test_config_preserves_mahalanobis_default_and_disables_rul(self):
        config = HealthMonitorConfig(output_mode="full")
        self.assertEqual(config.openset_method, "mahalanobis")
        self.assertEqual(config.mahalanobis_method, "ledoit_wolf")
        with self.assertRaises(ValueError):
            HealthMonitorConfig(rul_enabled=True)
