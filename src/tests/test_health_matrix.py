import unittest

from experiments.health_index_matrix import expected_run_ids


class HealthMatrixTests(unittest.TestCase):
    def test_default_matrix_has_three_seeds_and_two_methods(self):
        run_ids = expected_run_ids()
        self.assertEqual(len(run_ids), 6)
        self.assertIn("seed_42_mahalanobis", run_ids)
        self.assertIn("seed_2026_knn", run_ids)

    def test_method_aliases_are_canonicalized(self):
        self.assertEqual(expected_run_ids([7], ["maha", "k-nn"]), ["seed_7_mahalanobis", "seed_7_knn"])

    def test_invalid_method_is_rejected(self):
        with self.assertRaises(ValueError):
            expected_run_ids([1], ["invalid"])
