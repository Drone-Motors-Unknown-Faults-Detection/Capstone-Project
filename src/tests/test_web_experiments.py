import json
import math
import unittest
from pathlib import Path

import numpy as np

from web.experiments import CATALOG, ROOT, ExperimentRunner, jsonable

FAKE_DATASETS = [
    {"motor": "T1", "rpm": "8000rpm", "path": "unused"},
    {"motor": "T2", "rpm": "6000rpm", "path": "unused"},
]


class CatalogTests(unittest.TestCase):
    def test_every_entry_points_at_existing_code_and_doc(self):
        for entry in CATALOG:
            with self.subTest(entry=entry["id"]):
                self.assertTrue((ROOT / entry["code"]).is_file(), entry["code"])
                self.assertTrue((ROOT / entry["doc"]).is_file(), entry["doc"])
                self.assertTrue(entry["no"].startswith("實驗"))

    def test_every_entry_states_what_it_fits(self):
        for entry in CATALOG:
            with self.subTest(entry=entry["id"]):
                self.assertTrue(entry["fits"].strip())
                self.assertIsInstance(entry["seconds"], (int, float))
                self.assertGreater(entry["seconds"], 0)

    def test_sub_labels_are_unique_within_experiment(self):
        labels = [entry["sub"] for entry in CATALOG if "sub" in entry]
        labels += [entry["committed_sub"] for entry in CATALOG if "committed_sub" in entry]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(sorted(label for label in labels if label.startswith("8-")), ["8-1", "8-2", "8-3"])
        self.assertEqual(sorted(label for label in labels if label.startswith("6-")), ["6-1", "6-2", "6-3"])

    def test_ids_are_unique(self):
        ids = [entry["id"] for entry in CATALOG]
        self.assertEqual(len(ids), len(set(ids)))


class ParameterTests(unittest.TestCase):
    def setUp(self):
        self.runner = ExperimentRunner(FAKE_DATASETS, "data")
        self.spec = {e["id"]: e for e in CATALOG}

    def test_defaults_fill_missing_values(self):
        params = self.runner._clean(self.spec["exp3"], {})
        self.assertEqual(params, {
            "dataset": "T1/8000rpm", "openset_method": "mahalanobis", "seed": 42, "n_trials": 20,
        })

    def test_rejects_unknown_dataset(self):
        with self.assertRaises(ValueError):
            self.runner._clean(self.spec["exp1"], {"dataset": "T9/1rpm"})

    def test_rejects_unlisted_method(self):
        with self.assertRaises(ValueError):
            self.runner._clean(self.spec["exp1"], {"openset_method": "svm"})

    def test_rejects_out_of_range_int(self):
        with self.assertRaises(ValueError):
            self.runner._clean(self.spec["exp3"], {"n_trials": 0})
        with self.assertRaises(ValueError):
            self.runner._clean(self.spec["exp1"], {"seed": -1})

    def test_unknown_experiment(self):
        with self.assertRaises(ValueError):
            self.runner.run("exp99", {})


class JsonableTests(unittest.TestCase):
    def test_numpy_and_non_finite_values(self):
        value = jsonable({"a": np.float64(0.5), "b": np.int64(3), "c": float("nan"),
                          "d": np.array([1.0, math.inf]), 1: (np.bool_(True),)})
        self.assertEqual(value, {"a": 0.5, "b": 3, "c": None, "d": [1.0, None], "1": [True]})
        json.dumps(value, allow_nan=False)


class CommittedTests(unittest.TestCase):
    def setUp(self):
        self.runner = ExperimentRunner(FAKE_DATASETS, "data")

    def test_exp6_formal_matrix(self):
        data = self.runner.committed("exp6_formal_matrix")
        self.assertTrue(data["available"])
        self.assertEqual(data["completed_runs"], 54)
        self.assertEqual({row["method"] for row in data["macro_summary"]}, {"mahalanobis", "knn"})
        json.dumps(data, allow_nan=False)

    def test_exp8_health_index_results(self):
        data = self.runner.committed("exp8_health_index_results")
        self.assertTrue(data["available"])
        self.assertEqual(data["rows"], 54)
        self.assertEqual(len(data["by_condition"]), 18)
        self.assertIsInstance(data["by_condition"][0]["auroc_mean"], float)
        json.dumps(data, allow_nan=False)

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            self.runner.committed("reports")


class SaveTests(unittest.TestCase):
    def test_results_go_under_experiments_subdir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(FAKE_DATASETS, "data", out_dir=tmp)
            first = runner._save("exp1", {"x": 1})
            second = runner._save("exp1", {"x": 2})
            self.assertNotEqual(first, second)
            files = sorted((Path(tmp) / "experiments").glob("exp1_*.json"))
            self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
