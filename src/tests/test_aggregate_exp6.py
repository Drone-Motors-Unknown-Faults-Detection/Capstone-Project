"""Tests for strict exp6 matrix aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.aggregate_exp6 import aggregate, write_aggregate


def _write_fixture(root: Path) -> Path:
    matrix = root / "matrix"
    runs = []
    metrics = {"known_accuracy": 0.8, "open_set_accuracy": 0.7, "auroc": 0.9, "aupr_unknown_positive": 0.85, "fpr_at_tpr95": 0.1, "unknown_precision": 0.75, "unknown_recall": 0.65, "unknown_f1": 0.7}
    for seed in (42, 123):
        for method, offset in (("mahalanobis", 0.0), ("knn", 0.05)):
            run_id = f"T1_6000rpm_seed{seed}_{method}"
            run_dir = matrix / "runs" / run_id
            run_dir.mkdir(parents=True)
            row = {"dataset": "T1/6000rpm", "seed": seed, "method": method, "status": "completed", **{key: value + offset for key, value in metrics.items()}}
            summary = {"status": "completed", "method": method, "seed": seed, "dataset_fingerprint": "abc", "rows": [row]}
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            runs.append({"run_id": run_id, "method": method, "seed": seed, "status": "completed", "dataset_fingerprint": "abc", "summary": f"runs/{run_id}/summary.json"})
    manifest = {"schema_version": 1, "status": "completed", "data_root": "fixture", "seeds": [42, 123], "methods": ["mahalanobis", "knn"], "expected_runs": 4, "completed_runs": 4, "runs": runs}
    path = matrix / "matrix_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class AggregateTests(unittest.TestCase):
    def test_aggregate_computes_macro_and_paired_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = aggregate(_write_fixture(Path(directory)))
            self.assertEqual(result["completed_runs"], 4)
            self.assertEqual(result["macro_summary"][1]["method"], "knn")
            auroc = next(row for row in result["paired_differences"] if row["metric"] == "auroc")
            self.assertAlmostEqual(auroc["mean_difference"], 0.05)

    def test_writer_emits_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _write_fixture(Path(directory))
            output = Path(directory) / "aggregate"
            write_aggregate(manifest, output)
            self.assertTrue((output / "aggregate.json").is_file())
            self.assertTrue((output / "condition_summary.csv").is_file())
            self.assertTrue((output / "macro_summary.csv").is_file())
            self.assertTrue((output / "paired_differences.csv").is_file())


if __name__ == "__main__":
    unittest.main()
