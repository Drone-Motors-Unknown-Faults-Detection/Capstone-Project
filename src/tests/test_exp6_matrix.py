"""Tests for the formal exp6 matrix contract without running full data."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp6_matrix import _is_complete, expected_run_matrix


class MatrixTests(unittest.TestCase):
    def test_matrix_requires_all_nine_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expected 9"):
                expected_run_matrix(Path(directory), seeds=(42,), methods=("mahalanobis",))

    def test_completed_schema_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                '{"status":"completed","method":"knn","seed":42,"rows":[{"status":"completed","motor":"T1","rpm":"6000rpm","method":"knn","seed":42}]}',
                encoding="utf-8",
            )
            expected = {"method": "knn", "seed": 42, "motor": "T1", "rpm": "6000rpm"}
            self.assertTrue(_is_complete(path, expected))
            path.write_text(
                '{"status":"completed","method":"knn","seed":42,"rows":[{"status":"failed","motor":"T1","rpm":"6000rpm","method":"knn","seed":42}]}',
                encoding="utf-8",
            )
            self.assertFalse(_is_complete(path, expected))


if __name__ == "__main__":
    unittest.main()
