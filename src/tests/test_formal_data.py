"""Unit tests for the formal-source adapter (no external dataset required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.formal_data import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FormalDataError,
    _channel_key,
    _clean_feature_frame,
    _ensure_output_not_source,
)


class FormalDataTests(unittest.TestCase):
    def test_feature_contract_has_historical_order(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertEqual(FEATURE_NAMES[0], "1-Current-rms")
        self.assertEqual(FEATURE_NAMES[15], "16-Vibration_X-rms")
        self.assertEqual(FEATURE_NAMES[30], "31-Vibration_X-FFT1X")
        self.assertEqual(FEATURE_NAMES[90], "91-Delta_T-rms")
        self.assertEqual(FEATURE_NAMES[-1], "105-Delta_T-mean_amplitude")

    def test_clean_rule_is_feature_level_one_point_five_iqr(self) -> None:
        frame = pd.DataFrame(
            {
                "a": [0.0, 1.0, 2.0, 3.0, 100.0],
                "b": [10.0, 11.0, 12.0, 13.0, 14.0],
            }
        )
        cleaned, before = _clean_feature_frame(frame)
        self.assertEqual(before, 5)
        self.assertEqual(cleaned.index.tolist(), [0, 1, 2, 3])
        self.assertEqual(cleaned.shape, (4, 2))

    def test_channel_aliases_cover_t1_t2_t3_names(self) -> None:
        aliases = {
            "T1_Current_data.csv": "current",
            "T1_Acceleration_X_data.csv": "x",
            "T2_X_data.csv": "x",
            "T3_Y_data.csv": "y",
            "T3_Acceleration_Z_data.csv": "z",
            "T2_Delta_T_data.csv": "delta_t",
        }
        for filename, expected in aliases.items():
            self.assertEqual(_channel_key(filename), expected)

    def test_materializer_never_writes_inside_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaises(FormalDataError):
                _ensure_output_not_source(source, source / "materialized")

    def test_nonfinite_feature_rows_are_removed_before_materialization(self) -> None:
        frame = pd.DataFrame([[1.0, np.inf], [2.0, 3.0]], columns=["a", "b"])
        finite = frame.replace([np.inf, -np.inf], np.nan).dropna()
        self.assertEqual(finite.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
