"""Unit tests for the formal-source adapter (no external dataset required)."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from core.formal_data import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FormalDataError,
    _channel_key,
    _clean_feature_frame,
    _convert_condition,
    _ensure_output_not_source,
    _parse_condition,
    _statistical_features,
)


class FormalDataTests(unittest.TestCase):
    def test_feature_contract_has_historical_order(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertEqual(FEATURE_NAMES[0], "1-Current-rms")
        self.assertEqual(FEATURE_NAMES[15], "16-Vibration_X-rms")
        self.assertEqual(FEATURE_NAMES[30], "31-Vibration_X-FFT1X")
        self.assertEqual(FEATURE_NAMES[90], "91-Delta_T-rms")
        self.assertEqual(FEATURE_NAMES[-1], "105-Delta_T-mean_amplitude")

    def test_clearance_differs_from_impulse(self) -> None:
        """#277：clearance = max/(mean sqrt|x|)^2，不能退化成 impulse = max/mean|x|。"""
        column = np.array([1.0, -4.0, 9.0, 0.25])
        features = _statistical_features(pd.DataFrame({"c": column}))[0]
        clearance, impulse = features[7], features[9]
        self.assertAlmostEqual(clearance, 9.0 / np.mean(np.sqrt(np.abs(column))) ** 2)
        self.assertAlmostEqual(impulse, 9.0 / np.mean(np.abs(column)))
        self.assertNotAlmostEqual(clearance, impulse)

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
        # 每個 CSV 欄是一個視窗；全零的電流視窗讓 rms=0，峰值因子等特徵變成非有限值
        rng = np.random.default_rng(0)
        n_windows = 12
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            for name in ("Current", "Acceleration_X", "Acceleration_Y", "Acceleration_Z", "Delta_T"):
                signal = rng.normal(0.0, 1.0, (400, n_windows))
                if name == "Current":
                    signal[:, 3] = 0.0
                archive.writestr(f"8screws/T1_{name}_data.csv", pd.DataFrame(signal).to_csv(index=False))
        pending: list[tuple[Path, bytes]] = []
        with tempfile.TemporaryDirectory() as directory, ZipFile(buffer) as archive, \
                np.errstate(divide="ignore", invalid="ignore"):
            (record,) = _convert_condition(
                archive, "synthetic", Path(directory), "T1", "8000rpm",
                force=False, selected_conditions=None, pending_writes=pending,
            )
        # rows_before_clean 是 IQR 過濾前的列數：只少了非有限那一列
        self.assertEqual(record.rows_before_clean, n_windows - 1)
        (_, payload), = pending
        written = pd.read_csv(io.BytesIO(payload))
        self.assertEqual(len(written), record.rows_after_clean)
        self.assertTrue(np.isfinite(written.to_numpy()).all())

    def test_condition_filter_matches_archive_alias_1screw(self) -> None:
        """--condition 用正規名 1screws、archive 目錄叫 1screw 也要物化，不能靜默略過（#285 R2-01）。"""
        rng = np.random.default_rng(1)
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            for name in ("Current", "Acceleration_X", "Acceleration_Y", "Acceleration_Z", "Delta_T"):
                archive.writestr(f"1screw/T1_{name}_data.csv",
                                 pd.DataFrame(rng.normal(0.0, 1.0, (400, 12))).to_csv(index=False))
        for selected, expected in (({("T1", "8000rpm", "1screws")}, 1), ({("T1", "8000rpm", "2screws")}, 0)):
            pending: list[tuple[Path, bytes]] = []
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory, \
                    ZipFile(buffer) as archive, np.errstate(divide="ignore", invalid="ignore"):
                records = _convert_condition(
                    archive, "synthetic", Path(directory), "T1", "8000rpm",
                    force=False, selected_conditions=selected, pending_writes=pending,
                )
                self.assertEqual(len(records), expected)

    def test_parse_condition_accepts_alias_and_returns_canonical(self) -> None:
        self.assertEqual(_parse_condition("T2/8000rpm/1screw"), ("T2", "8000rpm", "1screws"))
        self.assertEqual(_parse_condition("T2/8000rpm/8screws"), ("T2", "8000rpm", "8screws"))
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_condition("T2/8000rpm/9screws")


if __name__ == "__main__":
    unittest.main()
