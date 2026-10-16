"""105維完整欄位版本與拒絕政策；不更改來源CSV。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_DIM = 105
REGISTRY_VERSION = "formal_feature_registry_v1"
CONFIGURATIONS = ("8screws", "7screws", "6screws", "5screws", "4screws",
                  "3screws", "2screws", "1screws", "3_14screws", "4_146screws")
CONFIG_ALIASES = {"1screw": "1screws"}


def adapter_columns() -> list[str]:
    """保留main 22a253b的FEATURE_NAMES順序與拼字。"""
    stats = ("rms", "mean", "kurtosis", "std", "skewness", "peak2peak",
             "crest_indicator", "clearance_indicator", "shape_indicator",
             "impulse_indicator", "max", "min", "msa", "variance", "mean_amplitude")
    names = []
    for prefix, offset in (("Current", 0), ("Vibration_X", 15), ("Vibration_Y", 40),
                           ("Vibration_Z", 65), ("Delta_T", 90)):
        names.extend(f"{offset + i}-{prefix}-{name}" for i, name in enumerate(stats, 1))
        if prefix.startswith("Vibration"):
            names.extend(f"{offset + 15 + i}-{prefix}-FFT{i}{prefix[-1]}" for i in range(1, 11))
    return names


def _historical_columns() -> tuple[str, ...]:
    names = adapter_columns()
    for i in range(12, 15):
        names[i] = names[i].replace("Current", "current")
    for start in (55, 80):
        for i in range(start, start + 10):
            names[i] = names[i][:-1] + "X"
    return tuple(names)


SCHEMAS = {"historical_clean_v1": _historical_columns(),
           "adapter_generated_v1": tuple(adapter_columns())}


class FeatureSchemaError(ValueError):
    """可公開的拒絕原因與計數；不含開發機絕對路徑。"""

    def __init__(self, reason: str, file_id: str = "", *, rows: int = 0,
                 rejected_rows: int | None = None, row_indices=(), detail: str = ""):
        self.audit = {"status": "REJECTED", "reason": reason, "file_id": file_id,
                      "rows": rows, "rejected_rows": rows if rejected_rows is None else rejected_rows,
                      "rejected_row_indices": list(row_indices), "discarded_rows": 0}
        super().__init__(f"{reason}：{file_id}；拒絕列數={self.audit['rejected_rows']}" +
                         (f"；{detail}" if detail else ""))


def read_feature_csv(path: Path, file_id: str) -> tuple[np.ndarray, dict]:
    """精確欄位與品質驗證後，沿用pandas的原始浮點解析。"""
    try:
        payload = path.read_bytes()
        records = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FeatureSchemaError("unreadable_csv", file_id, detail=type(exc).__name__) from exc
    if not records:
        raise FeatureSchemaError("empty_file", file_id)
    header, rows = records[0], records[1:]
    if len(header) != len(set(header)):
        raise FeatureSchemaError("duplicate_header", file_id, rows=len(rows))
    version = next((name for name, columns in SCHEMAS.items() if tuple(header) == columns), None)
    if version is None:
        raise FeatureSchemaError("ordered_header_mismatch", file_id, rows=len(rows),
                                 detail=f"預期105欄完整已登錄版本，實際{len(header)}欄")
    if not rows:
        raise FeatureSchemaError("empty_configuration", file_id)
    invalid = [i for i, row in enumerate(rows) if len(row) != FEATURE_DIM]
    if invalid:
        raise FeatureSchemaError("row_width_mismatch", file_id, rows=len(rows),
                                 rejected_rows=len(invalid), row_indices=invalid)
    try:
        frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
        values = frame.to_numpy(dtype=float)
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        invalid = []
        for i, row in enumerate(rows):
            try:
                np.asarray(row, dtype=float)
            except (ValueError, TypeError):
                invalid.append(i)
        raise FeatureSchemaError("non_numeric", file_id, rows=len(rows),
                                 rejected_rows=len(invalid) or len(rows), row_indices=invalid) from exc
    if values.shape != (len(rows), FEATURE_DIM):
        raise FeatureSchemaError("parsed_shape_mismatch", file_id, rows=len(rows))
    invalid = np.flatnonzero(~np.isfinite(values).all(axis=1)).tolist()
    if invalid:
        raise FeatureSchemaError("non_finite", file_id, rows=len(rows),
                                 rejected_rows=len(invalid), row_indices=invalid)
    return values, {"status": "VERIFIED", "file_id": file_id, "schema_version": version,
                    "rows": len(values), "columns": FEATURE_DIM, "rejected_rows": 0, "discarded_rows": 0,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "header_sha256": hashlib.sha256(json.dumps(header, ensure_ascii=False,
                                                               separators=(",", ":")).encode()).hexdigest()}


def validate_pool_coverage(pools, *, require_complete: bool = False) -> dict:
    """格式與完整矩陣coverage分開；不改原key與樣本順序。"""
    names = list(pools)
    canonical = [CONFIG_ALIASES.get(name, name) for name in names]
    unknown = sorted(set(canonical) - set(CONFIGURATIONS))
    if unknown:
        raise FeatureSchemaError("unknown_configuration", detail=", ".join(unknown))
    if len(canonical) != len(set(canonical)):
        raise FeatureSchemaError("ambiguous_configuration_alias")
    for name, values in pools.items():
        if len(values) == 0:
            raise FeatureSchemaError("empty_configuration", name)
    if "8screws" not in canonical:
        raise FeatureSchemaError("missing_healthy", detail="缺少健康基準 8screws")
    missing = [name for name in CONFIGURATIONS if name not in canonical]
    result = {"status": "INCOMPLETE" if missing else "VERIFIED", "missing_configurations": missing,
              "present_configurations": canonical, "required_complete": require_complete,
              "raw_session_independence": "UNKNOWN"}
    if require_complete and missing:
        raise FeatureSchemaError("incomplete_matrix_coverage", detail=", ".join(missing))
    return result
