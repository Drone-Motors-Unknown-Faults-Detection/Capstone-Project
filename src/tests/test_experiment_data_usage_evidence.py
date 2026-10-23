"""資料用量以合成 CSV 驗證；不依賴忽略的正式 data。"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.data import HEALTHY
from core.feature_schema import SCHEMAS, FeatureSchemaError
from tests.experiment_data_usage_evidence import allocation, inspect


def fixture(root, motor="T1", n=53):
    rng = np.random.default_rng(42)
    columns = list(SCHEMAS["historical_clean_v1"])
    for config in (HEALTHY, "1screw", "2screws", "3screws", "4screws", "5screws"):
        folder = root / "Step-1/myfeature" / motor / "8000rpm" / config
        folder.mkdir(parents=True)
        pd.DataFrame(rng.normal(size=(n, 105)), columns=columns).to_csv(
            folder / "x_Group_feature_data_clean.csv", index=False)


def test_counts_rounding_known_roles_and_no_training(tmp_path):
    fixture(tmp_path)
    with patch("core.monitor.OpenSetMonitor._refit", side_effect=AssertionError("禁止擬合")):
        result = inspect(tmp_path)
    ds = result["datasets"][0]
    assert result["models_fitted"] == 0
    assert result["raw_session_independence"] == "UNKNOWN"
    assert result["total_finite_rows"] == 6 * 53
    assert len(result["source_inventory"]) == 6
    for cold in ds["cold_start"]:
        assert (cold["train"], cold["cal"], cold["holdout"]) == (31, 10, 12)
        assert cold["classes"][0]["overlap_n"] == 0
        assert cold["classes"][0]["all_rows_accounted"]
    assert ds["ancestor_a"][0]["registration_order"] == [HEALTHY, "1screws", "2screws", "3screws", "4screws"]
    one = next(r for r in ds["configurations"] if r["config"] == "1screws")
    assert one["directory"] == "1screw"
    assert ds["ancestor_a_unknown_n"] == 53
    assert ds["monitor_default_120"]["5screws"] == 53
    assert ds["monitor_default_120"][HEALTHY] == 12
    assert ds["transfer_support"][0]["scratch"]["holdout"] == 2
    assert ds["transfer_support"][0]["target_healthy_test_n"] == 43
    for row in result["exp5_strategy_overlap"]:
        assert row["per_condition_train_overlap_n"] == row["per_condition_cal_overlap_n"] == 0
    assert inspect(tmp_path) == result


def test_nonfinite_rejected_and_exact_source_bytes(tmp_path):
    """正式 loader 遇非有限值即拋 FeatureSchemaError（non_finite），不刪列；檢查過程不改來源位元組。"""
    fixture(tmp_path)
    result = inspect(tmp_path, (42,))
    row = next(r for r in result["datasets"][0]["configurations"] if r["config"] == "5screws")
    assert (row["raw_feature_rows"], row["finite_rows"], row["discarded_nonfinite"]) == (53, 53, 0)
    assert all(not Path(r["relative_path"]).is_absolute() for r in result["source_inventory"])
    file = next(tmp_path.rglob("5screws/*csv"))
    frame = pd.read_csv(file)
    frame.iloc[0, 0] = np.inf
    frame.to_csv(file, index=False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.csv")}
    with pytest.raises(FeatureSchemaError) as error:
        inspect(tmp_path, (42,))
    assert error.value.audit["reason"] == "non_finite"
    assert error.value.audit["rejected_rows"] == 1
    assert all(p.read_bytes() == payload for p, payload in before.items())


def test_missing_data_duplicate_conditions_and_seeds_rejected(tmp_path):
    with pytest.raises(ValueError):
        inspect(tmp_path)
    fixture(tmp_path)
    with pytest.raises(ValueError):
        inspect(tmp_path, (42, 42))
    ds_path = tmp_path / "Step-1/myfeature/T1/8000rpm"
    duplicate = {"motor": "T1", "rpm": "8000rpm", "path": ds_path}
    with patch("tests.experiment_data_usage_evidence.discover_datasets", return_value=[duplicate, duplicate]), pytest.raises(ValueError):
        inspect(tmp_path)


def test_index_digests_depend_on_seed_but_counts_do_not():
    pool = {HEALTHY: np.zeros((100, 105))}
    a, b = allocation(pool, [HEALTHY], 42), allocation(pool, [HEALTHY], 123)
    assert a["classes"][0]["indices_sha256"] != b["classes"][0]["indices_sha256"]
    assert a["train"] == b["train"] == 60
