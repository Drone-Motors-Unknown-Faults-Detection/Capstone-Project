"""健康指數批次的來源版本、內容快取及已知專用 fit 角色。"""
from pathlib import Path
from unittest.mock import patch
import json
import subprocess
import types

import pytest

from core.data import PoolCache
from experiments import health_index_benchmark as module
from tests.test_exp6_benchmark import _fixture


@pytest.mark.parametrize("method", ["mahalanobis", "knn"])
def test_health_source_roles_cache_and_original_metrics(tmp_path, method):
    root = _fixture(tmp_path / "data")
    project = Path(__file__).resolve().parents[2]
    code = subprocess.check_output(["git", "show",
        "c38b51a59700c9ea7a0ee7378062d1de1d4bf7c2:src/experiments/health_index_benchmark.py"], cwd=project)
    old = types.ModuleType("_paired_health")
    old.__file__ = module.__file__
    exec(compile(code, "health_index_benchmark", "exec"), old.__dict__)
    baseline = old.run(root, openset_method=method, require_nine=False)
    cache = PoolCache()
    first = module.run(root, openset_method=method, require_nine=False, pool_cache=cache)
    second = module.run(root, openset_method=method, require_nine=False, pool_cache=cache)
    assert first["rows"] == second["rows"] == baseline["rows"]
    assert cache.hits == 1
    audit = first["fit_audit"]["T1/8000"]
    assert audit["known_configuration"] == "8screws"
    assert set(audit["train"]).isdisjoint(audit["calibration"])
    assert set(audit["known_test"]).isdisjoint(audit["train"] + audit["calibration"])
    assert set(audit["unknown_test"]) == {"7screws"}
    assert first["provenance"]["source"]["status"] == "VERIFIED"
    assert str(root) not in json.dumps(first)


def test_health_cli_missing_source_preserves_failed_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.main(["--data-root", str(tmp_path / "missing")])
    source = next(Path("output").glob("exp8_health_index_benchmark/*/source.json"))
    assert json.loads(source.read_text())["status"] == "FAILED"
    assert (source.parent / "environment.json").is_file()


def test_health_web_uses_traced_run_and_saves_failure(tmp_path):
    from core.data import discover_datasets
    from core.provenance import ProvenanceError
    from web.experiments import ExperimentRunner

    root = _fixture(tmp_path / "data")
    runner = ExperimentRunner(discover_datasets(root), str(root), out_dir=tmp_path / "out")
    actual = runner.run("exp8", {})["result"]
    assert actual["provenance"]["source"]["status"] == "VERIFIED"
    with patch.object(module, "run", side_effect=ProvenanceError("私人測試訊息")):
        with pytest.raises(ProvenanceError):
            runner.run("exp8", {})
    artifacts = list((tmp_path / "out/experiments").glob("*.json"))
    assert any(json.loads(path.read_text()).get("status") == "FAILED" for path in artifacts)
    assert all("私人測試訊息" not in path.read_text() for path in artifacts)


def test_monitor_row_positions_and_changed_source_are_not_raw_windows(tmp_path):
    from experiments.health_monitor import iter_run
    from core.provenance import ProvenanceError

    root = _fixture(tmp_path / "data")
    iterator = iter_run(root, "T1", "8000rpm", "8screws", motor_id="m", session_id="s", max_windows=2)
    fitted, one, two = next(iterator), next(iterator), next(iterator)
    source = fitted["provenance"]["source"]
    assert source["row_semantics"] == "zero_based_clean_csv_data_rows_not_raw_time_windows"
    assert one["window"]["source_pool_index"] != two["window"]["source_pool_index"]
    assert one["window"]["provenance"]["source"]["data_version"] == source["data_version"]
    path = next(root.rglob("*_clean.csv"))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ProvenanceError):
        next(iterator)


def test_monitor_cli_failure_preserves_environment(tmp_path):
    from experiments.health_monitor import main
    with pytest.raises(SystemExit):
        main(["--data-root", str(tmp_path / "missing"), "--motor", "T1", "--rpm", "8000"])
    source = next(Path("output").glob("exp8_health_monitor/*/source.json"))
    assert json.loads(source.read_text())["status"] == "FAILED"
    assert (source.parent / "environment.json").is_file()
