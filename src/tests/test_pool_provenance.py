"""已載入 bytes、陣列與公開來源的綁定；不使用被忽略的正式資料。"""
from pathlib import Path
from unittest.mock import patch
import argparse
import copy
import json
import subprocess
import types

import numpy as np
import pytest

from core.data import PoolCache, discover_datasets, load_pools
from core.logger import setup_run
from core.provenance import ProvenanceError, load_pool_source, source_for_pools
from core.runner import execution_context, run_dataset
from tests.test_exp6_benchmark import _fixture


def fixture(tmp_path):
    root = _fixture(tmp_path / "data")
    return root, discover_datasets(root)[0]["path"]


def test_actual_parser_bytes_and_original_csv_rows(tmp_path):
    root, dataset = fixture(tmp_path)
    pools, context = load_pool_source(root, dataset)
    public = source_for_pools(pools, context)
    assert public["status"] == "VERIFIED"
    assert len(public["row_mapping"]) == 2
    assert all(row["csv_row_start"] == 0 and row["csv_row_stop"] == 30
               for row in public["row_mapping"])
    assert public["raw_session_independence"] == public["fresh_test"] == "UNKNOWN"
    assert str(root) not in json.dumps(public)
    assert (Path(".lineage_private") / (public["private_context_id"] + ".json")).is_file()


def test_memory_has_no_invented_file_identity():
    result = source_for_pools({"8screws": np.zeros((3, 105))})
    assert result["status"] == "UNVERIFIED_MEMORY"
    assert result["files"] == [] and result["data_version"] == "UNKNOWN"


@pytest.mark.parametrize("tamper", ["array", "metadata", "file"])
def test_tampered_array_metadata_or_source_rejected(tmp_path, tamper):
    root, dataset = fixture(tmp_path)
    pools, context = load_pool_source(root, dataset)
    if tamper == "array":
        pools["8screws"][0, 0] += 1
    elif tamper == "metadata":
        context.public["data_version"] = "fake"
    else:
        path = next(dataset.rglob("*_clean.csv"))
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ProvenanceError):
        source_for_pools(pools, context)


def test_content_cache_never_binds_new_sha_to_old_array(tmp_path):
    root, dataset = fixture(tmp_path)
    cache = PoolCache()
    pools, context = load_pool_source(root, dataset, cache=cache)
    again, second = load_pool_source(root, dataset, cache=cache)
    assert cache.hits == 1
    assert context.public["data_version"] == second.public["data_version"]
    path = next(dataset.rglob("*_clean.csv"))
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    fields = lines[1].split(",")
    fields[0] = str(float(fields[0]) + 1)
    lines[1] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    updated, third = load_pool_source(root, dataset, cache=cache)
    assert third.public["data_version"] != context.public["data_version"]
    assert any(not np.array_equal(updated[k], pools[k]) for k in pools)
    assert all(np.array_equal(again[k], pools[k]) for k in pools)


def test_loader_sha_mismatch_even_when_before_after_snapshots_match(tmp_path):
    root, dataset = fixture(tmp_path)
    def wrong(path, *, audit):
        pools = load_pools(path, audit=audit)
        audit["files"][0]["sha256"] = "0" * 64
        return pools
    with patch("core.data.load_pools", side_effect=wrong), pytest.raises(ProvenanceError):
        load_pool_source(root, dataset)


def test_sidecar_failure_is_private_and_fatal(tmp_path):
    root, dataset = fixture(tmp_path)
    with patch("core.provenance.write_private_context", side_effect=OSError("私人路徑秘密")):
        with pytest.raises(ProvenanceError) as exc:
            load_pool_source(root, dataset)
    assert "私人路徑秘密" not in str(exc.value)


def test_missing_dataset_cli_still_saves_environment_and_failure(tmp_path):
    _, paths = setup_run("exp1_cold_start", unique=True)
    args = argparse.Namespace(data_root=tmp_path / "missing", motor="T1", rpm="8000")
    with pytest.raises(SystemExit):
        run_dataset(args, paths, lambda _: None)
    assert (paths.output_dir / "environment.json").is_file()
    assert json.loads((paths.output_dir / "source.json").read_text())["status"] == "FAILED"


def test_polar_api_iterator_and_file_binding(tmp_path):
    from experiments.exp4_polar_map import run, iter_run
    from tests.test_stream_integration import pools as make_pools
    from core.feature_schema import SCHEMAS
    import pandas as pd

    root, dataset = fixture(tmp_path)
    for config, values in make_pools().items():
        target = dataset / config
        target.mkdir(exist_ok=True)
        pd.DataFrame(values, columns=SCHEMAS["historical_clean_v1"]).to_csv(
            target / "T1_Group_feature_data_clean.csv", index=False)
    pools, context = load_pool_source(root, dataset)
    batch = run(pools, parts="a", source_context=context)
    streamed = list(iter_run(pools, parts="a", source_context=context))[-1]["result"]
    assert batch["provenance"]["source"]["row_mapping"] == streamed["provenance"]["source"]["row_mapping"]
    assert batch["geometry"] == streamed["geometry"]
    pools["8screws"][0, 0] += 1
    with pytest.raises(ProvenanceError):
        run(pools, parts="a", source_context=context)


@pytest.mark.parametrize("method", ["mahalanobis", "knn"])
def test_comparison_result_records_actual_method_and_memory_status(tmp_path, method):
    from experiments.compare_openset import run

    root, dataset = fixture(tmp_path)
    pools, context = load_pool_source(root, dataset)
    result = run(pools, methods=(method,), source_context=context)["results"][0]
    assert result["provenance"]["source"]["status"] == "VERIFIED"
    assert result["provenance"]["resolved_config"]["openset_method"] == method
    memory = run(pools, methods=(method,))["results"][0]
    assert memory["provenance"]["source"]["status"] == "UNVERIFIED_MEMORY"
    for row in (result, memory):
        row.pop("provenance")
    assert result == memory


@pytest.mark.parametrize("name", ["exp4_polar_map", "compare_openset"])
def test_polar_and_comparison_match_current_scientific_baseline(name):
    import importlib
    from tests.test_stream_integration import pools
    from web.experiments import jsonable

    root = Path(__file__).resolve().parents[2]
    text = subprocess.check_output([
        "git", "show", f"edc6a0f052379df52c6874b702badcc000ba742d:src/experiments/{name}.py"], cwd=root)
    old = types.ModuleType("_source_pair_" + name)
    exec(compile(text, name, "exec"), old.__dict__)
    current = importlib.import_module("experiments." + name)
    expected, actual = jsonable(old.run(pools())), jsonable(current.run(pools()))
    rows = actual["results"] if name == "compare_openset" else [actual]
    for row in rows:
        assert row.pop("provenance")["source"]["status"] == "UNVERIFIED_MEMORY"
    assert actual == expected


@pytest.mark.parametrize("exp_id", ["exp4", "exp7"])
def test_web_polar_and_comparison_receive_actual_source(tmp_path, exp_id):
    from web.experiments import ExperimentRunner

    root, _ = fixture(tmp_path)
    runner = ExperimentRunner(discover_datasets(root), str(root))
    module = "exp4_polar_map" if exp_id == "exp4" else "compare_openset"
    with patch("experiments." + module + ".run", return_value={}) as callback:
        runner.run(exp_id, {})
    assert callback.call_args.kwargs["source_context"].public["status"] == "VERIFIED"


@pytest.mark.parametrize("name", ["exp4_polar_map", "compare_openset"])
def test_polar_and_comparison_cli_load_failure_is_same_run(tmp_path, name):
    import importlib

    module = importlib.import_module("experiments." + name)
    with patch("sys.argv", [name, "--data-root", str(tmp_path / "missing")]):
        with pytest.raises(SystemExit):
            module.main()
    program = "exp4_polar_map" if name == "exp4_polar_map" else "openset_comparison"
    source = next(Path("output").glob(program + "/*/source.json"))
    assert json.loads(source.read_text())["status"] == "FAILED"
    assert (source.parent / "environment.json").is_file()


def test_existing_config_resolver_and_memory_context():
    pools = {"8screws": np.zeros((3, 105))}
    before = copy.deepcopy(pools)
    result = execution_context(pools, openset_method="knn", method="oas", knn_neighbors=3)
    assert result["resolved_config"]["effective"]["knn_neighbors"] == 3
    assert result["resolved_config"]["ignored"][0]["parameter"] == "mahalanobis_method"
    assert np.array_equal(pools["8screws"], before["8screws"])


@pytest.mark.parametrize("name", ["exp1_cold_start", "exp2_scale_growth", "exp3_trend"])
def test_all_events_and_scientific_outputs_match_fixed_baseline(name):
    import importlib
    from web.experiments import jsonable

    root = Path(__file__).resolve().parents[2]
    baseline = ("edc6a0f052379df52c6874b702badcc000ba742d" if name.startswith("exp2")
                else "db47d244345e0263ff9101f0eeec4757ed92d0fe")
    text = subprocess.check_output(["git", "show", f"{baseline}:src/experiments/{name}.py"], cwd=root)
    old = types.ModuleType("_pool_provenance_" + name)
    exec(compile(text, name, "exec"), old.__dict__)
    current = importlib.import_module("experiments." + name)
    rng = np.random.default_rng(100)
    pools = {config: rng.normal(center, .3, (40, 105)) for config, center in
             (("8screws", 0), ("7screws", 2), ("6screws", 3), ("5screws", 4), ("4screws", 5))}
    iterator = "iter_playback" if name.startswith("exp2") else "iter_run"
    kwargs = {"max_ticks_per_stage": 5} if name.startswith("exp2") else (
        {"n_trials": 1} if name.startswith("exp3") else {})
    expected = jsonable(list(getattr(old, iterator)(pools, seed=42, **kwargs)))
    actual = jsonable(list(getattr(current, iterator)(pools, seed=42, **kwargs)))
    provenance = actual[-1]["result"].pop("provenance")
    assert provenance["source"]["status"] == "UNVERIFIED_MEMORY"
    assert actual == expected


def test_web_batch_stream_and_file_edit_use_actual_source(tmp_path):
    from web.experiments import ExperimentRunner

    root, dataset = fixture(tmp_path)
    runner = ExperimentRunner(discover_datasets(root), str(root), out_dir=tmp_path / "output")
    batch = runner.run("exp1", {"seed": 42})
    frames = list(runner.stream("exp1", {"seed": 42}))
    streamed = frames[-1][0]["payload"]
    one = batch["result"]["provenance"]["source"]
    two = streamed["result"]["provenance"]["source"]
    assert one["dataset_fingerprint"] == two["dataset_fingerprint"]
    assert one["row_mapping"] == two["row_mapping"]
    assert runner._source_cache.hits == 1
    source = next(dataset.rglob("*_clean.csv"))
    lines = source.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[0] = str(float(fields[0]) + 1)
    lines[1] = ",".join(fields)
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    changed = runner.run("exp1", {})
    assert changed["result"]["provenance"]["source"]["data_version"] != one["data_version"]


@pytest.mark.parametrize("name", ["exp1_cold_start", "exp2_scale_growth", "exp3_trend"])
def test_cli_uses_traced_loader_and_same_run_environment(tmp_path, name):
    import importlib

    root, _ = fixture(tmp_path)
    module = importlib.import_module("experiments." + name)
    result = {"rows": [], "stages": [], "trials": [], "summary": {}, "final_known": [],
              "healthy_fp_rate": 0, "macro_detect_rate": 0, "macro_auroc": 0}
    with patch.object(module, "run", return_value=result) as callback, \
            patch("sys.argv", [name, "--data-root", str(root)]):
        if name.startswith("exp1"):
            with patch.object(module, "_make_figure"), patch.object(module, "save_plot"):
                module.main()
        else:
            module.main()
    source = callback.call_args.kwargs["source_context"]
    assert source.public["status"] == "VERIFIED"
    files = list(Path("output").glob(name + "/*/source.json"))
    assert files and json.loads(files[-1].read_text())["status"] == "VERIFIED"
    assert (files[-1].parent / "environment.json").is_file()
    assert "source_context" not in json.dumps(result)


@pytest.mark.parametrize("stream", [False, True])
def test_web_source_failure_is_saved_without_private_payload(tmp_path, stream):
    from web.experiments import ExperimentRunner

    root, dataset = fixture(tmp_path)
    runner = ExperimentRunner(discover_datasets(root), str(root), out_dir=tmp_path / "output")
    next(dataset.rglob("*_clean.csv")).write_text("私人損壞內容", encoding="utf-8")
    with pytest.raises(ProvenanceError):
        list(runner.stream("exp1", {})) if stream else runner.run("exp1", {})
    files = list((tmp_path / "output/experiments").glob("exp1_*.json"))
    saved = files[0].read_text(encoding="utf-8")
    assert json.loads(saved)["status"] == "FAILED"
    assert str(root) not in saved and "私人損壞內容" not in saved
