"""跨工況 runner 的實際來源、失敗及固定版本數值配對。"""
from pathlib import Path
from unittest.mock import patch
import importlib
import json
import shutil
import subprocess
import types

import numpy as np
import pytest

from core.provenance import ProvenanceError
from core.runner import RootSources, run_root_sources
from tests.test_exp6_benchmark import _fixture

MODULES = ("exp5_cross_condition", "exp6_osr_benchmark", "exp10_transfer",
           "exp11_ancestor_comparison", "exp12_confusion_tsne")


def fixture(tmp_path):
    root = _fixture(tmp_path / "data")
    path = root / "Step-1/myfeature/T1/8000rpm"
    shutil.copytree(path, path.parent / "6000rpm")
    shutil.copytree(path.parent, path.parent.parent / "T3")
    return root


@pytest.mark.parametrize("name", MODULES)
def test_root_cli_failure_has_same_run_environment(tmp_path, name):
    module = importlib.import_module("experiments." + name)
    with patch("sys.argv", [name, "--data-root", str(tmp_path / "missing")]):
        with pytest.raises(FileNotFoundError):
            module.main()
    source = next(Path("output").glob(name + "/*/source.json"))
    assert json.loads(source.read_text())["status"] == "FAILED"
    assert (source.parent / "environment.json").is_file()
    assert str(tmp_path) not in source.read_text()


@pytest.mark.parametrize("name", MODULES)
def test_root_methods_match_fixed_scientific_baseline(tmp_path, name):
    from core.detectors import MahalanobisLW, MahalanobisLegacy
    from web.experiments import jsonable

    root = fixture(tmp_path)
    project = Path(__file__).resolve().parents[2]
    code = subprocess.check_output(["git", "show",
        f"edc6a0f052379df52c6874b702badcc000ba742d:src/experiments/{name}.py"], cwd=project)
    old = types.ModuleType("_paired_" + name)
    exec(compile(code, name, "exec"), old.__dict__)
    current = importlib.import_module("experiments." + name)
    options = {"n_values": (10,)} if name.startswith("exp10") else (
        {"seeds": (42,)} if name.startswith("exp11") else {})
    # 此配對核對混淆矩陣；嵌入函數未改，避免為來源測試執行昂貴 t-SNE。
    if name.startswith("exp12"):
        def embedding(pools, *args):
            return {"embedding": np.zeros((1, 2)), "config_labels": ["8screws"], "verdicts": ["unknown"]}
        with patch.object(old, "_tsne_one_dataset", side_effect=embedding), \
                patch.object(current, "_tsne_one_dataset", side_effect=embedding):
            expected, actual = jsonable(old.run(root)), jsonable(current.run(root))
    elif name.startswith("exp6"):
        with patch.object(old, "ALL_DETECTORS", [MahalanobisLW, MahalanobisLegacy]), \
                patch.object(current, "ALL_DETECTORS", [MahalanobisLW, MahalanobisLegacy]):
            expected, actual = old.run(root), current.run(root)
    else:
        expected, actual = old.run(root, **options), current.run(root, **options)
    provenance = actual.pop("provenance")
    assert actual == expected
    assert len(provenance["source"]["datasets"]) == 4
    versions = {s["data_version"] for s in provenance["source"]["datasets"].values()}
    assert versions == {provenance["source"]["data_version"]}
    assert provenance["source"]["raw_session_independence"] == "UNKNOWN"
    assert str(root) not in json.dumps(provenance)


def test_root_rejects_late_mutation_and_saves_failure(tmp_path):
    root = fixture(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    def compute(data_root, *, source_run):
        pools = source_run.load_all()
        pools["T1/8000"]["8screws"][0, 0] += 1
        return {}
    with pytest.raises(ProvenanceError):
        run_root_sources(compute, root, output_dir=out)
    assert json.loads((out / "source.json").read_text())["status"] == "FAILED"


def test_root_environment_and_saved_run_use_actual_root(tmp_path):
    from core.logger import setup_run
    from core.runtime_environment import data_info

    root = fixture(tmp_path / "actual_features")
    default = _fixture(Path.cwd() / "data")
    csv = next(default.rglob("*_clean.csv"))
    csv.write_bytes(csv.read_bytes() + b"\n")
    def compute(data_root, *, source_run):
        source_run.load_all()
        return {}
    with patch("sys.argv", ["api", "--data-root", str(default)]):
        _, paths = setup_run("root_fixture", unique=True)
        result = run_root_sources(compute, root, output_dir=paths.output_dir)
    expected = data_info(root)["dataset_fingerprint"]
    assert result["provenance"]["source"]["dataset_fingerprint"] == expected
    assert result["provenance"]["environment"]["data"]["dataset_fingerprint"] == expected
    saved = json.loads((paths.output_dir / "environment.json").read_text())
    assert saved["data"]["dataset_fingerprint"] == expected
    assert str(root) not in json.dumps(result)


def test_root_rejects_version_mixing_between_datasets(tmp_path):
    root = fixture(tmp_path)
    source = RootSources(root)
    datasets = importlib.import_module("core.data").discover_datasets(root)
    source.load(datasets[0])
    path = next(datasets[1]["path"].rglob("*_clean.csv"))
    path.write_bytes(path.read_bytes().replace(b"0.000", b"0.001"))
    # 保證內容確實變動，不倚賴特定浮點字串是否出現。
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ProvenanceError):
        source.load(datasets[1])


@pytest.mark.parametrize("name,exp_id", list(zip(MODULES, ("exp5", "exp6", "exp10", "exp11", "exp12"))))
def test_web_root_failure_is_saved_without_private_payload(tmp_path, name, exp_id):
    from core.data import discover_datasets
    from web.experiments import ExperimentRunner

    root = fixture(tmp_path)
    runner = ExperimentRunner(discover_datasets(root), str(root), out_dir=tmp_path / "out")
    with patch("experiments." + name + ".run", side_effect=ProvenanceError("來源驗證失敗")):
        with pytest.raises(ProvenanceError):
            runner.run(exp_id, {})
    artifact = next((tmp_path / "out/experiments").glob(exp_id + "_*.json"))
    assert json.loads(artifact.read_text())["status"] == "FAILED"
    assert str(root) not in artifact.read_text()
