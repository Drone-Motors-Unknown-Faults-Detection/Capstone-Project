"""實驗六矩陣的合成產物：依完成判定規範寫出每格 summary.json、results.csv、run.log。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.exp6_matrix import METRIC_COLUMNS, run_key

BASE_METRICS = dict(zip(METRIC_COLUMNS, (0.8, 0.7, 0.9, 0.85, 0.1, 0.75, 0.65, 0.7)))


def condition_row(motor: str, rpm: str, seed: int, method: str, offset: float = 0.0) -> dict:
    return {"dataset": f"{motor}/{rpm.replace('rpm', '')}", "motor": motor, "rpm": rpm, "seed": seed,
            "method": method, "status": "completed",
            **{key: round(value + offset, 6) for key, value in BASE_METRICS.items()}}


def write_run(run_dir: Path, row: dict, fingerprint: str = "abc") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "completed", "method": row["method"], "seed": row["seed"],
               "dataset_fingerprint": fingerprint, "commit_sha": "fixture", "rows": [row]}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame([row]).to_csv(run_dir / "results.csv", index=False)
    (run_dir / "run.log").write_text(
        f"started=fixture\ncondition={row['motor']}/{row['rpm']}\nseed={row['seed']}\n"
        f"method={row['method']}\nstatus=completed\nduration_seconds=0.1\n", encoding="utf-8")


def write_matrix(root: Path, conditions=(("T1", "6000rpm"),), seeds=(42, 123), methods=("mahalanobis", "knn")) -> Path:
    matrix = root / "matrix"
    runs = []
    for motor, rpm in conditions:
        for seed in seeds:
            for method in methods:
                run_id = run_key(motor, rpm, seed, method)
                offset = 0.05 if method == "knn" else 0.0
                write_run(matrix / "runs" / run_id, condition_row(motor, rpm, seed, method, offset))
                runs.append({"run_id": run_id, "condition": f"{motor}/{rpm}", "motor": motor, "rpm": rpm,
                             "seed": seed, "method": method, "status": "completed",
                             "dataset_fingerprint": "abc", "summary": f"runs/{run_id}/summary.json"})
    manifest = {"schema_version": 1, "status": "completed", "data_root": "fixture", "seeds": list(seeds),
                "methods": list(methods), "expected_runs": len(runs), "completed_runs": len(runs), "runs": runs}
    path = matrix / "matrix_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def source_fixture(root: Path, conditions=(("T1", "6000rpm"),)) -> Path:
    from core.feature_schema import SCHEMAS

    rng = np.random.default_rng(17)
    for motor, rpm in conditions:
        for config, center in (("8screws", 0.0), ("7screws", 2.0)):
            target = root / "Step-1/myfeature" / motor / rpm / config
            target.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rng.normal(center, .1, (20, 105)), columns=SCHEMAS["historical_clean_v1"]).to_csv(
                target / f"{motor}_Group_feature_data_clean.csv", index=False)
    return root


def producer_result(data_root: Path, row: dict) -> dict:
    """來源工程合成回傳，不訓練模型、不當成研究成績。"""
    from core.provenance import capture_source
    from core.runtime_environment import collect_environment
    from experiments.exp6_matrix import PROJECT, method_configuration

    source, _ = capture_source(data_root)
    environment = collect_environment(PROJECT, data_root=data_root)
    config = method_configuration(row["method"])
    return {"schema_version": 3, "status": "completed", "method": row["method"], "seed": row["seed"],
            "dataset_fingerprint": source["dataset_fingerprint"], "fingerprint_version": source["schema_version"],
            "commit_sha": environment["git"]["head"], "source": source, "environment": environment,
            "config": config["benchmark_config"], "resolved_config": config["resolved_config"],
            "formal_condition_count": 1, "rows": [row]}


def write_verified_matrix(root: Path, conditions=(("T1", "6000rpm"),), seeds=(42, 123),
                          methods=("mahalanobis", "knn")) -> Path:
    from experiments import exp6_matrix as matrix

    data = source_fixture(root / "features", conditions)
    entries = [{"run_id": run_key(motor, rpm, seed, method), "condition": f"{motor}/{rpm}",
                "motor": motor, "rpm": rpm, "seed": seed, "method": method, "status": "pending"}
               for motor, rpm in conditions for seed in seeds for method in methods]
    def benchmark(data_path, *, seed, openset_method, **kwargs):
        rows = [condition_row(motor, rpm, seed, openset_method, .05 if openset_method == "knn" else 0)
                for motor, rpm in conditions]
        return dict(producer_result(data_path, rows[0]), rows=rows)
    with patch.object(matrix, "expected_run_matrix", return_value=entries), \
            patch.object(matrix, "run", side_effect=benchmark):
        result = matrix.run_matrix(data, root / "matrix", seeds=seeds, methods=methods)
    if result["status"] != "completed":
        raise AssertionError("來源fixture矩陣未完成")
    return root / "matrix/matrix_manifest.json"
