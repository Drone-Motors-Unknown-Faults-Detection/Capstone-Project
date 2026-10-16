"""實驗六矩陣的合成產物：依完成判定契約寫出每格 summary.json、results.csv、run.log。"""

from __future__ import annotations

import json
from pathlib import Path

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
