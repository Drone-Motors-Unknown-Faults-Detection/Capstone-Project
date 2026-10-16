"""實驗六矩陣：Resumable formal exp6 matrix runner (9 conditions × seeds × methods)."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from core.data import PoolCache, discover_datasets
from core.logger import setup_run
from experiments.exp6_formal_benchmark import (
    FORMAL_CONDITION_COUNT,
    FORMAL_METHODS,
    dataset_fingerprint,
    run,
)


DEFAULT_SEEDS = (42, 123, 2026)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def expected_run_matrix(data_root: Path | str, seeds: Iterable[int] = DEFAULT_SEEDS, methods: Iterable[str] = FORMAL_METHODS) -> list[dict]:
    datasets = sorted(
        discover_datasets(Path(data_root).expanduser().resolve()),
        key=lambda row: (row["motor"], row["rpm"]),
    )
    if len(datasets) != FORMAL_CONDITION_COUNT:
        raise ValueError(
            f"expected {FORMAL_CONDITION_COUNT} formal conditions, discovered {len(datasets)}"
        )
    canonical_methods = tuple(methods)
    invalid = set(canonical_methods) - set(FORMAL_METHODS)
    if invalid:
        raise ValueError(f"unsupported formal methods: {sorted(invalid)}")
    matrix = []
    for dataset in datasets:
        for seed in seeds:
            for method in canonical_methods:
                run_id = run_key(dataset["motor"], dataset["rpm"], seed, method)
                matrix.append(
                    {
                        "run_id": run_id,
                        "condition": f"{dataset['motor']}/{dataset['rpm']}",
                        "motor": dataset["motor"],
                        "rpm": dataset["rpm"],
                        "seed": int(seed),
                        "method": method,
                        "status": "pending",
                        "log_path": f"runs/{run_id}/run.log",
                    }
                )
    return matrix


IDENTITY_COLUMNS = ("dataset", "motor", "rpm", "seed", "method", "status")
METRIC_COLUMNS = (
    "known_accuracy",
    "open_set_accuracy",
    "auroc",
    "aupr_unknown_positive",
    "fpr_at_tpr95",
    "unknown_precision",
    "unknown_recall",
    "unknown_f1",
)


def run_key(motor: str, rpm: str, seed: int, method: str) -> str:
    """唯一 run 鍵；目錄固定為 runs/{run_key}/。"""
    return f"{motor}_{rpm}_seed{int(seed)}_{method}"


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return str(left) == str(right)


def validate_run(run_dir: Path, expected: dict, fingerprint: str | None = None) -> list[str]:
    """依完成判定契約核對單格，回傳全部不符原因；空清單代表完成。

    ``expected`` 需含 method、seed，可含 motor、rpm、condition。
    ``fingerprint`` 給定時 summary 的 dataset_fingerprint 必須相同。
    """
    reasons: list[str] = []
    summary_path = run_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ["missing summary.json"]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable summary.json: {type(exc).__name__}"]
    if not isinstance(summary, dict):
        return ["summary.json is not an object"]
    if summary.get("status") != "completed":
        reasons.append(f"summary status={summary.get('status')!r}")
    if summary.get("method") != expected["method"]:
        reasons.append(f"summary method={summary.get('method')!r}, expected {expected['method']!r}")
    try:
        summary_seed = int(summary.get("seed", -1))
    except (TypeError, ValueError):
        summary_seed = None
    if summary_seed != int(expected["seed"]):
        reasons.append(f"summary seed={summary.get('seed')!r}, expected {expected['seed']}")
    if not summary.get("dataset_fingerprint"):
        reasons.append("summary has no dataset_fingerprint")
    elif fingerprint is not None and summary["dataset_fingerprint"] != fingerprint:
        reasons.append("summary dataset_fingerprint does not match the expected source")
    rows = summary.get("rows")
    row: dict | None = None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        reasons.append(f"summary must hold exactly one row, found {len(rows) if isinstance(rows, list) else 'none'}")
    else:
        row = rows[0]
        if row.get("status") != "completed":
            reasons.append(f"summary row status={row.get('status')!r}")
        for key in ("motor", "rpm"):
            if expected.get(key) is not None and row.get(key) != expected[key]:
                reasons.append(f"summary row {key}={row.get(key)!r}, expected {expected[key]!r}")
        if row.get("method") != expected["method"]:
            reasons.append(f"summary row method={row.get('method')!r}, expected {expected['method']!r}")
        if not _same_value(row.get("seed"), int(expected["seed"])):
            reasons.append(f"summary row seed={row.get('seed')!r}, expected {expected['seed']}")

    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        reasons.append("missing results.csv")
    else:
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:  # pandas 對壞 CSV 會丟多種例外
            reasons.append(f"unreadable results.csv: {type(exc).__name__}")
        else:
            if len(frame) != 1:
                reasons.append(f"results.csv must hold exactly one row, found {len(frame)}")
            elif row is not None:
                saved = frame.iloc[0].to_dict()
                for column in IDENTITY_COLUMNS + METRIC_COLUMNS:
                    if column not in saved:
                        reasons.append(f"results.csv missing column {column}")
                    elif column in row and not _same_value(saved[column], row[column]):
                        reasons.append(f"results.csv {column}: summary={row[column]!r} csv={saved[column]!r}")

    log_path = run_dir / "run.log"
    if not log_path.is_file():
        reasons.append("missing run.log")
    else:
        lines = set(log_path.read_text(encoding="utf-8", errors="replace").splitlines())
        required = {"status=completed", f"seed={int(expected['seed'])}", f"method={expected['method']}"}
        if expected.get("condition"):
            required.add(f"condition={expected['condition']}")
        for line in sorted(required - lines):
            reasons.append(f"run.log lacks '{line}'")
    return reasons


def _is_complete(summary_path: Path, expected: dict, fingerprint: str | None = None) -> bool:
    return not validate_run(summary_path.parent, expected, fingerprint)


def _write_run_csv(run_dir: Path, result: dict) -> None:
    frame = pd.DataFrame(result["rows"])
    temporary = run_dir / "results.csv.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, run_dir / "results.csv")


def run_matrix(
    data_root: Path | str = "data",
    output_root: Path | str = "output/exp6_formal_matrix",
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    methods: Sequence[str] = FORMAL_METHODS,
    resume: bool = True,
    log=None,
) -> dict:
    data_path = Path(data_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    matrix = expected_run_matrix(data_path, seeds, methods)
    fingerprint = dataset_fingerprint(data_path)
    manifest = {
        "schema_version": 1,
        "status": "running",
        # The manifest is committed as evidence; never serialize a local absolute path.
        "data_root": data_path.name,
        "seeds": [int(seed) for seed in seeds],
        "methods": list(methods),
        "expected_runs": len(matrix),
        "runs": matrix,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "matrix_manifest.json"
    _atomic_json(manifest_path, manifest)
    pool_cache = PoolCache()

    for entry in manifest["runs"]:
        run_dir = output_path / "runs" / entry["run_id"]
        summary_path = run_dir / "summary.json"
        if resume and _is_complete(summary_path, entry, fingerprint):
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            entry.update(
                {
                    "status": "completed",
                    "resumed": True,
                    "dataset_fingerprint": cached["dataset_fingerprint"],
                    "commit_sha": cached.get("commit_sha"),
                    "summary": f"runs/{entry['run_id']}/summary.json",
                }
            )
            if log is not None:
                log.info(f"{entry['run_id']}：沿用既有結果")
            continue
        entry["status"] = "running"
        entry["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(manifest_path, manifest)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"
        log_path.write_text(
            f"started={entry['started_at_utc']}\ncondition={entry['condition']}\n"
            f"seed={entry['seed']}\nmethod={entry['method']}\n",
            encoding="utf-8",
        )
        started = time.perf_counter()
        try:
            result = run(
                data_path,
                seed=entry["seed"],
                openset_method=entry["method"],
                require_nine=True,
                pool_cache=pool_cache,
            )
            condition_rows = [
                row
                for row in result["rows"]
                if row.get("motor") == entry["motor"] and row.get("rpm") == entry["rpm"]
            ]
            if len(condition_rows) != 1:
                raise ValueError(
                    f"expected one row for {entry['condition']}, found {len(condition_rows)}"
                )
            result = dict(result, formal_condition_count=1, rows=condition_rows)
            fingerprint = result["dataset_fingerprint"]
            _atomic_json(summary_path, result)
            _write_run_csv(run_dir, result)
            entry.update(
                {
                    "status": "completed",
                    "dataset_fingerprint": result["dataset_fingerprint"],
                    "commit_sha": result["commit_sha"],
                    "duration_seconds": round(time.perf_counter() - started, 6),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": f"runs/{entry['run_id']}/summary.json",
                }
            )
            log_path.write_text(
                log_path.read_text(encoding="utf-8")
                + f"status=completed\nduration_seconds={entry['duration_seconds']}\n",
                encoding="utf-8",
            )
            if log is not None:
                log.info(f"{entry['run_id']}：完成")
        except Exception as exc:
            entry.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_seconds": round(time.perf_counter() - started, 6),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            log_path.write_text(
                log_path.read_text(encoding="utf-8")
                + f"status=failed\nerror={type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            if log is not None:
                log.error(f"{entry['run_id']}：失敗 {entry['error']}")
        _atomic_json(manifest_path, manifest)

    completed = sum(entry["status"] == "completed" for entry in manifest["runs"])
    failed = sum(entry["status"] == "failed" for entry in manifest["runs"])
    missing = sum(entry["status"] not in {"completed", "failed"} for entry in manifest["runs"])
    if log is not None:
        log.info(f"CSV cache：命中 {pool_cache.hits}、未命中 {pool_cache.misses}")
    manifest.update(
        {
            "status": "completed" if completed == len(matrix) else "incomplete",
            "completed_runs": completed,
            "failed_runs": failed,
            "missing_runs": missing,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="output/exp6_formal_matrix")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--method", action="append", choices=FORMAL_METHODS, dest="methods")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    log, _ = setup_run("exp6_formal_matrix", make_output=False)
    log.info(f"參數：{vars(args)}")
    manifest = run_matrix(
        args.data_root,
        args.output_root,
        seeds=tuple(args.seeds or DEFAULT_SEEDS),
        methods=tuple(args.methods or FORMAL_METHODS),
        resume=not args.no_resume,
        log=log,
    )
    log.info(f"矩陣 {manifest['status']}：完成 {manifest['completed_runs']}／{manifest['expected_runs']}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "expected_runs": manifest["expected_runs"],
                "completed_runs": manifest["completed_runs"],
                "failed_runs": manifest["failed_runs"],
                "missing_runs": manifest["missing_runs"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
