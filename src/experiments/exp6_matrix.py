"""實驗六矩陣：Resumable formal exp6 matrix runner (9 conditions × seeds × methods)."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from core.data import discover_datasets
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
                run_id = f"{dataset['motor']}_{dataset['rpm']}_seed{int(seed)}_{method}"
                matrix.append(
                    {
                        "run_id": run_id,
                        "condition": f"{dataset['motor']}/{dataset['rpm']}",
                        "motor": dataset["motor"],
                        "rpm": dataset["rpm"],
                        "seed": int(seed),
                        "method": method,
                        "status": "pending",
                        "log_path": str(Path("runs") / run_id / "run.log"),
                    }
                )
    return matrix


def _is_complete(summary_path: Path, expected: dict, fingerprint: str | None = None) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("status") != "completed" or summary.get("method") != expected["method"]:
        return False
    if int(summary.get("seed", -1)) != expected["seed"]:
        return False
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or any(row.get("status") != "completed" for row in rows):
        return False
    row = rows[0]
    if expected.get("motor") is not None and row.get("motor") != expected["motor"]:
        return False
    if expected.get("rpm") is not None and row.get("rpm") != expected["rpm"]:
        return False
    if row.get("method") not in {None, expected["method"]} or row.get("seed") not in {None, expected["seed"]}:
        return False
    if fingerprint is not None and summary.get("dataset_fingerprint") != fingerprint:
        return False
    return True


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

    for entry in manifest["runs"]:
        run_dir = output_path / "runs" / entry["run_id"]
        summary_path = run_dir / "summary.json"
        if resume and _is_complete(summary_path, entry, fingerprint):
            entry["status"] = "completed"
            entry["resumed"] = True
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
                    "summary": str(Path("runs") / entry["run_id"] / "summary.json"),
                }
            )
            log_path.write_text(
                log_path.read_text(encoding="utf-8")
                + f"status=completed\nduration_seconds={entry['duration_seconds']}\n",
                encoding="utf-8",
            )
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
        _atomic_json(manifest_path, manifest)

    completed = sum(entry["status"] == "completed" for entry in manifest["runs"])
    failed = sum(entry["status"] == "failed" for entry in manifest["runs"])
    missing = sum(entry["status"] not in {"completed", "failed"} for entry in manifest["runs"])
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
    manifest = run_matrix(
        args.data_root,
        args.output_root,
        seeds=tuple(args.seeds or DEFAULT_SEEDS),
        methods=tuple(args.methods or FORMAL_METHODS),
        resume=not args.no_resume,
    )
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
