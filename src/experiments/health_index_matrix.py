"""實驗八矩陣：Resumable 9-condition × seed × method health-index result writer."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from core.data import PoolCache
from core.logger import setup_run
from core.openset import SUPPORTED_OPENSET_METHODS, canonical_openset_method
from experiments.exp6_formal_benchmark import dataset_fingerprint
from experiments.health_index_benchmark import FORMAL_CONDITION_COUNT, run


DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_METHODS = ("mahalanobis", "knn")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def expected_run_ids(seeds: Sequence[int] = DEFAULT_SEEDS, methods: Sequence[str] = DEFAULT_METHODS) -> list[str]:
    canonical = [canonical_openset_method(method) for method in methods]
    invalid = set(canonical) - set(SUPPORTED_OPENSET_METHODS)
    if invalid:
        raise ValueError(f"unsupported methods: {sorted(invalid)}")
    return [f"seed_{int(seed)}_{method}" for seed in seeds for method in canonical]


def run_matrix(
    data_root: Path | str,
    output_root: Path | str,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    methods: Sequence[str] = DEFAULT_METHODS,
    resume: bool = True,
    log=None,
) -> dict:
    output_path = Path(output_root).expanduser().resolve()
    canonical_methods = tuple(canonical_openset_method(method) for method in methods)
    expected = expected_run_ids(seeds, canonical_methods)
    # 續跑只沿用同一份資料內容的結果：指紋寫進 manifest，每格 JSON 的指紋也要相同（#269 C06）
    fingerprint = dataset_fingerprint(data_root)
    manifest = {
        "schema_version": 1,
        "experiment": "health_index_benchmark",
        "status": "running",
        "data_root": Path(data_root).name,
        "dataset_fingerprint": fingerprint,
        "seeds": [int(seed) for seed in seeds],
        "methods": list(canonical_methods),
        "expected_runs": len(expected),
        "expected_conditions_per_run": FORMAL_CONDITION_COUNT,
        "runs": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "matrix_manifest.json"
    pool_cache = PoolCache()
    for seed in seeds:
        for method in canonical_methods:
            run_id = f"seed_{int(seed)}_{method}"
            json_path = output_path / f"seed_{int(seed)}" / f"{method}.json"
            csv_path = output_path / f"seed_{int(seed)}" / f"{method}.csv"
            entry = {"run_id": run_id, "seed": int(seed), "method": method, "status": "pending"}
            if resume and json_path.is_file():
                try:
                    cached = json.loads(json_path.read_text(encoding="utf-8"))
                    if (
                        cached.get("status") == "completed"
                        and cached.get("seed") == int(seed)
                        and cached.get("method") == method
                        and cached.get("formal_condition_count") == FORMAL_CONDITION_COUNT
                        and cached.get("dataset_fingerprint") == fingerprint
                    ):
                        entry.update({"status": "completed", "resumed": True, "dataset_fingerprint": fingerprint,
                                      "path": str(json_path.relative_to(output_path))})
                        if log is not None:
                            log.info(f"{run_id}：沿用既有結果")
                        manifest["runs"].append(entry)
                        _atomic_json(manifest_path, manifest)
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            try:
                result = run(data_root, seed=int(seed), openset_method=method, require_nine=True, pool_cache=pool_cache)
                if result.get("dataset_fingerprint") != fingerprint:
                    raise ValueError("dataset_fingerprint changed during the matrix run")
                _atomic_json(json_path, result)
                pd.DataFrame(result["rows"]).to_csv(csv_path, index=False)
                entry.update({"status": "completed", "dataset_fingerprint": fingerprint,
                              "path": str(json_path.relative_to(output_path))})
                if log is not None:
                    log.info(f"{run_id}：完成 {len(result['rows'])} 列")
            except Exception as exc:
                entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                if log is not None:
                    log.error(f"{run_id}：失敗 {entry['error']}")
            manifest["runs"].append(entry)
            _atomic_json(manifest_path, manifest)
    completed = sum(item["status"] == "completed" for item in manifest["runs"])
    failed = sum(item["status"] == "failed" for item in manifest["runs"])
    manifest.update(
        {
            "status": "completed" if completed == len(expected) else "incomplete",
            "completed_runs": completed,
            "failed_runs": failed,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", help="續跑既有輸出根時才給；預設寫本次 output/exp8_health_index_matrix/{ts}/")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    log, paths = setup_run("exp8_health_index_matrix", data_root=args.data_root)
    output_root = Path(args.output_root) if args.output_root else paths.output_dir
    log.info(f"參數：{vars(args)}；輸出根 {output_root}")
    manifest = run_matrix(
        args.data_root,
        output_root,
        seeds=tuple(args.seeds or DEFAULT_SEEDS),
        methods=tuple(args.methods or DEFAULT_METHODS),
        resume=not args.no_resume,
        log=log,
    )
    log.info(f"矩陣 {manifest['status']}：完成 {manifest['completed_runs']}／{manifest['expected_runs']}")
    summary = {k: manifest[k] for k in ("status", "expected_runs", "completed_runs", "failed_runs")}
    summary["output_root"] = str(output_root)
    print(json.dumps(summary))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
