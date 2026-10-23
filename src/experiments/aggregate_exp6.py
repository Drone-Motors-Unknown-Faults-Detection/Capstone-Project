"""實驗六彙總：Aggregate the completed formal exp6 matrix without inventing missing evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from core.logger import setup_run
from experiments.exp6_matrix import METRIC_COLUMNS as METRICS
from experiments.exp6_matrix import run_key, validate_run


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON evidence: {path}") from exc


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _mean_std(series: pd.Series) -> tuple[float, float]:
    values = [_finite(value, str(series.name)) for value in series]
    return float(pd.Series(values).mean()), float(pd.Series(values).std(ddof=1)) if len(values) > 1 else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _entry_problems(entry: dict, manifest: dict) -> list[str]:
    """回傳 manifest 單格的不符原因（身分、status、summary 路徑、fingerprint）；不讀檔。"""
    problems = []
    for key in ("run_id", "motor", "rpm", "seed", "method"):
        if entry.get(key) in (None, ""):
            problems.append(f"manifest entry has no {key}")
    if problems:
        return problems
    if entry["run_id"] != run_key(entry["motor"], entry["rpm"], entry["seed"], entry["method"]):
        problems.append(f"run_id does not match {run_key(entry['motor'], entry['rpm'], entry['seed'], entry['method'])}")
    if entry.get("condition") not in (None, f"{entry['motor']}/{entry['rpm']}"):
        problems.append(f"condition={entry['condition']!r} does not match motor/rpm")
    if int(entry["seed"]) not in {int(seed) for seed in manifest.get("seeds", [])}:
        problems.append(f"seed {entry['seed']} not in manifest seeds")
    if entry["method"] not in manifest.get("methods", []):
        problems.append(f"method {entry['method']!r} not in manifest methods")
    if entry.get("status") != "completed":
        problems.append(f"manifest status={entry.get('status')!r}")
    reference = entry.get("summary")
    expected_reference = f"runs/{entry['run_id']}/summary.json"
    if not isinstance(reference, str) or reference.replace("\\", "/") != expected_reference:
        problems.append(f"summary reference {reference!r} is not {expected_reference}")
    if not entry.get("dataset_fingerprint"):
        problems.append("manifest entry has no dataset_fingerprint")
    return problems


def _format_rejections(rejected: list[dict]) -> str:
    return "; ".join(f"{item['run_id']}: {' | '.join(item['reasons'])}" for item in rejected)


def load_matrix(manifest_path: Path | str, *, allow_partial: bool = False) -> tuple[dict, pd.DataFrame, list[dict], str]:
    """核對矩陣 manifest 與每格產物，回傳 (manifest, 列表, 被拒格, 接受格共同的 dataset fingerprint)。

    每格先以 _entry_problems 核對 manifest 欄位，再以 exp6_matrix.validate_run
    核對產物，最後確認 summary 指標皆為有限值。預設任何矩陣層或單格不符就丟
    ValueError 並列出全部原因；``allow_partial=True`` 時把不符的格放進被拒清單、
    矩陣層問題以 run_id="<matrix>" 放在清單最前，其餘照常回傳。無論是否
    allow_partial，接受的格之間 dataset fingerprint 不一致、或沒有任何接受的格，
    都會丟 ValueError。
    """
    path = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(path)
    matrix_problems: list[str] = []
    if manifest.get("status") != "completed":
        matrix_problems.append(f"matrix status={manifest.get('status')!r}")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("matrix manifest has no run list")
    if len(runs) != int(manifest.get("expected_runs", -1)):
        matrix_problems.append(f"manifest lists {len(runs)} runs, expected_runs={manifest.get('expected_runs')}")

    run_ids = [str(entry.get("run_id")) for entry in runs]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        matrix_problems.append(f"duplicate run_id: {', '.join(duplicates)}")
    conditions = sorted({f"{entry.get('motor')}/{entry.get('rpm')}" for entry in runs})
    expected_keys = {
        (condition, int(seed), str(method))
        for condition in conditions
        for seed in manifest.get("seeds", [])
        for method in manifest.get("methods", [])
    }
    present_keys = set()
    for entry in runs:
        try:
            present_keys.add((f"{entry.get('motor')}/{entry.get('rpm')}", int(entry.get("seed")), str(entry.get("method"))))
        except (TypeError, ValueError):
            pass
    missing = sorted(expected_keys - present_keys)
    if missing:
        matrix_problems.append("missing runs: " + ", ".join(f"{c} seed{s} {m}" for c, s, m in missing))
    rows: list[dict] = []
    rejected: list[dict] = []
    fingerprints: dict[str, list[str]] = {}
    for entry in sorted(runs, key=lambda item: str(item.get("run_id"))):
        run_id = str(entry.get("run_id", "<missing-run-id>"))
        reasons = _entry_problems(entry, manifest)
        if run_id in duplicates:
            reasons.append("duplicate run_id")
        if not reasons:
            expected = dict(entry, condition=f"{entry['motor']}/{entry['rpm']}")
            reasons = validate_run(path.parent / "runs" / run_id, expected, entry["dataset_fingerprint"])
        summary_rows = []
        if not reasons:
            summary = _read_json(path.parent / "runs" / run_id / "summary.json")
            summary_rows = summary["rows"]
            for metric in METRICS:
                try:
                    _finite(summary_rows[0].get(metric), f"{metric}")
                except (TypeError, ValueError) as exc:
                    reasons.append(f"{exc}")
        if reasons:
            rejected.append({"run_id": run_id, "reasons": reasons})
            continue
        fingerprints.setdefault(summary["dataset_fingerprint"], []).append(run_id)
        rows.append(dict(summary_rows[0], run_id=run_id))
    if len(fingerprints) > 1:
        detail = "; ".join(f"{key[:12]}…: {len(ids)} runs" for key, ids in sorted(fingerprints.items()))
        raise ValueError(f"dataset fingerprint mismatch across runs ({detail})")
    if (matrix_problems or rejected) and not allow_partial:
        parts = [f"matrix: {problem}" for problem in matrix_problems]
        if rejected:
            parts.append(f"{len(rejected)} run(s) rejected: {_format_rejections(rejected)}")
        raise ValueError("matrix rejected; " + "; ".join(parts))
    if not rows:
        raise ValueError("matrix contains no accepted rows")
    for problem in matrix_problems:
        rejected.insert(0, {"run_id": "<matrix>", "reasons": [problem]})
    frame = pd.DataFrame(rows).sort_values("run_id", kind="mergesort").reset_index(drop=True)
    (fingerprint,) = fingerprints
    return manifest, frame, rejected, fingerprint


def aggregate(manifest_path: Path | str, *, allow_partial: bool = False) -> dict:
    manifest, frame, rejected, fingerprint = load_matrix(manifest_path, allow_partial=allow_partial)
    partial = bool(rejected)
    methods = [str(method) for method in manifest.get("methods", sorted(frame["method"].unique()))]
    methods = [method for method in methods if method in set(frame["method"])]
    seeds = [int(seed) for seed in manifest.get("seeds", sorted(frame["seed"].unique()))]
    condition_rows = []
    for (condition, method), group in frame.groupby(["dataset", "method"], sort=True):
        row = {"dataset": condition, "method": method, "n_seeds": int(group["seed"].nunique())}
        for metric in METRICS:
            mean, std = _mean_std(group[metric])
            row[f"{metric}_mean"] = _round(mean)
            row[f"{metric}_std"] = _round(std)
        condition_rows.append(row)

    macro_rows = []
    for method in methods:
        group = frame[frame["method"] == method]
        row = {"method": method, "n_condition_seed_rows": int(len(group))}
        for metric in METRICS:
            mean, std = _mean_std(group[metric])
            row[f"{metric}_mean"] = _round(mean)
            row[f"{metric}_std"] = _round(std)
        macro_rows.append(row)

    paired_rows = []
    if set(methods) == {"mahalanobis", "knn"}:
        key = ["dataset", "seed"]
        left = frame[frame["method"] == "mahalanobis"].set_index(key)
        right = frame[frame["method"] == "knn"].set_index(key)
        joined = left[list(METRICS)].join(right[list(METRICS)], lsuffix="_mahalanobis", rsuffix="_knn", how="inner")
        if not partial and (len(joined) != len(left) or len(joined) != len(right)):
            raise ValueError("Mahalanobis and kNN rows are not paired one-to-one")
        for metric in METRICS:
            difference = joined[f"{metric}_knn"] - joined[f"{metric}_mahalanobis"]
            # allow_partial 下兩方法可能沒有共同的 (dataset, seed)：空 join 寫 None，不寫 NaN（#285 R2-02）
            mean, std = _mean_std(difference) if len(difference) else (None, None)
            paired_rows.append(
                {
                    "comparison": "knn_minus_mahalanobis",
                    "metric": metric,
                    "n_pairs": int(len(difference)),
                    "mean_difference": None if mean is None else _round(mean),
                    "std_difference": None if std is None else _round(std),
                    "knn_better_pairs": int((difference > 0).sum()),
                    "mahalanobis_better_pairs": int((difference < 0).sum()),
                    "ties": int((difference == 0).sum()),
                }
            )

    result = {
        "schema_version": 1,
        "status": "PARTIAL" if partial else "completed",
        "source_manifest": "matrix_manifest.json",
        "dataset_root": manifest.get("data_root"),
        # 只用 load_matrix 核對過、所有接受格一致的指紋；不取 runs[0]（它可能是被拒的格，#269 C07）
        "dataset_fingerprint": fingerprint,
        "expected_runs": int(manifest["expected_runs"]),
        "completed_runs": int(manifest.get("completed_runs", 0)),
        "methods": methods,
        "seeds": seeds,
        "condition_summary": condition_rows,
        "macro_summary": macro_rows,
        "paired_differences": paired_rows,
    }
    if partial:
        result["accepted_runs"] = int(frame["run_id"].nunique())
        result["rejected_runs"] = rejected
    return result


def write_aggregate(manifest_path: Path | str, output_root: Path | str | None = None, *, allow_partial: bool = False) -> dict:
    manifest_file = Path(manifest_path).expanduser().resolve()
    result = aggregate(manifest_file, allow_partial=allow_partial)
    default_name = "aggregate_partial" if result["status"] == "PARTIAL" else "aggregate"
    destination = Path(output_root).expanduser().resolve() if output_root else manifest_file.parent / default_name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "aggregate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")
    pd.DataFrame(result["condition_summary"]).to_csv(destination / "condition_summary.csv", index=False)
    pd.DataFrame(result["macro_summary"]).to_csv(destination / "macro_summary.csv", index=False)
    pd.DataFrame(result["paired_differences"]).to_csv(destination / "paired_differences.csv", index=False)
    lines = ["# exp6 formal matrix aggregate", "", f"- status: {result['status']}", f"- expected runs: {result['expected_runs']}", f"- completed runs: {result['completed_runs']}", f"- methods: {', '.join(result['methods'])}", f"- seeds: {', '.join(map(str, result['seeds']))}", "", "## Macro summary", "", "| method | AUROC mean ± std | open-set accuracy mean ± std | unknown F1 mean ± std |", "|---|---:|---:|---:|"]
    for row in result["macro_summary"]:
        lines.append(f"| {row['method']} | {row['auroc_mean']:.6f} ± {row['auroc_std']:.6f} | {row['open_set_accuracy_mean']:.6f} ± {row['open_set_accuracy_std']:.6f} | {row['unknown_f1_mean']:.6f} ± {row['unknown_f1_std']:.6f} |")
    if result["paired_differences"]:
        lines.extend(["", "## Paired differences", "", "Positive values mean kNN is higher than Mahalanobis for that metric.", "", "| metric | mean difference | std difference | pairs |", "|---|---:|---:|---:|"])
        for row in result["paired_differences"]:
            cells = ["n/a" if row[key] is None else f"{row[key]:.6f}" for key in ("mean_difference", "std_difference")]
            lines.append(f"| {row['metric']} | {cells[0]} | {cells[1]} | {row['n_pairs']} |")
    if result.get("rejected_runs"):
        lines.extend(["", "## Rejected runs (PARTIAL)", "", "| run | reasons |", "|---|---|"])
        for item in result["rejected_runs"]:
            lines.append(f"| {item['run_id']} | {'; '.join(item['reasons'])} |")
    (destination / "aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="exp6_matrix 輸出目錄下的 matrix_manifest.json（#253）")
    parser.add_argument("--output-root")
    parser.add_argument("--allow-partial", action="store_true", help="不符規範的格列入 rejected_runs，status 寫 PARTIAL")
    args = parser.parse_args(argv)
    log, _ = setup_run("exp6_aggregate", make_output=False)
    log.info(f"參數：{vars(args)}")
    result = write_aggregate(args.manifest, args.output_root, allow_partial=args.allow_partial)
    for item in result.get("rejected_runs", []):
        log.warning(f"拒絕 {item['run_id']}：{'; '.join(item['reasons'])}")
    log.info(f"彙總 {result['status']}")
    print(json.dumps({"status": result["status"], "expected_runs": result["expected_runs"], "completed_runs": result["completed_runs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
