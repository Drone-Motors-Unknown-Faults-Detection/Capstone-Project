"""實驗六彙總：Aggregate the completed formal exp6 matrix without inventing missing evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd


METRICS = (
    "known_accuracy",
    "open_set_accuracy",
    "auroc",
    "aupr_unknown_positive",
    "fpr_at_tpr95",
    "unknown_precision",
    "unknown_recall",
    "unknown_f1",
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON evidence: {path}") from exc


def _summary_path(manifest_path: Path, reference: str, run_id: str) -> Path:
    candidate = manifest_path.parent / Path(reference)
    if not candidate.is_file():
        raise ValueError(f"missing summary for {run_id}: {candidate}")
    return candidate


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


def load_matrix(manifest_path: Path | str) -> tuple[dict, pd.DataFrame]:
    """Load and strictly validate every completed summary referenced by a matrix."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(path)
    if manifest.get("status") != "completed":
        raise ValueError(f"matrix is not complete: status={manifest.get('status')!r}")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != int(manifest.get("expected_runs", -1)):
        raise ValueError("matrix manifest has an incomplete run list")
    fingerprint = None
    rows: list[dict] = []
    for entry in runs:
        run_id = str(entry.get("run_id", "<missing-run-id>"))
        if entry.get("status") != "completed":
            raise ValueError(f"run {run_id} is not completed")
        reference = entry.get("summary")
        if not isinstance(reference, str):
            raise ValueError(f"run {run_id} has no relative summary reference")
        summary = _read_json(_summary_path(path, reference, run_id))
        if summary.get("status") != "completed":
            raise ValueError(f"summary {run_id} is not completed")
        if summary.get("method") != entry.get("method") or int(summary.get("seed", -1)) != int(entry.get("seed", -2)):
            raise ValueError(f"summary identity mismatch for {run_id}")
        current_fingerprint = summary.get("dataset_fingerprint")
        if not current_fingerprint:
            raise ValueError(f"summary {run_id} has no dataset fingerprint")
        if fingerprint is None:
            fingerprint = current_fingerprint
        elif current_fingerprint != fingerprint:
            raise ValueError(f"dataset fingerprint mismatch at {run_id}")
        summary_rows = summary.get("rows")
        if not isinstance(summary_rows, list) or not summary_rows:
            raise ValueError(f"summary {run_id} has no condition rows")
        for row in summary_rows:
            if row.get("status") != "completed":
                raise ValueError(f"summary {run_id} contains a non-completed row")
            if row.get("method") != entry.get("method") or int(row.get("seed", -1)) != int(entry.get("seed", -2)):
                raise ValueError(f"condition row identity mismatch for {run_id}")
            for metric in METRICS:
                _finite(row.get(metric), f"{run_id}.{metric}")
            rows.append(dict(row, run_id=run_id))
    if not rows:
        raise ValueError("completed matrix contains no rows")
    frame = pd.DataFrame(rows)
    return manifest, frame


def aggregate(manifest_path: Path | str) -> dict:
    manifest, frame = load_matrix(manifest_path)
    methods = [str(method) for method in manifest.get("methods", sorted(frame["method"].unique()))]
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
        if len(joined) != len(left) or len(joined) != len(right):
            raise ValueError("Mahalanobis and kNN rows are not paired one-to-one")
        for metric in METRICS:
            difference = joined[f"{metric}_knn"] - joined[f"{metric}_mahalanobis"]
            mean, std = _mean_std(difference)
            paired_rows.append(
                {
                    "comparison": "knn_minus_mahalanobis",
                    "metric": metric,
                    "n_pairs": int(len(difference)),
                    "mean_difference": _round(mean),
                    "std_difference": _round(std),
                    "knn_better_pairs": int((difference > 0).sum()),
                    "mahalanobis_better_pairs": int((difference < 0).sum()),
                    "ties": int((difference == 0).sum()),
                }
            )

    return {
        "schema_version": 1,
        "status": "completed",
        "source_manifest": "matrix_manifest.json",
        "dataset_root": manifest.get("data_root"),
        "dataset_fingerprint": manifest.get("runs", [{}])[0].get("dataset_fingerprint"),
        "expected_runs": int(manifest["expected_runs"]),
        "completed_runs": int(manifest.get("completed_runs", 0)),
        "methods": methods,
        "seeds": seeds,
        "condition_summary": condition_rows,
        "macro_summary": macro_rows,
        "paired_differences": paired_rows,
    }


def write_aggregate(manifest_path: Path | str, output_root: Path | str | None = None) -> dict:
    manifest_file = Path(manifest_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve() if output_root else manifest_file.parent / "aggregate"
    destination.mkdir(parents=True, exist_ok=True)
    result = aggregate(manifest_file)
    (destination / "aggregate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(result["condition_summary"]).to_csv(destination / "condition_summary.csv", index=False)
    pd.DataFrame(result["macro_summary"]).to_csv(destination / "macro_summary.csv", index=False)
    pd.DataFrame(result["paired_differences"]).to_csv(destination / "paired_differences.csv", index=False)
    lines = ["# exp6 formal matrix aggregate", "", f"- expected runs: {result['expected_runs']}", f"- completed runs: {result['completed_runs']}", f"- methods: {', '.join(result['methods'])}", f"- seeds: {', '.join(map(str, result['seeds']))}", "", "## Macro summary", "", "| method | AUROC mean ± std | open-set accuracy mean ± std | unknown F1 mean ± std |", "|---|---:|---:|---:|"]
    for row in result["macro_summary"]:
        lines.append(f"| {row['method']} | {row['auroc_mean']:.6f} ± {row['auroc_std']:.6f} | {row['open_set_accuracy_mean']:.6f} ± {row['open_set_accuracy_std']:.6f} | {row['unknown_f1_mean']:.6f} ± {row['unknown_f1_std']:.6f} |")
    if result["paired_differences"]:
        lines.extend(["", "## Paired differences", "", "Positive values mean kNN is higher than Mahalanobis for that metric.", "", "| metric | mean difference | std difference | pairs |", "|---|---:|---:|---:|"])
        for row in result["paired_differences"]:
            lines.append(f"| {row['metric']} | {row['mean_difference']:.6f} | {row['std_difference']:.6f} | {row['n_pairs']} |")
    (destination / "aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="output/exp6_formal_matrix/matrix_manifest.json")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    result = write_aggregate(args.manifest, args.output_root)
    print(json.dumps({"status": result["status"], "expected_runs": result["expected_runs"], "completed_runs": result["completed_runs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
