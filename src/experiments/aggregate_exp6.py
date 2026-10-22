"""實驗六彙總：Aggregate the completed formal exp6 matrix without inventing missing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from core.logger import setup_run
from core.provenance import file_sha
from core.runtime_environment import collect_environment
from experiments.exp6_matrix import METRIC_COLUMNS as METRICS
from experiments.exp6_matrix import (MATRIX_SCHEMA, artifact_hashes, artifact_snapshot,
                                    run_contract_problems, run_directory, run_key, validate_manifest_contract,
                                    validate_run)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON證據無法讀取：{type(exc).__name__}") from None


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
    if not isinstance(entry["seed"], int) or isinstance(entry["seed"], bool):
        return ["manifest seed不是integer"]
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
        problems.append("summary reference不是固定portable路徑")
    if not entry.get("dataset_fingerprint"):
        problems.append("manifest entry has no dataset_fingerprint")
    return problems


def _format_rejections(rejected: list[dict]) -> str:
    return "; ".join(f"{item['run_id']}: {' | '.join(item['reasons'])}" for item in rejected)


def _verify_parents(path: Path, proof: dict) -> None:
    try:
        if file_sha(path) != proof["manifest_sha256"]:
            raise ValueError("讀取期間matrix manifest bytes變動")
        for parent in proof["checked_runs"]:
            current = artifact_hashes(artifact_snapshot(run_directory(path.parent, parent["run_id"])))
            if current != parent["artifact_sha256"]:
                raise ValueError("讀取期間父run bytes變動")
    except OSError:
        raise ValueError("父產物無法再核對，拒絕完成") from None


def load_matrix(manifest_path: Path | str, *, allow_partial: bool = False,
                _audit: dict | None = None) -> tuple[dict, pd.DataFrame, list[dict]]:
    """讀同一份bytes快照再驗父SHA；partial不能掩蓋程式／資料／設定混版。"""
    path = Path(manifest_path).expanduser().resolve()
    try:
        raw_manifest = path.read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f"matrix manifest無法讀取：{type(exc).__name__}") from None
    if not isinstance(manifest, dict):
        raise ValueError("matrix manifest不是object")
    schema = manifest.get("schema_version")
    contract = None
    if schema == MATRIX_SCHEMA:
        contract = validate_manifest_contract(manifest, allow_missing_runs=allow_partial)
    elif schema != 1 or "contract" in manifest:
        raise ValueError("未知或降級的matrix schema，拒絕混版")
    environment = collect_environment(include_data=False)
    environment["data"]["reason"] = "ARTIFACT_INPUTS_ONLY"
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    proof = {"version": "formal_aggregate_provenance_v2",
             "verification_status": "VERIFIED" if contract is not None else "LEGACY_UNVERIFIED",
             "manifest_source_id": f"sha256:{manifest_sha}",
             "manifest_sha256": manifest_sha,
             "contract_sha256": manifest.get("contract_sha256"),
             "fingerprint_version": contract["source"]["schema_version"] if contract else "UNKNOWN",
             "data_version": contract["source"]["data_version"] if contract else "UNKNOWN",
             "environment": environment, "parent_runs": [], "checked_runs": []}
    matrix_problems: list[str] = []
    if manifest.get("status") != "completed":
        matrix_problems.append(f"matrix status={manifest.get('status')!r}")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or any(not isinstance(entry, dict) for entry in runs):
        raise ValueError("matrix manifest沒有合法run清單")
    if len(runs) != int(manifest.get("expected_runs", -1)):
        matrix_problems.append(f"manifest lists {len(runs)} runs, expected_runs={manifest.get('expected_runs')}")
    observed_completed = sum(entry.get("status") == "completed" for entry in runs)
    if manifest.get("completed_runs", observed_completed) != observed_completed:
        matrix_problems.append("manifest completed_runs與實際狀態列數不符")
    proof["observed_completed_runs"] = observed_completed
    run_ids = [str(entry.get("run_id")) for entry in runs]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        matrix_problems.append(f"duplicate run_id: {', '.join(duplicates)}")
    conditions = sorted({f"{entry.get('motor')}/{entry.get('rpm')}" for entry in runs})
    expected_keys = {
        (condition, int(seed), str(method))
        for condition in conditions
        for seed in manifest.get("seeds", []) for method in manifest.get("methods", [])}
    if contract is not None:
        expected_keys = {(f"{row['motor']}/{row['rpm']}", row["seed"], row["method"])
                         for row in contract["run_keys"]}
    present_keys = set()
    for entry in runs:
        try:
            present_keys.add((f"{entry.get('motor')}/{entry.get('rpm')}", int(entry.get("seed")), str(entry.get("method"))))
        except (TypeError, ValueError):
            pass
    missing = sorted(expected_keys - present_keys)
    if missing:
        matrix_problems.append("missing runs: " + ", ".join(f"{c} seed{s} {m}" for c, s, m in missing))
    rows, rejected, fingerprints = [], [], {}
    for entry in sorted(runs, key=lambda item: str(item.get("run_id"))):
        run_id = str(entry.get("run_id", "<missing-run-id>"))
        run_dir = run_directory(path.parent, run_id)
        payloads = artifact_snapshot(run_dir)
        hashes = artifact_hashes(payloads)
        parent = {"run_id": run_id, "artifact_sha256": hashes}
        proof["checked_runs"].append(parent)
        summary = None
        if "summary.json" in payloads:
            try:
                summary = json.loads(payloads["summary.json"])
            except (ValueError, UnicodeError):
                pass
        if contract is not None and isinstance(summary, dict):
            incompatible = run_contract_problems(summary, entry, contract)
            if incompatible:
                raise ValueError("不相容的資料／程式／設定不能用partial排除：" + run_id)
        reasons = _entry_problems(entry, manifest)
        if contract is not None and entry.get("summary") != f"runs/{run_id}/summary.json":
            reasons.append("新契約summary路徑必須使用固定POSIX相對ID")
        if run_id in duplicates:
            reasons.append("duplicate run_id")
        if not reasons:
            expected = dict(entry, condition=f"{entry['motor']}/{entry['rpm']}")
            reasons = validate_run(run_dir, expected, entry["dataset_fingerprint"],
                                   contract=contract, payloads=payloads)
        if not reasons:
            for metric in METRICS:
                try:
                    _finite(summary["rows"][0].get(metric), metric)
                except (TypeError, ValueError):
                    reasons.append(f"指標缺少或不是有限數值：{metric}")
        if reasons:
            rejected.append({"run_id": run_id, "reasons": reasons})
            continue
        fingerprints.setdefault(summary["dataset_fingerprint"], []).append(run_id)
        rows.append(dict(summary["rows"][0], run_id=run_id))
        proof["parent_runs"].append(parent)
    if len(fingerprints) > 1:
        raise ValueError("dataset fingerprint mismatch across runs")
    if (matrix_problems or rejected) and not allow_partial:
        parts = [f"matrix: {problem}" for problem in matrix_problems]
        if rejected:
            parts.append(f"{len(rejected)} run(s) rejected: {_format_rejections(rejected)}")
        raise ValueError("matrix rejected; " + "; ".join(parts))
    if not rows:
        raise ValueError("matrix contains no accepted rows")
    _verify_parents(path, proof)
    proof["dataset_fingerprint"] = next(iter(fingerprints))
    for problem in matrix_problems:
        rejected.insert(0, {"run_id": "<matrix>", "reasons": [problem]})
    if _audit is not None:
        _audit.update(proof)
    frame = pd.DataFrame(rows).sort_values("run_id", kind="mergesort").reset_index(drop=True)
    return manifest, frame, rejected


def aggregate(manifest_path: Path | str, *, allow_partial: bool = False) -> dict:
    provenance = {}
    manifest, frame, rejected = load_matrix(manifest_path, allow_partial=allow_partial, _audit=provenance)
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
        if joined.empty:
            raise ValueError("沒有共同工況／seed配對，不能以空集合計算方法差值")
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

    result = {
        "schema_version": 2,
        "status": "PARTIAL" if partial else "completed",
        "source_manifest": "matrix_manifest.json",
        "dataset_root": manifest.get("data_root"),
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "expected_runs": int(manifest["expected_runs"]),
        "completed_runs": provenance["observed_completed_runs"],
        "methods": methods,
        "seeds": seeds,
        "condition_summary": condition_rows,
        "macro_summary": macro_rows,
        "paired_differences": paired_rows,
        "provenance": provenance,
    }
    label = result["dataset_root"]
    if isinstance(label, str) and any(value in label for value in ("/", "\\", ":")):
        result["dataset_root"] = "LEGACY_PRIVATE_ROOT"
    if partial:
        result["accepted_runs"] = int(frame["run_id"].nunique())
        result["rejected_runs"] = rejected
    _verify_parents(Path(manifest_path).expanduser().resolve(), provenance)
    return result


def write_aggregate(manifest_path: Path | str, output_root: Path | str | None = None, *, allow_partial: bool = False) -> dict:
    manifest_file = Path(manifest_path).expanduser().resolve()
    result = aggregate(manifest_file, allow_partial=allow_partial)
    default_name = "aggregate_partial" if result["status"] == "PARTIAL" else "aggregate"
    destination = Path(output_root).expanduser().resolve() if output_root else manifest_file.parent / default_name
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("彙總目的地非空，請使用新輸出根；不覆寫封存")
    _verify_parents(manifest_file, result["provenance"])
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["condition_summary"]).to_csv(destination / "condition_summary.csv", index=False, mode="x")
    pd.DataFrame(result["macro_summary"]).to_csv(destination / "macro_summary.csv", index=False, mode="x")
    pd.DataFrame(result["paired_differences"]).to_csv(destination / "paired_differences.csv", index=False, mode="x")
    lines = ["# exp6 formal matrix aggregate", "", f"- status: {result['status']}", f"- expected runs: {result['expected_runs']}", f"- completed runs: {result['completed_runs']}", f"- methods: {', '.join(result['methods'])}", f"- seeds: {', '.join(map(str, result['seeds']))}", "", "## Macro summary", "", "| method | AUROC mean ± std | open-set accuracy mean ± std | unknown F1 mean ± std |", "|---|---:|---:|---:|"]
    for row in result["macro_summary"]:
        lines.append(f"| {row['method']} | {row['auroc_mean']:.6f} ± {row['auroc_std']:.6f} | {row['open_set_accuracy_mean']:.6f} ± {row['open_set_accuracy_std']:.6f} | {row['unknown_f1_mean']:.6f} ± {row['unknown_f1_std']:.6f} |")
    if result["paired_differences"]:
        lines.extend(["", "## Paired differences", "", "Positive values mean kNN is higher than Mahalanobis for that metric.", "", "| metric | mean difference | std difference | pairs |", "|---|---:|---:|---:|"])
        for row in result["paired_differences"]:
            lines.append(f"| {row['metric']} | {row['mean_difference']:.6f} | {row['std_difference']:.6f} | {row['n_pairs']} |")
    if result.get("rejected_runs"):
        lines.extend(["", "## Rejected runs (PARTIAL)", "", "| run | reasons |", "|---|---|"])
        for item in result["rejected_runs"]:
            lines.append(f"| {item['run_id']} | {'; '.join(item['reasons'])} |")
    lines.extend(["", "來源驗證狀態：" + result["provenance"]["verification_status"],
                  "父manifest bytes SHA256：" + result["provenance"]["manifest_sha256"]])
    with (destination / "aggregate.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    summary_created = False
    try:
        _verify_parents(manifest_file, result["provenance"])
        with (destination / "aggregate.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        summary_created = True
        _verify_parents(manifest_file, result["provenance"])
    except ValueError:
        result["status"] = "FAILED"
        result["provenance"]["verification_status"] = "INCOMPLETE_PARENT_CHANGED"
        with (destination / "aggregate.json").open("w" if summary_created else "x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (destination / "aggregate.md").write_text("父產物在寫入期間變動，彙總未完成；請使用新輸出根核對來源。\n", encoding="utf-8")
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="exp6_matrix 輸出目錄下的 matrix_manifest.json（#253）")
    parser.add_argument("--output-root")
    parser.add_argument("--allow-partial", action="store_true", help="不符規範的格列入 rejected_runs，status 寫 PARTIAL")
    args = parser.parse_args(argv)
    log, _ = setup_run("exp6_aggregate", make_output=False, include_data=False)
    log.info("彙總partial={}；父來源由實際bytes SHA識別", args.allow_partial)
    result = write_aggregate(args.manifest, args.output_root, allow_partial=args.allow_partial)
    for item in result.get("rejected_runs", []):
        log.warning(f"拒絕 {item['run_id']}：{'; '.join(item['reasons'])}")
    log.info(f"彙總 {result['status']}")
    print(json.dumps({"status": result["status"], "expected_runs": result["expected_runs"], "completed_runs": result["completed_runs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
