"""實驗八：唯讀重算封存健康指數矩陣，逐欄核對歷史彙總。"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from core.data import dataset_key
from core.logger import setup_run
from core.provenance import sha256_bytes
from core.runtime_environment import git_head

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "reports/exp8_health_index_results"
DEFAULT_CONTRACT = Path(__file__).resolve().parent / "exp8_health_index_aggregate_contract.json"
METRICS = (
    "health_gap_known_minus_unknown", "open_set_accuracy", "auroc",
    "unknown_recall", "unknown_f1", "known_health_mean", "unknown_health_mean",
)
COUNTS = ("n_train", "n_calibration", "n_known_test", "n_unknown_test", "n_unknown_classes")
NUMERIC = (
    "known_health_mean", "known_health_std", "known_health_median", "known_healthy_fraction",
    "unknown_health_mean", "unknown_health_std", "unknown_health_median", "unknown_critical_fraction",
    "health_gap_known_minus_unknown", "known_score_median", "unknown_score_median",
    "open_set_accuracy", "auroc", "aupr_unknown_positive", "unknown_recall", "unknown_f1",
    "calibration_healthy_anchor", "calibration_critical_anchor",
)
ROW_FIELDS = (
    "dataset", "motor", "rpm", "seed", "method", "mahalanobis_method", "knn_neighbors",
    *COUNTS, *NUMERIC[:8], "health_gap_known_minus_unknown", "health_effect_size",
    *NUMERIC[9:], "rul_available", "status",
)
# README 是人寫的封存說明，改字不改資料：列入來源檔案集合但不釘 SHA；其餘 15 檔逐 byte 核對（#278）。
UNPINNED_SOURCES = frozenset({"README.md"})
RUN_FIELDS = (
    "schema_version", "status", "experiment", "method", "seed", "confidence",
    "formal_condition_count", "dataset_root", "dataset_fingerprint", "score_direction",
    "health_direction", "health_interpretation", "unknown_calibration_leakage",
    "rul_available", "commit_sha", "python", "rows",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pairs(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"JSON 欄位重複：{key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict:
    def reject(value):
        raise ValueError(f"JSON 非有限值：{value}")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=reject)
    _require(isinstance(value, dict), "JSON 根節點必須是物件")
    return value


def _number(value: object, label: str) -> float:
    _require(type(value) in (float, int) and math.isfinite(value), f"非有限數值或型別錯誤：{label}")
    return float(value)


def _integer(value: object, label: str, positive: bool = True) -> int:
    _require(type(value) is int and (value > 0 if positive else value >= 0), f"整數錯誤：{label}")
    return value


def _csv(data: bytes) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    fields = reader.fieldnames or []
    _require(bool(fields) and len(fields) == len(set(fields)), "CSV 欄名缺失或重複")
    rows = list(reader)
    _require(all(None not in row and None not in row.values() for row in rows), "CSV 欄位數錯誤")
    return fields, rows


def _contract(data: bytes) -> dict:
    contract = _json(data)
    # 本版本只實作手冊先固定的規則，不接受悄悄換權重或放大容差。
    fixed = {
        "contract_version": "exp8_recompute_v1", "source_schema_version": 1,
        "seeds": [42, 123, 2026], "methods": ["mahalanobis", "knn"],
        "motors": ["T1", "T2", "T3"], "rpms": ["6000rpm", "8000rpm", "11000rpm"],
        "confidence": 0.95, "knn_neighbors": 5, "n_unknown_classes": 9,
        "weighting": "equal_condition_seed_rows", "ddof": 1, "summary_digits": 6,
        "summary_atol": 0.0000005, "summary_rtol": 0.0,
        "csv_atol": 1e-12, "csv_rtol": 1e-10, "historical_formula_status": "UNKNOWN",
    }
    for field, value in fixed.items():
        _require(contract.get(field) == value and type(contract.get(field)) is type(value), f"規範規則不符：{field}")
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(contract.get("dataset_fingerprint", "")))), "規範資料指紋錯誤")
    names = {"matrix_manifest.json", "aggregate_summary.json", "aggregate_by_condition.csv"}
    names.update(f"seed_{seed}/{method}.{suffix}" for seed in fixed["seeds"]
                 for method in fixed["methods"] for suffix in ("json", "csv"))
    hashes = contract.get("source_sha256", {})
    _require(isinstance(hashes, dict) and set(hashes) == names, "規範必須列出完整 15 檔資料 SHA，README 不釘")
    _require(all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in hashes.values()), "規範 SHA 格式錯誤")
    git_hashes = contract.get("source_git_lf_sha256", {})
    _require(isinstance(git_hashes, dict) and set(git_hashes) == names, "規範必須列出完整 Git LF SHA")
    _require(all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in git_hashes.values()), "規範 Git SHA 格式錯誤")
    return contract


def _snapshot(root: Path, contract: dict) -> dict[str, bytes]:
    expected = contract["source_sha256"]
    names = set(expected) | UNPINNED_SOURCES
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    _require(actual == names, f"來源檔案集合不符：缺 {sorted(names-actual)}；多 {sorted(actual-names)}")
    snapshot = {}
    for name in sorted(names):
        path = (root / name).resolve()
        _require(path.is_relative_to(root), f"來源路徑超出目錄：{name}")
        data = path.read_bytes()
        if name in expected:
            _require(sha256_bytes(data) in (expected[name], contract["source_git_lf_sha256"][name]), f"來源 SHA 不符：{name}")
        snapshot[name] = data
    return snapshot


def _csv_matches_json(data: bytes, rows: list[dict], label: str, contract: dict) -> None:
    fields, table = _csv(data)
    _require(set(fields) == set(ROW_FIELDS) and len(table) == len(rows), f"附帶 CSV 欄位或列數不符：{label}")
    lookup = {}
    for row in table:
        key = row["dataset"]
        _require(key not in lookup, f"附帶 CSV 重複工況：{label}/{key}")
        lookup[key] = row
    _require(set(lookup) == {row["dataset"] for row in rows}, f"附帶 CSV 工況不符：{label}")
    for row in rows:
        for field, value in row.items():
            text = lookup[row["dataset"]][field]
            if type(value) is int:
                _require(text == str(value), f"附帶 CSV 整數或 ID 不符：{label}/{field}")
            elif type(value) is float:
                parsed = float(text)
                _require(math.isfinite(parsed) and math.isclose(parsed, value, abs_tol=contract["csv_atol"],
                         rel_tol=contract["csv_rtol"]), f"附帶 CSV 數值不符：{label}/{row['dataset']}/{field}")
            else:
                _require(text == ("" if value is None else str(value)), f"附帶 CSV 值不符：{label}/{field}")


def load_matrix(source_root: Path | str, contract_path: Path | str) -> tuple[dict, list[dict], dict]:
    """驗證來源位元組與矩陣；只讀封存，不載入模型或原始資料。"""
    root = Path(source_root).resolve()
    contract_bytes = Path(contract_path).read_bytes()
    contract = _contract(contract_bytes)
    snapshot = _snapshot(root, contract)
    manifest = _json(snapshot["matrix_manifest.json"])
    for field, value in {"schema_version": 1, "experiment": "health_index_benchmark", "status": "completed",
                         "expected_runs": 6, "expected_conditions_per_run": 9, "completed_runs": 6,
                         "failed_runs": 0}.items():
        _require(manifest.get(field) == value and type(manifest.get(field)) is type(value), f"manifest 欄位不符：{field}")
    for field in ("seeds", "methods"):
        _require(manifest.get(field) == contract[field], f"manifest {field} 不符")
    if "dataset_fingerprint" in manifest:
        _require(manifest["dataset_fingerprint"] == contract["dataset_fingerprint"], "manifest 指紋不符")
    entries = manifest.get("runs")
    _require(isinstance(entries, list) and len(entries) == 6, "manifest run 數不足")
    expected = {(seed, method) for seed in contract["seeds"] for method in contract["methods"]}
    conditions = {(motor, rpm) for motor in contract["motors"] for rpm in contract["rpms"]}
    seen, rows, provenance, count_reference = set(), [], [], {}
    for entry in entries:
        _require(isinstance(entry, dict), "manifest run 必須是物件")
        key = (entry.get("seed"), entry.get("method"))
        _require(type(key[0]) is int and key in expected and key not in seen, "run 重複或 seed／方法錯誤")
        seen.add(key)
        seed, method = key
        run_id = f"seed_{seed}_{method}"
        name = f"seed_{seed}/{method}.json"
        _require(entry.get("run_id") == run_id and entry.get("status") == "completed", f"run 身分或狀態錯誤：{run_id}")
        _require(str(entry.get("path", "")).replace("\\", "/") == name, f"run 路徑錯誤：{run_id}")
        result = _json(snapshot[name])
        _require(set(RUN_FIELDS).issubset(result), f"run 必備欄位缺失：{run_id}")
        for field, value in {"schema_version": 1, "experiment": "health_index_benchmark", "status": "completed",
                             "seed": seed, "method": method, "formal_condition_count": 9,
                             "dataset_fingerprint": contract["dataset_fingerprint"], "confidence": contract["confidence"],
                             "score_direction": "higher_is_worse", "health_direction": "higher_is_better",
                             "unknown_calibration_leakage": False, "rul_available": False,
                             "dataset_root": manifest.get("data_root")}.items():
            _require(result[field] == value and type(result[field]) is type(value), f"run 欄位不符：{run_id}/{field}")
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", str(result["commit_sha"]))), f"來源 commit 格式錯誤：{run_id}")
        _require(bool(re.fullmatch(r"\d+\.\d+\.\d+", str(result["python"]))), f"來源 Python 格式錯誤：{run_id}")
        run_rows = result["rows"]
        _require(isinstance(run_rows, list) and len(run_rows) == 9, f"run 工況數不足：{run_id}")
        seen_conditions = set()
        for row in run_rows:
            _require(isinstance(row, dict) and set(row) == set(ROW_FIELDS), f"列欄位集合不符：{run_id}")
            condition = (row["motor"], row["rpm"])
            _require(condition in conditions and condition not in seen_conditions, f"重複或錯誤工況：{run_id}")
            seen_conditions.add(condition)
            _require(row["dataset"] == dataset_key(row["motor"], row["rpm"]), f"dataset 與 motor/rpm 不符：{run_id}")
            _require(type(row["seed"]) is int and row["seed"] == seed and row["method"] == method
                     and row["status"] == "completed" and row["rul_available"] is False, f"列身分不符：{run_id}")
            _require(row["mahalanobis_method"] == ("ledoit_wolf" if method == "mahalanobis" else None), "共變異數方法不符")
            _require((type(row["knn_neighbors"]) is int and row["knn_neighbors"] == 5) if method == "knn"
                     else row["knn_neighbors"] is None, "鄰居數不符")
            counts = tuple(_integer(row[field], field) for field in COUNTS)
            _require(row["n_unknown_classes"] == 9, "未知配置數不符")
            _require(count_reference.setdefault(condition, counts) == counts, "配對來源樣本數不符")
            for field in NUMERIC:
                value = _number(row[field], field)
                if field == "health_gap_known_minus_unknown":
                    _require(-1 <= value <= 1, f"健康差距超出範圍：{run_id}")
                elif field not in ("known_score_median", "unknown_score_median", "calibration_healthy_anchor", "calibration_critical_anchor"):
                    _require(0 <= value <= 1, f"數值超出範圍：{field}")
                else:
                    _require(value >= 0, f"分數或錨點小於零：{field}")
            _require(row["calibration_critical_anchor"] > row["calibration_healthy_anchor"], "校準錨點順序錯誤")
            effect = row["health_effect_size"]
            if effect is None:
                _require(row["known_health_std"] == row["unknown_health_std"] == 0, "效果量缺失但分母非零")
            else:
                _number(effect, "health_effect_size")
            _require(math.isclose(row["health_gap_known_minus_unknown"], row["known_health_mean"]-row["unknown_health_mean"],
                                 abs_tol=1.5e-6, rel_tol=0), "健康差距與平均不符")
            rows.append(dict(row))
        _require(seen_conditions == conditions, f"工況矩陣不完整：{run_id}")
        _csv_matches_json(snapshot[name[:-5]+".csv"], run_rows, run_id, contract)
        provenance.append({"run_id": run_id, "path": name, "schema_version": result["schema_version"],
                           "source_commit": result["commit_sha"], "source_python": result["python"], "rows": len(run_rows)})
    _require(seen == expected and len(rows) == 54, "完整矩陣必須有六 run、54 列")
    old = _json(snapshot["aggregate_summary.json"])
    _require(old.get("dataset_fingerprint") == contract["dataset_fingerprint"], "舊彙總指紋不符")
    rows.sort(key=lambda row: (row["method"], row["dataset"], row["seed"]))
    inventory = {"contract_sha256": sha256_bytes(contract_bytes), "dataset_fingerprint": contract["dataset_fingerprint"],
                 "files": [{"path": name, "sha256": sha256_bytes(data), "bytes": len(data), "pinned": name not in UNPINNED_SOURCES}
                           for name, data in sorted(snapshot.items())],
                 "runs": sorted(provenance, key=lambda item: item["run_id"]),
                 "configuration_names_status": "UNKNOWN：來源 JSON 只記錄未知配置數，不含名稱或逐窗 ID"}
    return contract, rows, {"inventory": inventory, "snapshot": snapshot}


def aggregate_rows(rows: list[dict], contract: dict) -> tuple[dict, list[dict]]:
    """等權列平均與 ddof=1；不重建模型、不產生選定方法。"""
    summary = {"schema_version": 1, "experiment": "health_index_benchmark", "seeds": sorted(contract["seeds"]),
               "methods": sorted(contract["methods"]), "runs": 6, "rows": len(rows),
               "dataset_fingerprint": contract["dataset_fingerprint"], "overall": {}}
    table, raw_means = [], {}
    for method in summary["methods"]:
        subset = [row for row in rows if row["method"] == method]
        _require(len(subset) == 27, "每方法必須有 27 列")
        raw_means[method] = {}
        summary["overall"][method] = {}
        for metric in METRICS:
            values = [_number(row[metric], metric) for row in subset]
            mean, std = statistics.mean(values), statistics.stdev(values)
            raw_means[method][metric] = mean
            summary["overall"][method][metric] = {"mean": round(mean, 6), "std": round(std, 6)}
        for dataset in sorted({row["dataset"] for row in subset}):
            group = [row for row in subset if row["dataset"] == dataset]
            _require({row["seed"] for row in group} == set(contract["seeds"]) and len(group) == 3, "逐工況 seed 不完整")
            record = {"index": len(table), "method": method, "dataset": dataset}
            for metric in METRICS:
                values = [_number(row[metric], metric) for row in group]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_std"] = statistics.stdev(values)
            table.append(record)
    # 差值用已 round 的兩個平均相減，讀者手算 overall 兩欄會得到同一個數（與封存公式一致，#288）
    overall = summary["overall"]
    summary["difference_knn_minus_mahalanobis"] = {
        metric: round(overall["knn"][metric]["mean"]-overall["mahalanobis"][metric]["mean"], 6) for metric in METRICS}
    return summary, table


def _flatten(value: object, prefix: str = "") -> dict:
    if isinstance(value, dict):
        if not value:
            return {prefix: {}}
        return {key: item for field, nested in sorted(value.items())
                for key, item in _flatten(nested, f"{prefix}/{field}").items()}
    return {prefix: value}


def _difference(key: str, old: object, new: object, atol: float, rtol: float, exact: bool = False) -> dict:
    numeric = not exact and type(old) in (int, float) and type(new) in (int, float)
    delta = float(new)-float(old) if numeric else None
    tolerance = atol+rtol*abs(float(old)) if numeric else 0.0
    matches = math.isfinite(delta) and abs(delta) <= tolerance if numeric else type(old) is type(new) and old == new
    return {"key": key, "old": old, "new": new, "difference": delta, "allowed_tolerance": tolerance,
            "matches": bool(matches)}


def compare(snapshot: dict[str, bytes], summary: dict, table: list[dict], contract: dict) -> list[dict]:
    """欄位集合、匹配鍵與值一起比對，不只核對總平均。"""
    old = _flatten(_json(snapshot["aggregate_summary.json"]))
    new = _flatten(summary)
    differences = []
    for key in sorted(set(old) | set(new)):
        exact = not (key.startswith("/overall/") or key.startswith("/difference_knn_minus_mahalanobis/"))
        differences.append(_difference("summary"+key, old.get(key, "<缺欄>"), new.get(key, "<缺欄>"),
                                       contract["summary_atol"], contract["summary_rtol"], exact))
    fields, old_table = _csv(snapshot["aggregate_by_condition.csv"])
    new_fields = list(table[0])
    differences.append(_difference("condition/columns", sorted(fields), sorted(new_fields), 0, 0, True))
    old_lookup = {}
    for row in old_table:
        _require("method" in row and "dataset" in row, "舊逐工況彙總缺匹配鍵")
        key = (row["method"], row["dataset"])
        _require(key not in old_lookup, "舊逐工況彙總重複鍵")
        old_lookup[key] = row
    new_lookup = {(row["method"], row["dataset"]): row for row in table}
    for key in sorted(set(old_lookup) | set(new_lookup)):
        left, right = old_lookup.get(key, {}), new_lookup.get(key, {})
        for field in sorted(set(fields) | set(new_fields)):
            a, b = left.get(field, "<缺欄>"), right.get(field, "<缺欄>")
            exact = field in ("index", "method", "dataset")
            if not exact and a != "<缺欄>":
                a = float(a)
                _require(math.isfinite(a), f"舊彙總非有限值：{key}/{field}")
            if field == "index" and a != "<缺欄>":
                a = int(a)
            differences.append(_difference(f"condition/{key[0]}/{key[1]}/{field}", a, b,
                                           contract["csv_atol"], contract["csv_rtol"], exact))
    return differences


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source_root: Path | str = DEFAULT_SOURCE, contract_path: Path | str = DEFAULT_CONTRACT) -> dict:
    """重算並寫入 logger 配置的 logs/output；封存只讀。"""
    logger, paths = setup_run("exp8_health_index_aggregate")
    root = Path(source_root).resolve()
    output = paths.output_dir.resolve()
    _require(not output.is_relative_to(root) and not root.is_relative_to(output), "輸出不得覆寫封存來源")
    try:
        contract, rows, evidence = load_matrix(root, contract_path)
        summary, table = aggregate_rows(rows, contract)
        differences = compare(evidence["snapshot"], summary, table, contract)
        # 核對處理期間未修改來源；第二次也檢查額外或缺失檔案。
        _require(_snapshot(root, contract) == evidence["snapshot"], "來源在重算期間變動")
        packages = {}
        for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "hdbscan", "loguru", "tornado"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "未安裝"
        # 沒有 .git 或 git 指令時記 None，不讓重算整個失敗（#269 C16）
        sha = git_head(ROOT)
        audit = {"status": "COMPLETED", "comparison_status": "MATCH" if all(row["matches"] for row in differences) else "DIFFERENT",
                 "runs": 6, "rows": 54, "condition_groups": len(table), "compared_fields": len(differences),
                 "mismatches": sum(not row["matches"] for row in differences), "source_unchanged": True,
                 "historical_formula_status": "UNKNOWN", "source_root": str(root), "contract_path": str(Path(contract_path).resolve()),
                 "contract_sha256": evidence["inventory"]["contract_sha256"], "commit_sha": sha,
                 "python": platform.python_version(), "packages": packages, "argv": sys.argv,
                 "model_fit_performed": False, "data_loaded": False,
                 "limitations": ["原生成程式與 ddof 未提交；吻合不證實歷史公式", "配置名稱、逐窗與採集來源仍未記錄", "未提供新的可靠性或盲測證據"]}
        _write_json(output / "aggregate_summary.json", summary)
        _write_csv(output / "aggregate_by_condition.csv", table)
        _write_csv(output / "field_differences.csv", differences)
        _write_json(output / "source_inventory.json", evidence["inventory"])
        _write_json(output / "recomputation_audit.json", audit)
        logger.info(f"已重算六 run、54 列；比對 {len(differences)} 欄，差異 {audit['mismatches']} 欄；未擬合模型")
        return {**audit, "output_dir": str(output), "log_file": str(paths.log_file.resolve())}
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        _write_json(output / "failure.json", {"status": "FAILED", "error": str(exc), "source_root": str(root)})
        logger.error(f"重算失敗：{exc}")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE, help="唯讀封存目錄")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT, help="事前提交的來源規範")
    args = parser.parse_args(argv)
    try:
        result = run(args.source_root, args.contract)
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError):
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["comparison_status"] == "MATCH" else 3


if __name__ == "__main__":
    raise SystemExit(main())
