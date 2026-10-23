"""實驗六矩陣：Resumable formal exp6 matrix runner (9 conditions × seeds × methods)."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from core.data import PoolCache, discover_datasets
from core.logger import setup_run
from core.openset import resolve_openset_config
from core.provenance import (CONTENT_VERSION, ProvenanceError, canonical_sha, capture_source,
                             file_sha, sha256_bytes, write_private_context)
from core.runtime_environment import collect_environment, source_data_info
from experiments.exp6_formal_benchmark import (
    FORMAL_CONDITION_COUNT,
    FORMAL_METHODS,
    dataset_fingerprint,
    run,
)


DEFAULT_SEEDS = (42, 123, 2026)
MATRIX_SCHEMA = 2
MATRIX_CONTRACT = "formal_matrix_provenance_v2"
RUN_CONTRACT = "formal_matrix_run_v2"
ARTIFACT_NAMES = ("summary.json", "results.csv", "run.log")
PROJECT = Path(__file__).resolve().parents[2]
CODE_FILES = ("src/core/data.py", "src/core/feature_schema.py", "src/core/logger.py",
              "src/core/openset.py", "src/core/mahalanobis.py",
              "src/core/runner.py", "src/core/provenance.py", "src/core/runtime_environment.py",
              "src/experiments/exp6_formal_benchmark.py", "src/experiments/exp6_matrix.py")


def method_configuration(method: str) -> dict:
    """記錄矩陣原有固定參數；兩個方法各自解析，不比較無效參數。"""
    return {"resolved_config": resolve_openset_config(method, confidence=0.95,
                mahalanobis_method="ledoit_wolf", knn_neighbors=5),
            "benchmark_config": {
                "feature_count": 105,
                "normalization": "RobustScaler fit on known training split only",
                "split": "known health 60/20/20 train/calibration/holdout; all other configurations unknown positive",
                "threshold": 1.0, "threshold_strategy": "known_calibration_quantile_0.95",
                "openset_method": method,
                "mahalanobis_method": "ledoit_wolf" if method == "mahalanobis" else None,
                "knn_neighbors": 5 if method == "knn" else None, "checkpoint_identifier": None}}


def _execution_identity(environment: dict) -> dict:
    return {key: copy.deepcopy(environment.get(key)) for key in
            ("git", "constraints_sha256", "python", "numeric", "packages")} | {
            "platform": {key: environment.get("os", {}).get(key) for key in ("system", "machine")},
            "code_sha256": {name: file_sha(PROJECT / name) for name in CODE_FILES}}


def matrix_contract(data_root, matrix: list[dict], methods: Sequence[str]) -> tuple[dict, dict]:
    """固定起始資料及程式身分；不消耗模型 RNG。"""
    source, _ = capture_source(data_root)
    environment = collect_environment(data_root=data_root)
    if environment["data"]["dataset_fingerprint"] != source["dataset_fingerprint"]:
        raise ProvenanceError("矩陣起始來源與環境不符")
    contract = {"version": MATRIX_CONTRACT, "source": source,
                "execution": _execution_identity(environment),
                "method_configs": {method: method_configuration(method) for method in methods},
                "run_keys": [{key: row[key] for key in ("run_id", "motor", "rpm", "seed", "method")}
                             for row in matrix]}
    validate_contract(contract)
    return contract, environment


def validate_contract(contract: dict) -> None:
    """只核對可讀契約；彙總不要求生產者 HEAD 等於目前讀取工具 HEAD。"""
    if not isinstance(contract, dict) or contract.get("version") != MATRIX_CONTRACT:
        raise ValueError("矩陣來源契約版本不符，請使用新輸出根")
    source = contract.get("source", {})
    if (not isinstance(source, dict) or source.get("schema_version") != CONTENT_VERSION or not source.get("files")
            or source.get("dataset_fingerprint") != canonical_sha(source["files"])
            or source.get("data_version") != f"{CONTENT_VERSION}:{source.get('dataset_fingerprint')}"):
        raise ValueError("矩陣資料版本或指紋契約不符")
    files = source["files"]
    if (not isinstance(files, list) or any(not isinstance(row, dict)
            or not isinstance(row.get("source_id"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))) for row in files)):
        raise ValueError("矩陣檔案清冊欄位不合法")
    identities = [row["source_id"] for row in files]
    if (len(set(identities)) != len(identities) or any(
            "\\" in identity or ":" in identity or len(identity.split("/")) != 6
            or any(part in {"", ".", ".."} for part in identity.split("/"))
            or not identity.startswith("Step-") or identity.split("/")[1] != "myfeature"
            for identity in identities)):
        raise ValueError("矩陣source ID不是唯一portable正式檔案路徑")
    execution = contract.get("execution", {})
    if not isinstance(execution, dict) or not isinstance(execution.get("git"), dict):
        raise ValueError("矩陣執行身分欄位缺少")
    git = execution.get("git", {})
    code = execution.get("code_sha256")
    if not isinstance(code, dict):
        raise ValueError("矩陣程式內容SHA清冊缺少")
    if (git.get("status") != "VERIFIED" or not re.fullmatch(r"[0-9a-f]{40}", str(git.get("head", "")))
            or not isinstance(git.get("tracked_dirty"), bool)
            or not re.fullmatch(r"[0-9a-f]{64}", str(execution.get("constraints_sha256", "")))
            or set(code) != set(CODE_FILES)
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(value))
                   for value in code.values())):
        raise ValueError("矩陣缺少可信 Git／鎖版／程式內容身分")
    if (not all(isinstance(execution.get(key), dict) and execution[key]
                for key in ("python", "numeric", "packages", "platform"))
            or not execution["python"].get("version") or not execution["numeric"].get("blas")):
        raise ValueError("矩陣Python／數值環境證據不足")
    configs = contract.get("method_configs", {})
    if (not isinstance(configs, dict) or not configs or set(configs) - set(FORMAL_METHODS)
            or any(value != method_configuration(method) for method, value in configs.items())):
        raise ValueError("矩陣有效設定契約不符")
    keys = contract.get("run_keys")
    if (not isinstance(keys, list) or not keys
            or any(not isinstance(row, dict) or not {"run_id", "motor", "rpm", "seed", "method"}.issubset(row)
                   for row in keys)
            or len({row["run_id"] for row in keys}) != len(keys)):
        raise ValueError("矩陣預期鍵缺少或重複")
    for row in keys:
        if row["method"] not in configs or row["run_id"] != run_key(row["motor"], row["rpm"], row["seed"], row["method"]):
            raise ValueError("矩陣預期鍵與方法不符")


def run_directory(root: Path, identity: str) -> Path:
    """產物不得借相對路徑或 symlink 跳出所選矩陣根。"""
    if not isinstance(identity, str) or not identity or any(c in identity for c in ("/", "\\", ":")) or identity in {".", ".."}:
        raise ValueError("run ID不是portable單一名稱")
    target = root / "runs" / identity
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("run產物跳出矩陣根") from None
    return target


def artifact_snapshot(run_dir: Path) -> dict[str, bytes]:
    payloads = {}
    for name in ARTIFACT_NAMES:
        path = run_dir / name
        if path.is_symlink():
            raise ValueError("run產物不得以symlink指向其他檔案")
        try:
            payloads[name] = path.read_bytes()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError(f"run產物無法讀取：{type(exc).__name__}") from None
    return payloads


def artifact_hashes(payloads: dict[str, bytes]) -> dict[str, str]:
    return {name: sha256_bytes(payload) for name, payload in sorted(payloads.items())}


def run_contract_problems(summary: dict, expected: dict, contract: dict) -> list[str]:
    """版本／資料／程式／有效設定不相容不能用partial掩蓋。"""
    source = contract["source"]
    config = contract["method_configs"].get(expected["method"])
    if config is None:
        return ["summary方法不在固定契約內"]
    proof = {"version": RUN_CONTRACT, "contract_sha256": canonical_sha(contract),
             "fingerprint_version": source["schema_version"], "data_version": source["data_version"],
             "effective_config_sha256": canonical_sha(config), "execution": contract["execution"]}
    reasons = []
    if summary.get("matrix_provenance") != proof:
        reasons.append("summary來源／版本／程式契約不符")
    if (summary.get("source") != source or summary.get("fingerprint_version") != source["schema_version"]
            or summary.get("commit_sha") != contract["execution"]["git"]["head"]):
        reasons.append("summary資料或執行程式身分不符")
    if (summary.get("resolved_config") != config["resolved_config"]
            or summary.get("config") != config["benchmark_config"]):
        reasons.append("summary實際有效設定不符")
    if summary.get("environment", {}).get("data", {}).get("dataset_fingerprint") != source["dataset_fingerprint"]:
        reasons.append("summary環境來源與固定資料不符")
    return reasons


def validate_manifest_contract(manifest: dict, *, allow_missing_runs: bool = False) -> dict:
    """新契約不能降級成 legacy；固定部分不得由進度欄位覆寫。"""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("legacy矩陣只讀，請使用新輸出根")
    contract = manifest.get("contract")
    validate_contract(contract)
    if manifest.get("contract_sha256") != canonical_sha(contract):
        raise ValueError("矩陣contract SHA不符")
    source = contract["source"]
    if (manifest.get("fingerprint_version") != source["schema_version"]
            or manifest.get("data_version") != source["data_version"]):
        raise ValueError("矩陣資料版本欄位不符")
    if (manifest.get("source_verification_status") != "VERIFIED"
            or set(manifest.get("methods", [])) != set(contract["method_configs"])
            or set(manifest.get("seeds", [])) != {row["seed"] for row in contract["run_keys"]}
            or manifest.get("expected_runs") != len(contract["run_keys"])):
        raise ValueError("矩陣來源狀態或固定coverage不符")
    actual = [{key: row.get(key) for key in ("run_id", "motor", "rpm", "seed", "method")}
              for row in manifest.get("runs", [])]
    if allow_missing_runs:
        expected = {row["run_id"]: row for row in contract["run_keys"]}
        if any(row != expected.get(row["run_id"]) for row in actual):
            raise ValueError("矩陣run身分不符固定coverage")
    elif sorted(actual, key=lambda x: x["run_id"]) != sorted(contract["run_keys"], key=lambda x: x["run_id"]):
        raise ValueError("矩陣run清單不符固定coverage")
    return contract


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


def validate_run(run_dir: Path, expected: dict, fingerprint: str | None = None,
                 *, contract: dict | None = None, payloads: dict | None = None) -> list[str]:
    """核對單格產物，回傳全部不符原因；空清單代表完成。

    檢查項目：summary.json 可讀、status=completed、method/seed 相符、有
    dataset_fingerprint 且（給定時）相同、恰一列且該列 status/motor/rpm/method/seed
    相符；results.csv 恰一列、含全部身分與指標欄且與 summary 列一致；run.log
    含 status=completed、seed=、method= 與（給定時）condition= 行。

    ``expected`` 需含 method、seed，可含 motor、rpm、condition。
    ``fingerprint`` 給定時 summary 的 dataset_fingerprint 必須相同。
    """
    reasons: list[str] = []
    payloads = artifact_snapshot(run_dir) if payloads is None else payloads
    if "summary.json" not in payloads:
        return ["missing summary.json"]
    try:
        summary = json.loads(payloads["summary.json"])
    except (UnicodeError, json.JSONDecodeError) as exc:
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

    if "results.csv" not in payloads:
        reasons.append("missing results.csv")
    else:
        try:
            frame = pd.read_csv(io.BytesIO(payloads["results.csv"]))
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

    if "run.log" not in payloads:
        reasons.append("missing run.log")
    else:
        lines = set(payloads["run.log"].decode("utf-8", errors="replace").splitlines())
        required = {"status=completed", f"seed={int(expected['seed'])}", f"method={expected['method']}"}
        if expected.get("condition"):
            required.add(f"condition={expected['condition']}")
        for line in sorted(required - lines):
            reasons.append(f"run.log lacks '{line}'")
    if contract is not None:
        reasons.extend(run_contract_problems(summary, expected, contract))
        if expected.get("artifact_sha256") != artifact_hashes(payloads):
            reasons.append("run原始bytes SHA不符或缺少")
    return reasons


def _is_complete(summary_path: Path, expected: dict, fingerprint: str | None = None,
                 *, contract: dict | None = None) -> bool:
    return not validate_run(summary_path.parent, expected, fingerprint, contract=contract)


def _condition_result(result: dict, entry: dict) -> dict:
    """benchmark 結果只留目標工況那一列；找不到或多於一列就丟 ValueError。不修改輸入。"""
    condition_rows = [
        row
        for row in result["rows"]
        if row.get("motor") == entry["motor"] and row.get("rpm") == entry["rpm"]
    ]
    if len(condition_rows) != 1:
        raise ValueError(f"expected one row for {entry['condition']}, found {len(condition_rows)}")
    return dict(result, formal_condition_count=1, rows=condition_rows)


def _write_run_csv(run_dir: Path, result: dict) -> None:
    frame = pd.DataFrame(result["rows"])
    temporary = run_dir / "results.csv.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, run_dir / "results.csv")


def _assert_source(data_root: Path, contract: dict) -> None:
    if capture_source(data_root)[0] != contract["source"]:
        raise ProvenanceError("矩陣執行期間來源變動，不能沿用或完成")
    if _execution_identity(collect_environment(data=source_data_info(None))) != contract["execution"]:
        raise ProvenanceError("矩陣執行期間程式或數值環境變動")


def _preserve_attempt(run_dir: Path) -> dict | None:
    payloads = artifact_snapshot(run_dir)
    if not payloads:
        return None
    root = run_dir / "attempts"
    try:
        root.resolve().relative_to(run_dir.resolve())
    except ValueError:
        raise ValueError("嘗試備份目錄跳出run目錄") from None
    root.mkdir(exist_ok=True)
    index = 1
    while (root / f"{index:04d}").exists():
        index += 1
    destination = root / f"{index:04d}"
    destination.mkdir()
    for name, payload in payloads.items():
        (destination / name).write_bytes(payload)
    return {"attempt_id": f"attempts/{index:04d}", "artifact_sha256": artifact_hashes(payloads)}


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
    if not matrix or len({row["run_id"] for row in matrix}) != len(matrix):
        raise ValueError("矩陣預期鍵不能為空或重複")
    contract, environment = matrix_contract(data_path, matrix, methods)
    fingerprint = contract["source"]["dataset_fingerprint"]
    if dataset_fingerprint(data_path) != fingerprint:
        raise ProvenanceError("共用benchmark與矩陣的來源指紋不符")
    contract_sha = canonical_sha(contract)
    manifest_path = output_path / "matrix_manifest.json"
    original_bytes = None
    if manifest_path.exists():
        if not resume:
            raise ValueError("已有矩陣不能以no-resume覆寫，請使用新輸出根")
        original_bytes = manifest_path.read_bytes()
        try:
            manifest = json.loads(original_bytes)
        except (ValueError, UnicodeError):
            raise ValueError("既有矩陣manifest損壞，保留原檔並使用新根") from None
        saved_contract = validate_manifest_contract(manifest)
        if saved_contract != contract:
            raise ValueError("來源、有效設定或程式版本不相容，拒絕混版續跑")
        for entry in manifest["runs"]:
            run_dir = run_directory(output_path, entry["run_id"])
            if entry["status"] == "completed":
                reasons = validate_run(run_dir, entry, fingerprint, contract=contract)
                if reasons:
                    raise ValueError("完成格證據不符，保留原檔並使用新根：" + entry["run_id"])
        manifest = copy.deepcopy(manifest)
    else:
        if output_path.exists() and any(output_path.iterdir()):
            raise ValueError("非空輸出根沒有相容manifest，不能挪作新矩陣")
        manifest = {
            "schema_version": MATRIX_SCHEMA, "status": "running", "data_root": data_path.name,
            "seeds": [int(seed) for seed in seeds], "methods": list(methods),
            "expected_runs": len(matrix), "runs": matrix,
            "contract": contract, "contract_sha256": contract_sha, "environment": environment,
            "fingerprint_version": contract["source"]["schema_version"],
            "data_version": contract["source"]["data_version"],
            "source_verification_status": "VERIFIED",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    _assert_source(data_path, contract)
    if original_bytes is not None and manifest_path.read_bytes() != original_bytes:
        raise ValueError("唯讀核對期間manifest變動，拒絕覆寫")
    manifest["private_context_id"] = write_private_context(capture_source(data_path)[1])
    manifest["data_root"] = data_path.name
    manifest["environment"] = environment
    output_path.mkdir(parents=True, exist_ok=True)
    manifest["status"] = "running"
    _atomic_json(manifest_path, manifest)
    pool_cache = PoolCache()
    source_changed = False

    for entry in manifest["runs"]:
        run_dir = run_directory(output_path, entry["run_id"])
        summary_path = run_dir / "summary.json"
        try:
            _assert_source(data_path, contract)
        except (ProvenanceError, OSError, ValueError):
            source_changed = True
            manifest["source_verification_status"] = "INCOMPLETE_SOURCE_CHANGED"
            break
        if entry["status"] == "completed":
            if not _is_complete(summary_path, entry, fingerprint, contract=contract):
                manifest.update(status="incomplete", source_verification_status="INCOMPLETE")
                _atomic_json(manifest_path, manifest)
                raise ValueError("完成格在續跑核對後變動，拒絕沿用")
            entry["resumed"] = True
            if log is not None:
                log.info("{}：沿用已核對bytes", entry["run_id"])
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt = _preserve_attempt(run_dir)
        if attempt is not None:
            entry.setdefault("attempt_history", []).append(
                {**attempt, "status": entry["status"], "error_type": entry.get("error_type")})
        for key in ("error", "error_type", "resumed"):
            entry.pop(key, None)
        entry.update(status="running", started_at_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(manifest_path, manifest)
        log_path = run_dir / "run.log"
        log_path.write_text(
            f"started={entry['started_at_utc']}\ncondition={entry['condition']}\n"
            f"seed={entry['seed']}\nmethod={entry['method']}\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            result = run(data_path, seed=entry["seed"], openset_method=entry["method"],
                         confidence=0.95, mahalanobis_method="ledoit_wolf", knn_neighbors=5,
                         require_nine=True, evidence_dir=run_dir, pool_cache=pool_cache)
            _assert_source(data_path, contract)
            result = _condition_result(result, entry)
            config = contract["method_configs"][entry["method"]]
            result["matrix_provenance"] = {
                "version": RUN_CONTRACT, "contract_sha256": contract_sha,
                "fingerprint_version": contract["source"]["schema_version"],
                "data_version": contract["source"]["data_version"],
                "effective_config_sha256": canonical_sha(config), "execution": contract["execution"]}
            _atomic_json(summary_path, result)
            _write_run_csv(run_dir, result)
            log_path.write_text(log_path.read_text(encoding="utf-8")
                               + "status=completed\n", encoding="utf-8")
            payloads = artifact_snapshot(run_dir)
            entry["artifact_sha256"] = artifact_hashes(payloads)
            reasons = validate_run(run_dir, entry, fingerprint, contract=contract, payloads=payloads)
            if reasons:
                raise ProvenanceError("benchmark回傳不符固定來源或設定")
            entry.update(status="completed", dataset_fingerprint=fingerprint,
                         fingerprint_version=contract["source"]["schema_version"],
                         data_version=contract["source"]["data_version"],
                         commit_sha=result["commit_sha"],
                         summary=f"runs/{entry['run_id']}/summary.json")
        except Exception as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
            log_path.write_text(log_path.read_text(encoding="utf-8")
                               + f"status=failed\nerror_type={type(exc).__name__}\n", encoding="utf-8")
            if log is not None:
                log.error("{}：失敗類型{}；保留證據，請查來源與設定", entry["run_id"], type(exc).__name__)
        entry.update(duration_seconds=round(time.perf_counter() - started, 6),
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        log_path.write_text(log_path.read_text(encoding="utf-8")
                           + f"duration_seconds={entry['duration_seconds']}\n", encoding="utf-8")
        entry["artifact_sha256"] = artifact_hashes(artifact_snapshot(run_dir))
        try:
            _assert_source(data_path, contract)
        except (ProvenanceError, OSError, ValueError):
            source_changed = True
            manifest["source_verification_status"] = "INCOMPLETE_SOURCE_CHANGED"
        _atomic_json(manifest_path, manifest)
        if source_changed:
            break
    try:
        _assert_source(data_path, contract)
    except (ProvenanceError, OSError, ValueError):
        source_changed = True
        manifest["source_verification_status"] = "INCOMPLETE_SOURCE_CHANGED"
    for entry in manifest["runs"]:
        if entry["status"] == "completed" and validate_run(
                run_directory(output_path, entry["run_id"]), entry, fingerprint, contract=contract):
            entry.update(status="failed", error_type="ArtifactChanged")
    completed = sum(entry["status"] == "completed" for entry in manifest["runs"])
    failed = sum(entry["status"] == "failed" for entry in manifest["runs"])
    manifest.update(
        status="completed" if completed == len(matrix) and not source_changed else "incomplete",
        completed_runs=completed, failed_runs=failed, missing_runs=len(matrix) - completed - failed,
        finished_at_utc=datetime.now(timezone.utc).isoformat())
    if log is not None:
        log.info("CSV cache：命中{}、未命中{}", pool_cache.hits, pool_cache.misses)
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", default="data")
    # 預設開新的時間戳目錄；output/exp6_formal_matrix 本身是 formal_local 歷史封存，不得被續跑覆寫（#253）
    parser.add_argument("--output-root", default=None,
                        help="輸出目錄（預設 output/exp6_formal_matrix/<時間戳>；續跑時傳同一個目錄）")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--method", action="append", choices=FORMAL_METHODS, dest="methods")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    log, _ = setup_run("exp6_formal_matrix", make_output=False, data_root=args.data_root)
    log.info("矩陣seed={}、method={}、resume={}", args.seeds, args.methods, not args.no_resume)
    output_root = args.output_root or f"output/exp6_formal_matrix/{datetime.now():%Y-%m-%d-%H-%M-%S}"
    manifest = run_matrix(
        args.data_root,
        output_root,
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
