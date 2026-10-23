"""實驗八：Formal health-index benchmark using the existing Open Set split contract."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from core.data import HEALTHY, dataset_key, discover_datasets, make_split
from core.logger import setup_run
from core.openset import resolve_openset_config
from core.runner import run_root_sources
from core.runtime_environment import git_head
from experiments.health.evaluation import evaluate_open_set
from experiments.health.index import CalibratedHealthIndex
from experiments.health.severity import DEFAULT_SEVERITY_POLICY


FORMAL_CONDITION_COUNT = 9
# 健康映射或結果欄位一改就升版；矩陣續跑只沿用同版結果（#301 F-01）。
# 2：三錨點映射（拒絕線映到 0.5、兩倍拒絕線映到 0）。
RESULT_SCHEMA_VERSION = 2


def _validate_datasets(datasets: list[dict], data_path: Path, require_nine: bool) -> None:
    if not datasets:
        raise FileNotFoundError(f"formal data not found under {data_path}")
    if require_nine and len(datasets) != FORMAL_CONDITION_COUNT:
        raise ValueError(
            f"health benchmark requires {FORMAL_CONDITION_COUNT} conditions; "
            f"discovered {len(datasets)} under {data_path}"
        )


def _health_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "median": round(float(np.median(values)), 6),
        "healthy_fraction": round(float(np.mean(values >= DEFAULT_SEVERITY_POLICY.healthy_min)), 6),
        "critical_fraction": round(float(np.mean(values < DEFAULT_SEVERITY_POLICY.degraded_min)), 6),
    }


def _effect_size(known: np.ndarray, unknown: np.ndarray) -> float | None:
    pooled = np.sqrt((np.var(known) + np.var(unknown)) / 2.0)
    if pooled <= np.finfo(float).eps:
        return None
    return round(float((np.mean(known) - np.mean(unknown)) / pooled), 6)


def run(
    data_root: Path | str = "data",
    seed: int = 42,
    openset_method: str = "mahalanobis",
    mahalanobis_method: str = "ledoit_wolf",
    confidence: float = 0.95,
    knn_neighbors: int = 5,
    *,
    require_nine: bool = True,
    pool_cache=None,
    output_dir=None,
) -> dict:
    """Run one seed/method across all formal conditions.

    Only healthy train/calibration windows fit the scaler, detector and health
    mapping.  Unknown configuration labels are consumed after scoring for
    evaluation only; they never alter the health scale.
    """

    return run_root_sources(_run, data_root, output_dir=output_dir, pool_cache=pool_cache,
                            seed=seed, openset_method=openset_method,
                            mahalanobis_method=mahalanobis_method, confidence=confidence,
                            knn_neighbors=knn_neighbors, require_nine=require_nine)


def _run(data_root, seed, openset_method, mahalanobis_method, confidence,
         knn_neighbors, require_nine, *, source_run):
    resolved = resolve_openset_config(openset_method, confidence=confidence,
                                      mahalanobis_method=mahalanobis_method, knn_neighbors=knn_neighbors)
    canonical_method = resolved["openset_method"]
    data_path = Path(data_root).expanduser().resolve()
    datasets = sorted(discover_datasets(data_path), key=lambda row: (row["motor"], row["rpm"]))
    _validate_datasets(datasets, data_path, require_nine)
    if canonical_method == "mahalanobis" and mahalanobis_method == "legacy":
        raise ValueError("formal health benchmark rejects legacy training-distance calibration")
    random.seed(seed)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    fit_audit = {}
    for dataset in datasets:
        pools = source_run.load(dataset)
        healthy_pool = pools[HEALTHY]
        split = make_split(len(healthy_pool), rng)
        X_train = healthy_pool[split.train]
        X_cal = healthy_pool[split.cal]
        X_known = healthy_pool[split.holdout]
        unknown_configs = [config for config in sorted(pools) if config != HEALTHY]
        fit_audit[dataset_key(dataset["motor"], dataset["rpm"])] = {
            "known_configuration": HEALTHY, "train": split.train.tolist(),
            "calibration": split.cal.tolist(), "known_test": split.holdout.tolist(),
            "unknown_test": {config: {"pool_start": 0, "pool_stop": len(pools[config])}
                             for config in unknown_configs},
            "row_semantics": "zero_based_clean_pool_indices_not_raw_windows"}
        if not unknown_configs:
            raise ValueError(f"{dataset['motor']}/{dataset['rpm']} has no unknown configurations")
        X_unknown = np.vstack([pools[config] for config in unknown_configs])
        model = CalibratedHealthIndex.fit(
            X_train,
            np.zeros(len(X_train), dtype=int),
            X_cal,
            np.zeros(len(X_cal), dtype=int),
            openset_method=canonical_method,
            mahalanobis_method=mahalanobis_method,
            confidence=confidence,
            knn_neighbors=knn_neighbors,
            random_state=seed,
        )
        known_results = model.predict(X_known, condition=f"{dataset['motor']}/{dataset['rpm']}")
        unknown_results = model.predict(X_unknown, condition=f"{dataset['motor']}/{dataset['rpm']}")
        known_health = np.asarray([item.health_index for item in known_results], dtype=float)
        unknown_health = np.asarray([item.health_index for item in unknown_results], dtype=float)
        all_results = known_results + unknown_results
        open_set = evaluate_open_set(all_results, [0] * len(known_results) + [1] * len(unknown_results))
        known_dist = _health_distribution(known_health)
        unknown_dist = _health_distribution(unknown_health)
        metadata = model.metadata()
        rows.append(
            {
                "dataset": dataset_key(dataset["motor"], dataset["rpm"]),
                "motor": dataset["motor"],
                "rpm": dataset["rpm"],
                "seed": int(seed),
                "method": canonical_method,
                "mahalanobis_method": mahalanobis_method if canonical_method == "mahalanobis" else None,
                "knn_neighbors": knn_neighbors if canonical_method == "knn" else None,
                "n_train": int(len(X_train)),
                "n_calibration": int(len(X_cal)),
                "n_known_test": int(len(known_results)),
                "n_unknown_test": int(len(unknown_results)),
                "n_unknown_classes": int(len(unknown_configs)),
                "known_health_mean": known_dist["mean"],
                "known_health_std": known_dist["std"],
                "known_health_median": known_dist["median"],
                "known_healthy_fraction": known_dist["healthy_fraction"],
                "unknown_health_mean": unknown_dist["mean"],
                "unknown_health_std": unknown_dist["std"],
                "unknown_health_median": unknown_dist["median"],
                "unknown_critical_fraction": unknown_dist["critical_fraction"],
                "health_gap_known_minus_unknown": round(float(np.mean(known_health) - np.mean(unknown_health)), 6),
                "health_effect_size": _effect_size(known_health, unknown_health),
                "known_score_median": round(float(np.median([item.openset_score for item in known_results])), 6),
                "unknown_score_median": round(float(np.median([item.openset_score for item in unknown_results])), 6),
                "open_set_accuracy": round(open_set["open_set_accuracy"], 6),
                "auroc": round(open_set["auroc"], 6) if open_set["auroc"] is not None else None,
                "aupr_unknown_positive": round(open_set["aupr_unknown_positive"], 6)
                if open_set["aupr_unknown_positive"] is not None
                else None,
                "unknown_recall": round(open_set["unknown_recall"], 6),
                "unknown_f1": round(open_set["unknown_f1"], 6),
                "calibration_healthy_anchor": metadata["calibration"]["healthy_anchor"],
                "calibration_warning_anchor": metadata["calibration"]["warning_anchor"],
                "calibration_critical_anchor": metadata["calibration"]["critical_anchor"],
                "rul_available": False,
                "status": "completed",
            }
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "experiment": "health_index_benchmark",
        "method": canonical_method,
        "resolved_config": resolved,
        "seed": int(seed),
        "confidence": float(confidence),
        "formal_condition_count": len(rows),
        "dataset_root": "formal_features",
        "dataset_fingerprint": _dataset_fingerprint(data_path),
        "score_direction": "higher_is_worse",
        "health_direction": "higher_is_better",
        "health_interpretation": "relative calibration only; no physical damage or RUL claim",
        "unknown_calibration_leakage": False,
        "rul_available": False,
        "commit_sha": git_head() or "unknown",
        "python": platform.python_version(),
        "rows": rows,
        "fit_audit": fit_audit,
    }


def _dataset_fingerprint(data_path: Path) -> str:
    from experiments.exp6_formal_benchmark import dataset_fingerprint

    return dataset_fingerprint(data_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openset-method", default="mahalanobis")
    parser.add_argument("--method", dest="mahalanobis_method", default="ledoit_wolf")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    log, paths = setup_run("exp8_health_index_benchmark", unique=True, data_root=args.data_root)
    log.info("執行實驗八健康指數比較；來源絕對路徑僅留本機 sidecar")
    result = run(
        args.data_root,
        seed=args.seed,
        openset_method=args.openset_method,
        mahalanobis_method=args.mahalanobis_method,
        confidence=args.confidence,
        knn_neighbors=args.knn_neighbors,
        require_nine=not args.allow_partial,
        output_dir=paths.output_dir,
    )
    (paths.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(result["rows"]).to_csv(paths.output_dir / "rows.csv", index=False)
    log.info(f"完成：{len(result['rows'])} 列，結果寫入 {paths.output_dir}")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "method": result["method"],
                      "seed": result["seed"], "output_dir": str(paths.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
