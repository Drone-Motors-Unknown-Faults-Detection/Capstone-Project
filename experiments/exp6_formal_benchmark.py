"""實驗六（正式版）：Mahalanobis vs k-NN 開集基準，供 exp6_matrix 多 seed 矩陣使用。

與 ``exp6_osr_benchmark``（七種偵測器的廣度比較、core.detectors）互補：
本模組只比較主線兩種 core.openset 偵測器，並記錄完整可重現 provenance。
（原於 feat/knn-openset-comparison 分支名為 exp6_osr_benchmark，合併時更名；
該分支 output/、logs/ 下 2026-09-19 的 exp6_osr_benchmark 紀錄即出自本模組。）

    venv/bin/python -m experiments.exp6_formal_benchmark --openset-method knn --seed 42

Formal OSR benchmark using the shared :mod:`core.openset` factory.

The benchmark deliberately has one detector construction path for both
methods.  It evaluates a health-only open-set task: ``8screws`` is the known
class and every other configuration is the unknown positive class.  Threshold
calibration uses only the known calibration split; unknown holdout labels are
used only after scoring to calculate evaluation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

from core.data import HEALTHY, discover_datasets, load_pools, make_split
from core.logger import save_plot, setup_run
from core.openset import OpenSetMethod, canonical_openset_method, create_openset_detector
from core.runner import save_json
from core.provenance import CONTENT_VERSION, capture_source, write_private_context


FORMAL_CONDITION_COUNT = 9
FORMAL_METHODS: tuple[OpenSetMethod, ...] = ("mahalanobis", "knn")


def _git_sha(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def legacy_dataset_fingerprint(data_root: Path | str) -> str:
    """Hash the formal manifest, excluding its volatile generation timestamp."""
    root = Path(data_root).expanduser().resolve()
    manifest_path = root / "formal_materialization_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable = {
            "archive_sha256": manifest.get("archive_sha256", {}),
            "formal_contract": manifest.get("formal_contract", {}),
            "files": [
                {
                    key: item.get(key)
                    for key in ("stage", "motor", "rpm", "config", "source_member", "source_sha256", "rows_after_clean", "columns")
                }
                for item in sorted(manifest.get("files", []), key=lambda row: row.get("output", ""))
            ],
        }
        payload = json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    else:
        entries = []
        for path in sorted(root.glob("Step-*/myfeature/*/*/*/*_Group_feature_data_clean.csv")):
            stat = path.stat()
            entries.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
        payload = json.dumps(entries, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_fingerprint(data_root: Path | str, *, version: str = CONTENT_VERSION) -> str:
    """明確版本化；預設驗證實際內容，legacy僅供歷史唯讀相容。"""
    if version == "legacy_fingerprint_v1":
        return legacy_dataset_fingerprint(data_root)
    if version != CONTENT_VERSION:
        raise ValueError("未知fingerprint版本")
    return capture_source(data_root)[0]["dataset_fingerprint"]


def _fpr_at_tpr95(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """Return evaluation-only FPR at 95% unknown recall.

    This threshold is *not* the detector threshold and is never used for
    predictions or calibration.  It is the standard post-hoc ROC operating
    point, calculated from held-out scores solely to report the metric.
    """
    operating_threshold = float(np.quantile(unknown_scores, 0.05))
    return float((known_scores >= operating_threshold).mean())


def _validate_datasets(datasets: list[dict], data_root: Path, require_nine: bool) -> None:
    if not datasets:
        raise FileNotFoundError(
            f"formal data not found under {data_root}; expected "
            "Step-*/myfeature/*/*/*_Group_feature_data_clean.csv. "
            "Run core.formal_data with the verified formal source first."
        )
    conditions = {(row["motor"], row["rpm"]) for row in datasets}
    if len(conditions) != len(datasets):
        raise ValueError(f"duplicate formal conditions discovered: {sorted(conditions)}")
    if require_nine and len(datasets) != FORMAL_CONDITION_COUNT:
        raise ValueError(
            f"formal OSR benchmark requires {FORMAL_CONDITION_COUNT} conditions; "
            f"discovered {len(datasets)} under {data_root}: {sorted(conditions)}"
        )


def _detector_metadata(detector, confidence: float, openset_method: str, knn_neighbors: int) -> dict:
    summaries = detector.class_summaries()
    raw_thresholds = [float(item["threshold"]) for item in summaries]
    return {
        "openset_method": openset_method,
        "canonical_method": openset_method,
        "mahalanobis_method": getattr(detector, "method", None),
        "knn_neighbors": knn_neighbors if openset_method == "knn" else None,
        "threshold": 1.0,
        "raw_thresholds": raw_thresholds,
        "threshold_strategy": f"known_calibration_quantile_{confidence:.4g}",
        "score_direction": "higher_is_unknown",
        "calibration_source": "known training/calibration only",
        "reference_bank_samples": [int(item["n_reference"]) for item in summaries]
        if openset_method == "knn"
        else None,
        "effective_neighbors": [int(item["effective_neighbors"]) for item in summaries]
        if openset_method == "knn"
        else None,
        "distance_metric": "euclidean" if openset_method == "knn" else "ledoit_wolf_mahalanobis",
        "neighbor_aggregation": "mean_distance" if openset_method == "knn" else None,
    }


def _run_benchmark(
    data_root: Path | str = "data",
    seed: int = 42,
    confidence: float = 0.95,
    openset_method: str = "mahalanobis",
    mahalanobis_method: str = "ledoit_wolf",
    knn_neighbors: int = 5,
    *,
    require_nine: bool = True,
) -> dict:
    """Run one condition×seed×method benchmark over the formal data root."""
    data_path = Path(data_root).expanduser().resolve()
    datasets = sorted(discover_datasets(data_path), key=lambda row: (row["motor"], row["rpm"]))
    _validate_datasets(datasets, data_path, require_nine)
    requested_method = str(openset_method)
    openset_method = canonical_openset_method(requested_method)
    if openset_method not in FORMAL_METHODS:
        raise ValueError(f"unsupported formal method {requested_method!r}; choose mahalanobis or knn")
    if openset_method == "mahalanobis" and mahalanobis_method == "legacy":
        raise ValueError(
            "formal exp6 rejects mahalanobis_method='legacy': its historical "
            "training-distance threshold is not the shared known-calibration policy"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    # The formal benchmark currently uses NumPy/scikit-learn only, but seed the
    # Python RNG as well so future helpers cannot silently add an uncontrolled
    # source of split or sampling randomness.
    random.seed(seed)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for dataset in datasets:
        input_audit = {}
        pools = load_pools(dataset["path"], require_complete=require_nine, audit=input_audit)
        if HEALTHY not in pools:
            raise ValueError(f"{dataset['motor']}/{dataset['rpm']} lacks {HEALTHY}")
        split = make_split(len(pools[HEALTHY]), rng)
        X_train = pools[HEALTHY][split.train]
        X_cal = pools[HEALTHY][split.cal]
        X_known = pools[HEALTHY][split.holdout]
        unknown_configs = [config for config in sorted(pools) if config != HEALTHY]
        if not unknown_configs:
            raise ValueError(f"{dataset['motor']}/{dataset['rpm']} has no unknown classes")
        X_unknown = np.vstack([pools[config] for config in unknown_configs])

        scaler = RobustScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_cal_scaled = scaler.transform(X_cal)
        detector = create_openset_detector(
            openset_method,
            confidence=confidence,
            mahalanobis_method=mahalanobis_method,
            knn_neighbors=knn_neighbors,
        ).fit(
            X_train_scaled,
            np.zeros(len(X_train), dtype=int),
            X_cal_scaled,
            np.zeros(len(X_cal), dtype=int),
        )
        started = time.perf_counter()
        known_scores = np.asarray(detector.score_samples(scaler.transform(X_known)), dtype=float)
        unknown_scores = np.asarray(detector.score_samples(scaler.transform(X_unknown)), dtype=float)
        inference_seconds = time.perf_counter() - started
        if not np.isfinite(known_scores).all() or not np.isfinite(unknown_scores).all():
            raise ValueError(f"{dataset['motor']}/{dataset['rpm']} produced non-finite scores")
        scores = np.concatenate((known_scores, unknown_scores))
        labels = np.concatenate((np.zeros(len(known_scores), dtype=int), np.ones(len(unknown_scores), dtype=int)))
        predictions = (scores > 1.0).astype(int)
        metadata = _detector_metadata(detector, confidence, openset_method, knn_neighbors)
        rows.append(
            {
                "dataset": f"{dataset['motor']}/{dataset['rpm'].replace('rpm', '')}",
                "motor": dataset["motor"],
                "rpm": dataset["rpm"],
                "seed": int(seed),
                "method": openset_method,
                "requested_method": requested_method,
                "mahalanobis_method": mahalanobis_method if openset_method == "mahalanobis" else None,
                "knn_neighbors": knn_neighbors if openset_method == "knn" else None,
                "n_train": int(len(X_train)),
                "n_calibration": int(len(X_cal)),
                "n_known_test": int(len(X_known)),
                "n_unknown_test": int(len(X_unknown)),
                "n_unknown_classes": int(len(unknown_configs)),
                "known_accuracy": round(float((predictions[: len(known_scores)] == 0).mean()), 6),
                "open_set_accuracy": round(float(accuracy_score(labels, predictions)), 6),
                "auroc": round(float(roc_auc_score(labels, scores)), 6),
                "aupr_unknown_positive": round(float(average_precision_score(labels, scores)), 6),
                "fpr_at_tpr95": round(float(_fpr_at_tpr95(known_scores, unknown_scores)), 6),
                "unknown_precision": round(float(precision_score(labels, predictions, zero_division=0)), 6),
                "unknown_recall": round(float(recall_score(labels, predictions, zero_division=0)), 6),
                "unknown_f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
                "known_score_median": round(float(np.median(known_scores)), 6),
                "unknown_score_median": round(float(np.median(unknown_scores)), 6),
                "inference_seconds": round(float(inference_seconds), 6),
                "peak_memory_mb": None,
                "status": "completed",
                "input_schema_audit": input_audit,
                **metadata,
            }
        )

    return {
        "schema_version": 3,
        "status": "completed",
        "method": openset_method,
        "requested_method": requested_method,
        "seed": int(seed),
        "confidence": float(confidence),
        "formal_condition_count": len(rows),
        "dataset_fingerprint": dataset_fingerprint(data_path),
        "fingerprint_version": CONTENT_VERSION,
        # Keep committed summaries portable and free of user-specific absolute paths.
        "dataset_root": "formal_features",
        "positive_class": "unknown",
        "score_direction": "higher_is_unknown",
        "polarmap_base_method": "mahalanobis",
        "checkpoint_identifier": None,
        "config": {
            "feature_count": 105,
            "normalization": "RobustScaler fit on known training split only",
            "split": "known health 60/20/20 train/calibration/holdout; all other configurations unknown positive",
            "threshold": 1.0,
            "threshold_strategy": f"known_calibration_quantile_{confidence:.4g}",
            "openset_method": openset_method,
            "mahalanobis_method": mahalanobis_method if openset_method == "mahalanobis" else None,
            "knn_neighbors": knn_neighbors if openset_method == "knn" else None,
            "checkpoint_identifier": None,
        },
        "commit_sha": _git_sha(Path(__file__).resolve().parents[1]),
        "python": platform.python_version(),
        "rows": rows,
    }


def run(data_root: Path | str = "data", seed: int = 42, confidence: float = 0.95,
        openset_method: str = "mahalanobis", mahalanobis_method: str = "ledoit_wolf",
        knn_neighbors: int = 5, *, require_nine: bool = True, evidence_paths=None) -> dict:
    """同一科學計算外記錄API／CLI的環境、內容來源與失敗狀態。"""
    if evidence_paths is None:
        _, evidence_paths = setup_run("exp6_formal_api", unique=True)
    try:
        recorded_method = canonical_openset_method(openset_method)
        recorded_requested_method = str(openset_method)
    except ValueError:
        recorded_method = recorded_requested_method = "INVALID"
    config = {"seed": int(seed) if isinstance(seed, (int, np.integer)) else "INVALID",
              "confidence": float(confidence) if isinstance(confidence, (int, float)) and np.isfinite(confidence) else "INVALID",
              "openset_method": recorded_method, "requested_method": recorded_requested_method,
              "mahalanobis_method": mahalanobis_method if mahalanobis_method in {"legacy", "ledoit_wolf", "oas", "mcd"} else "INVALID",
              "knn_neighbors": int(knn_neighbors) if isinstance(knn_neighbors, (int, np.integer)) else "INVALID",
              "require_nine": bool(require_nine)}
    evidence = {"schema_version": "formal_run_provenance_v1", "status": "RUNNING",
                "configuration_arguments": config, "fingerprint_version": CONTENT_VERSION}
    try:
        source, private = capture_source(data_root)
        evidence["private_context_id"] = write_private_context(private)
        evidence["source"] = source
        result = _run_benchmark(data_root, seed, confidence, openset_method, mahalanobis_method,
                                knn_neighbors, require_nine=require_nine)
        after = capture_source(data_root)[0]
        if source != after:
            raise ValueError("評估期間來源內容或manifest變動，拒絕完成")
        result["environment"] = json.loads((evidence_paths.output_dir / "environment.json").read_text(encoding="utf-8"))
        result["source"] = source
        evidence["effective_configuration"] = result["config"]
        evidence["environment"] = result["environment"]
        evidence["status"] = "COMPLETED"
    except BaseException as exc:
        evidence.update(status="FAILED", error_type=type(exc).__name__)
        save_json(evidence_paths.output_dir / "provenance.json", evidence)
        raise
    save_json(evidence_paths.output_dir / "provenance.json", evidence)
    result["provenance_record_id"] = f"{evidence_paths.program}/{evidence_paths.timestamp}"
    return result


def _figure(result: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(result["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(frame["dataset"], frame["auroc"], color="#3b82f6")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].set_title(f"{result['method']} AUROC")
    axes[0].tick_params(axis="x", rotation=55)
    axes[1].bar(frame["dataset"], frame["unknown_f1"], color="#16a34a")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_title("Unknown F1")
    axes[1].tick_params(axis="x", rotation=55)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--openset-method", default="mahalanobis")
    parser.add_argument("--method", dest="mahalanobis_method", default="ledoit_wolf")
    parser.add_argument("--knn-neighbors", type=int, default=5)
    args = parser.parse_args(argv)

    log, paths = setup_run("exp6_formal_benchmark", unique=True)
    started = datetime.now(timezone.utc).isoformat()
    log.info(
        f"正式 OSR benchmark method={args.openset_method}, seed={args.seed}, "
        f"confidence={args.confidence}"
    )
    result = run(
        args.data_root,
        args.seed,
        args.confidence,
        args.openset_method,
        args.mahalanobis_method,
        args.knn_neighbors,
        evidence_paths=paths,
    )
    result["started_at_utc"] = started
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["output_dir"] = str(Path("output") / "exp6_formal_benchmark" / paths.output_dir.name)
    pd.DataFrame(result["rows"]).to_csv(paths.output_dir / "results.csv", index=False)
    save_json(paths.output_dir / "summary.json", result)
    save_plot(_figure(result), paths, "osr_benchmark.png")
    log.info(f"completed {len(result['rows'])} conditions; output={paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
