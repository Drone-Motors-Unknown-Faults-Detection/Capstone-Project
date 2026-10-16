"""實驗七：在相同資料切分與指標下比較 Mahalanobis 與 k-NN Open Set detector。

unknown 是正類。所有 threshold 都只由 known calibration split 決定；holdout
健康資料與未知故障資料僅用於最終評估，不參與擬合或調參。
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from core.data import HEALTHY
from core.logger import save_plot, setup_run
from core.monitor import OpenSetMonitor
from core.openset import SUPPORTED_OPENSET_METHODS
from core.runner import add_dataset_args, resolve_dataset, save_json


def _fpr_at_tpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    eligible = fpr[tpr >= 0.95]
    return float(np.min(eligible)) if len(eligible) else 1.0


def evaluate_method(
    pools: dict[str, np.ndarray],
    openset_method: str,
    *,
    seed: int = 42,
    confidence: float = 0.95,
    mahalanobis_method: str = "ledoit_wolf",
    knn_neighbors: int = 5,
) -> tuple[dict, list[dict]]:
    monitor = OpenSetMonitor(
        pools,
        seed=seed,
        confidence=confidence,
        method=mahalanobis_method,
        openset_method=openset_method,
        knn_neighbors=knn_neighbors,
    )
    monitor.fit_initial()

    known_X = monitor.holdout(HEALTHY)
    known_scores = monitor.score(known_X)
    known_predictions = monitor.classify(known_X)
    unknown_scores_parts = []
    details = []
    for config, pool in pools.items():
        if config == HEALTHY:
            continue
        scores = monitor.score(pool)
        predictions = monitor.classify(pool)
        unknown_scores_parts.append(scores)
        details.append(
            {
                "openset_method": openset_method,
                "config": config,
                "kind": "unknown-fault",
                "n": int(len(scores)),
                "unknown_recall": float(np.mean([p is None for p in predictions])),
                "median_score": float(np.median(scores)),
            }
        )
    unknown_scores = np.concatenate(unknown_scores_parts)
    all_scores = np.r_[known_scores, unknown_scores]
    y_true = np.r_[np.zeros(len(known_scores), dtype=int), np.ones(len(unknown_scores), dtype=int)]
    y_pred = (all_scores > 1.0).astype(int)
    model = monitor.summary()

    details.insert(
        0,
        {
            "openset_method": openset_method,
            "config": HEALTHY,
            "kind": "known-holdout",
            "n": int(len(known_scores)),
            "unknown_recall": float(np.mean(known_scores > 1.0)),
            "median_score": float(np.median(known_scores)),
        },
    )
    result = {
        "openset_method": openset_method,
        "mahalanobis_method": mahalanobis_method if openset_method == "mahalanobis" else None,
        "knn_neighbors": knn_neighbors if openset_method == "knn" else None,
        "score_type": model["score_type"],
        "score_direction": "larger_is_more_unknown",
        "threshold": 1.0,
        "threshold_strategy": model["threshold_strategy"],
        "calibration_source": model["calibration_source"],
        "random_seed": seed,
        "split": model["split"],
        "n_known_test": int(len(known_scores)),
        "n_unknown_test": int(len(unknown_scores)),
        "auroc": float(roc_auc_score(y_true, all_scores)),
        "aupr_unknown_positive": float(average_precision_score(y_true, all_scores)),
        "fpr_at_95_tpr": _fpr_at_tpr95(y_true, all_scores),
        "open_set_accuracy": float(accuracy_score(y_true, y_pred)),
        "known_class_accuracy": float(
            np.mean([prediction == HEALTHY for prediction in known_predictions])
        ),
        "unknown_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "unknown_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_unknown": float(f1_score(y_true, y_pred, zero_division=0)),
        "class_thresholds": [c["threshold"] for c in model["classes"]],
    }
    return result, details


def run(
    pools: dict[str, np.ndarray],
    methods: tuple[str, ...] = SUPPORTED_OPENSET_METHODS,
    *,
    seed: int = 42,
    confidence: float = 0.95,
    mahalanobis_method: str = "ledoit_wolf",
    knn_neighbors: int = 5,
) -> dict:
    results, details = [], []
    for openset_method in methods:
        result, method_details = evaluate_method(
            pools,
            openset_method,
            seed=seed,
            confidence=confidence,
            mahalanobis_method=mahalanobis_method,
            knn_neighbors=knn_neighbors,
        )
        results.append(result)
        details.extend(method_details)
    return {
        "results": results,
        "details": details,
        "fairness": {
            "same_random_seed": seed,
            "same_split_ratios": {"train": 0.6, "calibration": 0.2, "holdout": 0.2},
            "same_features": "RobustScaler-transformed 105-dimensional clean features",
            "same_threshold_data": "known-only calibration split",
            "unknown_positive": True,
        },
    }


def _make_figure(results: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["auroc", "aupr_unknown_positive", "open_set_accuracy", "unknown_recall"]
    names = [row["openset_method"] for row in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, row in enumerate(results):
        offset = (index - (len(results) - 1) / 2) * width
        ax.bar(x + offset, [row[m] for m in metrics], width, label=names[index])
    ax.set_xticks(x, ["AUROC", "AUPR\n(unknown+)", "Open Set\naccuracy", "Unknown\nrecall"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Open Set detector comparison — same split and calibration data")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument(
        "--openset-methods",
        nargs="+",
        choices=SUPPORTED_OPENSET_METHODS,
        default=list(SUPPORTED_OPENSET_METHODS),
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--method",
        choices=("legacy", "ledoit_wolf", "oas", "mcd"),
        default="ledoit_wolf",
        help="Mahalanobis covariance estimator（向後相容名稱）",
    )
    parser.add_argument("--knn-neighbors", type=int, default=5)
    args = parser.parse_args()

    ds, pools = resolve_dataset(args)
    log, paths = setup_run("openset_comparison")
    result = run(
        pools,
        tuple(args.openset_methods),
        seed=args.seed,
        confidence=args.confidence,
        mahalanobis_method=args.method,
        knn_neighbors=args.knn_neighbors,
    )
    result["dataset"] = {"motor": ds["motor"], "rpm": ds["rpm"]}

    import pandas as pd

    pd.DataFrame(result["results"]).to_csv(paths.output_dir / "summary.csv", index=False)
    pd.DataFrame(result["details"]).to_csv(paths.output_dir / "details.csv", index=False)
    save_json(paths.output_dir / "summary.json", result)
    save_plot(_make_figure(result["results"]), paths, "method_comparison.png")

    for row in result["results"]:
        log.info(
            f"{row['openset_method']}: AUROC={row['auroc']:.4f}, "
            f"AUPR={row['aupr_unknown_positive']:.4f}, "
            f"FPR@95TPR={row['fpr_at_95_tpr']:.4f}, "
            f"OpenSetAcc={row['open_set_accuracy']:.4f}, F1={row['f1_unknown']:.4f}"
        )
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
