"""實驗一：冷啟動偵測能力（Cold-start Unknown Detection）。

只用 8screws（Healthy）建立健康基準——Mahalanobis–Taguchi 式的單類監測
（逐類 Ledoit–Wolf 收縮共變異數，校準集第 95 百分位當閾值）——
然後量測：每種故障配置被偵測為未知的比例、健康保留集的誤報率、AUROC。

批次執行（產出論文數據）：
    venv/bin/python -m experiments.exp1_cold_start --motor T1 --rpm 8000rpm

Web 介面經由同一個 OpenSetMonitor（階段 0）呼叫相同邏輯。
"""

from __future__ import annotations

import argparse

import numpy as np

from core.data import HEALTHY, display_name
from core.logger import save_plot, setup_run
from core.monitor import OpenSetMonitor
from core.runner import add_dataset_args, add_openset_args, resolve_dataset, save_json


def run(
    pools: dict[str, np.ndarray],
    seed: int = 42,
    confidence: float = 0.95,
    method: str = "ledoit_wolf",
    openset_method: str = "mahalanobis",
    knn_neighbors: int = 5,
) -> dict:
    """批次介面：把 iter_run 跑完，回傳最後的結果。"""
    for event in iter_run(pools, seed, confidence, method, openset_method, knn_neighbors):
        pass
    return event["result"]


def iter_run(
    pools: dict[str, np.ndarray],
    seed: int = 42,
    confidence: float = 0.95,
    method: str = "ledoit_wolf",
    openset_method: str = "mahalanobis",
    knn_neighbors: int = 5,
):
    """逐步產生事件，供 Web 實驗頁邊跑邊畫；計算與 run() 完全相同。

    事件依序為 fitted（擬合完成）、每個配置一組 scores（逐筆開集分數）與 row
    （該配置的統計列），最後一個是 result（與 run() 回傳值相同）。
    """
    monitor = OpenSetMonitor(
        pools,
        seed=seed,
        confidence=confidence,
        method=method,
        openset_method=openset_method,
        knn_neighbors=knn_neighbors,
    )
    monitor.fit_initial()
    yield {"event": "fitted", "model": monitor.summary()}

    healthy_scores = monitor.score(monitor.holdout(HEALTHY))
    yield {"event": "scores", "config": HEALTHY, "display": display_name(HEALTHY),
           "kind": "healthy-holdout", "scores": healthy_scores.tolist()}
    rows = [
        {
            "config": HEALTHY,
            "display": display_name(HEALTHY),
            "kind": "healthy-holdout",
            "n": int(len(healthy_scores)),
            "flagged": int((healthy_scores > 1).sum()),
            "detect_rate": float((healthy_scores > 1).mean()),
            "auroc": None,
            "median_score": float(np.median(healthy_scores)),
        }
    ]
    yield {"event": "row", "row": rows[-1]}
    for config, pool in pools.items():
        if config in monitor.known:
            continue
        scores = monitor.score(pool)
        yield {"event": "scores", "config": config, "display": display_name(config),
               "kind": "unknown-fault", "scores": scores.tolist()}
        rows.append(
            {
                "config": config,
                "display": display_name(config),
                "kind": "unknown-fault",
                "n": int(len(scores)),
                "flagged": int((scores > 1).sum()),
                "detect_rate": float((scores > 1).mean()),
                "auroc": _auroc(healthy_scores, scores),
                "median_score": float(np.median(scores)),
            }
        )
        yield {"event": "row", "row": rows[-1]}

    fault_rows = rows[1:]
    yield {"event": "result", "result": {
        "rows": rows,
        "healthy_fp_rate": rows[0]["detect_rate"],
        "macro_detect_rate": float(np.mean([r["detect_rate"] for r in fault_rows])),
        "macro_auroc": float(np.mean([r["auroc"] for r in fault_rows])),
        "model": monitor.summary(),
        "confidence": confidence,
        "openset_method": openset_method,
        "method": method,
        "score_type": monitor.summary()["score_type"],
        "threshold": 1.0,
        "threshold_strategy": monitor.summary()["threshold_strategy"],
        "calibration_source": "known-only 20% calibration split",
        "split": {"train": 0.6, "calibration": 0.2, "holdout": 0.2},
        "knn_neighbors": knn_neighbors if openset_method == "knn" else None,
        "seed": seed,
    }}


def _auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.zeros(len(neg_scores)), np.ones(len(pos_scores))]
    return float(roc_auc_score(y, np.r_[neg_scores, pos_scores]))


def _make_figure(result: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    names = [r["config"] for r in rows]
    rates = [r["detect_rate"] * 100 for r in rows]
    colors = ["#22c55e" if r["kind"] == "healthy-holdout" else "#ef4444" for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, rates, color=colors)
    ax.set_ylabel("Flagged as unknown (%)")
    ax.set_title("Exp1 Cold-start detection (healthy-only training) — green: healthy FP, red: fault detection")
    ax.axhline(95, ls="--", lw=1, color="#94a3b8")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    add_openset_args(parser)
    args = parser.parse_args()

    ds, pools = resolve_dataset(args)
    log, paths = setup_run("exp1_cold_start")
    log.info(f"實驗一：冷啟動偵測 — {ds['motor']}/{ds['rpm']}"
             f"（openset={args.openset_method}, method={args.method}, "
             f"confidence={args.confidence}, seed={args.seed}）")

    result = run(
        pools,
        seed=args.seed,
        confidence=args.confidence,
        method=args.method,
        openset_method=args.openset_method,
        knn_neighbors=args.knn_neighbors,
    )
    result["dataset"] = {"motor": ds["motor"], "rpm": ds["rpm"]}

    import pandas as pd

    pd.DataFrame(result["rows"]).to_csv(paths.output_dir / "results.csv", index=False)
    save_json(paths.output_dir / "summary.json", {k: v for k, v in result.items() if k != "rows"})
    save_plot(_make_figure(result), paths, "detect_rates.png")

    log.info(f"{'配置':<14}{'樣本':>6}{'偵測率':>10}{'AUROC':>10}{'分數中位':>10}")
    for r in result["rows"]:
        auroc = f"{r['auroc']:.4f}" if r["auroc"] is not None else "—"
        log.info(f"{r['config']:<14}{r['n']:>6}{r['detect_rate']*100:>9.1f}%{auroc:>10}{r['median_score']:>10.2f}")
    log.info(f"健康誤報率 {result['healthy_fp_rate']*100:.1f}%｜"
             f"平均偵測率 {result['macro_detect_rate']*100:.1f}%｜平均 AUROC {result['macro_auroc']:.4f}")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
