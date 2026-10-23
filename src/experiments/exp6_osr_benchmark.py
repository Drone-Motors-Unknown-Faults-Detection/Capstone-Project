"""實驗六：開集偵測方法基準比較（OSR Benchmark）——嚴謹性章節。

問題：主線採用的「逐類 Ledoit–Wolf 馬氏距離」在健康-only 冷啟動設定下，
相對其他常見單類/異常偵測方法是不是合理選擇？

方法（core/detectors.py，全部 sklearn；除 maha_legacy 外皆在校準集定閾值）：
    maha_ledoit_wolf（主線）、maha_legacy（論文版對照）、ocsvm、
    iforest、lof、knn_dist、pca_recon、mlp_autoencoder

流程：對 discover_datasets 找到的所有工況各自——健康池 60/20/20 切訓練/校準/保留 →
各方法在訓練集擬合、校準集定閾值（--confidence 分位，預設 0.95 = 正規化 1.0；
maha_legacy 依論文版改在訓練集自身定閾值）→
以健康保留集（誤報）與全部故障池（偵測）評估 → 跨資料集平均 ± 標準差。

指標：AUROC、FPR@TPR95、健康誤報率、故障偵測率。

批次執行：
    venv/bin/python -m experiments.exp6_osr_benchmark
"""

from __future__ import annotations

import argparse

import numpy as np

from core.data import HEALTHY, load_pools, make_split, discover_datasets
from core.detectors import ALL_DETECTORS
from core.logger import save_plot, setup_run
from core.openset import openset_metrics
from core.runner import save_json


def run(data_root: str = "data", seed: int = 42, confidence: float = 0.95) -> dict:
    datasets = discover_datasets(data_root)
    rng = np.random.default_rng(seed)
    rows = []
    for ds in datasets:
        key = f"{ds['motor']}/{ds['rpm'].replace('rpm','')}"
        pools = load_pools(ds["path"])
        sp = make_split(len(pools[HEALTHY]), rng)
        X_tr = pools[HEALTHY][sp.train]
        X_ca = pools[HEALTHY][sp.cal]
        X_ho = pools[HEALTHY][sp.holdout]
        X_faults = np.vstack([p for c, p in pools.items() if c != HEALTHY])

        for det_cls in ALL_DETECTORS:
            det = det_cls(confidence=confidence, seed=seed).fit(X_tr, X_ca)
            h = det.score(X_ho)
            f = det.score(X_faults)
            metrics = openset_metrics(h, f)
            rows.append({
                "dataset": key,
                "method": det.name,
                "auroc": round(metrics["auroc"], 4),
                "fpr_at_tpr95": round(metrics["fpr_at_tpr95"], 4),
                "healthy_fp": round(float((h > 1).mean()), 4),
                "fault_detect": round(float((f > 1).mean()), 4),
            })

    summary = []
    for det_cls in ALL_DETECTORS:
        sub = [r for r in rows if r["method"] == det_cls.name]
        summary.append({
            "method": det_cls.name,
            **{
                f"{m}_mean": round(float(np.mean([r[m] for r in sub])), 4)
                for m in ["auroc", "fpr_at_tpr95", "healthy_fp", "fault_detect"]
            },
            **{
                f"{m}_std": round(float(np.std([r[m] for r in sub])), 4)
                for m in ["auroc", "healthy_fp"]
            },
        })
    summary.sort(key=lambda r: (-r["auroc_mean"], r["fpr_at_tpr95_mean"]))
    return {"rows": rows, "summary": summary, "seed": seed, "confidence": confidence,
            "n_datasets": len(datasets)}


def _figure(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s["method"] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(names, [s["auroc_mean"] for s in summary],
                yerr=[s["auroc_std"] for s in summary], color="#3b82f6", capsize=3)
    axes[0].set_ylim(0.9, 1.001)
    axes[0].set_title("AUROC (mean ± std over 9 datasets)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(names, [s["healthy_fp_mean"] * 100 for s in summary],
                yerr=[s["healthy_fp_std"] * 100 for s in summary], color="#ef4444", capsize=3)
    axes[1].axhline(5, ls="--", c="#94a3b8", lw=1)
    axes[1].set_title("Healthy false-positive rate % (nominal 5%)")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    log, paths = setup_run("exp6_osr_benchmark", data_root=args.data_root)
    log.info(f"實驗六：OSR 方法基準比較（seed={args.seed}）")
    result = run(args.data_root, args.seed, args.confidence)

    import pandas as pd

    pd.DataFrame(result["rows"]).to_csv(paths.output_dir / "results.csv", index=False)
    save_json(paths.output_dir / "summary.json", result)
    save_plot(_figure(result["summary"]), paths, "osr_benchmark.png")

    log.info(f"{'方法':<18}{'AUROC':>14}{'FPR@95':>9}{'健康誤報':>9}{'故障偵測':>9}")
    for s in result["summary"]:
        log.info(f"{s['method']:<18}{s['auroc_mean']:.4f}±{s['auroc_std']:.4f}"
                 f"{s['fpr_at_tpr95_mean']:>9.4f}{s['healthy_fp_mean']*100:>8.1f}%"
                 f"{s['fault_detect_mean']*100:>8.1f}%")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
