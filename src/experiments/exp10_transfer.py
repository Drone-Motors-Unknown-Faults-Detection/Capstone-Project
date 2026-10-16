"""實驗十：遷移學習——目標工況少量健康資料的跨工況基準遷移。

問題：exp5 的 9×9 矩陣顯示直接搬運健康基準到別的工況，AUROC 最差只有 0.10。
若目標工況能拿到少量（10/25/50 筆）健康資料，中心平移適應來源基準，或乾脆
只用這些筆數從零建基準，哪一種比較划算？

三種策略（每組 (src, dst) 配對、每個 N）：
    direct    零適應，N=0，直接用 src 基準評 dst 全部健康＋故障池
    adapted   中心平移：shift = median(目標支援集) - median(src 訓練集)
    scratch   目標端只用 N 筆健康資料從零冷啟動

批次執行：
    venv/bin/python -m experiments.exp10_transfer
"""

from __future__ import annotations

import argparse

import numpy as np

from core.data import HEALTHY, discover_datasets, load_pools
from core.logger import save_plot, setup_run
from core.monitor import OpenSetMonitor
from core.runner import save_json


def _dataset_key(ds: dict) -> str:
    return f"{ds['motor']}/{ds['rpm'].replace('rpm', '')}"


def _load_all(data_root: str) -> dict[str, dict[str, np.ndarray]]:
    return {_dataset_key(ds): load_pools(ds["path"]) for ds in discover_datasets(data_root)}


def _draw_support(healthy_pool: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """從健康池抽 n 筆當目標端支援集，回傳 (support, eval)；兩者索引不重疊。"""
    order = rng.permutation(len(healthy_pool))
    support_idx, eval_idx = order[:n], order[n:]
    return healthy_pool[support_idx], healthy_pool[eval_idx]


def _metrics(healthy_scores: np.ndarray, fault_scores: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.zeros(len(healthy_scores)), np.ones(len(fault_scores))]
    s = np.r_[healthy_scores, fault_scores]
    return {
        "healthy_accept": round(float((healthy_scores <= 1).mean()), 4),
        "fault_detect": round(float((fault_scores > 1).mean()), 4),
        "auroc": round(float(roc_auc_score(y, s)), 4),
    }


def run(data_root: str = "data", seed: int = 42, confidence: float = 0.95,
        method: str = "ledoit_wolf", n_values: tuple[int, ...] = (10, 25, 50)) -> dict:
    all_pools = _load_all(data_root)
    keys = sorted(all_pools)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for src in keys:
        monitor = OpenSetMonitor({HEALTHY: all_pools[src][HEALTHY]}, seed=seed,
                                  confidence=confidence, method=method)
        monitor.fit_initial()
        src_train = monitor.pools[HEALTHY][monitor.splits[HEALTHY].train]

        for dst in keys:
            if dst == src:
                continue
            dst_healthy = all_pools[dst][HEALTHY]
            dst_faults = np.vstack([p for c, p in all_pools[dst].items() if c != HEALTHY])

            # direct：N=0，零適應，用 dst 的全部健康池
            direct = _metrics(monitor.score(dst_healthy), monitor.score(dst_faults))
            rows.append({"src": src, "dst": dst, "n": 0, "strategy": "direct", **direct})

            for n in n_values:
                support, eval_pool = _draw_support(dst_healthy, n, rng)

                shift = np.median(support, axis=0) - np.median(src_train, axis=0)
                adapted = _metrics(
                    monitor.score(eval_pool - shift),
                    monitor.score(dst_faults - shift),
                )
                rows.append({"src": src, "dst": dst, "n": n, "strategy": "adapted", **adapted})

                scratch_monitor = OpenSetMonitor({HEALTHY: support}, seed=seed,
                                                  confidence=confidence, method=method)
                scratch_monitor.fit_initial()
                scratch = _metrics(scratch_monitor.score(eval_pool), scratch_monitor.score(dst_faults))
                rows.append({"src": src, "dst": dst, "n": n, "strategy": "scratch", **scratch})

    summary: dict[str, list[dict]] = {"direct": [], "adapted": [], "scratch": []}
    for strategy in summary:
        n_levels = [0] if strategy == "direct" else list(n_values)
        for n in n_levels:
            sub = [r for r in rows if r["strategy"] == strategy and r["n"] == n]
            summary[strategy].append({
                "n": n,
                **{f"{m}_mean": round(float(np.mean([r[m] for r in sub])), 4)
                   for m in ["auroc", "healthy_accept", "fault_detect"]},
            })

    return {"rows": rows, "summary": summary, "seed": seed, "confidence": confidence,
            "method": method, "n_values": list(n_values), "n_datasets": len(keys)}


def _figure(summary: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    direct_auroc = summary["direct"][0]["auroc_mean"]
    ax.axhline(direct_auroc, ls="--", c="#94a3b8", lw=1, label="direct (N=0)")
    for strategy, color in [("adapted", "#3b82f6"), ("scratch", "#ef4444")]:
        ns = [r["n"] for r in summary[strategy]]
        aurocs = [r["auroc_mean"] for r in summary[strategy]]
        ax.plot(ns, aurocs, marker="o", color=color, label=strategy)
    ax.set_xlabel("目標端健康樣本數 N")
    ax.set_ylabel("AUROC（跨 72 組 src→dst 配對平均）")
    ax.set_title("實驗十：遷移 vs 從零冷啟動")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--method", default="ledoit_wolf")
    parser.add_argument("--n-values", default="10,25,50")
    args = parser.parse_args()

    n_values = tuple(int(v) for v in args.n_values.split(","))

    log, paths = setup_run("exp10_transfer")
    log.info(f"實驗十：遷移學習（N={n_values}, method={args.method}, seed={args.seed}）")
    result = run(args.data_root, args.seed, args.confidence, args.method, n_values)
    save_json(paths.output_dir / "summary.json", result)

    for strategy, rows in result["summary"].items():
        for r in rows:
            log.info(f"{strategy:<8} N={r['n']:<3} AUROC={r['auroc_mean']:.4f}  "
                      f"健康接受={r['healthy_accept_mean']*100:.1f}%  "
                      f"故障偵測={r['fault_detect_mean']*100:.1f}%")

    save_plot(_figure(result["summary"]), paths, "transfer_curve.png")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
