"""實驗五：跨工況冷啟動泛化（Cross-condition Generalization）。

問題：在某一工況（馬達壽命期 × 轉速）建立的健康基準，能否直接用於其他工況？
需不需要逐工況建基準？老化（T1→T2→T3）本身造成多大的漂移？

三個子實驗：
(a) matrix    9×9 遷移矩陣：在 (motor, rpm) 建健康基準 → 對每組資料集測
              健康接受率／故障偵測率／AUROC（zero-adaptation，對角線 = 自身）
(b) strategy  部署策略比較：逐工況基準（現行）vs 全混合基準（九組健康合併）
              vs 同轉速混合基準（三個壽命期合併）——誤報/漏報的取捨
(c) drift     壽命期漂移：以 T1 各轉速的健康基準看 T1/T2/T3「健康」資料的
              分數分布位移——「老化即慢性漂移」的量化（連結漸進退化敘事）

批次執行：
    venv/bin/python -m experiments.exp5_cross_condition
    venv/bin/python -m experiments.exp5_cross_condition --part a
"""

from __future__ import annotations

import argparse

import numpy as np

from core.data import HEALTHY, load_pools, make_split
from core.logger import save_plot, setup_run
from core.monitor import OpenSetMonitor
from core.runner import save_json
from core.data import discover_datasets


def _dataset_key(ds: dict) -> str:
    return f"{ds['motor']}/{ds['rpm'].replace('rpm', '')}"


def _load_all(data_root: str) -> dict[str, dict[str, np.ndarray]]:
    return {_dataset_key(ds): load_pools(ds["path"]) for ds in discover_datasets(data_root)}


def _fit_healthy_monitor(pools, seed: int, confidence: float, method: str) -> OpenSetMonitor:
    monitor = OpenSetMonitor(pools, seed=seed, confidence=confidence, method=method)
    monitor.fit_initial()
    return monitor


def _evaluate(monitor: OpenSetMonitor, target_pools, self_target: bool) -> dict:
    """健康接受率 / 故障偵測率 / AUROC。自身資料集只用健康 holdout 避免洩漏。"""
    from sklearn.metrics import roc_auc_score

    if self_target:
        healthy_scores = monitor.score(monitor.holdout(HEALTHY))
    else:
        healthy_scores = monitor.score(target_pools[HEALTHY])
    fault_scores = np.concatenate([
        monitor.score(pool) for config, pool in target_pools.items() if config != HEALTHY
    ])
    y = np.r_[np.zeros(len(healthy_scores)), np.ones(len(fault_scores))]
    s = np.r_[healthy_scores, fault_scores]
    return {
        "healthy_accept": round(float((healthy_scores <= 1).mean()), 4),
        "fault_detect": round(float((fault_scores > 1).mean()), 4),
        "auroc": round(float(roc_auc_score(y, s)), 4),
    }


# ── (a) 9×9 遷移矩陣 ────────────────────────────────────────────────────────


def run_matrix(all_pools, seed: int = 42, confidence: float = 0.95,
               method: str = "ledoit_wolf") -> dict:
    keys = sorted(all_pools)
    cells = {}
    for src in keys:
        monitor = _fit_healthy_monitor(all_pools[src], seed, confidence, method)
        for dst in keys:
            cells[f"{src}->{dst}"] = _evaluate(monitor, all_pools[dst], self_target=(src == dst))
    diag = [cells[f"{k}->{k}"]["auroc"] for k in keys]
    off = [v["auroc"] for k, v in cells.items()
           if k.split("->")[0] != k.split("->")[1]]
    same_rpm_off = [v["auroc"] for k, v in cells.items()
                    if (lambda a, b: a != b and a.split("/")[1] == b.split("/")[1])(*k.split("->"))]
    same_motor_off = [v["auroc"] for k, v in cells.items()
                      if (lambda a, b: a != b and a.split("/")[0] == b.split("/")[0])(*k.split("->"))]
    return {
        "keys": keys,
        "cells": cells,
        "summary": {
            "diag_auroc_mean": round(float(np.mean(diag)), 4),
            "offdiag_auroc_mean": round(float(np.mean(off)), 4),
            "offdiag_auroc_min": round(float(np.min(off)), 4),
            "same_rpm_cross_motor_auroc_mean": round(float(np.mean(same_rpm_off)), 4),
            "same_motor_cross_rpm_auroc_mean": round(float(np.mean(same_motor_off)), 4),
        },
    }


# ── (b) 部署策略比較 ────────────────────────────────────────────────────────


def _pooled_monitor(healthy_arrays, seed, confidence, method) -> OpenSetMonitor:
    pooled = {HEALTHY: np.vstack(healthy_arrays)}
    return _fit_healthy_monitor(pooled, seed, confidence, method)


def run_strategy(all_pools, seed: int = 42, confidence: float = 0.95,
                 method: str = "ledoit_wolf") -> dict:
    keys = sorted(all_pools)
    rng = np.random.default_rng(seed + 7)
    # 各資料集的健康池先各自留 20% 當評估用 holdout（混合策略不得看到）
    holdout_idx = {k: make_split(len(all_pools[k][HEALTHY]), rng) for k in keys}

    def eval_with(monitor, k) -> dict:
        sp = holdout_idx[k]
        healthy_scores = monitor.score(all_pools[k][HEALTHY][sp.holdout])
        fault_scores = np.concatenate([
            monitor.score(pool) for c, pool in all_pools[k].items() if c != HEALTHY
        ])
        return {
            "healthy_accept": float((healthy_scores <= 1).mean()),
            "fault_detect": float((fault_scores > 1).mean()),
        }

    strategies: dict[str, dict] = {"per_condition": {}, "global_mixed": {}, "per_rpm_mixed": {}}

    for k in keys:  # 逐工況（現行做法）
        monitor = _fit_healthy_monitor(all_pools[k], seed, confidence, method)
        strategies["per_condition"][k] = eval_with(monitor, k)

    train_part = {k: all_pools[k][HEALTHY][np.r_[holdout_idx[k].train, holdout_idx[k].cal]]
                  for k in keys}
    global_monitor = _pooled_monitor(list(train_part.values()), seed, confidence, method)
    for k in keys:
        strategies["global_mixed"][k] = eval_with(global_monitor, k)

    rpms = sorted({k.split("/")[1] for k in keys})
    for rpm in rpms:
        group = [k for k in keys if k.endswith("/" + rpm)]
        monitor = _pooled_monitor([train_part[k] for k in group], seed, confidence, method)
        for k in group:
            strategies["per_rpm_mixed"][k] = eval_with(monitor, k)

    summary = {
        name: {
            "healthy_accept_mean": round(float(np.mean([v["healthy_accept"] for v in rows.values()])), 4),
            "fault_detect_mean": round(float(np.mean([v["fault_detect"] for v in rows.values()])), 4),
        }
        for name, rows in strategies.items()
    }
    return {"strategies": strategies, "summary": summary}


# ── (c) 壽命期漂移 ──────────────────────────────────────────────────────────


def run_drift(all_pools, seed: int = 42, confidence: float = 0.95,
              method: str = "ledoit_wolf") -> dict:
    rows = []
    rpms = sorted({k.split("/")[1] for k in all_pools})
    for rpm in rpms:
        base_key = f"T1/{rpm}"
        if base_key not in all_pools:
            continue
        monitor = _fit_healthy_monitor(all_pools[base_key], seed, confidence, method)
        for motor in ["T1", "T2", "T3"]:
            key = f"{motor}/{rpm}"
            if key not in all_pools:
                continue
            X = (monitor.holdout(HEALTHY) if motor == "T1" else all_pools[key][HEALTHY])
            scores = monitor.score(X)
            rows.append({
                "rpm": rpm, "motor": motor, "n": int(len(scores)),
                "median_score": round(float(np.median(scores)), 3),
                "p90_score": round(float(np.percentile(scores, 90)), 3),
                "flagged_unknown": round(float((scores > 1).mean()), 4),
            })
    return {"rows": rows}


# ── 圖表與 CLI ──────────────────────────────────────────────────────────────


def _figure(matrix, drift):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

    keys = matrix["keys"]
    M = np.array([[matrix["cells"][f"{s}->{d}"]["auroc"] for d in keys] for s in keys])
    im = axes[0].imshow(M, vmin=0.5, vmax=1.0, cmap="viridis")
    axes[0].set_xticks(range(len(keys)), keys, rotation=45, ha="right")
    axes[0].set_yticks(range(len(keys)), keys)
    axes[0].set_xlabel("evaluated on")
    axes[0].set_ylabel("baseline trained on")
    for i in range(len(keys)):
        for j in range(len(keys)):
            axes[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                         color="w" if M[i, j] < 0.85 else "k", fontsize=7)
    axes[0].set_title("(a) Cross-condition AUROC (healthy baseline transfer)")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    for rpm in sorted({r["rpm"] for r in drift["rows"]}):
        rows = [r for r in drift["rows"] if r["rpm"] == rpm]
        axes[1].plot([r["motor"] for r in rows], [r["median_score"] for r in rows],
                     marker="o", label=f"{rpm} rpm")
    axes[1].axhline(1, ls="--", c="#94a3b8", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_title("(c) Aging drift: T1 healthy baseline vs T1/T2/T3 'healthy' data")
    axes[1].set_ylabel("median open-set score (log)")
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def run(data_root: str = "data", seed: int = 42, parts: str = "abc",
        confidence: float = 0.95, method: str = "ledoit_wolf") -> dict:
    all_pools = _load_all(data_root)
    result = {"seed": seed, "datasets": sorted(all_pools)}
    if "a" in parts:
        result["matrix"] = run_matrix(all_pools, seed, confidence, method)
    if "b" in parts:
        result["strategy"] = run_strategy(all_pools, seed, confidence, method)
    if "c" in parts:
        result["drift"] = run_drift(all_pools, seed, confidence, method)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--part", default="abc")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--method", default="ledoit_wolf")
    args = parser.parse_args()

    log, paths = setup_run("exp5_cross_condition")
    log.info(f"實驗五：跨工況冷啟動泛化（part={args.part}, seed={args.seed}）")
    result = run(args.data_root, args.seed, args.part, args.confidence, args.method)
    save_json(paths.output_dir / "summary.json", result)

    if "matrix" in result:
        s = result["matrix"]["summary"]
        log.info(f"(a) AUROC：自身 {s['diag_auroc_mean']}｜跨工況平均 {s['offdiag_auroc_mean']}"
                 f"（最低 {s['offdiag_auroc_min']}）｜同轉速跨壽命 {s['same_rpm_cross_motor_auroc_mean']}"
                 f"｜同壽命跨轉速 {s['same_motor_cross_rpm_auroc_mean']}")
    if "strategy" in result:
        for name, s in result["strategy"]["summary"].items():
            log.info(f"(b) {name:<16} 健康接受 {s['healthy_accept_mean']*100:.1f}%"
                     f"｜故障偵測 {s['fault_detect_mean']*100:.1f}%")
    if "drift" in result:
        for r in result["drift"]["rows"]:
            log.info(f"(c) {r['rpm']}rpm 基準=T1 → {r['motor']} 健康分數中位 {r['median_score']}"
                     f"（判未知 {r['flagged_unknown']*100:.1f}%）")
    if "matrix" in result and "drift" in result:
        save_plot(_figure(result["matrix"], result["drift"]), paths, "cross_condition.png")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
