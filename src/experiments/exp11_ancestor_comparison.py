"""實驗十一：與 Ancestor 的 Accuracy / F1 對照（Protocol A）＋冷啟動對照（Protocol B）。

問題：專題指標要求相對 Ancestor 提升 Accuracy/F1 5%~10%。主指標先寫定為
Balanced Accuracy 與 macro-F1（Accuracy 照列不作達標依據），達標門檻是
兩者相對 legacy 絕對提升 ≥ 5 個百分點（pp）。

Protocol A（Ancestor 協定）：已知 = 8/1/2/3/4screws，未知 = 5/6/7/3_14/4_146screws。
Protocol B（冷啟動協定）：已知只有 8screws，其餘全部未知，呼應 exp1。

批次執行：
    venv/bin/python -m experiments.exp11_ancestor_comparison
"""

from __future__ import annotations

import argparse

import numpy as np

from core.data import HEALTHY, discover_datasets, load_pools
from core.logger import save_plot, setup_run
from core.monitor import OpenSetMonitor
from core.runner import save_json

_ANCESTOR_KNOWN_CANDIDATES = ["8screws", "1screw", "1screws", "2screws", "3screws", "4screws"]
_ANCESTOR_UNKNOWN_CANDIDATES = ["5screws", "6screws", "7screws", "3_14screws", "4_146screws"]


def _ancestor_known_configs(pools: dict[str, np.ndarray]) -> list[str]:
    """Ancestor 協定的已知類別：8/1/2/3/4screws，交集判斷 1screw/1screws 目錄名陷阱。"""
    seen_one = False
    result = []
    for name in _ANCESTOR_KNOWN_CANDIDATES:
        if name not in pools:
            continue
        if name.startswith("1screw"):
            if seen_one:
                continue
            seen_one = True
        result.append(name)
    return result


def build_ancestor_monitor(pools: dict[str, np.ndarray], seed: int = 42,
                            confidence: float = 0.95, method: str = "ledoit_wolf") -> OpenSetMonitor:
    """建立 Ancestor 協定（8/1/2/3/4screws 已知）的 OpenSetMonitor，供本實驗與 exp12 共用。"""
    known = _ancestor_known_configs(pools)
    monitor = OpenSetMonitor(pools, seed=seed, confidence=confidence, method=method)
    monitor.fit_initial(healthy=HEALTHY)
    for config in known:
        if config == HEALTHY:
            continue
        monitor.add_class(config)
    return monitor


def _evaluate_protocol(monitor: OpenSetMonitor, unknown_configs: list[str],
                        pools: dict[str, np.ndarray]) -> dict:
    """已知類別各自 holdout + 全部未知池丟 classify()，算多類 Accuracy/Balanced Accuracy/macro-F1/unknown F1。"""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y_true: list[str] = []
    y_pred: list[str] = []
    for config in monitor.known:
        X = monitor.holdout(config)
        preds = monitor.classify(X)
        y_true += [config] * len(X)
        y_pred += [p if p is not None else "unknown" for p in preds]
    for config in unknown_configs:
        X = pools[config]
        preds = monitor.classify(X)
        y_true += ["unknown"] * len(X)
        y_pred += [p if p is not None else "unknown" for p in preds]

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return {
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true_arr, y_pred_arr)), 4),
        "macro_f1": round(float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)), 4),
        "unknown_f1": round(float(f1_score(y_true_arr == "unknown", y_pred_arr == "unknown",
                                            zero_division=0)), 4),
        "n": int(len(y_true_arr)),
    }


def _summarize(rows: list[dict], methods: tuple[str, ...]) -> dict:
    summary: dict[str, dict] = {}
    for protocol in ["A", "B"]:
        summary[protocol] = {}
        for method in methods:
            sub = [r for r in rows if r["protocol"] == protocol and r["method"] == method]
            summary[protocol][method] = {
                f"{m}_mean": round(float(np.mean([r[m] for r in sub])), 4)
                for m in ["accuracy", "balanced_accuracy", "macro_f1", "unknown_f1"]
            }
    return summary


def _paired_wilcoxon(rows: list[dict], methods: tuple[str, ...]) -> dict | None:
    if "legacy" not in methods or "ledoit_wolf" not in methods:
        return None
    from scipy.stats import wilcoxon

    result = {}
    for protocol in ["A", "B"]:
        by_key: dict[tuple, dict[str, float]] = {}
        for r in rows:
            if r["protocol"] != protocol:
                continue
            by_key.setdefault((r["dataset"], r["seed"]), {})[r["method"]] = r["macro_f1"]
        pairs = [(v["legacy"], v["ledoit_wolf"]) for v in by_key.values()
                 if "legacy" in v and "ledoit_wolf" in v]
        legacy_vals = np.array([p[0] for p in pairs])
        lw_vals = np.array([p[1] for p in pairs])
        diffs = lw_vals - legacy_vals
        if np.allclose(diffs, 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(lw_vals, legacy_vals)
        result[f"protocol_{protocol}_macro_f1"] = {
            "n_pairs": len(pairs),
            "mean_diff_pp": round(float(np.mean(diffs) * 100), 2),
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
        }
    return result


def run(data_root: str = "data", seeds: tuple[int, ...] = (42, 123, 2026),
        confidence: float = 0.95, methods: tuple[str, ...] = ("legacy", "ledoit_wolf")) -> dict:
    datasets = discover_datasets(data_root)
    rows: list[dict] = []
    for ds in datasets:
        key = f"{ds['motor']}/{ds['rpm'].replace('rpm', '')}"
        pools = load_pools(ds["path"])
        unknown_a = [c for c in _ANCESTOR_UNKNOWN_CANDIDATES if c in pools]

        for seed in seeds:
            for method in methods:
                monitor_a = build_ancestor_monitor(pools, seed, confidence, method)
                rows.append({"protocol": "A", "dataset": key, "seed": seed, "method": method,
                             **_evaluate_protocol(monitor_a, unknown_a, pools)})

                monitor_b = OpenSetMonitor(pools, seed=seed, confidence=confidence, method=method)
                monitor_b.fit_initial()
                unknown_b = [c for c in pools if c != HEALTHY]
                rows.append({"protocol": "B", "dataset": key, "seed": seed, "method": method,
                             **_evaluate_protocol(monitor_b, unknown_b, pools)})

    summary = _summarize(rows, methods)
    wilcoxon_result = _paired_wilcoxon(rows, methods)
    return {"rows": rows, "summary": summary, "wilcoxon": wilcoxon_result,
            "seeds": list(seeds), "confidence": confidence, "methods": list(methods),
            "n_datasets": len(datasets),
            "primary_metrics": ["balanced_accuracy", "macro_f1"],
            "threshold_pp": 5.0}


def _figure(summary: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocols = list(summary)
    methods = list(summary[protocols[0]])
    width = 0.35
    x = np.arange(len(protocols))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, method in enumerate(methods):
        axes[0].bar(x + i * width, [summary[p][method]["balanced_accuracy_mean"] for p in protocols],
                    width, label=method)
        axes[1].bar(x + i * width, [summary[p][method]["macro_f1_mean"] for p in protocols],
                    width, label=method)
    for ax, title in zip(axes, ["Balanced Accuracy", "macro-F1"]):
        ax.set_xticks(x + width / 2, [f"Protocol {p}" for p in protocols])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
    fig.suptitle("實驗十一：legacy vs ledoit_wolf")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--methods", default="legacy,ledoit_wolf")
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    methods = tuple(args.methods.split(","))

    log, paths = setup_run("exp11_ancestor_comparison")
    log.info(f"實驗十一：Ancestor 對照（seeds={seeds}, methods={methods}）")
    result = run(args.data_root, seeds, args.confidence, methods)
    save_json(paths.output_dir / "summary.json", result)

    for protocol, per_method in result["summary"].items():
        for method, m in per_method.items():
            log.info(f"Protocol {protocol} {method:<12} Accuracy={m['accuracy_mean']:.4f}  "
                      f"BalancedAcc={m['balanced_accuracy_mean']:.4f}  macroF1={m['macro_f1_mean']:.4f}  "
                      f"unknownF1={m['unknown_f1_mean']:.4f}")
    if result["wilcoxon"]:
        for k, v in result["wilcoxon"].items():
            log.info(f"{k}: Δ={v['mean_diff_pp']}pp  W={v['statistic']}  p={v['p_value']}  n={v['n_pairs']}")

    save_plot(_figure(result["summary"]), paths, "ancestor_comparison.png")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
