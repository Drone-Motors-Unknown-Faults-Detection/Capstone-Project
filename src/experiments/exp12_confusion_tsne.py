"""實驗十二：視覺化——開集混淆矩陣與 t-SNE 特徵分布圖。

問題：專題要求混淆矩陣與 t-SNE，展示模型區分正常與故障訊號的效果。
repo 既有的圖只有分數折線與 PCA 2D 投影，沒有混淆矩陣，也沒有 t-SNE。

三張圖：
    (a) 二元混淆矩陣     健康 vs 故障，呼應 exp1
    (b) 開集多類混淆矩陣  Ancestor 協定（5 已知 + unknown），legacy vs ledoit_wolf 並排
    (c) t-SNE 9 工況網格  顏色=螺絲配置，標記=系統判定

批次執行：
    venv/bin/python -m experiments.exp12_confusion_tsne
"""

from __future__ import annotations

import argparse

import numpy as np

from core.data import HEALTHY, config_sort_key, discover_datasets, load_pools
from core.logger import save_plot, setup_run
from core.monitor import OpenSetMonitor
from core.runner import save_json
from experiments.exp11_ancestor_comparison import _ANCESTOR_UNKNOWN_CANDIDATES, build_ancestor_monitor


def _canon(config: str) -> str:
    """把 1screw/1screws 兩種目錄名併成同一個標籤，避免跨資料集疊混淆矩陣時多出虛假類別。"""
    return "1screw" if config in ("1screw", "1screws") else config


def _dataset_key(ds: dict) -> str:
    return f"{ds['motor']}/{ds['rpm'].replace('rpm', '')}"


# ── (a) 二元混淆矩陣 ─────────────────────────────────────────────────────────


def _binary_confusion(all_pools: dict[str, dict[str, np.ndarray]], seed: int, confidence: float) -> dict:
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = [], []
    for pools in all_pools.values():
        monitor = OpenSetMonitor({HEALTHY: pools[HEALTHY]}, seed=seed, confidence=confidence)
        monitor.fit_initial()
        healthy = monitor.holdout(HEALTHY)
        faults = np.vstack([p for c, p in pools.items() if c != HEALTHY])

        healthy_pred = ["healthy" if s <= 1 else "fault" for s in monitor.score(healthy)]
        fault_pred = ["healthy" if s <= 1 else "fault" for s in monitor.score(faults)]
        y_true += ["healthy"] * len(healthy) + ["fault"] * len(faults)
        y_pred += healthy_pred + fault_pred

    labels = ["healthy", "fault"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": labels, "matrix": matrix.tolist()}


# ── (b) 開集多類混淆矩陣 ──────────────────────────────────────────────────────


def _collect_labels(monitor: OpenSetMonitor, unknown_configs: list[str],
                     pools: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    y_true, y_pred = [], []
    for config in monitor.known:
        X = monitor.holdout(config)
        preds = monitor.classify(X)
        y_true += [_canon(config)] * len(X)
        y_pred += [_canon(p) if p is not None else "unknown" for p in preds]
    for config in unknown_configs:
        X = pools[config]
        preds = monitor.classify(X)
        y_true += ["unknown"] * len(X)
        y_pred += [_canon(p) if p is not None else "unknown" for p in preds]
    return y_true, y_pred


def _openset_confusion(all_pools: dict[str, dict[str, np.ndarray]], seed: int,
                        confidence: float, method: str) -> dict:
    from sklearn.metrics import confusion_matrix

    y_true, y_pred = [], []
    for pools in all_pools.values():
        unknown = [c for c in _ANCESTOR_UNKNOWN_CANDIDATES if c in pools]
        monitor = build_ancestor_monitor(pools, seed=seed, confidence=confidence, method=method)
        dy_true, dy_pred = _collect_labels(monitor, unknown, pools)
        y_true += dy_true
        y_pred += dy_pred

    labels = ["8screws", "1screw", "2screws", "3screws", "4screws", "unknown"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": labels, "matrix": matrix.tolist()}


# ── (c) t-SNE ────────────────────────────────────────────────────────────────


def _tsne_one_dataset(pools: dict[str, np.ndarray], seed: int, confidence: float,
                       perplexity: float, max_per_class: int) -> dict:
    from sklearn.manifold import TSNE

    monitor = build_ancestor_monitor(pools, seed=seed, confidence=confidence, method="ledoit_wolf")
    rng = np.random.default_rng(seed)

    X_parts, config_labels, verdicts = [], [], []
    for config, pool in sorted(pools.items(), key=lambda kv: config_sort_key(kv[0])):
        n = min(len(pool), max_per_class)
        idx = rng.choice(len(pool), size=n, replace=False)
        sample = pool[idx]
        preds = monitor.classify(sample)
        for pred in preds:
            if config in monitor.known:
                verdicts.append("known_correct" if pred == config else
                                 ("unknown" if pred is None else "known_wrong"))
            else:
                verdicts.append("unknown" if pred is None else "known_wrong")
        X_parts.append(sample)
        config_labels += [_canon(config)] * n

    X = np.vstack(X_parts)
    X_scaled = monitor.scaler.transform(X)
    effective_perplexity = min(perplexity, max(5, X_scaled.shape[0] // 4))
    embedding = TSNE(random_state=seed, perplexity=effective_perplexity, init="pca").fit_transform(X_scaled)
    return {"embedding": embedding, "config_labels": config_labels, "verdicts": verdicts}


def _figure_confusion(binary: dict, openset_lw: dict, openset_legacy: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ConfusionMatrixDisplay(np.array(binary["matrix"]), display_labels=binary["labels"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("(a) 健康 vs 故障")
    ConfusionMatrixDisplay(np.array(openset_lw["matrix"]), display_labels=openset_lw["labels"]).plot(
        ax=axes[1], colorbar=False, cmap="Blues", xticks_rotation=45)
    axes[1].set_title("(b) 開集多類：ledoit_wolf")
    ConfusionMatrixDisplay(np.array(openset_legacy["matrix"]), display_labels=openset_legacy["labels"]).plot(
        ax=axes[2], colorbar=False, cmap="Blues", xticks_rotation=45)
    axes[2].set_title("(b) 開集多類：legacy")
    fig.tight_layout()
    return fig


def _figure_tsne(tsne_by_dataset: dict[str, dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(tsne_by_dataset)
    all_configs = sorted({c for v in tsne_by_dataset.values() for c in v["config_labels"]},
                          key=config_sort_key)
    palette = plt.get_cmap("tab10" if len(all_configs) <= 10 else "tab20")
    color_map = {c: palette(i / max(1, len(all_configs) - 1)) for i, c in enumerate(all_configs)}
    marker_map = {"known_correct": "o", "known_wrong": "x", "unknown": "^"}

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for ax, key in zip(axes.flat, keys):
        data = tsne_by_dataset[key]
        emb = data["embedding"]
        for verdict, marker in marker_map.items():
            mask = [v == verdict for v in data["verdicts"]]
            if not any(mask):
                continue
            colors = [color_map[c] for c, m in zip(data["config_labels"], mask) if m]
            ax.scatter(emb[mask, 0], emb[mask, 1], c=colors, marker=marker, s=14, alpha=0.7)
        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(keys):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c],
                           markersize=8, label=c) for c in all_configs]
    handles += [plt.Line2D([0], [0], marker=m, color="k", linestyle="", label=v)
                for v, m in marker_map.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=8)
    fig.suptitle("實驗十二：t-SNE（僅供視覺化，距離不具量尺意義）")
    fig.subplots_adjust(bottom=0.12)
    return fig


def run(data_root: str = "data", seed: int = 42, confidence: float = 0.95,
        perplexity: float = 30.0, tsne_max_per_class: int = 200) -> dict:
    all_pools = {_dataset_key(ds): load_pools(ds["path"]) for ds in discover_datasets(data_root)}

    binary = _binary_confusion(all_pools, seed, confidence)
    openset_lw = _openset_confusion(all_pools, seed, confidence, "ledoit_wolf")
    openset_legacy = _openset_confusion(all_pools, seed, confidence, "legacy")

    tsne_by_dataset = {
        key: _tsne_one_dataset(pools, seed, confidence, perplexity, tsne_max_per_class)
        for key, pools in all_pools.items()
    }

    return {
        "seed": seed, "confidence": confidence, "perplexity": perplexity,
        "n_datasets": len(all_pools),
        "binary_confusion": binary,
        "openset_confusion": {"ledoit_wolf": openset_lw, "legacy": openset_legacy},
        "_tsne_by_dataset": tsne_by_dataset,  # 內部用，_figure_tsne 消耗；summary.json 不保留座標
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-per-class", type=int, default=200)
    args = parser.parse_args()

    log, paths = setup_run("exp12_confusion_tsne")
    log.info(f"實驗十二：混淆矩陣與 t-SNE（seed={args.seed}, perplexity={args.perplexity}）")
    result = run(args.data_root, args.seed, args.confidence, args.perplexity, args.tsne_max_per_class)
    tsne_by_dataset = result.pop("_tsne_by_dataset")
    save_json(paths.output_dir / "summary.json", result)

    log.info(f"(a) 二元混淆矩陣 {result['binary_confusion']['labels']}："
              f"{result['binary_confusion']['matrix']}")
    for method, cm in result["openset_confusion"].items():
        log.info(f"(b) 開集多類混淆矩陣 [{method}] 對角線：{np.diag(cm['matrix']).tolist()}")

    save_plot(_figure_confusion(result["binary_confusion"], result["openset_confusion"]["ledoit_wolf"],
                                 result["openset_confusion"]["legacy"]),
              paths, "confusion_matrices.png")
    save_plot(_figure_tsne(tsne_by_dataset), paths, "tsne_grid.png")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
