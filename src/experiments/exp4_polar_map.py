"""實驗四：極座標健康地圖（Polar Health Map）——方向 = 故障類型、半徑 = 嚴重度。

把「星狀幾何」假設（健康為中心、故障各成射線）做成可量測的機制與誠實的檢驗。
初步實測（T1/8000rpm）顯示假設**部分成立**：中段均勻鬆動（6~3screws）共享一束
緊密射線（cos 0.94–0.97）、4_146 是獨立方向（vs 均勻族 0.45–0.50）；但兩端
（7s/1s）偏離主束、3_14 在方向上仍屬鬆動家族——因此本實驗以**連續的方向熟悉度
分數**（max cos）取代硬性 same-ray 判定，三個子實驗各自輸出可報告的量測：

(a) geometry   全類別射線結構：兩兩 cos 矩陣、族內/族間統計（星狀假設的直接證據）
(b) direction  方向熟悉度：只認識部分類別時，max-cos 能否把「未知的鬆動家族成員」
               與「真正的新方向（4_146）」分開——AUROC 與各配置分布
(c) severity   嚴重度可回復性：兩端錨點（7s、1s）下比較三種嚴重度定義
               （半徑 / 最佳射線投影 / 主束投影）的 Spearman 排序一致性

批次執行：
    venv/bin/python -m experiments.exp4_polar_map --motor T1 --rpm 8000rpm
    venv/bin/python -m experiments.exp4_polar_map --part a   # 只跑幾何驗證
"""

from __future__ import annotations

import argparse

import numpy as np

from core.data import HEALTHY, config_sort_key
from core.geometry import PolarMap
from core.logger import save_plot, setup_run
from core.monitor import OpenSetMonitor
from core.runner import add_dataset_args, resolve_dataset, save_json

UNIFORM = ["7screws", "6screws", "5screws", "4screws", "3screws", "2screws", "1screws"]
COMPOUND = ["3_14screws", "4_146screws"]


def _fit_monitor(pools, known_faults, seed, confidence=0.95, method="ledoit_wolf"):
    monitor = OpenSetMonitor(pools, seed=seed, confidence=confidence, method=method)
    monitor.fit_initial()
    for config in known_faults:
        monitor.add_class(config)
    return monitor


# ── (a) 射線結構 ────────────────────────────────────────────────────────────


def run_geometry(pools, seed: int = 42) -> dict:
    """全類別擬合 → 射線兩兩 cos 與族內/族間統計。"""
    faults = sorted((c for c in pools if c != HEALTHY), key=config_sort_key)
    pm = PolarMap(_fit_monitor(pools, faults, seed))
    names, M = pm.ray_cos_matrix()

    idx = {n: i for i, n in enumerate(names)}
    mid = [c for c in ["6screws", "5screws", "4screws", "3screws", "2screws"] if c in idx]

    def _pair_mean(group_a, group_b):
        vals = [M[idx[a], idx[b]] for a in group_a for b in group_b if a != b]
        return float(np.mean(vals)) if vals else None

    stats = {
        "uniform_all_mean": _pair_mean(UNIFORM, UNIFORM),
        "uniform_mid_mean": _pair_mean(mid, mid),
        "mid_vs_7screws": _pair_mean(mid, ["7screws"]),
        "mid_vs_1screws": _pair_mean(mid, ["1screws"]),
        "uniform_vs_3_14": _pair_mean(UNIFORM, ["3_14screws"]),
        "uniform_vs_4_146": _pair_mean(UNIFORM, ["4_146screws"]),
    }
    return {"ray_names": names, "cos_matrix": M.round(4).tolist(),
            "group_stats": {k: round(v, 4) for k, v in stats.items() if v is not None},
            "polar": pm.summary()}


# ── (b) 方向熟悉度 ──────────────────────────────────────────────────────────

KNOWN_SETS = [
    ["7screws"],
    ["1screws"],
    ["7screws", "1screws"],
    ["5screws"],
    ["3_14screws"],
]


def run_direction(pools, seed: int = 42) -> dict:
    """已知部分類別時，max-cos（方向熟悉度）分開「鬆動家族」與「新方向」的能力。

    正類 = 未知的均勻鬆動配置（期望熟悉度高），負類 = 4_146（期望低）；
    3_14 為困難案例，單獨報告不計入 AUROC。
    """
    from sklearn.metrics import roc_auc_score

    scenarios = []
    for known in KNOWN_SETS:
        pm = PolarMap(_fit_monitor(pools, known, seed))
        per_config, fam_scores, new_scores = {}, [], []
        for config in sorted(pools, key=config_sort_key):
            if config == HEALTHY or config in known:
                continue
            cos = pm.cosines(pools[config]).max(axis=1)
            per_config[config] = {
                "median": round(float(np.median(cos)), 4),
                "p10": round(float(np.percentile(cos, 10)), 4),
                "p90": round(float(np.percentile(cos, 90)), 4),
            }
            if config in UNIFORM:
                fam_scores.append(cos)
            elif config == "4_146screws":
                new_scores.append(cos)
        pos = np.concatenate(fam_scores)
        neg = np.concatenate(new_scores)
        auroc = float(roc_auc_score(
            np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
        scenarios.append({
            "known": known,
            "auroc_family_vs_4_146": round(auroc, 4),
            "median_family": round(float(np.median(pos)), 4),
            "median_4_146": round(float(np.median(neg)), 4),
            "median_3_14": per_config.get("3_14screws", {}).get("median"),
            "per_config": per_config,
        })
    return {"scenarios": scenarios}


# ── (c) 嚴重度可回復性 ──────────────────────────────────────────────────────


def run_severity(pools, seed: int = 42,
                 anchors: tuple[str, str] = ("7screws", "1screws")) -> dict:
    """兩端錨點下，三種嚴重度定義對中間等級的排序一致性（Spearman）。"""
    from scipy.stats import spearmanr

    monitor = _fit_monitor(pools, list(anchors), seed)
    pm = PolarMap(monitor)
    rays = {r.config: r for r in pm.rays}
    severe = rays[anchors[1]]
    # 主束方向：兩錨點射線單位向量的平均，再以最嚴重錨點在主束上的投影正規化
    bundle = sum(r.direction for r in pm.rays)
    bundle = bundle / np.linalg.norm(bundle)
    severe_proj = float(severe.radius * float(severe.direction @ bundle))

    ladder = [c for c in UNIFORM if c in pools]
    rows, samples = [], {"radius": ([], []), "best_ray": ([], []), "bundle": ([], [])}
    for rank, config in enumerate(ladder):
        X = pools[config]
        Z = pm.whiten_raw(X)
        radius = np.linalg.norm(Z, axis=1)
        defs = {
            "radius": radius / severe.radius,
            "best_ray": np.array([a["severity"] for a in pm.analyze(X)]),
            "bundle": (Z @ bundle) / severe_proj,
        }
        row = {"config": config, "rank": rank, "n": int(len(X))}
        for name, values in defs.items():
            row[f"sev_{name}_median"] = round(float(np.median(values)), 4)
            samples[name][0].extend([rank] * len(values))
            samples[name][1].extend(values.tolist())
        rows.append(row)

    spearman = {
        name: round(float(spearmanr(ranks, vals)[0]), 4)
        for name, (ranks, vals) in samples.items()
    }
    return {"anchors": list(anchors), "rows": rows, "spearman": spearman}


# ── 圖表與 CLI ──────────────────────────────────────────────────────────────


def _figure(geometry, severity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    names = [n.replace("screws", "s") for n in geometry["ray_names"]]
    M = np.array(geometry["cos_matrix"])
    im = axes[0].imshow(M, vmin=0.4, vmax=1.0, cmap="viridis")
    axes[0].set_xticks(range(len(names)), names, rotation=45, ha="right")
    axes[0].set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                         color="w" if M[i, j] < 0.8 else "k", fontsize=7)
    axes[0].set_title("(a) Pairwise ray cosine (fault directions from healthy center)")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    ladder = [r["config"].replace("screws", "s") for r in severity["rows"]]
    for name, marker in [("radius", "o"), ("best_ray", "s"), ("bundle", "^")]:
        axes[1].plot(ladder, [r[f"sev_{name}_median"] for r in severity["rows"]],
                     marker=marker, label=f"{name} (ρ={severity['spearman'][name]})")
    axes[1].set_title("(c) Severity definitions vs loosening grade (anchors: 7s & 1s)")
    axes[1].set_ylabel("median severity")
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def run(pools, seed: int = 42, parts: str = "abc") -> dict:
    """批次介面：把 iter_run 跑完，回傳最後的結果。"""
    for event in iter_run(pools, seed, parts):
        pass
    return event["result"]


def iter_run(pools, seed: int = 42, parts: str = "abc"):
    """逐段產生事件，供 Web 實驗頁邊跑邊畫；計算與 run() 完全相同。

    每個子實驗開始時產生 part_start、完成時產生 part（name = geometry / direction /
    severity，data 為該段結果），最後一個是 result（與 run() 回傳值相同）。
    """
    result = {"seed": seed}
    for key, name, fn in (("a", "geometry", run_geometry), ("b", "direction", run_direction),
                          ("c", "severity", run_severity)):
        if key in parts:
            yield {"event": "part_start", "name": name}
            result[name] = fn(pools, seed)
            yield {"event": "part", "name": name, "data": result[name]}
    yield {"event": "result", "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument("--part", default="abc", help="要跑的子實驗（a/b/c 任意組合）")
    args = parser.parse_args()

    ds, pools = resolve_dataset(args)
    log, paths = setup_run("exp4_polar_map")
    log.info(f"實驗四：極座標健康地圖 — {ds['motor']}/{ds['rpm']}（part={args.part}, seed={args.seed}）")

    result = run(pools, seed=args.seed, parts=args.part)
    result["dataset"] = {"motor": ds["motor"], "rpm": ds["rpm"]}
    save_json(paths.output_dir / "summary.json", result)

    if "geometry" in result:
        g = result["geometry"]["group_stats"]
        log.info(f"(a) 射線結構：中段均勻族內 cos={g['uniform_mid_mean']}, "
                 f"均勻 vs 3_14={g['uniform_vs_3_14']}, 均勻 vs 4_146={g['uniform_vs_4_146']}")
    if "direction" in result:
        for s in result["direction"]["scenarios"]:
            log.info(f"(b) 已知={'+'.join(s['known'])}: AUROC(家族 vs 4_146)={s['auroc_family_vs_4_146']}"
                     f"｜家族中位 {s['median_family']}｜4_146 中位 {s['median_4_146']}"
                     f"｜3_14 中位 {s['median_3_14']}")
    if "severity" in result:
        log.info(f"(c) 嚴重度 Spearman：{result['severity']['spearman']}")
        for r in result["severity"]["rows"]:
            log.info(f"    {r['config']:<10} radius={r['sev_radius_median']:.3f} "
                     f"best_ray={r['sev_best_ray_median']:.3f} bundle={r['sev_bundle_median']:.3f}")
    if "geometry" in result and "severity" in result:
        save_plot(_figure(result["geometry"], result["severity"]), paths, "polar_map.png")
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
