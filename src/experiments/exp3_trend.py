"""實驗三：漸進磨損 vs 突發事件判別（Trend Discrimination）。

在冷啟動監測器（實驗一的階段 0 狀態）之上，對開集分數的時間序列做
EWMA 異常比例 + CUSUM 變化點分析（core.trend），判別故障的發生模式：

- 劇本 A（漸進磨損）：健康 → 少量混入 7 螺絲 → 比例升高 → 6 → 5 螺絲
  模擬「螺絲逐漸鬆脫」的早期間歇性症狀與緩慢惡化。
- 劇本 B（突發外力）：健康 → 直接切換為 4 螺絲
  模擬外力損傷（如異物撞擊）造成的狀態跳變。

批次執行（多次隨機重複、產出判別準確率）：
    venv/bin/python -m experiments.exp3_trend --motor T1 --rpm 8000rpm --trials 20
Web 介面使用同一份 SCENARIOS 定義驅動即時劇本。
"""

from __future__ import annotations

import argparse

import numpy as np

from core.data import HEALTHY, CycleSampler
from core.logger import setup_run
from core.monitor import OpenSetMonitor
from core.runner import add_dataset_args, add_openset_args, resolve_dataset, save_json
from core.trend import TrendMonitor

# phase = (fault_config, 故障混入比例, ticks)；比例 < 1 時其餘機率抽健康 holdout
SCENARIOS = {
    "A": {
        "name": "劇本 A：漸進磨損",
        "expected": "gradual",
        "phases": [
            ("8screws", 0.0, 40),
            ("7screws", 0.25, 50),
            ("7screws", 0.7, 50),
            ("6screws", 1.0, 60),
            ("5screws", 1.0, 60),
        ],
    },
    "B": {
        "name": "劇本 B：突發外力",
        "expected": "sudden",
        "phases": [
            ("8screws", 0.0, 40),
            ("4screws", 1.0, 120),
        ],
    },
}

KIND_DISPLAY = {"gradual": "漸進退化（漂移）", "sudden": "突發事件（跳變）"}


def iter_stream(pools, monitor: OpenSetMonitor, phases, rng: np.random.Generator):
    """依劇本 phase 表產生 (樣本, 真實配置) 串流。"""
    healthy = CycleSampler(pools[HEALTHY], monitor.splits[HEALTHY].holdout, rng)
    faults: dict[str, CycleSampler] = {}
    for config, ratio, ticks in phases:
        fault = None
        if config != HEALTHY:
            fault = faults.setdefault(
                config, CycleSampler(pools[config], np.arange(len(pools[config])), rng)
            )
        for _ in range(ticks):
            if fault is not None and rng.random() < ratio:
                yield fault.draw(), config
            else:
                yield healthy.draw(), HEALTHY


def run(
    pools: dict[str, np.ndarray],
    n_trials: int = 20,
    seed: int = 42,
    confidence: float = 0.95,
    method: str = "ledoit_wolf",
    openset_method: str = "mahalanobis",
    knn_neighbors: int = 5,
) -> dict:
    """批次介面：把 iter_run 跑完，回傳最後的結果。"""
    for event in iter_run(pools, n_trials, seed, confidence, method, openset_method, knn_neighbors):
        pass
    return event["result"]


def iter_run(
    pools: dict[str, np.ndarray],
    n_trials: int = 20,
    seed: int = 42,
    confidence: float = 0.95,
    method: str = "ledoit_wolf",
    openset_method: str = "mahalanobis",
    knn_neighbors: int = 5,
):
    """逐步產生事件，供 Web 實驗頁邊跑邊畫；計算與 run() 完全相同。

    事件依序為 fitted、每筆串流一個 tick（分數、EWMA、CUSUM、是否警報）、
    每次重複結束一個 trial，最後一個是 result（與 run() 回傳值相同）。
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

    trials = []
    for key, scen in SCENARIOS.items():
        onset = scen["phases"][0][2]  # 第一段健康 phase 的長度 = 故障開始時間
        for k in range(n_trials):
            rng = np.random.default_rng(seed + 1000 * (ord(key) - ord("A") + 1) + k)
            trend = TrendMonitor()
            verdict, alarm_t, transition = None, None, None
            for t, (x, truth) in enumerate(iter_stream(pools, monitor, scen["phases"], rng), 1):
                score = float(monitor.score(x)[0])
                tr = trend.update(score)
                yield {"event": "tick", "scenario": key, "trial": k, "t": t, "truth": truth,
                       "score": score, "ewma": tr["ewma"], "cusum": tr["cusum"],
                       "alarm_now": tr["alarm_now"], "kind": tr["kind"], "onset": onset}
                if tr["alarm_now"]:
                    verdict, alarm_t, transition = tr["kind"], t, tr["transition"]
                    break
            trials.append(
                {
                    "scenario": key,
                    "trial": k,
                    "expected": scen["expected"],
                    "verdict": verdict,
                    "correct": verdict == scen["expected"],
                    "alarm_t": alarm_t,
                    "latency": None if alarm_t is None else alarm_t - onset,
                    "transition": transition,
                }
            )
            yield {"event": "trial", "trial": trials[-1]}

    summary = {}
    for key, scen in SCENARIOS.items():
        rows = [t for t in trials if t["scenario"] == key]
        alarmed = [t for t in rows if t["verdict"] is not None]
        summary[key] = {
            "name": scen["name"],
            "expected": scen["expected"],
            "trials": len(rows),
            "alarm_rate": len(alarmed) / len(rows),
            "correct_rate": float(np.mean([t["correct"] for t in rows])),
            "mean_latency": None if not alarmed else float(np.mean([t["latency"] for t in alarmed])),
            "mean_transition": None if not alarmed else float(np.mean([t["transition"] for t in alarmed])),
        }
    model = monitor.summary()
    yield {"event": "result", "result": {
        "trials": trials,
        "summary": summary,
        "model": model,
        "seed": seed,
        "n_trials": n_trials,
        "openset_method": openset_method,
        "method": method,
        "score_type": model["score_type"],
        "threshold": 1.0,
        "threshold_strategy": model["threshold_strategy"],
        "calibration_source": "known-only 20% calibration split",
        "split": {"train": 0.6, "calibration": 0.2, "holdout": 0.2},
        "knn_neighbors": knn_neighbors if openset_method == "knn" else None,
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument("--trials", type=int, default=20)
    add_openset_args(parser)
    args = parser.parse_args()

    ds, pools = resolve_dataset(args)
    log, paths = setup_run("exp3_trend")
    log.info(f"實驗三：漸進 vs 突發判別 — {ds['motor']}/{ds['rpm']}"
             f"（trials={args.trials}, openset={args.openset_method}, "
             f"method={args.method}, seed={args.seed}）")

    result = run(pools, n_trials=args.trials, seed=args.seed,
                 confidence=args.confidence, method=args.method,
                 openset_method=args.openset_method, knn_neighbors=args.knn_neighbors)
    result["dataset"] = {"motor": ds["motor"], "rpm": ds["rpm"]}

    import pandas as pd

    pd.DataFrame(result["trials"]).to_csv(paths.output_dir / "trials.csv", index=False)
    save_json(paths.output_dir / "summary.json",
              {k: v for k, v in result.items() if k != "trials"})

    for key, s in result["summary"].items():
        latency = "—" if s["mean_latency"] is None else f"{s['mean_latency']:.1f}"
        transition = "—" if s["mean_transition"] is None else f"{s['mean_transition']:.1f}"
        log.info(
            f"{s['name']}：警報率 {s['alarm_rate']*100:.0f}%｜判別正確率 {s['correct_rate']*100:.0f}%｜"
            f"平均警報延遲 {latency} 筆｜平均中間帶停留 {transition} 筆"
        )
    log.info(f"輸出：{paths.output_dir}")


if __name__ == "__main__":
    main()
