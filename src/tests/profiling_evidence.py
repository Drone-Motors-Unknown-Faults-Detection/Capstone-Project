"""CSV 載入與 k-NN 評分的效能量測；量測規範與門檻見 docs/profiling_contract.md。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from core.data import HEALTHY, PoolCache, load_pools, make_split
from core.feature_schema import CONFIGURATIONS, SCHEMAS
from core.logger import setup_run
from core.openset import create_openset_detector

CONTRACT = "docs/profiling_contract.md"
MATRIX_LOADS = 9 * 54  # exp6_matrix：每個 run_id 都重算 9 個工況
THRESHOLDS = {
    "cache_load_share": 0.50,
    "cache_matrix_load_seconds": 60.0,
    "knn_class_ratio": 5.0,
    "knn_ten_class_seconds": 0.5,
}
HEALTHY_ROWS = 313
FAULT_ROWS = 304


def write_synthetic_dataset(root: Path, *, healthy_rows: int = HEALTHY_ROWS, fault_rows: int = FAULT_ROWS,
                            seed: int = 0) -> Path:
    """與 T1/8000rpm 同量級的合成工況；只用來看時間比例。"""
    rng = np.random.default_rng(seed)
    dataset = root / "Step-1" / "myfeature" / "T1" / "8000rpm"
    columns = list(SCHEMAS["historical_clean_v1"])
    for index, config in enumerate(CONFIGURATIONS):
        folder = dataset / config
        folder.mkdir(parents=True)
        rows = healthy_rows if config == HEALTHY else fault_rows
        frame = pd.DataFrame(rng.normal(index * 0.5, 1.0, size=(rows, len(columns))), columns=columns)
        frame.to_csv(folder / f"{config}_Group_feature_data_clean.csv", index=False)
    return dataset


def _measure(action: Callable[[], object]) -> tuple[object, dict]:
    """先不開 tracemalloc 計時，再另跑一次量記憶體峰值；tracemalloc 會拖慢純 Python 程式碼。"""
    started = time.perf_counter()
    value = action()
    seconds = time.perf_counter() - started
    tracemalloc.start()
    try:
        action()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, {"seconds": seconds, "python_peak_mb": peak / 2**20}


def _stats(samples: list[dict]) -> dict:
    seconds = [item["seconds"] for item in samples]
    memory = [item["python_peak_mb"] for item in samples]
    return {"n": len(samples), "median_seconds": statistics.median(seconds), "min_seconds": min(seconds),
            "max_seconds": max(seconds), "median_python_peak_mb": statistics.median(memory)}


def _pipeline_once(dataset: Path, method: str, seed: int) -> dict:
    pools, load = _measure(lambda: load_pools(dataset))
    rng = np.random.default_rng(seed)

    def fit():
        split = make_split(len(pools[HEALTHY]), rng)
        scaler = RobustScaler().fit(pools[HEALTHY][split.train])
        detector = create_openset_detector(method).fit(
            scaler.transform(pools[HEALTHY][split.train]), np.zeros(len(split.train), dtype=int),
            scaler.transform(pools[HEALTHY][split.cal]), np.zeros(len(split.cal), dtype=int))
        return split, scaler, detector

    (split, scaler, detector), fit_sample = _measure(fit)
    batch = np.vstack([pools[HEALTHY][split.holdout]] + [pools[c] for c in sorted(pools) if c != HEALTHY])
    scaled = scaler.transform(batch)
    _, score = _measure(lambda: detector.score_samples(scaled))
    return {"load": load, "fit": fit_sample, "score": score,
            "rows_loaded": int(sum(len(p) for p in pools.values())), "score_batch": int(len(batch)),
            "reference_bank": int(len(split.train))}


def _knn_by_classes(dataset: Path, class_counts: Sequence[int], seed: int, repeats: int, warmup: int) -> list[dict]:
    pools = load_pools(dataset)
    configs = [c for c in CONFIGURATIONS if c in pools]
    rng = np.random.default_rng(seed)
    splits = {c: make_split(len(pools[c]), rng) for c in configs}
    batch = np.vstack([pools[c] for c in configs])
    results = []
    for count in class_counts:
        known = configs[:count]
        X_train = np.vstack([pools[c][splits[c].train] for c in known])
        y_train = np.concatenate([np.full(len(splits[c].train), i) for i, c in enumerate(known)])
        X_cal = np.vstack([pools[c][splits[c].cal] for c in known])
        y_cal = np.concatenate([np.full(len(splits[c].cal), i) for i, c in enumerate(known)])
        scaler = RobustScaler().fit(X_train)
        detector = create_openset_detector("knn").fit(
            scaler.transform(X_train), y_train, scaler.transform(X_cal), y_cal)
        scaled = scaler.transform(batch)
        samples = [_measure(lambda: detector.score_samples(scaled))[1] for _ in range(warmup + repeats)]
        results.append({"known_classes": len(known), "score_batch": int(len(batch)),
                        "reference_bank": int(len(X_train)), "warmup": samples[:warmup],
                        "samples": samples[warmup:], "stats": _stats(samples[warmup:])})
    return results


def profile(dataset: Path, *, methods: Sequence[str] = ("mahalanobis", "knn"), repeats: int = 5,
            warmup: int = 1, seed: int = 42, class_counts: Sequence[int] = (1, 5, 10)) -> dict:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats 至少 1，warmup 不得為負")
    pipelines = {}
    for method in methods:
        runs = [_pipeline_once(dataset, method, seed) for _ in range(warmup + repeats)]
        measured = runs[warmup:]
        stats = {stage: _stats([run[stage] for run in measured]) for stage in ("load", "fit", "score")}
        total = sum(item["median_seconds"] for item in stats.values())
        pipelines[method] = {
            "warmup": runs[:warmup], "samples": measured, "stats": stats,
            "share_of_run": {stage: stats[stage]["median_seconds"] / total for stage in stats},
            "rows_loaded": measured[0]["rows_loaded"], "score_batch": measured[0]["score_batch"],
            "reference_bank": measured[0]["reference_bank"],
        }
    knn = _knn_by_classes(dataset, class_counts, seed, repeats, warmup)
    cache = PoolCache()
    cache.load(dataset)
    cached = [_measure(lambda: cache.load(dataset))[1] for _ in range(warmup + repeats)]
    uncached = min(item["stats"]["load"]["median_seconds"] for item in pipelines.values())
    cached_load = {"warmup": cached[:warmup], "samples": cached[warmup:], "stats": _stats(cached[warmup:])}
    cached_load["speedup_vs_load"] = uncached / cached_load["stats"]["median_seconds"]
    return {"schema_version": "profiling_v1", "contract": CONTRACT, "seed": seed, "repeats": repeats,
            "warmup": warmup, "os_page_cache": "not_controlled", "cpu_count": os.cpu_count(),
            "pipelines": pipelines, "knn_score_by_classes": knn, "cached_load": cached_load,
            "decision": decide(pipelines, knn)}


def decide(pipelines: dict, knn: list[dict]) -> dict:
    """依事先固定的門檻判定；不因量測結果改門檻。"""
    load_share = max(item["share_of_run"]["load"] for item in pipelines.values())
    load_median = max(item["stats"]["load"]["median_seconds"] for item in pipelines.values())
    matrix_seconds = load_median * MATRIX_LOADS
    cache = load_share >= THRESHOLDS["cache_load_share"] or matrix_seconds >= THRESHOLDS["cache_matrix_load_seconds"]
    by_classes = {row["known_classes"]: row["stats"]["median_seconds"] for row in knn}
    smallest, largest = min(by_classes), max(by_classes)
    ratio = by_classes[largest] / by_classes[smallest] if by_classes[smallest] > 0 else float("inf")
    vectorize = ratio >= THRESHOLDS["knn_class_ratio"] and by_classes[largest] >= THRESHOLDS["knn_ten_class_seconds"]
    return {
        "thresholds": THRESHOLDS,
        "csv_cache": {"worth_it": bool(cache), "max_load_share": load_share,
                      "estimated_matrix_load_seconds": matrix_seconds, "matrix_loads": MATRIX_LOADS},
        "knn_vectorization": {"worth_it": bool(vectorize), "class_ratio": ratio,
                              "classes_compared": [smallest, largest],
                              "largest_class_count_seconds": by_classes[largest]},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="單一工況目錄（含各配置子目錄）；不給則用合成資料")
    parser.add_argument("--methods", nargs="+", default=["mahalanobis", "knn"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    log, paths = setup_run("profiling_evidence")
    log.info(f"參數：{vars(args)}；規範 {CONTRACT}")
    with tempfile.TemporaryDirectory() as directory:
        if args.data_root:
            dataset, source = Path(args.data_root), "data_root"
        else:
            dataset, source = write_synthetic_dataset(Path(directory)), "synthetic_same_size_as_T1_8000rpm"
        result = profile(dataset, methods=args.methods, repeats=args.repeats, warmup=args.warmup, seed=args.seed)
    result["data_source"] = source
    (paths.output_dir / "profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    for method, item in result["pipelines"].items():
        shares = "、".join(f"{k} {v:.1%}" for k, v in item["share_of_run"].items())
        log.info(f"{method}：{shares}；load median {item['stats']['load']['median_seconds']:.4f}s")
    for row in result["knn_score_by_classes"]:
        log.info(f"k-NN {row['known_classes']} 類：median {row['stats']['median_seconds']:.4f}s")
    cached_load = result["cached_load"]
    log.info(f"PoolCache 命中：median {cached_load['stats']['median_seconds']:.4f}s，"
             f"比未快取快 {cached_load['speedup_vs_load']:.1f} 倍")
    decision = result["decision"]
    log.info(f"CSV cache 值得做：{decision['csv_cache']['worth_it']}；k-NN 批次化值得做：{decision['knn_vectorization']['worth_it']}")
    print(json.dumps({"output_dir": str(paths.output_dir), "csv_cache": decision["csv_cache"]["worth_it"],
                      "knn_vectorization": decision["knn_vectorization"]["worth_it"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
