"""唯讀重播現行 loader／splitter，保存有效資料用量；不擬合或重跑研究。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.data import HEALTHY, discover_datasets, load_pools, make_split
from core.feature_schema import CONFIG_ALIASES
from core.logger import setup_run
from experiments.exp11_ancestor_comparison import (
    _ANCESTOR_UNKNOWN_CANDIDATES,
    _ancestor_known_configs,
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def allocation(pools: dict, known: list[str], seed: int) -> dict:
    """依註冊順序使用真正的 make_split；保留計數、互斥檢查及索引摘要。"""
    rng = np.random.default_rng(seed)
    rows = []
    for config in known:
        n = len(pools[config])
        split = make_split(n, rng)
        roles = {name: getattr(split, name).tolist() for name in ("train", "cal", "holdout")}
        sets = {name: set(ids) for name, ids in roles.items()}
        overlap = sum(len(sets[a] & sets[b]) for a, b in (
            ("train", "cal"), ("train", "holdout"), ("cal", "holdout")))
        covered = set.union(*sets.values()) == set(range(n))
        rows.append({"config": config, "pool_n": n, **{k: len(v) for k, v in roles.items()},
                     "ratio_denominator": n,
                     "fractions": {k: len(v) / n for k, v in roles.items()},
                     "overlap_n": overlap, "all_rows_accounted": covered,
                     "indices_sha256": _digest(roles)})
    return {"seed": seed, "registration_order": known, "classes": rows,
            **{name: sum(r[name] for r in rows) for name in ("train", "cal", "holdout")}}


def inspect(data_root: Path | str, seeds: tuple[int, ...] = (42, 123, 2026)) -> dict:
    root = Path(data_root)
    datasets = discover_datasets(root)
    if not datasets or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("需有可用工況及不重複的 seeds")
    inventory = []
    loaded = []
    seen = set()
    for ds in datasets:
        key = f"{ds['motor']}/{ds['rpm']}"
        if key in seen:
            raise ValueError(f"工況鍵重複，不能默默合併：{key}")
        seen.add(key)
        pools = load_pools(ds["path"])
        config_rows = []
        for folder in sorted(p for p in ds["path"].iterdir() if p.is_dir()):
            files = sorted(folder.glob("*_Group_feature_data_clean.csv"))
            if not files:
                continue
            config = CONFIG_ALIASES.get(folder.name, folder.name)  # 池的 key 是正規名（#271）
            frames = []
            for file in files:
                payload = file.read_bytes()
                inventory.append({"relative_path": file.relative_to(root).as_posix(),
                                  "sha256": hashlib.sha256(payload).hexdigest()})
                frames.append(pd.read_csv(file))
            matrix = pd.concat(frames, ignore_index=True).select_dtypes("number").to_numpy(dtype=float)
            finite_n = int(np.isfinite(matrix).all(axis=1).sum())
            if finite_n != len(pools.get(config, [])):
                raise ValueError("數量與現行 loader 不一致")
            config_rows.append({"config": config, "directory": folder.name, "files": len(files),
                                "raw_feature_rows": len(matrix),
                                "numeric_columns": matrix.shape[1], "finite_rows": finite_n,
                                "discarded_nonfinite": len(matrix) - finite_n})
        unknown = {c: len(p) for c, p in pools.items() if c != HEALTHY}
        known_a = _ancestor_known_configs(pools)
        loaded.append({"dataset": key, "motor": ds["motor"], "rpm": ds["rpm"],
                       "configurations": config_rows, "total_finite_rows": sum(map(len, pools.values())),
                       "cold_start": [allocation(pools, [HEALTHY], s) for s in seeds],
                       "unknown_test": unknown, "unknown_test_n": sum(unknown.values()),
                       "ancestor_a": [allocation(pools, known_a, s) for s in seeds],
                       "ancestor_a_unknown_n": sum(len(pools[c]) for c in _ANCESTOR_UNKNOWN_CANDIDATES if c in pools),
                       "tsne_plot_n_at_cap_200": sum(min(len(p), 200) for p in pools.values()),
                       "monitor_default_120": {c: min(120, len(make_split(len(p), np.random.default_rng(seeds[0])).holdout)
                                                      if c == HEALTHY else len(p)) for c, p in pools.items()},
                       "transfer_support": [{"requested_n": n, "effective_n": min(n, len(pools[HEALTHY])),
                                             "target_healthy_test_n": max(0, len(pools[HEALTHY]) - n),
                                             "scratch": allocation({HEALTHY: pools[HEALTHY][:n]}, [HEALTHY], seeds[0])}
                                            for n in (10, 25, 50)]})
    # exp5 的策略子實驗：三種基準都只用外層 split(seed+7) 的 train+cal 擬合，monitor 再在其中內切。
    strategy = []
    by_key = {row["dataset"]: row for row in loaded}
    for seed in seeds:
        rng = np.random.default_rng(seed + 7)
        for key in sorted(by_key):
            n = next(r["finite_rows"] for r in by_key[key]["configurations"] if r["config"] == HEALTHY)
            outer = make_split(n, rng)
            fit_rows = np.r_[outer.train, outer.cal]
            inner = make_split(len(fit_rows), np.random.default_rng(seed))
            strategy.append({"seed": seed, "dataset": key, "outer_healthy_test_n": len(outer.holdout),
                             "per_condition_train_overlap_n": len(np.intersect1d(outer.holdout, fit_rows[inner.train])),
                             "per_condition_cal_overlap_n": len(np.intersect1d(outer.holdout, fit_rows[inner.cal]))})
    # 讀取後再驗證原位元組；不把 mtime 或換行標準化當內容證據。
    if any(hashlib.sha256((root / r["relative_path"]).read_bytes()).hexdigest() != r["sha256"] for r in inventory):
        raise ValueError("查核期間來源內容改變")
    return {"schema_version": "experiment_data_usage_v1", "basis": "現行 loader 與 splitter 唯讀重播；配置數量非實際消耗或新研究分數",
            "models_fitted": 0, "source_inventory": inventory, "content_inventory_sha256": _digest(inventory),
            "seeds": list(seeds), "datasets": loaded, "n_datasets": len(loaded),
            "total_finite_rows": sum(r["total_finite_rows"] for r in loaded),
            "exp5_strategy_overlap": strategy, "raw_session_independence": "UNKNOWN", "fresh_test": "UNKNOWN",
            "limitations": ["索引屬過濾後特徵列，不能回推 raw 視窗或錄製群組",
                            "循環重播／多 seed／多方法重用資料，不增加獨立樣本",
                            "候選到達與更新後的實際消耗須查該 session，不能從配置用量推算"]}


def run(data_root: Path | str, seeds: tuple[int, ...] = (42, 123, 2026)) -> dict:
    log, paths = setup_run("experiment_data_usage_evidence", data_root=data_root)
    result = inspect(data_root, seeds)
    (paths.output_dir / "usage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("唯讀 {} 工況、{} 有效列；模型擬合 0 次", result["n_datasets"], result["total_finite_rows"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seeds", default="42,123,2026")
    args = parser.parse_args()
    run(args.data_root, tuple(int(v) for v in args.seeds.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
