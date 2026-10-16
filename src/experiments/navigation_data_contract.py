"""導覽展示的唯讀來源契約，不提供新的科學計算。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core.data import HEALTHY, discover_datasets, load_pools, config_sort_key
from core.logger import setup_run


def run(dataset: dict, seed: int = 42) -> dict:
    pools = load_pools(dataset["path"])
    if any(len(p) < 10 for p in pools.values()):
        raise ValueError("展示資料池不足10筆，不能建立可重播的基準")
    configs = sorted(pools, key=config_sort_key)
    sources = {}
    for config in configs:
        sources[config] = [
            {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "path": str(p.resolve())}
            for p in sorted((Path(dataset["path"]) / config).glob("*_Group_feature_data_clean.csv"))
        ]
    return {"schema": "navigation_data_v1", "motor": dataset["motor"],
            "rpm": dataset["rpm"], "seed": seed, "feature_dim": 105,
            "source_type": "CSV重播", "healthy": HEALTHY,
            "configs": [{"id": f"S{i:02d}", "config": c, "n": len(pools[c]),
                         "sha256": [s["sha256"] for s in sources[c]]}
                        for i, c in enumerate(configs)],
            "sources": sources, "raw_session_independence": "UNKNOWN",
            "fresh_final": "INCOMPLETE",
            "row_id_meaning": "既有concat後有限值列索引，非raw視窗"}


def split_audit(demo) -> dict:
    return {
        "schema": "navigation_fit_v1", "seed": demo.seed,
        "known_only": True, "fit_source": "既有完整標記配置池",
        "row_id_meaning": "concat後有限值列索引，搭配資料契約CSV SHA",
        "roles": {c: {"train": sp.train.tolist(), "calibration": sp.cal.tolist(),
                      "holdout": sp.holdout.tolist()}
                  for c, sp in demo.session.monitor.splits.items()},
        "model": demo.session.monitor.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--motor", default="T1")
    parser.add_argument("--rpm", default="8000rpm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    log, paths = setup_run("navigation_data_contract")
    ds = next(d for d in discover_datasets(args.data_root)
              if d["motor"] == args.motor and d["rpm"] == args.rpm)
    result = run(ds, args.seed)
    (paths.output_dir / "data_contract.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("唯讀資料契約完成；沒有訓練或評估")


if __name__ == "__main__":
    main()
