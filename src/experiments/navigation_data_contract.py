"""導覽展示的唯讀來源規範，不提供新的科學計算。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.data import HEALTHY, discover_datasets, load_pools, config_sort_key
from core.logger import setup_run
from core.provenance import sha256_file


def run(dataset: dict, seed: int = 42) -> dict:
    pools = load_pools(dataset["path"])
    if any(len(p) < 10 for p in pools.values()):
        raise ValueError("展示資料池不足10筆，不能建立可重播的基準")
    configs = sorted(pools, key=config_sort_key)
    sources = {}
    # 只記相對資料根的 source_id（Step-*/myfeature/...），不記主機絕對路徑（#252）
    dataset_dir = Path(dataset["path"]).resolve()
    data_root = next((p.parent for p in (dataset_dir, *dataset_dir.parents) if p.name.startswith("Step-")), dataset_dir)
    for config in configs:
        sources[config] = [
            {"sha256": sha256_file(p),
             "path": p.resolve().relative_to(data_root).as_posix()}
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


def split_audit(demo, include_rows: bool = True) -> dict:
    """每個已知類的資料角色。配置池類別的索引是 concat 後有限值列索引；discovered-N 是 adopt() 只用候選群
    擬合的類，索引是候選群矩陣內的位置，`cluster_rows` 另記每個位置的來源池列（[配置, 池列號]，與矩陣同序）。
    導覽送給瀏覽器的版本 include_rows=False：cluster_rows 帶真值配置名，會洩漏匿名來源的答案（#296 R3-03）。"""
    session = demo.session
    mon = session.monitor
    roles = {}
    for config, sp in mon.splits.items():
        entry = {"train": sp.train.tolist(), "calibration": sp.cal.tolist(), "holdout": sp.holdout.tolist()}
        if config in mon.observed:
            entry.update({"fit_source": "隔離區候選群（adopt() 只用這一群已到達樣本）",
                          "row_id_meaning": "候選群矩陣內的位置（0～size-1），不是池列號",
                          "cluster_size": int(len(mon.observed[config]))})
            if include_rows:
                # 串流端沒給列號時（直接塞列的測試）每個位置是 [配置, None]
                entry["cluster_rows"] = session.learned_rows.get(config, {}).get("rows")
        else:
            entry.update({"fit_source": "既有完整標記配置池",
                          "row_id_meaning": "concat後有限值列索引，搭配資料規範CSV SHA"})
        roles[config] = entry
    return {
        "schema": "navigation_fit_v2", "seed": demo.seed,
        "known_only": True,
        "roles": roles,
        "model": mon.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--motor", default="T1")
    parser.add_argument("--rpm", default="8000rpm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    log, paths = setup_run("navigation_data_contract", data_root=args.data_root)
    ds = next(d for d in discover_datasets(args.data_root)
              if d["motor"] == args.motor and d["rpm"] == args.rpm)
    result = run(ds, args.seed)
    (paths.output_dir / "data_contract.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("唯讀資料規範完成；沒有訓練或評估")


if __name__ == "__main__":
    main()
