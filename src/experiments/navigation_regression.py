"""相同來源／seed 下，首頁 LiveDemo 與導覽 GuideHub 的有界逐筆配對回歸。"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from core.data import discover_datasets
from core.logger import setup_run
from experiments.navigation_data_contract import run as contract, split_audit
from web.guide import GuideHub
from web.live import LiveDemo

SCHEMA = "navigation_regression_v3"
MAX_MISMATCH_DETAILS = 20


class MismatchLedger:
    """逐項比對原 LiveDemo（cli）與導覽（guide）；不一致計數並留前幾筆細節。
    不用 assert：python -O 會略過 assert，回歸就會在不一致時照樣回報 0 筆（#301 F-04）。"""

    def __init__(self, limit: int = MAX_MISMATCH_DETAILS):
        self.count = 0
        self.details: list[dict] = []
        self.limit = limit

    def check(self, pair_id: str, field: str, cli, guide) -> bool:
        if cli == guide:
            return True
        self.count += 1
        if len(self.details) < self.limit:
            self.details.append({"pair_id": pair_id, "field": field, "cli": cli, "guide": guide})
        return False


class Collector:
    def __init__(self):
        self.sample = None

    def write_message(self, raw):
        msg = json.loads(raw)
        if msg["type"] == "sample":
            self.sample = msg


def tick_with_row(demo, tick):
    """只記錄既有draw結果，不改RNG或科學計算；同值列保留全部索引。"""
    rows = []
    originals = []
    for config, sampler in demo.samplers.items():
        draw = sampler.draw
        originals.append((sampler, draw))

        def recorded(draw=draw, config=config):
            x = draw()
            candidates = np.flatnonzero(np.all(demo.pools[config] == x, axis=1)).tolist()
            rows.append({"config": config, "finite_row_indices": candidates,
                         "feature_sha256": hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()})
            return x

        sampler.draw = recorded
    try:
        value = tick()
        if len(rows) != 1:
            raise RuntimeError(f"一拍應抽 1 筆來源列，實際 {len(rows)} 筆")
        return value, rows[0]
    finally:
        for sampler, draw in originals:
            sampler.draw = draw


def compare_adoption(mismatch: MismatchLedger, method: str, original, guide) -> dict | None:
    """先比兩入口是否都完成 adopt()；只有一邊有結果（任一方向）就記不一致並回 None，
    不去讀另一邊 None 的欄位（#304 F5-05）。兩邊都有才比內容與 fit audit，回傳 CLI 的結果。"""
    pair_id = f"{method}/adopted"
    cli_done, guide_done = original.adoption is not None, guide.adoption is not None
    if not mismatch.check(pair_id, "adoption_present", cli_done, guide_done) or not cli_done:
        return None
    adopted = {k: v for k, v in original.adoption.items() if k != "label"}
    mismatch.check(pair_id, "adoption", adopted, {k: v for k, v in guide.adoption.items() if k != "label"})
    mismatch.check(pair_id, "fit_audit", split_audit(original), split_audit(guide))
    return adopted


def run(dataset, seed=42):
    source = contract(dataset, seed)
    results = []
    keys = ("t", "score", "verdict", "pca", "ewma", "cusum", "quarantine", "attempts", "phase")
    for method in ("mahalanobis", "knn"):
        options = {"openset_method": method, "mahalanobis_method": "ledoit_wolf",
                   "confidence": .95, "knn_neighbors": 5}
        hub = GuideHub([dataset], seed=seed, detector_options=options)
        hub.build(dataset["motor"], dataset["rpm"])
        original = LiveDemo(dataset, seed=seed, **options)
        collector = Collector()
        hub.clients.add(collector)
        paired = 0
        ledger = []
        mismatch = MismatchLedger()
        audits = [{"stage": "initial", "fit": split_audit(original)}]
        def tick():
            nonlocal paired
            messages, expected_row = tick_with_row(original, original.tick)
            expected = next(m for m in messages if m["type"] == "sample")
            hub.running = True
            _, guide_row = tick_with_row(hub.demo, hub.tick)
            pair_id = f"{method}/{paired}"
            mismatch.check(pair_id, "source", expected_row, guide_row)
            for k in keys:
                mismatch.check(pair_id, k, expected[k], collector.sample[k])
            mismatch.check(pair_id, "monitor_summary", original.session.monitor.summary(),
                           hub.demo.session.monitor.summary())
            ledger.append({"pair_id": pair_id, "epoch": original.epoch,
                           "source": expected_row, "cli": {k: expected[k] for k in keys},
                           "guide": {k: collector.sample[k] for k in keys}})
            paired += 1
        try:
            for _ in range(40):
                tick()
            original.set_source("5screws")
            hub.demo.set_source("5screws")
            arrival = 0
            # 成群當拍 LiveDemo.tick() 就 adopt()：等到第一次自學結果出現
            while arrival < 600 and hub.demo.adoption is None:
                tick()
                arrival += 1
            found = hub.demo.adoption is not None
            adopted = compare_adoption(mismatch, method, original, hub.demo)
            if adopted is not None:
                audits.append({"stage": "adopted", "fit": split_audit(original)})
                for _ in range(60):
                    tick()
            trends = {}
            for name, total in (("A", 260), ("B", 160)):
                original._build()
                hub.demo._build()
                original.start_scenario(name)
                hub.demo.start_scenario(name)
                hub.scenario_end = total
                for _ in range(total):
                    tick()
                mismatch.check(f"{method}/scenario_{name}", "guide_running_after_end", False, hub.running)
                mismatch.check(f"{method}/scenario_{name}", "alarms", original.alarms, hub.demo.alarms)
                trends[name] = hub.demo.alarms
            results.append({"method": method, "candidate_found": found,
                            "fault_arrival_count": arrival, "adoption": adopted,
                            "paired_samples": paired, "mismatches": mismatch.count,
                            "mismatch_details": mismatch.details, "trend_alarms": trends,
                            "ledger": ledger, "fit_audits": audits})
        finally:
            original.close()
            hub.demo.close()
            hub.executor.shutdown()
    source_unchanged = source == contract(dataset, seed)
    mismatches = sum(item["mismatches"] for item in results)
    return {"schema": SCHEMA, "status": "VERIFIED" if mismatches == 0 and source_unchanged else "FAILED",
            "data": source, "seed": seed, "paired_methods": results, "mismatches": mismatches,
            "source_unchanged": source_unchanged,
            "claim": "工程計算一致性，非模型可靠性或fresh盲測"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    log, paths = setup_run("navigation_regression", data_root=args.data_root)
    dataset = next(d for d in discover_datasets(args.data_root)
                   if d["motor"] == "T1" and d["rpm"] == "8000rpm")
    summary_path = paths.output_dir / "summary.json"
    try:
        result = run(dataset, args.seed)
    except Exception as exc:
        # 中途崩潰也留失敗紀錄，不讓 output 只剩 environment.json
        failure = {"schema": SCHEMA, "status": "FAILED", "seed": args.seed,
                   "error": f"{type(exc).__name__}: {exc}"}
        summary_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        log.error("回歸中斷：{}", failure["error"])
        raise
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["status"] != "VERIFIED":
        log.error("回歸不一致：{} 筆不一致，來源未變={}", result["mismatches"], result["source_unchanged"])
        return 1
    log.info("兩方法逐筆配對完成；來源未變")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
