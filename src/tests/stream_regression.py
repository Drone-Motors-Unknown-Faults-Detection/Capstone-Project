"""真實 CSV 與整合前 main 配對；工程回歸，不選模型。"""
import argparse
import hashlib
import importlib
import json

from core.data import discover_datasets, load_pools
from core.logger import setup_run
from tests.test_stream_integration import BASE, baseline
from tests.integration_evidence import data_index
from web.experiments import jsonable


def run(data_root, seed=42):
    log, paths = setup_run("stream_integration_regression")
    before = data_index(data_root)
    dataset = next(d for d in discover_datasets(data_root) if d["motor"] == "T1" and d["rpm"] == "8000rpm")
    pools = load_pools(dataset["path"])
    rows = []
    for module in ("exp1_cold_start", "exp3_trend", "exp4_polar_map"):
        new = importlib.import_module("experiments." + module)
        methods = ("mahalanobis", "knn") if module != "exp4_polar_map" else (None,)
        for method in methods:
            opts = {"seed": seed}
            if method:
                opts["openset_method"] = method
            if module == "exp3_trend":
                opts["n_trials"] = 2
            original = jsonable(baseline(module)(pools, **opts))
            updated = jsonable(new.run(pools, **opts))
            events = list(new.iter_run(pools, **opts))
            assert original == updated == jsonable(events[-1]["result"]), module
            payload = json.dumps(original, sort_keys=True).encode()
            rows.append({"module": module, "params": opts, "matched_original": True,
                         "result_sha256": hashlib.sha256(payload).hexdigest(),
                         "event_count": len(events), "result": updated})
    assert before == data_index(data_root)
    result = {"baseline": BASE, "source_unchanged": True, "data_inventory": before,
              "comparisons": rows, "claim": "同工況工程配對，非fresh研究；health_monitor排除"}
    (paths.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("{} 組與原 main 配對，全部相同，來源未變", len(rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_root, args.seed)


if __name__ == "__main__":
    main()
