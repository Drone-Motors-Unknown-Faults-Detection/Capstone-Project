"""唯讀實際來源的格式／值／split配對稽核；不重新擬合模型。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from core.data import discover_datasets, load_pools, make_split
from core.feature_schema import SCHEMAS
from core.logger import setup_run

BASELINE = "22a253b14eb5df14058800f5a086bf4d7afeae58"
ROOT = Path(__file__).resolve().parents[1]


def baseline_module():
    source = subprocess.check_output(["git", "show", f"{BASELINE}:core/data.py"], cwd=ROOT)
    module = types.ModuleType("_schema_unchanged_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, "fixed_baseline/core/data.py", "exec"), module.__dict__)
    return module


def source_inventory(data_root):
    return {path.relative_to(data_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(data_root.glob("Step-*/myfeature/*/*/*/*_Group_feature_data_clean.csv"))}


def run(data_root: Path | str, *, generation_script: Path | str | None = None) -> dict:
    log, paths = setup_run("feature_schema_evidence")
    root = Path(data_root).resolve()
    before = source_inventory(root)
    old = baseline_module()
    datasets = discover_datasets(root)
    if not datasets:
        raise ValueError("找不到正式來源；不能用空清冊宣稱通過")
    results = []
    for dataset in datasets:
        audit = {}
        current = load_pools(dataset["path"], require_complete=True, audit=audit)
        previous = old.load_pools(dataset["path"])
        if list(current) != list(previous):
            raise AssertionError("配置／alias／排序改變")
        arrays, splits = {}, {}
        for config, values in current.items():
            np.testing.assert_array_equal(values, previous[config])
            arrays[config] = {"rows": len(values), "columns": values.shape[1],
                              "value_sha256": hashlib.sha256(values.tobytes()).hexdigest()}
            splits[config] = {}
            for seed in (42, 123, 2026):
                a = make_split(len(values), np.random.default_rng(seed))
                b = old.make_split(len(values), np.random.default_rng(seed))
                splits[config][str(seed)] = {}
                for role in ("train", "cal", "holdout"):
                    np.testing.assert_array_equal(getattr(a, role), getattr(b, role))
                    splits[config][str(seed)][role] = hashlib.sha256(getattr(a, role).tobytes()).hexdigest()
        results.append({"condition_id": dataset["path"].relative_to(root).as_posix(),
                        "motor": dataset["motor"], "rpm": dataset["rpm"], "audit": audit,
                        "array_audit": arrays, "split_audit": splits})
    script = None
    if generation_script is not None:
        payload = Path(generation_script).read_bytes()
        assignments = [node for node in ast.parse(payload.decode("utf-8-sig")).body
                       if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "feature_name"
                                                              for t in node.targets)]
        if len(assignments) != 1:
            raise ValueError("原腳本的feature_name不是唯一靜態assignment")
        columns = ast.literal_eval(assignments[0].value)
        if tuple(columns) != SCHEMAS["historical_clean_v1"]:
            raise ValueError("原腳本header與事前registry不符")
        script = {"source_id": Path(generation_script).name, "sha256": hashlib.sha256(payload).hexdigest(),
                  "line_start": assignments[0].lineno, "line_end": assignments[0].end_lineno,
                  "header_verified": True}
    after = source_inventory(root)
    if before != after:
        raise AssertionError("來源前後內容SHA改變")
    result = {"schema_version": "feature_schema_evidence_v1", "status": "VERIFIED", "baseline_commit": BASELINE,
              "file_count": len(before), "condition_count": len(datasets),
              "rows": sum(item["audit"]["accepted_rows"] for item in results),
              "source_content_unchanged": True, "values_and_split_indices_unchanged": True,
              "identity_scope": "來源相對ID／原檔SHA／原始列序不變；core.data未新增或重寫歷史sample ID",
              "raw_session_independence": "UNKNOWN", "fresh_test": False, "models_fitted": 0,
              "source_inventory": before, "generation_script": script, "datasets": results}
    (paths.output_dir / "schema_evidence.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("來源{}檔／{}工況／{}列；值與split相同；模型fit=0", len(before), len(datasets), result["rows"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--generation-script")
    args = parser.parse_args()
    run(args.data_root, generation_script=args.generation_script)


if __name__ == "__main__":
    main()
