"""唯讀比對本輪內容來源與既有逐檔 SHA 清冊，不重新訓練模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.logger import setup_run
from core.provenance import capture_source, file_sha


def run(data_root: Path, baseline: Path) -> dict:
    log, paths = setup_run("provenance_delivery_evidence", unique=True)
    before, _ = capture_source(data_root)
    inventory = {row["source_id"]: row["sha256"] for row in before["files"]}
    previous = json.loads(baseline.read_text(encoding="utf-8"))
    if inventory != previous["source_inventory"]:
        raise AssertionError("來源與既有逐檔SHA不符，不宣稱資料未變")
    if capture_source(data_root)[0] != before:
        raise AssertionError("核對期間來源改變")
    root = Path(__file__).resolve().parents[1]
    result = {"schema_version": "provenance_delivery_evidence_v1", "status": "VERIFIED",
              "file_count": len(inventory), "dataset_fingerprint": before["dataset_fingerprint"],
              "fingerprint_version": before["schema_version"], "baseline_evidence_sha256": file_sha(baseline),
              "baseline_file_count": previous["file_count"], "baseline_rows": previous["rows"],
              "source_bytes_unchanged": True, "models_fitted": 0,
              "manifest_validation": before["manifest_validation"],
              "raw_session_independence": "UNKNOWN", "fresh_test": "UNKNOWN",
              "source_code_sha256": {name: file_sha(root / name) for name in (
                  "core/provenance.py", "core/logger.py", "core/formal_data.py", "experiments/exp6_formal_benchmark.py")}}
    (paths.output_dir / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("{} 檔與既有來源SHA完全相同；沒有擬合正式模型", len(inventory))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_root, args.baseline)


if __name__ == "__main__":
    main()
