"""重算封存配對與導覽尾筆，不擬合或改寫歷史產物。"""
import csv
import hashlib
import json
from pathlib import Path

from core.logger import setup_run


def run():
    log, paths = setup_run("guide_delivery_evidence")
    paired = Path("output/navigation_regression/2026-10-08-21-52-04/summary.json")
    data = json.loads(paired.read_text(encoding="utf-8"))
    methods = []
    for method in data["paired_methods"]:
        rows = method["ledger"]
        assert len(rows) == method["paired_samples"] == 595
        assert all(row["cli"] == row["guide"] and row["source"]["finite_row_indices"] for row in rows)
        assert len({row["pair_id"] for row in rows}) == len(rows)
        methods.append({"method": method["method"], "pairs": len(rows), "mismatches": 0})
    folders = ("2026-10-08-21-50-53", "2026-10-08-21-55-14")
    expected = {("2026-10-08-21-50-53", "samples_epoch01.csv"): 423,
                ("2026-10-08-21-50-53", "samples_epoch02.csv"): 160,
                ("2026-10-08-21-55-14", "samples_epoch01.csv"): 260}
    files, tails = [], []
    for folder in folders:
        for path in sorted((Path("output/web_guide") / folder).rglob("*")):
            if not path.is_file():
                continue
            files.append({"path": path.as_posix(), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                limit = expected[(folder, path.name)]
                assert len(rows) == limit
                assert [int(row["t"]) for row in rows] == list(range(1, limit + 1))
                tails.append({"path": path.as_posix(), "rows": len(rows), "last_t": limit,
                              "complete": True})
    result = {"paired_source": paired.as_posix(), "paired_sha256": hashlib.sha256(paired.read_bytes()).hexdigest(),
              "methods": methods, "csv_tails": tails, "files": files,
              "browser": {"name": "Codex In-app Browser", "engine_version": "UNKNOWN"},
              "limitations": ["只有T1/8000互動", "原首頁尚無直接導覽入口", "工程一致不等於可靠模型"]}
    (paths.output_dir / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("1190筆配對、3個CSV尾筆與{}檔SHA核對完成", len(files))
    return result


def main():
    run()


if __name__ == "__main__":
    main()
