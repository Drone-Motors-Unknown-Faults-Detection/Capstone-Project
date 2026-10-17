"""本輪文件收尾的相對目標與版本證據；不載入正式資料或訓練模型。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from core.logger import setup_run
from tests.issue_delivery_evidence import local_links


def run() -> dict:
    logger, paths = setup_run("project_closeout_evidence")
    root = Path(__file__).resolve().parents[2]
    names = ("TODO.md", "docs/README.md", "docs/experiments/exp8_health.md",
             "reports/Andy_20261009_主線交付/README.md",
             "reports/Andy_20261009_主線交付/execution_log.md", "docs/runtime_policy.md",
             "docs/ci_contract.md")
    documents = [root / name for name in names]
    links = [row for path in documents for row in local_links(path, root)]
    result = {
        "schema": "project_closeout_evidence_v1",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python": platform.python_version(),
        "files": [{"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                  for name in names],
        "local_links": links,
        "broken_links": [row for row in links if not row["exists"]],
        "limitations": ["僅核對選定文件的相對檔案目標，錨點未驗證", "遠端 URL 與原文內容另行稽核",
                        "不讀正式 data，不擬合，不提供新研究效能"],
    }
    (paths.output_dir / "evidence.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("{} 個文件，{} 個相對目標，{} 個失效", len(names), len(links), len(result["broken_links"]))
    return result


def main() -> int:
    return 1 if run()["broken_links"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
