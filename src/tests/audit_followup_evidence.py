"""#30 最新主線來源與文件目標核對；不重跑歷史研究或修改模型。"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from core.logger import setup_run
from tests.issue_delivery_evidence import SOURCE_FILES, local_links


def run() -> dict:
    logger, paths = setup_run("audit_followup_evidence")
    root = Path(__file__).resolve().parents[2]
    baseline = "4f783c676849a27185cd28d2410b4e76639b7357"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    sources = []
    for name in SOURCE_FILES:
        old = subprocess.check_output(["git", "show", f"{baseline}:{name}"], cwd=root)
        current = subprocess.check_output(["git", "show", f"{head}:{name}"], cwd=root)
        record = {"path": name, "baseline_sha256": hashlib.sha256(old).hexdigest(),
                  "head_sha256": hashlib.sha256(current).hexdigest(), "changed": old != current}
        if name.endswith(".py"):
            record["functions"] = [{"name": node.name, "line": node.lineno, "end_line": node.end_lineno}
                                   for node in ast.walk(ast.parse(current.decode("utf-8")))
                                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        sources.append(record)
    documents = [root / "docs/health_and_reports.md", root / "reports/Andy_20261009_主線交付/README.md",
                 root / "reports/Andy_20261008_議題交付/delivery.md"]
    links = [row for path in documents for row in local_links(path, root)]
    result = {"schema": "audit_followup_evidence_v1", "head": head, "baseline": baseline,
              "source_files": sources, "local_links": links,
              "broken_links": [row for row in links if not row["exists"]],
              "limits": ["Git blob／AST 核對不是重跑所有歷史反例", "PERF 未 profiling，root escape 未證明",
                         "不載入正式 data，不 fit，不宣稱新議題已修好"]}
    (paths.output_dir / "evidence.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("{} 份來源、{} 個文件目標、{} 個失效", len(sources), len(links), len(result["broken_links"]))
    return result


def main() -> int:
    return 1 if run()["broken_links"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
