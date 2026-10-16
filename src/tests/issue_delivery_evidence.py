"""本輪issue交付的唯讀來源、函數行號及相對連結證據；不擬合模型。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

from core.data import discover_datasets
from core.logger import setup_run
from tests.integration_evidence import data_index

BASE = "4f783c676849a27185cd28d2410b4e76639b7357"
OLD = "64cb71d84663e1745ec54db74abad68def67e8e6"
HISTORY = "83846b2967bb1f925e457db8093cfdb2d430c1bc"
SOURCE_FILES = (
    "core/data.py", "core/formal_data.py", "core/runner.py", "core/openset.py",
    "experiments/exp1_cold_start.py", "experiments/exp2_scale_growth.py",
    "experiments/exp3_trend.py", "experiments/exp6_formal_benchmark.py",
    "experiments/exp6_matrix.py", "experiments/aggregate_exp6.py", "web/server.py",
    "web/live.py", "docs/README.md", "README.md", "build_uv.sh", "run_web.sh",
)
HISTORY_FILES = (
    "reports/issue_delivery_20261005/main_reaudit.md",
    "reports/issue_delivery_20261005/roadmap_issue_drafts.md",
    "reports/issue_delivery_20261005/remaining_issue_triage.md",
    "reports/issue_delivery_20261005/health_and_reports_section22_patch.md",
    "output/fault_type_issue_triage/2026-10-06-18-37-13/main_review.json",
)


def git_bytes(*args):
    return subprocess.check_output(["git", *args])


def source_record(path):
    old = git_bytes("show", f"{OLD}:{path}")
    current = git_bytes("show", f"{BASE}:{path}")
    record = {"path": path, "old_ref": OLD, "main_ref": BASE,
              "old_sha256": hashlib.sha256(old).hexdigest(),
              "main_sha256": hashlib.sha256(current).hexdigest(),
              "changed_by_integration": old != current}
    if path.endswith(".py"):
        record["functions"] = [{"name": node.name, "line": node.lineno,
                                "end_line": node.end_lineno}
                               for node in ast.walk(ast.parse(current.decode("utf-8")))
                               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return record


def local_links(path, root):
    rows = []
    for value in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        target = value.strip().split(' "')[0].strip("<>")
        parts = urlsplit(target)
        if parts.scheme or target.startswith("//"):
            continue
        resolved = (path.parent / unquote(parts.path)).resolve() if parts.path else path.resolve()
        rows.append({"source": str(path.relative_to(root)), "link": target,
                     "target": str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved),
                     "exists": resolved.exists(),
                     "anchor": parts.fragment, "anchor_checked": False})
    return rows


def run(data_root):
    log, paths = setup_run("issue_delivery_evidence")
    root = Path.cwd().resolve()
    documents = [root / "docs/README.md", root / "docs/health_and_reports.md",
                 root / "README.md", *sorted((root / "reports/Andy_20261008_議題交付").glob("*.md")),
                 *sorted((root / "docs/navigation").glob("*.md"))]
    links = [row for path in documents for row in local_links(path, root)]
    data = data_index(data_root)
    conditions = []
    for ds in discover_datasets(data_root):
        conditions.append({"motor": ds["motor"], "rpm": ds["rpm"],
                           "configurations": sorted({p.parent.name for p in ds["path"].glob("*/*_Group_feature_data_clean.csv")})})
    historical = [{"ref": HISTORY, "path": path,
                   "sha256": hashlib.sha256(git_bytes("show", f"{HISTORY}:{path}")).hexdigest()}
                  for path in HISTORY_FILES]
    result = {"schema": "issue_delivery_evidence_v1", "head": git_bytes("rev-parse", "HEAD").decode().strip(),
              "main_ref": BASE, "source_files": [source_record(p) for p in SOURCE_FILES],
              "historical_sources": historical, "data": data, "conditions": conditions,
              "local_links": links, "broken_links": [row for row in links if not row["exists"]],
              "limits": ["只核相對檔案目標，不核所有Markdown錨點", "未做profiling或root escape",
                         "未知raw/session獨立性不由hash證明", "不是重跑17項舊fixture"]}
    (paths.output_dir / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("{}檔來源、{}個工況、{}個相對目標，{}個失效", len(SOURCE_FILES), len(conditions), len(links), len(result["broken_links"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    result = run(args.data_root)
    raise SystemExit(1 if result["broken_links"] else 0)


if __name__ == "__main__":
    main()
