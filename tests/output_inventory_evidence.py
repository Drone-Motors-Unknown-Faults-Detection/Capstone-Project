"""逐檔唯讀核對 output 與 writer；不刪除檔案，不將 writer 相符當作歷史執行證明。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from core.logger import setup_run


WRITERS = {
    "branch_integration": ["tests/integration_evidence.py"],
    "exp8_aggregate_baseline": ["tests/exp8_aggregate_baseline.py"],
    "exp8_aggregate_validation_full": ["tests/exp8_aggregate_validation.py"],
    "exp8_aggregate_validation_related": ["tests/exp8_aggregate_validation.py"],
    "exp8_health_index_aggregate": ["experiments/health_index_aggregate.py"],
    "navigation_regression": ["experiments/navigation_regression.py"],
    "stream_integration_regression": ["tests/stream_regression.py"],
    "web_server": ["web/live.py", "web/experiments.py"],
    "web_guide": ["web/live.py", "web/guide.py"],
    "exp6_formal_matrix": ["experiments/exp6_matrix.py", "experiments/aggregate_exp6.py"],
    "exp6_formal_api": ["experiments/exp6_formal_benchmark.py"],
    "source_provenance": ["core/provenance.py"],
    "environment_install": ["core/runtime_environment.py", "core/logger.py"],
}

# 僅列已由瀏覽器操作紀錄確認的外部截圖，不依 png 副檔名推斷來源。
EXTERNAL_SCREENSHOTS = {
    **{f"output/integration_browser/{name}.png":
       f"reports/Andy_20261008_分支整合/browser_qa/screenshots/{name}.png" for name in (
           "guide_reconnected", "guide_trend_a_done", "guide_trend_a", "guide_trend_b_done",
           "stream_batch_recovered", "stream_exp1_done", "stream_exp3_done", "stream_exp4_done",
           "stream_invalid_parameter")},
    **{f"output/web_guide/2026-10-08-21-50-53/{name}.png":
       f"reports/Andy_20261008_議題交付/browser_screenshots/2026-10-08-21-50-53/{name}.png"
       for name in ("anonymous", "candidate", "confirmed", "reset_reconnected", "trend_b")},
    "output/web_guide/2026-10-08-21-55-14/trend_a.png":
    "reports/Andy_20261008_議題交付/browser_screenshots/2026-10-08-21-55-14/trend_a.png",
}


def verify_external_relocations(root, baseline, moves=None):
    """固定版本讀取原檔 bytes；搬移不得變更內容，也不得留下 output 副本。"""
    rows = []
    for old, new in sorted((EXTERNAL_SCREENSHOTS if moves is None else moves).items()):
        original = subprocess.check_output(["git", "show", f"{baseline}:{old}"], cwd=root)
        target = root / new
        rows.append({"old": old, "new": new, "source_ref": baseline,
                     "source_sha256": hashlib.sha256(original).hexdigest(),
                     "target_sha256": sha(target) if target.is_file() else None,
                     "same_bytes": target.is_file() and original == target.read_bytes(),
                     "old_absent": not (root / old).exists()})
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(root: Path) -> dict:
    root = root.resolve(strict=True)
    output = (root / "output").resolve(strict=True)
    output.relative_to(root)
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "output"], cwd=root, text=True, encoding="utf-8").splitlines())
    sources, rows = {}, []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        path.resolve(strict=True).relative_to(output)
        relative = path.relative_to(root).as_posix()
        group = path.relative_to(output).parts[0]
        candidates = ["core/logger.py"] if path.name == "environment.json" else WRITERS.get(group, [])
        if not candidates:
            candidates = [f"tests/{group}.py", f"experiments/{group}.py"]
        evidence = []
        for name in candidates:
            source = root / name
            if not source.is_file():
                continue
            if name not in sources:
                lines = source.read_text(encoding="utf-8").splitlines()
                sources[name] = {"sha256": sha(source), "writer_lines": [
                    {"line": i, "code": line.strip()} for i, line in enumerate(lines, 1)
                    if any(token in line for token in ("write_text(", "to_csv(", "save_plot(", "save_json(",
                                                       "json.dump(", "open(", "setup_run(", "write_bytes("))]}
            text = source.read_text(encoding="utf-8")
            dynamic = ((name == "web/live.py" and re.fullmatch(r"(?:samples_epoch\d+\.csv|model_epoch\d+_\d+classes\.json)", path.name))
                       or (name == "web/guide.py" and re.fullmatch(r"fit_epoch\d+_classes\d+\.json", path.name))
                       or (name == "web/experiments.py" and re.fullmatch(r"exp\d+_[\d-]+\.json", path.name))
                       or (name == "tests/exp8_aggregate_validation.py" and re.fullmatch(r"check_\d+\.txt", path.name)))
            if sources[name]["writer_lines"] and (path.name in text or dynamic):
                evidence.append(name)
        category = "WRITER_MATCH_HISTORY_UNCONFIRMED" if evidence else "UNKNOWN_HISTORY"
        if relative in EXTERNAL_SCREENSHOTS:
            category, evidence = "DOCUMENTED_EXTERNAL_SCREENSHOT", []
        if group == "integration_browser" and path.name in {"guide_qa.md", "stream_qa.md"}:
            category, evidence = "CONFIRMED_MANUAL_QA_PR58", []
        parts = path.relative_to(output).parts
        log = root / "logs" / group / (parts[1] + ".log") if len(parts) > 1 else None
        rows.append({"path": relative, "sha256_bytes": sha(path), "bytes": path.stat().st_size,
                     "tracked": relative in tracked, "category": category,
                     "candidate_writers": evidence, "log": log.relative_to(root).as_posix() if log and log.is_file() else None,
                     "log_sha256": sha(log) if log and log.is_file() else None, "action": "KEEP"})
    return {"schema": "output_inventory_v1", "head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "scope": "僅指定 checkout 的 output；writer 相符不保證歷史版本、命令或完整來源鏈",
        "counts": dict(Counter(row["category"] for row in rows)), "files": rows, "sources": sources,
        "limits": ["未核實的歷史檔案全部保留", "截圖為外部工具操作證據，不是演算法輸出",
                   "原始 bytes SHA 與 Git 換行正規化 blob 分開解讀", "既有 PR58 搬移不在此程式重做"]}


def verify_document_relocations(root: Path, baseline: str, relocations: dict[str, str]) -> list[dict]:
    """核對指定文字搬移的 Git 正規化內容；原始位元組差異另由搬移紀錄說明。"""
    moves = []
    for old, new in sorted(relocations.items()):
        old_blob = subprocess.check_output(["git", "rev-parse", f"{baseline}:{old}"], cwd=root, text=True).strip()
        new_blob = subprocess.check_output(["git", "hash-object", "--path", new, new], cwd=root, text=True).strip()
        moves.append({"old": old, "new": new, "old_blob": old_blob, "new_normalized_blob": new_blob,
                      "same_git_content": old_blob == new_blob, "old_absent": not (root / old).exists(),
                      "new_exists": (root / new).is_file()})
    return moves


def verify_cleanup(root: Path) -> dict:
    """用 Git blob 核對既有搬移，另查目的檔及 manifest。"""
    baseline = "5a7865610fff07a455c0a23cec34fc5957ba3569"
    relocations = {f"output/integration_browser/{name}":
                   f"reports/Andy_20261008_分支整合/browser_qa/{name}"
                   for name in ("guide_qa.md", "stream_qa.md")}
    # 原臨時 python -c 命令已核對；未補寫專案 writer，不把摘要當演算法結果。
    relocations["output/exp8_aggregate_baseline/2026-10-09-03-28-48/test_summary.json"] = (
        "reports/Andy_20261009_實驗八基線測試/test_summary.json")
    moves = verify_document_relocations(root, baseline, relocations)
    changed = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=MDR", baseline, "HEAD", "--", "output"],
                                      cwd=root, text=True).splitlines()
    manifest = json.loads((root / "reports/Andy_20261008_分支整合/manifest.json").read_text(encoding="utf-8"))
    targets = manifest["verification"]["browser_qa"]
    from tests.issue_delivery_evidence import local_links
    documents = [root / "reports/Andy_20261008_分支整合/browser_qa/guide_qa.md",
                 root / "reports/Andy_20261008_分支整合/browser_qa/stream_qa.md"]
    links = [row for document in documents for row in local_links(document, root)]
    external = verify_external_relocations(root, "791216cf5602c370091fa516264f1ce6aaad6ab1")
    expected = [row["old"] for row in moves] + list(EXTERNAL_SCREENSHOTS)
    return {"baseline": baseline, "moves": moves, "external_relocations": external,
            "changed_output_paths": changed,
            "other_tracked_output_unchanged": sorted(changed) == sorted(expected),
            "manifest_targets": targets, "manifest_targets_exist": all((root / name).is_file() for name in targets),
            "local_links": links, "broken_links": [row for row in links if not row["exists"]]}


def run(root: Path) -> dict:
    result = inspect(root)
    result["cleanup_verification"] = verify_cleanup(root)
    log, paths = setup_run("output_inventory_evidence")
    (paths.output_dir / "inventory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("逐檔唯讀清冊 {} 份；本程式沒有刪除或搬移功能", len(result["files"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    result = run(parser.parse_args().root)
    verification = result["cleanup_verification"]
    valid = (verification["other_tracked_output_unchanged"] and verification["manifest_targets_exist"]
             and not verification["broken_links"] and all(
                 row["same_git_content"] and row["old_absent"] and row["new_exists"]
                 for row in verification["moves"]) and all(
                     row["same_bytes"] and row["old_absent"]
                     for row in verification["external_relocations"]))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
