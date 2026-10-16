"""主線整合的測試證據入口；不訓練研究模型。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import unittest

from core.logger import setup_run

MAIN_BEFORE = "64cb71d84663e1745ec54db74abad68def67e8e6"


def command(args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return {"command": args, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def data_index(root):
    root = Path(root).resolve()
    rows = [{"path": str(p.relative_to(root)).replace("\\", "/"),
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*_Group_feature_data_clean.csv"))]
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"root": str(root), "files": rows, "file_count": len(rows),
            "inventory_sha256": hashlib.sha256(payload).hexdigest(),
            "version_note": "本輪實際可用唯讀檔案清冊；非歷史90檔指紋"}


def run(phase, data_root):
    log, paths = setup_run("branch_integration")
    # 沙箱系統暫存目錄可能缺少子目錄權限；只改本測試行程的暫存根。
    test_temp = paths.output_dir / "test_temp"
    test_temp.mkdir()
    tempfile.tempdir = str(test_temp.resolve())
    stream = io.StringIO()
    loader = unittest.TestLoader()
    if phase == "baseline":
        tracked = command(["git", "ls-tree", "-r", "--name-only", MAIN_BEFORE, "tests"])
        if tracked["returncode"]:
            raise RuntimeError(tracked["stderr"])
        modules = [p[:-3].replace("/", ".") for p in tracked["stdout"].splitlines()
                   if p.startswith("tests/test_") and p.endswith(".py")]
        suite = loader.loadTestsFromNames(modules)
    else:
        suite = loader.discover("src/tests", pattern="test_*.py", top_level_dir="src")
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    # 一個 test method 可以有數個 subTest 失敗，不能扣成數個未通過方法。
    failed_tests = {id(getattr(test, "test_case", test)) for test, _ in result.failures}
    error_tests = {id(getattr(test, "test_case", test)) for test, _ in result.errors}
    evidence = {"phase": phase, "python": platform.python_version(), "executable": sys.executable,
                "head": command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
                "main_before": MAIN_BEFORE, "tests_run": result.testsRun,
                "failed": len(failed_tests), "errors": len(error_tests),
                "failure_entries": len(result.failures), "error_entries": len(result.errors),
                "skipped": len(result.skipped),
                "passed": result.testsRun - len(failed_tests | error_tests) - len(result.skipped),
                "skip_reasons": [(str(test), reason) for test, reason in result.skipped],
                "seconds": time.monotonic() - started, "data": data_index(data_root),
                "pip_check": command([sys.executable, "-m", "pip", "check"]),
                "packages": command([sys.executable, "-m", "pip", "freeze"]),
                "cli": [command([sys.executable, "-m", module, "--help"])
                        for module in ("experiments.exp1_cold_start", "experiments.exp4_polar_map",
                                       "experiments.compare_openset", "web.server")],
                "git_diff_check": command(["git", "diff", "--check"])}
    (paths.output_dir / "tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    (paths.output_dir / "validation.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("{}：{} passed／{} failed／{} errors／{} skipped；{}",
             phase, evidence["passed"], evidence["failed"], evidence["errors"], evidence["skipped"], paths.output_dir)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "candidate", "post_merge"), required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    result = run(args.phase, args.data_root)
    ok = not (result["failed"] or result["errors"] or result["pip_check"]["returncode"]
              or any(row["returncode"] for row in result["cli"]))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
