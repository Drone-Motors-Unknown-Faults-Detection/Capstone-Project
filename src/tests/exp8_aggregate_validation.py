"""保存實驗八重算相關測試、完整回歸與環境檢查的實際輸出。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from core.logger import setup_run

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("related", "full"), default="related")
    args = parser.parse_args()
    logger, paths = setup_run(f"exp8_aggregate_validation_{args.scope}")
    target = ["tests.test_health_index_aggregate", "tests.test_health_matrix", "tests.test_health_benchmark",
              "tests.test_openset", "tests.test_geometry"] if args.scope == "related" else ["discover", "-s", "tests", "-t", "."]
    commands = [[sys.executable, "-m", "unittest", *target], [sys.executable, "-m", "pip", "check"]]
    checks = []
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
        text = result.stdout+result.stderr
        (paths.output_dir / f"check_{index}.txt").write_text(text, encoding="utf-8")
        match = re.search(r"Ran (\d+) tests?", text)
        checks.append({"command": command, "exit_code": result.returncode,
                       "tests_run": int(match.group(1)) if match else None, "output": f"check_{index}.txt"})
        logger.info(f"檢查 {index}：exit={result.returncode}，測試數={checks[-1]['tests_run']}")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = ("src/experiments/health_index_aggregate.py", "src/tests/test_health_index_aggregate.py", "src/tests/exp8_aggregate_validation.py")
    summary = {"scope": args.scope, "python": platform.python_version(), "commit_sha": sha,
               "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
               "checks": checks, "all_passed": all(item["exit_code"] == 0 for item in checks)}
    (paths.output_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
