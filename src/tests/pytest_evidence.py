"""執行 pytest 並保存公開摘要；不保存任意 traceback 或私人路徑。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from core.logger import setup_run


def summarize(payload: bytes, returncode: int) -> dict:
    root = ET.fromstring(payload)
    suites = list(root.iter("testsuite"))
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    status = "FAILED" if returncode or counts["failures"] or counts["errors"] or not counts["tests"] else (
        "INCOMPLETE" if counts["skipped"] else "PASS")
    return {**counts, "status": status, "returncode": returncode}


def validate_targets(targets: list[str]) -> None:
    if not targets or any(not re.fullmatch(r"src/tests/?|src/tests/test_[A-Za-z0-9_]+\.py", value) for value in targets):
        raise ValueError("只接受專案 tests 的相對目標，不公開私人路徑")


def run(targets: list[str]) -> dict:
    validate_targets(targets)
    log, paths = setup_run("pytest_evidence")
    with tempfile.TemporaryDirectory() as directory:
        xml = Path(directory) / "pytest.xml"
        command = [sys.executable, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"]
        process = subprocess.run(command, capture_output=True)
        result = summarize(xml.read_bytes(), process.returncode) if xml.is_file() else {
            "status": "FAILED", "returncode": process.returncode, "reason": "MISSING_JUNIT"}
    result.update(schema_version="pytest_evidence_v1", targets=targets,
                  output_sha256=hashlib.sha256(process.stdout + process.stderr).hexdigest(),
                  environment=json.loads((paths.output_dir / "environment.json").read_text(encoding="utf-8")),
                  diagnostic_policy="原始輸出與JUnit只留暫存／記憶體；skip明記INCOMPLETE，不算通過")
    (paths.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    log.info("pytest {}：{}", result["status"], {k: result.get(k) for k in ("tests", "failures", "errors", "skipped")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+")
    result = run(parser.parse_args().targets)
    return 0 if result["status"] == "PASS" else (2 if result["status"] == "INCOMPLETE" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
