"""CI 工程證據：真實執行指定命令，只公開白名單摘要。

pytest／ruff 改由 CI yaml 的獨立 step（./scripts/quantity/run_pytest.sh、./scripts/quantity/run_ruff.sh）把關，
不在這裡重複跑；這裡只留 pip check、CLI help 與文件相對連結檢查。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from core.logger import setup_run
from core.runtime_environment import collect_environment
from tests.issue_delivery_evidence import local_links

ROOT = Path(__file__).resolve().parents[2]
CLIS = ("experiments.exp1_cold_start", "experiments.exp2_scale_growth",
        "experiments.exp3_trend", "experiments.exp4_polar_map",
        "experiments.compare_openset", "experiments.exp6_formal_benchmark", "web.server")
DOCUMENTS = ("TODO.md", "docs/README.md", "docs/health_and_reports.md",
             "docs/runtime_policy.md", "docs/ci_contract.md",
             "reports/Andy_20261009_主線交付/README.md",
             "reports/Andy_20261009_主線交付/execution_log.md",
             "docs/formal_materialization_contract.md")


def execute(arguments: list[str], timeout: int = 600, tool: str | None = None) -> dict:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    try:
        result = subprocess.run([tool or sys.executable, *arguments], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout)
        output = result.stdout + "\n" + result.stderr
        # 任意 exception payload、絕對路徑與 token 都不進公開 artifact。
        count = re.search(r"Ran (\d+) tests? in", output)
        failures = re.search(r"FAILED \(([^\r\n]+)\)", output)
        totals = {key: int(value) for key, value in
                  re.findall(r"(failures|errors|skipped)=(\d+)", failures.group(1) if failures else output[-1000:])}
        tests_run = int(count.group(1)) if count else None
        failed, errors, skipped = totals.get("failures", 0), totals.get("errors", 0), totals.get("skipped", 0)
        failure_ids = re.findall(r"^(?:FAIL|ERROR): ([\w]+) \(([\w.]+)\)", output, re.MULTILINE)
        if tests_run is None:
            # pytest 摘要列格式：「X passed, Y failed, Z error(s), W skipped in Ns」。
            pytest_counts = {label: int(value) for value, label in
                             re.findall(r"(\d+) (passed|failed|skipped|errors?)\b", output)}
            if pytest_counts:
                failed = pytest_counts.get("failed", 0)
                errors = pytest_counts.get("error", pytest_counts.get("errors", 0))
                skipped = pytest_counts.get("skipped", 0)
                tests_run = pytest_counts.get("passed", 0) + failed + errors + skipped
                failure_ids = re.findall(r"^(?:FAILED|ERROR) (\S+)", output, re.MULTILINE)
        return {"arguments": arguments, "returncode": result.returncode,
                "tests_run": tests_run, "failed": failed, "errors": errors, "skipped": skipped,
                "failure_ids": failure_ids,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "diagnostic_policy": "原始輸出只留行程記憶體；公開摘要不收錄任意例外文字"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"arguments": arguments, "returncode": -1, "error_type": type(exc).__name__}


def run() -> dict:
    logger, paths = setup_run("ci_evidence")
    environment = collect_environment(ROOT)
    checks = []
    # 即使正式 data 在開發機存在，本入口與 tests 也不將它當測試輸入。
    with tempfile.TemporaryDirectory(prefix="lineage_ci_") as directory:
        previous_tmp = {name: os.environ.get(name) for name in ("TMP", "TEMP", "TMPDIR")}
        try:
            for name in previous_tmp:
                os.environ[name] = directory
            # uv sync 不把 pip 裝進最終 venv，改用 uv pip check；傳相對路徑避免洩漏絕對路徑。
            # 不對 sys.executable 呼叫 resolve()：uv 建的 venv 裡 python 是指向共用安裝目錄的
            # symlink，resolve 後的路徑會跑到 ROOT 之外，uv pip check 反而認得這個相對路徑。
            try:
                python_rel = str(Path(sys.executable).relative_to(ROOT))
                checks.append(execute(["pip", "check", "--python", python_rel], tool="uv"))
            except ValueError:
                checks.append({"arguments": ["pip", "check"], "returncode": -1, "error_type": "PythonNotUnderRoot"})
            checks.extend(execute(["-m", module, "--help"], 60) for module in CLIS)
        finally:
            for name, old in previous_tmp.items():
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old
    links = [row for name in DOCUMENTS for row in local_links(ROOT / name, ROOT)]
    broken = [{"source": row["source"], "link": row["link"]} for row in links if not row["exists"]]
    result = {"schema_version": "ci_evidence_v1", "environment": environment,
              "checks": checks, "document_count": len(DOCUMENTS), "relative_targets": len(links),
              "broken_targets": broken, "ignored_data_required": False,
              "status": "PASS" if not broken and all(row["returncode"] == 0 for row in checks) else "FAILED",
              "scope": "工程契約；任意Origin現狀不代表安全，無新模型成績，macOS未驗證"}
    (paths.output_dir / "public_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("CI 工程驗證：{}；失效目標={}", result["status"], len(broken))
    for check in checks:
        if check["returncode"]:
            logger.error("失敗命令={}；測試ID={}", check["arguments"], check.get("failure_ids", []))
    return result


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    return 0 if run()["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
