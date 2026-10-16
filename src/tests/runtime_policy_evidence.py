"""保存乾淨安裝或 fixture 驗證的真實指令與回傳碼；不讀正式 data。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from core.logger import setup_run
from core.runtime_environment import lock_versions
from tests.integration_evidence import command

ROOT = Path(__file__).resolve().parents[2]


def _mismatches(packages: dict) -> list[dict]:
    """比對實際安裝版本與 uv.lock 鎖定版本；忽略 lock 沒有的名稱（如本機殘留套件）。"""
    lock = lock_versions(ROOT)
    found = []
    for name, actual in packages.items():
        if name in {"lineage", "drone-motors-unknown-faults-detection"}:
            continue
        expected = lock.get(name)
        if expected is not None and actual != expected:
            found.append({"package": name, "expected": expected, "actual": actual})
    return found


def run(mode: str, python: str, venv: str | None = None) -> dict:
    log, paths = setup_run("runtime_policy_evidence")
    results = []
    mismatches = []
    if mode == "install":
        if venv is None:
            raise ValueError("install 必須指定新的 --venv 路徑")
        if Path(venv).exists():
            raise ValueError("目標已存在；拒絕覆寫")
        results.append(command(["bash", "scripts/opt/build_uv.sh", "--python", python, "--venv", venv]))
        if results[-1]["returncode"] == 0:
            installed = str(Path(venv) / "bin/python")
            results.append(command(["uv", "pip", "check", "--python", installed]))
            versions = command([installed, "-c", "import json; from core.runtime_environment import collect_environment; print(json.dumps(collect_environment()))"])
            results.append(versions)
            if versions["returncode"] == 0:
                mismatches.extend(_mismatches(json.loads(versions["stdout"])["packages"]))
    else:
        empty_data = paths.output_dir / "empty_data"
        empty_data.mkdir()
        results.append(command([python, "-m", "unittest", "tests.test_runtime_environment", "-v"]))
        results.append(command([python, "-m", "tests.integration_evidence", "--phase", "candidate", "--data-root", str(empty_data)]))
        results.append(command([python, "-m", "tests.monitor_guard_evidence"]))
    environment = json.loads((paths.output_dir / "environment.json").read_text(encoding="utf-8"))
    if mode == "verify":
        # verify 必須由受測的新環境本身呼叫；不把外層 runner 的版本當作子行程版本。
        if Path(python).resolve() != Path(sys.executable).resolve():
            raise ValueError("verify 必須使用目前 Python executable")
        mismatches.extend(_mismatches(environment["packages"]))
    evidence = {"mode": mode, "scope": "工程驗證，不依賴 ignored data；不重新訓練研究模型",
                "environment": environment,
                "commands": results, "constraint_mismatches": mismatches,
                "passed": all(row["returncode"] == 0 for row in results) and not mismatches}
    (paths.output_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("{} 驗證結果：{}", mode, evidence["passed"])
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("install", "test", "verify"), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--venv")
    args = parser.parse_args()
    result = run(args.mode, args.python, args.venv)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
