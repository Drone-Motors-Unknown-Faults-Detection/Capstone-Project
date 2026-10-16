"""保存執行環境版本；不收集 token、環境變數或使用者絕對路徑。"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path
import platform
import subprocess


def lock_versions(root: Path) -> dict[str, str]:
    """讀取 uv.lock 鎖定的套件版本；只供測試／驗證入口呼叫，不在正式執行路徑上。"""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    data = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    return {pkg["name"].lower().replace("_", "-"): pkg["version"] for pkg in data.get("package", [])}


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_environment(root: Path | None = None) -> dict:
    """回傳可公開的版本紀錄；Git 缺失以 UNKNOWN 保留。"""
    root = Path.cwd() if root is None else Path(root)
    top = _git(root, "rev-parse", "--show-toplevel")
    head = _git(root, "rev-parse", "HEAD") if top else None
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no") if top else None
    constraint = Path(top) / "uv.lock" if top else root / "uv.lock"
    constraint_sha = hashlib.sha256(constraint.read_bytes()).hexdigest() if constraint.is_file() else None
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    return {
        "schema_version": "runtime_environment_v1",
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": dict(sorted(packages.items())),
        "git": {"head": head, "tracked_dirty": bool(dirty) if dirty is not None else None,
                "status": "VERIFIED" if head and dirty is not None else "UNKNOWN"},
        "constraints_sha256": constraint_sha,
        "scope": "呼叫 setup_run 的執行；版本紀錄不等於依賴相容或資料獨立性驗證",
    }


def run(program: str = "environment_install") -> dict:
    """初始化既有日誌慣例並保存當前版本，供安裝後驗證。"""
    from core.logger import setup_run
    log, paths = setup_run(program)
    result = json.loads((paths.output_dir / "environment.json").read_text(encoding="utf-8"))
    log.info("執行環境已保存：Python {}", result["python"]["version"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", default="environment_install", choices=("environment_install", "runtime_environment"))
    args = parser.parse_args()
    run(args.program)


if __name__ == "__main__":
    main()
