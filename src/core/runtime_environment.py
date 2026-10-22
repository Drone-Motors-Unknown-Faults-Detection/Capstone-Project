"""保存執行環境版本；不收集 token、環境變數或使用者絕對路徑。"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys


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


def _program_root() -> Path | None:
    """只辨識實際載入的 src/core 程式樹，不借用呼叫端工作目錄。"""
    try:
        module = Path(__file__).resolve()
        if module.parent.name == "core" and module.parent.parent.name == "src":
            root = module.parents[2]
            if (root / "pyproject.toml").is_file():
                return root
    except OSError:
        pass
    return None


def blas_info() -> dict:
    """numpy 連結的 BLAS／LAPACK。同 seed 的數字會因 BLAS 不同而分岔（macOS Accelerate vs OpenBLAS，#224）。"""
    try:
        import numpy as np

        deps = np.show_config(mode="dicts")["Build Dependencies"]
        return {"blas": deps["blas"].get("name"), "blas_version": deps["blas"].get("version"),
                "lapack": deps["lapack"].get("name")}
    except Exception:  # 取不到時以 UNKNOWN 保留，不中斷執行
        return {"blas": None, "blas_version": None, "lapack": None, "status": "UNKNOWN"}


def _argv_data_root(argv: list[str]) -> str:
    """實驗 CLI 一律用 --data-root（預設 data）；從命令列取，免得逐支傳入。"""
    for i, arg in enumerate(argv):
        if arg == "--data-root" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--data-root="):
            return arg.split("=", 1)[1]
    return "data"


def data_info(data_root: str | Path) -> dict:
    """資料內容指紋（core.provenance 的 source_content 版本）；只記相對 source_id 數量，不記絕對路徑（#242）。

    呼叫端沒有傳入 data_root 時，才從命令列的 --data-root 推得（#249）。
    """
    try:
        from core.provenance import capture_source

        public, _ = capture_source(data_root)
        return {"content_version": public["schema_version"], "dataset_fingerprint": public["dataset_fingerprint"],
                "n_files": len(public["files"]), "status": "VERIFIED"}
    except Exception:  # 沒有資料目錄（例如單元測試）時保留 UNKNOWN，不中斷執行
        return {"content_version": None, "dataset_fingerprint": None, "n_files": 0, "status": "UNKNOWN"}


def collect_environment(root: Path | None = None, data_root: str | Path | None = None,
                        *, include_data: bool = True) -> dict:
    """回傳可公開的版本紀錄；Git 缺失以 UNKNOWN 保留。"""
    explicit = root is not None
    root = Path(root) if explicit else _program_root()
    top = _git(root, "rev-parse", "--show-toplevel") if root is not None else None
    git_reason = "GIT_UNAVAILABLE_OR_INCOMPLETE" if root is not None else "PROGRAM_ROOT_UNAVAILABLE"
    if not explicit and top and Path(top).resolve() != root:
        # 匯出程式樹可能放在另一個 repo 裡；外層版本不能替這份程式背書。
        top = None
        git_reason = "NOT_PROGRAM_CHECKOUT"
    head = _git(root, "rev-parse", "HEAD") if top else None
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no") if top else None
    constraint_root = Path(top) if explicit and top else root
    constraint_sha = None
    lock_reason = "LOCK_MISSING" if constraint_root is not None else "PROGRAM_ROOT_UNAVAILABLE"
    if constraint_root is not None:
        try:
            constraint_sha = hashlib.sha256((constraint_root / "uv.lock").read_bytes()).hexdigest()
        except FileNotFoundError:
            pass
        except OSError:
            lock_reason = "LOCK_UNREADABLE"
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    return {
        "schema_version": "runtime_environment_v1",
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
               "host": platform.node()},
        "numeric": blas_info(),
        "data": (data_info(data_root if data_root is not None else _argv_data_root(sys.argv))
                 if include_data else {"content_version": None, "dataset_fingerprint": None,
                                       "n_files": 0, "status": "UNKNOWN", "reason": "UNVERIFIED_MEMORY"}),
        "packages": dict(sorted(packages.items())),
        "git": {"head": head, "tracked_dirty": bool(dirty) if dirty is not None else None,
                "status": "VERIFIED" if head and dirty is not None else "UNKNOWN"},
        "constraints_sha256": constraint_sha,
        "program_source": {"schema_version": "program_location_v1",
                           "selection": "EXPLICIT_ROOT" if explicit else "LOADED_MODULE",
                           "git_reason": None if head and dirty is not None else git_reason,
                           "lock_status": "VERIFIED" if constraint_sha is not None else "UNKNOWN",
                           "lock_reason": None if constraint_sha is not None else lock_reason},
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
