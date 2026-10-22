"""保存執行環境版本；不收集 token、環境變數或使用者絕對路徑。"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path
import platform
import subprocess


DEFAULT_DATA_ROOT = "data"


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


def git_head(root: Path) -> str | None:
    """HEAD 的 commit SHA；沒有 git、沒有 .git 或逾時都回 None，不中斷呼叫端。"""
    return _git(Path(root), "rev-parse", "HEAD")


def blas_info() -> dict:
    """numpy 連結的 BLAS／LAPACK。同 seed 的數字會因 BLAS 不同而分岔（macOS Accelerate vs OpenBLAS，#224）。"""
    try:
        import numpy as np

        deps = np.show_config(mode="dicts")["Build Dependencies"]
        return {"blas": deps["blas"].get("name"), "blas_version": deps["blas"].get("version"),
                "lapack": deps["lapack"].get("name")}
    except Exception:  # 取不到時以 UNKNOWN 保留，不中斷執行
        return {"blas": None, "blas_version": None, "lapack": None, "status": "UNKNOWN"}


def source_data_info(public: dict) -> dict:
    """把 core.provenance.capture_source 的公開紀錄縮成 environment.json 的 data 欄；不再重算雜湊。"""
    return {"content_version": public["schema_version"], "dataset_fingerprint": public["dataset_fingerprint"],
            "n_files": len(public["files"]), "status": "VERIFIED"}


def data_info(data_root: str | Path) -> dict:
    """資料內容指紋（core.provenance 的 source_content 版本）；只記相對 source_id 數量，不記絕對路徑（#242）。

    沒有資料目錄或沒有正式 CSV（例如單元測試）記 UNKNOWN；manifest 與內容不符等
    ProvenanceError、其他讀取錯誤記 FAILED 並留 error_type，兩種都不中斷執行（#269）。
    """
    from core.provenance import ProvenanceError, capture_source

    try:
        public, _ = capture_source(data_root)
        return source_data_info(public)
    except FileNotFoundError:
        return {"content_version": None, "dataset_fingerprint": None, "n_files": 0, "status": "UNKNOWN"}
    except (ProvenanceError, OSError, ValueError) as exc:
        return {"content_version": None, "dataset_fingerprint": None, "n_files": 0, "status": "FAILED",
                "error_type": type(exc).__name__}


def collect_environment(root: Path | None = None, data_root: str | Path | None = None, *,
                        data: dict | None = None) -> dict:
    """回傳可公開的版本紀錄；Git 缺失以 UNKNOWN 保留。

    ``data_root`` 由呼叫端（CLI 的 ``--data-root``、Web 的 ``args.data_root``）明確傳入，
    沒傳就記預設的 ``data``（不再從 sys.argv 推斷，#269）。``data`` 給定時直接採用
    已算好的 data_info（例如 exp6 正式版 run() 已 capture_source 過），不再雜湊一次。
    """
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
    if data is None:
        data = data_info(DEFAULT_DATA_ROOT if data_root is None else data_root)
    return {
        "schema_version": "runtime_environment_v1",
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
               "host": platform.node()},
        "numeric": blas_info(),
        "data": data,
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
