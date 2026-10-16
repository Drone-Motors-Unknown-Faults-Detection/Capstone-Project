"""資料池載入、切分與串流抽樣。

本專案只讀取論文版（Ancestor）管線已產出的 105 維 clean 特徵檔：
    data/Step-*/myfeature/{Motor}/{RPM}/{Screws}/*_Group_feature_data_clean.csv
（clean = 逐列 IQR scale=1.5 過濾後的版本，與論文 Step 4/5 的輸入相同），
不依賴論文版程式碼。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core.feature_schema import (FEATURE_DIM as FEATURE_DIM, REGISTRY_VERSION, CONFIGURATIONS, CONFIG_ALIASES,
                                 FeatureSchemaError, read_feature_csv, validate_pool_coverage)

HEALTHY = "8screws"

# 展示與報表排序：健康 → 鬆動由輕到重 → 複合配置
CONFIG_ORDER = [
    "8screws", "7screws", "6screws", "5screws", "4screws",
    "3screws", "2screws", "1screws", "1screw", "3_14screws", "4_146screws",
]

DISPLAY_NAMES = {
    "8screws": "Healthy（8 螺絲全鎖）",
    "7screws": "7 螺絲（最輕微鬆動）",
    "6screws": "6 螺絲（輕微鬆動）",
    "5screws": "5 螺絲（中度鬆動）",
    "4screws": "4 螺絲（明顯鬆動）",
    "3screws": "3 螺絲（嚴重鬆動）",
    "2screws": "2 螺絲（很嚴重鬆動）",
    "1screws": "1 螺絲（最嚴重鬆動）",
    "1screw": "1 螺絲（最嚴重鬆動）",
    "3_14screws": "3_14（複合不均勻鬆動）",
    "4_146screws": "4_146（複合不均勻鬆動）",
}


def config_sort_key(name: str) -> tuple[int, str]:
    try:
        return (CONFIG_ORDER.index(name), name)
    except ValueError:
        return (len(CONFIG_ORDER), name)


def display_name(config: str | None) -> str:
    if config is None:
        return "未知"
    return DISPLAY_NAMES.get(config, config)


def discover_datasets(data_root: Path | str) -> list[dict]:
    """列出可用的 (motor, rpm) 組合與其 myfeature 目錄。"""
    found: list[dict] = []
    for step_dir in sorted(Path(data_root).glob("Step-*")):
        myfeature = step_dir / "myfeature"
        if not myfeature.is_dir():
            continue
        for motor_dir in sorted(p for p in myfeature.iterdir() if p.is_dir()):
            for rpm_dir in sorted(p for p in motor_dir.iterdir() if p.is_dir()):
                if any(rpm_dir.glob("*/*_Group_feature_data_clean.csv")):
                    found.append({"motor": motor_dir.name, "rpm": rpm_dir.name, "path": rpm_dir})
    return found


def load_pools(dataset_path: Path | str, *, require_complete: bool = False,
               audit: dict | None = None) -> dict[str, np.ndarray]:
    """只接受登錄的ordered105欄；不刪列，coverage按用途另驗。"""
    report = audit if audit is not None else {}
    report.clear()
    report.update(schema_version=REGISTRY_VERSION, status="RUNNING", files=[], rejected_rows=0,
                  discarded_rows=0, raw_session_independence="UNKNOWN")
    try:
        pools = _load_pools(Path(dataset_path), report)
        report["coverage"] = validate_pool_coverage(pools, require_complete=require_complete)
    except FeatureSchemaError as exc:
        report.update(status="REJECTED", rejection=exc.audit, rejected_rows=exc.audit["rejected_rows"])
        raise
    except OSError as exc:
        rejection = FeatureSchemaError("unreadable_dataset", detail=type(exc).__name__)
        report.update(status="REJECTED", rejection=rejection.audit, rejected_rows=0)
        raise rejection from exc
    report["status"] = "VERIFIED"
    report["accepted_rows"] = sum(len(values) for values in pools.values())
    return pools


def _load_pools(dataset_path: Path, report: dict) -> dict[str, np.ndarray]:
    pools: dict[str, np.ndarray] = {}
    if not dataset_path.is_dir():
        raise FeatureSchemaError("missing_dataset")
    config_dirs = sorted(
        (p for p in dataset_path.iterdir() if p.is_dir()),
        key=lambda p: config_sort_key(p.name),
    )
    for config_dir in config_dirs:
        files = sorted(config_dir.glob("*_Group_feature_data_clean.csv"))
        canonical = CONFIG_ALIASES.get(config_dir.name, config_dir.name)
        if canonical not in CONFIGURATIONS:
            if files:
                raise FeatureSchemaError("unknown_configuration", config_dir.name)
            continue
        if not files:
            raise FeatureSchemaError("empty_configuration", config_dir.name)
        matrices = []
        for path in files:
            values, record = read_feature_csv(path, path.relative_to(dataset_path).as_posix())
            report["files"].append(record)
            matrices.append(values)
        pools[config_dir.name] = np.concatenate(matrices, axis=0)
    return pools


def run(dataset_path: Path | str, *, require_complete: bool = False) -> dict:
    """唯讀格式稽核入口；結果只含相對file ID與品質摘要。"""
    import json
    from core.logger import setup_run

    log, paths = setup_run("formal_schema_audit")
    audit = {}
    try:
        load_pools(dataset_path, require_complete=require_complete, audit=audit)
    except FeatureSchemaError as exc:
        log.error("輸入拒絕：{}", exc)
    (paths.output_dir / "schema_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("格式稽核={}；刪除列數=0；raw/session=UNKNOWN", audit["status"])
    return audit


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="唯讀核對正式105維欄位、數值與配置coverage")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    return 0 if run(args.dataset, require_complete=args.require_complete)["status"] == "VERIFIED" else 1


@dataclass(frozen=True)
class Split:
    """單一配置資料池的索引切分：train/cal 供擬合與校準，holdout 供串流與評估。"""

    train: np.ndarray
    cal: np.ndarray
    holdout: np.ndarray


def make_split(n: int, rng: np.random.Generator, ratios: tuple[float, float] = (0.6, 0.2)) -> Split:
    order = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_cal = max(1, int(n * ratios[1]))
    return Split(order[:n_train], order[n_train:n_train + n_cal], order[n_train + n_cal:])


class CycleSampler:
    """在指定索引上循環抽樣；每輪重新洗牌，抽完自動續圈。"""

    def __init__(self, pool: np.ndarray, indices: np.ndarray, rng: np.random.Generator):
        if len(indices) == 0:
            raise ValueError("CycleSampler 需要至少一筆樣本")
        self.pool = pool
        self.indices = np.asarray(indices)
        self.rng = rng
        self._order: list[int] = []

    def draw(self) -> np.ndarray:
        if not self._order:
            self._order = list(self.rng.permutation(self.indices))
        return self.pool[self._order.pop()]


if __name__ == "__main__":
    raise SystemExit(main())
