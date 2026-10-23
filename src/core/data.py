"""資料池載入、切分與串流抽樣。

本專案只讀取論文版（原始論文）管線已產出的 105 維 clean 特徵檔：
    data/Step-*/myfeature/{Motor}/{RPM}/{Screws}/*_Group_feature_data_clean.csv
（clean = 逐列 IQR scale=1.5 過濾後的版本，與論文 Step 4/5 的輸入相同），
不依賴論文版程式碼。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core.feature_schema import (FEATURE_DIM as FEATURE_DIM, REGISTRY_VERSION, CONFIGURATIONS, CONFIG_ALIASES,
                                 FeatureSchemaError, read_feature_csv, validate_pool_coverage)
from core.provenance import sha256_bytes, sha256_file

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


def dataset_key(motor: str, rpm: str) -> str:
    """實驗輸出的工況鍵：`T1/8000`（目錄名 `8000rpm` 去掉 rpm 後綴）。

    exp5／exp6／exp8／exp10／exp11／exp12 的 `dataset` 欄與彙整都用這個格式；
    web 實驗頁的資料集參數是 `T1/8000rpm`（含後綴），另有定義（#276 K02）。
    """
    return f"{motor}/{rpm.replace('rpm', '')}"


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


class PoolCache:
    """以內容 SHA 為鍵的 load_pools() 記憶體快取；規範見 docs/profiling_contract.md。

    範圍由呼叫端決定（例如一次 run_matrix）。命中時回傳深拷貝；載入失敗不快取。
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[dict[str, np.ndarray], dict]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def content_key(dataset_path: Path | str, *, require_complete: bool = False,
                    file_hashes: dict[str, str] | None = None) -> str:
        """內容鍵。``file_hashes`` 是 load_pools audit 已算好的 {相對 posix 路徑: sha256}，
        有列的檔案直接取用，不再讀第二次；目錄結構仍重新掃描。"""
        root = Path(dataset_path)
        known = file_hashes or {}
        listing = []
        if root.is_dir():
            for folder in sorted(p for p in root.iterdir() if p.is_dir()):
                files = [[path.name, known.get(path.relative_to(root).as_posix()) or sha256_file(path)]
                         for path in sorted(folder.glob("*_Group_feature_data_clean.csv"))]
                listing.append([folder.name, files])
        payload = {"registry": REGISTRY_VERSION, "require_complete": bool(require_complete),
                   "is_dir": root.is_dir(), "folders": listing}
        return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())

    def load(self, dataset_path: Path | str, *, require_complete: bool = False,
             audit: dict | None = None) -> dict[str, np.ndarray]:
        key = self.content_key(dataset_path, require_complete=require_complete)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            pools, report = cached
            if audit is not None:
                audit.clear()
                audit.update(copy.deepcopy(report))
            return {name: values.copy() for name, values in pools.items()}
        self.misses += 1
        report: dict = {}
        pools = load_pools(dataset_path, require_complete=require_complete, audit=report)
        # 載入後重算鍵時沿用 read_feature_csv 對同一批 bytes 算出的 SHA：鍵對應的就是存進快取的內容，
        # 讀檔途中被改掉的檔案會讓兩次鍵不同而不存（docs/profiling_contract.md）。
        loaded = {record["file_id"]: record["sha256"] for record in report.get("files", [])}
        if self.content_key(dataset_path, require_complete=require_complete, file_hashes=loaded) == key:
            self._entries[key] = ({name: values.copy() for name, values in pools.items()}, copy.deepcopy(report))
        if audit is not None:
            audit.clear()
            audit.update(report)
        return pools


def _load_pools(dataset_path: Path, report: dict) -> dict[str, np.ndarray]:
    """池的 key 一律是正規配置名（`1screw` 目錄讀成 `1screws`，CONFIG_ALIASES）；
    原目錄名留在 audit 的 ``configuration_dirs`` 與每個檔案的 file_id，實驗端不再各自處理別名（#271）。"""
    pools: dict[str, np.ndarray] = {}
    report["configuration_dirs"] = {}
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
        if canonical in pools:
            raise FeatureSchemaError("ambiguous_configuration_alias", config_dir.name,
                                     detail=f"{report['configuration_dirs'][canonical]} 與 {config_dir.name} 同時存在")
        matrices = []
        for path in files:
            values, record = read_feature_csv(path, path.relative_to(dataset_path).as_posix())
            report["files"].append(record)
            matrices.append(values)
        pools[canonical] = np.concatenate(matrices, axis=0)
        report["configuration_dirs"][canonical] = config_dir.name
    return pools


def dataset_data_root(dataset_path: Path | str) -> Path:
    """RPM 目錄（data/Step-*/myfeature/T1/8000rpm）所屬的資料根；不在 Step-* 底下就回傳目錄本身。"""
    path = Path(dataset_path).expanduser().resolve()
    return next((p.parent for p in (path, *path.parents) if p.name.startswith("Step-")), path)


def run(dataset_path: Path | str, *, require_complete: bool = False) -> dict:
    """唯讀格式稽核入口；結果只含相對file ID與品質摘要。"""
    import json
    from core.logger import setup_run

    # 環境指紋記這個 RPM 目錄所屬的資料根（Step-* 的上一層），不是預設的 data（#301 F-05）
    log, paths = setup_run("formal_schema_audit", data_root=dataset_data_root(dataset_path))
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
        self.last_index: int | None = None  # 最近一次抽到的池列號，供串流端記錄列來源

    def draw(self) -> np.ndarray:
        if not self._order:
            self._order = list(self.rng.permutation(self.indices))
        self.last_index = int(self._order.pop())
        return self.pool[self.last_index]


if __name__ == "__main__":
    raise SystemExit(main())
