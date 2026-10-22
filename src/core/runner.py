"""實驗批次執行的共用小工具：資料集選擇參數、Open Set CLI 參數（add_openset_args）與 JSON 儲存。

（日誌與輸出目錄由 core.logger.setup_run 依 logs/ + output/ 慣例建立。）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.data import discover_datasets, load_pools
from core.openset import SUPPORTED_OPENSET_METHODS, resolve_openset_config
from core.provenance import capture_source, load_pool_source, ProvenanceError, source_for_pools
from core.runtime_environment import collect_environment


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="data", help="資料根目錄（預設 data）")
    parser.add_argument("--motor", default="T1", help="馬達代碼（預設 T1）")
    parser.add_argument("--rpm", default="8000rpm", help="轉速目錄名（預設 8000rpm）")
    parser.add_argument("--seed", type=int, default=42)


def add_openset_args(parser: argparse.ArgumentParser) -> None:
    """加入所有實驗共用的 Open Set 方法與校準參數。

    ``--method`` 只控制 Mahalanobis 的 covariance estimator；openset_method=knn 時忽略。
    """
    parser.add_argument(
        "--openset-method",
        choices=SUPPORTED_OPENSET_METHODS,
        default="mahalanobis",
        help="Open Set detector 家族（預設 mahalanobis）",
    )
    parser.add_argument(
        "--method",
        choices=("legacy", "ledoit_wolf", "oas", "mcd"),
        default="ledoit_wolf",
        help="Mahalanobis covariance estimator；使用 knn 時忽略",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=5,
        help="k-NN detector 的鄰居數（預設 5）",
    )


def _dataset_info(args: argparse.Namespace) -> dict:
    rpm = args.rpm if args.rpm.endswith("rpm") else f"{args.rpm}rpm"
    datasets = discover_datasets(args.data_root)
    for ds in datasets:
        if ds["motor"] == args.motor and ds["rpm"] == rpm:
            return ds
    available = ", ".join(f"{d['motor']}/{d['rpm']}" for d in datasets) or "（無）"
    raise SystemExit(f"找不到資料集 {args.motor}/{rpm}；可用：{available}")


def resolve_dataset(args: argparse.Namespace) -> tuple[dict, dict]:
    """回傳 (dataset 資訊, pools)。rpm 可寫 8000 或 8000rpm。"""
    ds = _dataset_info(args)
    return ds, load_pools(ds["path"])


def execution_context(pools, source_context=None, *, seed=42, confidence=0.95,
                      method="ledoit_wolf", openset_method="mahalanobis", knn_neighbors=5,
                      **parameters) -> dict:
    """共用來源／環境與既有設定解析，不參與實驗數值計算。"""
    return {"schema_version": "experiment_provenance_v1",
            "source": source_for_pools(pools, source_context),
            "environment": collect_environment(),
            "resolved_config": resolve_openset_config(openset_method, confidence=confidence,
                mahalanobis_method=method, knn_neighbors=knn_neighbors),
            "parameters": {"seed": seed, **parameters}}


def run_dataset(args, paths, callback, **parameters):
    """CLI 已初始化 run 後才讀檔；任一步失敗保存去敏狀態。"""
    target = paths.output_dir / "source.json"
    source = {"pool_source_schema": "pool_source_v1", "status": "LOADING",
              "raw_session_independence": "UNKNOWN"}
    save_json(target, source)
    try:
        ds = _dataset_info(args)
        pools, context = load_pool_source(args.data_root, ds["path"])
        source = source_for_pools(pools, context)
        save_json(target, {**source, "status": "RUNNING"})
        result = callback(pools, source_context=context, **parameters)
        source = source_for_pools(pools, context)
    except (Exception, SystemExit) as exc:
        save_json(target, {**source, "status": "FAILED", "error_type": type(exc).__name__})
        raise
    save_json(target, source)
    return ds, result


def finish_execution(context, pools, source_context=None):
    """回傳結果前再驗來源；保留執行開始時的環境快照。"""
    return {**context, "source": source_for_pools(pools, source_context)}


class RootSources:
    """同一次跨工況執行的內容版本；不參與切分或消耗模型 RNG。"""

    def __init__(self, data_root, *, pool_cache=None):
        self.root = Path(data_root).resolve()
        self.initial, _ = capture_source(self.root)
        self.loaded = {}
        self.pool_cache = pool_cache

    def load(self, dataset):
        key = f"{dataset['motor']}/{dataset['rpm'].replace('rpm', '')}"
        if key in self.loaded:
            raise ProvenanceError("跨工況來源鍵重複")
        pools, context = load_pool_source(self.root, dataset["path"], cache=self.pool_cache)
        if context.public["data_version"] != self.initial["data_version"]:
            raise ProvenanceError("跨工況載入混用資料版本")
        self.loaded[key] = (pools, context)
        return pools

    def load_all(self):
        return {f"{ds['motor']}/{ds['rpm'].replace('rpm', '')}": self.load(ds)
                for ds in discover_datasets(self.root)}

    def finish(self):
        current, _ = capture_source(self.root)
        if current != self.initial:
            raise ProvenanceError("跨工況執行期間來源變動")
        if not self.loaded:
            raise ProvenanceError("沒有實際載入的工況，不能宣稱來源已驗證")
        return {**current, "status": "VERIFIED",
                "datasets": {key: source_for_pools(pools, context)
                             for key, (pools, context) in self.loaded.items()}}


def run_root_sources(callback, data_root, *, output_dir=None, pool_cache=None, **parameters):
    """root 入口共用前後核對；API 不另建 run，CLI 在 setup_run 後傳輸出目錄。"""
    target = Path(output_dir) / "source.json" if output_dir is not None else None
    source = {"status": "LOADING", "data_version": "UNKNOWN",
              "raw_session_independence": "UNKNOWN", "fresh_test": "UNKNOWN"}
    environment = collect_environment()
    if target is not None:
        save_json(target, source)
    try:
        tracker = RootSources(data_root, pool_cache=pool_cache)
        result = callback(data_root, source_run=tracker, **parameters)
        source = tracker.finish()
        result["provenance"] = {"schema_version": "experiment_provenance_v1",
                                "source": source, "environment": environment,
                                "parameters": parameters,
                                "implementation": callback.__module__}
    except (Exception, SystemExit) as exc:
        if target is not None:
            save_json(target, {**source, "status": "FAILED", "error_type": type(exc).__name__})
        raise
    if target is not None:
        save_json(target, source)
    return result


def save_json(path: Path, obj) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"無法序列化 {type(o)}")

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
