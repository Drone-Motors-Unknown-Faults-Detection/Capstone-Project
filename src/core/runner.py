"""實驗批次執行的共用小工具：資料集選擇參數與 JSON 儲存。

（日誌與輸出目錄由 core.logger.setup_run 依 logs/ + output/ 慣例建立。）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.data import discover_datasets, load_pools
from core.openset import SUPPORTED_OPENSET_METHODS


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="data", help="資料根目錄（預設 data）")
    parser.add_argument("--motor", default="T1", help="馬達代碼（預設 T1）")
    parser.add_argument("--rpm", default="8000rpm", help="轉速目錄名（預設 8000rpm）")
    parser.add_argument("--seed", type=int, default=42)


def add_openset_args(parser: argparse.ArgumentParser) -> None:
    """加入所有實驗共用的 Open Set 方法與校準參數。

    ``--method`` 保留原行為，仍只控制 Mahalanobis 的 covariance estimator。
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


def resolve_dataset(args: argparse.Namespace) -> tuple[dict, dict]:
    """回傳 (dataset 資訊, pools)。rpm 可寫 8000 或 8000rpm。"""
    rpm = args.rpm if args.rpm.endswith("rpm") else f"{args.rpm}rpm"
    datasets = discover_datasets(args.data_root)
    for ds in datasets:
        if ds["motor"] == args.motor and ds["rpm"] == rpm:
            return ds, load_pools(ds["path"])
    available = ", ".join(f"{d['motor']}/{d['rpm']}" for d in datasets) or "（無）"
    raise SystemExit(f"找不到資料集 {args.motor}/{rpm}；可用：{available}")


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
