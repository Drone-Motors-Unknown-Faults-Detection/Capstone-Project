"""實驗八串流：CLI for calibrated window inference and optional session health streaming."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from core.data import HEALTHY, discover_datasets, make_split
from core.logger import setup_run
from core.provenance import load_pool_source
from core.runner import execution_context, finish_execution, run_dataset, save_json
from experiments.health.index import CalibratedHealthIndex
from experiments.health.trajectory import SessionTrajectoryMonitor


def _dataset(data_root: Path | str, motor: str, rpm: str) -> dict:
    rpm_name = str(rpm)
    if not rpm_name.endswith("rpm"):
        rpm_name = f"{rpm_name}rpm"
    matches = [
        row
        for row in discover_datasets(Path(data_root).expanduser().resolve())
        if row["motor"] == motor and row["rpm"] == rpm_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one dataset for {motor}/{rpm_name}, found {len(matches)}")
    return matches[0]


def run(
    data_root: Path | str,
    motor: str,
    rpm: str,
    config: str,
    *,
    seed: int = 42,
    openset_method: str = "mahalanobis",
    mahalanobis_method: str = "ledoit_wolf",
    knn_neighbors: int = 5,
    motor_id: str | None = None,
    session_id: str | None = None,
    max_windows: int | None = None,
    require_identity: bool = True,
) -> list[dict]:
    """批次介面：把 iter_run 跑完，回傳逐窗結果列表。"""
    return [
        event["window"]
        for event in iter_run(
            data_root, motor, rpm, config, seed=seed, openset_method=openset_method,
            mahalanobis_method=mahalanobis_method, knn_neighbors=knn_neighbors,
            motor_id=motor_id, session_id=session_id, max_windows=max_windows,
            require_identity=require_identity,
        )
        if event["event"] == "window"
    ]


def iter_run(
    data_root: Path | str,
    motor: str,
    rpm: str,
    config: str,
    *,
    seed: int = 42,
    openset_method: str = "mahalanobis",
    mahalanobis_method: str = "ledoit_wolf",
    knn_neighbors: int = 5,
    motor_id: str | None = None,
    session_id: str | None = None,
    max_windows: int | None = None,
    require_identity: bool = True,
):
    """逐窗產生事件，供 Web 實驗頁邊跑邊畫；計算與 run() 相同。

    事件依序為 fitted（健康指數模型擬合完成）與每個窗口一個 window（內容同
    run() 列表的一筆）。
    """
    dataset = _dataset(data_root, motor, rpm)
    pools, source_context = load_pool_source(data_root, dataset["path"])
    yield from _iter_pools(pools, motor, dataset["rpm"], config, seed=seed,
        openset_method=openset_method, mahalanobis_method=mahalanobis_method,
        knn_neighbors=knn_neighbors, motor_id=motor_id, session_id=session_id,
        max_windows=max_windows, require_identity=require_identity, source_context=source_context)


def _iter_pools(pools, motor, rpm, config, *, seed=42, openset_method="mahalanobis",
                mahalanobis_method="ledoit_wolf", knn_neighbors=5, motor_id=None,
                session_id=None, max_windows=None, require_identity=True, source_context=None):
    context = execution_context(pools, source_context, seed=seed, method=mahalanobis_method,
        openset_method=openset_method, knn_neighbors=knn_neighbors, motor=motor, rpm=rpm,
        configuration=config, max_windows=max_windows, require_identity=require_identity)
    if config not in pools:
        raise KeyError("指定配置不在已載入的資料池")
    rng = np.random.default_rng(seed)
    healthy = pools[HEALTHY]
    split = make_split(len(healthy), rng)
    model = CalibratedHealthIndex.fit(
        healthy[split.train],
        np.zeros(len(split.train), dtype=int),
        healthy[split.cal],
        np.zeros(len(split.cal), dtype=int),
        openset_method=openset_method,
        mahalanobis_method=mahalanobis_method,
        knn_neighbors=knn_neighbors,
    )
    monitor = SessionTrajectoryMonitor(model, require_identity=require_identity)
    metadata = model.metadata()
    context["health_calibration"] = metadata["calibration"]
    yield {"event": "fitted", "model": metadata, "provenance": context}
    # 8screws 取與擬合不重疊的 holdout；其他配置沒參與擬合，依原順序取整池
    indices = split.holdout if config == HEALTHY else np.arange(len(pools[config]))
    samples = pools[config][indices]
    if max_windows is not None:
        if max_windows < 1:
            raise ValueError("max_windows must be positive")
        samples = samples[:max_windows]
        indices = indices[:max_windows]
    condition = f"{motor}/{rpm}"
    for index, sample in enumerate(samples):
        result = monitor.update(
            sample,
            motor_id=motor_id,
            session_id=session_id,
            timestamp=None,
            condition=condition,
        )
        payload = result.to_dict("full")
        payload["window_index"] = int(index)
        payload["source_pool_index"] = int(indices[index])
        payload["provenance"] = context
        yield {"event": "window", "window": payload}
    finish_execution(context, pools, source_context)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--motor", required=True)
    parser.add_argument("--rpm", required=True)
    parser.add_argument("--config", default=HEALTHY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openset-method", default="mahalanobis")
    parser.add_argument("--method", dest="mahalanobis_method", default="ledoit_wolf")
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--motor-id")
    parser.add_argument("--session-id")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output-mode", choices=("binary", "health", "full"), default="full")
    parser.add_argument(
        "--allow-no-identity",
        action="store_true",
        help="allow output without temporal identity; trend remains insufficient_history when absent",
    )
    args = parser.parse_args(argv)
    log, paths = setup_run("exp8_health_monitor", unique=True, data_root=args.data_root)
    log.info("執行健康指數逐窗回放；來源絕對路徑僅留本機 sidecar")
    def collect(pools, *, source_context, **options):
        return [event["window"] for event in _iter_pools(pools, args.motor,
            args.rpm if args.rpm.endswith("rpm") else args.rpm + "rpm", args.config,
            source_context=source_context, **options) if event["event"] == "window"]
    _, results = run_dataset(args, paths, collect,
        seed=args.seed,
        openset_method=args.openset_method,
        mahalanobis_method=args.mahalanobis_method,
        knn_neighbors=args.knn_neighbors,
        motor_id=args.motor_id,
        session_id=args.session_id,
        max_windows=args.max_windows,
        require_identity=not args.allow_no_identity,
    )
    save_json(paths.output_dir / "metadata.json", {
        "provenance": results[0]["provenance"] if results else None,
        "configuration": args.config, "output_mode": args.output_mode,
        "source_pool_indices": [row["source_pool_index"] for row in results]})
    lines = []
    for result in results:
        # iter_run 已輸出 full 模式；此處依 --output-mode 篩欄，health 欄位須與
        # HealthMonitoringResult.to_dict("health") 一致（另加 window_index）。
        if args.output_mode == "binary":
            lines.append(json.dumps({"is_fault": result["is_fault"]}, ensure_ascii=False))
        elif args.output_mode == "health":
            lines.append(json.dumps({key: result[key] for key in (
                "is_fault", "health_index", "degradation_score", "severity_stage", "trend",
                "degradation_rate", "openset_method", "openset_score", "is_unknown_fault",
                "prediction_confidence", "uncertainty", "timestamp", "condition", "data_quality",
                "alarm_state", "alarm_reason", "severity_basis", "raw_health_index",
                "smoothed_health_index", "change_point_state", "window_index",
            )}, ensure_ascii=False))
        else:
            lines.append(json.dumps(result, ensure_ascii=False))
    (paths.output_dir / "windows.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    for line in lines:
        print(line)
    log.info(f"完成：{len(lines)} 窗，結果寫入 {paths.output_dir / 'windows.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
