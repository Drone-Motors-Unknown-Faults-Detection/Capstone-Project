"""執行期日誌與輸出目錄——沿用論文版專案的 logs/ + output/ 慣例，改以 loguru 實作。

每次執行自動建立（以程式名與時間戳隔離）：
    logs/{program}/{YYYY-MM-DD-HH-MM-SS}.log     ← loguru 檔案日誌（同步輸出到終端）
    output/{program}/{YYYY-MM-DD-HH-MM-SS}/      ← 結果檔與圖表（save_plot 自動編號）

論文版專案因 Jupyter 的遞迴問題自製 SimpleFileLogger；本專案全部是純 Python 程式，
可直接使用 loguru。
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from core.runtime_environment import collect_environment


@dataclass
class RunPaths:
    program: str
    timestamp: str
    log_file: Path
    output_dir: Path | None
    _plot_index: int = field(default=0, repr=False)


def setup_run(program: str, make_output: bool = True, *, unique: bool = False):
    """初始化本次執行的 loguru logger 與輸出目錄，回傳 (logger, RunPaths)。"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path("logs") / program
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{timestamp}.log"

    output_dir: Path | None = None
    if unique:
        base_timestamp = timestamp
        index = 0
        while True:
            timestamp = base_timestamp if index == 0 else f"{base_timestamp}_{index:03d}"
            log_file = log_dir / f"{timestamp}.log"
            try:
                with log_file.open("x", encoding="utf-8"):
                    pass
            except FileExistsError:
                index += 1
                continue
            if make_output:
                output_dir = Path("output") / program / timestamp
                try:
                    output_dir.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    log_file.unlink()  # 只清除本次exclusive保留的空檔。
                    index += 1
                    continue
            break
    elif make_output:
        output_dir = Path("output") / program / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
        encoding="utf-8",
    )

    paths = RunPaths(program, timestamp, log_file, output_dir)
    logger.info(f"Logger initialized: {program}")
    logger.info(f"log_file={log_file}")
    if output_dir is not None:
        logger.info(f"output_dir={output_dir}")
    environment = collect_environment()
    if output_dir is not None:
        (output_dir / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        logger.info("執行環境：{}", json.dumps(environment, ensure_ascii=False, sort_keys=True))
    return logger, paths


def save_plot(fig, paths: RunPaths, name: str | None = None) -> Path:
    """儲存 matplotlib figure 至本次執行的 output 目錄（未命名時自動編號）。"""
    if paths.output_dir is None:
        raise RuntimeError("此執行以 make_output=False 初始化，沒有 output 目錄")
    filename = name if name else f"plot_{paths._plot_index:03d}.png"
    paths._plot_index += 1
    target = paths.output_dir / filename
    fig.savefig(target, dpi=150, bbox_inches="tight")
    logger.info(f"圖表已儲存：{target}")
    return target
