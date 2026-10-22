"""在 Aero／Spark 對既有共享模型做三題有界抽驗；原始回答只留受測主機。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from core.logger import setup_run
from web.assistant import Assistant
from web.playback import PlaybackTracker

QUESTIONS = (
    "說明實驗一的健康冷啟動方法與資料切分。187／62／64是否適用所有工況？95%代表95%分類準確嗎？",
    "查實驗二這次保存結果：own_holdout_acc、eval_known_rate、cluster_purity分別代表什麼？能算未知召回或AUROC嗎？",
    "T1、T2、T3馬達型號、KV、額定電壓與使用時數為何？若沒有直接證據，請明說。",
)


def validate_url(value):
    if value.rstrip("/") not in {"http://127.0.0.1:6767/v1", "http://127.0.0.1:8000/v1"}:
        raise ValueError("只允許既有本機共享模型，不另啟模型或使用外部服務")


async def run(base_url, source_file):
    validate_url(base_url)
    log, paths = setup_run("assistant_semantics_evidence", unique=True)
    source = Path(source_file)
    raw = source.read_bytes()
    payload = json.loads(raw)
    result = payload.get("result", payload)
    if not isinstance(result, dict) or not isinstance(result.get("stages"), list):
        raise ValueError("必須指定既有實驗二結果，不建立假實驗資料")
    tracker = PlaybackTracker()
    tracker.record_run("exp2", {"result": result, "params": payload.get("params", {})})
    assistant = Assistant(base_url)
    receipt = {"schema_version": "assistant_semantics_evidence_v1", "source_sha256": hashlib.sha256(raw).hexdigest(),
               "limit": {"questions": 3, "seconds_per_question": 80}, "cases": [],
               "scope": "實驗頁紀錄／存檔；不讀首頁、不部署。COMPLETED只代表流程，語意需逐段審閱。",
               "semantic_status": "REQUIRES_REVIEW", "browser": "NOT_VERIFIED"}
    for question in QUESTIONS:
        frames = []
        started = time.monotonic()
        async def collect():
            async for frame in assistant.ask(tracker.snapshot(), question, []):
                frames.append(frame)
        try:
            await asyncio.wait_for(collect(), timeout=80)
            completed = any(f.get("event") == "tool" for f in frames) and any(
                f.get("event") == "delta" for f in frames) and not any(f.get("event") == "error" for f in frames)
            status = "COMPLETED" if completed else "INCOMPLETE"
        except Exception as exc:
            status = type(exc).__name__
        receipt["cases"].append({"question": question, "status": status,
            "seconds": round(time.monotonic() - started, 3), "frames": frames})
        (paths.output_dir / "summary.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.info("第{}題流程={}；語意仍待逐段審閱", len(receipt["cases"]), status)
    receipt["source_unchanged"] = source.read_bytes() == raw
    (paths.output_dir / "summary.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:6767/v1")
    parser.add_argument("--source-file", required=True)
    result = asyncio.run(run(**vars(parser.parse_args())))
    return 0 if result["source_unchanged"] and all(c["status"] == "COMPLETED" for c in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
