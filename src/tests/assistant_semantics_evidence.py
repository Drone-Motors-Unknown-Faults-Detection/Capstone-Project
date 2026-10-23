"""在 Aero／Spark 對既有共享模型做有界抽驗；原始回答只留受測主機。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from langchain_core.callbacks import BaseCallbackHandler

from core.logger import setup_run
from web.assistant import Assistant
from web.playback import PlaybackTracker

QUESTIONS = (
    "說明實驗一的健康冷啟動方法與資料切分。187／62／64是否適用所有工況？95%代表95%分類準確嗎？",
    "查實驗二這次保存結果：own_holdout_acc、eval_known_rate、cluster_purity分別代表什麼？能算未知召回或AUROC嗎？",
    "T1、T2、T3馬達型號、KV、額定電壓與使用時數為何？若沒有直接證據，請明說。",
    "實驗二這次外部接受率高，就代表正確辨識自身配置嗎？比較eval_known_rate、eval_matched_rate與final_eval，final的unknown_rate能當學會前未知召回嗎？",
    "說明實驗二HDBSCAN分群的一般原理、適用條件與限制。",
)


class ModelCalls(BaseCallbackHandler):
    """只記模型客戶端呼叫事件，不保存prompt或推算遠端GPU次數。"""
    def __init__(self):
        self.started = self.completed = self.errors = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.started += 1

    def on_llm_end(self, response, **kwargs):
        self.completed += 1

    def on_llm_error(self, error, **kwargs):
        self.errors += 1


class ProbeAssistant(Assistant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_entries = 0
        self.calls = ModelCalls()

    async def llm(self):
        self.path_entries += 1
        model = await super().llm()
        callbacks = list(model.callbacks or [])
        if self.calls not in callbacks:
            model.callbacks = [*callbacks, self.calls]
        return model


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
    source_sha = hashlib.sha256(raw).hexdigest()
    tracker.record_run("exp2", {"result": result, "params": payload.get("params", {}), "saved_sha256": source_sha})
    assistant = ProbeAssistant(base_url)
    receipt = {"schema_version": "assistant_semantics_evidence_v2", "source_sha256": source_sha,
               "source_mode": "既有JSON載入記憶體快照，不重新執行實驗。",
               "limit": {"questions": len(QUESTIONS), "seconds_per_question": 80}, "cases": [],
               "scope": "實驗頁紀錄／存檔；不讀首頁、不部署。COMPLETED只代表流程，語意需逐段審閱。",
               "model_audit_scope": "LangChain模型呼叫起始／結束事件；不等同遠端推理完成或GPU次數。",
               "semantic_status": "REQUIRES_REVIEW", "browser": "NOT_VERIFIED"}
    for question in QUESTIONS:
        frames = []
        before = (assistant.path_entries, assistant.calls.started, assistant.calls.completed, assistant.calls.errors)
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
            "seconds": round(time.monotonic() - started, 3), "frames": frames,
            "route": next((f["mode"] for f in frames if f.get("event") == "mode"),
                          "model" if assistant.path_entries > before[0] else "NOT_OBSERVED"),
            "model_path_entered": assistant.path_entries > before[0],
            "model_invocations_started": assistant.calls.started - before[1],
            "model_invocations_completed": assistant.calls.completed - before[2],
            "model_invocation_errors": assistant.calls.errors - before[3],
            "model_output_seen": any(f.get("event") == "delta" and f.get("origin") == "model" and f.get("text") for f in frames),
            "semantic_status": "REQUIRES_REVIEW"})
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
