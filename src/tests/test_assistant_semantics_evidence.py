"""抽驗器只確認資料來源及有界流程，不能自動宣稱語意正確。"""
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.assistant_semantics_evidence import QUESTIONS, ModelCalls, ProbeAssistant, run, validate_url


@pytest.mark.parametrize("url", ["http://example.com/v1", "http://127.0.0.1:8600", "http://127.0.0.1:6767/v1/extra"])
def test_probe_rejects_other_targets(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_probe_preserves_source_and_does_not_claim_semantic_pass(tmp_path):
    source = tmp_path / "summary.json"
    raw = json.dumps({"stages": [], "seed": 42}).encode()
    source.write_bytes(raw)
    async def ask(self, *args):
        yield {"event": "tool", "name": "get_run_status", "sources": []}
        yield {"event": "delta", "text": "測試回答，不代表語意正確"}
    with patch("web.assistant.Assistant.ask", ask):
        result = asyncio.run(run("http://127.0.0.1:6767/v1", source))
    assert 3 <= len(QUESTIONS) <= 6
    assert len(result["cases"]) == len(QUESTIONS) and result["source_unchanged"]
    assert result["semantic_status"] == "REQUIRES_REVIEW"
    assert all(c["semantic_status"] == "REQUIRES_REVIEW" and c["route"] == "NOT_OBSERVED"
               and not c["model_path_entered"] and not c["model_output_seen"]
               and c["model_invocations_started"] == 0 for c in result["cases"])
    assert source.read_bytes() == raw
    assert next(Path("output").glob("assistant_semantics_evidence/*/summary.json")).is_file()


def test_probe_tracks_model_callbacks_without_counting_path_entry_as_inference():
    from langchain_core.messages import AIMessage
    from tests.test_assistant import fake_model
    calls = ModelCalls()
    calls.on_chat_model_start({}, [])
    calls.on_llm_end(None)
    calls.on_llm_error(RuntimeError("測試"))
    assert (calls.started, calls.completed, calls.errors) == (1, 1, 1)
    model = fake_model(AIMessage(content="測試模型文字"))
    assistant = ProbeAssistant("unused", llm=model)
    async def collect():
        await assistant.llm()
        assert assistant.path_entries == 1 and assistant.calls.started == 0
        return [f async for f in assistant.ask(None, "一般方法題")]
    frames = asyncio.run(collect())
    assert model.callbacks == [assistant.calls]
    assert assistant.path_entries == 2
    assert assistant.calls.started >= 1
    assert assistant.calls.completed == assistant.calls.started
    assert any(f.get("origin") == "model" for f in frames)


def test_probe_records_protected_routes_without_claiming_model_call(tmp_path):
    source = tmp_path / "summary.json"
    source.write_text(json.dumps({"stages": []}), encoding="utf-8")
    with patch("tests.assistant_semantics_evidence.QUESTIONS", QUESTIONS[:2]), patch(
            "web.assistant.Assistant.llm", side_effect=AssertionError("保護題不可呼叫模型")):
        result = asyncio.run(run("http://127.0.0.1:6767/v1", source))
    assert [c["route"] for c in result["cases"]] == ["data_usage", "exp2_metrics"]
    assert all(not c["model_path_entered"] and c["model_invocations_started"] == 0 for c in result["cases"])
    assert result["semantic_status"] == "REQUIRES_REVIEW"
