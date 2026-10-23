"""一般方法問答的來源隔離；合成資料與模型 stub 不代表真實模型驗收。"""

import asyncio
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.test_assistant import exp2_tracker, fake_model
from web.assistant import Assistant, load_knowledge, make_tools, method_only_question, to_messages
from web.playback import PlaybackTracker


def calls(*items):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"c{i}"} for i, (name, args) in enumerate(items)])


def ask(question, model, **kwargs):
    assistant = Assistant("http://unused/v1", llm=model)

    async def collect():
        return [frame async for frame in assistant.ask(**kwargs, question=question)]

    return asyncio.run(collect())


def evidence(model):
    return "\n".join(message.content for message in model.seen[-1] if isinstance(message, ToolMessage))


@pytest.mark.parametrize("question", [
    "HDBSCAN 的原理是什麼？", "群純度是否能保證未知召回？", "標準化是否保證消除跨馬達漂移？",
    "T1 馬達規格是什麼？", "實驗十二的原理", "實驗二在做什麼？",
])
def test_method_does_not_read_records_or_reuse_old_result_context(question):
    reader = Mock(side_effect=AssertionError("方法問題不應讀取存檔"))
    tracker = exp2_tracker()
    tracker.record_run("exp12", {"result": {"irrelevant": "UNRELATED_RESULT_9137"}})
    model = fake_model(calls(("get_run_status", {"number": "12"}), ("get_run_events", {})),
                       AIMessage(content="方法資訊不足時不補造數值"))
    frames = ask(question, model, snap=tracker.snapshot(), load_saved=reader,
                 history=[{"role": "assistant", "content": "OLD_ANSWER_8361"},
                          {"role": "user", "content": "請看先前實驗十二結果 USER_8123"}],
                 context={"mode": "result", "experiment": "exp12"})
    assert model.seen
    assert reader.call_count == 0
    context = "\n".join(str(message.content) for batch in model.seen for message in batch)
    for canary in ("UNRELATED_RESULT_9137", "OLD_ANSWER_8361", "USER_8123", "5screws"):
        assert canary not in context
    assert not any(frame["event"] in ("facts", "presentation") for frame in frames)
    assert all(not frame.get("sources") for frame in frames if frame.get("name") in (
        "get_run_status", "get_run_events"))


def test_method_lookup_excludes_historical_results_and_keeps_source():
    model = fake_model(calls(("lookup_knowledge", {"query": "HDBSCAN 自適應密度階梯"})),
                       AIMessage(content="依據本輪方法段落說明"))
    frames = ask("HDBSCAN 自適應密度階梯原理？", model, snap=None)
    text = evidence(model)
    assert "min_cluster_size" in text
    assert "25, 3" in text
    assert "summary.json" not in text
    assert "11-56-10" not in text
    assert "7／9" not in text
    assert "1430" not in text
    assert any("methods.md › HDBSCAN" in source for frame in frames for source in frame.get("sources", []))


def test_method_experiment_tool_does_not_open_handbook_results(tmp_path):
    (tmp_path / "exp2_example.md").write_text("# 合成手冊\n## 方法\nHDBSCAN\n## HDBSCAN 結果\nHANDBOOK_9127\n", encoding="utf-8")
    model = fake_model(calls(("get_experiment", {"number": "2"}),
                             ("lookup_knowledge", {"query": "HDBSCAN"}),
                             ("get_experiment", {"number": "12"})),
                       AIMessage(content="只解釋指定方法"))
    assistant = Assistant("http://unused/v1", llm=model, experiments_dir=tmp_path)

    async def collect():
        return [frame async for frame in assistant.ask(None, "實驗二的原理")]

    asyncio.run(collect())
    text = evidence(model)
    assert "HANDBOOK_9127" not in text
    assert "exp2_scale_growth" in text
    assert "exp12_confusion_tsne.py" not in text


@pytest.mark.parametrize("question,query,expected", [
    ("群純度是否保證未知召回？", "群純度", ("0～1", "未知召回", "物理故障原因")),
    ("標準化能保證消除跨馬達漂移嗎？", "標準化", ("RobustScaler", "訓練列", "不保證")),
    ("T1 馬達規格是什麼？", "T1 馬達規格", ("UNKNOWN", "規格尚未提供")),
])
def test_method_evidence_contains_limits_not_prediction_guarantees(question, query, expected):
    model = fake_model(calls(("lookup_knowledge", {"query": query})), AIMessage(content="只依工具來源回答"))
    ask(question, model, snap=None)
    text = evidence(model)
    for value in expected:
        assert value in text
    assert "187／62／64" not in text


def test_explicit_result_and_method_question_keeps_result_path():
    tracker = PlaybackTracker()
    tracker.record_run("exp1", {"result": {"synthetic_marker": "REQUESTED_RESULT_4173"}})
    model = fake_model(calls(("get_run_status", {"number": "1"})), AIMessage(content="依指定結果解釋方法"))
    ask("請解釋實驗一這次結果與方法", model, snap=tracker.snapshot())
    assert "REQUESTED_RESULT_4173" in evidence(model)


def test_method_does_not_weaken_invalid_context_validation():
    model = fake_model(AIMessage(content="不應呼叫"))
    frames = ask("HDBSCAN 原理", model, snap=None, context={"mode": 7})
    assert frames[0]["event"] == "error"
    assert not model.seen


def test_normal_history_interface_still_preserves_valid_messages():
    messages = to_messages([{"role": "assistant", "content": "已核對的前文"}], "目前狀況？")
    assert any(message.content == "已核對的前文" for message in messages)
    assert messages[-1].content == "目前狀況？"


def test_unmarked_knowledge_fails_closed_and_marker_is_not_shown(tmp_path):
    (tmp_path / "methods.md").write_text(
        "# 合成知識\n## 方法\n<!-- assistant: method -->\nSAFE_METHOD_4731\n"
        "## 未核對來源\nHISTORY_8732\n", encoding="utf-8")
    (tmp_path / "unreviewed.md").write_text(
        "# 未核對文件\n## 方法\n<!-- assistant: method -->\nUNREVIEWED_9917\n", encoding="utf-8")
    sections = load_knowledge(tmp_path)
    tools = {t.name: t for t in make_tools(None, sections, method_only=True)}
    text = tools["lookup_knowledge"].invoke({"query": "方法"})
    assert "SAFE_METHOD_4731" in text
    assert "HISTORY_8732" not in text
    assert "UNREVIEWED_9917" not in text
    assert "<!-- assistant:" not in text


@pytest.mark.parametrize("question", [
    "實驗一這次結果的原理", "HDBSCAN 存檔解釋", "說明 exp1_2026-10-23-12-00-00.json 的方法",
    "confidence 0.95 是 95% 正確機率嗎？",
])
def test_result_sources_and_usage_guard_are_not_classified_as_free_method(question):
    assert not method_only_question(question)


def test_method_mode_keeps_four_tools_and_only_requested_catalog():
    assert method_only_question("請查這個名詞", {"mode": "method"})
    tools = {t.name: t for t in make_tools(None, [], method_only=True, method_number=2)}
    assert set(tools) == {"get_run_status", "get_run_events", "get_experiment", "lookup_knowledge"}
    assert "查無資料" in tools["get_run_status"].invoke({"number": "2"})
    assert "查無資料" in tools["get_run_events"].invoke({"number": "2"})
    assert "exp2_scale_growth" in tools["get_experiment"].invoke({})


def test_pure_method_does_not_prefetch_latest_saved_record():
    reader = Mock(side_effect=AssertionError("方法問題不可預讀最新結果"))
    model = fake_model(calls(("lookup_knowledge", {"query": "群純度"})), AIMessage(content="方法定義與限制"))
    ask("群純度是否保證未知召回？", model, snap=None, load_saved=reader)
    reader.assert_not_called()
    assert "0～1" in evidence(model)
