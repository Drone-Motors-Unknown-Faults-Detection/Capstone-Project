"""工具圖的有限終止與降級；不需要線上模型。"""

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tests.test_assistant import exp2_tracker, fake_model
from web.assistant import Assistant, build_graph, experiment_number, make_tools


def call(query, index=0, name="lookup_knowledge"):
    return AIMessage(content="不應顯示的工具前言", tool_calls=[
        {"name": name, "args": {"number": query} if name.startswith("get_") else {"query": query}, "id": f"c{index}"}])


def graph_result(model, *, limits=None):
    audit = {}
    tools = make_tools(exp2_tracker().snapshot(), [])
    graph = build_graph(model, tools, audit=audit, limits=limits)
    result = asyncio.run(graph.ainvoke({"messages": []}, {"recursion_limit": 10}))
    return result["messages"][-1], audit


def test_duplicate_normalized_tool_arguments_reuse_evidence_and_force_finalizer():
    model = fake_model(call("實驗二", name="get_run_status"), call("二", 1, "get_run_status"),
                       call("2", 2, "get_run_status"))
    answer, audit = graph_result(model)
    assert len(audit["evidence"]) == 1
    assert audit["models"] == 3
    assert model.choices == ["required", "auto", "none"]
    assert audit["degraded"]
    assert "查詢已達上限" in answer.content
    assert "播放中" in answer.content
    assert "尚未完成" in answer.content


def test_different_queries_cannot_escape_round_budget():
    model = fake_model(*(call(f"不同查詢{i}", i) for i in range(4)))
    answer, audit = graph_result(model)
    assert audit["rounds"] == 3
    assert audit["models"] == 4
    assert model.choices[-1] == "none"
    assert "未完整完成" in answer.content


def test_normal_finite_query_still_answers_without_degradation():
    answer, audit = graph_result(fake_model(call("二", name="get_run_status"), AIMessage(content="已查到播放狀態")))
    assert answer.content == "已查到播放狀態"
    assert not audit["degraded"]


def test_batch_larger_than_tool_budget_is_not_executed():
    reply = AIMessage(content="", tool_calls=[{"name": "get_run_status", "args": {}, "id": str(i)} for i in range(7)])
    answer, audit = graph_result(fake_model(reply))
    assert audit["tools"] == 0
    assert "超出預算" in answer.content


class SlowModel:
    def __init__(self, *, cancel=False):
        self.cancel = cancel
        self.started = asyncio.Event()
        self.stopped = False

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, config):
        self.started.set()
        try:
            await asyncio.sleep(10)
        finally:
            self.stopped = True


def test_model_timeout_is_bounded_and_not_reported_as_experiment_failure():
    answer, audit = graph_result(SlowModel(), limits={"call_seconds": 0.01})
    assert audit["models"] == 1
    assert "模型回覆逾時" in answer.content
    assert "並非實驗失敗" in answer.content


def test_near_deadline_reserves_finalizer_instead_of_forcing_tool():
    model = fake_model(call("二", name="get_run_status"))
    answer, audit = graph_result(model, limits={"seconds": 0.1, "final_seconds": 0.2})
    assert model.choices == ["none"]
    assert audit["tools"] == 0
    assert "最後整理仍要求工具" in answer.content


def test_evidence_character_budget_preserves_whole_facts_and_source():
    @tool(response_format="content_and_artifact")
    def long_source(query: str):
        """提供預算邊界的測試資料。"""
        return "x" * 17000, {"sources": ["固定來源"], "facts": "判對2736／支持2938，非純故障。"}

    model = fake_model(call("資料", name="long_source"), call("資料", 1, "long_source"))
    audit = {}
    graph = build_graph(model, [long_source], audit=audit)
    result = asyncio.run(graph.ainvoke({"messages": []}, {"recursion_limit": 10}))
    text = result["messages"][-1].content
    assert "2736／支持2938" in text
    assert audit["chars"] < 16000
    assert audit["evidence"][0].artifact["sources"] == ["固定來源"]


def test_finalizer_timeout_preserves_prior_evidence():
    model = fake_model(call("二", name="get_run_status"), call("2", 1, "get_run_status"))
    original = model.bind_tools

    def bind(tools, **kwargs):
        return SlowModel() if kwargs.get("tool_choice") == "none" else original(tools, **kwargs)

    with patch.object(type(model), "bind_tools", lambda self, tools, **kwargs: bind(tools, **kwargs)):
        answer, audit = graph_result(model, limits={"call_seconds": 0.02})
    assert "模型回覆逾時" in answer.content
    assert "播放中" in answer.content
    assert audit["evidence"]


def test_recursion_error_becomes_bounded_verified_fallback_with_sources():
    @tool(response_format="content_and_artifact")
    def lookup_knowledge(query: str):
        """固定短證據，讓案例只觸發圖步數而非文字預算。"""
        return "固定短證據", {"sources": ["固定來源"]}

    model = fake_model(*(call(f"不同{i}", i) for i in range(8)))
    assistant = Assistant("http://unused/v1", llm=model)
    original = build_graph

    async def run():
        return [f async for f in assistant.ask(None, "方法綜合問題")]

    with patch("web.assistant.make_tools", return_value=[lookup_knowledge]), patch(
            "web.assistant.build_graph", side_effect=lambda *a, **k: original(*a, **k, limits={"rounds": 100, "models": 100, "tools": 100})):
        frames = asyncio.run(run())
    text = "".join(f.get("text", "") for f in frames)
    assert "流程上限" in text
    assert "GraphRecursionError" not in text
    assert any(f.get("mode") == "verified_fallback" for f in frames)
    assert not any(f.get("code") == "model_unavailable" for f in frames)


def test_cancel_stops_model_and_following_request_still_works():
    async def run():
        slow = SlowModel()
        assistant = Assistant("http://unused/v1", llm=slow)

        async def collect():
            return [f async for f in assistant.ask(None, "方法問題")]

        pending = asyncio.create_task(collect())
        await slow.started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert slow.stopped
        assistant._llm = fake_model(call("二", name="get_run_status"), AIMessage(content="下一輪完成"))
        frames = await collect()
        assert "下一輪完成" in "".join(f.get("text", "") for f in frames)

    asyncio.run(run())


def test_normalizer_accepts_equivalent_experiment_numbers():
    assert experiment_number("二") == experiment_number("實驗2")
