"""實驗二保存欄位的資料角色；不把 fixture 當真實 Qwen 驗收。"""

import copy
import json

import pytest
from langchain_core.messages import ToolMessage

from web.assistant import load_knowledge, make_tools, search_knowledge
from web.playback import PlaybackTracker, summarize_result
from web.presentation import exp2_metric_contract


def fixture():
    return {"stages": [{"config": "7screws", "learned": "discovered-1",
        "own_holdout_acc": 0.9, "own_holdout_n": 10,
        "eval_known_rate": 0.6, "eval_matched_rate": 0.55, "eval_n": 65, "cluster_purity": 0.8, "cluster_size": 50,
        "healthy_holdout_acc": 0.95}],
        "final_eval": {"7screws": {"n": 65, "unknown_rate": 0.3, "healthy_rate": 0.1, "known_rate": 0.6,
                                   "matched_rate": 0.5, "top_class": "discovered-1"}},
        "seed": 42, "confidence": 0.95}


def test_contract_keeps_values_but_separates_four_populations():
    result = fixture()
    before = copy.deepcopy(result)
    contract = exp2_metric_contract(result)
    rows = {row["metric"]: row for row in contract["rows"]}
    assert (rows["own_holdout_acc"]["value"], rows["own_holdout_acc"]["denominator"]) == (0.9, 10)
    assert (rows["eval_known_rate"]["value"], rows["eval_known_rate"]["denominator"]) == (0.6, 65)
    assert rows["healthy_holdout_acc"]["denominator"] is None
    assert (rows["eval_matched_rate"]["value"], rows["eval_matched_rate"]["denominator"]) == (0.55, 65)
    assert rows["eval_matched_rate"]["status"] == "VERIFIED_SAVED_FIELDS"
    assert contract["unknown_auroc"] == contract["unknown_recall"] == "NOT_PROVIDED"
    assert "不是未知召回" in contract["text"] or "不等於未知召回" in contract["text"]
    assert "不要求判對自身配置" in contract["text"]
    assert result == before


def test_complete_result_gives_complete_status():
    # 沒有分母欄位的 healthy_holdout_acc 只看值本身，不再把每份完整結果都標成 incomplete（#279）
    contract = exp2_metric_contract(fixture())
    assert contract["status"] == "complete"
    rows = {row["metric"]: row for row in contract["rows"]}
    assert rows["healthy_holdout_acc"]["status"] == "VERIFIED_SAVED_FIELDS"
    assert all(row["status"] == "VERIFIED_SAVED_FIELDS" for row in contract["rows"])


def test_final_eval_matched_rate_rows_are_listed_per_config():
    contract = exp2_metric_contract(fixture())
    final = {row["metric"]: row for row in contract["rows"] if row.get("stage_index") == "final_eval"}
    assert set(final) == {"known_rate", "matched_rate", "unknown_rate"}
    assert (final["unknown_rate"]["value"], final["unknown_rate"]["denominator"]) == (0.3, 65)
    assert (final["matched_rate"]["config"], final["matched_rate"]["value"], final["matched_rate"]["denominator"]) == ("7screws", 0.5, 65)
    assert "final_eval：7screws" in contract["text"]
    assert "matched_rate 才要求判進事後多數等於該配置的新類" in contract["text"]
    broken = fixture()
    broken["final_eval"]["7screws"]["n"] = 0
    bad = exp2_metric_contract(broken)
    assert bad["status"] == "incomplete"
    assert all(row["value"] is None for row in bad["rows"] if row.get("stage_index") == "final_eval")


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -0.1, 1.1, "0.9"])
def test_invalid_fraction_not_presented_as_supported_value(value):
    result = fixture()
    result["stages"][0]["own_holdout_acc"] = value
    row = exp2_metric_contract(result)["rows"][0]
    assert row["value"] is None and row["status"] == "INCOMPLETE"


@pytest.mark.parametrize("count", [0, -1, True, "10"])
def test_invalid_denominator_does_not_produce_a_rate_claim(count):
    result = fixture()
    result["stages"][0]["own_holdout_n"] = count
    row = exp2_metric_contract(result)["rows"][0]
    assert row["value"] is None and row["denominator"] is None
    assert row["status"] == "INVALID_DENOMINATOR"


@pytest.mark.parametrize("result", [None, {}, {"stages": []}, {"stages": [None]}, {"stages": [{}]}])
def test_missing_fields_remain_incomplete_without_invented_counts(result):
    contract = exp2_metric_contract(result)
    assert contract["status"] == "incomplete"
    assert contract["rows"] == []
    assert "187／62／64" in contract["text"] and "指定示例" in contract["text"]


def test_budget_does_not_cut_out_an_unqualified_number():
    assert exp2_metric_contract(fixture(), budget=3)["text"] == "不完整"
    assert exp2_metric_contract(fixture(), budget=1)["text"] == "略"
    with pytest.raises(ValueError):
        exp2_metric_contract(fixture(), budget=0)


def test_run_and_stream_tools_share_the_same_semantic_facts():
    payload = {"params": {"seed": 42}, "result": fixture(), "seconds": 1.0}
    tracker = PlaybackTracker()
    tracker.record_run("exp2", payload)
    batch = tracker.snapshot()["records"]["exp2"]["fact_text"]
    tracker.start("exp2", {})
    tracker.update({"event": "done", "payload": payload})
    stream = tracker.snapshot()["records"]["exp2"]["fact_text"]
    assert batch == stream == summarize_result(fixture(), exp_id="exp2")
    tools = {tool.name: tool for tool in make_tools(tracker.snapshot(), [])}
    for name in ("get_run_status", "get_run_events"):
        reply = tools[name].invoke({"name": name, "args": {"number": "2"}, "id": name, "type": "tool_call"})
        assert isinstance(reply, ToolMessage)
        assert reply.artifact["facts"] == batch
        assert "群內保留列" in reply.content


def test_saved_exp2_reads_the_selected_file_and_preserves_bytes(tmp_path):
    root = tmp_path / "web_server"
    path = root / "session" / "experiments" / "exp2_2026-10-18-20-00-00.json"
    path.parent.mkdir(parents=True)
    raw = json.dumps({"experiment": "exp2", "params": {}, "result": fixture()}).encode()
    path.write_bytes(raw)
    rec = PlaybackTracker(root).load_saved_record(path.name, "exp2")
    assert rec["mode"] == "saved"
    assert rec["metric_contract"]["unknown_auroc"] == "NOT_PROVIDED"
    assert path.read_bytes() == raw


def test_method_and_unknown_spec_remain_searchable():
    sections = load_knowledge()
    assert "alpha = 0.08" in search_knowledge(sections, "EWMA 警報")
    assert "UNKNOWN" in search_knowledge(sections, "T1 馬達規格")
    assert "不是" in search_knowledge(sections, "confidence 分位數")


def paired_result():
    result = fixture()
    result["stages"][0].update(eval_known_rate=.9, eval_matched_rate=.2)
    result["final_eval"] = {"7screws": {"known_rate": .6, "matched_rate": .1, "unknown_rate": .25, "n": 65}}
    return result


def test_stage_and_final_matched_are_separate_saved_fields():
    contract = exp2_metric_contract(paired_result())
    rows = {(r["timepoint"], r["metric"]): r for r in contract["rows"]}
    for key, value in (("eval_known_rate", .9), ("eval_matched_rate", .2)):
        assert rows[("stage", key)]["value"] == value
        assert rows[("stage", key)]["denominator"] == 65
    for key, value in (("known_rate", .6), ("matched_rate", .1), ("unknown_rate", .25)):
        assert rows[("final", key)]["value"] == value
        assert rows[("final", key)]["denominator"] == 65
    assert contract["version"] == "exp2_metric_semantics_v2"
    assert contract["unknown_auroc"] == contract["unknown_recall"] == "NOT_PROVIDED"
    assert "全部階段結束" in contract["text"]


@pytest.mark.parametrize("key", ["eval_matched_rate", "eval_known_rate"])
def test_absent_stage_field_is_not_filled_from_other_timepoint_or_metric(key):
    result = paired_result()
    del result["stages"][0][key]
    contract = exp2_metric_contract(result)
    assert not any(r["metric"] == key for r in contract["rows"])
    assert any(r["timepoint"] == "final" and r["metric"] == "matched_rate" and r["value"] == .1
               for r in contract["rows"])


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1, 2, "0.5"])
def test_invalid_final_fraction_cannot_reach_visible_numeric_claim(value):
    result = paired_result()
    result["final_eval"]["7screws"]["matched_rate"] = value
    row = next(r for r in exp2_metric_contract(result)["rows"] if r["metric"] == "matched_rate")
    assert row["value"] is None and row["status"] == "INCOMPLETE"


@pytest.mark.parametrize("count", [None, 0, -1, True, "65"])
def test_final_missing_or_bad_denominator_never_borrows_stage_count(count):
    result = paired_result()
    result["final_eval"]["7screws"]["n"] = count
    rows = [r for r in exp2_metric_contract(result)["rows"] if r["timepoint"] == "final"]
    assert len(rows) == 3
    assert all(r["value"] is None and r["denominator"] is None for r in rows)


def test_exp2_hostile_model_cannot_contradict_facts_in_any_frame():
    import asyncio
    from langchain_core.messages import AIMessage
    from tests.test_assistant import fake_model
    from web.assistant import Assistant

    tracker = PlaybackTracker()
    tracker.record_run("exp2", {"result": paired_result()})
    model = fake_model(AIMessage(content="", tool_calls=[
        {"name": "get_run_status", "args": {"number": "2"}, "id": "metrics"}]),
        AIMessage(content="錯答：外部接受率等於自身配置正確率，未知召回也包含正確新類，AUROC為0.99。"))
    async def collect():
        return [frame async for frame in Assistant("unused", llm=model).ask(
            tracker.snapshot(), "實驗二這次結果的eval_known_rate、matched_rate與未知召回是什麼？")]
    frames = asyncio.run(collect())
    assert model.seen == []
    assert frames[0]["mode"] == "exp2_metrics"
    assert all("錯答" not in f.get("text", "") and f.get("origin") != "model" for f in frames)
    assert "不是模型自由生成" in frames[-1]["text"]


@pytest.mark.parametrize("remove", ["stages", "final_eval"])
def test_missing_timepoint_does_not_erase_other_saved_fields(remove):
    result = paired_result()
    del result[remove]
    contract = exp2_metric_contract(result)
    assert contract["status"] == "incomplete"
    expected = "final" if remove == "stages" else "stage"
    assert contract["rows"] and all(r["timepoint"] == expected for r in contract["rows"])


@pytest.mark.parametrize("status", ["FAILED", "running", "stopped", "unexpected"])
@pytest.mark.parametrize("location", ["payload", "result"])
def test_saved_status_cannot_become_a_completed_result(tmp_path, status, location):
    from tests.test_assistant_results import OfflineAssistant, collect
    payload = {"experiment": "exp2", "result": paired_result()}
    (payload if location == "payload" else payload["result"])["status"] = status
    folder = tmp_path / "session/experiments"
    folder.mkdir(parents=True)
    path = folder / "exp2_2026-10-23-00-00-00.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    tracker = PlaybackTracker(tmp_path)
    rec = tracker.load_saved_record(path.name, "exp2")
    assert rec["status"] != "done"
    frames = collect(OfflineAssistant("unused"), f"實驗二 `{path.name}` 結果摘要",
                     snap={"latest": None, "records": {}}, load_selected=tracker.load_saved_record)
    assert frames[0]["mode"] == "exp2_metrics" and frames[0]["status"] == "incomplete"
    assert "尚未完成" in frames[-1]["text"]
    assert not any(f["event"] == "presentation" for f in frames)


def test_exp2_exact_source_survives_followup_then_explicitly_switches_latest(tmp_path):
    import hashlib
    from tests.test_assistant_results import OfflineAssistant, collect
    folder = tmp_path / "session/experiments"
    folder.mkdir(parents=True)
    paths = []
    for index, value in enumerate((.2, .3)):
        result = paired_result()
        result["stages"][0]["eval_matched_rate"] = value
        path = folder / f"exp2_2026-10-23-00-00-0{index}.json"
        path.write_text(json.dumps({"experiment": "exp2", "result": result}), encoding="utf-8")
        paths.append(path)
    before = [p.read_bytes() for p in paths]
    tracker = PlaybackTracker(tmp_path)
    options = dict(snap={"latest": None, "records": {}}, load_selected=tracker.load_saved_record, load_saved=tracker.load_saved)
    assistant = OfflineAssistant("unused")
    first = collect(assistant, f"摘要 `{paths[0].name}` 的matched_rate", **options)
    follow = collect(assistant, "那接受率與未知召回呢？", context=first[0]["context"], **options)
    history = collect(assistant, "那接受率呢？", history=[{"role": "user", "content": f"摘要 `{paths[0].name}`"},
        {"role": "assistant", "content": f"改讀 `{paths[1].name}`"}], **options)
    latest = collect(assistant, "實驗二最新結果摘要", context=first[0]["context"], **options)
    for frames, index in ((first, 0), (follow, 0), (history, 0), (latest, 1)):
        p = next(f["presentation"] for f in frames if f["event"] == "presentation")
        assert p["source"]["saved_sha256"] == hashlib.sha256(before[index]).hexdigest()
        assert frames[0]["source_id"] == p["source_id"]
        table = next(s for s in p["sections"] if s["id"] == "stages")
        assert table["rows"][0][table["headers"].index("eval_matched_rate")] == f"{(.2, .3)[index]}（n=65）"
    assert [p.read_bytes() for p in paths] == before


@pytest.mark.parametrize("reference", ["exp2_2026-10-23-00-00-00.json", "../exp2_2026-10-23-00-00-00.json"])
def test_missing_or_unsafe_exp2_source_never_falls_back_to_latest(tmp_path, reference):
    from tests.test_assistant_results import OfflineAssistant, collect
    tracker = PlaybackTracker(tmp_path)
    tracker.record_run("exp2", {"result": paired_result()})
    frames = collect(OfflineAssistant("unused"), f"摘要 `{reference}` 的matched_rate", snap=tracker.snapshot(),
                     load_selected=tracker.load_saved_record, load_saved=tracker.load_saved)
    assert "資料不足" in frames[-1]["text"] and "0.2" not in frames[-1]["text"]
    assert frames[0]["status"] == "missing"


def test_exp2_context_conflict_and_over_budget_source_never_reach_model():
    from tests.test_assistant_results import OfflineAssistant, collect
    tracker = PlaybackTracker()
    tracker.record_run("exp2", {"result": paired_result()})
    frames = collect(OfflineAssistant("unused"), "實驗二結果", snap=tracker.snapshot(), context={"experiment": "exp1"})
    assert frames[0]["event"] == "error"
    tracker.record_run("exp2", {"params": {"padding": "x" * 9000}, "result": paired_result()})
    frames = collect(OfflineAssistant("unused"), "實驗二結果", snap=tracker.snapshot())
    assert frames[0]["status"] == "incomplete" and "摘要不完整" in frames[-1]["text"]
    assert not any(f["event"] == "presentation" for f in frames)


def test_general_method_question_keeps_model_path_after_exp2_record():
    from langchain_core.messages import AIMessage
    from tests.test_assistant import fake_model
    from tests.test_assistant_results import collect
    from web.assistant import Assistant
    tracker = PlaybackTracker()
    tracker.record_run("exp2", {"result": paired_result()})
    model = fake_model(AIMessage(content="", tool_calls=[
        {"name": "lookup_knowledge", "args": {"query": "HDBSCAN"}, "id": "method"}]), AIMessage(content="一般方法測試回答"))
    frames = collect(Assistant("unused", llm=model), "實驗二HDBSCAN分群的原理是什麼？", snap=tracker.snapshot())
    assert model.seen
    assert any(f.get("origin") == "model" for f in frames)
    assert not any(f.get("mode") == "exp2_metrics" for f in frames)


def test_shared_metric_helper_preserves_valid_fraction_without_defined_denominator():
    from unittest.mock import patch
    from web import presentation
    from tests.test_assistant_results import OfflineAssistant, collect
    result = fixture()
    with patch.object(presentation, "_metric_row", wraps=presentation._metric_row) as shared:
        contract = exp2_metric_contract(result)
    assert shared.call_count == len(presentation.EXP2_METRICS) + len(presentation.FINAL_EVAL_METRICS)
    healthy = next(r for r in contract["rows"] if r["metric"] == "healthy_holdout_acc")
    assert healthy["value"] == .95 and healthy["status"] == "VERIFIED_SAVED_FIELDS"
    assert healthy["denominator"] is None and not healthy["denominator_required"]
    tracker = PlaybackTracker()
    tracker.record_run("exp2", {"result": result})
    frames = collect(OfflineAssistant("unused"), "實驗二保存指標結果", snap=tracker.snapshot())
    assert frames[0]["status"] == "complete"
    assert "0.95（此欄未定義分母）" in frames[-1]["text"]
    assert "n=None" not in frames[-1]["text"]


def test_saved_configuration_mismatch_cannot_be_replaced_with_unsupported_success():
    from langchain_core.messages import AIMessage
    from tests.test_assistant import fake_model
    from tests.test_assistant_results import collect
    from web.assistant import Assistant
    result = fixture()
    result["stages"][0].update(config="6screws", matched="7screws", cluster_purity=.4)
    tracker = PlaybackTracker()
    tracker.record_run("exp2", {"result": result})
    model = fake_model(AIMessage(content="錯誤合成主張：所有階段純度必為100%，完全匹配。"))
    frames = collect(Assistant("unused", llm=model), "實驗二這次保存結果的匹配與純度", snap=tracker.snapshot())
    assert model.seen == []
    assert "6screws | 7screws" in frames[-1]["text"]
    assert "0.4（n=50）" in frames[-1]["text"]
    assert all("錯誤合成主張" not in f.get("text", "") for f in frames)
