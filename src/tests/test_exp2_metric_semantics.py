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
    assert set(final) == {"known_rate", "matched_rate"}
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
