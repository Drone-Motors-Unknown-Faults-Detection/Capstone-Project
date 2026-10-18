"""主要答案與每個中間frame的語意回歸；網站錯答線索不冒充已取得原檔。"""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from tests.exp12_counts_fixture import REPORTED_WRONG_ANSWER, historical_result
from tests.test_assistant import fake_model
from web.assistant import Assistant
from web.playback import PlaybackTracker


def result_snapshot(result=None):
    tracker = PlaybackTracker()
    tracker.record_run("exp12", {"params": {"seed": 42, "confidence": 0.95}, "result": historical_result() if result is None else result,
                                 "saved": "output/web_server/test-session/experiments/exp12_2026-10-09-14-27-55.json"})
    return tracker.snapshot()


def hostile_model():
    return fake_model(
        AIMessage(content="", tool_calls=[{"name": "get_run_status", "args": {"number": "12"}, "id": "c1"}]),
        AIMessage(content=REPORTED_WRONG_ANSWER))


def collect(assistant, question, snap=None, **kwargs):
    async def run():
        return [frame async for frame in assistant.ask(snap or result_snapshot(), question, **kwargs)]
    return asyncio.run(run())


def test_hostile_model_cannot_leak_wrong_text_in_any_delta_or_final_answer():
    model = hostile_model()
    frames = collect(Assistant("http://unused/v1", llm=model), "總結實驗十二的結果")
    visible = "".join(frame["text"] for frame in frames if frame["event"] == "delta")
    for fragment in ("legacy 模型完全失效", "接近100%", "46～47", "unknown無誤報", "output/exp12_confusion_tsne/"):
        assert all(fragment not in frame.get("text", "") for frame in frames)
        assert fragment not in visible
    assert "2736／2938＝93.12%" in visible
    assert "202／2938＝6.88%" in visible
    assert "結果摘要由程式依矩陣產生" in visible
    assert model.seen == []


@pytest.mark.parametrize("question", [
    "實驗12結果如何", "實驗十二結果是不是很好", "exp12的legacy是不是失效", "legacy是否沒有任何判對、應立即停用",
    "2938筆是不是都誤拒", "未知全部判unknown，代表全部分類錯了嗎", "這能保證換馬達仍可靠嗎", "目前結果如何",
])
def test_result_phrasings_do_not_call_hostile_model(question):
    model = hostile_model()
    frames = collect(Assistant("http://unused/v1", llm=model), question)
    text = "".join(f.get("text", "") for f in frames if f["event"] == "delta")
    assert "2736／2938＝93.12%" in text
    assert "8screws（健康）" in text
    for count in (47, 39, 46, 30, 40):
        assert f"誤拒 {count}" in text
    assert "未知全拒不代表整個系統沒有誤報" in text
    assert "本紀錄未提供可核驗版本" in text
    assert "不證明未見馬達泛化" in text
    assert not any(f["event"] == "facts" for f in frames)
    assert model.seen == []


class OfflineAssistant(Assistant):
    async def llm(self):
        raise TimeoutError("不應初始化離線模型")


def test_valid_result_is_available_when_model_is_offline():
    frames = collect(OfflineAssistant("http://unused/v1"), "總結實驗十二")
    assert frames[0]["mode"] == "matrix_summary"
    assert frames[0]["status"] == "complete"
    assert "45／572＝7.87%" in frames[-1]["text"]


@pytest.mark.parametrize("missing,notice,available", [
    ("binary_confusion", "缺少二元矩陣", "2736／2938＝93.12%"),
    ("openset_confusion", "缺少多類開集矩陣", "45／572＝7.87%"),
])
def test_partial_legacy_payload_names_missing_section_without_model(missing, notice, available):
    result = historical_result()
    del result[missing]
    frames = collect(OfflineAssistant("http://unused/v1"), "實驗十二結果", snap=result_snapshot(result))
    assert frames[0]["status"] == "incomplete"
    assert notice in frames[-1]["text"]
    assert available in frames[-1]["text"]


def test_wrong_assistant_history_cannot_select_or_rewrite_result():
    snap = result_snapshot()
    tracker = PlaybackTracker()
    tracker.record_run("exp1", {"result": {"wrong": 999}})
    snap["records"].update(tracker.snapshot()["records"])
    snap["latest"] = "exp1"
    frames = collect(OfflineAssistant("http://unused/v1"), "那legacy真的完全失效嗎", snap=snap,
                     history=[{"role": "user", "content": "總結實驗十二"},
                              {"role": "assistant", "content": REPORTED_WRONG_ANSWER}])
    assert "正確拒絕 14381" in frames[-1]["text"]
    assert "exp12" in frames[0]["sources"][0]
    assert "wrong" not in frames[-1]["text"]


@pytest.mark.parametrize("question", ["實驗十二在做什麼", "實驗12的t-SNE原理", "legacy方法是什麼"])
def test_method_questions_still_use_model(question):
    model = fake_model(
        AIMessage(content="", tool_calls=[{"name": "get_experiment", "args": {"number": "12"}, "id": "c1"}]),
        AIMessage(content="方法說明文字"))
    frames = collect(Assistant("http://unused/v1", llm=model), question,
                     context={"experiment": "exp12", "mode": "result", "saved": "上一筆.json"})
    assert model.seen
    assert not any(f["event"] == "mode" for f in frames)


def test_changed_matrix_and_label_order_change_visible_facts():
    result = historical_result()
    result["binary_confusion"]["matrix"][0][1] += 7
    entry = result["openset_confusion"]["ledoit_wolf"]
    entry["matrix"][0][1] = 3
    entry["matrix"][5][0] = 2
    order = [5, 4, 3, 2, 1, 0]
    entry["labels"] = [entry["labels"][i] for i in order]
    entry["matrix"] = [[entry["matrix"][i][j] for j in order] for i in order]
    frames = collect(OfflineAssistant("http://unused/v1"), "實驗十二結果", snap=result_snapshot(result))
    text = frames[-1]["text"]
    assert "52／579＝8.98%" in text
    assert "已知互相錯分 3" in text
    assert "unknown→known 未拒 2" in text
    assert "8screws（健康）：支持 575、判對 525、誤拒 47、錯分其他已知 3" in text


@pytest.mark.parametrize("case", ["missing", "normalized", "zero", "unfinished", "budget"])
def test_inadequate_results_do_not_fall_back_to_model_or_history(case):
    result = historical_result()
    if case == "missing":
        result = {}
    if case == "normalized":
        result["value_kind"] = "normalized"
    if case == "zero":
        result = {"binary_confusion": {"labels": ["healthy", "fault"], "matrix": [[0, 0], [0, 0]]}}
    snap = result_snapshot(result)
    if case == "unfinished":
        snap["records"]["exp12"]["status"] = "running"
    if case == "budget":
        tracker = PlaybackTracker()
        tracker.record_run("exp12", {"params": {"padding": "x" * 9000}, "result": result})
        snap = tracker.snapshot()
    frames = collect(OfflineAssistant("http://unused/v1"), "實驗十二結果", snap=snap)
    assert frames[0]["status"] in ("invalid", "incomplete", "complete")
    assert "不適用" in frames[-1]["text"] if case == "zero" else (
        "失敗" in frames[-1]["text"] or "不足" in frames[-1]["text"] or "未完成" in frames[-1]["text"] or "不完整" in frames[-1]["text"])
    assert REPORTED_WRONG_ANSWER not in frames[-1]["text"]


@pytest.fixture
def saved_pair(tmp_path):
    root = tmp_path / "saved"
    folder = root / "session/experiments"
    folder.mkdir(parents=True)
    paths = []
    for i in (0, 1):
        result = historical_result()
        result["binary_confusion"]["matrix"][0][1] += i
        path = folder / f"exp12_2026-01-01-00-00-0{i}.json"
        path.write_text(json.dumps({"experiment": "exp12", "params": {"seed": 42}, "result": result}), encoding="utf-8")
        paths.append(path)
    return PlaybackTracker(root), paths


def test_exact_file_is_not_replaced_by_latest_record_and_bytes_are_unchanged(saved_pair):
    tracker, paths = saved_pair
    before = paths[0].read_bytes()
    frames = collect(OfflineAssistant("http://unused/v1"), f"總結實驗十二 `{paths[0].name}`",
                     load_selected=tracker.load_saved_record, load_saved=tracker.load_saved)
    assert "45／572＝7.87%" in frames[-1]["text"]
    assert str(paths[0]) in frames[0]["sources"][0]
    assert paths[1].name not in frames[-1]["text"]
    assert paths[0].read_bytes() == before


def test_context_followup_stays_on_file_but_latest_request_can_change_source(saved_pair):
    tracker, paths = saved_pair
    context = {"experiment": "exp12", "mode": "result", "saved": str(paths[0])}
    kwargs = {"load_selected": tracker.load_saved_record, "load_saved": tracker.load_saved, "context": context,
              "snap": {"latest": None, "records": {}}}
    old = collect(OfflineAssistant("http://unused/v1"), "那誤報率是多少", **kwargs)
    new = collect(OfflineAssistant("http://unused/v1"), "實驗十二最新結果", **kwargs)
    assert "45／572＝7.87%" in old[-1]["text"]
    assert "46／573＝8.03%" in new[-1]["text"]
    assert str(paths[1]) in new[0]["sources"][0]


def test_missing_explicit_file_never_uses_latest_and_remains_explicit(saved_pair):
    tracker, _ = saved_pair
    requested = "exp12_2026-01-02-00-00-00.json"
    with patch.object(tracker, "load_saved", side_effect=AssertionError("不得取最新")):
        frames = collect(OfflineAssistant("http://unused/v1"), f"總結 `{requested}`",
                         load_selected=tracker.load_saved_record, load_saved=tracker.load_saved)
    assert frames[0]["status"] == "missing"
    assert frames[0]["context"]["saved"] == requested
    assert "7.87%" not in frames[-1]["text"]


@pytest.mark.parametrize("reference", ["../exp12_2026-01-01-00-00-00.json", "/etc/passwd",
                                      "session/../experiments/exp12_2026-01-01-00-00-00.json",
                                      "output/web_server/%2e%2e/secret.json", "exp1_2026-01-01-00-00-00.json"])
def test_unsafe_or_wrong_experiment_references_are_rejected(saved_pair, reference):
    tracker, _ = saved_pair
    with pytest.raises(ValueError):
        tracker.load_saved_record(reference)


def test_ambiguous_basename_requires_full_path(saved_pair):
    tracker, paths = saved_pair
    duplicate = tracker.results_root / "other/experiments" / paths[0].name
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="同名"):
        tracker.load_saved_record(paths[0].name)
    assert tracker.load_saved_record(str(paths[0]))["exp_id"] == "exp12"


def test_root_escape_and_symlink_are_rejected_before_read(saved_pair, monkeypatch):
    tracker, paths = saved_pair
    original = Path.resolve

    def outside(path, *args, **kwargs):
        return tracker.results_root.parent / "outside.json" if path == paths[0] else original(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "resolve", outside)
        with pytest.raises(ValueError):
            tracker.load_saved_record(str(paths[0]))
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "is_symlink", lambda path: path == paths[0])
        with pytest.raises(ValueError):
            tracker.load_saved_record(str(paths[0]))


@pytest.mark.parametrize("context", [{"mode": "unsupported"}, {"experiment": 12},
                                      {"experiment": "exp1", "saved": "exp12_2026-01-01-00-00-00.json"}])
def test_invalid_context_does_not_initialize_model(context):
    frames = collect(OfflineAssistant("http://unused/v1"), "結果如何", context=context)
    assert frames[0]["event"] == "error"
    assert "93.12%" not in str(frames)


def test_unknown_label_absent_and_unknown_only_never_invent_known_accuracy():
    for labels, matrix in [(["8screws"], [[2]]), (["unknown"], [[5]])]:
        result = {"openset_confusion": {"ledoit_wolf": {"labels": labels, "matrix": matrix}}}
        frames = collect(OfflineAssistant("http://unused/v1"), "實驗十二結果", snap=result_snapshot(result))
        assert "不適用" in frames[-1]["text"]


def test_older_client_followup_uses_user_selected_file_not_assistant_path(saved_pair):
    tracker, paths = saved_pair
    frames = collect(OfflineAssistant("http://unused/v1"), "那是不是都誤拒",
                     history=[{"role": "user", "content": f"總結 `{paths[0].name}`"},
                              {"role": "assistant", "content": REPORTED_WRONG_ANSWER}],
                     load_selected=tracker.load_saved_record, load_saved=tracker.load_saved)
    assert str(paths[0]) in frames[0]["sources"][0]
    assert "45／572＝7.87%" in frames[-1]["text"]


def test_saved_only_context_is_explicit_even_without_experiment_or_mode(saved_pair):
    tracker, paths = saved_pair
    frames = collect(OfflineAssistant("http://unused/v1"), "請說明",
                     context={"saved": str(paths[0])}, load_selected=tracker.load_saved_record)
    assert frames[0]["mode"] == "matrix_summary"
    assert str(paths[0]) in frames[0]["sources"][0]
