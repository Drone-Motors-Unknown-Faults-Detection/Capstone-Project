"""執行時間取自runner，讀檔與完成後建紀錄不得另造起點。"""

import hashlib
import json
from datetime import datetime

import pytest

from tests.exp12_counts_fixture import historical_result
from web.assistant import Assistant, describe_record
from web.run_timing import timing_fields

from web.playback import PlaybackTracker


def test_completed_record_keeps_runner_start():
    payload = {"params": {}, "started_at": "2026-10-18T23:59:30.000+00:00",
               "finished_at": "2026-10-19T00:00:20.150+00:00", "seconds": 50.15, "result": {}}
    tracker = PlaybackTracker()
    tracker.record_run("exp1", payload)
    assert tracker.state["started_at"] == payload["started_at"]


def test_old_saved_file_does_not_borrow_finish_as_start(tmp_path):
    root = tmp_path / "web_server"
    path = root / "session/experiments/exp12_2026-10-18-12-11-13.json"
    path.parent.mkdir(parents=True)
    raw = json.dumps({"experiment": "exp12", "params": {"seed": 42}, "seconds": 50.15,
                      "finished_at": "2026-10-18T12:11:13", "result": {}}).encode()
    path.write_bytes(raw)
    tracker = PlaybackTracker(root)
    selected = tracker.load_saved_record(path.name)
    assert selected["started_at"] is None
    assert tracker.load_saved("exp12")["started_at"] is None
    assert selected["result"]["saved_sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    assert "開始時間：原檔未記錄" in selected["result_text"]
    assert "原檔未提供時區" in describe_record(selected)


@pytest.mark.parametrize("start,finish,seconds,status,notice", [
    ("2026-10-18T12:11:13", "2026-10-18T12:11:13", 50.15, "INCONSISTENT", "時間紀錄不一致"),
    ("2026-10-18T12:11:13.000+00:00", "2026-10-18T12:11:13.000+00:00", 0, "VERIFIED", "耗時：0 秒"),
    ("2026-10-18T12:11:13.000+00:00", "2026-10-18T12:11:13.004+00:00", 0, "VERIFIED", "耗時：0 秒"),
    ("2026-10-18T23:59:30.000+00:00", "2026-10-19T08:00:20.150+08:00", 50.15, "VERIFIED", "+08:00"),
    ("2026-10-18T12:01:00.000+00:00", "2026-10-18T12:00:00.000+00:00", 50.15, "INCONSISTENT", "保留原值"),
    ("2026-10-18T12:11:13", "2026-10-18T12:12:03", 50.15, "INCOMPLETE", "原檔未提供時區"),
    ("2026-10-18T12:11:13+00:00", "2026-10-18T12:12:03", 50.15, "INCOMPLETE", "不能比較"),
    ("壞時間", "壞時間", 1, "INCOMPLETE", "格式無效"),
    ({"錯誤": 1}, None, -1, "INCOMPLETE", "耗時無效"),
    (None, None, float("nan"), "INCOMPLETE", "耗時無效"),
    (None, None, float("inf"), "INCOMPLETE", "耗時無效"),
    (None, None, True, "INCOMPLETE", "耗時無效"),
])
def test_time_quality_preserves_truth_and_precision(start, finish, seconds, status, notice):
    original = {"started_at": start, "finished_at": finish, "seconds": seconds}
    result = timing_fields(original)
    assert result["timing_status"] == status
    assert notice in result["timing_text"]
    if isinstance(start, str):
        assert result["started_at"] == start
    json.dumps(result, allow_nan=False)


def test_stream_stopped_and_batch_error_do_not_invent_start():
    t = PlaybackTracker()
    t.start("exp1", {})
    assert t.state["started_at"] is None
    t.update({"event": "start", "started_at": "2026-10-18T12:00:00.000+00:00"})
    t.finish("stopped")
    assert t.state["status"] == "stopped"
    assert t.state["finished_at"] is None
    assert t.state["started_at"] == "2026-10-18T12:00:00.000+00:00"
    t.record_run_error("exp1", {}, "失敗")
    assert t.state["status"] == "error"
    assert t.state["started_at"] is None


def test_time_correction_keeps_exp12_data_and_source(tmp_path):
    result = historical_result()
    before = json.dumps(result, sort_keys=True)
    t = PlaybackTracker()
    t.record_run("exp12", {"params": {"seed": 42}, "finished_at": "2026-10-18T12:11:13",
                           "seconds": 50.15, "result": result})
    text = t.state["matrix_answer"]["text"]
    for fragment in ("2736／2938＝93.12%", "202／2938＝6.88%", "45／572＝7.87%", "14381"):
        assert fragment in text
    assert json.dumps(result, sort_keys=True) == before


def test_time_question_uses_no_model_and_never_infers_old_start(tmp_path):
    import asyncio

    class Offline(Assistant):
        async def llm(self):
            raise AssertionError("時間查詢不應初始化模型")

    root = tmp_path / "web_server"
    folder = root / "session/experiments"
    folder.mkdir(parents=True)
    path = folder / "exp12_2026-10-18-12-11-13.json"
    raw = json.dumps({"experiment": "exp12", "params": {"seed": 42}, "seconds": 50.15,
                      "finished_at": "2026-10-18T12:11:13", "result": historical_result()}).encode()
    path.write_bytes(raw)
    (folder / "exp12_2026-10-18-23-59-59.json").write_text(json.dumps({
        "experiment": "exp12", "started_at": "另一筆", "result": {}}), encoding="utf-8")
    tracker = PlaybackTracker(root)

    async def collect():
        return [f async for f in Offline("http://unused").ask(None, "這次實驗何時開始、何時完成、花多久？",
                context={"experiment": "exp12", "mode": "result", "saved": path.name},
                load_saved=tracker.load_saved, load_selected=tracker.load_saved_record)]
    frames = asyncio.run(collect())
    assert frames[0]["mode"] == "record_timing"
    text = frames[-1]["text"]
    assert "開始時間：原檔未記錄" in text
    assert "原檔未提供時區" in text
    assert hashlib.sha256(raw).hexdigest() in text
    assert "另一筆" not in text
    assert path.read_bytes() == raw


def test_new_record_times_are_identical_after_restart(tmp_path):
    from web.experiments import ExperimentRunner

    root = tmp_path / "web_server"
    runner = ExperimentRunner([], "unused", out_dir=root / "session")
    runner._dispatch = lambda *_: {}
    payload = runner.run("exp12", {})
    t = PlaybackTracker(root)
    t.record_run("exp12", payload)
    for rec in (t.state, PlaybackTracker(root).load_saved("exp12"), t.load_saved_record(payload["saved"])):
        for key in ("started_at", "finished_at", "seconds", "timing_text"):
            assert rec[key] == payload[key]
        assert datetime.fromisoformat(rec["started_at"]).utcoffset().total_seconds() == 0
