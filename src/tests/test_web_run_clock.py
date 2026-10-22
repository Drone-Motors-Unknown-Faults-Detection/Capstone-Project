"""用可控制時鐘與stub實驗驗證runner，不等待真實50秒。"""

import json

import pytest

from web import run_timing
from web.experiments import ExperimentRunner


@pytest.fixture
def runner(tmp_path):
    return ExperimentRunner([], "unused", out_dir=tmp_path / "experiments")


def clocks(monkeypatch, wall, counter):
    walls, counters = iter(wall), iter(counter)
    monkeypatch.setattr(run_timing, "utc_now", lambda: next(walls))
    monkeypatch.setattr(run_timing.time, "perf_counter", lambda: next(counters))


def test_batch_real_boundaries_and_json_conversion(monkeypatch, runner):
    clocks(monkeypatch, ["2026-10-18T23:59:30.000+00:00", "2026-10-19T00:00:20.150+00:00"], [10, 60.15])
    monkeypatch.setattr(runner, "_dispatch", lambda *_: {"value": 7})
    payload = runner.run("exp12", {})
    raw = json.loads(next(runner.out_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["started_at"] == "2026-10-18T23:59:30.000+00:00"
    assert payload["finished_at"] == "2026-10-19T00:00:20.150+00:00"
    assert raw == {k: v for k, v in payload.items() if k != "saved"}
    assert payload["seconds"] == 50.15
    assert payload["timing_scope"] == "batch_execution"


def test_stream_starts_when_consumed_and_keeps_playback_wait(monkeypatch, runner):
    monkeypatch.setattr(runner, "_clean", lambda *_: {"seed": 42})
    monkeypatch.setattr(runner, "_dispatch", lambda *_, **__: iter([
        {"event": "tick", "score": 1}, {"event": "result", "result": {"value": 7}}]))
    # 建立generator時不能取時鐘；兩個counter差含消費端的播放等待。
    gen = runner.stream("exp1", {})
    clocks(monkeypatch, ["2026-10-18T12:00:00.000+00:00", "2026-10-18T12:00:50.150+00:00"], [100, 150.15])
    start, _ = next(gen)
    next(gen)
    done, _ = next(gen)
    payload = done["payload"]
    assert start["started_at"] == payload["started_at"]
    assert payload["seconds"] == 50.15
    assert payload["timing_scope"] == "stream_elapsed_including_playback"
    raw = json.loads(next(runner.out_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert raw["started_at"] == start["started_at"]


def test_clock_rollback_does_not_change_elapsed(monkeypatch, runner):
    clocks(monkeypatch, ["2026-10-18T12:01:00.000+00:00", "2026-10-18T12:00:00.000+00:00"], [10, 60.15])
    monkeypatch.setattr(runner, "_dispatch", lambda *_: {})
    p = runner.run("exp12", {})
    assert p["seconds"] == 50.15
    assert p["finished_at"] < p["started_at"]


def test_failed_run_does_not_write_done(monkeypatch, runner):
    def fail(*_):
        raise RuntimeError("刻意失敗")
    monkeypatch.setattr(runner, "_dispatch", fail)
    with pytest.raises(RuntimeError):
        runner.run("exp12", {})
    artifacts = list(runner.out_dir.glob("*.json"))
    assert len(artifacts) == 1
    saved = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert saved["status"] == "FAILED" and saved["error_type"] == "RuntimeError"
    assert "result" not in saved and "finished_at" not in saved
    assert "刻意失敗" not in artifacts[0].read_text(encoding="utf-8")
