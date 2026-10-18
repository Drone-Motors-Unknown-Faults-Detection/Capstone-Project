"""時間writer的離線負例，不連SSH、不依賴data。"""

import json
import pytest

from tests.execution_timing_evidence import validate_reference, verify_time


@pytest.mark.parametrize("path", ["../file", "output/web_server/../exp12.json", "/home/andy/secret", "output/web_server/a/experiments/x.json"])
def test_reject_nonstandard_paths(path):
    with pytest.raises(ValueError):
        validate_reference(path)


def test_source_and_time_must_match():
    value = {"started_at": None, "finished_at": "2026-10-18T12:11:13", "seconds": 50.15, "saved_sha256": "a"*64}
    text = "開始時間：原檔未記錄；原檔未提供時區\n來源識別：" + json.dumps(value)
    frames = [{"event": "mode", "mode": "record_timing"}, {"event": "delta", "origin": "record", "text": text}, {"event": "done"}]
    assert verify_time(frames, value, old=True, sha="a"*64) == value
    with pytest.raises(AssertionError):
        verify_time(frames, {**value, "seconds": 4}, old=True)
    with pytest.raises(AssertionError):
        verify_time(frames, value, old=True, sha="b"*64)


def test_new_presentation_keeps_raw_time_and_rejects_source_tampering():
    from web.presentation import timing_presentation, markdown
    source = {"exp_id": "exp12", "mode": "saved", "status": "done", "started_at": None,
              "finished_at": "2026-10-18T12:11:13", "seconds": 50.15, "saved_sha256": "a" * 64}
    p = timing_presentation(source)
    frames = [{"event": "mode", "mode": "record_timing", "source_id": p["source_id"], "presentation_source": source},
              {"event": "presentation", "presentation": p},
              {"event": "delta", "origin": "record", "text": markdown(p)}, {"event": "done"}]
    assert verify_time(frames, source, old=True, sha="a" * 64) == source
    frames[0]["source_id"] = "b" * 64
    with pytest.raises(AssertionError):
        verify_time(frames, source, old=True)
