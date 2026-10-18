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
