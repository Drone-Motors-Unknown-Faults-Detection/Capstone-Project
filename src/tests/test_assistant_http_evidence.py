"""測試站驗證器拒絕正式目標、舊模式與來源不一致；不依賴外網。"""

import json

import pytest

from tests.assistant_http_evidence import json_headers, parse_sse, validate_url, verify_result
from tests.test_assistant_results import OfflineAssistant, collect, result_snapshot


@pytest.mark.parametrize("url", ["https://phm.jw-albert.dev", "http://140.134.138.33:8600", "http://127.0.0.1:8000"])
def test_formal_service_targets_are_rejected(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_real_result_frames_are_accepted_but_source_or_mode_tampering_is_not():
    frames = collect(OfflineAssistant("unused"), "實驗十二結果", snap=result_snapshot())
    raw = "\n\n".join("data: " + json.dumps(f) for f in frames) + "\n\ndata: " + json.dumps({"event": "done"}) + "\n\n"
    decoded = parse_sse(raw)
    assert verify_result(decoded, "指定不存在的來源SHA")
    decoded[0]["mode"] = "model"
    assert "結果未走新版程式核算模式" in verify_result(decoded, "指定不存在的來源SHA")


def test_json_headers_carry_same_origin():
    """#304 F5-02：伺服器拒絕三種來源標頭都沒有的狀態端點請求，證據腳本要帶同站 Origin（不含路徑）。"""
    assert json_headers("http://127.0.0.1:8611/") == {"Content-Type": "application/json", "Origin": "http://127.0.0.1:8611"}
