"""測試站驗證器拒絕正式目標、舊模式與來源不一致；不依賴外網。"""

import io
import json
import urllib.error
import urllib.request

import pytest

from tests.assistant_http_evidence import (HostNotAllowedError, json_headers, open_url, parse_sse, validate_url,
                                          verify_result)
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


def test_json_headers_send_same_origin():
    """#304 F5-02：伺服器拒絕沒有來源標頭的狀態端點請求，證據腳本的 POST 帶測試站本身的 Origin。"""
    assert json_headers("http://127.0.0.1:8611/sub/") == {"Content-Type": "application/json",
                                                         "Origin": "http://127.0.0.1:8611"}


def _http_error(code, body):
    return urllib.error.HTTPError("http://140.134.138.33:8787/api/assistant/ask", code, "x", {}, io.BytesIO(body.encode()))


def test_host_403_names_the_allow_host_setting(monkeypatch):
    """#304：8787 容器沒設 ALLOW_HOST 時，伺服器以 Host 允許清單回 403；腳本要直接說出要設哪個值。"""
    body = json.dumps({"error": "Host 不是本機名稱，也不在 --allow-host 名單"}, ensure_ascii=False)

    def refuse(request, timeout):
        raise _http_error(403, body)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    request = urllib.request.Request("http://140.134.138.33:8787/api/assistant/ask", data=b"{}")
    with pytest.raises(HostNotAllowedError, match="ALLOW_HOST=140.134.138.33"):
        open_url(request, 1)


def test_other_http_errors_pass_through(monkeypatch):
    for code, body in ((403, '{"error": "跨站請求被拒絕"}'), (500, "boom")):
        def refuse(request, timeout, code=code, body=body):
            raise _http_error(code, body)

        monkeypatch.setattr(urllib.request, "urlopen", refuse)
        with pytest.raises(urllib.error.HTTPError) as error:
            open_url("http://140.134.138.33:8787/", 1)
        assert not isinstance(error.value, HostNotAllowedError)
        assert error.value.code == code
