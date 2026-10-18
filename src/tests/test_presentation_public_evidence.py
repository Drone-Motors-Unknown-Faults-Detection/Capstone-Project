"""公開欄位採白名單；額外欄位、問答、伺服器識別與來源路徑均不能外流。"""

import json
import pytest

from tests.presentation_public_evidence import public_summary, read_report


def fixture():
    counts = {"tests": 252, "failures": 0, "errors": 0, "skipped": 0, "returncode": 0, "status": "PASS"}
    private = "不得公開的路徑與問答"
    source = {"path": private}
    p = {"status": "VERIFIED_HTTP_NODE", "deployment_sha": "a" * 40, "browser": "NOT_VERIFIED",
         "sources_before": source, "sources_after": source, "server_before": source, "server_after": source,
         "cases": [{"view": view, "status": "COMPLETED_LIMITED_PROBE" if view == "method" else "VERIFIED_HTTP_NODE",
                    "question": private, "frames": [private], "container": private}
                   for view in ("default", "time", "default", "lw_classes", "full", "tsne", "method")], "extra": private}
    return p, {**counts, "environment": private}, counts


def test_only_public_allowlisted_fields_are_exported():
    result = public_summary(*fixture())
    raw = json.dumps(result, ensure_ascii=False)
    for forbidden in ("不得公開", "question", "frames", "container", '"environment"', '"path"', '"extra"'):
        assert forbidden not in raw
    assert result["related_tests"]["tests"] == 252
    assert result["source_bytes_unchanged"] is True


@pytest.mark.parametrize("case", ["failed", "browser", "sha", "view", "case-status", "count"])
def test_invalid_or_unverified_summary_is_not_published(case):
    p, related, windows = fixture()
    if case == "failed":
        p["status"] = "FAILED"
    elif case == "browser":
        p["browser"] = "VERIFIED"
    elif case == "sha":
        p["deployment_sha"] = "private/path"
    elif case == "view":
        p["cases"][0]["view"] = "private/path"
    elif case == "case-status":
        p["cases"][0]["status"] = "FAILED"
    else:
        related["tests"] = "private/path"
    with pytest.raises(ValueError):
        public_summary(p, related, windows)


def test_export_reader_rejects_unrelated_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        read_report(tmp_path / "outside/summary.json", "pytest_evidence")
