"""抽驗器只確認資料來源及有界流程，不能自動宣稱語意正確。"""
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.assistant_semantics_evidence import run, validate_url


@pytest.mark.parametrize("url", ["http://example.com/v1", "http://127.0.0.1:8600", "http://127.0.0.1:6767/v1/extra"])
def test_probe_rejects_other_targets(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_probe_preserves_source_and_does_not_claim_semantic_pass(tmp_path):
    source = tmp_path / "summary.json"
    raw = json.dumps({"stages": [], "seed": 42}).encode()
    source.write_bytes(raw)
    async def ask(self, *args):
        yield {"event": "tool", "name": "get_run_status", "sources": []}
        yield {"event": "delta", "text": "測試回答，不代表語意正確"}
    with patch("web.assistant.Assistant.ask", ask):
        result = asyncio.run(run("http://127.0.0.1:6767/v1", source))
    assert len(result["cases"]) == 3 and result["source_unchanged"]
    assert result["semantic_status"] == "REQUIRES_REVIEW"
    assert source.read_bytes() == raw
    assert next(Path("output").glob("assistant_semantics_evidence/*/summary.json")).is_file()
