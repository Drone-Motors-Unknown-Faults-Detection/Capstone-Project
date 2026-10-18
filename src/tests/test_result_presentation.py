"""格式先重現缺陷；數值仍由既有矩陣彙整產生。"""

import json
from pathlib import Path
import shutil
import subprocess
from copy import deepcopy

import pytest

from tests.exp12_counts_fixture import historical_result
from web.playback import _exp12_answer, _render
from web.presentation import display_time, markdown, select_view


def markdown_html(text):
    node = shutil.which("node")
    assert node, "Markdown呈現測試需要Node"
    script = Path(__file__).resolve().parents[2] / "src/web/static/markdown.js"
    code = "const fs=require('fs'),vm=require('vm');vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8'));process.stdout.write(renderMarkdown(JSON.parse(fs.readFileSync(0,'utf8'))));"
    return subprocess.run([node, "-e", code, str(script)], input=json.dumps(text), capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def test_generic_matrix_is_a_real_markdown_table():
    lines = []
    _render("矩陣", {"labels": ["真值", "未知"], "matrix": [[2, 1], [0, 3]]}, 0, lines)
    html = markdown_html("\n".join(lines))
    assert "<table>" in html and "<thead>" in html and "<tbody>" in html


def test_exp12_source_and_settings_are_readable_sections():
    source = {"exp_id": "exp12", "mode": "saved", "status": "done", "params": {"seed": 42},
              "started_at": "2026-10-18T12:37:51.477+00:00", "finished_at": "2026-10-18T12:38:42.345+00:00",
              "seconds": 50.87, "timing_scope": "batch_execution"}
    text = _exp12_answer(historical_result(), 6000, source)["text"]
    assert "## 來源與執行資訊" in text
    assert "來源識別：{" not in text
    assert "結果設定：{" not in text
    assert "2026-10-18 20:37:51.477" in text


def test_details_and_markdown_share_counts_and_do_not_mutate_source():
    result = historical_result()
    result["_tsne_by_dataset"] = {f"{m}/{r}": {"config_labels": ["8screws"] * n,
        "verdicts": ["known_correct"] * n, "embedding": [[0, 0]] * n}
        for m in ("T1", "T2", "T3") for r in (6000, 8000, 11000)
        for n in (1980 if (m, r) == ("T2", 6000) else 2000,)}
    source = {"exp_id": "exp12", "mode": "saved", "status": "done", "saved": "output/長路徑/原檔.json",
              "saved_sha256": "a" * 64, "params": {"seed": 42}}
    before = deepcopy((result, source))
    answer = _exp12_answer(result, 6000, source)
    p = answer["presentation"]
    assert answer["text"] == markdown(p)
    assert (result, source) == before
    sections = {s["id"]: s for s in p["sections"]}
    assert sections["methods"]["rows"][1] == ["已知判對", "2736／2938＝93.12%", "0／2938＝0.00%"]
    assert len(sections["summary"]["items"]) <= 3
    assert all(not sections[k]["detail"] for k in ("source", "limits", "warnings"))
    assert all(sections[k]["detail"] for k in ("classes-ledoit_wolf", "tsne", "source_details", "matrix-binary"))
    assert sections["tsne"]["rows"][-1] == ["合計", 17980]
    assert p["source"] == source
    assert "a" * 64 in sections["source_details"]["text"]
    assert len(markdown_html(answer["text"]).split("<table>")) >= 7


@pytest.mark.parametrize("value,expected", [(None, "原檔未記錄"), ("壞時間", "格式無效"),
    ("2026-10-18T12:11:13", "2026-10-18T12:11:13（原檔未提供時區）"),
    ("2026-10-18T12:37:51.477Z", "2026-10-18 20:37:51.477"),
    ("2026-10-18T23:59:30.000+00:00", "2026-10-19 07:59:30.000")])
def test_fixed_timezone_and_legacy_values(value, expected):
    assert expected in display_time(value)


def test_generic_tables_escape_pipes_backticks_lines_and_html():
    lines = []
    label = "配置|`名稱`\n<script>不執行</script>"
    _render("資料", [{label: "a|b", "其他": "**純文字**"}], 0, lines)
    html = markdown_html("\n".join(lines))
    assert html.count("<th>") == 2 and html.count("<td>") == 2
    assert "配置|`名稱`\\n&lt;script&gt;" in html
    assert "<script>" not in html and "<strong>" not in html
    assert "<td>a|b</td>" in html


def test_focus_views_keep_warnings_source_and_full_view_opens_details():
    p = _exp12_answer(historical_result(), 6000, None)["presentation"]
    lw = select_view(p, "lw_classes")
    assert {s["id"] for s in lw["sections"]} == {"source", "warnings", "limits", "source_details", "classes-ledoit_wolf"}
    assert next(s for s in lw["sections"] if s["id"] == "classes-ledoit_wolf")["detail"] is False
    assert select_view(p, "tsne")["source_id"] == p["source_id"]
    assert all(s["open"] for s in select_view(p, "full")["sections"])
    assert not all(s["open"] for s in p["sections"])


def test_zero_missing_invalid_and_budget_cannot_hide_critical_warnings():
    result = {"openset_confusion": {"ledoit_wolf": {"labels": ["8screws"], "matrix": [[0]]}}}
    p = _exp12_answer(result, 6000, None)["presentation"]
    assert p["status"] == "incomplete"
    assert "不適用" in markdown(p) and "缺unknown" in markdown(p)
    bad = _exp12_answer({"value_kind": "normalized"}, 6000, None)
    assert bad["status"] == "invalid" and "核算失敗" in bad["text"]
    assert not next(s for s in bad["presentation"]["sections"] if s["kind"] == "warning")["detail"]
    limited = _exp12_answer(historical_result(), 150, None)
    assert limited["status"] == "incomplete" and "presentation" not in limited
