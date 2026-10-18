"""可信區段、SSE與DOM共用同一筆來源；不呼叫模型或讀data。"""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.test_assistant_results import OfflineAssistant, collect, result_snapshot
from web.presentation import markdown


@pytest.mark.parametrize("question,visible,hidden", [
    ("只看LW逐類結果", "classes-ledoit_wolf", "methods"),
    ("只看t-SNE圖例", "legend", "methods"),
    ("展開全部矩陣", "matrix-binary", None),
])
def test_question_focus_is_program_generated_and_keeps_same_source(question, visible, hidden):
    frames = collect(OfflineAssistant("unused"), question, context={"experiment": "exp12", "mode": "result"})
    p = next(f["presentation"] for f in frames if f["event"] == "presentation")
    ids = {s["id"] for s in p["sections"]}
    assert visible in ids and (hidden is None or hidden not in ids)
    assert {"source", "warnings", "limits", "source_details"} <= ids
    assert frames[0]["source_id"] == p["source_id"]
    assert frames[0]["presentation_source"] == p["source"]
    assert frames[-1]["text"] == markdown(p) and frames[-1]["origin"] == "matrix"
    if hidden is None:
        assert all(s["open"] for s in p["sections"])


def test_actual_presentation_dom_contract():
    frames = collect(OfflineAssistant("unused"), "實驗十二結果", snap=result_snapshot())
    p = next(f["presentation"] for f in frames if f["event"] == "presentation")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([shutil.which("node"), str(root / "src/tests/presentation_frontend.cjs"),
        str(root / "src/web/static/assistant_presentation.js")],
        input=json.dumps({"presentation": p, "binding": frames[0], "text": frames[-1]["text"]}),
        capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_chat_consumes_program_frames_without_duplicate_markdown():
    frames = collect(OfflineAssistant("unused"), "實驗十二結果")
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run([shutil.which("node"), str(root / "src/tests/ai_chat_frontend.cjs"),
        str(root / "src/web/static/markdown.js"), str(root / "src/web/static/ai_chat.js"), "--frames"],
        input=json.dumps([*frames, {"event": "done"}]), capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr


def test_narrow_layout_has_local_scroll_and_no_fixed_minimum_for_result():
    # CSS契約不是實際360px或200%畫面驗收。
    root = Path(__file__).resolve().parents[2]
    html = (root / "src/web/static/index.html").read_text(encoding="utf-8")
    assert html.index('/static/assistant_presentation.js') < html.index('/static/ai_chat.js')
    assert ".ai-table-scroll{max-width:100%; overflow-x:auto" in html
    assert ".ai-row,.ai-msg,.ai-md,.ai-presentation,.ai-section{min-width:0;}" in html
    assert "@media (max-width:420px){.ai-fields{grid-template-columns:minmax(0,1fr);}" in html
