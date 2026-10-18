"""8787驗收器的離線正反例；不連外、不讀data、不修改原始來源。"""

from copy import deepcopy
import pytest

from tests.assistant_presentation_evidence import source_receipt, verify_presentation
from tests.exp12_counts_fixture import historical_result
from tests.test_assistant_results import OfflineAssistant, collect
from web.playback import PlaybackTracker


def cases(question="實驗十二結果"):
    result = historical_result()
    result["_tsne_by_dataset"] = {"T1/8000": {"config_labels": ["8screws"], "verdicts": ["known_correct"], "embedding": [[0, 0]]}}
    saved = "output/web_server/session/experiments/exp12_2026-10-18-12-38-42.json"
    tracker = PlaybackTracker()
    tracker.record_run("exp12", {"params": {"seed": 42}, "saved": saved, "saved_sha256": "a" * 64, "result": result})
    frames = collect(OfflineAssistant("unused"), question, snap=tracker.snapshot())
    receipt = {"saved": saved, "sha256": "a" * 64, "result": result, "visualization": {"T1/8000": 1}}
    return [*frames, {"event": "done"}], receipt


@pytest.mark.parametrize("question,view", [("實驗十二結果", "default"), ("只看LW逐類結果", "lw_classes"),
    ("t-SNE實際抽樣", "tsne"), ("展開全部矩陣", "full")])
def test_program_frames_are_recomputable_from_same_receipt(question, view):
    frames, receipt = cases(question)
    assert verify_presentation(frames, receipt, view)["source"]["saved"] == receipt["saved"]


@pytest.mark.parametrize("tamper", ["source", "text", "count", "done", "origin"])
def test_tampered_frame_never_passes_acceptance(tamper):
    frames, receipt = cases()
    frames = deepcopy(frames)
    p = next(f["presentation"] for f in frames if f.get("event") == "presentation")
    if tamper == "source":
        p["source"]["saved_sha256"] = "b" * 64
    elif tamper == "text":
        next(f for f in frames if f["event"] == "delta")["text"] = "假成功"
    elif tamper == "count":
        next(s for s in p["sections"] if s["id"] == "methods")["rows"][1][1] = "錯誤數字"
    elif tamper == "origin":
        next(f for f in frames if f["event"] == "delta")["origin"] = "model"
    else:
        frames.pop()
    with pytest.raises(AssertionError):
        verify_presentation(frames, receipt)


def test_source_reader_rejects_escape_before_ssh(monkeypatch):
    monkeypatch.setattr("tests.assistant_presentation_evidence.ssh_python", lambda _: pytest.fail("不應連SSH"))
    with pytest.raises(ValueError):
        source_receipt("../secret.json")
