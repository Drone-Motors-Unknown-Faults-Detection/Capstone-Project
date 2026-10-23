"""experiments.navigation_regression：不一致要計數並留失敗紀錄，不靠 assert（#301 F-04）。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from experiments import navigation_regression
from experiments.navigation_regression import MismatchLedger, main, tick_with_row

DATASET = {"motor": "T1", "rpm": "8000rpm", "path": Path("data")}


def test_ledger_counts_every_mismatch_and_caps_details():
    ledger = MismatchLedger(limit=2)
    assert ledger.check("m/0", "score", 1.0, 1.0)
    for index in range(3):
        assert not ledger.check(f"m/{index}", "verdict", "known", "unknown")
    assert ledger.count == 3
    assert ledger.details == [{"pair_id": "m/0", "field": "verdict", "cli": "known", "guide": "unknown"},
                              {"pair_id": "m/1", "field": "verdict", "cli": "known", "guide": "unknown"}]


def test_tick_without_draw_raises_instead_of_assert():
    demo = SimpleNamespace(pools={}, samplers={})
    with pytest.raises(RuntimeError, match="1 筆"):
        tick_with_row(demo, lambda: None)


def _main(monkeypatch, fake_run):
    monkeypatch.setattr(navigation_regression, "discover_datasets", lambda root: [DATASET])
    monkeypatch.setattr(navigation_regression, "run", fake_run)
    try:
        return main(["--data-root", "data"])
    finally:
        logger.remove()


def _summary() -> dict:
    (path,) = Path("output/navigation_regression").glob("*/summary.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_mismatch_writes_failure_record_and_nonzero_exit(monkeypatch):
    def fake_run(dataset, seed):
        return {"schema": "navigation_regression_v3", "status": "FAILED", "seed": seed, "mismatches": 2,
                "source_unchanged": True,
                "paired_methods": [{"method": "knn", "mismatches": 2,
                                    "mismatch_details": [{"pair_id": "knn/3", "field": "score",
                                                          "cli": 1.0, "guide": 1.5}]}]}

    assert _main(monkeypatch, fake_run) == 1
    summary = _summary()
    assert summary["mismatches"] == 2
    assert summary["paired_methods"][0]["mismatch_details"][0]["field"] == "score"


def test_verified_run_exits_zero(monkeypatch):
    def fake_run(dataset, seed):
        return {"schema": "navigation_regression_v3", "status": "VERIFIED", "seed": seed, "mismatches": 0,
                "source_unchanged": True, "paired_methods": []}

    assert _main(monkeypatch, fake_run) == 0
    assert _summary()["status"] == "VERIFIED"


def test_crash_still_writes_failure_record(monkeypatch):
    def fake_run(dataset, seed):
        raise RuntimeError("一拍應抽 1 筆來源列，實際 0 筆")

    with pytest.raises(RuntimeError):
        _main(monkeypatch, fake_run)
    summary = _summary()
    assert summary["status"] == "FAILED"
    assert "RuntimeError" in summary["error"]
