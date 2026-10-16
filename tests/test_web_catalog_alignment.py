"""展示文案對上實際方法及資料角色，不變更演算法。"""
from pathlib import Path

from core.detectors import ALL_DETECTORS
from web.experiments import CATALOG


def test_catalog_matches_method_registry_and_roles():
    by_id = {entry["id"]: entry for entry in CATALOG}
    assert len(by_id) == 10
    assert str(len(ALL_DETECTORS)) in by_id["exp6"]["title"]
    assert "AutoEncoder" in by_id["exp6"]["fits"]
    for entry in CATALOG:
        assert entry["data_roles"] and entry["limitations"]
        assert "9 組全跑" not in entry["question"]
    assert "故障全池" in by_id["exp8_monitor"]["question"]
    assert "用途重疊" in by_id["exp5"]["limitations"]


def test_home_scale_is_not_physical_health():
    text = (Path(__file__).resolve().parents[1] / "web/static/index.html").read_text(encoding="utf-8")
    assert "相對偏離量尺（展示）" in text
    assert "不是損壞百分比" in text
    assert 'id="healthPct"' in text
