"""CATALOG 文案對上實際方法與資料角色、首頁量尺用語，以及 experiments.js 的 LIVE 畫面涵蓋所有串流實驗。"""
import re
from pathlib import Path

from core.detectors import ALL_DETECTORS
from web.experiments import CATALOG


def test_catalog_matches_method_registry_and_roles():
    by_id = {entry["id"]: entry for entry in CATALOG}
    assert len(by_id) == 14
    assert str(len(ALL_DETECTORS)) in by_id["exp6"]["title"]
    assert "AutoEncoder" in by_id["exp6"]["fits"]
    for entry in CATALOG:
        assert entry["data_roles"] and entry["limitations"]
        assert "9 組全跑" not in entry["question"]
    assert "故障全池" in by_id["exp8_monitor"]["question"]
    assert "外層健康 holdout" in by_id["exp5"]["limitations"]


def test_home_scale_is_not_physical_health():
    text = (Path(__file__).resolve().parents[2] / "src/web/static/index.html").read_text(encoding="utf-8")
    assert "相對偏離量尺（展示）" in text
    assert "不是損壞百分比" in text
    assert 'id="healthPct"' in text


def test_every_streaming_experiment_has_live_view():
    """CATALOG 標 stream 的實驗，experiments.js 的 LIVE 一定要有對應畫面，否則按「邊跑邊畫」會拋例外。"""
    js = (Path(__file__).resolve().parents[2] / "src/web/static/experiments.js").read_text(encoding="utf-8")
    block = js[js.index("const LIVE = {"):]
    block = block[:block.index("\n};\n")]
    views = set(re.findall(r"^  (exp\w+): \{", block, re.M))
    streaming = {entry["id"] for entry in CATALOG if entry.get("stream")}
    assert streaming, "至少要有一個實驗支援邊跑邊畫"
    assert streaming <= views, f"缺少 LIVE 畫面：{sorted(streaming - views)}"
