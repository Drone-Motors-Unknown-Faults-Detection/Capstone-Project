"""首頁即時展示 LiveDemo：候選成群當拍走與實驗二相同的 adopt()，沒有操作員確認（#287）。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.data import HEALTHY
from experiments.exp2_scale_growth import ScaleGrowthSession
from tests.test_navigation_guide import _force_candidate, fixture
from web.live import LiveDemo
from web.server import WSHandler


class AutoAdoptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.out = root / "out"
        self.out.mkdir()
        self.demo = LiveDemo(fixture(root), out_dir=self.out)

    def tearDown(self):
        self.demo.close()
        self.temp.cleanup()

    def test_confirm_path_is_gone(self):
        self.assertFalse(hasattr(LiveDemo, "confirm"))
        self.assertFalse(hasattr(ScaleGrowthSession, "confirm"))
        self.assertFalse(hasattr(WSHandler, "_do_confirm"))
        state = self.demo.state()
        self.assertIsNone(state["adoption"])
        self.assertNotIn("candidate", state)
        self.assertEqual(state["phase"], "monitoring")

    def test_candidate_is_adopted_in_the_same_tick(self):
        self.demo.set_source("5screws")
        with patch.object(ScaleGrowthSession, "process", _force_candidate(ScaleGrowthSession.process)):
            msgs = self.demo.tick()
        texts = [m["text"] for m in msgs if m["type"] == "event"]
        self.assertTrue(any("不等操作員" in t for t in texts))
        self.assertTrue(any(t.startswith("✔ 自動納入為「模型新類 1」") for t in texts))
        self.assertTrue(any(t.startswith("事後對照") and "5screws" in t for t in texts))
        self.assertEqual(msgs[-1]["type"], "state_patch")
        # 模型只多了 discovered-1；真值配置名沒有進模型，只留在事後對照的對應表
        self.assertEqual(list(self.demo.session.monitor.known), [HEALTHY, "discovered-1"])
        adoption = self.demo.state()["adoption"]
        self.assertEqual((adoption["action"], adoption["config"], adoption["t"]), ("learned", "discovered-1", 1))
        self.assertIsNone(self.demo.session.candidate)
        self.assertIn(self.demo.phase(), ("monitoring", "accumulating"))
        self.assertTrue((self.out / "model_epoch01_02classes.json").exists())
        json.dumps(self.demo.state(), ensure_ascii=False)   # 狀態可序列化

    def test_metrics_split_by_model_expansion_not_by_truth(self):
        # #296 R3-01：學到第一個新類之後，所有來源一律算「擴張後」；分段不讀真值
        self.demo.set_source("5screws")
        self.demo.tick()
        with patch.object(ScaleGrowthSession, "process", _force_candidate(ScaleGrowthSession.process)):
            self.demo.tick()
        self.demo.tick()
        self.demo.set_source("6screws")
        for _ in range(3):
            self.demo.tick()
        rows = {r["config"]: r for r in self.demo.metrics_msg()["rows"]}
        self.assertEqual((rows["5screws"]["pre_streamed"], rows["5screws"]["post_streamed"]), (2, 1))
        self.assertEqual((rows["6screws"]["pre_streamed"], rows["6screws"]["post_streamed"]), (0, 3))
        self.assertFalse(next(c for c in self.demo.state()["configs"] if c["config"] == "5screws")["known"])

    def test_fitted_rows_are_not_streamed_again_after_learning(self):
        # #296 R3-02：discovered-N 的 train／cal 列不再重抽，學後認出率只來自未擬合的列
        self.demo.set_source("5screws")
        with patch.object(ScaleGrowthSession, "process", _force_candidate(ScaleGrowthSession.process)):
            self.demo.tick()
        fitted = self.demo.fitted_rows["5screws"]
        split = self.demo.session.monitor.splits["discovered-1"]
        self.assertEqual(len(fitted), len(split.train) + len(split.cal))
        self.assertEqual(set(self.demo.samplers["5screws"].indices.tolist()) & fitted, set())
        self.assertEqual(len(self.demo.samplers["5screws"].indices), 60 - len(fitted))
        drawn = set()
        for _ in range(100):
            self.demo.tick()
            drawn.add(self.demo.samplers["5screws"].last_index)
        self.assertEqual(drawn & fitted, set())
        self.assertEqual(len(drawn), 60 - len(fitted))   # 100 筆循環把未擬合列都抽過一輪
        self.demo._build()
        self.assertEqual(self.demo.fitted_rows, {})
        self.assertEqual(len(self.demo.samplers["5screws"].indices), 60)

    def test_healthy_fitted_rows_excluded_and_exhausted_pool_keeps_streaming(self):
        # #299：健康列被併進 discovered-N 也要從健康 holdout 排除；整池擬合用完時不讓串流中斷
        holdout = self.demo.session.monitor.splits[HEALTHY].holdout
        self.demo.fitted_rows[HEALTHY] = {int(holdout[0])}
        self.demo.fitted_rows["5screws"] = set(range(len(self.demo.pools["5screws"])))
        self.demo._refresh_samplers()
        self.assertNotIn(int(holdout[0]), self.demo.samplers[HEALTHY].indices.tolist())
        self.assertEqual(len(self.demo.samplers["5screws"].indices), len(self.demo.pools["5screws"]))
        self.demo.set_source("5screws")
        self.demo.tick()

    def test_reset_clears_adoption(self):
        self.demo.set_source("5screws")
        with patch.object(ScaleGrowthSession, "process", _force_candidate(ScaleGrowthSession.process)):
            self.demo.tick()
        self.demo._build()
        self.assertIsNone(self.demo.adoption)
        self.assertEqual(list(self.demo.session.monitor.known), [HEALTHY])


if __name__ == "__main__":
    unittest.main()
