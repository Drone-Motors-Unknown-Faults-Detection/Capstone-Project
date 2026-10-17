"""web/playback.py：用真正的 ExperimentRunner.stream() 產生 frame，確認播放狀態彙整與真值隔離。"""
import json
import tempfile
import urllib.parse
import unittest
from pathlib import Path

import tornado.testing

import web.server as server
from tests.test_navigation_guide import fixture
from web.experiments import ExperimentRunner
from web.playback import PlaybackTracker

# fixture 的故障配置；這些名稱只能出現在事後對照
TRUTH = ("5screws", "7screws", "6screws", "4screws", "螺絲", "truth", "expected", "majority", "is_fault", "onset")


class StreamedPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        folder = root / "data/Step-0/myfeature/T1/8000rpm"   # 與正式資料相同層級（實驗八依此搜尋）
        folder.mkdir(parents=True)
        cls.ds = fixture(folder)
        cls.runner = ExperimentRunner([cls.ds], str(root / "data"), out_dir=root / "out")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def play(self, exp_id, params):
        tracker = PlaybackTracker()
        tracker.start(exp_id, params)      # 與 StreamHandler 相同順序：start → 每個 frame update
        for frame, _units in self.runner.stream(exp_id, params):
            tracker.update(json.loads(json.dumps(frame)))   # 走過 JSON，與 SSE 送出的內容一致
        tracker.finish("stopped")
        return tracker.snapshot()

    def assertNoTruth(self, snap):
        visible = json.dumps([snap["current"], snap["events"], snap["params"]], ensure_ascii=False)
        for word in TRUTH:
            self.assertNotIn(word, visible)

    def test_exp2_self_learning(self):
        snap = self.play("exp2", {"dataset": "T1/8000rpm"})
        self.assertEqual(snap["status"], "done")   # finish("stopped") 不覆蓋已完成
        self.assertEqual((snap["no"], snap["title"]), ("實驗二", "量尺擴張"))
        self.assertGreater(snap["samples"], 0)
        self.assertEqual(set(snap["current"]), {"t", "score", "verdict", "predicted", "quarantine", "n_known"})
        self.assertTrue(any(k.startswith("discovered-") for k in snap["known"]))
        self.assertTrue(any(e.startswith("隔離區聚出候選群") for e in snap["events"]))
        self.assertTrue(any(e.startswith("已自學為") for e in snap["events"]))
        self.assertTrue(any("majority" in r for r in snap["reference"]))
        self.assertIn("final_known", snap["result"])
        self.assertTrue(any("stage_injected" in r for r in snap["reference"]))
        self.assertNoTruth(snap)

    def test_exp3_alarm_events_and_trial_reference(self):
        snap = self.play("exp3", {"dataset": "T1/8000rpm", "n_trials": 1})
        self.assertEqual(snap["status"], "done")
        self.assertEqual(snap["params"]["n_trials"], 1)
        self.assertIn("ewma", snap["current"])
        self.assertTrue(any("變化點警報" in e for e in snap["events"]))
        trials = [r for r in snap["reference"] if "trial" in r]
        self.assertEqual({r["scenario"] for r in trials}, {"A", "B"})
        self.assertTrue(all("expected" in r for r in trials))
        self.assertEqual({r["scenario_summary"] for r in snap["reference"] if "scenario_summary" in r}, {"A", "B"})
        self.assertNoTruth(snap)

    def test_exp1_rows_are_reference(self):
        snap = self.play("exp1", {"dataset": "T1/8000rpm"})
        self.assertEqual(len(snap["reference"]), 5)   # 健康 holdout + 4 個未知配置
        self.assertIn("macro_detect_rate", snap["result"])
        self.assertEqual(snap["current"]["segment"], 5)
        self.assertNoTruth(snap)

    def test_exp8_source_config_kept_out_of_params(self):
        snap = self.play("exp8_monitor", {"dataset": "T1/8000rpm", "config": "5screws", "max_windows": 10})
        self.assertNotIn("config", snap["params"])
        self.assertEqual(snap["source_config"], "5screws")
        self.assertTrue(snap["source_is_fault"])
        self.assertNotIn("is_fault", snap["current"])
        self.assertIn("health_index", snap["current"])
        self.assertEqual(snap["result"]["windows"], 10)
        self.assertNoTruth(snap)


class TrackerLifecycleTests(unittest.TestCase):
    def test_nothing_played(self):
        self.assertIsNone(PlaybackTracker().snapshot())

    def test_interrupted_is_stopped(self):
        tracker = PlaybackTracker()
        tracker.start("exp2", {"dataset": "T1/8000rpm"})
        tracker.update({"event": "tick", "t": 1, "score": 2.0, "verdict": "unknown", "label": None,
                        "quarantine": 1, "known": 1, "config": "7screws"})
        tracker.finish("stopped")
        snap = tracker.snapshot()
        self.assertEqual(snap["status"], "stopped")
        self.assertEqual(snap["samples"], 1)

    def test_error_frame_and_exp4_parts(self):
        tracker = PlaybackTracker()
        tracker.start("exp4", {"dataset": "T1/8000rpm"})
        tracker.update({"event": "part_start", "name": "geometry"})
        self.assertEqual(tracker.snapshot()["current"], {"part": "geometry", "state": "計算中"})
        tracker.update({"event": "error", "error": "exp4 執行失敗"})
        tracker.finish("stopped")
        snap = tracker.snapshot()
        self.assertEqual((snap["status"], snap["error"]), ("error", "exp4 執行失敗"))

    def test_snapshot_is_a_copy(self):
        tracker = PlaybackTracker()
        tracker.start("exp1", {})
        tracker.snapshot()["events"].append("外部改動")
        self.assertEqual(tracker.snapshot()["events"], [])


class StreamHandlerTrackingTests(tornado.testing.AsyncHTTPTestCase):
    """StreamHandler 播放時更新 server.PLAYBACK；參數錯誤與忙碌不覆蓋上一次播放。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        folder = root / "data/Step-0/myfeature/T1/8000rpm"
        folder.mkdir(parents=True)
        ds = fixture(folder)
        self.saved = (server.JOBS, server.PLAYBACK)
        server.JOBS = server.ExperimentJobs(ExperimentRunner([ds], str(root / "data"), out_dir=root / "out"))
        server.PLAYBACK = PlaybackTracker()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        server.JOBS.executor.shutdown(wait=True)
        server.JOBS, server.PLAYBACK = self.saved
        self.temp.cleanup()

    def get_app(self):
        return server.make_app()

    def stream(self, exp_id, params):
        query = urllib.parse.quote(json.dumps(params))
        return self.fetch(f"/api/experiments/{exp_id}/stream?params={query}&rate=1000", request_timeout=60)

    def test_stream_updates_playback_and_bad_request_keeps_it(self):
        self.assertEqual(self.stream("exp3", {"dataset": "T1/8000rpm", "n_trials": 1}).code, 200)
        snap = server.PLAYBACK.snapshot()
        self.assertEqual((snap["exp_id"], snap["status"]), ("exp3", "done"))
        self.stream("exp2", "not-an-object")                     # 參數錯誤：不開始新播放
        self.assertEqual(server.PLAYBACK.snapshot()["exp_id"], "exp3")
        server.JOBS.current = "exp1"                             # 忙碌：不開始新播放
        self.stream("exp2", {"dataset": "T1/8000rpm"})
        server.JOBS.current = None
        self.assertEqual(server.PLAYBACK.snapshot()["exp_id"], "exp3")


if __name__ == "__main__":
    unittest.main()
