"""web/playback.py：用真正的 ExperimentRunner 產生 frame／結果，確認紀錄彙整、存檔讀取與真值隔離。"""
import json
import tempfile
import urllib.parse
import unittest
from pathlib import Path

import tornado.testing

import web.server as server
from tests.test_navigation_guide import fixture
from web.experiments import ExperimentRunner
from web.playback import PlaybackTracker, summarize_result

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
        snap = tracker.snapshot()
        self.assertEqual(snap["latest"], exp_id)
        return snap["records"][exp_id]

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
        self.assertTrue(any(k.startswith("discovered-") for k in snap["known"]))
        self.assertIn("stages：", snap["result_text"])
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
        self.assertIn("macro_detect_rate：", snap["result_text"])
        self.assertNotIn("resolved_config", snap["result_text"])   # 模型細節不進摘要
        self.assertEqual(snap["current"]["segment"], 5)
        self.assertNoTruth(snap)

    def test_exp8_source_config_kept_out_of_params(self):
        snap = self.play("exp8_monitor", {"dataset": "T1/8000rpm", "config": "5screws", "max_windows": 10})
        self.assertNotIn("config", snap["params"])
        self.assertEqual(snap["source_config"], "5screws")
        self.assertTrue(snap["source_is_fault"])
        self.assertNotIn("is_fault", snap["current"])
        self.assertIn("health_index", snap["current"])
        self.assertIn("（10 列）", snap["result_text"])
        self.assertNoTruth(snap)


class TrackerLifecycleTests(unittest.TestCase):
    def test_nothing_played(self):
        tracker = PlaybackTracker(results_root="不存在的目錄")
        self.assertIsNone(tracker.snapshot())
        self.assertIsNone(tracker.load_saved("exp12"))

    def test_interrupted_is_stopped(self):
        tracker = PlaybackTracker()
        tracker.start("exp2", {"dataset": "T1/8000rpm"})
        tracker.update({"event": "tick", "t": 1, "score": 2.0, "verdict": "unknown", "label": None,
                        "quarantine": 1, "known": 1, "config": "7screws"})
        tracker.finish("stopped")
        snap = tracker.snapshot()["records"]["exp2"]
        self.assertEqual(snap["status"], "stopped")
        self.assertEqual(snap["samples"], 1)

    def test_error_frame_and_exp4_parts(self):
        tracker = PlaybackTracker()
        tracker.start("exp4", {"dataset": "T1/8000rpm"})
        tracker.update({"event": "part_start", "name": "geometry"})
        self.assertEqual(tracker.snapshot()["records"]["exp4"]["current"], {"part": "geometry", "state": "計算中"})
        tracker.update({"event": "error", "error": "exp4 執行失敗"})
        tracker.finish("stopped")
        snap = tracker.snapshot()["records"]["exp4"]
        self.assertEqual((snap["status"], snap["error"]), ("error", "exp4 執行失敗"))

    def test_snapshot_is_a_copy(self):
        tracker = PlaybackTracker()
        tracker.start("exp1", {})
        tracker.snapshot()["records"]["exp1"]["events"].append("外部改動")
        self.assertEqual(tracker.snapshot()["records"]["exp1"]["events"], [])

    def test_each_experiment_keeps_its_own_latest_record(self):
        tracker = PlaybackTracker()
        tracker.record_run("exp12", {"params": {"seed": 42}, "seconds": 3, "result": {"seed": 42}})
        tracker.start("exp2", {"dataset": "T1/8000rpm"})
        snap = tracker.snapshot()
        self.assertEqual(snap["latest"], "exp2")
        self.assertEqual(snap["records"]["exp12"]["mode"], "run")
        self.assertEqual(snap["records"]["exp12"]["status"], "done")
        self.assertGreater(snap["records"]["exp2"]["seq"], snap["records"]["exp12"]["seq"])

    def test_run_error_recorded(self):
        tracker = PlaybackTracker()
        tracker.record_run_error("exp5", {"dataset": "T1/8000rpm"}, "exp5 執行失敗：ValueError()")
        rec = tracker.snapshot()["records"]["exp5"]
        self.assertEqual((rec["mode"], rec["status"]), ("run", "error"))


class SavedResultTests(unittest.TestCase):
    """伺服器重啟後記憶體沒有紀錄：讀 output/web_server/*/experiments/ 最新的存檔。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, session, exp_id, ts, seed):
        folder = self.root / session / "experiments"
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"experiment": exp_id, "params": {"seed": seed}, "seconds": 1.5,
                   "finished_at": ts.replace("-", ":"), "result": {"seed": seed, "binary_confusion": {
                       "labels": ["healthy", "fault"], "matrix": [[527, 0], [0, 2736]]}}}
        (folder / f"{exp_id}_{ts}.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_newest_saved_file_across_sessions(self):
        self.save("2026-10-17-14-08-37", "exp12", "2026-10-17-14-17-08", 42)
        self.save("2026-10-17-09-00-00", "exp12", "2026-10-17-09-10-00", 7)
        self.save("2026-10-17-14-08-37", "exp1", "2026-10-17-15-00-00", 1)
        self.save("2026-10-17-14-08-37", "exp12_extra", "2026-10-17-16-00-00", 3)   # 不同實驗 id，不能混入
        tracker = PlaybackTracker(results_root=self.root)
        rec = tracker.load_saved("exp12")
        self.assertEqual(rec["mode"], "saved")
        self.assertEqual(rec["params"], {"seed": 42})
        self.assertIn("exp12_2026-10-17-14-17-08.json", rec["result"]["saved"])
        self.assertIn("| healthy | 527 | 0 |", rec["result_text"])
        self.assertEqual(tracker.load_saved(None)["exp_id"], "exp12_extra")   # 不指定實驗：全部中最新

    def test_broken_file_skipped(self):
        self.save("s1", "exp12", "2026-10-17-14-17-08", 42)
        (self.root / "s1/experiments/exp12_2026-10-17-20-00-00.json").write_text("{壞掉", encoding="utf-8")
        self.assertEqual(PlaybackTracker(results_root=self.root).load_saved("exp12")["params"], {"seed": 42})


class SummarizeResultTests(unittest.TestCase):
    def test_matrix_table_and_internal_keys_dropped(self):
        text = summarize_result({
            "seed": 42, "perplexity": 30.0,
            "openset_confusion": {"ledoit_wolf": {"labels": ["8screws", "unknown"], "matrix": [[525, 0], [0, 2208]]}},
            "_tsne_by_dataset": {"T1/8000": {"x": list(range(5000))}},
            "model": {"resolved_config": {"version": "v1"}},
        })
        self.assertIn("ledoit_wolf：（列 = 真值、欄 = 預測）", text)
        self.assertIn("| 8screws | 525 | 0 |", text)
        self.assertNotIn("tsne", text)
        self.assertNotIn("resolved_config", text)
        self.assertIn("perplexity：30", text)

    def test_long_lists_and_budget(self):
        text = summarize_result({"scores": [float(i) for i in range(100)],
                                 "rows": [{"i": i, "v": i * 0.5} for i in range(45)]})
        self.assertIn("scores：100 個數值，最小 0、最大 99、平均 49.5", text)
        self.assertIn("…另有 15 列", text)
        self.assertTrue(summarize_result({"rows": [{"x": "y" * 300} for _ in range(30)]}, budget=500)
                        .endswith("…（以下略，完整結果見存檔）"))


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
        self.assertEqual((snap["latest"], snap["records"]["exp3"]["status"]), ("exp3", "done"))
        self.stream("exp2", "not-an-object")                     # 參數錯誤：不開始新播放
        self.assertEqual(server.PLAYBACK.snapshot()["latest"], "exp3")
        server.JOBS.current = "exp1"                             # 忙碌：不開始新播放
        self.stream("exp2", {"dataset": "T1/8000rpm"})
        server.JOBS.current = None
        self.assertEqual(set(server.PLAYBACK.snapshot()["records"]), {"exp3"})

    def test_run_button_recorded(self):
        resp = self.fetch("/api/experiments/exp12/run", method="POST", body=json.dumps({"seed": 42}),
                          request_timeout=120)
        self.assertEqual(resp.code, 200)
        rec = server.PLAYBACK.snapshot()["records"]["exp12"]
        self.assertEqual((rec["mode"], rec["status"], rec["params"]), ("run", "done", {"seed": 42}))
        self.assertIn("binary_confusion：（列 = 真值、欄 = 預測）", rec["result_text"])
        self.assertIn(json.loads(resp.body)["saved"], rec["result"]["saved"])


if __name__ == "__main__":
    unittest.main()
