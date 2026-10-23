"""導覽的來源、答案遮蔽、狀態與原計算一致性驗收。"""
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
import tornado.websocket
from tornado.testing import AsyncHTTPTestCase, gen_test

from core.data import HEALTHY
from core.feature_schema import SCHEMAS
from experiments.navigation_data_contract import run, split_audit
from experiments.exp2_scale_growth import ScaleGrowthSession
from web.guide import GuideHub, application
from web.live import LiveDemo


def fixture(root, healthy=50):
    rng = np.random.default_rng(42)
    ds = {"motor": "T1", "rpm": "8000rpm", "path": root}
    for c, n, shift in ((HEALTHY, healthy, 0), ("5screws", 60, 8),
                         ("7screws", 50, 2), ("6screws", 50, 4), ("4screws", 50, 10)):
        folder = root / c
        folder.mkdir()
        pd.DataFrame(rng.normal(size=(n, 105)) + shift, columns=SCHEMAS["historical_clean_v1"]).to_csv(
            folder / "fixture_Group_feature_data_clean.csv", index=False)
    return ds


def _force_candidate(original):
    """包住 process()：這一筆之後隔離區直接放 25 筆該配置列並成群，不跑 HDBSCAN。"""
    def process(session, sample, truth):
        out = original(session, sample, truth)
        rows = session.monitor.pools[truth][:25]
        session.quarantine_X = [row.copy() for row in rows]
        session.quarantine_truth = [truth] * 25
        session.quarantine_t = list(range(1, 26))
        session.candidate = {"indices": list(range(25)), "size": 25, "majority": truth,
                             "majority_display": truth, "purity": 1.0, "t_range": [1, 25], "cluster_params": [25, 3]}
        out["candidate_new"] = True
        return out
    return process


class Client:
    def __init__(self):
        self.messages = []

    def write_message(self, raw):
        self.messages.append(json.loads(raw))


class GuideTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ds = fixture(Path(self.temp.name))
        self.hub = GuideHub([self.ds])

    def tearDown(self):
        if self.hub.demo:
            self.hub.demo.close()
        self.hub.executor.shutdown()
        self.temp.cleanup()

    def test_no_initial_fit(self):
        self.assertIsNone(self.hub.demo)
        self.assertEqual(self.hub.full_state()["step"], "待建基準")

    def test_inspection_has_no_fit_and_no_unknown_name(self):
        self.hub.inspect("T1", "8000rpm")
        self.assertIsNone(self.hub.demo)
        self.assertNotIn("5screws", json.dumps(self.hub.full_state()))

    def test_invalid_dataset(self):
        with self.assertRaises(ValueError):
            self.hub.build("T2", "8000rpm")

    def test_small_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            ds = fixture(Path(folder), healthy=4)
            with self.assertRaises(ValueError):
                run(ds)

    def test_missing_healthy(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run({"path": Path(folder), "motor": "T1", "rpm": "8000rpm"})

    def test_zero_healthy_rows_cannot_build(self):
        with tempfile.TemporaryDirectory() as folder:
            ds = fixture(Path(folder), healthy=0)
            with self.assertRaises(ValueError):
                run(ds)

    def test_empty_catalog_has_explicit_reason_without_fit(self):
        hub = GuideHub([])
        try:
            self.assertEqual(hub.full_state()["catalog_status"], "no_datasets")
            self.assertIsNone(hub.demo)
        finally:
            hub.close()

    def test_only_healthy_fit(self):
        self.hub.build("T1", "8000rpm")
        audit = split_audit(self.hub.demo)
        self.assertEqual(list(audit["roles"]), [HEALTHY])
        roles = audit["roles"][HEALTHY]
        self.assertFalse(set(roles["train"]) & set(roles["calibration"]))
        self.assertFalse(set(roles["holdout"]) & set(roles["train"]))

    def test_anonymous_adoption_and_sample(self):
        self.hub.build("T1", "8000rpm")
        self.hub.demo.adoption = {"t": 40, "action": "learned", "config": "discovered-1", "display": "模型新類 1",
                                  "matched": "5screws", "matched_display": "5 螺絲", "size": 25, "purity": 1.0,
                                  "source": "cluster_only", "median_score": 30.0, "known_cal_max": 1.1, "label": 1,
                                  "pruned": 0}
        state = self.hub.full_state()
        self.assertEqual(set(state["adoption"]), {"t", "action", "config", "display", "size", "median_score", "known_cal_max"})
        self.assertNotIn("5screws", json.dumps(state))
        self.assertNotIn("purity", json.dumps(state))
        client = Client()
        self.hub.clients.add(client)
        self.hub.broadcast({"type": "sample", "truth": "5screws", "truth_display": "答案",
                            "health": 5, "cls": "答案", "verdict": "unknown", "score": 3})
        self.assertNotIn("truth", json.dumps(client.messages))
        self.assertNotIn("health", client.messages[0])
        self.hub.broadcast({"type": "event", "text": "5screws 純度100%"})
        self.assertNotIn("5screws", json.dumps(client.messages))

    def test_last_disconnect_pauses_without_refit(self):
        self.hub.build("T1", "8000rpm")
        demo = self.hub.demo
        c = Client()
        self.hub.clients.add(c)
        self.hub.running = True
        self.hub.close_client(c)
        self.assertFalse(self.hub.running)
        self.assertIs(demo, self.hub.demo)

    def test_pause_flushes_short_log_without_refit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.hub.out_dir = Path(folder)
            self.hub.build("T1", "8000rpm")
            demo = self.hub.demo
            for _ in range(3):
                demo.tick()
            self.hub.pause()
            with (Path(folder) / "session001/samples_epoch01.csv").open(encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 3)
            self.assertIs(self.hub.demo, demo)
            self.assertEqual(demo.t, 3)
            self.hub.demo.close()

    def test_stream_error_blocks_continuation(self):
        self.hub.build("T1", "8000rpm")
        self.hub.event("error", "串流錯誤，已暫停：測試")
        self.assertEqual(self.hub.full_state()["step"], "錯誤")
        self.assertFalse(self.hub.running)

    def test_busy_does_not_read_half_updated_model(self):
        self.hub.build("T1", "8000rpm")
        self.hub.busy_state = self.hub.full_state()
        self.hub.busy, self.hub.operation = True, "reset"
        with patch.object(self.hub.demo.session.monitor, "summary",
                          side_effect=AssertionError("不可讀取更新中的模型")):
            self.hub.tick()
            self.assertEqual(self.hub.full_state()["step"], "更新中")

    def test_same_source_seed_prediction_and_pause(self):
        self.hub.build("T1", "8000rpm")
        original = LiveDemo(self.ds)
        client = Client()
        self.hub.clients.add(client)
        self.hub.running = True
        for _ in range(12):
            expected = next(m for m in original.tick() if m["type"] == "sample")
            self.hub.tick()
            actual = [m for m in client.messages if m["type"] == "sample"][-1]
            self.assertEqual((expected["score"], expected["verdict"], expected["ewma"]),
                             (actual["score"], actual["verdict"], actual["ewma"]))
        self.hub.running = False
        self.hub.tick()
        self.assertEqual(self.hub.demo.t, 12)
        original.close()

    def test_paused_tick_does_not_broadcast(self):
        # 暫停時每拍不再送完整狀態；開始串流後每拍都有 sample 與 state（#274）
        self.hub.build("T1", "8000rpm")
        client = Client()
        self.hub.clients.add(client)
        for _ in range(3):
            self.hub.tick()
        self.assertEqual(client.messages, [])
        self.hub.running = True
        self.hub.tick()
        self.assertEqual([m["type"] for m in client.messages][-2:], ["sample", "state"])

    def test_scenario_stops_without_auto_confirmation(self):
        self.hub.build("T1", "8000rpm")
        self.hub.demo.start_scenario("A")
        self.assertIn("劇本A", self.hub.full_state()["source"])
        self.hub.running, self.hub.scenario_end = True, 3
        for _ in range(3):
            self.hub.tick()
        self.assertFalse(self.hub.running)
        self.assertEqual(list(self.hub.demo.session.monitor.known), [HEALTHY])
        self.assertTrue(self.hub.full_state()["scenario_completed"])

    def test_tick_adopts_candidate_and_masks_truth(self):
        # #287：成群當拍自學，導覽狀態只公開類名／筆數／分數，不公開真值多數與純度
        with tempfile.TemporaryDirectory() as folder:
            self.hub.out_dir = Path(folder)
            self.hub.build("T1", "8000rpm")
            self.hub.demo.set_source("5screws")
            client = Client()
            self.hub.clients.add(client)
            with patch.object(ScaleGrowthSession, "process", _force_candidate(ScaleGrowthSession.process)):
                self.hub.running = True
                self.hub.tick()
            state = self.hub.full_state()
            self.assertEqual(state["adoption"]["action"], "learned")
            self.assertEqual(state["adoption"]["config"], "discovered-1")
            self.assertEqual(sorted(state["adoption"]), ["action", "config", "display", "known_cal_max",
                                                         "median_score", "size", "t"])
            self.assertEqual(state["known"], [HEALTHY, "discovered-1"])
            self.assertEqual(self.hub.last_action, "learned")
            self.assertTrue((Path(folder) / "session001/fit_epoch1_classes2.json").exists())
            everything = json.dumps([state] + client.messages, ensure_ascii=False)
            self.assertNotIn("5screws", everything)
            self.assertNotIn("純度", everything)
            self.hub.running = False
            self.assertEqual(self.hub.full_state()["step"], "已更新")
            self.hub.demo.close()

    def test_new_session_retains_old_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            self.hub.out_dir = Path(folder)
            self.hub.build("T1", "8000rpm")
            self.hub.build("T1", "8000rpm")
            self.assertTrue((Path(folder) / "session001/data_contract.json").exists())
            self.assertTrue((Path(folder) / "session002/data_contract.json").exists())
            self.hub.demo.close()


class GuideSocketTests(AsyncHTTPTestCase):
    def get_app(self):
        self.temp = tempfile.TemporaryDirectory()
        self.hub = GuideHub([fixture(Path(self.temp.name))])
        return application(self.hub)

    def tearDown(self):
        if self.hub.demo:
            self.hub.demo.close()
        self.hub.executor.shutdown()
        self.temp.cleanup()
        super().tearDown()

    @gen_test
    async def test_unfitted_and_confirm_command_rejected(self):
        # #287：導覽沒有操作員確認；confirm 指令視為不支援
        ws = await tornado.websocket.websocket_connect(self.get_url("/ws").replace("http", "ws", 1))
        await ws.read_message()
        ws.write_message(json.dumps({"cmd": "start"}))
        error = json.loads(await ws.read_message())
        self.assertIn("請先", error["text"])
        await ws.read_message()
        self.hub.build("T1", "8000rpm")
        ws.write_message(json.dumps({"cmd": "confirm", "candidate_id": "fake"}))
        error = json.loads(await ws.read_message())
        self.assertIn("不支援", error["text"])
        ws.close()

    @gen_test
    async def test_invalid_arguments_not_echoed(self):
        # 用戶端字串不回送；非 CommandError 只回型別名（#279）
        payload = "<img src=x onerror=alert(1)>"
        ws = await tornado.websocket.websocket_connect(self.get_url("/ws").replace("http", "ws", 1))
        await ws.read_message()
        self.hub.build("T1", "8000rpm")
        for command in ({"cmd": "build", "motor": payload, "rpm": "x"}, {"cmd": "rate", "value": payload},
                        {"cmd": "source", "id": payload}, {"cmd": "scenario", "name": payload},
                        {"cmd": payload}, [payload]):
            with self.subTest(command=command):
                ws.write_message(json.dumps(command))
                error = json.loads(await ws.read_message())
                self.assertEqual(error["level"], "error")
                self.assertNotIn(payload, json.dumps(error, ensure_ascii=False))
                state = json.loads(await ws.read_message())
                self.assertFalse(state["running"])
        self.assertEqual(self.hub.rate, 4)
        ws.close()

    @gen_test
    async def test_busy_rejects_duplicate(self):
        ws = await tornado.websocket.websocket_connect(self.get_url("/ws").replace("http", "ws", 1))
        await ws.read_message()
        self.hub.busy = True
        ws.write_message(json.dumps({"cmd": "build", "motor": "T1", "rpm": "8000rpm"}))
        error = json.loads(await ws.read_message())
        self.assertIn("拒絕重複", error["text"])
        self.assertIsNone(self.hub.demo)
        ws.close()


class MountedGuideTests(AsyncHTTPTestCase):
    def get_app(self):
        from web import server
        self.temp = tempfile.TemporaryDirectory()
        self.ds = fixture(Path(self.temp.name))
        self.previous = server.HUB
        self.home = server.Hub(LiveDemo(self.ds), [self.ds], 4, 42)
        server.HUB = self.home
        self.guide = GuideHub([self.ds])
        return server.make_app(guide_hub=self.guide)

    def tearDown(self):
        from web import server
        self.guide.close()
        self.home.close()
        server.HUB = self.previous
        self.temp.cleanup()
        super().tearDown()

    def test_home_mount_assets_docs_and_no_implicit_fit(self):
        self.assertIn('href="/guide/"', self.fetch("/").body.decode())
        page = self.fetch("/guide/").body.decode()
        self.assertIn('data-guide-prefix="/guide"', page)
        self.assertIn('href="/guide/assets/guide.css"', page)
        self.assertIn('src="/guide/assets/guide.js"', page)
        self.assertIn('href="/guide/docs/exp13_實驗總覽.md"', page)
        self.assertEqual(self.fetch("/guide/assets/guide.js").code, 200)
        self.assertEqual(self.fetch("/guide/docs/exp13_README.md").code, 200)
        self.assertEqual(self.fetch("/guide/docs/not_found.md").code, 404)
        self.assertEqual(self.fetch("/guide/docs/../AGENT.md").code, 403)
        self.assertIsNone(self.guide.demo)
        self.assertEqual(self.home.demo.t, 0)

    def test_default_factory_uses_independent_lazy_hub_and_same_configuration(self):
        from web import server
        self.home.out_dir = Path(self.temp.name) / "main_output"
        app = server.make_app()
        mounted = app.settings["guide_hub"]
        try:
            self.assertIsNot(mounted, self.home)
            self.assertIsNone(mounted.demo)
            self.assertIsNot(mounted.executor, self.home.executor)
            self.assertEqual(mounted.datasets, self.home.datasets)
            self.assertEqual(mounted.seed, self.home.seed)
            self.assertEqual(mounted.detector_options, self.home.detector_options)
            self.assertEqual(mounted.out_dir, self.home.out_dir / "guide")
            mounted.build("T1", "8000rpm")
            self.assertIsNot(mounted.demo.stream_rng, self.home.demo.stream_rng)
            self.assertIsNot(mounted.demo.session.monitor, self.home.demo.session.monitor)
            self.assertEqual(self.home.demo.t, 0)
        finally:
            mounted.close()

    @gen_test
    async def test_mounted_socket_is_not_home_session_and_disconnect_pauses(self):
        guide = await tornado.websocket.websocket_connect(self.get_url("/guide/ws").replace("http", "ws", 1))
        state = json.loads(await guide.read_message())
        self.assertEqual(state["session_id"], 0)
        self.assertNotIn("5screws", json.dumps(state))
        self.guide.build("T1", "8000rpm")
        self.assertIsNot(self.guide.demo, self.home.demo)
        self.assertIsNot(self.guide.executor, self.home.executor)
        self.guide.running = True
        guide.close()
        await tornado.gen.sleep(.05)
        self.assertFalse(self.guide.running)
        self.assertEqual(self.home.demo.t, 0)
        self.assertEqual(list(self.home.demo.session.monitor.known), [HEALTHY])

    @gen_test
    async def test_cross_origin_guide_socket_remains_rejected(self):
        from tornado.httpclient import HTTPRequest, HTTPClientError
        request = HTTPRequest(self.get_url("/guide/ws").replace("http", "ws", 1),
                              headers={"Origin": "https://not-this-site.invalid"})
        with self.assertRaises(HTTPClientError) as error:
            await tornado.websocket.websocket_connect(request)
        self.assertEqual(error.exception.code, 403)

    def test_shutdown_closes_both_hubs_even_after_one_flush_failure(self):
        from web.server import serve
        from unittest.mock import Mock
        loop, first, second = Mock(), Mock(), Mock()
        first.close.side_effect = OSError("測試寫入失敗")
        with self.assertRaises(OSError):
            serve(loop, first, extra_hubs=(second,))
        second.close.assert_called_once_with()
