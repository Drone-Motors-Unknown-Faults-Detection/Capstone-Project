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
from experiments.navigation_data_contract import run, split_audit
from web.guide import GuideHub, application
from web.live import LiveDemo


def fixture(root, healthy=50):
    rng = np.random.default_rng(42)
    ds = {"motor": "T1", "rpm": "8000rpm", "path": root}
    for c, n, shift in ((HEALTHY, healthy, 0), ("5screws", 60, 8),
                         ("7screws", 50, 2), ("6screws", 50, 4), ("4screws", 50, 10)):
        folder = root / c
        folder.mkdir()
        pd.DataFrame(rng.normal(size=(n, 105)) + shift).to_csv(
            folder / "fixture_Group_feature_data_clean.csv", index=False)
    return ds


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

    def test_anonymous_candidate_and_sample(self):
        self.hub.build("T1", "8000rpm")
        self.hub.demo.session.candidate = {"size": 25, "majority": "5screws", "purity": 1}
        state = self.hub.full_state()
        self.assertEqual(set(state["candidate"]), {"id", "size", "t_range"})
        self.assertNotIn("5screws", json.dumps(state))
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

    def test_candidate_identity_uses_window_support(self):
        self.hub.build("T1", "8000rpm")
        self.hub.demo.session.candidate = {"size": 25, "indices": [0, 1], "t_range": [30, 40]}
        first = self.hub.candidate_id()
        self.hub.demo.session.candidate = {"size": 25, "indices": [0, 1], "t_range": [60, 70]}
        self.assertNotEqual(first, self.hub.candidate_id())

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
        self.hub.busy, self.hub.operation = True, "confirm"
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
    async def test_unfitted_and_stale_confirmation_rejected(self):
        ws = await tornado.websocket.websocket_connect(self.get_url("/ws").replace("http", "ws", 1))
        await ws.read_message()
        ws.write_message(json.dumps({"cmd": "start"}))
        error = json.loads(await ws.read_message())
        self.assertIn("請先", error["text"])
        await ws.read_message()
        self.hub.build("T1", "8000rpm")
        ws.write_message(json.dumps({"cmd": "confirm", "candidate_id": "fake"}))
        error = json.loads(await ws.read_message())
        self.assertIn("候選", error["text"])
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
