"""暫停、斷線及正常結束的實際 CSV 保存規範。"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.test_navigation_guide import fixture, Client
from web.guide import main as guide_main
from web.live import LiveDemo
from web.server import Hub, WSHandler, main as server_main, serve


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.ds = fixture(root)
        self.out = root / "output"
        self.out.mkdir()
        self.demo = LiveDemo(self.ds, out_dir=self.out)
        self.hub = Hub(self.demo, [self.ds], 4, 42, out_dir=self.out)

    def tearDown(self):
        self.hub.executor.shutdown(wait=True)
        self.demo._write_error = None
        self.demo.close()
        self.temp.cleanup()

    def rows(self):
        with (self.out / "samples_epoch01.csv").open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def ticks(self, n):
        self.hub.running = True
        for _ in range(n):
            self.hub.tick()

    def test_buffer_boundaries_pause_resume_without_duplicates(self):
        for target in (3, 14, 15, 16, 24, 30, 31):
            with self.subTest(target=target):
                self.ticks(target - self.demo.t)
                self.hub.pause()
                self.hub.pause()
                self.hub.tick()
                rows = self.rows()
                self.assertEqual([int(row["t"]) for row in rows], list(range(1, target + 1)))
                self.assertEqual(self.demo.flushed_t, target)

    def test_pause_queued_during_synchronous_tick(self):
        # IOLoop 不會在同步 process 中插入另一個 callback；以明確佇列模擬。
        callbacks = []
        original = self.demo.session.process
        def process(*args):
            callbacks.append(self.hub.pause)
            return original(*args)
        with patch.object(self.demo.session, "process", side_effect=process):
            self.ticks(1)
        callbacks.pop()()
        self.assertEqual(len(self.rows()), self.demo.t)
        self.assertEqual(self.demo.t, 1)

    def test_last_disconnect_preserves_session(self):
        client = Client()
        self.hub.clients.add(client)
        self.ticks(24)
        self.hub.close_client(client)
        self.assertEqual(len(self.rows()), 24)
        self.assertIs(self.hub.demo, self.demo)
        self.assertFalse(self.hub.running)
        self.assertFalse(self.demo._closed)

    def test_source_snapshot_matches_saved_metadata_and_csv(self):
        self.demo.set_source("5screws")
        self.ticks(3)
        self.hub.pause()
        saved = json.loads((self.out / "session_epoch01.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["source"], self.hub.full_state()["source"])
        self.assertEqual(saved["dataset"], self.hub.full_state()["meta"])
        self.assertEqual(saved["flushed_t"], len(self.rows()))
        self.assertEqual({row["truth"] for row in self.rows()}, {saved["source"]})

    def test_state_survives_legacy_covariance(self):
        # legacy 合併分布可能沒有健康一類的逐類閾值；state() 回 None 而不是中斷（#266）
        demo = LiveDemo(self.ds, mahalanobis_method="legacy")
        try:
            state = demo.state()
            self.assertIn("healthy_threshold", state)
            self.assertTrue(state["healthy_threshold"] is None or isinstance(state["healthy_threshold"], float))
        finally:
            demo.close()

    def test_server_entry_points_reject_legacy_at_startup(self):
        for module, main in (("web.server", server_main), ("web.guide", guide_main)):
            with self.subTest(module=module), patch.object(sys, "argv", [module, "--method", "legacy"]), \
                    patch("web.server.setup_run", side_effect=AssertionError("不應走到 setup_run")), \
                    patch("web.guide.setup_run", side_effect=AssertionError("不應走到 setup_run")):
                with self.assertRaisesRegex(SystemExit, "legacy"):
                    main()

    def test_sample_direction_carries_polar_map_verdict(self):
        # 方向判定由 PolarMap 的射線 τ 給出，前端不再用寫死的 cos 門檻重算（#270）
        self.ticks(1)
        self.hub.clients.add(Client())
        client = next(iter(self.hub.clients))
        self.ticks(2)
        sample = [m for m in client.messages if m["type"] == "sample"][-1]
        self.assertEqual(set(sample["direction"]), {"ray", "ray_display", "cos", "verdict", "radius"})
        # 階段 0 沒有故障射線：PolarMap 一律判 new_direction，cos 為 None
        self.assertEqual((sample["direction"]["verdict"], sample["direction"]["cos"]), ("new_direction", None))

    def test_dataset_changes_keep_prior_csv_metadata_and_model(self):
        self.ticks(3)
        self.hub.pause()
        saved = {path.name: path.read_bytes() for path in self.out.iterdir()}
        self.hub.datasets.append({**self.ds, "motor": "T3"})
        with patch("web.server.HUB", self.hub):
            WSHandler._do_dataset(self.hub.datasets[-1])
            self.demo = self.hub.demo
            self.assertEqual(self.demo.epoch, 2)
            self.ticks(2)
            self.hub.pause()
            for name, payload in saved.items():
                self.assertEqual((self.out / name).read_bytes(), payload)
            with (self.out / "samples_epoch02.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["t"]) for row in rows], [1, 2])
            self.assertEqual({int(row["epoch"]) for row in rows}, {2})
            metadata = json.loads((self.out / "session_epoch02.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset"]["motor"], "T3")
            self.assertEqual(metadata["flushed_t"], 2)
            saved = {path.name: path.read_bytes() for path in self.out.iterdir()}
            WSHandler._do_dataset(self.ds)
            self.demo = self.hub.demo
            self.assertEqual(self.demo.epoch, 3)
            self.ticks(1)
            self.hub.pause()
            for name, payload in saved.items():
                self.assertEqual((self.out / name).read_bytes(), payload)

    def test_new_epoch_skips_partial_history_without_overwriting(self):
        prior = {
            "model_epoch02_01classes.json": b"{\"history\": true}",
            "session_epoch03.json": b"{\"history\": true}",
            "samples_epoch04.csv": b"epoch,t\n4,1\n",
            "session_epoch05.json.tmp": b"{\"history\": true}",
        }
        for name, payload in prior.items():
            (self.out / name).write_bytes(payload)
        self.ticks(3)
        self.hub.pause()
        self.demo._build()
        self.assertEqual(self.demo.epoch, 6)
        self.assertEqual(len(self.rows()), 3)
        for name, payload in prior.items():
            self.assertEqual((self.out / name).read_bytes(), payload)

    def test_normal_stop_and_repeated_close(self):
        self.ticks(24)
        serve(Mock(), self.hub)
        self.hub.close()
        self.assertEqual(len(self.rows()), 24)
        with self.assertRaises(RuntimeError):
            self.demo.tick()

    def test_final_pause_snapshot_contains_exact_metrics(self):
        client = Client()
        self.hub.clients.add(client)
        self.ticks(24)
        self.hub.pause()
        final = client.messages[-1]
        self.assertFalse(final["running"])
        self.assertEqual(final["t"], 24)
        self.assertEqual(final["persistence"]["flushed_t"], 24)
        self.assertEqual(sum(r["pre_streamed"] + r["post_streamed"] for r in final["metrics"]["rows"]), 24)
        self.hub.tick()
        self.assertEqual(client.messages[-1], final)

    def test_shutdown_waits_worker_before_close(self):
        self.ticks(3)
        with patch.object(self.hub.executor, "shutdown") as shutdown, patch.object(
                self.demo, "close", wraps=self.demo.close) as close:
            calls = Mock()
            calls.attach_mock(shutdown, "wait")
            calls.attach_mock(close, "close")
            self.hub.close()
            self.assertEqual([c[0] for c in calls.mock_calls], ["wait", "close"])

    def test_busy_snapshot_does_not_read_mutating_model(self):
        self.ticks(3)
        self.hub.pause()
        self.hub._busy_state = self.hub.full_state()
        self.hub.busy = True
        with patch.object(self.demo, "state", side_effect=AssertionError("不可讀取 worker")), patch.object(
                self.demo, "metrics_msg", side_effect=AssertionError("不可讀取 worker")):
            state = self.hub.full_state()
            self.assertTrue(state["busy"])
            self.assertEqual(state["metrics"]["t"], 3)

    def test_ctrl_c_uses_same_shutdown(self):
        self.ticks(3)
        serve(Mock(start=Mock(side_effect=KeyboardInterrupt)), self.hub)
        self.assertEqual(len(self.rows()), 3)

    def test_flush_failure_does_not_acknowledge_pause(self):
        self.ticks(3)
        with patch.object(self.demo._sample_file, "flush", side_effect=OSError("測試寫入失敗")):
            with self.assertRaises(OSError):
                self.hub.pause()
        self.assertIn("保存失敗", self.hub.error)
        self.assertEqual(self.demo.flushed_t, 0)
        self.assertFalse(self.hub.running)

    def test_writer_failure_blocks_false_save_success(self):
        client = Client()
        self.hub.clients.add(client)
        with patch.object(self.demo, "_sample_writer") as writer:
            writer.writerow.side_effect = OSError("測試寫入失敗")
            self.ticks(1)
        self.assertFalse(any(m["type"] == "sample" for m in client.messages))
        self.assertIsNotNone(self.hub.error)
        with self.assertRaises(RuntimeError):
            self.hub.pause()
        with self.assertRaises(RuntimeError):
            self.demo.close()

    def fail_write(self):
        with patch.object(self.demo, "_sample_writer") as writer:
            writer.writerow.side_effect = OSError("測試寫入失敗")
            self.ticks(1)
        self.assertIsNotNone(self.hub.error)

    def test_write_error_recovers_through_reset(self):
        """#301 F-13：寫入失敗後一般串流仍拒絕，但重設會丟棄失敗紀錄、開新 epoch 檔並清除錯誤。"""
        import asyncio
        client = Client()
        self.hub.clients.add(client)
        self.fail_write()
        with patch("web.server.HUB", self.hub):
            with self.assertRaises(ValueError):
                asyncio.run(WSHandler.dispatch(Mock(), {"cmd": "start"}))
            asyncio.run(WSHandler._heavy(Mock(), WSHandler._do_reset, "重置"))
        self.assertIsNone(self.hub.error)
        self.assertTrue(self.demo.accepting)
        self.assertEqual(self.demo.epoch, 2)
        self.assertTrue(any(m["type"] == "event" and "寫入失敗" in m["text"] for m in client.messages))
        self.ticks(3)
        self.hub.pause()
        with (self.out / "samples_epoch02.csv").open(encoding="utf-8", newline="") as file:
            self.assertEqual([int(row["t"]) for row in csv.DictReader(file)], [1, 2, 3])

    def test_write_error_recovers_through_dataset_switch(self):
        """#301 F-13：換資料集時舊 session 的失敗紀錄被丟棄，close() 不再因舊錯誤丟例外。"""
        import asyncio
        self.fail_write()
        with patch("web.server.HUB", self.hub):
            asyncio.run(WSHandler._heavy(Mock(), lambda: WSHandler._do_dataset(self.ds), "載入資料集"))
        self.assertIsNone(self.hub.error)
        self.assertIsNot(self.hub.demo, self.demo)
        self.assertTrue(self.hub.demo.accepting)
        self.assertFalse(self.demo.accepting)
        self.hub.running = True
        self.hub.tick()
        self.assertEqual(self.hub.demo.t, 1)
        self.hub.demo.close()   # 暫存目錄清掉前先關新 session

    def test_busy_pause_does_not_touch_worker_file(self):
        self.hub.busy = True
        with patch.object(self.demo, "flush", side_effect=AssertionError("不可讀取 worker")):
            with self.assertRaises(RuntimeError):
                self.hub.pause()
            self.hub.close_client(Client())
        self.assertFalse(self.hub.running)

    def test_resume_rejected_after_write_error(self):
        import asyncio
        self.hub.error = "保存失敗"
        with patch("web.server.HUB", self.hub), self.assertRaises(ValueError):
            asyncio.run(WSHandler.dispatch(Mock(), {"cmd": "start"}))
