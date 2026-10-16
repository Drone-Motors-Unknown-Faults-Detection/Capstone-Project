"""暫停、斷線及正常結束的實際 CSV 保存契約。"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.test_navigation_guide import fixture, Client
from web.live import LiveDemo
from web.server import Hub, WSHandler, serve


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

    def test_dataset_changes_keep_prior_csv_metadata_and_model(self):
        self.ticks(3)
        self.hub.pause()
        saved = {path.name: path.read_bytes() for path in self.out.iterdir()}
        self.hub.datasets.append({**self.ds, "motor": "T3"})
        with patch("web.server.HUB", self.hub):
            WSHandler._do_dataset("T3", self.ds["rpm"])
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
            WSHandler._do_dataset(self.ds["motor"], self.ds["rpm"])
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
