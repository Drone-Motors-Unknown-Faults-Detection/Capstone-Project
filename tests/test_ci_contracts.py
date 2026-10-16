"""不讀 ignored data 的 loader、exp1～3、主 WebSocket 契約。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tornado.gen
import tornado.httpclient
import tornado.web
from tornado.testing import AsyncHTTPTestCase, gen_test
from tornado.websocket import websocket_connect

from core.data import discover_datasets, load_pools, make_split
from experiments import exp1_cold_start as exp1, exp2_scale_growth as exp2, exp3_trend as exp3
from tests.ci_evidence import execute
from tests.test_stream_integration import pools
from web import server

ROOT = Path(__file__).resolve().parents[1]


def write_fixture(root: Path, data: dict, chunks: bool = False) -> Path:
    dataset = root / "Step-fixture" / "myfeature" / "T1" / "8000rpm"
    for config, values in data.items():
        folder = dataset / config
        folder.mkdir(parents=True)
        pieces = np.array_split(values, 2) if chunks else [values]
        for index, piece in enumerate(pieces):
            pd.DataFrame(piece, columns=[f"f{i}" for i in range(105)]).to_csv(
                folder / f"{index}_Group_feature_data_clean.csv", index=False)
    return dataset


class LoaderContracts(unittest.TestCase):
    def test_scan_load_sorted_multiple_files_and_alias(self):
        data = pools()
        data["1screw"] = data.pop("1screws")
        with tempfile.TemporaryDirectory() as directory:
            dataset = write_fixture(Path(directory), data, chunks=True)
            found = discover_datasets(directory)
            self.assertEqual([(row["motor"], row["rpm"]) for row in found], [("T1", "8000rpm")])
            loaded = load_pools(dataset)
        self.assertEqual(next(iter(loaded)), "8screws")
        self.assertIn("1screw", loaded)
        for name, values in data.items():
            np.testing.assert_allclose(loaded[name], values, rtol=1e-14, atol=1e-14)

    def test_healthy_only_is_legal_and_missing_healthy_rejected(self):
        for healthy, accepted in ((True, True), (False, False)):
            with self.subTest(healthy=healthy), tempfile.TemporaryDirectory() as directory:
                data = {"8screws" if healthy else "7screws": pools()["8screws"]}
                dataset = write_fixture(Path(directory), data)
                if accepted:
                    self.assertEqual(list(load_pools(dataset)), ["8screws"])
                else:
                    with self.assertRaisesRegex(ValueError, "缺少健康基準"):
                        load_pools(dataset)

    def test_split_roles_disjoint_and_same_seed(self):
        a = make_split(120, np.random.default_rng(42))
        b = make_split(120, np.random.default_rng(42))
        for role in ("train", "cal", "holdout"):
            np.testing.assert_array_equal(getattr(a, role), getattr(b, role))
        roles = [set(a.train), set(a.cal), set(a.holdout)]
        self.assertEqual([len(role) for role in roles], [72, 24, 24])
        self.assertFalse(roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])


class ExperimentContracts(unittest.TestCase):
    def test_exp1_event_support_and_threshold(self):
        data = pools()
        events = list(exp1.iter_run(data, seed=42))
        self.assertEqual((events[0]["event"], events[-1]["event"]), ("fitted", "result"))
        scores = [row for row in events if row["event"] == "scores"]
        self.assertEqual(scores[0]["kind"], "healthy-holdout")
        self.assertEqual(len(scores[0]["scores"]), 24)
        self.assertTrue(all(len(row["scores"]) == 120 for row in scores[1:]))
        for score, row in zip(scores, events[-1]["result"]["rows"]):
            self.assertEqual(row["flagged"], int((np.asarray(score["scores"]) > 1).sum()))

    def test_exp2_requires_explicit_confirmation_before_growth(self):
        session = exp2.ScaleGrowthSession(pools(), seed=42, recluster_every=10000)
        self.assertEqual(list(session.monitor.known), ["8screws"])
        with self.assertRaises(RuntimeError):
            session.confirm()
        session.process(session.monitor.pools["7screws"][0], "7screws")
        self.assertEqual(list(session.monitor.known), ["8screws"])
        session.candidate = {"majority": "7screws", "size": 1, "purity": 1.0, "indices": []}
        result = session.confirm()
        self.assertEqual(result["action"], "learned")
        self.assertEqual(list(session.monitor.known), ["8screws", "7screws"])
        self.assertIsNone(session.candidate)

    def test_exp2_bounded_run_repeats_and_preserves_roles(self):
        a = exp2.run(pools(), sequence=["7screws"], seed=42, max_ticks_per_stage=2)
        b = exp2.run(pools(), sequence=["7screws"], seed=42, max_ticks_per_stage=2)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertFalse(a["stages"][0]["discovered"])
        self.assertEqual(a["final_known"], ["8screws"])
        self.assertEqual(a["threshold"], 1.0)

    def test_exp3_fixed_trials_events_and_determinism(self):
        events = list(exp3.iter_run(pools(), seed=42, n_trials=1))
        a = events[-1]["result"]
        self.assertEqual(a, exp3.run(pools(), seed=42, n_trials=1))
        self.assertEqual([row["trial"]["scenario"] for row in events if row["event"] == "trial"], ["A", "B"])
        self.assertEqual(a["threshold"], 1.0)
        self.assertEqual(a["n_trials"], 1)

    def test_exp1_to_3_real_cli_with_temporary_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root / "fixture", pools())
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
            # exp2 的空 sequence 只驗 CLI／載入／寫出；分群與確認另由上面 API fixture 驗證。
            for module, extra in (("exp1_cold_start", []), ("exp2_scale_growth", ["--sequence"]),
                                  ("exp3_trend", ["--trials", "1"])):
                with self.subTest(module=module):
                    result = subprocess.run([sys.executable, "-m", f"experiments.{module}",
                                             "--data-root", "fixture", "--seed", "42", *extra],
                                            cwd=root, env=env, capture_output=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
                    summaries = list((root / "output" / module).glob("*/summary.json"))
                    self.assertEqual(len(summaries), 1)
                    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
                    self.assertEqual(summary["seed"], 42)
                    self.assertEqual(summary["dataset"], {"motor": "T1", "rpm": "8000rpm"})


class FakeDemo:
    t = 0
    epoch = 0
    flushed = 0

    def flush(self):
        self.flushed += 1

    def state(self):
        return {"epoch": self.epoch}

    def metrics_msg(self):
        return {"type": "metrics"}

    def _build(self):
        self.epoch += 1


class WebSocketContracts(AsyncHTTPTestCase):
    def get_app(self):
        self.previous = server.HUB
        self.hub = server.Hub(FakeDemo(), [], 4, 42)
        server.HUB = self.hub
        return tornado.web.Application([(r"/ws", server.WSHandler)])

    def tearDown(self):
        if self.hub.periodic:
            self.hub.periodic.stop()
        for client in list(self.hub.clients):
            client.close()
        self.hub.executor.shutdown(wait=True)
        try:
            super().tearDown()
        finally:
            server.HUB = self.previous

    async def disconnect(self, socket):
        socket.close()
        for _ in range(50):
            if not self.hub.clients:
                return
            await tornado.gen.sleep(.01)
        self.fail("WebSocket 未完成關閉")

    async def connect(self):
        request = tornado.httpclient.HTTPRequest(self.get_url("/ws").replace("http:", "ws:"),
                                                headers={"Origin": "https://untrusted.invalid"})
        socket = await websocket_connect(request)
        self.assertEqual(json.loads(await socket.read_message())["type"], "state")
        self.assertEqual(json.loads(await socket.read_message())["type"], "metrics")
        return socket

    async def send_until(self, socket, command, wanted="state"):
        socket.write_message(command if isinstance(command, str) else json.dumps(command))
        messages = []
        for _ in range(10):
            message = json.loads(await socket.read_message())
            messages.append(message)
            if message["type"] == wanted:
                return messages
        self.fail("未收到預期的 WebSocket 訊息")

    @gen_test
    async def test_start_pause_rate_and_unknown_command(self):
        socket = await self.connect()
        try:
            self.assertTrue((await self.send_until(socket, {"cmd": "start"}))[-1]["running"])
            self.assertFalse((await self.send_until(socket, {"cmd": "pause"}))[-1]["running"])
            self.assertEqual(self.hub.demo.flushed, 1)
            self.assertEqual((await self.send_until(socket, {"cmd": "rate", "value": 50}))[-1]["rate"], 10)
            messages = await self.send_until(socket, {"cmd": "invalidcmd"})
            self.assertEqual([message["type"] for message in messages], ["state"])
        finally:
            await self.disconnect(socket)

    @gen_test
    async def test_reset_busy_then_state_and_metrics(self):
        socket = await self.connect()
        try:
            messages = await self.send_until(socket, {"cmd": "reset"}, "metrics")
            states = [message for message in messages if message["type"] == "state"]
            self.assertTrue(states[0]["busy"])
            self.assertFalse(states[-1]["busy"])
            self.assertEqual(states[-1]["epoch"], 1)
            self.assertFalse(states[-1]["running"])
        finally:
            await self.disconnect(socket)

    @gen_test
    async def test_invalid_json_reports_error_and_connection_survives(self):
        socket = await self.connect()
        try:
            messages = await self.send_until(socket, "{")
            self.assertEqual(messages[0]["level"], "error")
            self.assertFalse(messages[-1]["running"])
            self.assertTrue((await self.send_until(socket, {"cmd": "start"}))[-1]["running"])
        finally:
            await self.disconnect(socket)

    @gen_test
    async def test_untrusted_origin_is_current_insecure_baseline(self):
        # 成功連線只是 #28 尚待修的現狀，不稱為安全 PASS。
        socket = await self.connect()
        await self.disconnect(socket)


class EvidencePrivacyTests(unittest.TestCase):
    def test_public_failure_keeps_status_not_private_payload(self):
        with patch("tests.ci_evidence.subprocess.run") as command:
            command.return_value = subprocess.CompletedProcess([], 1, "private-token /private/path", "FAILED (failures=1)\nRan 1 test in 0.0s")
            result = execute(["-m", "unittest"])
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["tests_run"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertNotIn("private-token", json.dumps(result))
        self.assertNotIn("/private/path", json.dumps(result))

    def test_process_timeout_is_failed_not_silently_successful(self):
        with patch("tests.ci_evidence.subprocess.run", side_effect=subprocess.TimeoutExpired("python", 1)):
            self.assertEqual(execute(["-m", "unittest"])["returncode"], -1)


if __name__ == "__main__":
    unittest.main()
