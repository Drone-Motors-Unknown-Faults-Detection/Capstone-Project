"""PR36 部分抽取：固定主線配對、SSE 互斥與例外回歸。"""
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tornado.gen
import tornado.tcpclient
import tornado.web
from tornado.testing import AsyncHTTPTestCase, gen_test

from web import server
from web.experiments import CATALOG, ROOT, ExperimentRunner, jsonable

BASE = "64cb71d84663e1745ec54db74abad68def67e8e6"


@lru_cache(None)
def baseline(module):
    source = subprocess.check_output(["git", "show", f"{BASE}:experiments/{module}.py"])
    namespace = {"__name__": f"experiments._baseline_{module}"}
    exec(compile(source, f"{BASE}/{module}", "exec"), namespace)
    return namespace["run"]


def pools():
    rng = np.random.default_rng(7)
    configs = ["8screws", "7screws", "6screws", "5screws", "4screws", "3screws",
               "2screws", "1screws", "3_14screws", "4_146screws"]
    return {c: rng.normal(size=(120, 105)) + i for i, c in enumerate(configs)}


class PairedStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pools()

    def same(self, old, new):
        self.assertEqual(json.dumps(jsonable(old), sort_keys=True),
                         json.dumps(jsonable(new), sort_keys=True))

    def test_cold_both_detectors_match_original_main(self):
        from experiments.exp1_cold_start import run, iter_run
        for method in ("mahalanobis", "knn"):
            opts = dict(seed=42, openset_method=method)
            events = list(iter_run(self.data, **opts))
            self.same(baseline("exp1_cold_start")(self.data, **opts), run(self.data, **opts))
            self.same(run(self.data, **opts), events[-1]["result"])
            self.assertEqual(events[0]["event"], "fitted")
            self.assertEqual(sum(len(e["scores"]) for e in events if e["event"] == "scores"),
                             sum(r["n"] for r in events[-1]["result"]["rows"]))

    def test_trend_both_detectors_match_original_main(self):
        from experiments.exp3_trend import run, iter_run
        for method in ("mahalanobis", "knn"):
            opts = dict(seed=42, n_trials=2, openset_method=method)
            events = list(iter_run(self.data, **opts))
            self.same(baseline("exp3_trend")(self.data, **opts), events[-1]["result"])
            self.same(run(self.data, **opts), events[-1]["result"])
            self.assertEqual(sum(e["event"] == "trial" for e in events), 4)

    def test_polar_all_parts_match_original_main(self):
        from experiments.exp4_polar_map import run, iter_run
        events = list(iter_run(self.data, seed=42))
        self.same(baseline("exp4_polar_map")(self.data, seed=42), events[-1]["result"])
        self.same(run(self.data, seed=42), events[-1]["result"])
        self.assertEqual([e["name"] for e in events if e["event"] == "part"],
                         ["geometry", "direction", "severity"])

    def test_catalog_streams_health_monitor_after_issue_35(self):
        self.assertEqual({e["id"] for e in CATALOG if e.get("stream")},
                         {"exp1", "exp3", "exp4", "exp8_monitor"})
        with self.assertRaises(ValueError):
            list(ExperimentRunner([], "unused").stream("exp5", {}))

    def test_localhost_option_is_explicit_and_original_default_retained(self):
        source = (ROOT / "src/web/server.py").read_text(encoding="utf-8")
        self.assertIn('"--bind-address", default="0.0.0.0"', source)
        self.assertIn('app.listen(args.port, address=args.bind_address)', source)

    def test_done_saved_and_matches_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = ExperimentRunner([{"motor": "T1", "rpm": "8000rpm", "path": "unused"}],
                                      "unused", out_dir=folder)
            with patch.object(runner, "_pools", return_value=self.data):
                frames = list(runner.stream("exp1", {}))
                batch = runner.run("exp1", {})
            self.same(frames[-1][0]["payload"]["result"], batch["result"])
            self.assertEqual(sum(u for _, u in frames), sum(r["n"] for r in batch["result"]["rows"]))
            saved = Path(frames[-1][0]["payload"]["saved"])
            if not saved.is_absolute():
                saved = ROOT / saved
            self.assertTrue(saved.is_file())
            self.assertTrue(json.loads(saved.read_text(encoding="utf-8"))["streamed"])


class FakeRunner:
    mode = "normal"
    closed = False

    def run(self, exp_id, params):
        return {"seconds": 0, "saved": None, "result": {"ok": True}}

    def stream(self, exp_id, params):
        try:
            if self.mode == "error":
                raise ValueError("測試無效參數")
            yield {"event": "start"}, 0
            if self.mode in ("slow", "close_error"):
                for _ in range(20):
                    yield {"event": "scores", "scores": [1]}, 1
            yield {"event": "done", "payload": {"saved": None}}, 0
        finally:
            self.closed = True
            if self.mode == "close_error":
                raise RuntimeError("測試 generator.close 失敗")


class StreamHTTPTests(AsyncHTTPTestCase):
    def get_app(self):
        self.runner = FakeRunner()
        self.previous_jobs = server.JOBS
        server.JOBS = server.ExperimentJobs(self.runner)
        return tornado.web.Application([
            (r"/stream/([a-z0-9_]+)", server.StreamHandler),
            (r"/run/([a-z0-9_]+)", server.RunHandler),
        ])

    def tearDown(self):
        server.JOBS.executor.shutdown(wait=True)
        server.JOBS = self.previous_jobs
        super().tearDown()

    def frames(self, response):
        return [json.loads(line[6:]) for line in response.body.decode().splitlines() if line.startswith("data: ")]

    def test_done_releases_shared_lock(self):
        response = self.fetch("/stream/exp1")
        self.assertEqual([f["event"] for f in self.frames(response)], ["start", "done"])
        self.assertTrue(self.runner.closed)
        self.assertIsNone(server.JOBS.current)
        self.assertEqual(self.fetch("/run/exp1", method="POST", body="{}").code, 200)

    def test_busy_stream_and_batch_are_rejected(self):
        server.JOBS.current = "exp3"
        self.assertIn("執行中", self.frames(self.fetch("/stream/exp1"))[0]["error"])
        self.assertEqual(self.fetch("/run/exp1", method="POST", body="{}").code, 409)
        self.assertEqual(server.JOBS.current, "exp3")

    def test_bad_json_does_not_acquire_lock(self):
        self.assertEqual(self.frames(self.fetch("/stream/exp1?params=%7B"))[0]["event"], "error")
        self.assertIsNone(server.JOBS.current)

    def test_invalid_model_parameter_releases_lock(self):
        self.runner.mode = "error"
        self.assertEqual(self.frames(self.fetch("/stream/exp1"))[0]["event"], "error")
        self.assertIsNone(server.JOBS.current)
        self.assertTrue(self.runner.closed)

    @gen_test
    async def test_disconnect_closes_generator_and_releases_lock(self):
        self.runner.mode = "slow"
        conn = await tornado.tcpclient.TCPClient().connect("127.0.0.1", self.get_http_port())
        await conn.write(b"GET /stream/exp1?rate=10 HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await conn.read_until(b'"event": "start"')
        self.assertEqual(server.JOBS.current, "exp1")
        conn.close()
        for _ in range(50):
            if server.JOBS.current is None:
                break
            await tornado.gen.sleep(.02)
        self.assertIsNone(server.JOBS.current)
        self.assertTrue(self.runner.closed)

    @gen_test
    async def test_close_exception_still_releases_lock(self):
        self.runner.mode = "close_error"
        conn = await tornado.tcpclient.TCPClient().connect("127.0.0.1", self.get_http_port())
        await conn.write(b"GET /stream/exp1?rate=10 HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await conn.read_until(b'"event": "start"')
        conn.close()
        await tornado.gen.sleep(.2)
        self.assertIsNone(server.JOBS.current)

    def test_non_object_parameters_and_non_finite_rate_rejected(self):
        for query in ("params=%5B1%5D", "rate=nan", "rate=inf"):
            with self.subTest(query=query):
                self.assertEqual(self.frames(self.fetch("/stream/exp1?" + query))[0]["event"], "error")
                self.assertIsNone(server.JOBS.current)


if __name__ == "__main__":
    unittest.main()
