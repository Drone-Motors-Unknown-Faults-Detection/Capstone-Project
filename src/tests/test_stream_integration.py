"""固定主線配對、SSE 互斥與例外回歸。"""
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
from web.playback import PlaybackTracker

BASE = "64cb71d84663e1745ec54db74abad68def67e8e6"


@lru_cache(None)
def baseline(module):
    source = subprocess.check_output(["git", "show", f"{BASE}:experiments/{module}.py"], cwd=Path(__file__).resolve().parents[2])
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

    def test_catalog_stream_set_includes_health_monitor(self):
        self.assertEqual({e["id"] for e in CATALOG if e.get("stream")},
                         {"exp1", "exp2", "exp3", "exp4", "exp8_monitor"})
        with self.assertRaises(ValueError):
            list(ExperimentRunner([], "unused").stream("exp5", {}))

    def test_exp2_playback_learns_cluster_without_confirm(self):
        from experiments.exp2_scale_growth import ScaleGrowthSession, iter_playback

        original = ScaleGrowthSession.process

        def process(session, sample, truth, row_id=None):
            out = original(session, sample, truth, row_id)
            rows = np.repeat(session.monitor.pools[truth][:1], 25, axis=0)
            session.quarantine_X = [row.copy() for row in rows]
            session.quarantine_truth = [truth] * 25
            session.quarantine_t = list(range(1, 26))
            session.candidate = {
                "indices": list(range(25)), "size": 25, "majority": truth,
                "majority_display": truth, "purity": 1.0, "t_range": [1, 25],
                "is_known": False, "cluster_params": [25, 3],
            }
            return out

        with patch.object(ScaleGrowthSession, "process", process):
            learned = list(iter_playback(
                self.data, sequence=["7screws"], seed=42, max_ticks_per_stage=2,
            ))
        self.assertEqual(learned[-1]["event"], "result")
        self.assertEqual(learned[-1]["result"]["final_known"], ["8screws", "discovered-1"])
        self.assertEqual(learned[-1]["result"]["stages"][0]["matched"], "7screws")
        self.assertEqual(learned[-1]["result"]["stages"][0]["samples_to_candidate"], 1)

    def test_exp3_page_can_stream_one_scenario(self):
        from experiments.exp3_trend import iter_run, run

        events = list(iter_run(self.data, seed=42, n_trials=1, scenarios=("A",)))
        self.assertEqual([event["trial"]["scenario"] for event in events if event["event"] == "trial"], ["A"])
        self.assertEqual(set(run(self.data, seed=42, n_trials=1)["summary"]), {"A", "B"})
        runner = ExperimentRunner(
            [{"motor": "T1", "rpm": "8000rpm", "path": "unused"}], "unused",
        )
        with patch.object(runner, "_pools", return_value=self.data):
            frames = [frame for frame, _units in runner.stream("exp3", {
                "dataset": "T1/8000rpm", "seed": 42, "n_trials": 1,
                "openset_method": "mahalanobis", "scenario": "B",
            })]
        self.assertEqual([frame["trial"]["scenario"] for frame in frames if frame["event"] == "trial"], ["B"])
        with self.assertRaisesRegex(ValueError, "scenario"):
            list(runner.stream("exp3", {"scenario": "C"}))

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
        if self.mode == "run_crash":
            raise RuntimeError("測試執行失敗")
        return {"seconds": 0, "saved": None, "result": {"ok": True}}

    def clean_params(self, exp_id, params):
        if self.mode == "bad_params":
            raise ValueError("測試參數錯誤")
        return {"seed": 42}   # 清洗後的參數：原始用戶端鍵不保留

    def stream_params(self, exp_id, params):
        return self.clean_params(exp_id, params)

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

    def test_exp2_confirm_endpoint_is_removed(self):
        # 實驗二成群即 adopt() 自學，實驗頁不提供確認閘門
        patterns = [rule.matcher.regex.pattern for rule in server.make_app().wildcard_router.rules]
        self.assertFalse(any("confirm" in pattern for pattern in patterns))
        self.assertFalse(hasattr(server, "Exp2ConfirmHandler"))

    def test_done_releases_shared_lock(self):
        response = self.fetch("/stream/exp1")
        self.assertEqual([f["event"] for f in self.frames(response)], ["start", "done"])
        self.assertTrue(self.runner.closed)
        self.assertIsNone(server.JOBS.current)
        self.assertEqual(self.fetch("/run/exp1", method="POST", body="{}", headers={"Content-Type": "application/json"}).code, 200)

    def test_busy_stream_and_batch_are_rejected(self):
        server.JOBS.current = "exp3"
        self.assertIn("執行中", self.frames(self.fetch("/stream/exp1"))[0]["error"])
        self.assertEqual(self.fetch("/run/exp1", method="POST", body="{}", headers={"Content-Type": "application/json"}).code, 409)
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

    def test_invalid_parameters_keep_previous_playback_record(self):
        # 參數錯誤在 PLAYBACK.start 之前就回報；上一次完成的紀錄不會被換成錯誤（#274）
        previous = server.PLAYBACK
        server.PLAYBACK = PlaybackTracker()
        try:
            server.PLAYBACK.record_run("exp1", {"params": {"seed": 42}, "result": {"ok": True}, "seconds": 1.0})
            self.runner.mode = "bad_params"
            frames = self.frames(self.fetch("/stream/exp1"))
            self.assertEqual([f["event"] for f in frames], ["error"])
            self.assertIn("測試參數錯誤", frames[0]["error"])
            record = server.PLAYBACK.snapshot()["records"]["exp1"]
            self.assertEqual((record["mode"], record["status"]), ("run", "done"))
            self.assertIsNone(server.JOBS.current)
        finally:
            server.PLAYBACK = previous

    def test_batch_run_records_cleaned_params_only(self):
        """參數錯誤回 400 且不進 PLAYBACK；執行失敗只記清洗後的參數，不記原始用戶端字串（#285 R2-10）。"""
        previous = server.PLAYBACK
        server.PLAYBACK = PlaybackTracker()
        try:
            self.runner.mode = "bad_params"
            response = self.fetch("/run/exp1", method="POST", body='{"seed": "<img src=x>"}', headers={"Content-Type": "application/json"})
            self.assertEqual(response.code, 400)
            self.assertIsNone(server.PLAYBACK.snapshot())
            self.runner.mode = "run_crash"
            response = self.fetch("/run/exp1", method="POST", body='{"seed": 42, "evil": "<img src=x>"}',
                                  headers={"Content-Type": "application/json"})
            self.assertEqual(response.code, 500)
            record = server.PLAYBACK.snapshot()["records"]["exp1"]
            self.assertEqual(record["status"], "error")
            self.assertNotIn("<img", json.dumps(record, ensure_ascii=False))
            self.assertIsNone(server.JOBS.current)
        finally:
            server.PLAYBACK = previous

    def test_cross_site_run_and_stream_rejected(self):
        """#301 F-15：Origin／Referer 不同站 → 403；沒帶 Origin 但 Referer 同站（瀏覽器 EventSource）、
        兩者都沒帶的非瀏覽器用戶端照常；run 只收 application/json。被拒的請求不佔執行鎖。"""
        here = self.get_url("/")[:-1]
        json_type = {"Content-Type": "application/json"}
        for headers in ({"Origin": "https://evil.invalid"}, {"Referer": "https://evil.invalid/page"},
                        {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.fetch("/stream/exp1", headers=headers).code, 403)
                self.assertEqual(self.fetch("/run/exp1", method="POST", body="{}",
                                            headers={**headers, **json_type}).code, 403)
                self.assertIsNone(server.JOBS.current)
        for content_type in ("application/x-www-form-urlencoded", "text/plain"):
            with self.subTest(content_type=content_type):
                response = self.fetch("/run/exp1", method="POST", body="{}", headers={"Content-Type": content_type})
                self.assertEqual(response.code, 415)
        self.assertEqual([f["event"] for f in self.frames(self.fetch("/stream/exp1", headers={"Referer": here + "/"}))],
                         ["start", "done"])
        self.assertEqual(self.fetch("/run/exp1", method="POST", body="{}",
                                    headers={"Origin": here, **json_type}).code, 200)
        self.assertEqual(self.fetch("/run/exp1", method="POST", body="{}",
                                    headers={"Content-Type": "application/json; charset=utf-8"}).code, 200)
        server.ALLOWED_ORIGINS.add("https://demo.invalid")
        try:
            self.assertEqual(self.fetch("/run/exp1", method="POST", body="{}",
                                        headers={"Origin": "https://demo.invalid", **json_type}).code, 200)
        finally:
            server.ALLOWED_ORIGINS.discard("https://demo.invalid")

    def test_non_object_parameters_and_non_finite_rate_rejected(self):
        for query in ("params=%5B1%5D", "rate=nan", "rate=inf"):
            with self.subTest(query=query):
                self.assertEqual(self.frames(self.fetch("/stream/exp1?" + query))[0]["event"], "error")
                self.assertIsNone(server.JOBS.current)


if __name__ == "__main__":
    unittest.main()
