"""experiments.health_monitor：故障配置的窗口取樣與 8screws 輸出不變。"""
import contextlib
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from loguru import logger

from experiments.health.index import CalibratedHealthIndex
from experiments.health_monitor import iter_run, main, run
from core.feature_schema import SCHEMAS
from web.experiments import ExperimentRunner, jsonable

BASE = "64cb71d84663e1745ec54db74abad68def67e8e6"  # 比對 8screws 輸出不變的基準 commit
CONFIGS = ["8screws", "7screws", "6screws", "5screws", "4screws", "3screws",
           "2screws", "1screws", "3_14screws", "4_146screws"]
SIZES = {"8screws": 300, "5screws": 40}  # 5screws 比健康 holdout 最大索引還少


@lru_cache(None)
def baseline_run():
    source = subprocess.check_output(["git", "show", f"{BASE}:experiments/health_monitor.py"], cwd=Path(__file__).resolve().parents[2])
    namespace = {"__name__": "experiments._baseline_health_monitor"}
    exec(compile(source, f"{BASE}/health_monitor", "exec"), namespace)
    return namespace["run"]


def write_dataset(root: Path) -> None:
    rng = np.random.default_rng(3)
    base = root / "Step-5" / "myfeature" / "T1" / "8000rpm"
    columns = ",".join(SCHEMAS["adapter_generated_v1"])
    for i, config in enumerate(CONFIGS):
        folder = base / config
        folder.mkdir(parents=True)
        rows = rng.normal(size=(SIZES.get(config, 120), 105)) + i
        np.savetxt(folder / f"{config}_Group_feature_data_clean.csv", rows,
                   delimiter=",", header=columns, comments="")


def scientific_windows(rows):
    """科學結果逐欄不變；來源欄在專用案例另驗，不比較 opaque sidecar ID。"""
    return [{key: value for key, value in row.items()
             if key not in {"provenance", "provenance_record_id", "source_pool_index"}} for row in rows]


def stable_windows(rows):
    """每次載入各建私人 sidecar，溯源 ID 跟著變；只排除這個 ID，其餘欄位仍精確比較。"""
    result = json.loads(json.dumps(rows))
    for row in result:
        identity = row.pop("provenance_record_id")
        assert isinstance(identity, str) and identity.startswith("sha256:")
        assert "provenance" not in row
    return result


class HealthMonitorWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_dataset(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def call(self, fn, config, **kwargs):
        return fn(self.root, "T1", "8000", config, seed=42, motor_id="m", session_id="s", **kwargs)

    def test_small_fault_pool_streams_whole_pool_in_order(self):
        rows = self.call(run, "5screws")
        self.assertEqual(len(rows), SIZES["5screws"])
        self.assertEqual([r["window_index"] for r in rows], list(range(SIZES["5screws"])))
        with self.assertRaises(IndexError):
            self.call(baseline_run(), "5screws")

    def test_alias_config_1screw_resolves_to_canonical_pool(self):
        """池 key 是正規名；CLI/網頁打 1screw 不能 KeyError（#285 R2-03）。"""
        alias = self.call(run, "1screw", max_windows=3)
        canonical = self.call(run, "1screws", max_windows=3)
        self.assertEqual(stable_windows(alias), stable_windows(canonical))
        with self.assertRaises(KeyError):
            self.call(run, "9screws", max_windows=3)

    def test_legacy_covariance_is_rejected(self):
        """健康監測與實驗頁 8-3 一樣拒絕 legacy，跟正式基準同規則（#301 F-02）。"""
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.call(run, "8screws", mahalanobis_method="legacy")

    def test_max_windows_truncates_fault_pool(self):
        self.assertEqual(len(self.call(run, "5screws", max_windows=7)), 7)
        self.assertEqual(len(self.call(run, "7screws", max_windows=500)), 120)

    def test_healthy_sampling_unchanged(self):
        """健康 holdout 的取樣與開集分數和基準 commit 相同；健康指數映射在 #267 改過，不比對。"""
        keys = ("window_index", "openset_score", "is_fault", "fault_type", "openset_method", "condition")
        for kwargs in ({}, {"max_windows": 9}):
            old = self.call(baseline_run(), "8screws", **kwargs)
            new = self.call(run, "8screws", **kwargs)
            self.assertEqual(json.dumps(old, sort_keys=True), json.dumps(scientific_windows(new), sort_keys=True))
            self.assertEqual([{k: r[k] for k in keys} for r in old], [{k: r[k] for k in keys} for r in new])
            self.assertEqual(len(new), kwargs.get("max_windows", 60))

    def test_seed_reaches_detector_random_state(self):
        """detector 的 random_state 跟 --seed（#285 R2-06）；LW 不用亂數，結果與基準相同（見上一個測試）。"""
        with patch("experiments.health_monitor.CalibratedHealthIndex.fit",
                   wraps=CalibratedHealthIndex.fit) as fit:
            list(iter_run(self.root, "T1", "8000", "8screws", seed=9, motor_id="m", session_id="s", max_windows=3))
        self.assertEqual(fit.call_args.kwargs["random_state"], 9)

    def test_confidence_is_forwarded_to_calibration(self):
        events = {c: list(iter_run(self.root, "T1", "8000", "8screws", seed=42, motor_id="m",
                                   session_id="s", max_windows=5, confidence=c)) for c in (0.95, 0.8)}
        for c, items in events.items():
            calibration = items[0]["model"]["calibration"]
            self.assertEqual(calibration["confidence"], c)
            self.assertEqual(items[0]["model"]["resolved_config"]["effective"]["confidence"], c)
        self.assertNotEqual([e["window"]["openset_score"] for e in events[0.8][1:]],
                            [e["window"]["openset_score"] for e in events[0.95][1:]])
        self.assertEqual(stable_windows(self.call(run, "8screws", max_windows=5, confidence=0.8)),
                         stable_windows([e["window"] for e in events[0.8][1:]]))

    def test_iter_run_matches_run(self):
        for config in ("8screws", "5screws"):
            events = list(iter_run(self.root, "T1", "8000rpm", config, seed=5,
                                   motor_id="m", session_id="s", max_windows=30))
            self.assertEqual(events[0]["event"], "fitted")
            windows = [e["window"] for e in events if e["event"] == "window"]
            self.assertEqual(len(windows), len(events) - 1)
            batch = run(self.root, "T1", "8000rpm", config, seed=5,
                        motor_id="m", session_id="s", max_windows=30)
            self.assertEqual(json.dumps(scientific_windows(windows), sort_keys=True),
                             json.dumps(scientific_windows(batch), sort_keys=True))
            # 完整溯源只在 fitted 一次，每窗只帶指回它的 ID（#305 G-02）
            self.assertEqual({w["provenance_record_id"] for w in windows}, {events[0]["provenance_record_id"]})
            self.assertEqual(events[0]["provenance"]["schema_version"], "experiment_provenance_v1")
            self.assertTrue(all("provenance" not in w for w in windows + batch))
            self.assertEqual(len({w["provenance_record_id"] for w in batch}), 1)

    def test_web_stream_done_matches_batch_and_is_saved(self):
        from core.data import discover_datasets
        with tempfile.TemporaryDirectory() as out:
            runner = ExperimentRunner(discover_datasets(self.root), self.root, out_dir=out)
            params = {"dataset": "T1/8000rpm", "config": "5screws", "max_windows": 25}
            frames = list(runner.stream("exp8_monitor", params))
            batch = runner.run("exp8_monitor", params)
            done = frames[-1][0]
            self.assertEqual(done["event"], "done")
            self.assertEqual(json.dumps(scientific_windows(done["payload"]["result"]), sort_keys=True),
                             json.dumps(scientific_windows(jsonable(batch["result"])), sort_keys=True))
            self.assertEqual(sum(u for _, u in frames), 25)
            saved = json.loads(Path(done["payload"]["saved"]).read_text(encoding="utf-8"))
            self.assertTrue(saved["streamed"])
            # 存檔標頭有一份完整溯源，逐窗只留 ID（#305 G-02）
            for payload in (saved, batch):
                self.assertEqual(payload["provenance"]["schema_version"], "experiment_provenance_v1")
                self.assertEqual({w["provenance_record_id"] for w in payload["result"]},
                                 {payload["provenance_record_id"]})
                self.assertTrue(all("provenance" not in w for w in payload["result"]))
            for frame, _ in frames:
                json.dumps(frame, allow_nan=False)

    def test_cli_uses_setup_run_and_saves_printed_windows(self):
        """CLI 經 setup_run 寫 logs/output，windows.jsonl 與 stdout 及 API 逐行相同。"""
        argv = ["--data-root", str(self.root), "--motor", "T1", "--rpm", "8000rpm", "--config", "5screws",
                "--seed", "42", "--confidence", "0.95", "--motor-id", "m", "--session-id", "s", "--max-windows", "12"]
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as workspace:
            try:
                os.chdir(workspace)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(argv), 0)
                logs = list(Path("logs/exp8_health_monitor").glob("*.log"))
                outputs = list(Path("output/exp8_health_monitor").iterdir())
                self.assertEqual(len(logs), 1)
                self.assertEqual(len(outputs), 1)
                saved = (outputs[0] / "windows.jsonl").read_text(encoding="utf-8")
                self.assertTrue((outputs[0] / "environment.json").is_file())
            finally:
                logger.remove()
                os.chdir(previous)
        self.assertEqual(saved, stdout.getvalue())
        batch = self.call(run, "5screws", max_windows=12)
        self.assertEqual(scientific_windows([json.loads(line) for line in saved.splitlines()]),
                         json.loads(json.dumps(scientific_windows(batch), ensure_ascii=False)))


if __name__ == "__main__":
    unittest.main()
