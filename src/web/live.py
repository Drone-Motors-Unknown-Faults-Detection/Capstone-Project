"""Web 即時展示的協調器（LiveDemo）。

把三個實驗模組串成可互動的即時串流——本模組只做編排、訊息組裝與 session 持久化，
實驗邏輯全部住在 experiments / core：
    實驗一（冷啟動監測）  → ScaleGrowthSession 內的 OpenSetMonitor 階段 0
    實驗二（量尺擴張）    → ScaleGrowthSession（隔離、分群、確認、擴張）
    實驗三（變化點判別）  → TrendMonitor + exp3 的 SCENARIOS 劇本定義

session 持久化（out_dir 通常是 output/web_server/{ts}/，重置一次算一個 epoch）：
    samples_epoch{N}.csv          逐筆串流紀錄（t/真實注入/分數/判定/趨勢統計）
    model_epoch{N}_{K}classes.json  每次擬合後的模型快照（逐類閾值與樣本數）
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from core.data import (
    HEALTHY,
    CycleSampler,
    config_sort_key,
    display_name,
    load_pools,
)
from core.trend import TrendMonitor
from experiments.exp2_scale_growth import ScaleGrowthSession
from experiments.exp3_trend import KIND_DISPLAY, SCENARIOS

SAMPLE_FIELDS = [
    "epoch", "t", "truth", "score", "verdict", "predicted",
    "ewma", "cusum", "quarantine", "n_known", "source_mode",
    "dir_radius", "dir_ray", "dir_cos",
]


def _event(level: str, text: str, t: int) -> dict:
    return {"type": "event", "level": level, "text": text, "t": t}


class LiveDemo:
    def __init__(
        self,
        dataset: dict,
        seed: int = 42,
        out_dir: Path | str | None = None,
        openset_method: str = "mahalanobis",
        mahalanobis_method: str = "ledoit_wolf",
        confidence: float = 0.95,
        knn_neighbors: int = 5,
    ):
        self.meta = {"motor": dataset["motor"], "rpm": dataset["rpm"]}
        self.pools = load_pools(dataset["path"])
        self.seed = seed
        self.openset_method = openset_method
        self.mahalanobis_method = mahalanobis_method
        self.confidence = confidence
        self.knn_neighbors = knn_neighbors
        self.out_dir = Path(out_dir) if out_dir else None
        self.epoch = 0
        self._sample_file = None
        self._closed = False
        self._build()

    def _build(self) -> None:
        self.epoch += 1
        self.session = ScaleGrowthSession(
            self.pools,
            seed=self.seed,
            confidence=self.confidence,
            method=self.mahalanobis_method,
            openset_method=self.openset_method,
            knn_neighbors=self.knn_neighbors,
        )
        self.trend = TrendMonitor()
        self.stream_rng = np.random.default_rng(self.seed + 1)
        self.t = 0
        self.source = HEALTHY
        self.scenario: dict | None = None
        self.switch_t = 0
        self.fault_onset_t: int | None = None
        self.metrics = {
            c: {"pre_streamed": 0, "pre_flagged": 0, "post_streamed": 0, "post_flagged": 0}
            for c in self.pools
        }
        self.alarms: list[dict] = []
        self.samplers: dict[str, CycleSampler] = {}
        self._refresh_samplers()
        self._open_sample_log()
        self._snapshot_model("init")

    # -- session 持久化 --------------------------------------------------------

    def _open_sample_log(self) -> None:
        if self._sample_file is not None:
            self._sample_file.close()
            self._sample_file = None
        self._sample_writer = None
        self._closed = False
        self._write_error = None
        self.written_t = self.flushed_t = 0
        if self.out_dir is None:
            return
        # 換資料集會建立新的 LiveDemo；共用目錄中的舊 epoch 不可覆寫。
        while True:
            metadata = self.out_dir / f"session_epoch{self.epoch:02d}.json"
            if (metadata.exists() or metadata.with_suffix(".json.tmp").exists()
                    or any(self.out_dir.glob(f"model_epoch{self.epoch:02d}_*.json"))):
                self.epoch += 1
                continue
            path = self.out_dir / f"samples_epoch{self.epoch:02d}.csv"
            try:
                self._sample_file = open(path, "x", newline="", encoding="utf-8")
            except FileExistsError:
                self.epoch += 1
                continue
            break
        self._sample_writer = csv.writer(self._sample_file)
        self._sample_writer.writerow(SAMPLE_FIELDS)

    def _log_sample(self, row: list) -> None:
        if self._sample_writer is None:
            return
        try:
            self._sample_writer.writerow(row)
        except Exception as exc:
            self._write_error = exc
            raise
        self.written_t = self.t
        if self.t % 15 == 0:
            self.flush()

    def flush(self) -> None:
        """讓已寫入列可重新開檔讀取；不保證強制終止或掉電耐久性。"""
        if self._write_error is not None:
            raise RuntimeError("樣本寫入失敗，不能確認完整保存") from self._write_error
        if self._sample_file is not None:
            self._sample_file.flush()
            self.flushed_t = self.written_t
            payload = {"epoch": self.epoch, "t": self.t, "dataset": self.meta,
                       "source": self.source, "scenario": None if self.scenario is None else self.scenario["key"],
                       "seed": self.seed, "written_t": self.written_t, "flushed_t": self.flushed_t,
                       "openset_method": self.openset_method, "mahalanobis_method": self.mahalanobis_method,
                       "confidence": self.confidence, "knn_neighbors": self.knn_neighbors}
            path = self.out_dir / f"session_epoch{self.epoch:02d}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    def _snapshot_model(self, tag: str) -> None:
        if self.out_dir is None:
            return
        summary = self.session.monitor.summary()
        payload = {"t": self.t, "tag": tag, "dataset": self.meta, "seed": self.seed, **summary}
        path = self.out_dir / f"model_epoch{self.epoch:02d}_{summary['n_known']:02d}classes.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.flush()
        if self._sample_file is not None:
            self._sample_file.close()
            self._sample_file = None
            self._sample_writer = None
        self._closed = True

    def _refresh_samplers(self) -> None:
        """已知配置只從 holdout 抽（未參與擬合），未知配置從整池抽。"""
        mon = self.session.monitor
        for config, pool in self.pools.items():
            if config in mon.known:
                idx = mon.splits[config].holdout
            else:
                idx = np.arange(len(pool))
            self.samplers[config] = CycleSampler(pool, idx, self.stream_rng)

    # -- 控制 ------------------------------------------------------------------

    def set_source(self, config: str) -> list[dict]:
        if config not in self.pools:
            raise KeyError(config)
        self.scenario = None
        self.source = config
        self.switch_t = self.t
        self.fault_onset_t = self.t if config != HEALTHY else None
        self.trend.rearm()
        self.flush()
        return [_event("info", f"注入來源切換 → {display_name(config)}", self.t)]

    def start_scenario(self, key: str) -> list[dict]:
        scen = SCENARIOS[key]
        self.scenario = {"key": key, "phases": scen["phases"], "idx": 0,
                         "left": scen["phases"][0][2]}
        self.source = HEALTHY
        self.switch_t = self.t
        self.fault_onset_t = self.t + scen["phases"][0][2]
        self.trend.rearm()
        self.flush()
        return [_event("info", f"▶ 啟動{scen['name']}（先回到健康基線）", self.t)]

    def confirm(self) -> dict:
        """操作員確認候選（重擬合在執行緒池中跑）。兩種結果：learned / rejected_known。"""
        result = self.session.confirm()
        if result["action"] == "learned":
            self._refresh_samplers()
            self._snapshot_model(f"learned_{result['config']}")
            self.switch_t = self.t
        self.trend.rearm()
        return result

    # -- 串流 ------------------------------------------------------------------

    def _draw_config(self) -> tuple[str, list[dict]]:
        msgs: list[dict] = []
        if self.scenario is None:
            return self.source, msgs
        sc = self.scenario
        config, ratio, _ticks = sc["phases"][sc["idx"]]
        pick = config if (config != HEALTHY and self.stream_rng.random() < ratio) else HEALTHY
        sc["left"] -= 1
        if sc["left"] <= 0:
            if sc["idx"] + 1 < len(sc["phases"]):
                sc["idx"] += 1
                nxt = sc["phases"][sc["idx"]]
                sc["left"] = nxt[2]
                msgs.append(_event(
                    "info",
                    f"劇本進入下一階段：{display_name(nxt[0])}（混入比例 {int(nxt[1]*100)}%）",
                    self.t,
                ))
            else:
                sc["left"] = 10 ** 9
                msgs.append(_event("info", "劇本播畢，維持最後狀態", self.t))
        return pick, msgs

    def tick(self) -> list[dict]:
        if self._closed or self._write_error is not None:
            raise RuntimeError("session 已關閉或寫入失敗，不接受新樣本")
        config, msgs = self._draw_config()
        was_known = config in self.session.monitor.known
        prev_attempts = self.session.cluster_attempts
        x = self.samplers[config].draw()
        r = self.session.process(x, config)
        tr = self.trend.update(r["score"])
        self.t += 1

        phase_key = "post" if was_known else "pre"
        self.metrics[config][f"{phase_key}_streamed"] += 1
        if r["label"] is None:
            self.metrics[config][f"{phase_key}_flagged"] += 1

        if r["label"] is None:
            verdict = "unknown"
        elif r["label"] == HEALTHY:
            verdict = "healthy"
        else:
            verdict = "known"
        xy = self.session.monitor.project(x)[0]

        d = r["direction"]
        self._log_sample([
            self.epoch, self.t, config, round(float(r["score"]), 4), verdict,
            r["label"] or "", tr["ewma"], tr["cusum"], r["quarantine"],
            len(self.session.monitor.known),
            self.scenario["key"] if self.scenario else "manual",
            round(d["radius"], 3), d["best_ray"] or "", d["best_cos"] if d["best_cos"] is not None else "",
        ])

        dd = r["direction"]
        direction = {
            "ray": dd["best_ray"],
            "ray_display": display_name(dd["best_ray"]) if dd["best_ray"] else None,
            "cos": dd["best_cos"],
            "radius": round(dd["radius"], 2),
        }

        out = [{
            "type": "sample",
            "epoch": self.epoch,
            "t": self.t,
            "score": round(float(r["score"]), 3),
            "verdict": verdict,
            "cls": display_name(r["label"]),
            "truth": config,
            "truth_display": display_name(config),
            "direction": direction,
            "pca": [round(float(xy[0]), 3), round(float(xy[1]), 3)],
            "ewma": tr["ewma"],
            "cusum": tr["cusum"],
            "health": round(min(1.0, 1.0 / max(r["score"], 1e-9)) * 100, 1),
            "quarantine": r["quarantine"],
            "attempts": r["cluster_attempts"],
            "phase": self.phase(),
        }] + msgs

        if r["cluster_attempts"] > prev_attempts and r["cluster_attempts"] % 5 == 0:
            out.append(_event(
                "info",
                f"隔離區 {r['quarantine']} 筆，HDBSCAN 已嘗試 {r['cluster_attempts']} 次"
                f"仍未聚出穩定叢集——此故障樣本較發散，持續累積中"
                f"（密度門檻會隨樣本數自動放寬）",
                self.t,
            ))

        if tr["alarm_now"]:
            latency = None if self.fault_onset_t is None else self.t - self.fault_onset_t
            self.alarms.append({
                "t": self.t, "source": config, "kind": tr["kind"],
                "kind_display": KIND_DISPLAY.get(tr["kind"], tr["kind"]),
                "transition": tr["transition"], "latency": latency,
            })
            extra = "" if latency is None else f"，自故障注入起 {latency} 筆"
            out.append(_event(
                "warn",
                f"⚠ 變化點警報：判別為「{KIND_DISPLAY.get(tr['kind'])}」"
                f"（中間帶停留 {tr['transition']} 筆{extra}）",
                self.t,
            ))
        if r["candidate_new"]:
            c = self.session.candidate
            out.append(_event(
                "warn",
                f"◎ HDBSCAN 在隔離區發現穩定叢集（{c['size']} 筆、純度 {c['purity']*100:.0f}%）"
                f"→ 產生候選新故障，等待操作員確認",
                self.t,
            ))
            out.append(self.state_msg())
        return out

    # -- 狀態 ------------------------------------------------------------------

    def phase(self) -> str:
        if self.session.candidate is not None:
            return "pending"
        if len(self.session.quarantine_X) > 0:
            return "accumulating"
        return "monitoring"

    def state(self) -> dict:
        mon = self.session.monitor
        return {
            "meta": self.meta,
            "openset": self.session.monitor.summary(),
            "t": self.t,
            "epoch": self.epoch,
            "persistence": {"enabled": self.out_dir is not None,
                            "written_t": self.written_t, "flushed_t": self.flushed_t},
            "phase": self.phase(),
            "source": self.source,
            "scenario": None if self.scenario is None else {
                "key": self.scenario["key"],
                "name": SCENARIOS[self.scenario["key"]]["name"],
            },
            "known": [
                {"config": c, "display": display_name(c), "label": label}
                for c, label in mon.known.items()
            ],
            "configs": [
                {"config": c, "display": display_name(c), "n": int(len(p)),
                 "known": c in mon.known}
                for c, p in sorted(self.pools.items(), key=lambda kv: config_sort_key(kv[0]))
            ],
            "quarantine": len(self.session.quarantine_X),
            "min_cluster_size": self.session.min_cluster_size,
            "candidate": None if self.session.candidate is None else {
                k: v for k, v in self.session.candidate.items() if k != "indices"
            },
            "centroids": [
                {"config": c, "display": display_name(c),
                 "x": round(xy[0], 3), "y": round(xy[1], 3)}
                for c, xy in mon.centroids.items()
            ],
            "rays": [
                {"config": r["config"], "display": display_name(r["config"]),
                 "radius": r["radius"]}
                for r in self.session.polar.summary()["rays"]
            ],
            "healthy_threshold": round(next(
                c["threshold"] for c in mon.summary()["classes"]
                if c["config"] == HEALTHY), 2),
        }

    def state_msg(self) -> dict:
        return {"type": "state_patch", **self.state()}

    def metrics_msg(self) -> dict:
        rows = []
        for c, m in sorted(self.metrics.items(), key=lambda kv: config_sort_key(kv[0])):
            if m["pre_streamed"] + m["post_streamed"] == 0:
                continue
            rows.append({
                "config": c,
                "display": display_name(c),
                "pre_streamed": m["pre_streamed"],
                # 學會前：判未知 = 偵測成功
                "pre_rate": None if m["pre_streamed"] == 0
                else round(m["pre_flagged"] / m["pre_streamed"] * 100, 1),
                "post_streamed": m["post_streamed"],
                # 學會後：判為已知 = 認出成功
                "post_rate": None if m["post_streamed"] == 0
                else round((1 - m["post_flagged"] / m["post_streamed"]) * 100, 1),
            })
        return {"type": "metrics", "epoch": self.epoch, "t": self.t,
                "meta": self.meta, "rows": rows, "alarms": self.alarms[-5:]}
