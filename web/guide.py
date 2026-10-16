"""獨立逐步導覽；重用既有LiveDemo，不修改原串流或算法。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import tornado.ioloop
import tornado.web
import tornado.websocket
from loguru import logger

from core.data import HEALTHY, discover_datasets
from core.logger import setup_run
from core.runner import add_openset_args
from experiments.navigation_data_contract import run as data_contract, split_audit
from experiments.exp3_trend import SCENARIOS
from web.live import LiveDemo
from web.server import Hub, serve

ROOT = Path(__file__).resolve().parents[1]


class GuideHub(Hub):
    """只管理操作狀態與答案遮蔽，不更動核心判定。"""
    def __init__(self, datasets, seed=42, rate=4, out_dir=None, detector_options=None):
        super().__init__(None, datasets, rate, seed, out_dir, detector_options)
        self.contract = None
        self.error = None
        self.last_action = None
        self.operation = None
        self.session_id = 0
        self.scenario_end = None
        self.audit_dir = None
        self.preview = None
        self.busy_state = None
        self.catalog_status = "available" if datasets else "no_datasets"

    def candidate_id(self):
        if self.demo is None or self.demo.session.candidate is None:
            return None
        c = self.demo.session.candidate
        identity = json.dumps({"indices": [int(i) for i in c.get("indices", [])],
                               "t_range": c.get("t_range")}, sort_keys=True)
        token = hashlib.sha256(identity.encode()).hexdigest()[:12]
        return f"C{self.session_id}-{self.demo.epoch}-{token}"

    def public_config(self, config):
        if self.contract is None:
            return config
        known = self.demo is not None and config in self.demo.session.monitor.known
        row = next(r for r in self.contract["configs"] if r["config"] == config)
        return {"id": row["id"], "name": config if known or config == HEALTHY else "匿名來源" + row["id"],
                "n": row["n"], "known": known, "sha256": row["sha256"]}

    def persist(self):
        if self.audit_dir is not None and self.demo is not None:
            path = self.audit_dir / f"fit_epoch{self.demo.epoch}_classes{len(self.demo.session.monitor.known)}.json"
            path.write_text(json.dumps(split_audit(self.demo), ensure_ascii=False, indent=2), encoding="utf-8")

    def build(self, motor, rpm):
        ds = next((d for d in self.datasets if d["motor"] == motor and d["rpm"] == rpm), None)
        if ds is None:
            raise ValueError("所選工況不存在")
        contract = data_contract(ds, self.seed)
        self.session_id += 1
        folder = None if self.out_dir is None else Path(self.out_dir) / f"session{self.session_id:03d}"
        if folder:
            folder.mkdir(parents=True, exist_ok=False)
        new_demo = LiveDemo(ds, seed=self.seed, out_dir=folder, **self.detector_options)
        if contract != data_contract(ds, self.seed):
            new_demo.close()
            raise ValueError("載入期間來源SHA改變，拒絕沿用模型")
        if self.demo:
            self.demo.close()
        self.demo = new_demo
        self.contract = contract
        self.audit_dir = folder
        if folder:
            (folder / "data_contract.json").write_text(
                json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
        self.running = False
        self.scenario_end = None
        self.last_action = None
        self.persist()
        return "健康基準建立完成；僅健康train擬合、cal校準"

    def inspect(self, motor, rpm):
        ds = next((d for d in self.datasets if d["motor"] == motor and d["rpm"] == rpm), None)
        if ds is None:
            raise ValueError("所選工況不存在")
        c = data_contract(ds, self.seed)
        self.preview = {k: c[k] for k in ("motor", "rpm", "seed", "feature_dim", "source_type")}
        self.preview["configs"] = [
            {k: row[k] for k in ("id", "n", "sha256")} for row in c["configs"]]
        return "資料已檢查；各來源筆數與SHA可查看，尚未建立新基準"

    def full_state(self):
        if self.busy and self.busy_state is not None:
            step = "檢查資料中" if self.operation == "inspect" else "建基準中" if self.operation == "build" else "更新中"
            return {**self.busy_state, "busy": True, "running": False, "step": step}
        step = "待建基準" if self.datasets else "缺少資料"
        state = {"type": "state", "datasets": [{"motor": d["motor"], "rpm": d["rpm"]} for d in self.datasets],
                 "seed": self.seed, "busy": self.busy, "running": self.running,
                 "rate": self.rate, "error": self.error, "session_id": self.session_id,
                 "last_action": self.last_action, "t": 0, "epoch": 0, "candidate": None,
                 "configs": [], "quarantine": 0, "attempts": 0,
                 "evidence": "CSV舊資料展示；來源獨立性UNKNOWN；fresh INCOMPLETE"}
        state["preview"] = self.preview
        state["catalog_status"] = self.catalog_status
        if self.demo:
            d = self.demo
            step = "待確認" if self.candidate_id() else "累積" if d.session.quarantine_X else "監測中"
            if not self.running:
                step = "已更新" if self.last_action == "learned" else "可輸入" if d.t == 0 else "暫停"
            state.update(t=d.t, epoch=d.epoch, meta=d.meta, openset=d.session.monitor.summary(),
                         persistence=d.state()["persistence"],
                         metrics=self.public_metrics(d.metrics_msg()),
                         configs=[self.public_config(c["config"]) for c in self.contract["configs"]],
                         source=self.public_config(d.source)["id"],
                         known=list(d.session.monitor.known), quarantine=len(d.session.quarantine_X),
                         attempts=d.session.cluster_attempts, fit_audit=split_audit(d),
                         data={"feature_dim": 105, "source_type": "CSV重播", "seed": self.seed},
                         scenario=None if d.scenario is None else d.scenario["key"])
            if d.scenario:
                state["source"] = "劇本" + d.scenario["key"] + "（混合匿名來源）"
            state["scenario_completed"] = bool(d.scenario and self.scenario_end is None and not self.running)
            if self.candidate_id():
                state["candidate"] = {"id": self.candidate_id(), "size": int(d.session.candidate["size"]),
                                      "t_range": d.session.candidate.get("t_range")}
                if not self.busy and not self.error:
                    step = "待確認"
        if self.busy:
            step = "檢查資料中" if self.operation == "inspect" else "建基準中" if self.operation == "build" else "更新中"
        if self.error:
            step = "錯誤"
        state["step"] = step
        return state

    def public_metrics(self, msg):
        return {"type": "metrics", "session_id": self.session_id,
                "epoch": self.demo.epoch, "t": msg.get("t", self.demo.t),
                "rows": [
                    {**{k: v for k, v in row.items() if k not in ("config", "display")},
                     "source": self.public_config(row["config"])} for row in msg["rows"]],
                "alarms": [{k: v for k, v in alarm.items() if k not in ("source", "kind_display")}
                           for alarm in msg["alarms"]]}

    def broadcast(self, msg):
        kind = msg.get("type")
        if kind in ("state", "state_patch"):
            msg = self.full_state()
        elif kind == "sample":
            msg = {k: v for k, v in msg.items() if k not in ("truth", "truth_display", "health", "cls")}
            msg.update(session_id=self.session_id, epoch=self.demo.epoch)
            msg["prediction"] = {"unknown": "未知異常：原因未確認", "healthy": "接受為健康",
                                 "known": "接受為已知配置"}[msg["verdict"]]
        elif kind == "metrics":
            msg = self.public_metrics(msg)
        elif kind == "event" and not msg.get("guide_safe"):
            # 既有事件可能帶配置答案／純度，僅輸出可公開的動作提示。
            msg = {"type": "event", "level": msg.get("level", "info"), "t": msg.get("t", 0),
                   "text": "核心有新事件：請查看候選狀態、隔離數與趨勢警報"}
        super().broadcast(msg)

    def event(self, level, text):
        if level == "error" and text.startswith("串流錯誤"):
            self.error = text
            self.running = False
        logger.info(text)
        super().broadcast({"type": "event", "guide_safe": True, "level": level, "text": text})

    def pause(self, *, publish=True):
        super().pause(publish=publish)

    def tick(self):
        if not self.demo or self.busy:
            return
        super().tick()
        if self.scenario_end is not None and self.demo.t >= self.scenario_end:
            self.pause()
            self.scenario_end = None
            self.event("info", "劇本窗口播畢，已暫停；這是劇本結果，不是物理時間")
            self.broadcast(self.demo.metrics_msg())
        self.broadcast(self.full_state())

    def close_client(self, client):
        super().close_client(client)


class GuideSocket(tornado.websocket.WebSocketHandler):
    @property
    def hub(self):
        return self.application.settings["guide_hub"]

    def check_origin(self, origin):
        return urlparse(origin).netloc == self.request.host

    def open(self):
        self.hub.clients.add(self)
        self.hub.broadcast(self.hub.full_state())

    def on_close(self):
        self.hub.close_client(self)

    async def heavy(self, command, fn):
        h = self.hub
        h.pause(publish=False)
        h.busy_state = h.full_state()
        h.busy, h.operation, h.running = True, command, False
        h.broadcast(h.full_state())
        try:
            result = await tornado.ioloop.IOLoop.current().run_in_executor(h.executor, fn)
            h.error = None
            h.event("success", str(result))
        except Exception as exc:
            logger.exception("導覽操作失敗")
            h.error = (f"資料讀取失敗（{type(exc).__name__}）；請管理者確認讀取權限與正式資料。"
                       if isinstance(exc, OSError) else
                       f"操作失敗（{type(exc).__name__}）；可重新確認所選資料，仍失敗請查看伺服器日誌。")
            h.event("error", h.error)
        finally:
            h.busy, h.operation = False, None
        h.pause(publish=False)
        h.broadcast(h.full_state())

    async def on_message(self, raw):
        h = self.hub
        try:
            msg = json.loads(raw)
            cmd = msg["cmd"]
            if h.busy:
                raise ValueError("工作進行中，拒絕重複操作")
            if cmd == "inspect":
                await self.heavy(cmd, lambda: h.inspect(msg["motor"], msg["rpm"]))
            elif cmd == "build":
                await self.heavy(cmd, lambda: h.build(msg["motor"], msg["rpm"]))
            elif h.demo is None:
                raise ValueError("請先選資料並建立健康基準")
            elif cmd == "pause":
                h.pause(publish=False)
            elif cmd == "reset":
                def reset():
                    h.demo._build()
                    h.last_action, h.scenario_end = None, None
                    h.persist()
                    return "已重設健康基準；新epoch，保留舊紀錄"
                await self.heavy(cmd, reset)
            elif h.error:
                raise ValueError("目前有錯誤，請明確重設或重建後再監測")
            elif cmd == "start":
                h.running = True
            elif cmd == "rate":
                h.set_rate(msg["value"])
            elif cmd == "source":
                row = next(r for r in h.contract["configs"] if r["id"] == msg["id"])
                h.demo.set_source(row["config"])
                h.scenario_end = None
                h.event("info", "訊號來源已切換；匿名不代表推論出故障名稱")
            elif cmd == "scenario":
                name = msg["name"]
                if name not in SCENARIOS or any(c not in h.demo.pools for c, _, _ in SCENARIOS[name]["phases"]):
                    raise ValueError("劇本所需配置不完整")
                h.demo.start_scenario(name)
                h.scenario_end = h.demo.t + sum(n for _, _, n in SCENARIOS[name]["phases"])
                h.running = True
            elif cmd == "confirm":
                if not h.candidate_id() or msg.get("candidate_id") != h.candidate_id():
                    raise ValueError("沒有可確認候選，或候選ID已過期")
                def confirm():
                    result = h.demo.confirm()
                    h.last_action = result["action"]
                    h.persist()
                    return f"模擬操作員：{result['action']}，配置 {result.get('config')}；完整配置池重擬合／已知群退回"
                await self.heavy(cmd, confirm)
            else:
                raise ValueError("不支援的操作")
        except Exception as exc:
            h.event("error", str(exc))
        h.broadcast(h.full_state())


def application(hub):
    return tornado.web.Application([
        (r"/ws", GuideSocket),
        (r"/", tornado.web.RedirectHandler, {"url": "/assets/guide.html"}),
        (r"/assets/(guide\.(?:html|js|css))", tornado.web.StaticFileHandler, {"path": str(ROOT / "web/static")}),
        (r"/docs/(.*)", tornado.web.StaticFileHandler, {"path": str(ROOT / "docs/navigation")}),
    ], guide_hub=hub)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--port", type=int, default=8601)
    parser.add_argument("--rate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    add_openset_args(parser)
    args = parser.parse_args()
    log, paths = setup_run("web_guide")
    hub = GuideHub(discover_datasets(args.data_root), args.seed, args.rate, paths.output_dir,
                   {"openset_method": args.openset_method, "mahalanobis_method": args.method,
                    "confidence": args.confidence, "knn_neighbors": args.knn_neighbors})
    if not Path(args.data_root).is_dir():
        hub.catalog_status = "data_root_missing"
    application(hub).listen(args.port, address="127.0.0.1")
    hub.set_rate(args.rate)
    log.info(f"逐步導覽：http://127.0.0.1:{args.port}；尚未擬合")
    serve(tornado.ioloop.IOLoop.current(), hub)


if __name__ == "__main__":
    main()
