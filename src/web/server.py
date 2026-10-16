"""即時展示伺服器（Tornado + WebSocket）。

啟動：
    venv/bin/python -m web.server --motor T1 --rpm 8000rpm --port 8600
或使用專案根目錄的 ./run_web.sh。

伺服器本身不含實驗邏輯：以固定頻率呼叫 LiveDemo.tick()，把訊息廣播給
所有連線的瀏覽器；重擬合（確認納入、重置、換資料集）丟進執行緒池，
避免卡住串流迴圈。

實驗頁（實驗一～八，可在頁首切換）走 HTTP API，由 web/experiments.py 呼叫各
experiments/ 模組的 run()；用另一個單執行緒池，跑實驗時即時展示照常串流：
    GET  /api/experiments               實驗目錄、資料集、配置清單
    POST /api/experiments/{id}/run      以 JSON 參數執行一次，回傳結果並存檔
    GET  /api/experiments/{id}/stream   邊跑邊畫（Server-Sent Events）：?params=<JSON>&rate=<筆/秒>
    GET  /api/committed/{name}          已提交的矩陣結果（唯讀）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tornado.ioloop
import tornado.iostream
import tornado.web
import tornado.websocket
from loguru import logger

from core.data import discover_datasets
from core.logger import setup_run
from core.runner import add_openset_args
from web.experiments import ExperimentRunner
from web.live import LiveDemo

STATIC_DIR = Path(__file__).parent / "static"


class Hub:
    """連線集合 + 串流節拍器 + 重任務執行緒池。"""

    def __init__(
        self,
        demo: LiveDemo,
        datasets: list[dict],
        rate: int,
        seed: int,
        out_dir=None,
        detector_options: dict | None = None,
    ):
        self.demo = demo
        self.datasets = datasets
        self.seed = seed
        self.out_dir = out_dir
        self.detector_options = detector_options or {}
        self.clients: set[tornado.websocket.WebSocketHandler] = set()
        self.running = False
        self.busy = False
        self.error = None
        self.closed = False
        self._busy_state = None
        self.rate = rate
        self.periodic: tornado.ioloop.PeriodicCallback | None = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    # -- 廣播 ------------------------------------------------------------------

    def broadcast(self, msg: dict) -> None:
        data = json.dumps(msg, ensure_ascii=False)
        for client in list(self.clients):
            try:
                client.write_message(data)
            except Exception:
                self.clients.discard(client)

    def full_state(self) -> dict:
        if self.busy and self._busy_state is not None:
            return {**self._busy_state, "busy": True, "running": False}
        state = self.demo.state()
        state.update({
            "type": "state",
            "running": self.running,
            "busy": self.busy,
            "error": self.error,
            "rate": self.rate,
            "datasets": [{"motor": d["motor"], "rpm": d["rpm"]} for d in self.datasets],
            "metrics": self.demo.metrics_msg(),
        })
        return state

    def event(self, level: str, text: str) -> None:
        logger.log({"error": "ERROR", "warn": "WARNING"}.get(level, "INFO"), f"[t={self.demo.t}] {text}")
        self.broadcast({"type": "event", "level": level, "text": text, "t": self.demo.t})

    # -- 節拍 ------------------------------------------------------------------

    def pause(self, *, publish=True) -> None:
        """IOLoop 的同步 tick 先完成；停止後才確認尾批資料已保存。"""
        self.running = False
        if self.busy:
            raise RuntimeError("工作進行中，尚不能確認保存完成")
        if self.demo is not None:
            try:
                self.demo.flush()
            except Exception as exc:
                self.error = f"保存失敗：{exc!r}"
                raise
            if publish:
                self.broadcast(self.full_state())

    def close_client(self, client) -> None:
        self.clients.discard(client)
        if not self.clients:
            self.running = False
            if not self.busy:
                try:
                    self.pause()
                except Exception:
                    logger.exception("最後連線離開時保存失敗")

    def close(self) -> None:
        """先停止節拍並等重任務結束，再關閉 session；可重複呼叫。"""
        self.running = False
        if self.periodic is not None:
            self.periodic.stop()
        self.executor.shutdown(wait=True)
        if self.demo is not None:
            self.demo.close()
        self.closed = True

    def tick(self) -> None:
        if not self.running or self.busy or self.closed or self.error:
            return
        try:
            msgs = self.demo.tick()
        except Exception as exc:  # 展示現場永不讓伺服器死掉
            self.running = False
            self.error = f"串流錯誤：{exc!r}"
            self.event("error", f"串流錯誤，已暫停：{exc!r}")
            self.broadcast(self.full_state())
            return
        for msg in msgs:
            self.broadcast(msg)
        if self.demo.t % 15 == 0:
            self.broadcast(self.demo.metrics_msg())

    def set_rate(self, hz: float) -> None:
        self.rate = max(1, min(10, int(hz)))
        if self.periodic is not None:
            self.periodic.stop()
        self.periodic = tornado.ioloop.PeriodicCallback(self.tick, 1000 / self.rate)
        self.periodic.start()


HUB: Hub | None = None


class IndexHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        self.write((STATIC_DIR / "index.html").read_bytes())


class ExperimentJobs:
    """實驗頁的執行器：一次只跑一個實驗，避免兩個重任務搶 CPU。"""

    def __init__(self, runner: ExperimentRunner):
        self.runner = runner
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current: str | None = None


JOBS: ExperimentJobs | None = None


class _JSONHandler(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("Cache-Control", "no-store")

    def reply(self, payload: dict, status: int = 200) -> None:
        self.set_status(status)
        self.finish(json.dumps(payload, ensure_ascii=False, allow_nan=False))


class CatalogHandler(_JSONHandler):
    def get(self) -> None:
        payload = JOBS.runner.catalog()
        payload["running"] = JOBS.current
        self.reply(payload)


class RunHandler(_JSONHandler):
    async def post(self, exp_id: str) -> None:
        if JOBS.current is not None:
            self.reply({"error": f"{JOBS.current} 執行中，請稍候"}, 409)
            return
        try:
            params = json.loads(self.request.body or b"{}")
        except json.JSONDecodeError:
            self.reply({"error": "參數不是合法 JSON"}, 400)
            return
        JOBS.current = exp_id
        logger.info(f"實驗頁執行 {exp_id} {params}")
        try:
            loop = tornado.ioloop.IOLoop.current()
            payload = await loop.run_in_executor(JOBS.executor, JOBS.runner.run, exp_id, params)
        except ValueError as exc:
            self.reply({"error": str(exc)}, 400)
            return
        except Exception as exc:  # 展示現場永不讓伺服器死掉
            logger.exception(f"實驗 {exp_id} 失敗")
            self.reply({"error": f"{exp_id} 執行失敗：{exc!r}"}, 500)
            return
        finally:
            JOBS.current = None
        logger.info(f"實驗頁完成 {exp_id}（{payload['seconds']} 秒）→ {payload['saved']}")
        self.reply(payload)


class StreamHandler(tornado.web.RequestHandler):
    """以 Server-Sent Events 逐 frame 推送實驗進度；依 rate 控制每秒播放的樣本數。"""

    def initialize(self) -> None:
        self.closed = False

    def on_connection_close(self) -> None:
        self.closed = True

    async def send(self, frame: dict) -> None:
        self.write(f"data: {json.dumps(frame, ensure_ascii=False, allow_nan=False)}\n\n")
        await self.flush()

    async def get(self, exp_id: str) -> None:
        self.set_header("Content-Type", "text/event-stream; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        try:
            params = json.loads(self.get_argument("params", "{}"))
            rate = float(self.get_argument("rate", "40"))
            if not isinstance(params, dict) or not math.isfinite(rate):
                raise ValueError("參數必須是物件且速率必須有限")
            rate = max(1.0, min(1000.0, rate))
        except (json.JSONDecodeError, ValueError):
            await self.send({"event": "error", "error": "參數不是合法 JSON 或速率不是數字"})
            return
        if JOBS.current is not None:
            await self.send({"event": "error", "error": f"{JOBS.current} 執行中，請稍候"})
            return
        JOBS.current = exp_id
        logger.info(f"實驗頁串流 {exp_id} {params} rate={rate}")
        loop = tornado.ioloop.IOLoop.current()
        gen = None
        try:
            gen = JOBS.runner.stream(exp_id, params)
            budget = 0.0
            while not self.closed:
                item = await loop.run_in_executor(JOBS.executor, next, gen, None)
                if item is None:
                    break
                frame, units = item
                await self.send(frame)
                if frame["event"] == "done":
                    logger.info(f"實驗頁串流完成 {exp_id} → {frame['payload']['saved']}")
                budget += units / rate
                if budget >= 0.02:  # 累積到 20 ms 才睡，避免每筆都排程
                    await asyncio.sleep(budget)
                    budget = 0.0
            if self.closed:
                logger.info(f"實驗頁串流 {exp_id} 被瀏覽器中斷")
        except ValueError as exc:
            await self.send({"event": "error", "error": str(exc)})
        except tornado.iostream.StreamClosedError:
            logger.info(f"實驗頁串流 {exp_id} 連線已關閉")
        except Exception as exc:  # 展示現場永不讓伺服器死掉
            logger.exception(f"實驗 {exp_id} 串流失敗")
            try:
                await self.send({"event": "error", "error": f"{exp_id} 執行失敗：{exc!r}"})
            except tornado.iostream.StreamClosedError:
                pass
        finally:
            try:
                if gen is not None:
                    await loop.run_in_executor(JOBS.executor, gen.close)
            except Exception:
                logger.exception("串流產生器關閉失敗；仍釋放執行鎖")
            finally:
                JOBS.current = None


class CommittedHandler(_JSONHandler):
    def get(self, name: str) -> None:
        try:
            self.reply(JOBS.runner.committed(name))
        except ValueError as exc:
            self.reply({"error": str(exc)}, 404)


class WSHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin: str) -> bool:  # 區網展示用
        return True

    def open(self) -> None:
        HUB.clients.add(self)
        state = HUB.full_state()
        self.write_message(json.dumps(state, ensure_ascii=False))
        self.write_message(json.dumps(state["metrics"], ensure_ascii=False))

    def on_close(self) -> None:
        HUB.close_client(self)

    async def on_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            await self.dispatch(msg)
        except Exception as exc:
            HUB.event("error", f"指令失敗：{exc!r}")
            HUB.broadcast(HUB.full_state())

    async def dispatch(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        if HUB.busy:
            raise ValueError("工作進行中，拒絕重複操作")
        if HUB.error and cmd not in ("pause", "reset", "dataset"):
            raise ValueError("目前有錯誤，請明確重設／重建後再監測")
        if cmd == "start":
            HUB.running = True
            HUB.event("info", "▶ 串流開始")
        elif cmd == "pause":
            HUB.pause(publish=False)
            HUB.event("info", "⏸ 串流暫停；已寫入樣本完成 flush")
        elif cmd == "rate":
            HUB.set_rate(msg.get("value", 4))
        elif cmd == "set_source":
            for m in HUB.demo.set_source(msg["config"]):
                HUB.broadcast(m)
        elif cmd == "scenario":
            HUB.running = True
            for m in HUB.demo.start_scenario(msg["name"]):
                HUB.broadcast(m)
        elif cmd == "confirm":
            await self._heavy(self._do_confirm, "重訓練")
            return
        elif cmd == "reset":
            await self._heavy(self._do_reset, "重置")
            return
        elif cmd == "dataset":
            await self._heavy(
                lambda: self._do_dataset(msg["motor"], msg["rpm"]), "載入資料集"
            )
            return
        HUB.broadcast(HUB.full_state())

    async def _heavy(self, fn, label: str) -> None:
        if HUB.busy:
            HUB.event("warn", f"{label}略過：另一項工作進行中")
            return
        HUB.pause(publish=False)
        HUB._busy_state = HUB.full_state()
        HUB.busy = True
        HUB.broadcast(HUB.full_state())
        loop = tornado.ioloop.IOLoop.current()
        t0 = time.time()
        try:
            done_text = await loop.run_in_executor(HUB.executor, fn)
            HUB.error = None
            HUB.event("success", f"{done_text}（{time.time() - t0:.1f} 秒）")
        except Exception as exc:
            HUB.event("error", f"{label}失敗：{exc!r}")
        finally:
            HUB.busy = False
        HUB.pause(publish=False)
        HUB.broadcast(HUB.full_state())
        HUB.broadcast(HUB.demo.metrics_msg())

    # 下列三個在執行緒池中執行，回傳完成訊息文字
    @staticmethod
    def _do_confirm() -> str:
        res = HUB.demo.confirm()
        if res["action"] == "rejected_known":
            return (
                f"✋ 操作員辨識：候選叢集其實是已知類別「{res['display']}」的邊界樣本"
                f"（{res['size']} 筆——校準閾值天生會讓少量已知樣本被判未知，累積後自成一叢）"
                f"→ 已退回並清出隔離區，量尺不變"
            )
        return (
            f"✔ 操作員確認：叢集多數為「{res['display']}」（純度 {res['purity']*100:.0f}%）"
            f"→ 已納入為類別 {res['label']}，量尺擴張、監測器重新擬合完成"
        )

    @staticmethod
    def _do_reset() -> str:
        HUB.demo._build()
        HUB.running = False
        return f"↺ 已重置回階段 0（只認識健康）；session 紀錄換到 epoch {HUB.demo.epoch}"

    @staticmethod
    def _do_dataset(motor: str, rpm: str) -> str:
        for ds in HUB.datasets:
            if ds["motor"] == motor and ds["rpm"] == rpm:
                HUB.demo.close()
                HUB.demo = LiveDemo(
                    ds,
                    seed=HUB.seed,
                    out_dir=HUB.out_dir,
                    **HUB.detector_options,
                )
                HUB.running = False
                return f"已載入資料集 {motor}/{rpm}，回到階段 0"
        raise KeyError(f"{motor}/{rpm}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--motor", default="T1")
    parser.add_argument("--rpm", default="8000rpm")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--bind-address", default="0.0.0.0", help="監聽地址；本機驗證請用 127.0.0.1")
    parser.add_argument("--rate", type=int, default=4, help="每秒樣本數（1-10）")
    parser.add_argument("--seed", type=int, default=42)
    add_openset_args(parser)
    args = parser.parse_args()

    log, paths = setup_run("web_server")

    datasets = discover_datasets(args.data_root)
    if not datasets:
        raise SystemExit(f"在 {args.data_root} 找不到任何 clean 特徵資料集")
    rpm = args.rpm if args.rpm.endswith("rpm") else f"{args.rpm}rpm"
    chosen = next(
        (d for d in datasets if d["motor"] == args.motor and d["rpm"] == rpm),
        datasets[0],
    )

    global HUB, JOBS
    detector_options = {
        "openset_method": args.openset_method,
        "mahalanobis_method": args.method,
        "confidence": args.confidence,
        "knn_neighbors": args.knn_neighbors,
    }
    log.info(f"載入資料集 {chosen['motor']}/{chosen['rpm']} 並擬合健康基準…")
    log.info(f"Open Set method={args.openset_method}（Mahalanobis estimator={args.method}）")
    log.info(f"session 紀錄（逐筆樣本 + 模型快照）→ {paths.output_dir}")
    HUB = Hub(
        LiveDemo(chosen, seed=args.seed, out_dir=paths.output_dir, **detector_options),
        datasets, args.rate, args.seed, out_dir=paths.output_dir,
        detector_options=detector_options,
    )

    JOBS = ExperimentJobs(ExperimentRunner(
        datasets, args.data_root, out_dir=paths.output_dir, detector_options=detector_options,
    ))

    app = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/ws", WSHandler),
        (r"/api/experiments", CatalogHandler),
        (r"/api/experiments/([a-z0-9_]+)/run", RunHandler),
        (r"/api/experiments/([a-z0-9_]+)/stream", StreamHandler),
        (r"/api/committed/([a-z0-9_]+)", CommittedHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": str(STATIC_DIR)}),
    ])
    app.listen(args.port, address=args.bind_address)
    HUB.set_rate(args.rate)
    log.info(f"就緒 → http://localhost:{args.port}  （Ctrl+C 結束）")
    serve(tornado.ioloop.IOLoop.current(), HUB, JOBS)


def serve(loop, hub, jobs=None) -> None:
    """正常 stop 與 Ctrl+C 共用收尾；保存錯誤仍向呼叫端傳遞。"""
    try:
        loop.start()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            hub.close()
        finally:
            if jobs is not None:
                jobs.executor.shutdown(wait=True)
        logger.info("伺服器結束；session 已關閉")


if __name__ == "__main__":
    main()
