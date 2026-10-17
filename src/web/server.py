"""即時展示伺服器（Tornado + WebSocket）。

啟動：
    venv/bin/python -m web.server --motor T1 --rpm 8000rpm --port 8600
或使用 ./scripts/opt/run_web.sh。

伺服器本身不含實驗邏輯：以固定頻率呼叫 LiveDemo.tick()，把訊息廣播給
所有連線的瀏覽器；重擬合（確認納入、重置、換資料集）丟進執行緒池，
避免卡住串流迴圈。

實驗頁（實驗一～八，可在頁首切換）走 HTTP API，由 web/experiments.py 呼叫各
experiments/ 模組的 run()；用另一個單執行緒池，跑實驗時即時展示照常串流：
    GET  /api/experiments               實驗目錄、資料集、配置清單
    POST /api/experiments/{id}/run      以 JSON 參數執行一次，回傳結果並存檔
    GET  /api/experiments/{id}/stream   邊跑邊畫（Server-Sent Events）：?params=<JSON>&rate=<筆/秒>
    POST /api/experiments/exp2/confirm  實驗二逐筆播放遇到候選時放行確認
    GET  /api/committed/{name}          已提交的矩陣結果（唯讀）

AI 問答（web/assistant.py，LangGraph agent 連本地 OpenAI 相容模型）：
    GET  /api/assistant/status          模型端點是否可連、模型名稱
    POST /api/assistant/ask             {question, history} → SSE：tool / delta / done / error
    AI 讀的「目前狀況」是實驗頁「▶ 執行」「⏵ 邊跑邊畫」的紀錄（PLAYBACK，web/playback.py），
    記憶體沒有時讀 output/web_server/*/experiments/ 的存檔；不讀首頁即時展示。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
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
from web.assistant import Assistant, latest_record, record_label
from web.experiments import ExperimentRunner
from web.live import LiveDemo
from web.playback import PlaybackTracker

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


STATIC_REF = re.compile(r'(src|href)="/static/([^"?#]+)"')


class IndexHandler(tornado.web.RequestHandler):
    """首頁不快取；其中每個 /static/ 引用都補上內容雜湊 ?v=，檔案一改網址就變。

    Cloudflare 會把沒有版本的 .js 快取 4 小時（cf-cache-status: HIT），部署後新伺服器
    搭配舊前端（例：實驗二「邊跑邊畫」在舊 experiments.js 沒有對應畫面，按下去拋例外）。
    """

    def get(self) -> None:
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        html = (Path(self.settings["static_path"]) / "index.html").read_text(encoding="utf-8")
        self.write(STATIC_REF.sub(lambda m: f'{m.group(1)}="{self.static_url(m.group(2))}"', html))


class StaticHandler(tornado.web.StaticFileHandler):
    """帶 ?v= 的網址由 Tornado 給長快取；沒帶版本的一律要求重新驗證，不讓 CDN／瀏覽器拿舊檔。"""

    def set_extra_headers(self, path: str) -> None:
        if not self.get_argument("v", None):
            self.set_header("Cache-Control", "no-cache")


class ExperimentJobs:
    """實驗頁的執行器：一次只跑一個實驗，避免兩個重任務搶 CPU。"""

    def __init__(self, runner: ExperimentRunner):
        self.runner = runner
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current: str | None = None


JOBS: ExperimentJobs | None = None
PLAYBACK = PlaybackTracker()   # 實驗頁各實驗最近一次執行／播放紀錄，AI 問答讀這份


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
            PLAYBACK.record_run_error(exp_id, params, f"{exp_id} 執行失敗：{exc!r}")
            self.reply({"error": f"{exp_id} 執行失敗：{exc!r}"}, 500)
            return
        finally:
            JOBS.current = None
        PLAYBACK.record_run(exp_id, payload)
        logger.info(f"實驗頁完成 {exp_id}（{payload['seconds']} 秒）→ {payload['saved']}")
        self.reply(payload)


class StreamHandler(tornado.web.RequestHandler):
    """以 Server-Sent Events 逐 frame 推送實驗進度；依 rate 控制每秒播放的樣本數。"""

    def initialize(self) -> None:
        self.closed = False
        self.tracking = False   # 開始播放後才記進 PLAYBACK；參數錯誤或忙碌不覆蓋上一次播放

    def on_connection_close(self) -> None:
        self.closed = True
        runner = getattr(JOBS, "runner", None) if JOBS is not None else None
        gate = getattr(runner, "pending_confirm", None) if runner is not None else None
        if gate is not None:
            gate.release("stop")

    async def send(self, frame: dict) -> None:
        if self.tracking:
            PLAYBACK.update(frame)
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
        PLAYBACK.start(exp_id, params)
        self.tracking = True
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
                PLAYBACK.finish("stopped")   # 沒收到 done／error 就結束 = 被中斷


class Exp2ConfirmHandler(_JSONHandler):
    """放行實驗二邊跑邊畫停住的候選。沒有待確認候選時拒絕。"""

    def post(self) -> None:
        gate = getattr(JOBS.runner, "pending_confirm", None) if JOBS is not None else None
        if gate is None or not gate.waiting:
            self.reply({"error": "目前沒有待確認的實驗二候選"}, 409)
            return
        gate.release("confirm")
        self.reply({"ok": True})


class CommittedHandler(_JSONHandler):
    def get(self, name: str) -> None:
        try:
            self.reply(JOBS.runner.committed(name))
        except ValueError as exc:
            self.reply({"error": str(exc)}, 404)


ASSISTANT: Assistant | None = None


class AssistantStatusHandler(_JSONHandler):
    async def get(self) -> None:
        self.reply(await ASSISTANT.status())


class AssistantAskHandler(tornado.web.RequestHandler):
    """以 SSE 串流回答；快照在 IOLoop 執行緒取，模型只讀提問當下的狀態。"""

    def initialize(self) -> None:
        self.closed = False

    def on_connection_close(self) -> None:
        self.closed = True

    async def send(self, frame: dict) -> None:
        self.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n")
        await self.flush()

    async def post(self) -> None:
        self.set_header("Content-Type", "text/event-stream; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        try:
            body = json.loads(self.request.body or b"{}")
            question = body.get("question") if isinstance(body, dict) else None
            history = body.get("history", []) if isinstance(body, dict) else []
            if not isinstance(question, str) or not question.strip() or not isinstance(history, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            await self.send({"event": "error", "error": "需要 JSON：{question: 字串, history: 陣列}"})
            return
        snap = PLAYBACK.snapshot()
        logger.info(f"AI 問答（{record_label(latest_record(snap))}）：{question.strip()[:80]}")
        try:
            async for frame in ASSISTANT.ask(snap, question.strip(), history, load_saved=PLAYBACK.load_saved):
                if self.closed:
                    logger.info("AI 問答被瀏覽器中斷")
                    return
                await self.send(frame)
            await self.send({"event": "done", "as_of": record_label(latest_record(snap))})
        except tornado.iostream.StreamClosedError:
            logger.info("AI 問答連線已關閉")
        except Exception as exc:  # 模型離線或回應異常時回報，不讓伺服器死掉
            logger.exception("AI 問答失敗")
            try:
                await self.send({"event": "error",
                                 "error": f"本地模型呼叫失敗（{ASSISTANT.base_url}）：{exc!r}"})
            except tornado.iostream.StreamClosedError:
                pass


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
    parser.add_argument("--llm-url", default="http://127.0.0.1:8000/v1",
                        help="本地模型 OpenAI 相容端點（vLLM 預設 :8000；見 scripts/opt/run_llm.sh）")
    parser.add_argument("--llm-model", default="", help="模型名稱；留空取端點 /models 的第一個")
    parser.add_argument("--llm-api-key", default="EMPTY")
    args = parser.parse_args()

    log, paths = setup_run("web_server")
    PLAYBACK.results_root = Path(paths.output_dir).parent   # output/web_server：各次啟動的實驗存檔

    datasets = discover_datasets(args.data_root)
    if not datasets:
        raise SystemExit(f"在 {args.data_root} 找不到任何 clean 特徵資料集")
    rpm = args.rpm if args.rpm.endswith("rpm") else f"{args.rpm}rpm"
    chosen = next(
        (d for d in datasets if d["motor"] == args.motor and d["rpm"] == rpm),
        datasets[0],
    )

    global HUB, JOBS, ASSISTANT
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

    ASSISTANT = Assistant(args.llm_url, args.llm_model, args.llm_api_key)
    log.info(f"AI 問答模型端點 {args.llm_url}（model={args.llm_model or '自動偵測'}）")

    app = make_app()
    app.listen(args.port, address=args.bind_address)
    HUB.set_rate(args.rate)
    log.info(f"就緒 → http://localhost:{args.port}  （Ctrl+C 結束）")
    serve(tornado.ioloop.IOLoop.current(), HUB, JOBS)


def make_app(static_dir: Path = STATIC_DIR) -> tornado.web.Application:
    """路由與靜態檔設定；static_path 讓 IndexHandler 能用 static_url() 產生 ?v= 版本網址。"""
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/ws", WSHandler),
        (r"/api/experiments", CatalogHandler),
        (r"/api/experiments/([a-z0-9_]+)/run", RunHandler),
        (r"/api/experiments/exp2/confirm", Exp2ConfirmHandler),
        (r"/api/experiments/([a-z0-9_]+)/stream", StreamHandler),
        (r"/api/committed/([a-z0-9_]+)", CommittedHandler),
        (r"/api/assistant/status", AssistantStatusHandler),
        (r"/api/assistant/ask", AssistantAskHandler),
    ], static_path=str(static_dir), static_handler_class=StaticHandler)


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
