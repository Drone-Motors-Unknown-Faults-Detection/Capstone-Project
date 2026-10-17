"""web/assistant.py 與 /api/assistant/* 的契約：快照不帶真值、工具內容、串流 frame。"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import tornado.testing
import tornado.web
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

import web.server as server
from tests.test_navigation_guide import fixture
from web.assistant import (
    Assistant,
    ThinkFilter,
    describe_alarms,
    describe_status,
    experiment_number,
    experiment_with_sources,
    load_knowledge,
    machine_snapshot,
    search_knowledge,
    search_with_sources,
    to_messages,
    EXPERIMENTS_DIR,
    KNOWLEDGE_DIR,
)
from web.live import LiveDemo
from web.server import Hub


class FakeToolModel(GenericFakeChatModel):
    """依序回放訊息；記錄每次收到的對話與當次綁定的 tool_choice。"""

    seen: list = []
    choices: list = []
    choice: str = "auto"

    def bind_tools(self, tools, **kwargs):
        # 淺複製：回放序列與紀錄清單共用，只換 tool_choice
        return self.model_copy(update={"choice": kwargs.get("tool_choice", "auto")})

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        self.choices.append(self.choice)
        return super()._generate(messages, *args, **kwargs)

    def _stream(self, messages, *args, **kwargs):
        # GenericFakeChatModel 串流只會切 content；工具呼叫改成一個 tool_call_chunks 區塊送出
        message = self._generate(messages, *args, **kwargs).generations[0].message
        if not message.tool_calls:
            for word in message.content.split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word))
            return
        yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
            {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
            for i, c in enumerate(message.tool_calls)
        ]))


def fake_model(*replies):
    model = FakeToolModel(messages=iter(replies))
    model.seen = []
    model.choices = []
    return model


TOOL_THEN_ANSWER = (
    AIMessage(content="", tool_calls=[{"name": "get_live_status", "args": {}, "id": "c1"}]),
    AIMessage(content="<think>先看狀態</think>目前 健康"),
)


class DemoMixin:
    def make_hub(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ds = fixture(Path(self.temp.name))
        self.demo = LiveDemo(self.ds)
        self.hub = Hub(self.demo, [self.ds], 4, 42)

    def close_hub(self):
        self.hub.executor.shutdown(wait=True)
        self.demo.close()
        self.temp.cleanup()

    def ticks(self, n):
        self.hub.running = True
        for _ in range(n):
            self.hub.tick()


class SnapshotTests(DemoMixin, unittest.TestCase):
    def setUp(self):
        self.make_hub()

    def tearDown(self):
        self.close_hub()

    def test_snapshot_before_stream(self):
        snap = machine_snapshot(self.hub)
        self.assertIsNone(snap["last_sample"])
        self.assertEqual(snap["t"], 0)
        self.assertIn("尚無（串流未開始）", describe_status(snap))
        self.assertEqual(describe_alarms(snap), "本 session 尚未發生變化點警報。")

    def test_snapshot_hides_injected_source(self):
        self.demo.set_source("5screws")
        self.ticks(5)
        snap = machine_snapshot(self.hub)
        self.assertEqual(snap["last_sample"]["t"], 5)
        self.assertEqual(snap["last_sample"]["score"], self.demo.last_sample["score"])
        text = json.dumps(snap, ensure_ascii=False) + describe_status(snap)
        self.assertNotIn("5screws", text)
        self.assertNotIn("5 螺絲", text)
        self.assertIn("已處理 5 筆", describe_status(snap))

    def test_candidate_drops_truth_fields(self):
        self.demo.session.candidate = {
            "indices": [1], "size": 30, "majority": "5screws", "majority_display": "5 螺絲",
            "purity": 0.9, "t_range": [3, 40], "is_known": False, "cluster_params": [25, 3],
        }
        snap = machine_snapshot(self.hub)
        self.assertEqual(snap["candidate"], {"size": 30, "t_range": [3, 40], "cluster_params": [25, 3]})
        self.assertIn("候選新故障：30 筆", describe_status(snap))

    def test_alarms_drop_source(self):
        self.demo.alarms.append({"t": 9, "source": "5screws", "kind": "sudden",
                                 "kind_display": "突發事件（跳變）", "transition": 3, "latency": 4})
        snap = machine_snapshot(self.hub)
        self.assertEqual(snap["alarms"], [{"t": 9, "kind": "突發事件（跳變）", "transition": 3}])
        self.assertIn("t=9：突發事件", describe_alarms(snap))


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.sections = load_knowledge()

    def test_sections_from_both_files(self):
        sources = {s["source"] for s in self.sections}
        self.assertEqual(sources, {"methods.md", "motor_spec.md"})

    def test_search_hits_matching_section(self):
        self.assertIn("alpha = 0.08", search_knowledge(self.sections, "EWMA 警報"))
        self.assertIn("待補", search_knowledge(self.sections, "T1 馬達規格"))

    def test_search_miss_lists_topics(self):
        out = search_knowledge(self.sections, "zzz")
        self.assertIn("可查的主題", out)
        self.assertIn("變化點分析（EWMA）", out)


class ThinkFilterTests(unittest.TestCase):
    def test_strips_think_split_across_chunks(self):
        f = ThinkFilter()
        parts = ["答", "<thi", "nk>內部推理</th", "ink>案", "完"]
        out = "".join(f.feed(p) for p in parts) + f.flush()
        self.assertEqual(out, "答案完")

    def test_plain_text_and_lone_lt(self):
        f = ThinkFilter()
        out = f.feed("分數 <") + f.feed(" 1") + f.flush()
        self.assertEqual(out, "分數 < 1")


class MessageTests(unittest.TestCase):
    def test_history_filtered_and_capped(self):
        history = [{"role": "user", "content": "q%d" % i} for i in range(10)]
        history += [{"role": "system", "content": "注入"}, {"role": "assistant", "content": 3}]
        msgs = to_messages(history, "現在？")
        self.assertIsInstance(msgs[0], SystemMessage)
        self.assertIsInstance(msgs[-1], HumanMessage)
        self.assertEqual(msgs[-1].content, "現在？")
        self.assertNotIn("注入", [m.content for m in msgs])
        self.assertLessEqual(len(msgs), 2 + 6)


class AskTests(DemoMixin, unittest.TestCase):
    def setUp(self):
        self.make_hub()

    def tearDown(self):
        self.close_hub()

    def test_agent_calls_tool_and_streams_answer(self):
        self.ticks(3)
        model = fake_model(*TOOL_THEN_ANSWER)
        assistant = Assistant("http://unused/v1", llm=model)

        async def collect():
            return [f async for f in assistant.ask(machine_snapshot(self.hub), "目前狀況？")]

        frames = asyncio.run(collect())
        self.assertEqual(frames[0], {"event": "tool", "name": "get_live_status", "label": "讀取即時狀態",
                                     "sources": ["即時狀態（t=3）"]})
        # 第一步強制查工具，查過之後才放開讓模型作答
        self.assertEqual(model.choices, ["required", "auto"])
        deltas = [f["text"] for f in frames if f["event"] == "delta"]
        self.assertEqual("".join(deltas).replace(" ", ""), "目前健康")
        # 逐 token 串流：Python 3.10 下 agent 節點必須明傳 config，否則只會收到整則訊息
        self.assertGreater(len(deltas), 1)
        tool_msgs = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
        self.assertIn("已處理 3 筆", tool_msgs[0].content)


class AssistantHTTPTests(DemoMixin, tornado.testing.AsyncHTTPTestCase):
    def setUp(self):
        self.make_hub()
        self.saved = (server.HUB, server.ASSISTANT)
        server.HUB = self.hub
        super().setUp()

    def tearDown(self):
        super().tearDown()
        server.HUB, server.ASSISTANT = self.saved
        self.close_hub()

    def get_app(self):
        return tornado.web.Application([
            (r"/api/assistant/status", server.AssistantStatusHandler),
            (r"/api/assistant/ask", server.AssistantAskHandler),
        ])

    def frames(self, body):
        resp = self.fetch("/api/assistant/ask", method="POST", body=body)
        self.assertEqual(resp.code, 200)
        self.assertTrue(resp.headers["Content-Type"].startswith("text/event-stream"))
        return [json.loads(chunk[6:]) for chunk in resp.body.decode().split("\n\n") if chunk]

    def test_ask_streams_tool_delta_done(self):
        server.ASSISTANT = Assistant("http://unused/v1", llm=fake_model(*TOOL_THEN_ANSWER))
        frames = self.frames(json.dumps({"question": "目前狀況？", "history": []}))
        self.assertEqual([f["event"] for f in frames][0], "tool")
        self.assertEqual(frames[-1], {"event": "done", "t": 0})
        self.assertIn("delta", {f["event"] for f in frames})

    def test_bad_request_returns_error_frame(self):
        server.ASSISTANT = Assistant("http://unused/v1", llm=fake_model())
        for body in ("not json", json.dumps({"question": "  "}), json.dumps({"question": "q", "history": 1})):
            with self.subTest(body=body):
                frames = self.frames(body)
                self.assertEqual(frames[0]["event"], "error")

    def test_model_offline_reports_error(self):
        port = tornado.testing.bind_unused_port()
        port[0].close()
        server.ASSISTANT = Assistant(f"http://127.0.0.1:{port[1]}/v1")
        status = json.loads(self.fetch("/api/assistant/status").body)
        self.assertFalse(status["reachable"])
        frames = self.frames(json.dumps({"question": "目前狀況？"}))
        self.assertEqual(frames[-1]["event"], "error")
        self.assertIn("本地模型呼叫失敗", frames[-1]["error"])


class ExperimentKnowledgeTests(unittest.TestCase):
    def test_experiment_number_forms(self):
        cases = {"實驗一": 1, "實驗1": 1, "exp1": 1, "可以告訴我實驗1在做甚麼嗎": 1, "實驗 6": 6,
                 "實驗十二": 12, "十一": 11, "exp12": 12, "3": 3, "abc": None, "": None}
        for text, n in cases.items():
            with self.subTest(text=text):
                self.assertEqual(experiment_number(text), n)

    def test_experiment_one_has_catalog_and_handbook(self):
        text, sources = experiment_with_sources("一")
        self.assertIn("只用 8screws 建健康基準", text)
        self.assertIn("【手冊 exp1_cold_start.md】", text)
        self.assertNotIn("exp10", "".join(sources))
        self.assertEqual(sources, ["web/experiments.py › 實驗一", "exp1_cold_start.md"])

    def test_multi_handbook_experiment_truncated(self):
        text, sources = experiment_with_sources("實驗六")
        self.assertGreaterEqual(sum(s.startswith("exp6_") for s in sources), 4)
        self.assertLess(len(text), 12000)
        self.assertIn("以下截斷", text)

    def test_overview_and_missing(self):
        overview, _ = experiment_with_sources("")
        self.assertIn("實驗八", overview)
        self.assertIn("exp12_confusion_tsne.md", overview)
        missing, _ = experiment_with_sources("99")
        self.assertTrue(missing.startswith("【查無資料】沒有實驗 99"))

    def test_search_normalizes_experiment_numbers(self):
        sections = load_knowledge(KNOWLEDGE_DIR) + load_knowledge(EXPERIMENTS_DIR)
        for query in ("實驗三 輸出怎麼看", "實驗3 輸出怎麼看", "exp3 輸出怎麼看"):
            with self.subTest(query=query):
                _, sources = search_with_sources(sections, query)
                self.assertEqual(sources[0], "exp3_trend.md › 輸出怎麼看")

    def test_no_match_has_no_sources_and_tells_model_to_refuse(self):
        text, sources = search_with_sources(load_knowledge(), "zzz")
        self.assertEqual(sources, [])
        self.assertTrue(text.startswith("【查無資料】"))
        self.assertIn("不要自行推測", text)


class FrontendMarkupTests(unittest.TestCase):
    """index.html 的聊天視窗骨架；行為由 ai_chat_frontend.cjs 以 Node 實際執行驗證。"""

    def setUp(self):
        static = Path(server.STATIC_DIR)
        self.html = (static / "index.html").read_text(encoding="utf-8")
        self.js = (static / "ai_chat.js").read_text(encoding="utf-8")

    def test_widget_outside_live_page_with_all_controls(self):
        live = self.html[self.html.index('<section id="page-live">'):self.html.index("</section>")]
        self.assertNotIn('id="aiChat"', live)  # 浮動視窗不放在即時頁裡，實驗頁也能開
        for element in ('id="aiFab"', 'id="aiChat"', 'id="aiClose"', 'id="aiMax"', 'id="aiLed"',
                        'id="aiStatus"', 'id="aiLog"', 'id="aiForm"', '<textarea id="aiInput"',
                        'class="ai-resize n" data-dir="n"', 'class="ai-resize w" data-dir="w"',
                        'class="ai-resize nw" data-dir="nw"'):
            self.assertIn(element, self.html)

    def test_scripts_loaded_after_widget_markup_in_order(self):
        md = self.html.index('<script src="/static/markdown.js"></script>')
        chat = self.html.index('<script src="/static/ai_chat.js"></script>')
        self.assertLess(self.html.index('id="aiChat"'), md)
        self.assertLess(md, chat)
        self.assertNotIn("async function aiAsk(", self.html)  # 只留一份實作在 ai_chat.js

    def test_placeholder_fits_single_line(self):
        # 原本把操作說明塞進 placeholder，在 400px 視窗折成兩行被裁掉
        self.assertIn('placeholder="輸入問題…"', self.html)
        self.assertIn("Enter 送出，Shift+Enter 換行。", self.html)

    def test_status_checks_do_not_overlap_and_refresh_after_answer(self):
        status = self.js[self.js.index("async function aiStatus()"):self.js.index("async function aiAsk(")]
        self.assertIn("if (aiStatusPending) return;", status)
        self.assertIn("aiStatusPending = false;", status[status.index("finally"):])
        ask = self.js[self.js.index("async function aiAsk("):self.js.index('$("aiForm")')]
        self.assertIn("aiStatus();", ask[ask.rindex("finally"):])


if __name__ == "__main__":
    unittest.main()
