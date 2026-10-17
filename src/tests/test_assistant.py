"""web/assistant.py 與 /api/assistant/* 的契約：快照不帶真值、工具內容、串流 frame。"""
import asyncio
import json
import unittest
from pathlib import Path

import tornado.testing
import tornado.web
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

import web.server as server
from web.assistant import (
    Assistant,
    ThinkFilter,
    describe_playback,
    describe_playback_events,
    experiment_number,
    experiment_with_sources,
    load_knowledge,
    playback_label,
    search_knowledge,
    search_with_sources,
    to_messages,
    EXPERIMENTS_DIR,
    KNOWLEDGE_DIR,
)
from web.playback import PlaybackTracker


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
    AIMessage(content="", tool_calls=[{"name": "get_playback_status", "args": {}, "id": "c1"}]),
    AIMessage(content="<think>先看狀態</think>目前 健康"),
)


def exp2_tracker(ticks=3, candidate=True):
    """模擬實驗二邊跑邊畫播放到一半的狀態（frame 格式同 iter_run）。"""
    tracker = PlaybackTracker()
    tracker.start("exp2", {"dataset": "T1/8000rpm", "openset_method": "mahalanobis", "seed": 42})
    tracker.update({"event": "fitted", "known": ["8screws"]})
    for t in range(1, ticks + 1):
        tracker.update({"event": "tick", "config": "5screws", "display": "5 螺絲（中度鬆動）", "t": t,
                        "score": 2.5, "label": None, "verdict": "unknown", "quarantine": t, "known": 1})
    if candidate:
        tracker.update({"event": "candidate", "config": "5screws", "candidate": {
            "size": 30, "majority": "5screws", "majority_display": "5 螺絲（中度鬆動）", "purity": 0.9,
            "t_range": [1, 30], "is_known": False, "cluster_params": [25, 3]}})
    return tracker


class PlaybackDescribeTests(unittest.TestCase):
    def test_nothing_played_tells_user_where_to_start(self):
        self.assertIn("【查無資料】", describe_playback(None))
        self.assertIn("⏵ 邊跑邊畫", describe_playback(None))
        self.assertEqual(playback_label(None), "實驗頁播放：無紀錄")

    def test_status_shows_system_verdict_without_truth(self):
        snap = exp2_tracker().snapshot()
        text = describe_playback(snap)
        self.assertIn("實驗：實驗二 量尺擴張（exp2）", text)
        self.assertIn("狀態：播放中", text)
        self.assertIn("已播放：3 筆", text)
        self.assertIn("判定=未知", text)
        self.assertIn("隔離區筆數=3", text)
        self.assertNotIn("5screws", text)
        self.assertNotIn("螺絲", text)

    def test_events_separate_reference_truth(self):
        text = describe_playback_events(exp2_tracker().snapshot())
        events, _, reference = text.partition("【事後對照")
        self.assertIn("隔離區聚出候選群：30 筆", events)
        self.assertNotIn("螺絲", events)
        self.assertIn("不是系統當下的判定", reference)
        self.assertIn("5 螺絲（中度鬆動）", reference)

    def test_label_summarizes_progress(self):
        label = playback_label(exp2_tracker().snapshot())
        self.assertRegex(label, r"^實驗頁播放：實驗二 量尺擴張，播放中，已播 3 筆，更新於 \d\d:\d\d:\d\d$")


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


class AskTests(unittest.TestCase):
    def test_agent_calls_tool_and_streams_answer(self):
        snap = exp2_tracker(candidate=False).snapshot()
        model = fake_model(*TOOL_THEN_ANSWER)
        assistant = Assistant("http://unused/v1", llm=model)

        async def collect():
            return [f async for f in assistant.ask(snap, "目前狀況？")]

        frames = asyncio.run(collect())
        self.assertEqual(frames[0], {"event": "tool", "name": "get_playback_status", "label": "讀取實驗播放狀態",
                                     "sources": [playback_label(snap)]})
        # 第一步強制查工具，查過之後才放開讓模型作答
        self.assertEqual(model.choices, ["required", "auto"])
        deltas = [f["text"] for f in frames if f["event"] == "delta"]
        self.assertEqual("".join(deltas).replace(" ", ""), "目前健康")
        # 逐 token 串流：Python 3.10 下 agent 節點必須明傳 config，否則只會收到整則訊息
        self.assertGreater(len(deltas), 1)
        tool_msgs = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
        self.assertIn("已播放：3 筆", tool_msgs[0].content)


class AssistantHTTPTests(tornado.testing.AsyncHTTPTestCase):
    def setUp(self):
        self.saved = (server.PLAYBACK, server.ASSISTANT)
        server.PLAYBACK = PlaybackTracker()   # 尚未播放任何實驗
        super().setUp()

    def tearDown(self):
        super().tearDown()
        server.PLAYBACK, server.ASSISTANT = self.saved

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
        self.assertEqual(frames[-1], {"event": "done", "as_of": "實驗頁播放：無紀錄"})
        self.assertIn("delta", {f["event"] for f in frames})
        self.assertEqual(frames[0]["sources"], ["實驗頁播放：無紀錄"])

    def test_ask_reads_current_playback(self):
        server.PLAYBACK = exp2_tracker()
        server.ASSISTANT = Assistant("http://unused/v1", llm=fake_model(*TOOL_THEN_ANSWER))
        frames = self.frames(json.dumps({"question": "目前狀況？"}))
        self.assertRegex(frames[-1]["as_of"], r"^實驗頁播放：實驗二 量尺擴張，播放中，已播 3 筆")

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

    def test_page_title_and_icon(self):
        self.assertIn("<title>畢業專題 PHM 展示</title>", self.html)
        self.assertIn("<h1>畢業專題 PHM 展示</h1>", self.html)
        self.assertNotIn("冷啟動 PHM 展示", self.html)
        self.assertIn('<link rel="icon" type="image/png" href="/static/icon.png">', self.html)
        icon = (Path(server.STATIC_DIR) / "icon.png").read_bytes()
        self.assertEqual(icon[:8], b"\x89PNG\r\n\x1a\n")

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
