"""web/assistant.py 與 /api/assistant/* 的規範：快照不帶真值、工具內容、串流 frame。"""
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
    NO_PLAYBACK,
    describe_record,
    describe_record_events,
    experiment_number,
    experiment_with_sources,
    load_knowledge,
    latest_record,
    pick_record,
    record_label,
    search_knowledge,
    search_with_sources,
    to_messages,
    EXPERIMENTS_DIR,
    KNOWLEDGE_DIR,
)
from web.playback import PlaybackTracker
from tests.exp12_counts_fixture import historical_result


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
    AIMessage(content="", tool_calls=[{"name": "get_run_status", "args": {}, "id": "c1"}]),
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


def exp12_saved(exp_id):
    """假的存檔讀取：只有實驗十二有伺服器重啟前的結果。"""
    if exp_id not in ("exp12", None):
        return None
    tracker = PlaybackTracker()
    tracker.record_run("exp12", {"params": {"seed": 42}, "seconds": 56.71, "finished_at": "2026-10-17T14:17:08",
                                 "saved": "output/web_server/2026-10-17-14-08-37/experiments/exp12_2026-10-17-14-17-08.json",
                                 "result": {"binary_confusion": {"labels": ["healthy", "fault"],
                                                                 "matrix": [[527, 0], [0, 2736]]}}})
    rec = tracker.snapshot()["records"]["exp12"]
    rec.update(mode="saved", updated_at="2026-10-17T14:17:08")
    return rec


class RecordDescribeTests(unittest.TestCase):
    def test_nothing_recorded_tells_user_where_to_start(self):
        rec, missing = pick_record(None, "", load_saved=lambda _id: None)
        self.assertIsNone(rec)
        self.assertEqual(missing, NO_PLAYBACK)
        self.assertIn("▶ 執行", NO_PLAYBACK)
        self.assertEqual(record_label(None), "實驗頁紀錄：無")

    def test_stream_status_shows_system_verdict_without_truth(self):
        rec = latest_record(exp2_tracker().snapshot())
        text = describe_record(rec)
        self.assertIn("實驗：實驗二 量尺擴張（exp2）", text)
        self.assertIn("來源：⏵ 邊跑邊畫；狀態：播放中", text)
        self.assertIn("已播放：3 筆", text)
        self.assertIn("判定=未知", text)
        self.assertNotIn("5screws", text)
        self.assertNotIn("螺絲", text)

    def test_events_separate_reference_truth(self):
        text = describe_record_events(latest_record(exp2_tracker().snapshot()))
        events, _, reference = text.partition("【事後對照")
        self.assertIn("隔離區聚出候選群：30 筆", events)
        self.assertNotIn("螺絲", events)
        self.assertIn("不是系統當下的判定", reference)
        self.assertIn("5 螺絲（中度鬆動）", reference)

    def test_label_summarizes_progress(self):
        label = record_label(latest_record(exp2_tracker().snapshot()))
        self.assertRegex(label, r"^實驗頁紀錄：實驗二 量尺擴張（⏵ 邊跑邊畫），播放中，已播 3 筆，時間 \d\d:\d\d:\d\d$")

    def test_pick_by_number_prefers_memory_then_saved_file(self):
        snap = exp2_tracker().snapshot()
        for number in ("2", "二", "實驗二", "exp2"):
            with self.subTest(number=number):
                self.assertEqual(pick_record(snap, number, exp12_saved)[0]["exp_id"], "exp2")
        # 實驗十二用「▶ 執行」跑的，記憶體沒有（伺服器重啟），改讀存檔
        for number in ("12", "十二", "實驗12", "exp12_2026-10-17-14-17-08.json"):
            with self.subTest(number=number):
                rec, _ = pick_record(snap, number, exp12_saved)
                self.assertEqual((rec["exp_id"], rec["mode"]), ("exp12", "saved"))
        text = describe_record(pick_record(snap, "12", exp12_saved)[0])
        self.assertIn("來源：存檔；狀態：已完成", text)
        self.assertIn("先前執行留下的存檔", text)
        self.assertIn("| healthy | 527 | 0 |", text)
        self.assertIn("屬事後對照", text)

    def test_missing_experiment_lists_what_exists(self):
        rec, missing = pick_record(exp2_tracker().snapshot(), "7", exp12_saved)
        self.assertIsNone(rec)
        self.assertTrue(missing.startswith("【查無資料】實驗 7 在伺服器上沒有執行或播放紀錄"))
        self.assertIn("目前有紀錄的實驗：實驗二 量尺擴張", missing)

    def test_multi_card_number_takes_latest(self):
        tracker = PlaybackTracker()
        tracker.record_run("exp8", {"params": {}, "result": {}})
        tracker.start("exp8_monitor", {"dataset": "T1/8000rpm", "config": "5screws"})
        self.assertEqual(pick_record(tracker.snapshot(), "八")[0]["exp_id"], "exp8_monitor")

    def test_no_number_falls_back_to_newest_saved(self):
        rec, _ = pick_record(None, "", exp12_saved)
        self.assertEqual(rec["exp_id"], "exp12")


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
        self.assertEqual(frames[0], {"event": "tool", "name": "get_run_status", "label": "讀取實驗執行紀錄",
                                     "sources": [record_label(latest_record(snap))]})
        # 第一步強制查工具，查過之後才放開讓模型作答
        self.assertEqual(model.choices, ["required", "auto"])
        deltas = [f["text"] for f in frames if f["event"] == "delta"]
        self.assertEqual("".join(deltas).replace(" ", ""), "目前健康")
        # 逐 token 串流：Python 3.10 下 agent 節點必須明傳 config，否則只會收到整則訊息
        self.assertGreater(len(deltas), 1)
        tool_msgs = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
        self.assertIn("已播放：3 筆", tool_msgs[0].content)

    def test_exp12_primary_answer_overrides_wrong_history_without_model_generation(self):
        for name in ("get_run_status", "get_run_events"):
            with self.subTest(tool=name):
                tracker = PlaybackTracker()
                tracker.record_run("exp12", {"params": {"seed": 42}, "saved": "指定來源/exp12.json",
                                             "result": historical_result()})
                tracker.record_run("exp1", {"params": {}, "result": {"irrelevant": 123}})
                model = fake_model(
                    AIMessage(content="", tool_calls=[{"name": name, "args": {"number": "12"}, "id": "c1"}]),
                    AIMessage(content="模型仍可能答錯，這段不作為數值權威"))
                assistant = Assistant("http://unused/v1", llm=model)

                async def collect():
                    return [f async for f in assistant.ask(tracker.snapshot(), "更正實驗十二", history=[
                        {"role": "assistant", "content": "健康527筆，誤報率8.5%，2938筆都誤拒；legacy沒有判對"}])]

                frames = asyncio.run(collect())
                text = "".join(f["text"] for f in frames if f["event"] == "delta")
                self.assertEqual(frames[0]["mode"], "matrix_summary")
                self.assertIn("45／572＝7.87%", text)
                self.assertIn("202／2938＝6.88%", text)
                self.assertIn("正確拒絕 14381", text)
                self.assertIn("指定來源/exp12.json", frames[0]["sources"][0])
                self.assertNotIn("irrelevant", text)
                self.assertNotIn("模型仍可能答錯", text)
                self.assertFalse(any(f["event"] == "facts" for f in frames))
                self.assertEqual(model.seen, [])

    def test_exp12_without_matrix_emits_inadequacy_not_invented_facts(self):
        tracker = PlaybackTracker()
        tracker.record_run("exp12", {"result": {"seed": 42}})
        text = describe_record_events(tracker.snapshot()["records"]["exp12"])
        self.assertIn("矩陣核算失敗", text)
        self.assertNotIn("7.87%", text)

    def test_saved_record_does_not_change_between_tools_in_one_question(self):
        reads = []

        def changing_saved(exp_id):
            reads.append(exp_id)
            result = historical_result()
            result["binary_confusion"]["matrix"][0][1] += len(reads) - 1
            tracker = PlaybackTracker()
            tracker.record_run("exp12", {"saved": f"來源{len(reads)}.json", "result": result})
            return tracker.snapshot()["records"]["exp12"]

        model = fake_model(
            AIMessage(content="", tool_calls=[{"name": "get_run_status", "args": {"number": "12"}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[{"name": "get_run_events", "args": {"number": "12"}, "id": "c2"}]),
            AIMessage(content="核對同一份存檔"))
        assistant = Assistant("http://unused/v1", llm=model)

        async def collect():
            return [f async for f in assistant.ask(None, "實驗十二的原理", load_saved=changing_saved)]

        frames = asyncio.run(collect())
        facts = [f for f in frames if f["event"] == "facts"]
        self.assertEqual(reads, ["exp12"])
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["text"], facts[1]["text"])
        self.assertIn("來源1.json", facts[1]["text"])


class AssistantHTTPTests(tornado.testing.AsyncHTTPTestCase):
    def setUp(self):
        self.saved = (server.PLAYBACK, server.ASSISTANT)
        server.PLAYBACK = PlaybackTracker(results_root=Path("不存在的目錄"))   # 尚未執行，也沒有存檔
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
        self.assertEqual(frames[-1], {"event": "done", "as_of": "實驗頁紀錄：無"})
        self.assertIn("delta", {f["event"] for f in frames})
        self.assertEqual(frames[0]["sources"], ["實驗頁紀錄：無"])

    def test_ask_reads_current_playback(self):
        server.PLAYBACK = exp2_tracker()
        server.ASSISTANT = Assistant("http://unused/v1", llm=fake_model(*TOOL_THEN_ANSWER))
        frames = self.frames(json.dumps({"question": "目前狀況？"}))
        self.assertRegex(frames[-1]["as_of"], r"^實驗頁紀錄：實驗二 量尺擴張（⏵ 邊跑邊畫），播放中，已播 3 筆")

    def test_http_exp12_primary_answer_is_program_generated_with_consistent_source(self):
        server.PLAYBACK.record_run("exp12", {"saved": "指定來源/exp12.json", "result": historical_result()})
        server.ASSISTANT = Assistant("http://unused/v1", llm=fake_model(
            AIMessage(content="", tool_calls=[{"name": "get_run_status", "args": {"number": "12"}, "id": "c1"}]),
            AIMessage(content="模型的說明文字")))
        frames = self.frames(json.dumps({"question": "實驗十二總結", "history": []}))
        self.assertEqual(frames[0]["mode"], "matrix_summary")
        self.assertIn("指定來源/exp12.json", frames[0]["sources"][0])
        text = "".join(f["text"] for f in frames if f["event"] == "delta")
        self.assertIn("45／572＝7.87%", text)
        self.assertNotIn("模型的說明文字", text)
        self.assertFalse(any(f["event"] == "facts" for f in frames))
        self.assertTrue(all(f.get("origin") == "matrix" for f in frames if f["event"] == "delta"))
        self.assertEqual(frames[-1]["event"], "done")
        self.assertEqual(frames[-1]["as_of"], frames[0]["sources"][0])

    def test_status_exposes_only_minimal_app_diagnostics(self):
        server.ASSISTANT = Assistant("http://unused/v1", llm=fake_model())
        status = json.loads(self.fetch("/api/assistant/status").body)
        self.assertEqual(set(status["app"]), {"git_head", "tracked_dirty", "answer_schema"})
        self.assertEqual(status["app"]["answer_schema"], "exp12_standard_answer_v1")
        self.assertNotIn("packages", status["app"])

    def test_http_result_works_with_real_offline_model_endpoint(self):
        port = tornado.testing.bind_unused_port()
        port[0].close()
        server.ASSISTANT = Assistant(f"http://127.0.0.1:{port[1]}/v1")
        server.PLAYBACK.record_run("exp12", {"result": historical_result()})
        frames = self.frames(json.dumps({"question": "健康是不是只有527筆，誤報率8.5%？",
                                        "context": {"experiment": "exp12", "mode": "result"}}))
        self.assertFalse(any(f["event"] == "error" for f in frames))
        self.assertIn("45／572＝7.87%", "".join(f.get("text", "") for f in frames))
        self.assertIsNone(server.ASSISTANT._llm)

    def test_http_explicit_saved_context_survives_followup_without_switching_source(self):
        root = Path("saved")
        folder = root / "session/experiments"
        folder.mkdir(parents=True)
        names = ["exp12_2026-01-01-00-00-00.json", "exp12_2026-01-01-00-00-01.json"]
        for i, name in enumerate(names):
            result = historical_result()
            result["binary_confusion"]["matrix"][0][1] += i
            (folder / name).write_text(json.dumps({"experiment": "exp12", "result": result}), encoding="utf-8")
        server.PLAYBACK = PlaybackTracker(root)
        server.PLAYBACK.record_run("exp1", {"result": {"irrelevant": 999}})
        server.ASSISTANT = Assistant("http://unused/v1", llm=fake_model())
        frames = self.frames(json.dumps({"question": f"總結 `{names[0]}`"}))
        text = "".join(f.get("text", "") for f in frames)
        self.assertIn("45／572＝7.87%", text)
        self.assertNotIn(names[1], text)
        self.assertEqual(frames[-1]["as_of"], frames[0]["sources"][0])
        followup = self.frames(json.dumps({"question": "那是不是全部誤拒", "context": frames[0]["context"]}))
        self.assertIn("45／572＝7.87%", "".join(f.get("text", "") for f in followup))
        self.assertFalse(any(f["event"] == "error" for f in followup))

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
        # placeholder 只放短提示，操作說明另外顯示；放進 placeholder 會在 400px 視窗折成兩行被裁掉
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
