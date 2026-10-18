"""AI 問答：LangGraph 工具呼叫 agent，經 OpenAI 相容 API 連本地模型（vLLM／llama.cpp）。

server.py 在 IOLoop 執行緒取 ``PlaybackTracker.snapshot()``（web/playback.py），凍結提問當下
實驗頁各實驗最近一次的執行／播放紀錄，再交給 ``Assistant.ask()``；agent 只能透過四個工具讀資料：

    get_run_status       依編號取實驗頁紀錄（▶ 執行／⏵ 邊跑邊畫／存檔）：參數、進度、系統判定、結果摘要
    get_run_events       重點事件（候選群、自學、變化點警報、嚴重度改變）與事後對照（真值評估）
    get_experiment       依編號取實驗：web/experiments.py 的 CATALOG + docs/experiments/expN_*.md 手冊
    lookup_knowledge     關鍵字查 src/web/knowledge/*.md 與 docs/experiments/*.md 的「## 」段落

知識量小且有編號結構，用編號直取 + 關鍵字檢索，不另建向量資料庫：
「實驗一／實驗1／exp1」一律正規化成 exp1 再比對。

本模組不做任何判定：數字全部來自各實驗 iter_run() 的既有輸出，模型只負責解讀。
注入配置、expected、is_fault、候選群 majority／purity 等真值只放在「事後對照」，
並標明是評估用、不是系統當下的判定（AGENT.md 鐵則 6）。
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

import tornado.httpclient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from experiments.exp3_trend import KIND_DISPLAY
from web.experiments import CATALOG

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "experiments"
MAX_DOC_CHARS = 9000     # get_experiment 一次回傳的手冊字數上限（多份手冊平分）
MAX_HISTORY = 6          # 帶入的歷史訊息則數（user + assistant 合計）
MAX_TEXT = 2000          # 單則訊息字數上限
RECURSION_LIMIT = 10     # agent ↔ tools 來回上限

TOOL_DISPLAY = {
    "get_run_status": "讀取實驗執行紀錄",
    "get_run_events": "讀取事件與事後對照",
    "get_experiment": "讀取實驗手冊",
    "lookup_knowledge": "查詢知識庫",
}

SYSTEM_PROMPT = """你是無人機馬達冷啟動 PHM 監測系統的值班助理，用繁體中文回答操作員。

規則：
1. 問到目前狀況、執行或播放的結果、數據代表什麼、分數、判定、警報、隔離區、候選群、健康指數，
   先呼叫 get_run_status；使用者提到實驗編號（「實驗12」「exp12_….json」「我的實驗二結果」）就把編號填進
   number，沒提就留空取最近一次。需要事件經過或事後對照再呼叫 get_run_events。
   資料來自實驗頁「▶ 執行」「⏵ 邊跑邊畫」或其存檔，回答時說明是哪個實驗、哪種來源、是否已完成。
2. 問到某個實驗（例如「實驗一」「實驗1」「exp3」）在做什麼、怎麼跑、結果怎麼看，呼叫
   get_experiment 並帶入編號；問「有哪些實驗」時 number 留空取總覽。
3. 問到方法、參數、名詞意義或馬達規格，呼叫 lookup_knowledge。知識庫寫「待補」的欄位，
   回答「規格尚未提供」，不要推測。
4. 只能根據本輪工具回傳的內容回答，不要用一般常識、馬達或機器學習的通用知識補充或推測。
   工具結果沒有寫到的事，直接回答「文件與系統狀態裡沒有這項資訊」，並說明查了哪些來源。
5. 引用文件時標出檔名（例如 exp1_cold_start.md）；引用播放數字時標出實驗與樣本序號 t 或窗序號。
6. 系統當下的判定只有「健康／已知／未知」「漸進／突發」與健康指數；未知故障的身分就是未知，
   不要猜是哪種螺絲鬆動。「事後對照」裡的配置名稱是評估用真值，引用時必須明說是事後對照，
   不能說成系統判定。
7. 單筆偶發未知可能是健康誤報；判斷是否異常要一起看該次 EWMA 與隔離區。
   各次實驗的誤報率只能引用該次核算，不能挪用其他實驗的常數。
8. 實驗十二的數值引用本輪工具的「程式核算」，保留來源、任務、分子／分母；對話歷史可能有錯，
   不沿用舊答自行重算。核算失敗或摘要不完整時，說明不足，不能猜數字。
   known 支持總數不等於誤拒數；unknown→unknown 是正確拒絕，但不識別各未知配置。
   legacy 即使把已知全拒，unknown 對角仍可判對；-1 介面限制須有來源版本證據，不能從圖樣猜原因，
   不能判定原方法完全失效。此批不能證明跨馬達或部署可靠，也不能由誤拒率確認邊界過嚴，
   不建議用同一 test 調門檻後宣稱改善。手冊說明方法，模型特徵來自 CSV，不來自手冊。
9. 回答先講結論，再列支撐的內容。"""

NO_MATCH = "【查無資料】{what}。請直接告訴使用者：文件與系統狀態裡沒有這項資訊；不要自行推測或用一般知識補充。"


# -- 實驗頁播放狀態 -------------------------------------------------------------

NO_PLAYBACK = NO_MATCH.format(what="伺服器上沒有任何實驗的執行或播放紀錄，也找不到存檔") + \
    "可以請使用者到實驗頁按「▶ 執行」或「⏵ 邊跑邊畫」。"
STATUS_DISPLAY = {"running": "播放中", "done": "已完成", "stopped": "被停止（未完成）", "error": "失敗"}
MODE_DISPLAY = {"stream": "⏵ 邊跑邊畫", "run": "▶ 執行", "saved": "存檔"}
VERDICT_DISPLAY = {"healthy": "健康", "known": "已知類別", "unknown": "未知"}
FIELD_DISPLAY = {
    "t": "樣本序號 t", "score": "開集分數（> 1 為未知）", "verdict": "判定", "predicted": "模型判成的類別",
    "quarantine": "隔離區筆數", "n_known": "已知類別數", "scenario": "劇本", "trial": "第幾次重複（0 起算）",
    "ewma": "EWMA 異常比例（≥ 0.5 警報）", "cusum": "CUSUM", "alarm_kind": "警報判別",
    "segment": "第幾段", "segment_kind": "段落類型", "last_scores": "最近分數", "flagged_in_chunk": "本批判未知筆數",
    "chunk_size": "本批筆數", "window_index": "窗序號", "health_index": "相對健康指數（1 健康、0 最差）",
    "severity_stage": "嚴重度", "trend": "趨勢", "degradation_rate": "退化速率", "openset_score": "開集分數",
    "is_unknown_fault": "判為未知故障", "fault_type": "故障類型", "prediction_confidence": "信心",
    "uncertainty": "不確定度", "alarm_state": "告警狀態", "alarm_reason": "告警原因",
    "change_point_state": "變化點狀態", "data_quality": "資料品質", "part": "子實驗", "state": "狀態",
}


def record_name(rec: dict) -> str:
    sub = f" {rec['sub']}" if rec.get("sub") else ""
    return f"{rec['no']}{sub} {rec['title']}".strip()


def record_label(rec: dict | None) -> str:
    """一行摘要：依據清單與回答下方「狀態取自」共用。"""
    if rec is None:
        return "實驗頁紀錄：無"
    progress = f"，已播 {rec['samples']} 筆" if rec.get("mode") == "stream" else ""
    label = (f"實驗頁紀錄：{record_name(rec)}（{MODE_DISPLAY.get(rec.get('mode'), rec.get('mode'))}），"
             f"{STATUS_DISPLAY.get(rec['status'], rec['status'])}{progress}，時間 {rec['updated_at']}")
    saved = (rec.get("result") or {}).get("saved")
    return label + (f"，存檔 {saved}" if saved else "")


def latest_record(snap: dict | None) -> dict | None:
    if not snap or not snap.get("latest"):
        return None
    return snap["records"].get(snap["latest"])


def pick_record(snap: dict | None, number: str = "", load_saved=None) -> tuple[dict | None, str]:
    """依實驗編號取紀錄：記憶體優先，沒有再讀存檔；編號留空取最近一次。回傳（紀錄, 查無時的說明）。"""
    records = (snap or {}).get("records", {})
    if not (number or "").strip():
        rec = latest_record(snap) or (load_saved(None) if load_saved else None)
        return rec, NO_PLAYBACK
    n = experiment_number(number)
    if n is None:
        return None, NO_MATCH.format(what=f"看不懂實驗編號「{number}」")
    ids = [e["id"] for e in CATALOG if e["id"] == f"exp{n}" or e["id"].startswith(f"exp{n}_")]
    in_memory = [records[i] for i in ids if i in records]
    if in_memory:
        return max(in_memory, key=lambda r: r.get("seq", 0)), ""
    saved = [r for r in (load_saved(i) for i in ids) if r] if load_saved else []
    if saved:
        return max(saved, key=lambda r: r.get("updated_at") or ""), ""
    have = "、".join(record_name(r) for r in records.values()) or "無"
    return None, (NO_MATCH.format(what=f"實驗 {n} 在伺服器上沒有執行或播放紀錄，也找不到存檔")
                  + f"目前有紀錄的實驗：{have}。")


def _value(key: str, value) -> str:
    if key == "verdict":
        return VERDICT_DISPLAY.get(value, str(value))
    if key == "alarm_kind" and value:
        return KIND_DISPLAY.get(value, value)
    if isinstance(value, bool):
        return "是" if value else "否"
    return "—" if value is None else str(value)


def describe_record(rec: dict) -> str:
    params = "、".join(f"{k}={v}" for k, v in rec.get("params", {}).items()) or "—"
    lines = [
        f"實驗：{record_name(rec)}（{rec['exp_id']}）",
        f"來源：{MODE_DISPLAY.get(rec.get('mode'), rec.get('mode'))}；狀態：{STATUS_DISPLAY.get(rec['status'], rec['status'])}；"
        f"開始 {rec['started_at']}，最後更新 {rec['updated_at']}",
        f"參數：{params}",
    ]
    if rec.get("mode") == "stream":
        lines.append(f"已播放：{rec['samples']} 筆")
    if rec.get("mode") == "saved":
        lines.append("這份是先前執行留下的存檔（例如伺服器重啟前），不是本次啟動後的執行")
    if rec.get("known"):
        lines.append("模型的已知類別：" + "、".join(rec["known"]) + "（discovered-N 是模型自學的新類名稱）")
    current = rec.get("current")
    if current:
        lines.append("最新一筆的系統判定：" + "；".join(
            f"{FIELD_DISPLAY.get(k, k)}={_value(k, v)}" for k, v in current.items()))
    if rec.get("result"):
        lines.append("完成資訊：" + "；".join(f"{k}={v}" for k, v in rec["result"].items()))
    if rec.get("error"):
        lines.append(f"錯誤：{rec['error']}")
    text = "\n".join(f"- {x}" for x in lines)
    if rec.get("result_text"):
        text += ("\n\n【實驗結果（評估輸出；依注入配置比對的欄位，如混淆矩陣的真值列、偵測率、純度，屬事後對照）】\n"
                 + rec["result_text"])
    return text


def describe_record_events(rec: dict) -> str:
    events = rec.get("events") or []
    out = [f"【{record_name(rec)} 的重點事件】"]
    out += [f"- {e}" for e in events] or ["- 尚無重點事件"]
    reference = rec.get("reference") or []
    truth = []
    if rec.get("source_config"):
        truth.append(f"- 資料來源（操作者選的注入配置）：{rec['source_config']}")
    if "source_is_fault" in rec:
        truth.append(f"- 這段資料是否來自故障配置：{'是' if rec['source_is_fault'] else '否'}")
    truth += ["- " + "；".join(f"{k}={v}" for k, v in item.items()) for item in reference]
    if truth:
        out.append("【事後對照（由注入配置真值計算，只用於評估；不是系統當下的判定）】")
        out += truth
    if rec.get("fact_text"):
        out += ["【已完成結果的程式核算（事後對照，非即時系統判定）】", rec["fact_text"]]
    return "\n".join(out)


def _record_artifact(rec: dict, *, events: bool = False):
    sources = [record_label(rec) + (" › 事件與事後對照" if events else "")]
    if rec.get("fact_text"):
        return {"sources": sources, "facts": rec["fact_text"]}
    return sources


# -- 知識庫 ---------------------------------------------------------------------

def load_knowledge(directory: Path = KNOWLEDGE_DIR) -> list[dict]:
    """依「## 」標題把 *.md 切成段落。"""
    sections = []
    for path in sorted(directory.glob("*.md")):
        for block in re.split(r"(?m)^## ", path.read_text(encoding="utf-8"))[1:]:
            title, _, body = block.partition("\n")
            sections.append({"source": path.name, "title": title.strip(), "body": body.strip()})
    return sections


CN_DIGITS = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_number(text: str) -> int | None:
    """中文數字（一～九十九）或阿拉伯數字轉 int。"""
    if text.isdigit():
        return int(text)
    if not text or any(c not in CN_DIGITS and c != "十" for c in text):
        return None
    if "十" not in text:
        return CN_DIGITS[text] if len(text) == 1 else None
    tens, _, ones = text.partition("十")
    return (CN_DIGITS.get(tens, 1) if tens else 1) * 10 + (CN_DIGITS.get(ones, 0) if ones else 0)


def experiment_number(text: str) -> int | None:
    """「實驗一」「實驗 12」「exp3」「3」「十一」→ 實驗編號。"""
    text = text.strip().lower()
    m = re.search(r"(?:實驗|exp)\s*([0-9]+|[零一二兩三四五六七八九十]+)", text)
    return _cn_number(m.group(1) if m else text)


def _normalize(text: str) -> str:
    """把「實驗一」「實驗 1」改寫成 exp1，讓三種寫法比對到同一詞。"""
    def sub(m):
        n = _cn_number(m.group(1))
        return f" exp{n} " if n is not None else m.group(0)
    return re.sub(r"實驗\s*([0-9]+|[零一二兩三四五六七八九十]+)", sub, text)


def _terms(text: str) -> set[str]:
    """英數詞 + 中文二字詞，供簡單關鍵字比對。"""
    text = _normalize(text.lower())
    terms = set(re.findall(r"[a-z0-9_.]+", text))
    for run in re.findall(r"[一-鿿]+", text):
        terms.update(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return terms


def search_knowledge(sections: list[dict], query: str, limit: int = 3) -> str:
    return search_with_sources(sections, query, limit)[0]


def search_with_sources(sections: list[dict], query: str, limit: int = 3) -> tuple[str, list[str]]:
    """回傳（給模型的段落文字, 依據清單「檔名 › 段落標題」）；查無時依據為空。"""
    q = _terms(query)
    scored = []
    for sec in sections:
        # 手冊檔名帶 expN_，算進標題詞，讓「實驗三 輸出」對到 exp3_trend.md 的段落
        title_terms = _terms(sec["title"]) | _terms(sec["source"].split("_")[0])
        score = 3 * len(q & title_terms) + len(q & _terms(sec["body"]))
        if score:
            scored.append((score, sec))
    if not scored:
        titles = "、".join(dict.fromkeys(s["title"] for s in sections))
        return NO_MATCH.format(what=f"知識庫與實驗手冊沒有符合「{query}」的段落") + f"\n可查的主題：{titles}", []
    scored.sort(key=lambda x: -x[0])
    top = [sec for _, sec in scored[:limit]]
    text = "\n\n".join(f"### {s['title']}（{s['source']}）\n{s['body']}" for s in top)
    return text, [f"{s['source']} › {s['title']}" for s in top]


def _doc_title(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0] if path.stat().st_size else ""
    return first.lstrip("# ").strip() or path.stem


def experiment_overview(experiments_dir: Path = EXPERIMENTS_DIR) -> str:
    lines = ["網頁實驗頁（web/experiments.py CATALOG）："]
    for e in CATALOG:
        sub = f" {e['sub']}" if e.get("sub") else ""
        lines.append(f"- {e['no']}{sub}（{e['id']}）{e['title']}：{e['question']}")
    docs = sorted(experiments_dir.glob("exp*_*.md"), key=lambda p: (experiment_number(p.stem) or 0, p.name))
    if docs:
        lines.append("技術手冊（docs/experiments/）：")
        lines += [f"- {p.name}：{_doc_title(p)}" for p in docs]
    return "\n".join(lines)


def describe_experiment(number: str, experiments_dir: Path = EXPERIMENTS_DIR) -> str:
    return experiment_with_sources(number, experiments_dir)[0]


def experiment_with_sources(number: str, experiments_dir: Path = EXPERIMENTS_DIR) -> tuple[str, list[str]]:
    """實驗 N 的目錄條目（程式）+ 手冊全文（docs，依上限截斷），與依據清單。"""
    n = experiment_number(number or "")
    if n is None:
        return experiment_overview(experiments_dir), ["web/experiments.py › 實驗目錄", "docs/experiments/"]
    entries = [e for e in CATALOG if e["id"] == f"exp{n}" or e["id"].startswith(f"exp{n}_")]
    docs = sorted(experiments_dir.glob(f"exp{n}_*.md"))
    if not entries and not docs:
        return (NO_MATCH.format(what=f"沒有實驗 {n}") + "\n\n" + experiment_overview(experiments_dir),
                ["web/experiments.py › 實驗目錄"])
    parts = []
    for e in entries:
        sub = f" {e['sub']}" if e.get("sub") else ""
        parts.append(
            f"【{e['no']}{sub}：{e['title']}】\n研究問題：{e['question']}\n"
            f"擬合內容：{e.get('fits', '—')}\n程式：{e['code']}；手冊：{e['doc']}"
        )
    budget = MAX_DOC_CHARS // max(1, len(docs))
    for p in docs:
        text = p.read_text(encoding="utf-8")
        if len(text) > budget:
            text = text[:budget] + f"\n…（以下截斷，全文見 docs/experiments/{p.name}）"
        parts.append(f"【手冊 {p.name}】\n{text}")
    sources = [f"web/experiments.py › {e['no']}{(' ' + e['sub']) if e.get('sub') else ''}" for e in entries]
    return "\n\n".join(parts), sources + [p.name for p in docs]


# -- LangGraph agent -------------------------------------------------------------

def make_tools(snap: dict | None, sections: list[dict], experiments_dir: Path = EXPERIMENTS_DIR,
               load_saved=None) -> list:
    saved_cache = {}

    def frozen_saved(exp_id):
        if exp_id not in saved_cache:
            rec = load_saved(exp_id)
            if rec is not None:
                rec = saved_cache.setdefault(rec["exp_id"], rec)
            saved_cache[exp_id] = rec
        return saved_cache[exp_id]

    saved_reader = frozen_saved if load_saved else None
    # artifact = 依據清單，由程式決定、不經模型轉述，前端顯示成「📎 依據」
    @tool(response_format="content_and_artifact")
    def get_run_status(number: str = "") -> tuple[str, list[str] | dict]:
        """讀取實驗頁的執行紀錄與結果（「▶ 執行」或「⏵ 邊跑邊畫」）：哪個實驗、參數、狀態、播放進度、最新一筆的系統判定、完整結果摘要（混淆矩陣、偵測率等）。number 填實驗編號，例如「12」「十二」「實驗二」；使用者沒指定實驗時留空，取最近一次。"""
        rec, missing = pick_record(snap, number, saved_reader)
        if rec is None:
            return missing, [record_label(None)]
        return describe_record(rec), _record_artifact(rec)

    @tool(response_format="content_and_artifact")
    def get_run_events(number: str = "") -> tuple[str, list[str] | dict]:
        """讀取某次執行或播放的重點事件（候選群、模型自學、變化點警報、嚴重度改變）與事後對照（依注入配置真值算的評估結果）。number 規則同 get_run_status。"""
        rec, missing = pick_record(snap, number, saved_reader)
        if rec is None:
            return missing, [record_label(None)]
        return describe_record_events(rec), _record_artifact(rec, events=True)

    @tool(response_format="content_and_artifact")
    def get_experiment(number: str = "") -> tuple[str, list[str]]:
        """讀取某個實驗的目的、研究問題、方法、怎麼跑與輸出怎麼看。number 填編號，例如「1」「一」「實驗三」「exp8」；留空回傳所有實驗總覽。"""
        return experiment_with_sources(number, experiments_dir)

    @tool(response_format="content_and_artifact")
    def lookup_knowledge(query: str) -> tuple[str, list[str]]:
        """關鍵字查詢知識庫與各實驗手冊：偵測方法、參數、閾值、螺絲配置意義、處置原則、馬達規格。query 用關鍵字，例如「EWMA 警報」「T1 馬達規格」「實驗三 輸出」。"""
        return search_with_sources(sections, query)

    return [get_run_status, get_run_events, get_experiment, lookup_knowledge]


def _looked_up(messages: list) -> bool:
    """本輪提問（最後一則 HumanMessage）之後是否已經呼叫過工具。"""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return any(isinstance(m, ToolMessage) for m in messages[last_human + 1:])


def build_graph(llm, tools):
    # 每輪第一步 tool_choice="required"：模型一定先查工具，不能直接憑印象回答
    must_look = llm.bind_tools(tools, tool_choice="required")
    free = llm.bind_tools(tools)

    async def agent(state: MessagesState, config: RunnableConfig) -> dict:
        model = free if _looked_up(state["messages"]) else must_look
        # Python 3.10 的 asyncio 不會自動傳遞 config；不明傳，stream_mode="messages" 收不到逐 token
        return {"messages": [await model.ainvoke(state["messages"], config)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def to_messages(history: list, question: str) -> list:
    msgs = [SystemMessage(SYSTEM_PROMPT)]
    for item in (history or [])[-MAX_HISTORY:]:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            continue
        text = item["content"][:MAX_TEXT]
        if item.get("role") == "user":
            msgs.append(HumanMessage(text))
        elif item.get("role") == "assistant":
            msgs.append(AIMessage(text))
    msgs.append(HumanMessage(question[:MAX_TEXT]))
    return msgs


class ThinkFilter:
    """串流中剔除 <think>…</think>（Qwen3 等推理模型的思考段）。"""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while self.buf:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if not self.inside:
                    out.append(self.buf[:i])
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            # 保留可能是標籤開頭的尾巴，等下一段
            keep = next((k for k in range(min(len(tag) - 1, len(self.buf)), 0, -1)
                         if tag.startswith(self.buf[-k:])), 0)
            emit, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
            if not self.inside:
                out.append(emit)
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


def _chunk_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


class Assistant:
    """持有本地模型連線設定；每次提問以當下快照建一張 graph。"""

    def __init__(self, base_url: str, model: str = "", api_key: str = "EMPTY",
                 temperature: float = 0.0, llm=None, knowledge_dir: Path = KNOWLEDGE_DIR,
                 experiments_dir: Path = EXPERIMENTS_DIR):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._llm = llm
        self.knowledge_dir = knowledge_dir
        self.experiments_dir = experiments_dir

    async def list_models(self) -> list[str]:
        client = tornado.httpclient.AsyncHTTPClient()
        resp = await client.fetch(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            request_timeout=5,
        )
        return [m["id"] for m in json.loads(resp.body).get("data", [])]

    async def resolve_model(self) -> str:
        if not self.model:
            models = await self.list_models()
            if not models:
                raise RuntimeError(f"{self.base_url}/models 沒有回報任何模型")
            self.model = models[0]
        return self.model

    async def status(self) -> dict:
        info = {"url": self.base_url, "model": self.model or None}
        if self._llm is not None:
            return {**info, "reachable": True}
        try:
            models = await self.list_models()
            return {**info, "reachable": True, "model": self.model or (models[0] if models else None)}
        except Exception as exc:
            return {**info, "reachable": False, "error": f"{exc!r}"}

    async def llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI  # 延後載入：未用到 AI 問答不付匯入成本

            self._llm = ChatOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                model=await self.resolve_model(),
                temperature=self.temperature,
                streaming=True,
            )
        return self._llm

    async def ask(self, snap: dict | None, question: str, history: list | None = None,
                  load_saved=None) -> AsyncIterator[dict]:
        """回傳工具來源、非生成 facts 與模型 delta；facts 不由模型改寫。"""
        sections = load_knowledge(self.knowledge_dir) + load_knowledge(self.experiments_dir)
        graph = build_graph(await self.llm(), make_tools(snap, sections, self.experiments_dir, load_saved))
        think = ThinkFilter()
        async for chunk, meta in graph.astream(
            {"messages": to_messages(history or [], question)},
            {"recursion_limit": RECURSION_LIMIT},
            stream_mode="messages",
        ):
            node = meta.get("langgraph_node")
            if node == "tools" and isinstance(chunk, ToolMessage):
                artifact = chunk.artifact or []
                sources = artifact.get("sources", []) if isinstance(artifact, dict) else artifact
                yield {"event": "tool", "name": chunk.name,
                       "label": TOOL_DISPLAY.get(chunk.name, chunk.name),
                       "sources": list(sources)}
                if isinstance(artifact, dict) and artifact.get("facts"):
                    yield {"event": "facts", "text": artifact["facts"], "sources": list(sources)}
            elif node == "agent" and isinstance(chunk, (AIMessageChunk, AIMessage)):
                text = think.feed(_chunk_text(chunk.content))
                if text:
                    yield {"event": "delta", "text": text}
        tail = think.flush()
        if tail:
            yield {"event": "delta", "text": tail}
