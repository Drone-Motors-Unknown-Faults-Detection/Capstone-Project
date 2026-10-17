"""AI 問答：LangGraph 工具呼叫 agent，經 OpenAI 相容 API 連本地模型（vLLM／llama.cpp）。

server.py 在 IOLoop 執行緒呼叫 ``machine_snapshot(hub)`` 凍結提問當下的機台狀態，
再交給 ``Assistant.ask()``；agent 只能透過三個工具讀資料：

    get_live_status      即時狀態：資料集、階段、最近一筆分數／判定、EWMA、隔離區、候選叢集
    get_recent_alarms    最近的變化點警報（漸進／突發）
    lookup_knowledge     查 src/web/knowledge/*.md（偵測方法與馬達規格）

本模組不做任何判定：數字全部來自 LiveDemo／core 的既有輸出，模型只負責解讀。
注入來源、劇本、逐筆 truth、候選叢集的 majority／purity／is_known 都由真值算出，
屬於「操作員確認」才揭示的資訊（AGENT.md 鐵則 6），不放進快照。
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

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
MAX_HISTORY = 6          # 帶入的歷史訊息則數（user + assistant 合計）
MAX_TEXT = 2000          # 單則訊息字數上限
RECURSION_LIMIT = 10     # agent ↔ tools 來回上限

TOOL_DISPLAY = {
    "get_live_status": "讀取即時狀態",
    "get_recent_alarms": "讀取變化點警報",
    "lookup_knowledge": "查詢知識庫",
}

SYSTEM_PROMPT = """你是無人機馬達冷啟動 PHM 監測系統的值班助理，用繁體中文回答操作員。

規則：
1. 問到機台現況（分數、判定、警報、隔離區、候選故障、已知類別）一律先呼叫 get_live_status，
   需要警報歷史再呼叫 get_recent_alarms；不要憑印象回答數字。
2. 問到方法、參數、名詞意義或馬達規格，呼叫 lookup_knowledge。知識庫寫「待補」的欄位，
   回答「規格尚未提供」，不要推測。
3. 系統只知道「已知／未知」與「漸進／突發」；未經操作員確認的未知故障，身分就是未知，
   不要猜是哪種螺絲鬆動。
4. 單筆偶發未知可能是健康誤報（約 10.9%）；判斷是否異常要一起看 EWMA 與隔離區。
5. 回答先講結論，再列支撐的數字（附上樣本序號 t）。工具資料沒有的事就說不知道。"""


# -- 機台狀態快照 ---------------------------------------------------------------

def machine_snapshot(hub) -> dict:
    """在 IOLoop 執行緒呼叫：凍結提問當下的狀態，移除所有真值欄位。"""
    state = hub.full_state()
    demo = hub.demo
    openset = state.get("openset") or {}
    candidate = state.get("candidate")
    last = getattr(demo, "last_sample", None)
    return {
        "dataset": state.get("meta"),
        "openset_method": openset.get("openset_method"),
        "class_thresholds": [
            {"config": c.get("config"), "threshold": round(float(c["threshold"]), 3)}
            for c in openset.get("classes", []) if "threshold" in c
        ],
        "t": state.get("t"),
        "streaming": bool(state.get("running")),
        "busy": bool(state.get("busy")),
        "error": state.get("error"),
        "phase": state.get("phase"),
        "known_classes": [k["display"] for k in state.get("known", [])],
        "quarantine": state.get("quarantine"),
        "min_cluster_size": state.get("min_cluster_size"),
        "candidate": None if candidate is None else {
            k: candidate[k] for k in ("size", "t_range", "cluster_params") if k in candidate
        },
        "last_sample": None if last is None else {
            "t": last["t"],
            "score": last["score"],
            "verdict": last["verdict"],
            "predicted": last["cls"],
            "ewma": last["ewma"],
            "cusum": last["cusum"],
            "nearest_fault_ray": last["direction"]["ray_display"],
            "ray_cos": last["direction"]["cos"],
            "radius": last["direction"]["radius"],
            "cluster_attempts": last["attempts"],
        },
        "alarms": [
            {"t": a["t"], "kind": a["kind_display"], "transition": a["transition"]}
            for a in getattr(demo, "alarms", [])[-5:]
        ],
    }


PHASE_DISPLAY = {
    "monitoring": "監測中（隔離區空）",
    "accumulating": "異常樣本累積中（隔離區有樣本，等待成叢）",
    "pending": "候選新故障待操作員確認",
}
VERDICT_DISPLAY = {"healthy": "健康", "known": "已知故障", "unknown": "未知"}


def describe_status(snap: dict) -> str:
    """把快照整理成給模型讀的條列文字。"""
    ds = snap.get("dataset") or {}
    lines = [
        f"資料集：{ds.get('motor', '?')} / {ds.get('rpm', '?')}；開集方法：{snap.get('openset_method')}",
        f"串流：{'進行中' if snap['streaming'] else '暫停'}；已處理 {snap['t']} 筆"
        + ("；重任務執行中" if snap["busy"] else ""),
        f"階段：{PHASE_DISPLAY.get(snap['phase'], snap['phase'])}",
        f"已知類別（健康量尺）：{'、'.join(snap['known_classes']) or '無'}",
        f"隔離區：{snap['quarantine']} 筆（候選叢集需 ≥ {snap['min_cluster_size']} 筆）",
    ]
    if snap.get("error"):
        lines.append(f"錯誤：{snap['error']}")
    cand = snap.get("candidate")
    if cand:
        rng = cand.get("t_range") or ["?", "?"]
        lines.append(
            f"候選新故障：{cand['size']} 筆，樣本蒐集區間 t={rng[0]}~{rng[1]}，"
            f"HDBSCAN 參數 {cand.get('cluster_params')}；身分待操作員確認"
        )
    last = snap.get("last_sample")
    if last is None:
        lines.append("最近一筆樣本：尚無（串流未開始）")
    else:
        ray = last["nearest_fault_ray"]
        lines += [
            f"最近一筆樣本 t={last['t']}：開集分數 {last['score']}（> 1 為未知），"
            f"判定 {VERDICT_DISPLAY.get(last['verdict'], last['verdict'])}"
            + (f"（{last['predicted']}）" if last["verdict"] != "unknown" else ""),
            f"EWMA 異常比例 {last['ewma']}（≥ 0.5 警報，[0.2, 0.5) 中間帶）；CUSUM {last['cusum']}",
            "方向：" + (f"最接近已知故障射線「{ray}」，cos={last['ray_cos']}" if ray else "尚無已知故障射線")
            + f"；半徑 {last['radius']}",
            f"HDBSCAN 已嘗試分群 {last['cluster_attempts']} 次",
        ]
    thr = "、".join(f"{c['config']}={c['threshold']}" for c in snap.get("class_thresholds", []))
    if thr:
        lines.append(f"各類校準閾值（原始距離）：{thr}")
    return "\n".join(f"- {x}" for x in lines)


def describe_alarms(snap: dict) -> str:
    alarms = snap.get("alarms") or []
    if not alarms:
        return "本 session 尚未發生變化點警報。"
    return "\n".join(
        f"- t={a['t']}：{a['kind']}（中間帶停留 {a['transition']} 筆）" for a in alarms
    )


# -- 知識庫 ---------------------------------------------------------------------

def load_knowledge(directory: Path = KNOWLEDGE_DIR) -> list[dict]:
    """依「## 」標題把 *.md 切成段落。"""
    sections = []
    for path in sorted(directory.glob("*.md")):
        for block in re.split(r"(?m)^## ", path.read_text(encoding="utf-8"))[1:]:
            title, _, body = block.partition("\n")
            sections.append({"source": path.name, "title": title.strip(), "body": body.strip()})
    return sections


def _terms(text: str) -> set[str]:
    """英數詞 + 中文二字詞，供簡單關鍵字比對。"""
    text = text.lower()
    terms = set(re.findall(r"[a-z0-9_.]+", text))
    for run in re.findall(r"[一-鿿]+", text):
        terms.update(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return terms


def search_knowledge(sections: list[dict], query: str, limit: int = 3) -> str:
    q = _terms(query)
    scored = []
    for sec in sections:
        title_terms = _terms(sec["title"])
        score = 3 * len(q & title_terms) + len(q & _terms(sec["body"]))
        if score:
            scored.append((score, sec))
    if not scored:
        titles = "、".join(s["title"] for s in sections)
        return f"知識庫沒有符合「{query}」的段落。可查的主題：{titles}"
    scored.sort(key=lambda x: -x[0])
    return "\n\n".join(
        f"### {s['title']}（{s['source']}）\n{s['body']}" for _, s in scored[:limit]
    )


# -- LangGraph agent -------------------------------------------------------------

def make_tools(snap: dict, sections: list[dict]) -> list:
    @tool
    def get_live_status() -> str:
        """讀取提問當下的機台即時狀態：資料集、監測階段、最近一筆開集分數與判定、EWMA、隔離區、候選新故障、已知類別。"""
        return describe_status(snap)

    @tool
    def get_recent_alarms() -> str:
        """讀取最近 5 次變化點警報（漸進退化或突發事件）與發生的樣本序號。"""
        return describe_alarms(snap)

    @tool
    def lookup_knowledge(query: str) -> str:
        """查詢知識庫：偵測方法、參數、閾值、螺絲配置意義、處置原則、馬達規格。query 用關鍵字，例如「EWMA 警報」「T1 馬達規格」。"""
        return search_knowledge(sections, query)

    return [get_live_status, get_recent_alarms, lookup_knowledge]


def build_graph(llm, tools):
    bound = llm.bind_tools(tools)

    async def agent(state: MessagesState, config: RunnableConfig) -> dict:
        # Python 3.10 的 asyncio 不會自動傳遞 config；不明傳，stream_mode="messages" 收不到逐 token
        return {"messages": [await bound.ainvoke(state["messages"], config)]}

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
                 temperature: float = 0.2, llm=None, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self._llm = llm
        self.knowledge_dir = knowledge_dir

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

    async def ask(self, snap: dict, question: str, history: list | None = None) -> AsyncIterator[dict]:
        """逐 frame 產出：{"event": "tool", "name", "label"}、{"event": "delta", "text"}。"""
        graph = build_graph(await self.llm(), make_tools(snap, load_knowledge(self.knowledge_dir)))
        think = ThinkFilter()
        async for chunk, meta in graph.astream(
            {"messages": to_messages(history or [], question)},
            {"recursion_limit": RECURSION_LIMIT},
            stream_mode="messages",
        ):
            node = meta.get("langgraph_node")
            if node == "tools" and isinstance(chunk, ToolMessage):
                yield {"event": "tool", "name": chunk.name,
                       "label": TOOL_DISPLAY.get(chunk.name, chunk.name)}
            elif node == "agent" and isinstance(chunk, (AIMessageChunk, AIMessage)):
                text = think.feed(_chunk_text(chunk.content))
                if text:
                    yield {"event": "delta", "text": text}
        tail = think.flush()
        if tail:
            yield {"event": "delta", "text": tail}
