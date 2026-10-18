"""單次問答的有限工具圖；快取與已取得資料不跨請求共用。"""

from __future__ import annotations

import asyncio
import json
import re
import time

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

LIMITS = {"rounds": 3, "tools": 6, "models": 4, "seconds": 90.0, "call_seconds": 25.0, "final_seconds": 25.0}


def fallback_answer(audit: dict, reason: str) -> str:
    lines = ["【查詢未完整完成】" + reason + "；以下只保留已確認資料。"]
    for message in audit.get("evidence", []):
        artifact = message.artifact if isinstance(message.artifact, dict) else {}
        text = artifact.get("facts") or str(message.content)
        if sum(map(len, lines)) + len(text) < 18000:
            lines.append(f"工具 {message.name}：\n{text}")
        else:
            lines.append(f"工具 {message.name} 的完整資料超出呈現預算；請依來源讀取，不截出缺分母的數字。")
    if not audit.get("evidence"):
        lines.append("目前沒有已取得的工具資料。")
    lines.append("尚未完成：整體問題的完整解釋與未取得的證據；不補造答案。可縮小問題後再次提問。")
    return "\n\n".join(lines)


def build_bounded_graph(llm, tools, *, audit: dict, normalize_number, limits: dict | None = None):
    policy = {**LIMITS, **(limits or {})}
    audit.update(evidence=[], rounds=0, tools=0, models=0, chars=0, degraded=False)
    deadline = time.monotonic() + policy["seconds"]
    cache = {}
    node = ToolNode(tools)
    required = llm.bind_tools(tools, tool_choice="required")
    free = llm.bind_tools(tools)
    final = llm.bind_tools(tools, tool_choice="none")

    def key(call):
        args = dict(call["args"])
        definition = next((t for t in tools if t.name == call["name"]), None)
        if definition is not None:
            for field, schema in definition.args.items():
                if "default" in schema:
                    args.setdefault(field, schema["default"])
        for field, value in args.items():
            if isinstance(value, str):
                args[field] = re.sub(r"\s+", " ", value.strip().lower())
                if field == "number":
                    number = normalize_number(value)
                    if number is not None:
                        args[field] = str(number)
        return call["name"], json.dumps(args, ensure_ascii=False, sort_keys=True)

    def degrade(reason):
        audit.update(degraded=True, reason=reason)
        return {"messages": [AIMessage(content=fallback_answer(audit, reason))]}

    async def agent(state, config):
        remaining = deadline - time.monotonic()
        finish = (audit.get("stop") or audit["rounds"] >= policy["rounds"]
                  or audit["tools"] >= policy["tools"] or remaining <= policy["final_seconds"]
                  or audit["models"] >= policy["models"] - 1)
        if audit["models"] >= policy["models"] or remaining <= 0:
            return degrade("查詢已達時間或模型呼叫上限")
        model = final if finish else (required if audit["models"] == 0 else free)
        audit["models"] += 1
        messages = state["messages"]
        if finish:
            messages = [*messages, SystemMessage("工具查詢已停止。僅根據已取得資料整理一次答案，不再要求工具；不足處明說，禁止補造數字或原因。")]
        available = min(remaining, policy["final_seconds"]) if finish else max(0.001, remaining - policy["final_seconds"])
        try:
            response = await asyncio.wait_for(model.ainvoke(messages, config), min(policy["call_seconds"], available))
        except asyncio.TimeoutError:
            return degrade("模型回覆逾時，並非實驗失敗")
        if finish and response.tool_calls:
            return degrade("查詢已達上限，最後整理仍要求工具")
        if len(response.tool_calls) > max(0, policy["tools"] - audit["tools"]):
            return degrade("本輪要求的工具數超出預算，未執行這批查詢")
        if not response.tool_calls and not str(response.content).strip():
            return degrade("模型沒有提供可用答案")
        return {"messages": [response]}

    async def execute(state, config):
        audit["rounds"] += 1
        calls = state["messages"][-1].tool_calls
        messages = []
        for call in calls:
            audit["tools"] += 1
            if audit["tools"] > policy["tools"]:
                audit["stop"] = True
                messages.append(ToolMessage(content="工具呼叫預算已用盡，未執行此查詢。", tool_call_id=call["id"], name=call["name"]))
                continue
            identity = key(call)
            if identity in cache:
                audit["stop"] = True
                messages.append(cache[identity].model_copy(update={"tool_call_id": call["id"]}))
                continue
            remaining = deadline - time.monotonic() - policy["final_seconds"]
            if remaining <= 0:
                audit["stop"] = True
                messages.append(ToolMessage(content="已保留最後整理時間，未執行此查詢。", tool_call_id=call["id"], name=call["name"]))
                continue
            try:
                output = await asyncio.wait_for(node.ainvoke({"messages": [AIMessage(content="", tool_calls=[call])]}, config), min(5, remaining))
                message = output["messages"][0]
            except asyncio.TimeoutError:
                audit["stop"] = True
                message = ToolMessage(content="工具查詢逾時，沒有取得此項證據。", tool_call_id=call["id"], name=call["name"])
            if audit["chars"] + len(str(message.content)) > 16000:
                audit["stop"] = True
                facts = message.artifact.get("facts", "") if isinstance(message.artifact, dict) else ""
                content = facts if facts and audit["chars"] + len(facts) <= 16000 else "工具內容超出本輪文字預算；來源仍保留，不能猜測省略的數字。"
                message = message.model_copy(update={"content": content})
            audit["chars"] += len(str(message.content))
            cache[identity] = message
            audit["evidence"].append(message)
            messages.append(message)
        return {"messages": messages}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", execute)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", lambda state: "tools" if state["messages"][-1].tool_calls else END)
    graph.add_edge("tools", "agent")
    return graph.compile()
