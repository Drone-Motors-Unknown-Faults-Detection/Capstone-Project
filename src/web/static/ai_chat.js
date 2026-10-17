// AI 問答浮動聊天視窗（index.html 的 #aiFab／#aiChat）。需先載入 markdown.js（renderMarkdown）。
// 狀態燈定時檢查、SSE 串流回答（工具過程、Markdown、📎 依據）、拖曳改大小、放大鈕。
(function () {
  "use strict";
  const $ = id => document.getElementById(id);

  const aiHistory = [];      // {role, content}，送給伺服器的對話歷史
  let aiBusy = false;

  function aiBubble(cls, text){
    // 一列 = （AI 頭像）+ 泡泡；使用者靠右、AI 靠左
    const row = document.createElement("div");
    row.className = "ai-row " + cls;
    if (cls !== "user") {
      const avatar = document.createElement("div");
      avatar.className = "ai-avatar"; avatar.textContent = "AI";
      row.appendChild(avatar);
    }
    const div = document.createElement("div");
    div.className = "ai-msg " + cls;
    div.textContent = text;
    row.appendChild(div);
    $("aiLog").appendChild(row);
    $("aiLog").scrollTop = $("aiLog").scrollHeight;
    return div;
  }

  function aiOpen(open){
    $("aiChat").hidden = !open;
    $("aiFab").setAttribute("aria-label", open ? "關閉 AI 問答" : "開啟 AI 問答");
    if (open) { $("aiUnread").hidden = true; $("aiInput").focus(); $("aiLog").scrollTop = $("aiLog").scrollHeight; }
    try { localStorage.setItem("aiChatOpen", open ? "1" : "0"); } catch (e) {}
  }

  const AI_STATUS_MS = 15000;   // 模型狀態定時重新檢查（vLLM 啟動要數分鐘，好了不必重新整理頁面）
  let aiStatusPending = false;

  async function aiStatus(){
    if (aiStatusPending) return;  // 上一次還沒回來就不疊加
    aiStatusPending = true;
    const checked = new Date().toLocaleTimeString();
    try {
      const r = await (await fetch("/api/assistant/status")).json();
      $("aiStatus").textContent = r.reachable ? `● ${r.model || "已連線"}` : "○ 模型離線（啟動中或未開）";
      $("aiStatus").style.color = r.reachable ? "var(--green)" : "#fca5a5";
      $("aiLed").className = "ai-led " + (r.reachable ? "on" : "off");
      $("aiStatus").title = `${r.reachable ? r.url : (r.error || r.url)}\n上次檢查 ${checked}，每 ${AI_STATUS_MS / 1000} 秒自動更新`;
    } catch (e) {
      $("aiStatus").textContent = "○ 狀態未知";
      $("aiStatus").style.color = "var(--muted)";
      $("aiLed").className = "ai-led";
      $("aiStatus").title = `展示伺服器連不上：${e}\n上次檢查 ${checked}`;
    } finally {
      aiStatusPending = false;
    }
  }

  async function aiAsk(question){
    if (aiBusy || !question.trim()) return;
    aiBusy = true; $("aiSend").disabled = true;
    aiBubble("user", question);
    const box = aiBubble("assistant", "");
    const tools = document.createElement("div"); tools.className = "ai-tools";
    const body = document.createElement("div");
    body.innerHTML = '<span class="ai-typing"><span></span><span></span><span></span></span>';  // 等第一個字
    box.append(tools, body);
    let answer = "";
    const sources = new Set();   // 工具回傳的依據（程式決定，不經模型轉述）
    try {
      const resp = await fetch("/api/assistant/ask", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question, history: aiHistory.slice(-6)}),
      });
      const reader = resp.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const {value, done} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream: true});
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!line.startsWith("data: ")) continue;
          const f = JSON.parse(line.slice(6));
          if (f.event === "tool") {
            tools.textContent += (tools.textContent ? " → " : "🔧 ") + f.label;
            (f.sources || []).forEach(x => sources.add(x));
          }
          else if (f.event === "delta") { answer += f.text; body.className = "ai-md"; body.innerHTML = renderMarkdown(answer); }
          else if (f.event === "done") tools.textContent += (tools.textContent ? "　" : "") + `（狀態取自 t=${f.t}）`;
          else if (f.event === "error") { box.className = "ai-msg error"; body.textContent = f.error; }
          else if (f.event === "done" && !answer) body.textContent = "（模型沒有回覆文字）";
          if (f.event === "done") {
            const src = document.createElement("div"); src.className = "ai-sources";
            src.textContent = sources.size ? "📎 依據：" + [...sources].join("；") : "📎 依據：無（工具沒有找到相關資料）";
            box.appendChild(src);
          }
          $("aiLog").scrollTop = $("aiLog").scrollHeight;
        }
      }
      if (answer) aiHistory.push({role: "user", content: question}, {role: "assistant", content: answer});
    } catch (e) {
      box.className = "ai-msg error"; body.textContent = "連線失敗：" + e;
    } finally {
      aiBusy = false; $("aiSend").disabled = false;
      if ($("aiChat").hidden) $("aiUnread").hidden = false;   // 關著視窗時回答完成 → 黃點提示
      aiStatus();  // 回答成功或失敗後同步狀態燈
    }
  }

  $("aiForm").addEventListener("submit", e => {
    e.preventDefault();
    const q = $("aiInput").value; $("aiInput").value = ""; $("aiInput").style.height = "";
    aiAsk(q);
  });
  $("aiInput").addEventListener("keydown", e => {
    // Enter 送出、Shift+Enter 換行；中文輸入法選字中（isComposing）的 Enter 不送出
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      $("aiForm").requestSubmit();
    }
  });
  $("aiInput").addEventListener("input", () => {
    const box = $("aiInput");
    box.style.height = "";
    box.style.height = Math.min(box.scrollHeight, 120) + "px";
    box.style.overflowY = box.scrollHeight > 120 ? "auto" : "hidden";   // 未滿最大高度不出捲軸
  });
  $("aiFab").addEventListener("click", () => aiOpen($("aiChat").hidden));
  $("aiClose").addEventListener("click", () => aiOpen(false));

  // -- 視窗大小：左／上／左上把手拖曳，放大鈕切換；記在 localStorage --------------------
  const AI_MIN_W = 320, AI_MIN_H = 360;
  let aiRestoreSize = null;

  function aiApplySize(w, h){
    const maxW = window.innerWidth - 40, maxH = window.innerHeight - 108;
    w = Math.round(Math.max(AI_MIN_W, Math.min(maxW, w)));
    h = Math.round(Math.max(AI_MIN_H, Math.min(maxH, h)));
    $("aiChat").style.width = w + "px";
    $("aiChat").style.height = h + "px";
    return [w, h];
  }
  function aiSaveSize(){
    try {
      const w = parseInt($("aiChat").style.width), h = parseInt($("aiChat").style.height);
      if (w && h) localStorage.setItem("aiChatSize", JSON.stringify([w, h]));
      else localStorage.removeItem("aiChatSize");
    } catch (e) {}
  }
  function aiResetSize(){
    $("aiChat").style.width = ""; $("aiChat").style.height = "";
    aiRestoreSize = null; aiSaveSize();
  }
  document.querySelectorAll("#aiChat .ai-resize").forEach(handle => {
    handle.addEventListener("pointerdown", e => {
      if (window.innerWidth <= 600) return;
      e.preventDefault();
      const dir = handle.dataset.dir, rect = $("aiChat").getBoundingClientRect(), x0 = e.clientX, y0 = e.clientY;
      handle.setPointerCapture(e.pointerId);
      document.body.style.userSelect = "none";
      const move = ev => aiApplySize(
        dir.includes("w") ? rect.width + (x0 - ev.clientX) : rect.width,
        dir.includes("n") ? rect.height + (y0 - ev.clientY) : rect.height);
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        handle.removeEventListener("pointercancel", up);
        document.body.style.userSelect = "";
        aiRestoreSize = null; aiSaveSize();
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
      handle.addEventListener("pointercancel", up);
    });
  });
  document.querySelector("#aiChat .ai-resize.nw").addEventListener("dblclick", aiResetSize);
  $("aiMax").addEventListener("click", () => {
    if (aiRestoreSize) {   // 已放大 → 還原
      const [w, h] = aiRestoreSize; aiRestoreSize = null;
      if (w && h) aiApplySize(w, h); else { $("aiChat").style.width = ""; $("aiChat").style.height = ""; }
    } else {
      aiRestoreSize = [parseInt($("aiChat").style.width) || 0, parseInt($("aiChat").style.height) || 0];
      aiApplySize(window.innerWidth, window.innerHeight);
    }
    $("aiMax").textContent = aiRestoreSize ? "⤡" : "⤢";
    aiSaveSize();
  });
  window.addEventListener("resize", () => {
    if ($("aiChat").style.width) aiApplySize(parseInt($("aiChat").style.width), parseInt($("aiChat").style.height));
  });
  try {
    const saved = JSON.parse(localStorage.getItem("aiChatSize") || "null");
    if (Array.isArray(saved) && saved.length === 2) aiApplySize(saved[0], saved[1]);
  } catch (e) {}
  try { if (localStorage.getItem("aiChatOpen") === "1") aiOpen(true); } catch (e) {}
  document.querySelectorAll("#aiSuggest button").forEach(b => b.addEventListener("click", () => aiAsk(b.dataset.q)));
  aiStatus();
  setInterval(() => { if (!document.hidden) aiStatus(); }, AI_STATUS_MS);   // 背景分頁暫停
  document.addEventListener("visibilitychange", () => { if (!document.hidden) aiStatus(); });
})();
