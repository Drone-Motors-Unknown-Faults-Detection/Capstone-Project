// AI 問答浮動聊天視窗（index.html 的 #aiFab／#aiChat）。需先載入 markdown.js（renderMarkdown）。
// 狀態燈定時檢查、SSE 串流回答（工具過程、Markdown、📎 依據）、拖曳改大小、放大鈕。
(function () {
  "use strict";
  const $ = id => document.getElementById(id);

  const aiHistory = [];      // {role, content}，送給伺服器的對話歷史
  let aiContext = null;
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
      if (r.app) $("aiStatus").title += `\n程式版本：${r.app.git_head || "未知"}；未提交變更：${r.app.tracked_dirty ?? "未知"}；摘要：${r.app.answer_schema || "未知"}`;
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
    const progress = document.createElement("div"); progress.textContent = "正在讀取執行紀錄";
    progress.setAttribute("role", "status");
    const steps = document.createElement("details"), stepTitle = document.createElement("summary"), stepList = document.createElement("ul");
    stepTitle.textContent = "工具明細"; steps.append(stepTitle, stepList); tools.append(progress, steps);
    const body = document.createElement("div"); body.textContent = "等待回覆…";
    box.append(tools, body);
    let answer = "", factsBox = null, presentation = null, binding = {};
    let responseMode = "model", responseContext = null, completed = false, failed = false;
    const sources = new Map(), labels = new Set();
    function addSources(values) {
      for (const value of values || []) {
        const key = typeof value === "string" ? value : JSON.stringify(value);
        sources.set(key, value);
      }
    }
    try {
      const resp = await fetch("/api/assistant/ask", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question, history: aiHistory.slice(-6), ...(aiContext ? {context: aiContext} : {})}),
      });
      if (resp.ok === false || !resp.body) throw new Error("伺服器未提供可讀串流");
      const reader = resp.body.getReader(), dec = new TextDecoder("utf-8", {fatal: true});
      let buf = "";
      for (;;) {
        const {value, done} = await reader.read();
        if (done) { buf += dec.decode(); break; }
        buf = (buf + dec.decode(value, {stream: true})).replace(/\r\n/g, "\n");
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, i); buf = buf.slice(i + 2);
          if (!line.startsWith("data: ") || completed) continue;
          const f = JSON.parse(line.slice(6));
          if (failed && f.event !== "done") continue;
          if (f.event === "mode" && ["matrix_summary", "record_timing", "verified_fallback"].includes(f.mode) && !answer && !presentation) {
            binding = f; responseMode = f.mode; responseContext = f.context;
            box.setAttribute("data-response-mode", responseMode);
            progress.textContent = responseMode === "matrix_summary" ? "正在核算矩陣" :
              responseMode === "record_timing" ? "正在讀取時間欄位" : "查詢未完整完成";
            addSources(f.source_records || f.sources);
          }
          else if (f.event === "presentation") {
            if (!presentation && window.AssistantPresentation?.valid(f.presentation, binding)) presentation = f.presentation;
          }
          else if (f.event === "tool") {
            progress.textContent = f.label || "正在讀取依據";
            if (typeof f.label === "string" && !labels.has(f.label)) {
              const li = document.createElement("li"); li.textContent = f.label; stepList.appendChild(li); labels.add(f.label);
            }
            addSources(f.source_records || f.sources);
          }
          else if (f.event === "facts") {
            if (["matrix_summary", "record_timing"].includes(responseMode)) continue;
            if (!factsBox) {
              factsBox = document.createElement("div"); factsBox.className = "ai-facts";
              factsBox.style.whiteSpace = "pre-wrap"; box.appendChild(factsBox);
            }
            factsBox.textContent = f.text;
            addSources(f.sources);
          }
          else if (f.event === "delta") {
            if (responseMode === "matrix_summary" && f.origin !== "matrix") continue;
            if (responseMode === "record_timing" && f.origin !== "record") continue;
            if (responseMode === "verified_fallback" && f.origin !== "fallback") continue;
            if (typeof f.text !== "string") throw new Error("回覆文字格式無效");
            answer += f.text;
            if (responseMode === "model") { body.className = "ai-md"; body.innerHTML = renderMarkdown(answer); }
          }
          else if (f.event === "error") {
            failed = true; box.className = "ai-msg error"; body.textContent = f.error || "查詢失敗";
            progress.textContent = "查詢失敗";
          }
          else if (f.event === "done") {
            completed = true;
            if (!failed) {
              progress.textContent = responseMode === "verified_fallback" || ["invalid", "incomplete", "missing", "INCOMPLETE", "INCONSISTENT"].includes(binding.status) ?
                "查詢已結束；請注意資料限制" : "已完成";
              body.className = "ai-md";
              if (!answer) body.textContent = responseMode !== "model" ? "（程式未提供可用摘要）" : "（模型沒有回覆文字）";
              else if (!presentation || !window.AssistantPresentation?.render(body, presentation, binding, answer)) {
                body.innerHTML = renderMarkdown(answer);
                if (window.AssistantPresentation) window.AssistantPresentation.copyControl(body, "完整文字", answer);
              }
            }
            const src = document.createElement("div"); src.className = "ai-sources";
            if (window.AssistantPresentation) window.AssistantPresentation.sources(src, [...sources.values()]);
            else {
              const title = document.createElement("div"); title.textContent = "📎 依據"; src.appendChild(title);
              const ul = document.createElement("ul");
              for (const value of sources.values()) { const li = document.createElement("li"); li.textContent = typeof value === "string" ? value : value.label || "來源未提供"; ul.appendChild(li); }
              if (ul.children.length) src.appendChild(ul); else { const no = document.createElement("p"); no.textContent = "沒有取得依據"; src.appendChild(no); }
            }
            box.appendChild(src);
          }
          $("aiLog").scrollTop = $("aiLog").scrollHeight;
        }
      }
      if (!completed) throw new Error("回覆提前中斷，尚未收到完成事件；請重試。");
      if (!failed) {
        if (answer) {
          aiHistory.push({role: "user", content: question});
          if (responseMode === "model") aiHistory.push({role: "assistant", content: answer});
        }
        aiContext = ["matrix_summary", "record_timing"].includes(responseMode) ? responseContext : null;
      }
    } catch (e) {
      failed = true; box.className = "ai-msg error"; body.textContent = "連線或回覆失敗：" + e;
      progress.textContent = "查詢未完成";
    } finally {
      aiBusy = false; $("aiSend").disabled = false;
      if ($("aiChat").hidden) $("aiUnread").hidden = false;
      aiStatus();
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
