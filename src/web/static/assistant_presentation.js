// 可信程式區段的固定DOM模板；模型文字仍只走安全Markdown。
(function (root) {
  "use strict";
  const scalar = v => typeof v === "string" || (typeof v === "number" && Number.isFinite(v));
  const text = v => scalar(v) ? String(v) : "未提供";
  function node(tag, value, cls) {
    const e = document.createElement(tag);
    if (value !== undefined) e.textContent = text(value);
    if (cls) e.className = cls;
    return e;
  }
  function valid(p, binding) {
    const origin = {matrix_summary: "matrix", record_timing: "record"}[binding.mode];
    if (!origin || !p || p.schema_version !== "assistant_presentation_v1" || p.mode !== binding.mode || p.origin !== origin ||
        !/^[a-f0-9]{64}$/.test(binding.source_id || "") || p.source_id !== binding.source_id ||
        !p.source || JSON.stringify(p.source) !== JSON.stringify(binding.presentation_source) ||
        (binding.context?.saved && p.source.saved !== binding.context.saved) ||
        (binding.context?.experiment && p.source.exp_id !== binding.context.experiment) ||
        !["complete", "incomplete", "invalid"].includes(p.status) || !scalar(p.title) || !scalar(p.subtitle) ||
        !Array.isArray(p.sections) || p.sections.length > 40) return false;
    const ids = new Set();
    return p.sections.every(s => {
      if (!s || typeof s.id !== "string" || ids.has(s.id) || !scalar(s.title) || typeof s.detail !== "boolean" || typeof s.open !== "boolean") return false;
      ids.add(s.id);
      const rows = (width) => Array.isArray(s.rows) && s.rows.length <= 500 && s.rows.every(r => Array.isArray(r) && r.length === width && r.every(scalar));
      if (s.kind === "fields") return rows(2);
      if (s.kind === "table") return Array.isArray(s.headers) && s.headers.length > 0 && s.headers.length <= 30 && s.headers.every(scalar) && rows(s.headers.length);
      if (s.kind === "list" || s.kind === "warning") return !(s.kind === "warning" && s.detail) && Array.isArray(s.items) && s.items.every(scalar);
      return s.kind === "code" && typeof s.text === "string" && (!s.copies || (Array.isArray(s.copies) && s.copies.every(r => Array.isArray(r) && r.length === 2 && r.every(scalar))));
    });
  }
  function copyControl(parent, label, value) {
    const button = node("button", "複製" + label, "ai-copy"); button.type = "button";
    const status = node("span", "", "ai-copy-status"); status.setAttribute("role", "status");
    let manual = null;
    button.addEventListener("click", async () => {
      try {
        if (!root.navigator?.clipboard?.writeText) throw new Error("沒有Clipboard API");
        await root.navigator.clipboard.writeText(String(value));
        status.textContent = "已複製";
      } catch (_) {
        if (!manual) {
          manual = node("textarea", undefined, "ai-copy-manual"); manual.readOnly = true;
          manual.value = String(value); manual.setAttribute("aria-label", label + "：選取後手動複製"); parent.appendChild(manual);
        }
        status.textContent = "請在下方選取文字，以Ctrl+C或系統選單複製。";
        manual.focus(); manual.select();
      }
    });
    parent.append(button, status);
  }
  function fields(parent, rows) {
    const dl = node("dl", undefined, "ai-fields");
    for (const [key, value] of rows) dl.append(node("dt", key), node("dd", value));
    parent.appendChild(dl);
  }
  function render(parent, p, binding, plain) {
    if (!valid(p, binding)) return false;
    const wrapper = node("div", undefined, "ai-presentation");
    wrapper.append(node("h3", p.title), node("p", p.subtitle, "ai-subtitle"));
    if (p.status !== "complete") wrapper.appendChild(node("p", p.status === "invalid" ? "資料無效：不提供推測數字。" : "資料不完整：以下僅列現有部分。", "ai-warning"));
    for (const s of p.sections) {
      const part = node(s.detail ? "details" : "section", undefined, s.kind === "warning" ? "ai-warning" : "ai-section");
      part.setAttribute("data-section", s.id);
      if (s.detail) { part.open = s.open; part.appendChild(node("summary", s.title)); }
      else part.appendChild(node("h4", s.title));
      if (s.kind === "fields") fields(part, s.rows);
      else if (s.kind === "list" || s.kind === "warning") {
        const ul = node("ul"); s.items.forEach(v => ul.appendChild(node("li", v))); part.appendChild(ul);
      } else if (s.kind === "table") {
        const scroll = node("div", undefined, "ai-table-scroll"); scroll.tabIndex = 0;
        scroll.setAttribute("role", "region"); scroll.setAttribute("aria-label", s.title + "，可左右捲動");
        const table = node("table"), head = node("thead"), tr = node("tr"), body = node("tbody");
        s.headers.forEach(v => { const th = node("th", v); th.setAttribute("scope", "col"); tr.appendChild(th); });
        head.appendChild(tr);
        s.rows.forEach(r => { const row = node("tr"); r.forEach(v => row.appendChild(node("td", v))); body.appendChild(row); });
        table.append(head, body); scroll.appendChild(table); part.appendChild(scroll);
      } else if (s.kind === "code") {
        const pre = node("pre"); pre.appendChild(node("code", s.text)); part.appendChild(pre);
        (s.copies || []).forEach(([label, value]) => copyControl(part, label, value));
      }
      wrapper.appendChild(part);
    }
    copyControl(wrapper, "完整文字", plain);
    parent.textContent = ""; parent.appendChild(wrapper);
    return true;
  }
  function sources(parent, values) {
    const title = node("div", "📎 依據"); parent.appendChild(title);
    const seen = new Set(), ul = node("ul");
    for (const value of values) {
      const v = typeof value === "string" ? {id: value, label: value, type: "工具來源"} : value;
      if (!v || !scalar(v.label)) continue;
      const id = JSON.stringify([v.id || "", v.path || "", v.sha256 || "", v.url || "", v.id ? "" : v.label]);
      if (seen.has(id)) continue;
      seen.add(id);
      const li = node("li"), label = text(v.label);
      if (typeof v.url === "string" && /^https?:\/\/[^\s<>"']+$/.test(v.url)) {
        const a = node("a", label); a.href = v.url; a.target = "_blank"; a.rel = "noopener noreferrer"; li.appendChild(a);
      } else li.appendChild(node("span", label));
      if (v.type) li.appendChild(node("small", "（" + text(v.type) + "）"));
      if (v.path || v.sha256) {
        const detail = node("details"); detail.appendChild(node("summary", "來源詳情"));
        fields(detail, [["路徑", text(v.path)], ["SHA256", text(v.sha256)]]);
        if (v.path) copyControl(detail, "來源路徑", v.path);
        li.appendChild(detail);
      }
      ul.appendChild(li);
    }
    if (seen.size) parent.appendChild(ul);
    else parent.appendChild(node("p", "沒有取得依據"));
  }
  function displayTime(value) {
    if (!value) return "原檔未記錄";
    if (typeof value !== "string") return "格式無效，未推算";
    if (!/(?:Z|[+-]\d{2}:\d{2})$/.test(value)) return value + "（原檔未提供時區）";
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return value + "（格式無效，未推算）";
    const parts = Object.fromEntries(new Intl.DateTimeFormat("sv-SE", {timeZone: "Asia/Taipei", year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3, hourCycle: "h23"}).formatToParts(date).map(v => [v.type, v.value]));
    return parts.year + "-" + parts.month + "-" + parts.day + " " + parts.hour + ":" + parts.minute + ":" + parts.second + "." + parts.fractionalSecond + "（Asia/Taipei，UTC+08:00）";
  }
  function runInfo(parent, payload, expId) {
    const section = node("section", undefined, "ai-section"); section.appendChild(node("h4", "執行來源與時間"));
    fields(section, [["實驗", expId], ["開始時間", displayTime(payload.started_at)], ["完成時間", displayTime(payload.finished_at)],
      ["耗時", payload.seconds == null ? "原檔未記錄" : text(payload.seconds) + " 秒"],
      ["時間檢查", {VERIFIED: "起訖與量測耗時一致（僅時間欄位）", INCONSISTENT: "時間紀錄不一致", INCOMPLETE: "時間資訊不足"}[payload.timing_status] || "未提供"]]);
    fields(section, Object.entries(payload.params || {}).map(([k, v]) => [k === "confidence" ? "校準信心水準（不是準確率）" : k, scalar(v) ? v : JSON.stringify(v)]));
    const details = node("details"); details.appendChild(node("summary", "完整來源與技術詳情"));
    fields(details, [["存檔位置", payload.saved || "未提供"]]);
    if (payload.saved) copyControl(details, "存檔位置", payload.saved);
    const safe = {};
    for (const key of ["started_at", "finished_at", "seconds", "timing_version", "timing_scope", "timing_status", "timing_warnings", "timing_text", "saved", "saved_sha256", "params"])
      if (payload[key] !== undefined) safe[key] = payload[key];
    const pre = node("pre"); pre.appendChild(node("code", JSON.stringify(safe, null, 2))); details.appendChild(pre);
    section.appendChild(details); parent.appendChild(section);
    const warnings = Array.isArray(payload.timing_warnings) ? payload.timing_warnings : [];
    for (const warning of warnings) parent.appendChild(node("p", warning, "ai-warning"));
    if (!payload.started_at) parent.appendChild(node("p", "開始時間：原檔未記錄", "ai-warning"));
    if ([payload.started_at, payload.finished_at].some(v => typeof v === "string" && !/(?:Z|[+-]\d{2}:\d{2})$/.test(v)))
      parent.appendChild(node("p", "原檔未提供時區；未擅自轉為台灣時間。", "ai-warning"));
  }
  root.AssistantPresentation = {valid, render, sources, copyControl, runInfo, displayTime};
})(typeof window !== "undefined" ? window : globalThis);
