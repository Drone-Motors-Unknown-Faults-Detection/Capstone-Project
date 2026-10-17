// AI 問答回答用的精簡 Markdown → HTML。
// 先把整段文字的 HTML 跳脫，再套格式：模型輸出無法注入標籤或腳本；連結只接受 http(s)。
// 支援：標題、段落、粗體、斜體（*）、刪除線、行內程式碼、程式碼區塊、引用、分隔線、
// 有序／無序清單（依縮排巢狀）、表格（含對齊）。底線不當斜體：檔名 exp1_cold_start.md 很常見。
(function (root) {
  "use strict";

  const LIST = /^(\s*)([-*+]|\d+[.)])\s+/;
  const FENCE = /^\s*```/;
  const HR = /^\s*([-*_])(\s*\1){2,}\s*$/;
  const HEADING = /^\s*(#{1,6})\s+(.*?)\s*#*\s*$/;
  const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function inline(raw) {
    const codes = [];
    let s = raw.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
    s = esc(s);
    s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    s = s.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*\s][^*\n]*?)\*(?!\*)/g, "$1<em>$2</em>");
    s = s.replace(/~~([^~\n]+?)~~/g, "<del>$1</del>");
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => "<code>" + esc(codes[+i]) + "</code>");
  }

  function indentOf(line) {
    return line.match(/^\s*/)[0].length;
  }

  function isTableStart(lines, i) {
    return i + 1 < lines.length && lines[i].includes("|") && TABLE_SEP.test(lines[i + 1]);
  }

  function isBlockStart(lines, i) {
    const l = lines[i];
    return FENCE.test(l) || HEADING.test(l) || HR.test(l) || /^\s*>/.test(l) || LIST.test(l) || isTableStart(lines, i);
  }

  function cells(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    return s.split("|").map(c => c.trim());
  }

  function table(lines, i) {
    const head = cells(lines[i]);
    const aligns = cells(lines[i + 1]).map(c =>
      c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : "");
    const td = (tag, text, k) => `<${tag}${aligns[k] ? ` style="text-align:${aligns[k]}"` : ""}>${inline(text)}</${tag}>`;
    let html = '<div class="md-table"><table><thead><tr>' + head.map((h, k) => td("th", h, k)).join("") + "</tr></thead><tbody>";
    i += 2;
    while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
      const row = cells(lines[i]);
      html += "<tr>" + head.map((_, k) => td("td", row[k] || "", k)).join("") + "</tr>";
      i++;
    }
    return [html + "</tbody></table></div>", i];
  }

  function list(lines, i, indent) {
    const first = LIST.exec(lines[i]);
    const ordered = /\d/.test(first[2]);
    const start = ordered ? parseInt(first[2], 10) : 1;
    let html = ordered ? (start !== 1 ? `<ol start="${start}">` : "<ol>") : "<ul>";
    const items = [];
    while (i < lines.length) {
      const m = LIST.exec(lines[i]);
      if (!m) {
        if (!lines[i].trim()) {
          // 空行：下一行仍是同層以上的清單項目才延續
          const next = i + 1 < lines.length ? LIST.exec(lines[i + 1]) : null;
          if (next && next[1].length >= indent) { i++; continue; }
          break;
        }
        if (indentOf(lines[i]) > indent && items.length) {   // 項目的續行
          items[items.length - 1] += "<br>" + inline(lines[i].trim());
          i++;
          continue;
        }
        break;
      }
      const ind = m[1].length;
      if (ind < indent) break;
      if (ind > indent && items.length) {                     // 巢狀清單掛在上一個項目底下
        const [sub, j] = list(lines, i, ind);
        items[items.length - 1] += sub;
        i = j;
        continue;
      }
      if (/\d/.test(m[2]) !== ordered) break;                 // 有序／無序切換 → 另起一個清單
      items.push(inline(lines[i].slice(m[0].length)));
      i++;
    }
    html += items.map(x => "<li>" + x + "</li>").join("");
    return [html + (ordered ? "</ol>" : "</ul>"), i];
  }

  function render(src) {
    const lines = String(src == null ? "" : src).replace(/\r\n?/g, "\n").replace(/\t/g, "    ").split("\n");
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      if (FENCE.test(line)) {
        const buf = [];
        i++;
        while (i < lines.length && !FENCE.test(lines[i])) buf.push(lines[i++]);
        i++;   // 跳過結尾 ```（串流中尚未出現也照樣顯示）
        out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>");
        continue;
      }
      let m = HEADING.exec(line);
      if (m) {
        const level = Math.min(5, m[1].length + 2);   // 泡泡內縮小：# → h3、## → h4、### 以下 → h5
        out.push(`<h${level}>${inline(m[2])}</h${level}>`);
        i++;
        continue;
      }
      if (HR.test(line)) { out.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push("<blockquote>" + render(buf.join("\n")) + "</blockquote>");
        continue;
      }
      if (isTableStart(lines, i)) {
        [m, i] = table(lines, i);
        out.push(m);
        continue;
      }
      if (LIST.test(line)) {
        [m, i] = list(lines, i, indentOf(line));
        out.push(m);
        continue;
      }
      const buf = [];
      while (i < lines.length && lines[i].trim() && (buf.length === 0 || !isBlockStart(lines, i))) {
        buf.push(inline(lines[i].trim()));
        i++;
      }
      out.push("<p>" + buf.join("<br>") + "</p>");
    }
    return out.join("");
  }

  root.renderMarkdown = render;
})(typeof window !== "undefined" ? window : globalThis);
