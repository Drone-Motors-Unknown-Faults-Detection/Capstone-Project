// 以 Node 執行首頁 index.html 裡真正的 onMetrics()：健康來源擴張前的判未知率是誤報率，不上色；故障來源照偵測率上色（#305）
const fs = require("fs");
const assert = require("assert");
const html = fs.readFileSync(process.argv[2], "utf-8");
const start = html.indexOf("function onMetrics(m){");
const end = html.indexOf("\n}\n", start);
assert(start > 0 && end > start, "找不到 onMetrics");
const tbody = {innerHTML: ""};
const $ = () => ({querySelector: () => tbody});
const escapeHtml = s => String(s);
const state = null;
eval(html.slice(start, end + 2));
onMetrics({rows: [
  {config: "8screws", display: "健康", pre_streamed: 20, pre_rate: 3.0, post_streamed: 0},
  {config: "5screws", display: "5 顆", pre_streamed: 20, pre_rate: 98.0, post_streamed: 0},
  {config: "6screws", display: "6 顆", pre_streamed: 20, pre_rate: 40.0, post_streamed: 0},
]});
const rows = tbody.innerHTML.split("</tr>").filter(Boolean);
assert(!rows[0].includes("style="), "健康列不上色：" + rows[0]);
assert(rows[0].includes("3.0%"));
assert(rows[1].includes("var(--green)"), rows[1]);
assert(rows[2].includes("var(--yellow)"), rows[2]);
console.log("ok");
