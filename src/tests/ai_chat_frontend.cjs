// 真正的 markdown.js + ai_chat.js 跑在可控 DOM：狀態燈、拖曳改大小、放大、Enter／選字、串流 Markdown 與依據。
const vm = require('node:vm'), fs = require('node:fs'), assert = require('node:assert/strict');

class El {
  constructor(tag = 'div') {
    Object.assign(this, {tag, children: [], style: {}, dataset: {}, listeners: {}, attrs: {}, hidden: false,
      textContent: '', innerHTML: '', className: '', value: '', disabled: false, scrollHeight: 40, scrollTop: 0});
  }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter(x => x !== f); }
  fire(t, ev = {}) {
    ev.prevented = false;
    ev.preventDefault = () => { ev.prevented = true; };
    for (const f of [...(this.listeners[t] || [])]) f(ev);
    return ev;
  }
  appendChild(c) { this.children.push(c); return c; }
  append(...c) { this.children.push(...c); }
  setAttribute(k, v) { this.attrs[k] = v; }
  focus() {}
  setPointerCapture() {}
  requestSubmit() { this.fire('submit'); }
  getBoundingClientRect() { return {width: parseInt(this.style.width) || 400, height: parseInt(this.style.height) || 620}; }
}

const nodes = new Map();
const el = id => { if (!nodes.has(id)) nodes.set(id, new El()); return nodes.get(id); };
el('aiChat').hidden = true;
const handles = ['n', 'w', 'nw'].map(dir => { const h = new El(); h.dataset.dir = dir; return h; });
const suggest = new El('button'); suggest.dataset.q = '目前機台狀況如何？';

const store = new Map([['aiChatSize', JSON.stringify([5000, 50])]]);   // 存過超出範圍的大小
const requests = [], intervals = [];
const enc = new TextEncoder();
const sse = frames => frames.map(f => enc.encode(`data: ${JSON.stringify(f)}\n\n`));
let replies = [];

const document = {
  hidden: false, body: {style: {}}, listeners: {},
  getElementById: el,
  createElement: tag => new El(tag),
  querySelectorAll: sel => sel === '#aiChat .ai-resize' ? handles : sel === '#aiSuggest button' ? [suggest] : [],
  querySelector: sel => sel === '#aiChat .ai-resize.nw' ? handles[2] : null,
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
};
const ctx = {
  document, TextDecoder, JSON, Math, parseInt, Array, Set, Promise,
  innerWidth: 1200, innerHeight: 900, listeners: {},
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
  localStorage: {getItem: k => store.has(k) ? store.get(k) : null, setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k)},
  setInterval: (f, ms) => intervals.push([f, ms]),
  fetch: async (url, opts) => {
    requests.push([url, opts]);
    if (url === '/api/assistant/status') return {json: async () => ({reachable: true, model: 'qwen3', url: 'http://vllm:8000/v1'})};
    const chunks = replies.shift();
    return {body: {getReader: () => ({read: async () => chunks.length ? {value: chunks.shift(), done: false} : {done: true}})}};
  },
};
ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);   // markdown.js
vm.runInContext(fs.readFileSync(process.argv[3], 'utf8'), ctx);   // ai_chat.js
const tick = () => new Promise(r => setImmediate(r));

(async () => {
  // 狀態燈：開頁即查、每 15 秒輪詢
  await tick();
  assert.equal(requests[0][0], '/api/assistant/status');
  assert.equal(el('aiStatus').textContent, '● qwen3');
  assert.equal(el('aiLed').className, 'ai-led on');
  assert.equal(intervals[0][1], 15000);

  // 還原存過的大小時夾在視窗範圍內（寬 ≤ innerWidth-40，高 ≥ 360）
  assert.equal(el('aiChat').style.width, '1160px');
  assert.equal(el('aiChat').style.height, '360px');

  // 雙擊左上把手還原預設大小並清掉紀錄
  handles[2].fire('dblclick');
  assert.equal(el('aiChat').style.width, '');
  assert.equal(store.has('aiChatSize'), false);

  // 拖左上把手往左上 → 放大；放開才存
  const nw = handles[2];
  nw.fire('pointerdown', {clientX: 100, clientY: 100, pointerId: 1});
  nw.fire('pointermove', {clientX: 40, clientY: 20});
  assert.equal(el('aiChat').style.width, '460px');
  assert.equal(el('aiChat').style.height, '700px');
  assert.equal(store.has('aiChatSize'), false);
  nw.fire('pointerup');
  assert.equal(store.get('aiChatSize'), '[460,700]');
  assert.equal((nw.listeners.pointermove || []).length, 0);   // 放開後解除拖曳
  // 只拖上緣：寬度不變
  handles[0].fire('pointerdown', {clientX: 300, clientY: 300, pointerId: 2});
  handles[0].fire('pointermove', {clientX: 0, clientY: 250});
  handles[0].fire('pointerup');
  assert.equal(el('aiChat').style.width, '460px');
  assert.equal(el('aiChat').style.height, '750px');

  // 放大鈕：放大到視窗、再按還原
  el('aiMax').fire('click');
  assert.equal(el('aiChat').style.width, '1160px');
  assert.equal(el('aiChat').style.height, '792px');
  assert.equal(el('aiMax').textContent, '⤡');
  el('aiMax').fire('click');
  assert.equal(el('aiChat').style.width, '460px');
  assert.equal(el('aiMax').textContent, '⤢');

  // 開啟視窗
  el('aiFab').fire('click');
  assert.equal(el('aiChat').hidden, false);
  assert.equal(store.get('aiChatOpen'), '1');

  // Enter：選字中、Shift+Enter 不送出；一般 Enter 送出
  el('aiInput').value = '實驗一在做什麼？';
  // 選字中：Chrome／Firefox 給 isComposing，Safari 只給 keyCode 229，兩種都要擋
  assert.equal(el('aiInput').fire('keydown', {key: 'Enter', isComposing: true, keyCode: 13}).prevented, false);
  assert.equal(el('aiInput').fire('keydown', {key: 'Enter', isComposing: false, keyCode: 229}).prevented, false);
  assert.equal(el('aiInput').fire('keydown', {key: 'Enter', shiftKey: true}).prevented, false);
  assert.equal(requests.length, 1);
  replies.push(sse([
    {event: 'tool', name: 'get_experiment', label: '讀取實驗手冊', sources: ['web/experiments.py › 實驗一', 'exp1_cold_start.md']},
    {event: 'delta', text: '**結論**：只用健康資料\n'},
    {event: 'delta', text: '- 校準 62 筆\n- <b>不是</b>分類器'},
    {event: 'done', as_of: '實驗頁播放：實驗二 量尺擴張，播放中，已播 120 筆，更新於 21:30:00'},
  ]));
  assert.equal(el('aiInput').fire('keydown', {key: 'Enter'}).prevented, true);
  for (let k = 0; k < 10; k++) await tick();
  const ask = requests.find(r => r[0] === '/api/assistant/ask');
  assert.deepEqual(JSON.parse(ask[1].body), {question: '實驗一在做什麼？', history: []});
  assert.equal(el('aiInput').value, '');

  // 泡泡：使用者靠右純文字；AI 回答轉 Markdown、HTML 被跳脫、底下列出依據
  const rows = el('aiLog').children;
  const [userRow, aiRow] = rows.slice(-2);
  assert.equal(userRow.className, 'ai-row user');
  assert.equal(userRow.children[0].textContent, '實驗一在做什麼？');
  const msg = aiRow.children[1];
  const [tools, body, sources] = msg.children;
  assert.match(tools.textContent, /🔧 讀取實驗手冊/);
  // 「狀態取自」不顯示：伺服器只知道最近一筆紀錄，不知道 AI 實際查了哪份；以下方依據為準
  assert.doesNotMatch(tools.textContent, /實驗頁/);
  assert.equal(body.className, 'ai-md');
  assert.equal(body.innerHTML,
    '<p><strong>結論</strong>：只用健康資料</p><ul><li>校準 62 筆</li><li>&lt;b&gt;不是&lt;/b&gt;分類器</li></ul>');
  assert.equal(sources.textContent, '📎 依據：web/experiments.py › 實驗一；exp1_cold_start.md');

  // 第二問帶上一輪歷史；錯誤顯示為錯誤泡泡
  replies.push(sse([{event: 'error', error: '本地模型呼叫失敗'}, {event: 'done', as_of: '實驗頁播放：無紀錄'}]));
  suggest.fire('click');
  for (let k = 0; k < 10; k++) await tick();
  const second = requests.filter(r => r[0] === '/api/assistant/ask')[1];
  assert.equal(JSON.parse(second[1].body).history.length, 2);
  const errMsg = el('aiLog').children.at(-1).children[1];
  assert.equal(errMsg.className, 'ai-msg error');
  assert.equal(errMsg.children[1].textContent, '本地模型呼叫失敗');

  // 程式核算獨立於模型文字，來源保留，HTML 以純文字處理。
  replies.push(sse([
    {event: 'tool', name: 'get_run_status', label: '讀取實驗執行紀錄', sources: ['指定來源/exp12.json']},
    {event: 'facts', text: '【程式核算】45／572＝7.87% <script>不執行</script>', sources: ['指定來源/exp12.json']},
    {event: 'delta', text: '模型文字不取代程式核算'},
    {event: 'done'},
  ]));
  el('aiInput').value = '實驗十二';
  el('aiForm').requestSubmit();
  for (let k = 0; k < 10; k++) await tick();
  const factMsg = el('aiLog').children.at(-1).children[1];
  const factBox = factMsg.children.find(c => c.className === 'ai-facts');
  assert.equal(factBox.textContent, '【程式核算】45／572＝7.87% <script>不執行</script>');
  assert.equal(factBox.innerHTML, '');
  assert.equal(factBox.style.whiteSpace, 'pre-wrap');
  assert.equal(factMsg.children[1].innerHTML, '<p>模型文字不取代程式核算</p>');
  assert.match(factMsg.children.at(-1).textContent, /指定來源\/exp12.json/);
  replies.push(sse([{event: 'delta', text: '下一輪'}, {event: 'done'}]));
  el('aiInput').value = '繼續';
  el('aiForm').requestSubmit();
  for (let k = 0; k < 10; k++) await tick();
  const next = requests.filter(r => r[0] === '/api/assistant/ask').at(-1);
  const nextHistory = JSON.parse(next[1].body).history;
  assert.equal(nextHistory.at(-1).content, '模型文字不取代程式核算');
  assert.ok(!JSON.stringify(nextHistory).includes('【程式核算】'));
  console.log('ai chat frontend OK');
})().catch(err => { console.error(err); process.exit(1); });
