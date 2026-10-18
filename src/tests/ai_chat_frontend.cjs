// 真正的 markdown.js + ai_chat.js 跑在可控 DOM：狀態燈、拖曳改大小、放大、Enter／選字、串流 Markdown 與依據。
const vm = require('node:vm'), fs = require('node:fs'), assert = require('node:assert/strict');

class El {
  constructor(tag = 'div') {
    this.htmlWrites = [];
    Object.assign(this, {tag, children: [], style: {}, dataset: {}, listeners: {}, attrs: {}, hidden: false,
      textContent: '', innerHTML: '', className: '', value: '', disabled: false, scrollHeight: 40, scrollTop: 0});
  }
  set innerHTML(value) { this._html = value; this.htmlWrites.push(value); }
  get innerHTML() { return this._html; }
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
  select() { this.selected = true; }
  setPointerCapture() {}
  requestSubmit() { this.fire('submit'); }
  getBoundingClientRect() { return {width: parseInt(this.style.width) || 400, height: parseInt(this.style.height) || 620}; }
}

Object.defineProperty(El.prototype, 'textContent', {
  set(value) { this._text = String(value); this.children = []; this._html = ''; },
  get() { return (this._text || '') + (this.children || []).map(c => c.textContent).join(''); },
});

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
const acceptedFrames = process.argv[4] === '--frames' ? JSON.parse(fs.readFileSync(0, 'utf8')) : null;

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
vm.runInContext(fs.readFileSync(require('node:path').join(require('node:path').dirname(process.argv[3]), 'assistant_presentation.js'), 'utf8'), ctx);
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
  assert.match(tools.textContent, /已完成.*工具明細.*讀取實驗手冊/);
  assert.doesNotMatch(tools.textContent, /→/);
  // 「狀態取自」不顯示：伺服器只知道最近一筆紀錄，不知道 AI 實際查了哪份；以下方依據為準
  assert.doesNotMatch(tools.textContent, /實驗頁/);
  assert.equal(body.className, 'ai-md');
  assert.equal(body.innerHTML,
    '<p><strong>結論</strong>：只用健康資料</p><ul><li>校準 62 筆</li><li>&lt;b&gt;不是&lt;/b&gt;分類器</li></ul>');
  assert.equal(sources.children[1].children.length, 2);
  assert.match(sources.textContent, /web\/experiments.py › 實驗一/);
  assert.match(sources.textContent, /exp1_cold_start.md/);

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

  // 結果模式只顯示具matrix來源的正文；錯誤自由delta在任何中間畫面都不能閃出。
  const matrixContext = {experiment: 'exp12', mode: 'result', saved: 'output/web_server/session/experiments/exp12_2026-01-01-00-00-00.json'};
  replies.push(sse([
    {event: 'mode', mode: 'matrix_summary', context: matrixContext, sources: ['指定結果來源']},
    {event: 'delta', text: 'legacy模型完全失效，接近100%，每類46～47，unknown無誤報'},
    {event: 'facts', text: '不應重複顯示的核算框'},
    {event: 'delta', origin: 'matrix', text: '結果摘要由程式依矩陣產生：2736／2938＝93.12% <script>不執行</script>'},
    {event: 'done'},
  ]));
  el('aiInput').value = '總結實驗十二';
  el('aiForm').requestSubmit();
  for (let k = 0; k < 10; k++) await tick();
  const primary = el('aiLog').children.at(-1).children[1];
  assert.equal(primary.attrs['data-response-mode'], 'matrix_summary');
  assert.ok(primary.children[1].innerHTML.includes('2736／2938＝93.12%'));
  assert.ok(primary.children[1].innerHTML.includes('&lt;script&gt;'));
  assert.ok(primary.children[1].htmlWrites.every(value => !/完全失效|接近100%|46～47|unknown無誤報/.test(value)));
  assert.ok(!primary.children.some(c => c.className === 'ai-facts'));
  assert.match(primary.children.at(-1).textContent, /指定結果來源/);
  replies.push(sse([
    {event: 'mode', mode: 'matrix_summary', context: matrixContext, sources: ['指定結果來源']},
    {event: 'done'},
  ]));
  el('aiInput').value = '那是不是都誤拒';
  el('aiForm').requestSubmit();
  for (let k = 0; k < 10; k++) await tick();
  const follow = JSON.parse(requests.filter(r => r[0] === '/api/assistant/ask').at(-1)[1].body);
  assert.deepEqual(follow.context, matrixContext);
  assert.ok(!JSON.stringify(follow.history).includes('2736／2938'));
  assert.ok(!JSON.stringify(follow.history).includes('完全失效'));
  assert.equal(el('aiLog').children.at(-1).children[1].children[1].textContent, '（程式未提供可用摘要）');
  // 有限降級不冒充模型文字，按鈕恢復後仍能繼續提問。
  replies.push(sse([
    {event: 'mode', mode: 'verified_fallback', status: 'incomplete', sources: ['已確認來源']},
    {event: 'delta', text: '不應顯示的自由文字'},
    {event: 'delta', text: '查詢已達上限，以下只保留已確認資料。', origin: 'fallback'},
    {event: 'done'}
  ]));
  el('aiInput').value = '綜合方法問題'; el('aiForm').requestSubmit();
  for (let k = 0; k < 8; k++) await tick();
  const limited = el('aiLog').children.at(-1).children[1];
  assert.equal(limited.attrs['data-response-mode'], 'verified_fallback');
  assert.ok(limited.children[1].innerHTML.includes('查詢已達上限'));
  assert.ok(limited.children[1].htmlWrites.every(x => !x.includes('不應顯示的自由文字')));
  assert.equal(el('aiSend').disabled, false);
  replies.push(sse([{event: 'delta', text: '下一輪可用'}, {event: 'done'}]));
  el('aiInput').value = '下一輪'; el('aiForm').requestSubmit();
  for (let k = 0; k < 8; k++) await tick();
  assert.ok(!JSON.stringify(JSON.parse(requests.filter(r => r[0] === '/api/assistant/ask').at(-1)[1].body).history).includes('查詢已達上限'));
  replies.push(sse([
    {event:'mode', mode:'record_timing', context:{experiment:'exp1', mode:'result'}, sources:['時間原檔']},
    {event:'delta', text:'猜測大約開始時間'},
    {event:'delta', origin:'record', text:'開始時間：原檔未記錄；完成時間：2026-10-18T12:11:13（原檔未提供時區）；耗時：50.15 秒。'},
    {event:'done'}
  ]));
  el('aiInput').value = '何時開始、何時完成？'; el('aiForm').requestSubmit();
  for (let k = 0; k < 8; k++) await tick();
  const timed = el('aiLog').children.at(-1).children[1];
  assert.equal(timed.attrs['data-response-mode'], 'record_timing');
  assert.ok(timed.children[1].innerHTML.includes('原檔未記錄'));
  assert.ok(timed.children[1].htmlWrites.every(value => !value.includes('猜測大約')));
  assert.equal(el('aiSend').disabled, false);
  // UTF-8中文字拆成多個chunk，完成事件重送不重複依據。
  const utf = sse([{event:'delta', text:'分段中文正確'}, {event:'done'}, {event:'done'}]);
  const encoded = utf[0], at = encoded.indexOf(0xe5);
  replies.push([encoded.slice(0, at + 1), encoded.slice(at + 1, at + 2), encoded.slice(at + 2), ...utf.slice(1)]);
  el('aiInput').value = '分段'; el('aiForm').requestSubmit();
  for (let k = 0; k < 10; k++) await tick();
  const split = el('aiLog').children.at(-1).children[1];
  assert.ok(split.children[1].innerHTML.includes('分段中文正確'));
  assert.equal(split.children.filter(c => c.className === 'ai-sources').length, 1);
  // 沒有done、錯誤或壞JSON不能假完成，也不能污染下一輪歷史。
  for (const chunks of [sse([{event:'delta', text:'未封存的回答'}]),
    sse([{event:'delta', text:'未封存的回答'}, {event:'error', error:'中途失敗'}, {event:'done'}]),
    [enc.encode('data: {壞JSON}\n\n')]]) {
    replies.push(chunks); el('aiInput').value = '斷線負例'; el('aiForm').requestSubmit();
    for (let k = 0; k < 10; k++) await tick();
    const failed = el('aiLog').children.at(-1).children[1];
    assert.equal(failed.className, 'ai-msg error'); assert.equal(el('aiSend').disabled, false);
    assert.ok(!failed.children[0].textContent.startsWith('已完成'));
  }
  replies.push(sse([{event:'delta', text:'錯誤後可重試'}, {event:'done'}]));
  el('aiInput').value = '重試'; el('aiForm').requestSubmit();
  for (let k = 0; k < 10; k++) await tick();
  const retried = JSON.parse(requests.filter(r => r[0] === '/api/assistant/ask').at(-1)[1].body);
  assert.ok(!JSON.stringify(retried.history).includes('未封存的回答'));
  // 模型自己送presentation，沒有可信mode綁定，仍不能建立可信結果。
  replies.push(sse([{event:'presentation', presentation:{schema_version:'assistant_presentation_v1', mode:'matrix_summary', title:'假可信結果'}},
    {event:'delta', text:'模型普通文字'}, {event:'done'}]));
  el('aiInput').value = '假資料'; el('aiForm').requestSubmit();
  for (let k = 0; k < 10; k++) await tick();
  assert.equal(el('aiLog').children.at(-1).children[1].children[1].innerHTML, '<p>模型普通文字</p>');
  if (acceptedFrames) {
    // 直接消費正式HTTP handler產生的SSE，不以手寫正確答案替代資料鏈。
    replies.push(sse(acceptedFrames));
    el('aiInput').value = '實驗十二結果';
    el('aiForm').requestSubmit();
    for (let k = 0; k < 10; k++) await tick();
    const actual = el('aiLog').children.at(-1).children[1];
    if (acceptedFrames.some(f => f.event === 'presentation')) {
      assert.equal(actual.children[1].children.filter(c => c.className === 'ai-presentation').length, 1);
      assert.ok(!actual.children[1].innerHTML.includes('<table>'));
      const copy = actual.children[1].children[0].children.find(c => c.tag === 'button' && c.textContent === '複製完整文字');
      assert.ok(copy);
      const source = actual.children[1].children[0].children.find(c => c.attrs['data-section'] === 'source_details');
      assert.equal(source.tag, 'details');
      assert.equal(source.open, acceptedFrames.find(f => f.event === 'presentation').presentation.sections.find(s => s.id === 'source_details').open);
    }
    if (acceptedFrames[0].mode === 'record_timing') {
      assert.equal(actual.attrs['data-response-mode'], 'record_timing');
      const rendered = actual.children[1].innerHTML + actual.children[1].textContent;
      assert.ok(rendered.includes('開始時間'));
      assert.ok(rendered.includes('完成時間'));
      assert.ok(rendered.includes('耗時'));
      assert.equal(el('aiSend').disabled, false);
    } else {
    assert.equal(actual.attrs['data-response-mode'], 'matrix_summary');
    const rendered = actual.children[1].innerHTML + actual.children[1].textContent;
    const view = acceptedFrames.find(f => f.event === 'presentation')?.presentation;
    if (!view || view.sections.some(s => s.id === 'methods')) {
      assert.ok(rendered.includes('2736／2938＝93.12%'));
      assert.ok(rendered.includes('202／2938＝6.88%'));
    } else assert.ok(rendered.includes(view.title));
    assert.ok(actual.children[1].htmlWrites.every(value => !/legacy 模型完全失效|接近100%|46～47|unknown無誤報/.test(value)));
    assert.ok(!actual.children.some(c => c.className === 'ai-facts'));
    assert.ok(actual.children.at(-1).textContent.includes(acceptedFrames[0].sources[0]));
    }
    console.log('HTTP to frontend primary OK');
  }
  console.log('ai chat frontend OK');
})().catch(err => { console.error(err); process.exit(1); });
