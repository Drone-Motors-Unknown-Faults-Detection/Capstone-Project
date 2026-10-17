/* 執行真正 experiments.js；合成 DOM 只驗呈現契約，不冒充瀏覽器驗收。 */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value:'', innerHTML:'', textContent:'', disabled:false,
    hidden:false, querySelectorAll:() => []});
  return elements.get(id);
}
const sandbox = {console, document:{getElementById:element, querySelectorAll:() => []},
  window:{addEventListener(){}, dispatchEvent(){}}, Event:class {}, location:{hash:''}};
vm.createContext(sandbox);
let source = fs.readFileSync(path.join(__dirname, '../web/static/experiments.js'), 'utf8').replace(/\r\n/g, '\n');
assert.ok(source.includes('\ninit();\n'));
source = source.replace('\ninit();\n', '\nObject.assign(globalThis, {cardHtml, resultContext, refreshResultContext, showResult, runExp, streamExp, renderPage, RENDER, COMMITTED_RENDER, setCatalog:v=>CATALOG=v});\n');
vm.runInContext(source, sandbox);
const e = {id:'exp1', no:'實驗一', title:'測試', question:'測試', code:'測試', doc:'測試', fits:'健康基準',
  data_roles:'健康train/cal/test', limitations:'非獨立採集', stream:true,
  params:[{name:'dataset', type:'dataset'}, {name:'seed', type:'int', default:42}]};
const catalog = {datasets:['T1/8000rpm'], configs:[], experiments:[e]};
sandbox.setCatalog(catalog);
element('f-exp1-dataset').value = 'T1/8000rpm';
element('f-exp1-seed').value = '42';
const payload = {params:{dataset:'T1/8000rpm', seed:42}, seconds:1, finished_at:'2026-10-14T19:00:00',
  result:{healthy_fp_rate:0, macro_detect_rate:1, macro_auroc:1, rows:[], model:{score_type:'測試', threshold_strategy:'q95',
    classes:[{config:'8screws', n_train:187, n_cal:62, n_holdout:64}]}}};
assert.match(sandbox.resultContext(e, payload), /與目前選項一致/);
element('f-exp1-seed').value = '123';
assert.match(sandbox.resultContext(e, payload), /舊設定結果/);
assert.match(sandbox.resultContext(e, payload), /2026-10-14/);
sandbox.showResult(e, payload);
assert.match(element('out-exp1').innerHTML, /本次模型資料用途/);
assert.match(element('out-exp1').innerHTML, /187/);
assert.match(element('out-exp1').innerHTML, /舊設定結果/);
sandbox.setCatalog({...catalog, datasets:[]});
const empty = sandbox.cardHtml(e);
assert.match(empty, /id="run-exp1" disabled/);
assert.match(empty, /id="stream-exp1" disabled/);
assert.match(empty, /請管理者確認/);
sandbox.setCatalog(catalog);
const allCard = sandbox.cardHtml({...e, params:[{name:'seed', type:'int', default:42}], committed:'archive'});
assert.match(allCard, /掃描到的 1 工況/);
assert.match(allCard, /與上方選項及本次執行分開/);
const paired = sandbox.COMMITTED_RENDER.exp6_formal_matrix({paired_differences:[], macro_summary:[]});
assert.match(paired, /差值&gt;0/);
assert.match(paired, /FPR 越低越好/);
assert.doesNotMatch(paired, /k-NN 較好/);
const transfer = sandbox.RENDER.exp10({n_datasets:2, seed:42, method:'ledoit_wolf', rows:[], summary:{direct:[{n:0, auroc_mean:0.5, healthy_accept_mean:0.4, fault_detect_mean:0.6}], adapted:[], scratch:[]}});
assert.match(transfer, /direct/);
assert.match(transfer, /目標健康筆數/);
const ancestor = sandbox.RENDER.exp11({n_datasets:1, seeds:[42], primary_metrics:['macro_f1'], rows:[], summary:{A:{ledoit_wolf:{accuracy_mean:0.5, balanced_accuracy_mean:0.5, macro_f1_mean:0.5, unknown_f1_mean:0.5}}}, wilcoxon:null});
assert.match(ancestor, /Balanced Accuracy/);
const picture = sandbox.RENDER.exp12({n_datasets:1, seed:42, _tsne_by_dataset:{}, binary_confusion:{labels:['healthy','fault'], matrix:[[1,0],[0,1]]}, openset_confusion:{ledoit_wolf:{labels:['unknown'], matrix:[[1]]}, legacy:{labels:['unknown'], matrix:[[1]]}}});
assert.match(picture, /健康 vs 故障/);
assert.match(picture, /t-SNE/);
const formal = sandbox.RENDER.exp6_formal({rows:[], method:'mahalanobis', seed:42});
assert.match(formal, /健康訓練/);
assert.match(formal, /健康校準/);
assert.match(formal, /健康接受率/);

(async () => {
  element('f-exp1-seed').value = '42';
  sandbox.fetch = async () => ({ok:false, json:async () => ({error:'測試失敗'})});
  await sandbox.runExp(e);
  assert.match(element('status-exp1').textContent, /測試失敗/);
  assert.equal(element('run-exp1').disabled, false);
  sandbox.fetch = async () => ({ok:true, json:async () => payload});
  await sandbox.runExp(e);
  assert.match(element('status-exp1').textContent, /完成/);
  element('f-exp1-seed').value = '123';
  sandbox.refreshResultContext(e);
  assert.match(element('context-exp1').textContent, /舊設定結果/);
  sandbox.renderPage('exp1');
  assert.equal(typeof element('f-exp1-seed').oninput, 'function');
  const streams = [];
  sandbox.EventSource = class {
    constructor(url) { this.url = url; streams.push(this); }
    close() { this.closed = true; }
  };
  element('rate-exp1').value = '80';
  sandbox.streamExp(e);
  sandbox.streamExp(e);
  assert.equal(streams[0].closed, true);
  streams[0].onmessage({data:JSON.stringify({event:'done', payload:{...payload, finished_at:'過期結果'}})});
  assert.doesNotMatch(element('out-exp1').innerHTML, /過期結果/);
  streams[1].onmessage({data:JSON.stringify({event:'done', payload})});
  assert.equal(streams[1].closed, true);
  assert.match(element('out-exp1').innerHTML, /2026-10-14/);
  console.log('前端契約通過：設定／舊結果、空資料、來源與用途、失敗後重試');
})().catch(error => { console.error(error); process.exitCode = 1; });
