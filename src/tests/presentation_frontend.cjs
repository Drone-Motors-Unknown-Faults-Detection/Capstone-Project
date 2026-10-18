// 固定DOM模板的來源綁定、折疊與HTTP複製備援；不冒充真實瀏覽器。
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.events = {}; this._text = ''; this.htmlWrites = []; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(v) { this.htmlWrites.push(v); }
  append(...v) { this.children.push(...v); }
  appendChild(v) { this.append(v); return v; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(k, v) { this.events[k] = v; }
  focus() { this.focused = true; }
  select() { this.selected = true; }
}
const ctx = {document: {createElement: tag => new El(tag)}, navigator: {}};
ctx.window = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const api = ctx.AssistantPresentation;
const input = JSON.parse(fs.readFileSync(0, 'utf8')), p = input.presentation, binding = input.binding;
const all = e => [e, ...e.children.flatMap(all)];
(async () => {
  const target = new El('div');
  assert.equal(api.render(target, p, binding, input.text), true);
  const nodes = all(target);
  assert.ok(nodes.every(e => e.htmlWrites.length === 0));
  assert.equal(nodes.filter(e => e.attrs['data-section'] === 'source_details').length, 1);
  const source = nodes.find(e => e.attrs['data-section'] === 'source_details');
  assert.equal(source.tag, 'details'); assert.equal(source.children[0].tag, 'summary');
  assert.equal(source.open, false); source.open = true; assert.equal(source.open, true); source.open = false;
  const warning = nodes.find(e => e.attrs['data-section'] === 'warnings');
  assert.notEqual(warning.tag, 'details');
  assert.ok(target.textContent.includes('45／572＝7.87%'));
  assert.ok(target.textContent.includes('原檔未記錄'));
  assert.ok(nodes.some(e => e.tag === 'thead') && nodes.some(e => e.tag === 'tbody'));
  assert.ok(nodes.filter(e => e.className === 'ai-table-scroll').every(e => e.tabIndex === 0 && e.attrs.role === 'region'));
  // 無Clipboard及拒絕權限都只能提供手動選取，完整字元不縮寫。
  const button = all(source).find(e => e.tag === 'button');
  await button.events.click();
  let manual = all(source).find(e => e.tag === 'textarea');
  assert.equal(manual.value, p.source.saved); assert.equal(manual.selected, true); assert.equal(manual.readOnly, true);
  assert.ok(!source.textContent.includes('已複製'));
  ctx.navigator.clipboard = {writeText: async () => { throw new Error('拒絕'); }};
  await button.events.click();
  assert.equal(all(source).filter(e => e.tag === 'textarea').length, 1);
  assert.ok(!source.textContent.includes('已複製'));
  let copied;
  ctx.navigator.clipboard.writeText = async value => { copied = value; };
  await button.events.click(); assert.equal(copied, p.source.saved); assert.ok(source.textContent.includes('已複製'));
  // 任意模型HTML或換來源不能取得可信區段。
  assert.equal(api.valid(p, {...binding, mode: 'model'}), false);
  assert.equal(api.valid(p, {...binding, source_id: 'b'.repeat(64)}), false);
  const changed = structuredClone(p); changed.source.saved = '另一筆';
  assert.equal(api.valid(changed, binding), false);
  assert.equal(api.valid({...p, origin: 'model'}, binding), false);
  assert.equal(api.valid({...p, sections: [{...p.sections[0], kind: 'html'}]}, binding), false);
  assert.equal(api.valid({...p, sections: [{...warning, id: 'w', title: '警告', kind: 'warning', items: ['矛盾'], detail: true, open: false}]}, binding), false);
  const evil = structuredClone(p); evil.title = '<img src=x onerror=alert(1)>';
  evil.sections[0].rows[0][1] = '<script>不執行</script>|`\n';
  const safe = new El('div'); assert.equal(api.render(safe, evil, binding, input.text), true);
  assert.ok(safe.textContent.includes(evil.title)); assert.ok(all(safe).every(e => !['script', 'img'].includes(e.tag)));
  const sources = new El('div');
  api.sources(sources, ['同一來源', '同一來源', {id:'run', label:'同名', path:'a/exp12.json', sha256:'a'.repeat(64)},
    {id:'run', label:'同名', path:'b/exp12.json', sha256:'b'.repeat(64)},
    {id:'unsafe', label:'不連結', url:'javascript:alert(1)'}, {id:'safe', label:'手冊', url:'https://example.org/manual'}]);
  assert.equal(sources.children[1].children.length, 5);
  assert.equal(all(sources).filter(e => e.tag === 'a').length, 1);
  assert.ok(all(sources).some(e => e.tag === 'a' && e.rel === 'noopener noreferrer'));
  const empty = new El('div'); api.sources(empty, []); assert.ok(empty.textContent.includes('沒有取得依據'));
  assert.equal(api.displayTime('2026-10-18T12:37:51.477+00:00'), '2026-10-18 20:37:51.477（Asia/Taipei，UTC+08:00）');
  assert.equal(api.displayTime('2026-10-18T12:11:13'), '2026-10-18T12:11:13（原檔未提供時區）');
  const info = new El('div'); api.runInfo(info, {params: {seed:42}, finished_at:'2026-10-18T12:11:13', seconds:50.15}, 'exp1');
  assert.ok(info.textContent.includes('開始時間：原檔未記錄'));
  assert.ok(info.textContent.includes('原檔未提供時區'));
  assert.ok(!info.textContent.includes('已驗證'));
  console.log('presentation DOM OK：來源綁定、表格、折疊、複製與安全文字');
})().catch(err => { console.error(err); process.exit(1); });
