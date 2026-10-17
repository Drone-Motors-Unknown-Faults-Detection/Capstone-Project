// 真正的 src/web/static/markdown.js：跳脫、清單巢狀、表格與串流中未閉合的格式。
const vm = require('node:vm'), fs = require('node:fs'), assert = require('node:assert/strict');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const md = context.renderMarkdown;

// 模型輸出夾帶 HTML／腳本一律跳脫；javascript: 連結不轉成 <a>
assert.equal(md('<script>alert(1)</script>'), '<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>');
assert.equal(md('<img src=x onerror=alert(1)>'), '<p>&lt;img src=x onerror=alert(1)&gt;</p>');
assert.equal(md('[按我](javascript:alert(1))'), '<p>[按我](javascript:alert(1))</p>');
assert.equal(md('[手冊](https://example.com/a?b=1&c="x")'),
  '<p><a href="https://example.com/a?b=1&amp;c=&quot;x&quot;" target="_blank" rel="noopener noreferrer">手冊</a></p>');
assert.equal(md('`<b>`'), '<p><code>&lt;b&gt;</code></p>');
assert.equal(md('```\n<div>x</div>\n```'), '<pre><code>&lt;div&gt;x&lt;/div&gt;</code></pre>');

// 行內格式；檔名底線不當斜體；乘號不當斜體
assert.equal(md('**輸出指標**：健康誤報率'), '<p><strong>輸出指標</strong>：健康誤報率</p>');
assert.equal(md('*斜體* 與 ~~刪除~~'), '<p><em>斜體</em> 與 <del>刪除</del></p>');
assert.equal(md('見 exp1_cold_start.md 與 __init__'), '<p>見 exp1_cold_start.md 與 __init__</p>');
assert.equal(md('2 * 3 * 4'), '<p>2 * 3 * 4</p>');
assert.equal(md('**未閉合'), '<p>**未閉合</p>');   // 串流途中：保持原字

// 截圖中的回答形狀：有序清單 → 無序清單 → 巢狀項目
const answer = [
  ' 1. 從健康資料中以 **187** 筆訓練',
  ' 2. 使用 62 筆作為校準集',
  '- **輸出指標**：健康誤報率、AUROC',
  '- **關鍵限制**：',
  '  - 模型僅學習「健康長什麼樣」',
  '  - 分數僅表示偏離程度',
].join('\n');
assert.equal(md(answer),
  '<ol><li>從健康資料中以 <strong>187</strong> 筆訓練</li><li>使用 62 筆作為校準集</li></ol>' +
  '<ul><li><strong>輸出指標</strong>：健康誤報率、AUROC</li>' +
  '<li><strong>關鍵限制</strong>：<ul><li>模型僅學習「健康長什麼樣」</li><li>分數僅表示偏離程度</li></ul></li></ul>');
assert.equal(md('3. 第三步\n4. 第四步'), '<ol start="3"><li>第三步</li><li>第四步</li></ol>');
assert.equal(md('- 第一項\n  續行\n- 第二項'), '<ul><li>第一項<br>續行</li><li>第二項</li></ul>');

// 標題、段落、引用、分隔線
assert.equal(md('# 結論\n目前健康\n分數 0.94\n\n---\n> 依據 exp1'),
  '<h3>結論</h3><p>目前健康<br>分數 0.94</p><hr><blockquote><p>依據 exp1</p></blockquote>');

// 表格（含對齊與缺格）
assert.equal(md('| 指標 | 值 |\n|:--|--:|\n| EWMA | 0.10 |\n| CUSUM |'),
  '<div class="md-table"><table><thead><tr><th style="text-align:left">指標</th><th style="text-align:right">值</th></tr></thead>' +
  '<tbody><tr><td style="text-align:left">EWMA</td><td style="text-align:right">0.10</td></tr>' +
  '<tr><td style="text-align:left">CUSUM</td><td style="text-align:right"></td></tr></tbody></table></div>');

// 串流中尚未出現結尾 ``` 也能顯示；空字串不出錯
assert.equal(md('```\nprint(1)'), '<pre><code>print(1)</code></pre>');
assert.equal(md(''), '');
assert.equal(md(null), '');
console.log('markdown frontend OK');
