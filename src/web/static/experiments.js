/* 實驗頁：頁首切換列 + 實驗一～十二的參數表單、執行與結果呈現。
 * 只負責呈現；▶ 執行呼叫 run()，⏵ 邊跑邊畫呼叫 iter_run()／iter_playback()（見 web/experiments.py）。 */
"use strict";
(() => {
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let CATALOG = null;
const LAST = {};          // id -> 最近一次 payload（切頁回來還在）
const COMMITTED = {};     // name -> 已提交結果

/* ---------- 格式化 ---------- */
const num = (v, d = 3) => v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d);
const pct = (v, d = 1) => v === null || v === undefined ? "—" : (Number(v) * 100).toFixed(d) + "%";
const int = v => v === null || v === undefined ? "—" : String(Math.round(v));
const bool = v => v ? "✓" : "✗";
const pm = (m, s, f = num) => `${f(m)} ± ${f(s)}`;

function kpis(items){
  return `<div class="kpis">${items.map(([l, v]) =>
    `<div class="kpi"><div class="v">${v}</div><div class="l">${esc(l)}</div></div>`).join("")}</div>`;
}
function table(cols, rows){
  const head = cols.map(c => `<th>${esc(c[1])}</th>`).join("");
  const body = rows.map(r => "<tr>" + cols.map(c => {
    const v = c[2] ? c[2](r[c[0]], r) : r[c[0]];
    return `<td>${typeof v === "string" && v.startsWith("<") ? v : esc(v)}</td>`;
  }).join("") + "</tr>").join("");
  return `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
function bar(v, max = 1, color){
  const w = Math.max(0, Math.min(1, (v ?? 0) / max)) * 100;
  return `<div class="bar"><div style="width:${w}%;${color ? "background:" + color : ""}"></div></div>`;
}
// 0→紅、0.5→黃、1→綠
function heatColor(v, lo = 0, hi = 1){
  if (v === null || v === undefined) return "#334155";
  const x = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const h = x * 120;
  return `hsl(${h}, 70%, 55%)`;
}
function heatmap(rowLabels, colLabels, valueOf, opts = {}){
  const f = opts.fmt || (v => num(v, 2));
  const head = `<th>${esc(opts.corner || "")}</th>` + colLabels.map(c => `<th style="text-align:center">${esc(c)}</th>`).join("");
  const body = rowLabels.map(r => `<tr><th style="text-align:left">${esc(r)}</th>` + colLabels.map(c => {
    const v = valueOf(r, c);
    return `<td class="heat" style="background:${heatColor(v, opts.lo ?? 0, opts.hi ?? 1)}">${f(v)}</td>`;
  }).join("") + "</tr>").join("");
  return `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
const details = (title, html) => `<details><summary>${esc(title)}</summary>${html}</details>`;
const modelLine = m => m ? `<div class="mini">模型：${esc(m.openset_method)}${m.openset_method === "knn" ? `（k=${m.knn_neighbors}）` : `（${esc(m.method)}）`}；閾值策略 ${esc(m.threshold_strategy)}；seed ${esc(m.random_seed)}</div>` : "";

/* ---------- 各實驗的結果呈現 ---------- */
const exp1Table = rows => table([["display", "配置"], ["kind", "類型"], ["n", "樣本數"], ["detect_rate", "判未知比例", v => pct(v)],
  ["detect_rate", "", (v, row) => bar(v, 1, row.kind.startsWith("healthy") ? "var(--green)" : "var(--red)")],
  ["auroc", "AUROC", v => num(v, 4)], ["median_score", "分數中位數", v => num(v, 2)]], rows);
const RENDER = {
  exp1(r){
    return kpis([["健康誤報率（holdout）", pct(r.healthy_fp_rate)], ["故障平均偵測率", pct(r.macro_detect_rate)], ["故障平均 AUROC", num(r.macro_auroc, 4)]])
      + modelLine(r.model) + exp1Table(r.rows);
  },
  exp2(r){
    // 形成新類（discovered）不等於辨識出本段配置；判定欄位由實驗層給（n_matched_config／matched_is_config，#286）。
    // 舊存檔沒有總數欄位時才從 stage 列數。
    const n = r.n_stages ?? r.stages.length;
    const matched = r.n_matched_config ?? r.stages.filter(s => s.matched_is_config).length;
    const learned = r.n_learned ?? r.stages.filter(s => s.discovered).length;
    return kpis([["對上本階段配置", `${matched} / ${n}`], ["形成新類", `${learned} 次`], ["最終已知類別", r.final_known.length]])
      + `<div class="mini">「形成新類」只表示該段學到一個 discovered-N；對不對得上本段注入的配置，看事後對照（候選群真值多數）。</div>`
      + modelLine(r.model)
      + table([["display", "依序注入"], ["discovered", "形成新類", bool], ["matched_is_config", "對上本階段配置", bool],
               ["learned_display", "模型新類"], ["matched_display", "事後對照（真值多數）"], ["samples_to_candidate", "到候選所需筆數", int],
               ["cluster_size", "叢大小", int], ["cluster_purity", "純度", v => pct(v, 0)],
               ["own_holdout_acc", "本類 holdout 認出率", v => pct(v)], ["own_holdout_n", "本類 holdout 筆數", int],
               ["eval_known_rate", "外部評估被任一新類接受（學會當下）", v => pct(v)], ["eval_matched_rate", "判進對應新類（學會當下）", v => pct(v)], ["eval_n", "外部評估筆數", int],
               ["healthy_holdout_acc", "健康 holdout 認出率", v => pct(v)],
               ["known_after", "之後已知類別數", int]], r.stages);
  },
  exp3(r){
    const rows = Object.entries(r.summary).map(([k, s]) => ({key: k, ...s}));
    return modelLine(r.model)
      + table([["name", "劇本"], ["expected", "應判為"], ["trials", "次數"], ["alarm_rate", "有警報", v => pct(v, 0)],
               ["correct_rate", "判別正確", v => pct(v, 0)], ["correct_rate", "", v => bar(v)], ["mean_latency", "平均警報延遲（筆）", v => num(v, 1)],
               ["mean_transition", "平均中間帶停留（筆）", v => num(v, 1)]], rows)
      + details(`逐次結果（${r.trials.length} 次）`, table([["scenario", "劇本"], ["trial", "#"], ["expected", "應判"], ["verdict", "判為"],
               ["correct", "正確", bool], ["alarm_t", "警報時刻", int], ["latency", "延遲", int], ["transition", "中間帶停留", int]], r.trials));
  },
  exp4(r){
    let h = "";
    if (r.geometry){
      const g = r.geometry, names = g.ray_names;
      h += `<h3>(a) 射線結構：故障方向兩兩夾角 cos</h3>`
        + heatmap(names, names, (a, b) => g.cos_matrix[names.indexOf(a)][names.indexOf(b)], {lo: 0, hi: 1})
        + table([["k", "群組統計"], ["v", "平均 cos", v => num(v, 4)]], Object.entries(g.group_stats).map(([k, v]) => ({k, v})))
        + `<div class="mini">PolarMap 基準：${esc(g.polar.polarmap_base_method)}，τ 取第 ${esc(g.polar.tau_percentile)} 百分位</div>`
        + table([["config", "射線"], ["radius", "半徑", v => num(v, 1)], ["tau", "τ", v => num(v, 3)]], g.polar.rays);
    }
    if (r.direction){
      h += `<h3>(b) 方向熟悉度：只認識部分故障時，未知樣本的 cos</h3>`
        + table([["known", "已知射線", v => v.join("、")], ["auroc_family_vs_4_146", "家族 vs 4_146 AUROC", v => num(v, 4)],
                 ["median_family", "家族 cos 中位數", v => num(v, 3)], ["median_4_146", "4_146 cos 中位數", v => num(v, 3)],
                 ["median_3_14", "3_14 cos 中位數", v => num(v, 3)]], r.direction.scenarios);
    }
    if (r.severity){
      const s = r.severity;
      h += `<h3>(c) 嚴重度回復（錨點：${esc(s.anchors.join("、"))}）</h3>`
        + kpis(Object.entries(s.spearman).map(([k, v]) => [`Spearman（${k}）`, num(v, 3)]))
        + table([["config", "配置"], ["rank", "真實排序", int], ["n", "樣本數", int], ["sev_radius_median", "半徑嚴重度", v => num(v, 3)],
                 ["sev_best_ray_median", "最近射線嚴重度", v => num(v, 3)], ["sev_bundle_median", "射線束嚴重度", v => num(v, 3)]], s.rows);
    }
    return h;
  },
  exp5(r){
    const keys = r.matrix.keys, cell = (a, b) => r.matrix.cells[`${a}->${b}`];
    const sum = r.strategy.summary;
    return `<h3>(a) 9×9 遷移矩陣：列 = 建基準的工況，欄 = 測試工況</h3>`
      + `<div class="mini">AUROC</div>` + heatmap(keys, keys, (a, b) => cell(a, b)?.auroc, {corner: "基準 → 測試"})
      + `<div class="mini" style="margin-top:8px">健康接受率（測試工況的健康樣本被判健康的比例）</div>`
      + heatmap(keys, keys, (a, b) => cell(a, b)?.healthy_accept, {corner: "基準 → 測試", fmt: v => pct(v, 0)})
      + `<h3>(b) 部署策略</h3>`
      + table([["k", "策略"], ["healthy_accept_mean", "平均健康接受率", v => pct(v)], ["fault_detect_mean", "平均故障偵測率", v => pct(v)]],
              Object.entries(sum).map(([k, v]) => ({k, ...v})))
      + `<h3>(c) 跨馬達健康分布：以 T1 為基準</h3><div class="mini">不同馬達個體；個體與老化影響無法分離，不能視為同一馬達的退化歷程。</div>`
      + table([["motor", "馬達"], ["rpm", "RPM"], ["n", "樣本數"], ["median_score", "分數中位數", v => num(v, 2)],
               ["p90_score", "分數 P90", v => num(v, 2)], ["flagged_unknown", "被判未知", v => pct(v, 1)]], r.drift.rows);
  },
  exp6(r){
    const sorted = [...r.summary].sort((a, b) => a.healthy_fp_mean - b.healthy_fp_mean);
    const names = [...new Set(r.rows.map(x => x.method))], ds = [...new Set(r.rows.map(x => x.dataset))];
    const fp = (d, m) => r.rows.find(x => x.dataset === d && x.method === m)?.healthy_fp;
    return `<div class="mini">${r.n_datasets} 組資料集 × ${names.length} 種偵測器，seed ${esc(r.seed)}，校準信心 ${esc(r.confidence)}</div>`
      + table([["method", "偵測器"], ["auroc_mean", "AUROC", (v, x) => pm(v, x.auroc_std, z => num(z, 4))],
               ["healthy_fp_mean", "健康誤報率", (v, x) => `${pct(v)} ± ${pct(x.healthy_fp_std)}`], ["healthy_fp_mean", "", v => bar(v, 1, "var(--red)")],
               ["fault_detect_mean", "故障偵測率", v => pct(v)], ["fpr_at_tpr95_mean", "FPR@TPR95", v => num(v, 4)]], sorted)
      + `<h3>逐資料集健康誤報率</h3>`
      + heatmap(ds, names, (d, m) => { const v = fp(d, m); return v === undefined ? null : 1 - v; }, {corner: "資料集", fmt: v => v === null ? "—" : pct(1 - v, 1)})
      + `<div class="mini">顏色越綠誤報越低。</div>`;
  },
  exp6_formal(r){
    const m = k => r.rows.reduce((s, x) => s + (x[k] ?? 0), 0) / r.rows.length;
    return kpis([["二元正確率（平均）", pct(m("open_set_accuracy"), 2)], ["健康接受率（平均）", pct(m("known_accuracy"), 1)],
                 ["AUROC（平均）", num(m("auroc"), 4)], ["unknown F1（平均）", num(m("unknown_f1"), 4)]])
      + `<div class="mini">方法 ${esc(r.method)}，seed ${esc(r.seed)}，${r.rows.length} 組工況；資料指紋 ${esc(String(r.dataset_fingerprint).slice(0, 12))}…；commit ${esc(String(r.commit_sha).slice(0, 7))}</div>`
      + table([["dataset", "工況"], ["n_train", "健康訓練", int], ["n_calibration", "健康校準", int], ["n_known_test", "健康測試", int], ["n_unknown_test", "未知測試", int],
               ["known_accuracy", "健康接受率", v => pct(v, 1)], ["open_set_accuracy", "二元正確率", v => pct(v, 2)],
               ["auroc", "AUROC", v => num(v, 4)], ["fpr_at_tpr95", "FPR@TPR95", v => num(v, 4)], ["unknown_f1", "unknown F1", v => num(v, 4)],
               ["known_score_median", "健康分數中位數", v => num(v, 2)], ["unknown_score_median", "未知分數中位數", v => num(v, 2)]], r.rows);
  },
  exp7(r){
    return table([["openset_method", "方法"], ["auroc", "AUROC", v => num(v, 4)], ["aupr_unknown_positive", "AUPR", v => num(v, 4)],
                  ["fpr_at_95_tpr", "FPR@TPR95", v => num(v, 4)], ["open_set_accuracy", "Open Set accuracy", v => pct(v, 2)],
                  ["known_class_accuracy", "健康接受率", v => pct(v, 1)], ["unknown_recall", "unknown recall", v => pct(v, 1)],
                  ["unknown_precision", "unknown precision", v => pct(v, 2)], ["f1_unknown", "unknown F1", v => num(v, 4)]], r.results)
      + `<div class="mini">公平性：seed ${esc(r.fairness.same_random_seed)}、${esc(r.fairness.same_features)}、閾值資料 ${esc(r.fairness.same_threshold_data)}</div>`
      + details("逐配置結果", table([["openset_method", "方法"], ["config", "配置"], ["kind", "類型"], ["n", "樣本數"],
               ["unknown_recall", "判未知比例", v => pct(v, 1)], ["median_score", "分數中位數", v => num(v, 2)]], r.details));
  },
  exp8(r){
    const m = k => r.rows.reduce((s, x) => s + (x[k] ?? 0), 0) / r.rows.length;
    return kpis([["健康差距 known−unknown（平均）", num(m("health_gap_known_minus_unknown"), 3)], ["Open Set accuracy（平均）", pct(m("open_set_accuracy"), 2)],
                 ["AUROC（平均）", num(m("auroc"), 4)], ["unknown 健康度中位數（平均）", num(m("unknown_health_median"), 3)]])
      + `<div class="mini">${esc(r.health_interpretation)}；方法 ${esc(r.method)}，seed ${esc(r.seed)}</div>`
      + table([["dataset", "工況"], ["n_train", "健康訓練", int], ["n_calibration", "健康校準", int], ["n_known_test", "健康測試", int], ["n_unknown_test", "未知測試", int], ["known_health_mean", "健康樣本相對指數", (v, x) => pm(v, x.known_health_std)],
               ["unknown_health_mean", "未知樣本健康度", (v, x) => pm(v, x.unknown_health_std)],
               ["health_gap_known_minus_unknown", "差距", v => num(v, 3)], ["health_gap_known_minus_unknown", "", v => bar(v)],
               ["known_healthy_fraction", "健康樣本判 healthy 段", v => pct(v, 0)], ["unknown_critical_fraction", "未知判 critical 段", v => pct(v, 0)],
               ["open_set_accuracy", "Open Set accuracy", v => pct(v, 2)], ["auroc", "AUROC", v => num(v, 4)]], r.rows);
  },
  exp8_monitor(r){
    const n = r.length, alarms = r.filter(x => x.alarm_state !== "normal").length;
    const stages = {}; r.forEach(x => stages[x.severity_stage] = (stages[x.severity_stage] || 0) + 1);
    return kpis([["窗口數", n], ["告警中的窗口", `${alarms} / ${n}`],
                 ...Object.entries(stages).map(([k, v]) => [`${k} 段`, v])])
      + `<canvas class="exp-line" height="220" data-series='${esc(JSON.stringify(r.map(x => [x.raw_health_index, x.smoothed_health_index, x.alarm_state])))}'></canvas>`
      + `<div class="mini"><span style="color:#94a3b8">●</span> raw 健康度 <span style="color:var(--blue)">━</span> 平滑健康度；底色 = 告警（黃 warning、紅 critical）；虛線 = warning 0.5 / critical 0.2</div>`
      + details("逐窗結果", table([["window_index", "#"], ["openset_score", "開集分數", v => num(v, 2)], ["health_index", "健康度", v => num(v, 3)],
               ["smoothed_health_index", "平滑", v => num(v, 3)], ["severity_stage", "分段"], ["trend", "趨勢"], ["alarm_state", "告警"],
               ["change_point_state", "變化點"], ["fault_type", "fault_type"]], r));
  },
};

const COMMITTED_RENDER = {
  exp6_formal_matrix(c){
    return `<div class="mini">${esc(c.producer)}；${esc(c.completed_runs)}/${esc(c.expected_runs)} 格完成，seed ${esc((c.seeds || []).join("/"))}，資料 ${esc(c.dataset_root)}</div>`
      + table([["method", "方法"], ["open_set_accuracy_mean", "Open Set accuracy", (v, x) => `${pct(v, 2)} ± ${pct(x.open_set_accuracy_std, 2)}`],
               ["known_accuracy_mean", "已知認出率", (v, x) => `${pct(v, 1)} ± ${pct(x.known_accuracy_std, 1)}`],
               ["auroc_mean", "AUROC", v => num(v, 4)], ["n_condition_seed_rows", "列數", int]], c.macro_summary)
      + details("k-NN − Mahalanobis 成對差", table([["metric", "指標"], ["n_pairs", "配對數"], ["mean_difference", "平均差", v => num(v, 6)],
               ["knn_better_pairs", "差值>0"], ["mahalanobis_better_pairs", "差值<0"], ["ties", "平手"]], c.paired_differences))
      + `<div class="mini">正差不一律較好：FPR 越低越好。封存差值不會因上方選項改變而重算。</div>`;
  },
  exp8_health_index_results(c){
    const rows = Object.entries(c.overall).map(([method, v]) => ({method, ...v}));
    const ds = [...new Set(c.by_condition.map(x => x.dataset))];
    const gap = (d, m) => c.by_condition.find(x => x.dataset === d && x.method === m)?.health_gap_known_minus_unknown_mean;
    return `<div class="mini">${esc(c.producer)}；${esc(c.rows)} 列，seed ${esc((c.seeds || []).join("/"))}</div>`
      + table([["method", "方法"], ["health_gap_known_minus_unknown", "健康差距", v => pm(v.mean, v.std)],
               ["open_set_accuracy", "Open Set accuracy", v => pm(v.mean, v.std, z => num(z, 6))], ["auroc", "AUROC", v => num(v.mean, 4)],
               ["unknown_recall", "unknown recall", v => num(v.mean, 4)]], rows)
      + `<h3>逐工況健康差距（3 seed 平均）</h3>` + heatmap(ds, ["mahalanobis", "knn"], gap, {corner: "工況", lo: 0.3, hi: 0.8});
  },
};

/* ---------- 線圖（實驗八逐窗）與 t-SNE 散佈（實驗十二） ---------- */
function drawLines(root){
  root.querySelectorAll("canvas.exp-line").forEach(cv => {
    const data = JSON.parse(cv.dataset.series), dpr = devicePixelRatio;
    const ctx = cv.getContext("2d"), W = cv.width = cv.clientWidth * dpr, H = cv.height = 220 * dpr;
    if (!W || !data.length) return;
    const pad = 30 * dpr, X = i => pad + (data.length === 1 ? 0 : i / (data.length - 1)) * (W - pad * 1.4);
    const Y = v => H - pad - v * (H - pad * 1.6);
    const step = (W - pad * 1.4) / Math.max(1, data.length - 1);
    data.forEach(([, , a], i) => {
      if (a === "normal") return;
      ctx.fillStyle = a === "critical" ? "#ef444433" : "#eab30833";
      ctx.fillRect(X(i) - step / 2, Y(1), step, Y(0) - Y(1));
    });
    ctx.font = `${11 * dpr}px monospace`; ctx.fillStyle = "#64748b";
    for (const g of [0, 0.2, 0.5, 1]){
      ctx.strokeStyle = g === 0.2 || g === 0.5 ? "#94a3b8" : "#1e293b";
      ctx.setLineDash(g === 0.2 || g === 0.5 ? [5 * dpr, 4 * dpr] : []);
      ctx.beginPath(); ctx.moveTo(pad, Y(g)); ctx.lineTo(W, Y(g)); ctx.stroke();
      ctx.fillText(String(g), 4 * dpr, Y(g) + 4 * dpr);
    }
    ctx.setLineDash([]);
    ctx.fillStyle = "#94a3b8";
    data.forEach(([raw], i) => { ctx.beginPath(); ctx.arc(X(i), Y(raw ?? 0), 2.2 * dpr, 0, 7); ctx.fill(); });
    ctx.strokeStyle = "#60a5fa"; ctx.lineWidth = 2 * dpr; ctx.beginPath();
    data.forEach(([, s], i) => i ? ctx.lineTo(X(i), Y(s ?? 0)) : ctx.moveTo(X(i), Y(s ?? 0)));
    ctx.stroke();
  });
  root.querySelectorAll("canvas.exp-tsne").forEach(cv => {
    const pts = JSON.parse(cv.dataset.points || "[]");
    const dpr = devicePixelRatio;
    const ctx = cv.getContext("2d"), W = cv.width = cv.clientWidth * dpr, H = cv.height = 220 * dpr;
    if (!W || !pts.length) return;
    const pad = 16 * dpr;
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const X = x => pad + (maxX === minX ? 0.5 : (x - minX) / (maxX - minX)) * (W - pad * 2);
    const Y = y => H - pad - (maxY === minY ? 0.5 : (y - minY) / (maxY - minY)) * (H - pad * 2);
    const configs = [...new Set(pts.map(p => p[2]))];
    const color = c => `hsl(${(configs.indexOf(c) * 360) / Math.max(1, configs.length)}, 70%, 55%)`;
    pts.forEach(([x, y, c, v]) => {
      ctx.fillStyle = ctx.strokeStyle = color(c);
      ctx.beginPath();
      if (v === "unknown"){ ctx.moveTo(X(x), Y(y) - 4 * dpr); ctx.lineTo(X(x) + 3.5 * dpr, Y(y) + 3 * dpr); ctx.lineTo(X(x) - 3.5 * dpr, Y(y) + 3 * dpr); ctx.fill(); }
      else if (v === "known_wrong"){ ctx.strokeRect(X(x) - 2.5 * dpr, Y(y) - 2.5 * dpr, 5 * dpr, 5 * dpr); }
      else { ctx.arc(X(x), Y(y), 2.4 * dpr, 0, 7); ctx.fill(); }
    });
  });
}

/* ---------- 邊跑邊畫 ---------- */
let ACTIVE = null;   // {id, es}

function canvasSetup(cv, hpx){
  const dpr = devicePixelRatio, W = cv.width = cv.clientWidth * dpr, H = cv.height = hpx * dpr;
  const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, W, H);
  ctx.font = `${11 * dpr}px monospace`;
  return {ctx, W, H, dpr};
}
// 分數軸用平方根刻度，與即時展示一致；虛線 = 未知判定線 1.0
function drawScoreStrip(cv, pts, opts = {}){
  const {ctx, W, H, dpr} = canvasSetup(cv, opts.h || 220);
  if (!W || !pts.length) return;
  const pad = 30 * dpr, n = opts.n || pts.length;
  const maxS = Math.max(4, ...pts.map(p => p.s));
  const X = i => pad + (n <= 1 ? 0 : i / (n - 1)) * (W - pad * 1.3);
  const Y = s => H - pad - Math.sqrt(Math.max(s, 0)) / Math.sqrt(maxS) * (H - pad * 1.6);
  ctx.fillStyle = "#64748b"; ctx.strokeStyle = "#1e293b"; ctx.lineWidth = dpr;
  for (const g of [1, 10, Math.round(maxS)]){ if (g > maxS) continue;
    ctx.beginPath(); ctx.moveTo(pad, Y(g)); ctx.lineTo(W, Y(g)); ctx.stroke(); ctx.fillText(String(g), 4 * dpr, Y(g) + 4 * dpr); }
  ctx.strokeStyle = "#94a3b8"; ctx.setLineDash([6 * dpr, 5 * dpr]);
  ctx.beginPath(); ctx.moveTo(pad, Y(1)); ctx.lineTo(W, Y(1)); ctx.stroke(); ctx.setLineDash([]);
  for (const m of opts.marks || []){
    ctx.strokeStyle = m.color || "#475569"; ctx.beginPath(); ctx.moveTo(X(m.i), pad * 0.3); ctx.lineTo(X(m.i), H - pad); ctx.stroke();
    if (m.label){ ctx.fillStyle = "#cbd5e1"; ctx.fillText(m.label, X(m.i) + 3 * dpr, pad * 0.3 + 10 * dpr); }
  }
  pts.forEach((p, i) => { ctx.fillStyle = p.color || (p.s > 1 ? "#ef4444" : "#22c55e");
    ctx.beginPath(); ctx.arc(X(i), Y(p.s), 2 * dpr, 0, 7); ctx.fill(); });
  if (opts.line){   // 第二軸 0～1（EWMA），警報線 0.5
    const Y2 = v => H - pad - v * (H - pad * 1.6);
    ctx.strokeStyle = "#eab30899"; ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.beginPath(); ctx.moveTo(pad, Y2(0.5)); ctx.lineTo(W, Y2(0.5)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = "#eab308"; ctx.lineWidth = 2 * dpr; ctx.beginPath();
    opts.line.forEach((v, i) => i ? ctx.lineTo(X(i), Y2(v)) : ctx.moveTo(X(i), Y2(v))); ctx.stroke();
  }
}
let rafPending = false;
function scheduleDraw(fn){ if (rafPending) return; rafPending = true; requestAnimationFrame(() => { rafPending = false; fn(); }); }

const LIVE = {
  exp1: {
    init: () => ({pts: [], marks: [], rows: []}),
    html: () => `<canvas class="live-canvas" height="220"></canvas>
      <div class="mini">每點一筆樣本的開集分數（綠 ≤ 1 判健康、紅 > 1 判未知）；直線分隔配置。先播健康 holdout，再逐一播九種故障。</div>
      <div class="live-rows" style="margin-top:8px"></div>`,
    on(st, f, el){
      if (f.event === "scores"){
        if (f.offset === 0) st.marks.push({i: st.pts.length, label: f.config});
        f.scores.forEach((s, i) => st.pts.push({s, color: f.unknown[i] ? "#ef4444" : "#22c55e"}));   // 判定旗標由實驗端給
        scheduleDraw(() => drawScoreStrip(el.querySelector(".live-canvas"), st.pts, {marks: st.marks, n: Math.max(st.pts.length, 200)}));
      } else if (f.event === "row"){
        st.rows.push(f.row); el.querySelector(".live-rows").innerHTML = exp1Table(st.rows);
      }
    },
  },
  exp2: {
    init: () => ({pts: [], marks: [], config: null}),
    html: () => `<canvas class="live-canvas" height="220"></canvas>
      <div class="mini">每點一筆自動注入的樣本（綠 = 判健康、藍 = 判已知故障、紅 = 判未知）。直線分隔配置。形成群就只用這一群樣本自學，不等操作員指出故障程度。</div>
      <div class="live-known mini" style="margin-top:6px"></div>
      <div class="live-q mini"></div>
      <div class="live-cand panel" hidden style="margin-top:8px"></div>`,
    on(st, f, el){
      const known = el.querySelector(".live-known");
      if (f.event === "fitted") known.textContent = "已知：" + (f.known || []).join("、");
      if (f.event === "tick"){
        if (st.config !== f.config){ st.config = f.config; st.marks.push({i: st.pts.length, label: f.display || f.config}); }
        const color = f.verdict === "healthy" ? "#22c55e" : f.verdict === "known" ? "#60a5fa" : "#ef4444";
        st.pts.push({s: f.score, color});
        el.querySelector(".live-q").textContent = `隔離區 ${f.quarantine} 筆`;
        scheduleDraw(() => drawScoreStrip(el.querySelector(".live-canvas"), st.pts, {marks: st.marks, n: Math.max(st.pts.length, 200)}));
      } else if (f.event === "candidate"){
        const c = f.candidate;
        const box = el.querySelector(".live-cand");
        box.hidden = false;
        box.innerHTML = `<h2 style="color:var(--yellow)">模型發現一群，正在用這群樣本自學</h2>
          <div class="mini">叢集 ${esc(c.size)} 筆　蒐集區間 ${esc((c.t_range || []).join("–"))}。類名由模型編，不向操作員詢問故障程度。</div>`;
      } else if (f.event === "adopted"){
        const box = el.querySelector(".live-cand");
        const outcome = f.outcome || {};
        box.hidden = false;
        const after = `事後對照多數為 ${esc(outcome.matched_display)}，純度 ${esc(Math.round((outcome.purity || 0) * 100))}%。`;
        if (outcome.action === "learned"){
          box.innerHTML = `<h2>已自學為 ${esc(outcome.display || "")}</h2>
            <div class="mini">只用這一群 ${esc(outcome.size)} 筆；隔離區另有 ${esc(outcome.pruned || 0)} 筆已被新模型接受而移出。${after}</div>`;
        } else if (outcome.action === "rejected_boundary"){
          box.innerHTML = `<h2>退回這一群</h2>
            <div class="mini">群的中位分數 ${esc((outcome.median_score || 0).toFixed(2))} 沒有超過已知校準最大值 ${esc((outcome.known_cal_max || 0).toFixed(2))}，與已知邊界分不開，不學成新類。${after}</div>`;
        } else {
          box.innerHTML = `<h2>群太小，不學成新類</h2><div class="mini">${after}</div>`;
        }
        known.textContent = "已知：" + (f.known || []).join("、");
      }
    },
  },
  exp3: {
    init: () => ({cur: {}, trials: []}),
    html: (params = {}) => {
      const keys = params.scenario ? [params.scenario] : ["A", "B"];
      const names = {A: "劇本 A：漸進磨損", B: "劇本 B：突發外力"};
      return keys.map(k => `<h3>${names[k]}（第 1 次重複逐筆播放）</h3><canvas class="live-${k}" height="180"></canvas>`).join("")
      + `<div class="mini">點 = 開集分數（紅 = 判未知、綠 = 判健康）；黃線 = EWMA 異常比例（右軸 0～1，虛線 0.5 觸發警報）；直線 = 故障開始、⚠ = 警報。其餘重複直接計算，只列結果。</div>
         <div class="live-trials" style="margin-top:8px"></div>`;
    },
    on(st, f, el){
      if (f.event === "tick"){
        const c = st.cur[f.scenario] = st.cur[f.scenario] || {pts: [], line: [], marks: [{i: f.onset, label: "故障開始", color: "#a78bfa"}]};
        c.pts.push({s: f.score, color: f.unknown ? "#ef4444" : "#22c55e"}); c.line.push(f.ewma);   // 判定旗標由實驗端給
        if (f.alarm_now) c.marks.push({i: c.pts.length - 1, label: "⚠ " + (f.kind === "gradual" ? "漸進" : "突發"), color: "#eab308"});
        const k = f.scenario;
        scheduleDraw(() => drawScoreStrip(el.querySelector(".live-" + k), c.pts, {h: 180, line: c.line, marks: c.marks, n: Math.max(c.pts.length, 160)}));
      } else if (f.event === "trial"){
        st.trials.push(f.trial);
        el.querySelector(".live-trials").innerHTML = table([["scenario", "劇本"], ["trial", "#"], ["expected", "應判"], ["verdict", "判為"],
          ["correct", "正確", bool], ["latency", "延遲", int], ["transition", "中間帶停留", int]], st.trials.slice(-12))
          + `<div class="mini">已完成 ${st.trials.length} 次，正確 ${st.trials.filter(x => x.correct).length} 次（表只列最近 12 次）</div>`;
      }
    },
  },
  exp4: {
    init: () => ({res: {}, status: {geometry: "等待", direction: "等待", severity: "等待"}}),
    html: () => `<div class="live-status mini"></div><div class="live-parts"></div>`,
    on(st, f, el){
      const label = {geometry: "(a) 射線結構", direction: "(b) 方向熟悉度", severity: "(c) 嚴重度回復"};
      if (f.event === "part_start") st.status[f.name] = "⚙ 擬合中…";
      if (f.event === "part"){ st.status[f.name] = "✔ 完成"; st.res[f.name] = f.data;
        el.querySelector(".live-parts").innerHTML = RENDER.exp4(st.res); }
      el.querySelector(".live-status").textContent = Object.entries(st.status).map(([k, v]) => `${label[k]}：${v}`).join("　");
    },
  },
  exp8_monitor: {
    init: () => ({rows: []}),
    html: () => `<canvas class="exp-line" height="220" data-series="[]"></canvas>
      <div class="mini"><span style="color:#94a3b8">●</span> raw 健康度 <span style="color:var(--blue)">━</span> 平滑健康度；底色 = 告警（黃 warning、紅 critical）；虛線 = 0.5 / 0.2</div>
      <div class="live-kv" style="margin-top:6px"></div>`,
    on(st, f, el){
      if (f.event !== "window") return;
      const w = f.window; st.rows.push(w);
      const cv = el.querySelector(".exp-line");
      cv.dataset.series = JSON.stringify(st.rows.map(x => [x.raw_health_index, x.smoothed_health_index, x.alarm_state]));
      scheduleDraw(() => drawLines(el));
      el.querySelector(".live-kv").innerHTML = kpis([["窗口", `#${w.window_index}`], ["開集分數", num(w.openset_score, 2)], ["健康度", num(w.health_index, 3)],
        ["平滑", num(w.smoothed_health_index, 3)], ["分段", esc(w.severity_stage)], ["趨勢", esc(w.trend)], ["告警", esc(w.alarm_state)]]);
    },
  },
};
RENDER.exp10 = function(r){
  const rows = ["direct", "adapted", "scratch"].flatMap(strategy =>
    (r.summary[strategy] || []).map(row => ({strategy, ...row})));
  return `<div class="mini">${esc(r.n_datasets)} 組工況的有向配對，seed ${esc(r.seed)}，方法 ${esc(r.method)}。direct 的 N=0 用完整目標健康池。</div>`
    + table([["strategy", "策略"], ["n", "目標健康筆數 N", int], ["auroc_mean", "AUROC", v => num(v, 4)],
             ["healthy_accept_mean", "健康接受率", v => pct(v)], ["fault_detect_mean", "故障偵測率", v => pct(v)]], rows)
    + details(`逐配對（${r.rows.length} 列）`, table([["src", "來源"], ["dst", "目標"], ["strategy", "策略"], ["n", "N", int],
             ["auroc", "AUROC", v => num(v, 4)], ["healthy_accept", "健康接受率", v => pct(v)],
             ["fault_detect", "故障偵測率", v => pct(v)]], r.rows));
};
RENDER.exp11 = function(r){
  const summary = Object.entries(r.summary).flatMap(([protocol, methods]) =>
    Object.entries(methods).map(([method, m]) => ({protocol, method, ...m})));
  const paired = r.wilcoxon ? Object.entries(r.wilcoxon).map(([k, v]) => ({k, ...v})) : [];
  return `<div class="mini">${esc(r.n_datasets)} 組工況，seed ${esc((r.seeds || []).join("、"))}。主指標 ${esc((r.primary_metrics || []).join("、"))}；網頁只跑這一個 seed。</div>`
    + table([["protocol", "協定"], ["method", "方法"], ["accuracy_mean", "Accuracy", v => num(v, 4)],
             ["balanced_accuracy_mean", "Balanced Accuracy", v => num(v, 4)], ["macro_f1_mean", "macro-F1", v => num(v, 4)],
             ["unknown_f1_mean", "unknown F1", v => num(v, 4)]], summary)
    + (paired.length ? `<h3>Ledoit–Wolf 減 legacy</h3>` + table([["k", "比較"], ["n_pairs", "配對數", int],
             ["mean_diff_pp", "macro-F1 差（百分點）", v => num(v, 2)], ["p_value", "p", v => num(v, 4)]], paired) : "")
    + details(`逐列（${r.rows.length}）`, table([["protocol", "協定"], ["dataset", "工況"], ["seed", "seed", int], ["method", "方法"],
             ["n", "列數", int], ["accuracy", "Accuracy", v => num(v, 4)], ["balanced_accuracy", "Balanced Accuracy", v => num(v, 4)],
             ["macro_f1", "macro-F1", v => num(v, 4)], ["unknown_f1", "unknown F1", v => num(v, 4)]], r.rows));
};
function countHeat(cm){
  const labels = cm.labels;
  const max = Math.max(1, ...cm.matrix.flat());
  return heatmap(labels, labels, (a, b) => cm.matrix[labels.indexOf(a)][labels.indexOf(b)],
    {lo: 0, hi: max, fmt: v => int(v), corner: "真值 → 預測"});
}
RENDER.exp12 = function(r){
  const scatters = Object.entries(r._tsne_by_dataset || {}).map(([key, d]) => {
    const pts = d.embedding.map((xy, i) => [Math.round(xy[0] * 10) / 10, Math.round(xy[1] * 10) / 10, d.config_labels[i], d.verdicts[i]]);
    return `<h3>${esc(key)}</h3><canvas class="exp-tsne" height="220" data-points="${esc(JSON.stringify(pts))}"></canvas>`;
  }).join("");
  return `<div class="mini">${esc(r.n_datasets)} 組工況，seed ${esc(r.seed)}。t-SNE 含擬合列，只供看鄰近，不是嚴重度。</div>`
    + `<h3>(a) 健康 vs 故障</h3>` + countHeat(r.binary_confusion)
    + `<h3>(b) 開集多類：Ledoit–Wolf</h3>` + countHeat(r.openset_confusion.ledoit_wolf)
    + `<h3>(b) 開集多類：legacy</h3>` + countHeat(r.openset_confusion.legacy)
    + `<h3>(c) t-SNE</h3><div class="mini">顏色 = 配置；圓點 = 已知且判對，叉 = 判成別的已知類，三角 = 判未知。</div>`
    + scatters;
};

function setScenarioButtons(disabled){
  for (const id of ["scenario-A", "scenario-B"]){ const b = $(id); if (b) b.disabled = disabled; }
}

function stopStream(){
  if (ACTIVE){ ACTIVE.es.close(); const b = $(`stop-${ACTIVE.id}`); if (b) b.hidden = true;
    const s = $(`stream-${ACTIVE.id}`); if (s) s.disabled = false; setScenarioButtons(false); ACTIVE = null; }
}

function streamExp(e, extra){
  stopStream();
  const params = collectParams(e);
  if (extra) Object.assign(params, extra);
  const rate = $(`rate-${e.id}`).value;
  const live = $(`live-${e.id}`), st = $(`status-${e.id}`), view = LIVE[e.id];
  live.innerHTML = view.html(params); $(`out-${e.id}`).innerHTML = "";
  const state = view.init();
  const url = `/api/experiments/${e.id}/stream?params=${encodeURIComponent(JSON.stringify(params))}&rate=${rate}`;
  const es = new EventSource(url);
  ACTIVE = {id: e.id, es};
  $(`stream-${e.id}`).disabled = true; $(`stop-${e.id}`).hidden = false; setScenarioButtons(true);
  st.className = "mini"; st.textContent = "⚙ 擬合模型中…";
  es.onmessage = msg => {
    if (!ACTIVE || ACTIVE.es !== es) return;
    const f = JSON.parse(msg.data);
    if (f.event === "fitted") st.textContent = "⏵ 播放中…";
    else if (f.event === "candidate") st.textContent = "⚙ 模型自學中…";
    else if (f.event === "adopted") st.textContent = "⏵ 播放中…";
    else if (f.event === "error"){ st.className = "mini err"; st.textContent = "✗ " + f.error; stopStream(); return; }
    else if (f.event === "done"){
      LAST[e.id] = f.payload; stopStream();
      st.textContent = `✔ 完成（${f.payload.seconds} 秒，含播放時間）`;
      if (e.id === "exp4") live.innerHTML = "";
      showResult(e, f.payload);
      return;
    }
    view.on(state, f, live);
  };
  es.onerror = () => { if (ACTIVE && ACTIVE.es === es){ st.className = "mini err"; st.textContent = "✗ 串流中斷"; stopStream(); } };
}

/* ---------- 頁面 ---------- */
function groups(){
  const g = [];
  for (const e of CATALOG.experiments){
    let grp = g.find(x => x.no === e.no);
    if (!grp){ grp = {no: e.no, key: e.id, items: []}; g.push(grp); }
    grp.items.push(e);
  }
  return g;
}

function fieldHtml(e, p){
  const id = `f-${e.id}-${p.name}`;
  if (p.type === "dataset"){
    const def = CATALOG.datasets.includes("T1/8000rpm") ? "T1/8000rpm" : CATALOG.datasets[0];
    return `<label>${esc(p.label)}<select id="${id}">${CATALOG.datasets.map(d => `<option${d === def ? " selected" : ""}>${esc(d)}</option>`).join("")}</select></label>`;
  }
  if (p.type === "config")
    return `<label>${esc(p.label)}<select id="${id}">${CATALOG.configs.map(c => `<option value="${esc(c.config)}"${c.config === p.default ? " selected" : ""}>${esc(c.display)}</option>`).join("")}</select></label>`;
  if (p.type === "select")
    return `<label>${esc(p.label)}<select id="${id}">${p.options.map(o => `<option${o === p.default ? " selected" : ""}>${esc(o)}</option>`).join("")}</select></label>`;
  return `<label>${esc(p.label)}<input type="number" id="${id}" value="${esc(p.default)}" min="${p.min}" max="${p.max}"></label>`;
}

function cardHtml(e){
  const noData = !CATALOG.datasets.length;
  const coverage = e.executable === false ? "本卡不掃描工況、不擬合。"
    : e.params.some(p => p.type === "dataset") ? "本卡只使用所選工況。"
    : `本卡處理掃描到的 ${CATALOG.datasets.length} 工況：${CATALOG.datasets.join("、")}。`;
  return `<div class="panel" id="card-${e.id}">
    <div class="exp-head"><h2>${esc(e.no)}${e.sub ? " " + esc(e.sub) : ""}：${esc(e.title)}</h2>
      <div class="q">${esc(e.question)}</div>
      <div class="mini">程式 <code>${esc(e.code)}</code>　技術報告 <code>${esc(e.doc)}</code></div>
      <div class="mini">資料角色：${esc(e.data_roles)} ${esc(coverage)}</div>
      <div class="mini">解讀限制：${esc(e.limitations)}</div>
      <div class="mini" style="margin-top:4px">${e.executable === false
        ? `⚙ ${esc(e.fits)}。請到實驗六執行八方法比較。`
        : `⚙ 每次按「執行」都會從頭擬合：${esc(e.fits)}。耗時隨方法、資料與硬體而變。
        模型只在這次執行的記憶體裡，跑完即丟，不保存；只存結果 JSON 到 <code>output/web_server/{ts}/experiments/</code>。同來源、設定與 seed 才可核對重現。`}</div></div>
    <div class="form">${e.executable === false ? "" : e.params.map(p => fieldHtml(e, p)).join("")}
      ${e.executable === false ? "" : `<button class="primary" id="run-${e.id}"${noData ? " disabled" : ""}>▶ 執行</button>`}
      ${e.id === "exp3" ? `<button type="button" id="scenario-A"${noData ? " disabled" : ""}>劇本 A：漸進磨損</button><button type="button" id="scenario-B"${noData ? " disabled" : ""}>劇本 B：突發外力</button>` : ""}
      ${e.stream ? `<label>播放速度<select id="rate-${e.id}"><option value="20">慢（20 筆/秒）</option><option value="80" selected>中（80 筆/秒）</option><option value="400">快（400 筆/秒）</option></select></label>
      <button id="stream-${e.id}"${noData ? " disabled" : ""}>⏵ 邊跑邊畫</button><button id="stop-${e.id}" class="warn" hidden>■ 停止</button>` : ""}
      <span class="mini" id="status-${e.id}"></span></div>
    ${e.stream ? `<div id="live-${e.id}" style="margin-top:10px"></div>` : ""}
    <div id="out-${e.id}" style="margin-top:10px"></div>
    ${noData ? '<div class="mini err">沒有可用工況。請管理者確認 --data-root 與正式特徵 CSV，重啟服務後重新整理；目前不能執行或播放。</div>' : ""}
  </div>
  ${e.committed ? `<div class="panel"><div class="exp-head"><h2>${esc(e.no)}${e.committed_sub ? " " + esc(e.committed_sub) : ""}：${esc(e.committed_title || "已提交的正式結果")}</h2>
      <div class="mini">程式 <code>${esc(e.committed_code || "")}</code>　技術報告 <code>${esc(e.committed_doc || "")}</code></div>
      <div class="mini" style="margin-top:4px">封存結果，唯讀：與上方選項及本次執行分開，讀現成檔案，不擬合、不從網頁重跑；來源日期／SHA以該包manifest為準，頁面沒有另行核實新資料資格。</div></div>
      <div id="committed-${e.committed}" class="mini" style="margin-top:10px">載入中…</div></div>` : ""}`;
}

function showResult(e, payload){
  const out = $(`out-${e.id}`);
  if (!out) return;
  const p = Object.entries(payload.params).map(([k, v]) => `${k}=${v}`).join("，");
  out.innerHTML = `<div class="mini" role="status" id="context-${e.id}">${esc(resultContext(e, payload))}</div>`
    + (window.AssistantPresentation ? "" : `<div class="mini">結果參數 ${esc(p)}；${esc(payload.seconds)} 秒；結果存於 <code>${esc(payload.saved || "（未存檔）")}</code></div>`)
    + RENDER[e.id](payload.result);
  const model = payload.result?.model;
  if (model?.classes) out.innerHTML += details("本次模型資料用途（逐類池為分母）",
    table([["config", "已註冊配置"], ["n_train", "訓練", int], ["n_cal", "校準", int], ["n_holdout", "holdout", int]], model.classes)
    + `<div class="mini">分數 ${esc(model.score_type)}；門檻來源 ${esc(model.threshold_strategy)}。列級切分不證明採集獨立。</div>`);
  if (window.AssistantPresentation) window.AssistantPresentation.runInfo(out, payload, e.id);
  drawLines(out);
}

function resultContext(e, payload){
  const current = collectParams(e), actual = payload.params;
  const same = Object.keys(current).every(k => current[k] === actual[k]);
  const selection = same ? "與目前選項一致；這是該次執行的結果。" : "選項已變更；下方仍是舊設定結果，尚未按新選項執行。";
  if (window.AssistantPresentation) return selection;
  const stamp = value => value ? String(value) + (/T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? "" : "（原檔未提供時區）") : "原檔未記錄";
  const conflict = payload.started_at && payload.started_at === payload.finished_at && payload.seconds > 1.01 ? "時間紀錄不一致，保留原值。" : "";
  const fallback = `開始時間：${stamp(payload.started_at)}；完成時間：${stamp(payload.finished_at)}；耗時：${Number.isFinite(payload.seconds) && payload.seconds >= 0 ? payload.seconds + " 秒" : "原檔未記錄或數值無效"}。${conflict}`;
  return `${payload.timing_text || fallback}${selection}`;
}

function refreshResultContext(e){
  const box = $(`context-${e.id}`);
  if (box && LAST[e.id]) box.textContent = resultContext(e, LAST[e.id]);
}

function collectParams(e){
  const params = {};
  for (const p of e.params){
    const v = $(`f-${e.id}-${p.name}`).value;
    params[p.name] = p.type === "int" ? Number(v) : v;
  }
  return params;
}

async function runExp(e){
  stopStream();
  const params = collectParams(e);
  const btn = $(`run-${e.id}`), st = $(`status-${e.id}`);
  btn.disabled = true; st.className = "mini"; st.textContent = "⚙ 執行中…（多工況的實驗約需數秒到十幾秒）";
  try {
    const res = await fetch(`/api/experiments/${e.id}/run`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(params)});
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || res.statusText);
    LAST[e.id] = payload;
    st.textContent = `✔ 完成（${payload.seconds} 秒）`;
    showResult(e, payload);
  } catch (err){
    st.className = "mini err"; st.textContent = "✗ " + err.message;
  } finally {
    btn.disabled = false;
  }
}

async function loadCommitted(name){
  const box = $(`committed-${name}`);
  if (!box) return;
  try {
    if (!COMMITTED[name]){
      const res = await fetch(`/api/committed/${name}`);
      COMMITTED[name] = await res.json();
    }
    const c = COMMITTED[name];
    box.className = "";
    box.innerHTML = c.available ? `<div class="mini">檔案 <code>${esc(c.path)}</code></div>` + COMMITTED_RENDER[name](c)
      : `<span class="mini">找不到 <code>${esc(c.path)}</code></span>`;
  } catch (err){
    box.className = "mini err"; box.textContent = "✗ " + err.message;
  }
}

function renderPage(key){
  const grp = groups().find(g => g.key === key);
  if (!grp) return false;
  $("expRoot").innerHTML = grp.items.map(cardHtml).join("");
  stopStream();
  for (const e of grp.items){
    if (e.executable === false) continue;
    $(`run-${e.id}`).onclick = () => runExp(e);
    if (e.stream){ $(`stream-${e.id}`).onclick = () => streamExp(e); $(`stop-${e.id}`).onclick = () => {
      stopStream(); $(`status-${e.id}`).textContent = "■ 已停止；未收到完成結果不算完整執行，待伺服器釋放鎖再重試。"; }; }
    if (e.id === "exp3"){
      $("scenario-A").onclick = () => streamExp(e, {scenario: "A"});
      $("scenario-B").onclick = () => streamExp(e, {scenario: "B"});
    }
    for (const p of e.params){
      $(`f-${e.id}-${p.name}`).oninput = () => refreshResultContext(e);
      $(`f-${e.id}-${p.name}`).onchange = () => refreshResultContext(e);
    }
    if (LAST[e.id]) showResult(e, LAST[e.id]);
    if (e.committed) loadCommitted(e.committed);
  }
  return true;
}

function route(){
  const key = location.hash.slice(1) || "live";
  const isExp = CATALOG && key !== "live" && renderPage(key);
  $("page-live").hidden = !!isExp;
  $("page-exp").hidden = !isExp;
  document.querySelectorAll("#tabs a").forEach(a => a.classList.toggle("active", a.dataset.page === (isExp ? key : "live")));
  window.dispatchEvent(new Event("resize"));   // 依可見頁面重畫即時展示的 canvas（主圖或小窗）
}

async function init(){
  try {
    CATALOG = await (await fetch("/api/experiments")).json();
  } catch (err){
    console.error("無法載入實驗目錄", err);
    return;
  }
  $("tabs").insertAdjacentHTML("beforeend", groups().map(g =>
    `<a href="#${g.key}" data-page="${g.key}" title="${esc(g.items.map(i => i.title).join("／"))}">${esc(g.no)}</a>`).join(""));
  window.addEventListener("hashchange", route);
  window.addEventListener("resize", () => { if (!$("page-exp").hidden) drawLines($("expRoot")); });
  route();
}
init();
})();
