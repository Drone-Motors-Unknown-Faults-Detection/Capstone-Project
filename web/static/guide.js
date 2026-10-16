"use strict";
const el = id => document.getElementById(id);
let state = {}, flow = "cold", ws, research = false, samples = [], lastVersion = "";
let pendingSource = null;
let lastMetrics = {};
const names = {cold:"冷啟動：只有健康基準",learning:"未知學習：候選需要確認",trend:"趨勢：比較兩種 CSV 劇本"};
function send(cmd, extra={}) {if(ws && ws.readyState===1) ws.send(JSON.stringify({cmd,...extra}));}
function selectedData(){const v=el("dataset").value.split("|");return {motor:v[0],rpm:v[1]};}
function eventText(text){const li=document.createElement("li");li.textContent=text;el("events").prepend(li);if(el("events").children.length>80)el("events").lastChild.remove();}
function options(select, rows, value, label) {
  const old=select.value;
  select.replaceChildren(...rows.map(r=>{const o=document.createElement("option");o.value=value(r);o.textContent=label(r);return o;}));
  if([...select.options].some(o=>o.value===old))select.value=old;
}
function sourceOptions(restore){
  const rows=state.configs||[], active=state.source;
  if(restore)pendingSource=null;
  options(el("source"),rows,r=>r.id,r=>`${r.name}（${r.n}筆）`);
  const wanted=pendingSource??active;
  if(rows.some(r=>r.id===wanted))el("source").value=wanted;
  else {
    const unavailable=document.createElement("option");
    unavailable.value="";unavailable.textContent=`目前來源 ${active} 不在可選清單；請選可用來源`;
    unavailable.disabled=true;el("source").prepend(unavailable);el("source").value="";
    pendingSource=null;
  }
  const method=state.openset.openset_method==="knn"?`k-NN（k=${state.openset.knn_neighbors}）`:`Mahalanobis（${state.openset.method}）`;
  el("sourceStatus").textContent=`已套用來源：${active}；方法 ${method}。`+(pendingSource&&pendingSource!==active?`待套用：${pendingSource}，須按「輸入所選來源」。`:"選單與目前來源一致；選其他來源後須明確套用。");
  el("inject").disabled ||= !el("source").value;
}
function older(m, reference){
  if(m.session_id===undefined||m.epoch===undefined||!reference.meta)return false;
  return m.session_id<reference.session_id || (m.session_id===reference.session_id &&
    (m.epoch<reference.epoch || (m.epoch===reference.epoch && m.t<reference.t)));
}
function nextStep(ready,busy,online,matches,selected){
 if(!online)return "連線尚未完成；頁面會自動重連。持續失敗時請確認伺服器，再重新整理頁面。";
 if(busy)return "正在檢查／建立／更新，請等待完成。";
 if(!state.datasets?.length)return state.catalog_status==="data_root_missing"?
   "資料目錄不可用。請管理者確認 --data-root 指向現有正式 CSV 目錄並重啟伺服器，再重新整理頁面；目前不能建立基準。":
   "沒有可用資料集。請管理者確認 --data-root 內有論文管線產出的正式 clean 特徵 CSV 並重啟伺服器，再重新整理頁面；目前不能建立基準。";
 if(state.error)return ready?"操作失敗：查看操作紀錄；可明確重設，或重新確認所選資料後重建。資料／權限問題需管理者處理。":
   "資料載入／檢查失敗：查看操作紀錄，可按「確認所選資料」重試或改選可用工況；資料／權限問題需管理者處理。";
 if(!selected.motor||!selected.rpm)return "請先選擇清單中的馬達／轉速，再按「確認所選資料」。";
 const checked=state.preview?.motor===selected.motor&&state.preview?.rpm===selected.rpm;
 if(!ready||!matches)return checked?"資料已檢查；下一步可按「建立健康基準」。換工況需明確重建。":
   "下一步：按「確認所選資料」；檢查成功後才可建立健康基準。";
 if(flow==="trend")return state.running?"劇本演算中；可暫停，不在比較途中確認／重訓。":"下一步：記錄警報 → 重設健康基準 → 選劇本A或B。候選只在未知學習流程確認。";
 if(state.candidate)return "下一步：模擬操作員確認，或暫停等待。";
 return state.running?"逐筆監測中；可暫停，或選匿名訊號來源。":state.t===0?
   "基準已建立，已處理 0 筆；可按「開始／繼續」重播，尚無偵測統計。":
   "下一步：開始／繼續；觀看未知學習時可輸入匿名來源。";
}
function render(){
  const ready=!!state.openset, busy=state.busy, online=ws && ws.readyState===1;
  const selected=selectedData(), matches=ready && selected.motor===state.meta.motor && selected.rpm===state.meta.rpm;
  el("step").textContent=flow==="trend"&&state.scenario_completed?"劇本播畢（候選尚未確認）":state.step || "待建基準";
  el("data").textContent=ready?`${state.meta.motor}/${state.meta.rpm} · 105維 · seed ${state.seed} · session ${state.session_id}/epoch ${state.epoch} · ${state.t}筆 · 實際來源 ${state.source}`:state.datasets?.length?"尚無模型；可先檢查所選資料":"尚無模型；沒有可用資料，請管理者確認正式 CSV";
  el("evidence").textContent=state.evidence || "raw/session UNKNOWN；fresh INCOMPLETE";
  const p=state.preview;
  el("preview").textContent=p?`${p.motor}/${p.rpm} · ${p.feature_dim}維 · seed ${p.seed} · CSV有限值來源：${p.configs.map(r=>r.id+"："+r.n+"筆").join("，")}`:state.datasets?.length?"先確認所選資料；未擬合":"沒有可檢查資料；等待管理者確認資料目錄";
  el("inspect").disabled=!online||busy||!state.datasets?.length;
  el("dataset").disabled=busy;
  el("build").disabled=!online||busy||!p||p.motor!==selected.motor||p.rpm!==selected.rpm;
  for(const id of ["start","pause","reset","inject","source","scenarioA","scenarioB","rate"])el(id).disabled=!online||busy||!matches;
  el("confirm").disabled=!online||busy||!matches||!state.candidate||!!state.error;
  el("start").disabled ||= !!state.error;
  el("nextStep").textContent=nextStep(ready,busy,online,matches,selected);
  if(state.rate)el("rate").value=String(state.rate);
  el("quarantine").textContent=!ready?"尚無基準，尚未開始隔離／分群。":`隔離 ${state.quarantine||0} 筆；分群連續失敗 ${state.attempts||0} 次（成功歸零，非總嘗試）；`+(state.candidate?`候選 ${state.candidate.id}：${state.candidate.size} 筆，重播索引 ${JSON.stringify(state.candidate.t_range)}`:"沒有候選；可持續等待或暫停");
  if(ready)sourceOptions(false);
  else {el("source").replaceChildren();el("sourceStatus").textContent="尚無已套用來源；建立健康基準後才可選擇。";}
  el("technical").textContent=JSON.stringify({data:state.data,preview:state.preview,configs:state.configs,fit:state.fit_audit,persistence:state.persistence,active_source:state.source},null,2);
  const version=`${state.session_id}/${state.epoch}/${state.known?.length||0}`;
  if(version!==lastVersion){samples=[];lastVersion=version;el("judgment").textContent="尚未計算";el("metrics").replaceChildren();el("alarm").textContent="趨勢警報：尚未計算";}
  draw();
}
function draw(){
  const c=el("chart"),g=c.getContext("2d"),trend=flow==="trend",max=trend?1:3;
  g.clearRect(0,0,c.width,c.height);g.font="15px sans-serif";g.fillStyle="#274e60";
  g.fillText(trend?"EWMA（異常比例平滑）":"normalized score（畫面上限3）",14,22);
  const line=trend ? .5 : 1, y=v=>220-Math.min(max,Math.max(0,v))/max*175;
  g.strokeStyle="#9b7222";g.setLineDash([6,6]);g.beginPath();g.moveTo(45,y(line));g.lineTo(945,y(line));g.stroke();g.setLineDash([]);
  g.fillText(String(line),10,y(line)+5);g.strokeStyle="#17607c";g.lineWidth=2;g.beginPath();
  const rows=samples.slice(-180);rows.forEach((s,i)=>{const x=45+i*900/Math.max(179,rows.length-1),v=trend?s.ewma:s.score;i?g.lineTo(x,y(v)):g.moveTo(x,y(v));});g.stroke();
  el("chartCaption").textContent=trend?"EWMA警報線 .5；X軸是重播筆數，不是物理時間。":"正規化未知分數；虛線1是拒絕線。圖截頂不改原分數判定。";
}
function mode(){research=!research;el("research").hidden=!research;el("mode").textContent=research?"返回展示模式":"切換研究模式";el("mode").setAttribute("aria-pressed",String(research));}
function showMetrics(m){
 if(older(m,state)||older(m,lastMetrics))return;
 lastMetrics={...m,meta:true};
 el("metrics").replaceChildren(...m.rows.map(r=>{const p=document.createElement("p");p.textContent=`${r.source.name}：學會前拒絕為未知 ${r.pre_rate===null?"未計算":r.pre_rate+"%"}（n=${r.pre_streamed}）；已知後接受為已知 ${r.post_rate===null?"未計算":r.post_rate+"%"}（n=${r.post_streamed}）。接受率不等自身分類準確率。`;return p;}));
 el("alarm").textContent=m.alarms.length?"趨勢警報："+m.alarms.map(a=>`${a.kind}，t=${a.t}，中間帶${a.transition}筆／延遲${a.latency??"未計算"}筆`).join("；"):"趨勢警報：目前沒有";
}
function connect(){
 const socket=new WebSocket((location.protocol==="https:"?"wss://":"ws://")+location.host+"/ws");
 ws=socket;let firstSnapshot=true;
 socket.onopen=()=>{if(ws!==socket)return;el("connection").textContent="已連線 · 逐筆呼叫既有核心";render();};
 socket.onclose=()=>{if(ws!==socket)return;el("connection").textContent="已斷線，最後一個瀏覽器離線會暫停；重連不重訓";render();setTimeout(()=>{if(ws===socket)connect();},1500);};
 socket.onmessage=e=>{
  if(ws!==socket)return;
  const m=JSON.parse(e.data);
  if(m.type==="state"){
   if(!firstSnapshot&&older(m,state))return;
   const restore=firstSnapshot||m.session_id!==state.session_id||m.epoch!==state.epoch||m.source!==state.source;
   if(firstSnapshot)lastMetrics={};
   state=m;
   options(el("dataset"),m.datasets||[],r=>r.motor+"|"+r.rpm,r=>r.motor+" / "+r.rpm);
   if(restore&&m.meta)el("dataset").value=m.meta.motor+"|"+m.meta.rpm;
   if(restore)pendingSource=null;
   firstSnapshot=false;
   render();
   if(m.metrics)showMetrics(m.metrics);
  }else if(m.type==="sample"){
   if(older(m,state))return;
   samples.push(m);if(samples.length>400)samples.shift();el("judgment").textContent=`${m.prediction}；score ${m.score}，threshold 1`;draw();
  }else if(m.type==="event"){eventText(m.text);}
  else if(m.type==="metrics")showMetrics(m);
 };
}
document.querySelectorAll("[data-flow]").forEach(b=>b.onclick=()=>{flow=b.dataset.flow;el("flowTitle").textContent=names[flow];el("trend").hidden=flow!=="trend";el("learning").hidden=flow==="trend";document.querySelectorAll("[data-flow]").forEach(x=>x.classList.toggle("active",x===b));draw();});
el("mode").onclick=mode;
el("inspect").onclick=()=>send("inspect",selectedData());
el("build").onclick=()=>{if(state.openset&&!confirm("將明確重建所選工況的基準，停止原監測；舊紀錄保留。要繼續嗎？"))return;send("build",selectedData());};
el("dataset").onchange=render;
el("source").onchange=()=>{pendingSource=el("source").value;render();};
el("start").onclick=()=>send("start");
el("pause").onclick=()=>send("pause");
el("reset").onclick=()=>{if(confirm("重新擬合健康基準，清除本session隔離區；舊紀錄保留。要重設嗎？"))send("reset");};
el("rate").onchange=()=>send("rate",{value:Number(el("rate").value)});
el("inject").onclick=()=>send("source",{id:el("source").value});
el("confirm").onclick=()=>send("confirm",{candidate_id:state.candidate?.id});
el("scenarioA").onclick=()=>send("scenario",{name:"A"});
el("scenarioB").onclick=()=>send("scenario",{name:"B"});
fetch("/docs/exp24_實驗總覽.md").then(r=>r.text()).then(text=>{
 const rows=text.split("\n").filter(x=>x.startsWith("|")).filter((_,i)=>i!==1),table=document.createElement("table");
 rows.forEach((line,i)=>{const tr=document.createElement("tr");line.replace(/^\||\|$/g,"").split("|").forEach(cell=>{const td=document.createElement(i?"td":"th");td.textContent=cell.trim();tr.append(td);});table.append(tr);});el("overview").append(table);
}).catch(()=>el("overview").textContent="總覽載入失敗；請使用 Markdown 來源連結");
connect();
