// 真正前端程式的可控 DOM／WebSocket 測試；不連線、不擬合模型。
const vm=require('node:vm'), fs=require('node:fs'), assert=require('node:assert/strict');
class Element {
  constructor(){this.children=[];this.options=[];this._value='';this.disabled=false;this.textContent='';this.classList={toggle(){}};}
  get value(){return this._value;}
  set value(v){this._value=v;}
  replaceChildren(...rows){this.children=rows;this.options=rows;this._value=rows[0]?.value||'';}
  prepend(row){this.children.unshift(row);this.options=this.children;}
  append(row){this.children.push(row);}
  getContext(){return new Proxy({}, {get:()=>()=>{},set:()=>true});}
}
const nodes=new Map(), sockets=[];
const document={getElementById(id){if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);},
  createElement(){return new Element();},querySelectorAll(){return [];}};
class Socket {constructor(){this.readyState=1;sockets.push(this);}send(){} }
const context=vm.createContext({document,WebSocket:Socket,location:{protocol:'http:',host:'localhost'},
  fetch:()=>new Promise(()=>{}),setTimeout(){},confirm:()=>true});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const el=id=>document.getElementById(id), message=(socket,data)=>socket.onmessage({data:JSON.stringify(data)});
const initial={type:'state',session_id:1,epoch:1,t:24,seed:42,rate:4,step:'暫停',
  meta:{motor:'T1',rpm:'8000rpm'},datasets:[{motor:'T1',rpm:'8000rpm'}],
  openset:{openset_method:'mahalanobis',method:'ledoit_wolf'},known:['8screws'],
  configs:[{id:'S00',name:'8screws',n:50},{id:'S03',name:'匿名來源S03',n:60}],source:'S03'};
message(sockets[0],initial);
assert.equal(el('source').value,'S03');
assert.equal(el('dataset').value,'T1|8000rpm');
assert.match(el('sourceStatus').textContent,/已套用來源：S03/);
// 尚未套用的選項保留，但不冒稱實際來源。
el('source').value='S00';el('source').onchange();
message(sockets[0],{...initial,t:25});
assert.equal(el('source').value,'S00');assert.match(el('sourceStatus').textContent,/待套用：S00/);
// 同一 session 重連先收到空選單，再收到完整 snapshot。
vm.runInContext('connect()',context);
message(sockets[1],{type:'state',datasets:[],t:0});
message(sockets[1],{...initial,t:26});assert.equal(el('source').value,'S03');
message(sockets[0],{...initial,source:'S00',t:100});assert.equal(el('source').value,'S03');
message(sockets[1],{...initial,source:'S00',t:10});assert.equal(el('source').value,'S03');
message(sockets[1],{...initial,source:'S00',t:27});assert.equal(el('source').value,'S00');
message(sockets[1],{...initial,source:'S99',t:28});
assert.equal(el('source').value,'');assert.equal(el('inject').disabled,true);
assert.match(el('source').options[0].textContent,/不在可選清單/);
const metrics=n=>({type:'metrics',session_id:1,epoch:1,t:n,alarms:[],rows:[{source:{name:'匿名來源S03'},
  pre_streamed:n,pre_rate:100,post_streamed:0,post_rate:null}]});
message(sockets[1],{...initial,t:29,running:false,metrics:metrics(29)});
assert.match(el('metrics').children[0].textContent,/n=29/);
message(sockets[1],metrics(15));
assert.match(el('metrics').children[0].textContent,/n=29/);
message(sockets[0],metrics(100));
assert.match(el('metrics').children[0].textContent,/n=29/);
console.log('來源恢復、待套用選項、過期連線與不可用來源：通過');
message(sockets[1],{type:'state',datasets:[],catalog_status:'data_root_missing'});
assert.equal(el('inspect').disabled,true);assert.equal(el('build').disabled,true);
assert.match(el('nextStep').textContent,/資料目錄不可用/);
assert.match(el('nextStep').textContent,/管理者/);
message(sockets[1],{type:'state',datasets:[],catalog_status:'no_datasets'});
assert.match(el('nextStep').textContent,/沒有可用資料集/);
message(sockets[1],{type:'state',datasets:initial.datasets,error:'資料讀取失敗'});
assert.equal(el('inspect').disabled,false);assert.equal(el('build').disabled,true);
assert.match(el('nextStep').textContent,/確認所選資料.*重試/);
message(sockets[1],{...initial,t:0,session_id:2});
assert.equal(el('start').disabled,false);assert.match(el('nextStep').textContent,/已處理 0 筆/);
message(sockets[1],{type:'state',datasets:initial.datasets,busy:true});
assert.equal(el('inspect').disabled,true);assert.match(el('nextStep').textContent,/請等待/);
