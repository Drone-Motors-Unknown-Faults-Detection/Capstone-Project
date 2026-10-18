# #30／#19／#37／#29 統一交付

交付日期2026-10-09（Asia/Taipei），執行始於2026-10-08。工作樹`D:/schoolshit/專題/src/lineage_integration_20261008`；獨立分支`delivery/issues-30-19-37-29-20261008`，基線main `4f783c676849a27185cd28d2410b4e76639b7357`。本文是該次交付的歷史快照；當時文件位於docs，證據現集中於本reports目錄。當時未合併、未關issue，不代表目前PR狀態。

已建立[交付PR #40](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/pull/40)，本報告記錄其合併前驗收，不作目前狀態判定。

## 四部分與逐階段推送

| 部分 | 實際完成／限制 | 提交（已push並核對遠端相同SHA） |
|---|---|---|
| #30 | 17項差異表、五份固定來源、R1–R10去重草稿；health_and_reports §2.2更新；發布草稿仍待使用者確認 | 6863cda42280156bbe33689e2c889ad22062d220 |
| #19 | exp24繼承引用與9條核心來源、README過度主張修正；Page／Gebraeel全文、Wang舊指涉、Ye裸引與原始論文參考表尚未核實，不關閉 | c003e63a2506c5d89f991c33158a80f55d9558a9 |
| #37 | 瀏覽器功能檢查、逐列ledger、Maha／kNN各595筆配對；原首頁直接入口未完成、引擎完整版本UNKNOWN，不關閉 | e5a09d87789dbbfcc2e254051dd707bb9eebc7d0 |
| #29 | 四推論／摘要入口完整擬合guard、11項專屬測試、失敗refit拒絕混用舊模型；固定配對輸出不變，待PR合併及使用者授權才更新issue | 071d80498701ab524e181bf2f9fab28f4d4c54a6 |

各階段指令、失敗與修正均在[執行紀錄](execution_log.md)。最後整理提交只補交付索引與紅燈文字的一個尾端空白，沒有改研究數值。

## 驗證與效能差異

Python **3.10.19**；完整主線 **124 passed／0 failed／0 errors／0 skipped**，pip check成功，exp1／exp4／compare_openset／web.server四CLI `--help`成功。包含openset／PolarMap／HTTP／WebSocket回歸。未在本輪重建或驗證第二Python環境。

```powershell
python -m tests.monitor_guard_evidence
python -m tests.integration_evidence --phase candidate --data-root D:/schoolshit/專題/src/Lineage/data
python -m experiments.navigation_regression --data-root D:/schoolshit/專題/src/Lineage/data --seed 42
python -m tests.issue_delivery_evidence --data-root D:/schoolshit/專題/src/Lineage/data
```

固定舊版monitor比較2方法×2seeds×2階段、各16列fixture；score／PCA最大差0，classify／summary／splits完全相同。改guard後真實T1/8000再做1190筆配對，整份JSON SHA與改前完全一致。**沒有準確率提高的主張，也沒有profiling成績**。

資料唯讀：T1/T3×三RPM六工況60CSV，fingerprint `82534111c7901bfe0709637f4979976ce73445a54353e7c377e71e40b4cf2abd`不變；不是歷史90CSV全量重跑。未知原始視窗與session來源留UNKNOWN，fresh final仍INCOMPLETE。

## 證據位置

下面均相對本repo；本機絕對根為`D:/schoolshit/專題/src/lineage_integration_20261008`。

- [當次稽核與分組](#當次17項稽核與分組)、[引用判定與剩餘清單](citation_followup.md)、[導覽QA](guide_acceptance.md)、[未擬合規範](../../docs/experiments/README.md#共用模型生命週期)。
- 最後完整測試：`output/branch_integration/2026-10-09-03-11-15/{validation.json,tests.txt}`、對應`logs/branch_integration/2026-10-09-03-11-15.log`。
- guard紅／綠燈：`output/monitor_guard_evidence/2026-10-08-22-01-46`、`22-02-10`；原紅燈有subTest failures/errors，保留不改成PASS。
- 導覽修改前後1190筆ledger：`output/navigation_regression/2026-10-08-21-52-04/summary.json`及`2026-10-09-03-13-18/summary.json`；SHA `dbae28737e288cbf46d4dedd8d352dfce884b82a28ff4daa4a9f36b94139e8e0`。
- 外部瀏覽器截圖已移至 [驗收附件](browser_screenshots)；模型與fit仍在`output/web_guide/2026-10-08-21-50-53`、`21-55-14`。3CSV尾筆423／160／260與19檔SHA的歷史核對：`output/guide_delivery_evidence/2026-10-08-21-59-04/evidence.json`（保留當時路徑，搬移後對照由`tests.output_inventory_evidence`產生）。
- 來源與相對目標：`output/issue_delivery_evidence/2026-10-09-03-13-37/evidence.json`，73個檔案目標0失效；不等同全部歷史遠端連結／Markdown錨點皆通過。

## 排除的重疊與後續

PR36仍open，head544d4ed；沒有修改`web/server.py`、`web/static/index.html`、`experiments.js`、`experiments/health_monitor.py`與健康串流。#35保留；#24／#28與公開PR同檔，等後續協調。未知未推送工作與實際所有權不猜測。

正式105維、Mahalanobis／LW／q95預設、kNN factory／k5、PolarMap幾何與原切分均未改。#30文件與#29工程驗收可交付審閱；#19／#37仍有列出的缺口。所有issue草稿先留文件，待使用者確認才發布。

下一個可獨立安排的小任務是#25：先固定已存健康評估資料、指紋與指標定義，再重算aggregate並核對逐run一致；本輪只建議，不宣稱已接手，也不與R1的formal exp6完整性問題混同。#26環境決策另排，#27依賴該決策。

給老師的入口：[七問導覽](../../docs/navigation/exp13_README.md)，原始逐題回覆仍在[固定研究提交](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/b3d68f145cfa187e208e38cae74bf33f68d7c367/reports/teacher_reply_20261005/reply.md)。本輪是工程交付，不代表已取得可靠跨馬達fault-type模型或RUL。

Co-authored-by: Codex <noreply@openai.com>

## 當次17項稽核與分組

原反例來自44d98da／80bdf54，2026-10-06重驗基線64cb71d；本次對照固定4f783c6的Git blob及AST行號，未重跑所有舊fixture。PERF未profiling，SEC-003未證明root外寫入。原來源與五份查核附件見[固定稽核全文](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/issue_delivery_20261008/audit_followup.md)。

| ID | 舊64cb結論／證據 | 本輪4f783c6結論／檔案行號 | 限制／追蹤 |
|---|---|---|---|
| EXP-002 | summary缺CSV/log仍完整 | 仍成立；exp6_matrix.py:77 `_is_complete`，blob未變 | 原fixture未重跑；R1草案 |
| DATA-001 | healthy-only且任意105欄被載入 | 仍成立；core/data.py:54、68，blob未變 | 欄語意／exact labels仍未檢；R2 |
| DATA-002 | min(widths)截短且未記原寬度 | 仍成立；core/formal_data.py:314 `_convert_condition`，未變 | 不推論raw同步；R3 |
| DATA-003 | 無manifest同size/mtime不同bytes同指紋 | 仍成立；formal_benchmark.py:68，未變 | 原反例，不宣稱本輪正式資料遭竄改；R4 |
| DATA-004 | manifest保存absolute path | 仍成立；core/formal_data.py:433，未變 | portable／private分離未實作；R4 |
| SEC-002 | client取得exc repr | 仍成立且SSE也傳exc；web/server.py:177、241、279、324 | 本輪唯讀核新增SSE路徑，無外部攻擊；#28/R6 |
| SEC-003 | '..'接受，離開RPM目錄 | 仍成立，root escape未驗證；formal_data.py:314，未變 | 舊stub未越root，不寫完全安全或已證root escape；R3 |
| REPRO-001 | exp1–3未統一指紋與環境 | 仍成立；exp1:146、exp2:283、exp3:172 | exp1/3行移動，新增iter_run未統一metadata；R4 |
| REPRO-002 | formal結果缺packages/hardware | 仍成立；formal_benchmark.py:147，未變 | integration驗收記packages不等於所有研究runner補齊；R4 |
| ARCH-001 | parser與schema並行、無canonical config | 仍成立；runner.py:18、formal_benchmark.py:300，未變 | 維護議題，不推數值失效；R7 |
| ARCH-002 | 多責任大函數 | 結構仍成立；formal_benchmark.py:147、matrix.py:110、web/live.py:197，未變 | 未重構、未profiling；R10 |
| TEST-002 | catalog已測、WS/loader/輸出不足 | 部分再改善；test_navigation_guide.py、test_stream_integration.py、test_integration_navigation.py | 新增匿名/狀態/同算/中斷釋鎖測試；主WS Origin、正式loader仍缺專項，非全修；#27/R5 |
| AGG-001 | tampered manifest fingerprint被發布 | 仍成立；aggregate_exp6.py:56、103，未變 | 原fixture未重跑；R1，不和health彙總#25混同 |
| PERF-001 | CSV load無指紋cache | 結構未變；data.py:68、runner.py:51 | 未profiling，不能稱主要瓶頸；R8 |
| PERF-002 | 逐類kneighbors | 結構未變；openset.py:108、149 | 不沿用舊秒數當本輪量測；R8 |
| DOC-001 | index已修但歷史內部連結限制 | 現行選定索引目標重新驗證；docs/README.md:1及本輪新增文件 | 相對檔案存在不保證所有歷史錨點或遠端網址；R9 |
| DX-001 | 主流程POSIX，局部PS | 部分改善但仍成立；build_uv.sh:1、run_web.sh:1；exp24有PS入口 | 獨立3.10.19 Windows驗收不是所有install路徑；#26/R9 |

上表省略路徑前綴時，expN／matrix／aggregate均在experiments/；所有行號固定4f783c6。來源工具保存每檔SHA與完整函數跨度。原62（2026-10-03）、80等數字保留日期脈絡，不改寫成新測試數。歷史重驗工具24項與現行main suite是不同範圍。


以下是當時R1–R10去重邊界；已發布對應及其完成限制見[後續發布證據](../Andy_20261009_主線交付/README.md#已發布追蹤與去重補充)，不把原草稿當現行待辦。

| 分組 | 處理方式 | 最小重現／驗收 |
|---|---|---|
| R1 EXP-002/AGG-001 | 一份新草案，不重開#25 | 只有合法summary但缺CSV/log，及manifest指紋不符；完整性拒絕、合法數值不變、舊output不覆寫 |
| R2 DATA-001 | 新草案 | healthy-only/任意105欄fixture；先定exact schema/labels/finite/nonempty與compatibility，不重清資料 |
| R3 DATA-002/SEC-003 | 一份新草案 | 通道2/3窗、config '..'；明確通道截斷政策與resolved containment，drive/UNC/absolute測試只寫temp，root escape目前未驗證 |
| R4 DATA-003/004/REPRO | 一份metadata草案，引用#26 | 同size/mtime異bytes需異SHA；portable摘要/private sidecar、commit/packages/OS版本，排除raw寫入 |
| R5 TEST-002 | 補既有#27，不重建CI issue | loader/主WS/exp1–3規範；先保留新增導覽與SSE測試，不重寫 |
| R6 SEC-002 | 補既有#28，與PR36同server檔保留 | client不回path/trace，server有correlation；本輪不修 |
| R7 ARCH-001 | 新草案 | flags到canonical config到schema一致；fitted guard由#29處理 |
| R8 PERF | 一份量測草案 | 固定資料/seed量load walltime/memory及classes/batch/reference；先量再決定cache/vectorization |
| R9 DOC/DX | 文件入口本輪補，環境部分引用#26 | clean Windows與POSIX smoke、歷史鏈；不刪venv、不把局部驗收當全平台 |
| R10 ARCH-002 | 新草案 | 先fixture規範，再逐helper；metrics/event順序不變，排除Albert重疊檔案 |


#13–16需核對研究分支既有成果及各自驗收；main缺檔不代表全專案未做。#24／#28／#35的當時公開同檔重疊保留，未知未推送工作不推斷。原始資料獨立性及fresh資格沒有因文件交付改變。
