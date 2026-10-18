# 本輪交付紀錄

## 啟動與範圍

2026-10-08，Asia/Taipei。實際工作樹 `D:/schoolshit/專題/src/lineage_integration_20261008`，origin為正式Lineage。乾淨的4f783c6建立 `delivery/issues-30-19-37-29-20261008`；原Lineage checkout仍dda8910且clean。最新遠端main4f783c676849a27185cd28d2410b4e76639b7357、研究b3d68f、PR36 head544d4ed皆與提示詞相同。已完整讀取main AGENT與delivery、exp24入口、health_and_reports，沒有巢狀指引。

本輪依序處理#30、#19、#37、#29。只建立交付PR，不merge、不關issue、不發布issue留言／checkbox、不改Albert source branch。health_monitor、web/server.py、index.html、experiments.js不修改。所有open issues查詢時均無assignee，不能據作者推斷未推送工作。

AGENT寫作文章https://www.bnext.com.tw/article/90761/how-to-fix-ai-writing-style本輪瀏覽回non-retryable，未取得全文；遵循指引列出的五項要求。

## 事前影響與驗證

| 階段 | 修改範圍 | 驗證 |
|---|---|---|
| #30 | reports/Andy_20261008_議題交付、health_and_reports第2.2、docs索引；tests/issue_delivery_evidence.py只作來源工具 | 固定83846b資料SHA、17項差異、現main函數行、相對檔案連結、現行完整main測試、資料清冊 |
| #19 | 引用核對筆記、README引用表及明確主張、exp3文件註記 | 出版社／作者／合法原文，存在與內容及實作判定分開，不重訓 |
| #37 | exp24入口／CLI／交付文件與QA產物，不改重疊Web檔案 | 瀏覽器實際操作、兩detectors各自跨入口配對、終止flush與SHA |
| #29 | 先文件，再core/monitor.py及專屬fixture tests | prefit四入口、兩方法postfit與固定4f783c6配對、失敗fit不誤標完成、完整suite |

logs/output由core.logger.setup_run；手寫文件只放docs。效能差異本輪是工程等價，不聲稱準確率提高；正式105維唯讀，來源獨立性UNKNOWN、fresh INCOMPLETE。

## #30 完成的本輪驗證

提交6863cda42280156bbe33689e2c889ad22062d220，push成功，遠端同SHA。下一部分#19未涉及模型修改。

- 本輪重新測main：21-38-17，Python3.10.19，111 passed、0 failed/errors/skipped，pip check與四CLI通過。21-38-02另76舊基準測試通過，不把兩者相加成187個不同測試。
- 唯讀證據工具21-41-10：16檔來源SHA／AST行號、5份83846b來源SHA、6個工況60份資料清冊、65相對檔案目標全存在。資料fingerprint82534111c7901bfe0709637f4979976ce73445a54353e7c377e71e40b4cf2abd不變。
- 17項都有目前狀態、來源、限制與issue／去重草稿；health_and_reports第2.2已在交付分支補連結。PERF未profiling，root escape未驗證。僅文件收尾，不替代原稽核反例、不發布issues或勾選。

## #19 完成的本輪核對

提交c003e63a2506c5d89f991c33158a80f55d9558a9，push及遠端同SHA。

9條新／繼承來源逐條對照；另列Ye & Xie與原始論文原稿參考表缺口，不重建117條目錄。Roberts原文局部、Cover原文局部與Wang候選原文局部可讀；Page／Gebraeel全文未取得，其他只取得作者或出版社摘要者明列限制。README消除完整池重擬合防遺忘保證及三馬達run-to-failure錯誤。21-49-46連結工具69相對目標存在、0失效，資料指紋不變；git diff --check通過。沒有重訓或效能改動，issue草稿尚未發布。

## #37 本輪驗收

提交e5a09d87789dbbfcc2e254051dd707bb9eebc7d0，push成功、遠端SHA一致。

先在exp24手冊固定ledger語意，再補既有regression工具。21-52-04 Maha／kNN各595筆，0差異，1190筆列ID候選與特徵SHA都保存。新增兩測試；21-55-00第一輪112 passed／1 failed，原因是測試綁定draw太早，21-56-34修正後113 passed／0 failed/errors/skipped，pip check／四CLI通過。兩輪保存，不抹除失敗。

computer-use瀏覽器實際驗收含三入口、兩模式、六工況選單、T1/8000基準、匿名候選、確認兩known、重連、重設及A/B。重設工具逾時，之後log與重新開頁證明完成epoch2；引擎完整版本UNKNOWN。Maha B160筆，sudden t48／transition6／延遲8；kNN A260筆，gradual t100／transition33／延遲60。只作功能檢查，不作兩方法勝負。

關閉驗收clients後Ctrl+C停止兩個本輪服務，KeyboardInterrupt exit1為人工終止，不當成測試失敗；finally關檔。21-59-04封存核對3CSV：Maha epoch1=423、epoch2=160；kNN epoch1=260，各1…末筆連續、無尾筆缺失，19檔SHA保存。21-59-05相對檔案70目標0失效、資料SHA不變，diff check通過。原首頁整合仍待協調PR36，沒有修改重疊Web／健康串流；#37草稿尚未發布。

## #29 與2026-10-09接續

先完成monitor_guard文件、紅燈測試22-01-46，再改core/monitor.py。紅燈11個methods含14 failure entries／18 error entries（subTest），不是32個方法。最小guard在constructor／fit_initial／_refit追蹤完整成功擬合，score/classify/project/summary一致拒絕未擬合，其他公開讀取介面不放寬或改義。22-02-10專屬11項通過，8組固定舊版條件、各16列score/PCA差0，classify/summary/splits相同。

上次完整測試的提權審查因額度失敗，動作沒有執行；沒有繞過。2026-10-09使用者要求接續後，重新正常審查並執行：03-11-15全套124 passed／0 failed/errors/skipped，Python3.10.19，pip check／exp1、exp4、compare_openset、web.server四CLI通過。未重建或聲稱驗證第二Python環境。

重新fetch確認main4f783c6、交付e5a09d8、PR36 head544d4ed均未變，公開PR36不含core/monitor.py；原Lineage checkout仍clean。再次閱讀AGENT，只有根指引；寫作文章再次non-retryable，依列出的五項要求，不聲稱讀全文。未重訓研究模型、未動正式資料或Albert重疊檔案。

03-13-18再跑真實T1/8000導航配對：兩方法各595筆、0差異，整份JSON與guard修改前21-52-04的SHA完全相同（dbae28737e288cbf46d4dedd8d352dfce884b82a28ff4daa4a9f36b94139e8e0）。03-13-37文件檢查73相對目標0失效，資料清冊SHA不變，diff check通過。所有執行輸出依setup_run保存。

#29提交071d80498701ab524e181bf2f9fab28f4d4c54a6，push與遠端SHA一致。stage後diff check另發現紅燈tests.txt的一行尾端空白；PowerShell未因非零檢查自動中止後續commit，已在交付整理刪掉該空白並重新檢查整份PR差異。測試結果／trace未變，原始空白版本仍可從071d804追溯；後續提交命令加入失敗即停止檢查。

03-16-10最終交付文件加入後，80個相對檔案目標0失效、六工況60CSV指紋不變。相對檔案存在不代表全歷史錨點驗證。與main基線相比，mahalanobis/openset/geometry/data及server/index/experiments.js/health_monitor皆無差異。完整PR的git diff --check已通過，新增交付文件掃描未見列出的簡體字。PR只供審閱，不合併／發布issue留言／勾選或關閉。

## GitHub交付

最後整理提交db1915a3d90fecc13c952f6e9d6e5d9c5cdfe7fa已push、遠端SHA相同。已建立並附加到本聊天的[PR #40](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/pull/40)，base main、head本輪交付分支、state OPEN、mergeCommit null。最終唯讀核對main仍4f783c6、PR36仍OPEN且head544d4ed；#19／#29／#30／#37皆OPEN，沒有發布issue留言或勾選。本段後續提交只補PR連結，不改程式；該提交的SHA以Git歷史與遠端核對為準。

## 未擬合防護的原驗收邊界

當次規範在程式修改前固定：synthetic fixture seed7，模型 seeds42／123；與 4f783c676849a27185cd28d2410b4e76639b7357 的 monitor 比較健康初始與新增已確認配置兩階段，atol=rtol=1e-12。11 個 test methods 的首次紅燈含 14 個 failure entries／18 個 error entries，不能加成 32 個獨立測試；8 組配對的 score／PCA 最大差 0。原紅燈、綠燈與逐樣本來源仍在本紀錄連結的 output，沒有覆寫；這些是工程回歸，不是準確率提高。完整事前文字見 [固定原規範](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/issue_delivery_20261008/monitor_guard.md)，現行 API 操作統一見 [模型生命週期](../../docs/experiments/README.md#共用模型生命週期)。
