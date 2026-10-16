# 實驗八彙總重算：驗證與交付紀錄

## 本輪邊界

2026-10-09，checkout `D:\schoolshit\專題\src\lineage_integration_20261008`，分支 `delivery/exp8-aggregate-20261009`，從 main `4f783c676849a27185cd28d2410b4e76639b7357` 開始。開始時乾淨；先前 PR #40 分支保留在 `5a54cf1e500ddb8c8f1f031ce3cd0e0282429467`。已完整讀 AGENT.md，目錄內沒有其他 AGENT/AGENTS 指引。

重新查 GitHub：main 未變；PR #40、#36 都 OPEN、未合併。PR #36 head `544d4ed8c6516622e2f46c095351f9483a61635c`。15 個開放 issue 無 assignee；#25 只有 Albert 的改名留言，公開 PR 未見彙總實作。無法知道其他人尚未推送的工作。

不改 PR #40 防護，不改 Albert 同檔，不建立新 issue，不合併 PR、不直接推 main。#9、#13～16、#19、#24～30、#35、#37 保持原狀。關單需完整驗收與 main 整合，本輪尚未滿足。

## 分階段紀錄

| 階段 | 實際內容 | commit／推送 |
|---|---|---|
| 手冊 | 先固定 16 檔 SHA、六 run／54 列、等權重、ddof=1、容差與 UNKNOWN 邊界 | `aeab6eb48c27d1c63ef5fb3260632a204c8b1c3f`；已 push，ls-remote 相符 |
| 程式與 fixtures | 獨立彙總入口、來源與 CSV 檢查、失敗稽核、37 項 fixtures；無 fit/predict | `7b59381fef93272511bc547f8099815ae05a307d`；已 push，ls-remote 相符 |
| 來源換行補充 | 核對 16 個固定 main Git blob，補列 LF SHA；保留原 CRLF SHA，不改公式或容差；新增一項 fixture | `da0dc9415d53a99fb73c51db31829b58471ea751`；已 push，ls-remote 相符 |
| 實際重算與交付 | 兩次各六 run／54 列；349 欄中 347 吻合、2 不同；三個結果檔兩次 SHA 完全相同 | `24b6c51d937bd636cde2f39c98a71b409852375f`；已 push，遠端相符；PR #41 已建立 |

## 已執行的測試

既有環境 `D:\schoolshit\專題\src\tmp\lineage_integration310\Scripts\python.exe`，Python 3.10.19；沒有重建 venv。只驗證目前 pyproject 宣告的版本，未宣稱其他 Python 相容。

- main 程式基線：111 項、0 failure、0 error、0 skipped；在手冊 commit 執行，當時未新增程式。證據見 [基線摘要](../../reports/Andy_20261009_實驗八基線測試/test_summary.json) 與 `logs/exp8_aggregate_baseline/2026-10-09-03-28-48.log`。摘要由當次 `python -c` 呼叫 `unittest`、`setup_run('exp8_aggregate_baseline')` 後以 `write_text` 保存；該臨時命令未作為專案模組入庫，故摘要移至 reports，原始位元組 SHA256 為 `0c659a93d320342669783f40770e85ca2d43b2136cfdc127fbba14e32582d04f`。不是重新測試或以新 writer 補造歷史來源。
- 初次 fixture 執行在受限 Windows 暫存目錄遭 ACL 拒絕：27 項 setup error，並非成功測試。相同程式改用正常權限後，27 項通過；後續增補至 37 項。
- 新程式提交前相關測試：56 項通過；完整測試：148 項通過。pip check 都通過。保存 `output/exp8_aggregate_validation_related/2026-10-09-03-33-51/`、`output/exp8_aggregate_validation_full/2026-10-09-03-33-55/` 與對應 logs。這兩次 summary 的 HEAD 是手冊提交，程式仍在 working tree；正式重算前將再核對提交版程式與回歸。
- 完整測試內 generator.close 失敗、無效 CLI 參數等訊息是既有刻意失敗 fixtures；測試總結果仍為 OK，原訊息保留在 check_0.txt。

重現指令（專案根目錄）：

```powershell
python -m tests.exp8_aggregate_validation --scope related
python -m tests.exp8_aggregate_validation --scope full
python -m experiments.health_index_aggregate
```

完整測試需要允許本機 HTTP/WebSocket；Windows fixture 暫存也須有正常 ACL。遇到拒絕應使用正常權限流程，不改測試標準。

## 第一次封存重算

程式 HEAD `7b59381fef93272511bc547f8099815ae05a307d`，原契約 SHA `3805b5d00e0b86bf75c844378911956570df908f2f99a70fe25ce2a5bbad2f7e`。6 run、54 列、18 組工況摘要完整；16 檔 SHA 前後不變。逐欄比對 349 項：347 項通過容差，2 項超出。CLI exit=3 表示比較 DIFFERENT，非來源驗證失敗；wrapper 的 throw 保留，未抹去。

| 不一致欄位（k-NN 減 Mahalanobis） | 封存 | 本輪事前公式 | 差值 | 容差 |
|---|---:|---:|---:|---:|
| health_gap_known_minus_unknown | 0.018513 | 0.018512 | -0.000001 | 0.0000005 |
| known_health_mean | 0.018513 | 0.018512 | -0.000001 | 0.0000005 |

全部 global 平均與 ddof=1 標準差、逐工況 CSV 都在事前容差內。只讀診斷：k-NN 原列平均 `0.6016565185185185`、Mahalanobis `0.5831444444444445`，先相減得到 `0.01851207407407407`，四捨五入為 `0.018512`。如果先各自四捨五入，`0.601657−0.583144=0.018513`，可重現舊差值。這支持「四捨五入順序不同」的解釋；原生成程式缺失，仍不能確證歷史操作。未改本輪事前公式、未放大容差、未覆寫舊結果。兩欄尚有定義差異，#25 保持 OPEN。

精度不同不代表模型表現改變。本輪沒有 fit、predict 或讀正式 105 維 CSV；準確率變動 0。舊二元 open-set accuracy 的等權均值仍是 Mahalanobis 0.998849、k-NN 0.998842，AUROC／未知召回仍是 1.0；這些是封存數字重算，不是新實驗。

## 來源換行與後續驗證

發現 Windows checkout SHA 與 Git LF blob SHA 不同。例如 seed_42/knn.json：checkout `80d8759f…`、main blob `a3bed57d…`。16 檔逐一核對，內容只差 LF/CRLF。契約補列固定 main 的 16 個 LF SHA，程式接受原 SHA 或這份明列的 SHA，不使用模糊正規化。新增 fixture 確認已登錄換行可接受、額外空白仍被拒絕。`reports/` 完全未修改。

補充後相關 57 項、完整 149 項與 pip check 通過：`output/exp8_aggregate_validation_related/2026-10-09-03-36-59/`、`output/exp8_aggregate_validation_full/2026-10-09-03-37-03/`。summary 保存 HEAD、當時程式與測試 SHA；此次補充尚未提交，HEAD 仍為 `7b59381…`。提交後再保存正式重算與回歸證據。

### 最終提交版驗證

使用 `da0dc9415d53a99fb73c51db31829b58471ea751` 再跑封存重算與完整回歸；契約 SHA `c58e7f8d5d97f5af8d935a71704f399162c2a6b742ea9a9c69f0d22add54be29`。完整測試 149 項、0 failures/errors/skipped，pip check 通過，證據 `output/exp8_aggregate_validation_full/2026-10-09-03-38-55/{validation_summary.json,check_0.txt,check_1.txt}`。這 149 項含 38 項本輪 fixture，不引用其他 PR 的 124 項作本輪成績。

最後重算仍是 DIFFERENT、CLI exit=3；來源六個 JSON 的 schema 全為 1、每個九列、各自保存當時 commit/Python，完整清冊在 `output/exp8_health_index_aggregate/2026-10-09-03-38-52/source_inventory.json`。程式未 import benchmark、detector、資料載入或 sklearn；沒有訓練或故障資料選參。

| 兩次結果相同的檔案 | 實際檔案 SHA256 |
|---|---|
| aggregate_summary.json | `deeee196d4766f29da3783ceb1bd821d6e1a2364727c5bd8fed3a2272a7e30db` |
| aggregate_by_condition.csv | `e9895dcfc3bdbcd27699bf3ef636428083d3dbb2e657f8ce997f88ee40a2ce1b` |
| field_differences.csv | `d022661538c5c27c808e802f84eb19421841ac7a8f601863eaeab164f3579844` |

實際 Python 3.10.19；numpy 2.2.6、pandas 2.3.3、scikit-learn 1.7.2、scipy 1.15.3、matplotlib 3.10.9、hdbscan 0.8.44、loguru 0.7.3、tornado 6.5.10。彙總計算本身使用標準函式庫與既有 logger；列出其他套件是記錄完整執行環境，不代表都用於彙總。

## 驗收判定與其餘 issue

| 項目 | 判定 | 證據／缺口 |
|---|---|---|
| 程式已寫、來源完整、可重算 | VERIFIED | 六 run、54 列、16 檔 SHA；run/main、logger、嚴格失敗與 38 fixtures |
| 本輪指標、權重、NaN、ddof、容差 | VERIFIED | 手冊先提交；數值差異未回填改規則 |
| 與歷史全部欄位相符 | FAILED | 349 項有兩個差值欄超出容差；CSV 逐工況數字吻合 |
| 歷史差值的原始生成操作 | UNKNOWN | 先四捨五入可解釋舊值，沒有原程式確證 |
| 未知配置逐名／窗口／採集來源 | UNKNOWN | 封存沒保存；不填造 |
| main 整合、#25 全部驗收 | INCOMPLETE | 需 PR 審閱與整合；歷史差值規則仍須確認，不擅自合併或關單 |

本輪實際關閉 **0 個 issue**。回讀 15 個開放 issue 與 PR #40/#36 的本文、留言、檔案後，未發現新 #25 公開實作；main、PR heads 均未變。#29 等 PR #40 整合；#30 的新追蹤議題仍需另授權發布。#19 原文引用、#13～16 研究分支完整驗收、#26 版本／鎖版及 #27 CI 本輪沒有實作。#24/#28/#35/#37 涉及既有重疊或主首頁整合，保留不修改。

本輪只改彙總入口、fixtures、驗證工具、exp8 文件索引與 logs/output。正式 105 維 CSV、Mahalanobis/LW 預設、k-NN factory、PolarMap、health_monitor.py、web/server.py、index.html、experiments.js 及歷史 reports 的 diff 為零。沒有新的 reliable fault-type 或 fresh final 主張。

## 路徑與交付

- repo 根：`D:\schoolshit\專題\src\lineage_integration_20261008`。
- 最後重算：`D:\schoolshit\專題\src\lineage_integration_20261008\output\exp8_health_index_aggregate\2026-10-09-03-38-52`。
- 最後完整測試：`D:\schoolshit\專題\src\lineage_integration_20261008\output\exp8_aggregate_validation_full\2026-10-09-03-38-55`。
- 機器索引：[exp8_health_index_aggregate_delivery.json](exp8_health_index_aggregate_delivery.json)。完整來源 SHA 在契約與 source_inventory，不縮寫作驗證。
- 所有小型結果與 logs 納入本輪分支；不提交 raw、formal data、大型來源 ZIP 或其他工作樹修改。
- D 槽補充備份：`D:\schoolshit\專題\src\lineage_exp8_aggregate_artifacts\2026-10-09\exp8_recompute_20261009_0343.zip`，12 個成員，CRC 與逐檔位元組比對通過；整包 SHA256 `11f43991eaa330eebb4d53bcf8142afa4b5fe4d98377bb171a0a3f91cb13d9ca`。只含最終重算、完整回歸輸出、契約與三個新程式／測試，不含正式資料或舊大型 ZIP。

[PR #41](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/pull/41) 已建立並附加到目前聊天。回讀為 OPEN／MERGEABLE、mergedAt=null，base 仍為 `4f783c676849a27185cd28d2410b4e76639b7357`。#25 回讀仍 OPEN；本輪沒有修改任何 issue 狀態、checkbox 或發布新 issue。PR #40 的 head 仍為 `5a54cf1…`，Albert PR #36 head 仍為 `544d4ed…`。

本段只補記 PR 與交付索引，不改已驗證程式或公式。提交後再 push 並以 ls-remote 與 PR head 核對；main 尚未合併，本輪不是「驗收全部通過」。各階段提交均有 Codex Co-Authors。

Co-authored-by: Codex <noreply@openai.com>

## 未確認事項

原彙總生成程式沒有提交，ddof 與精度順序的歷史證據仍 UNKNOWN。九個未知配置逐名清單與逐窗來源不在六份 JSON 裡；本輪驗證工況、配置數、方法與封存指紋，不能填造採集來源。

給老師的本輪說明放在這份文件；七問與研究背景仍見 [exp24 導覽](../navigation/exp24_README.md)。本輪驗收只處理封存重算，不能把 exp8 的接近 100% 二元結果當成可靠 fault-type、RUL 或 fresh final。

## P0-B：更新至 PR #40 合併後的 main

以下為 2026-10-09 後續交付，不改寫上方歷史狀態。使用者已澄清 #25 驗收包含「有差異就回報」，不要求零差異；工程完成不等於原始公式已證實。

PR #40 已合併至 `c68b7cdbb5bddf1034a9e713c8e8efd5acc61c87`。PR #41 以非破壞性 merge 更新，commit 為 `471c04b0c3e0098d8da9a3b796094726f43003be`；`docs/README.md` 與 `docs/health_and_reports.md` 自動合併，逐項檢查保留兩側索引及 §2.2。正式資料、封存來源與組員檔案未修改。

受限權限下的相關測試有 Windows 暫存 ACL 錯誤，保存 [失敗紀錄](../../output/exp8_aggregate_validation_related/2026-10-09-11-29-22/check_0.txt)，不算通過。正常權限重跑，Python 3.10.19：相關 57 項與 pip check 通過，完整 **162 項**、0 failure/error/skipped 與 pip check 通過，見 [相關驗證](../../output/exp8_aggregate_validation_related/2026-10-09-11-29-45/validation_summary.json)、[完整驗證](../../output/exp8_aggregate_validation_full/2026-10-09-11-29-50/validation_summary.json)。

兩次可分辨的重算目錄為 [11-29-59](../../output/exp8_health_index_aggregate/2026-10-09-11-29-59/recomputation_audit.json) 與 [11-30-37](../../output/exp8_health_index_aggregate/2026-10-09-11-30-37/recomputation_audit.json)。最初緊接的兩次 CLI 落在同一秒，既有 logger 使用秒級目錄，因此另跑一次並核對兩個不同目錄；未將同一目錄當兩份獨立產物。

每次六 run、54 列、18 工況摘要，349 比對中 347 吻合、2 差異；三個結果 SHA 與上方封存表完全一致。兩欄仍為 `0.018513 → 0.018512`，歷史公式 UNKNOWN、CLI exit=3。契約內容未改；checkout 經 Git 轉為 CRLF，位元組 SHA 為 `dc0c2a54482356a5ad0cea267bb00d71d2f9c4b82a3effadc45bdea267ee1b1f`，不冒用先前 LF SHA。16 份來源前後 SHA 不變，`git diff -- reports/exp8_health_index_results` 為空。

本次審閱支持工程交付；需 PR #41 合併及真正 main 再測後才能關閉 #25。沒有 fit、新準確率、fresh final 或新可靠性證明。

另以 `tests.integration_evidence --phase candidate` 保存 [162 項完整測試及四個 CLI](../../output/branch_integration/2026-10-09-11-31-10/validation.json)，所有檢查成功。此產物記錄受測 commit `471c04b...`，不把之後純文件提交當成已重跑模型。
