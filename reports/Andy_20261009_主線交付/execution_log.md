# 2026-10-09 主線整合與關單紀錄

## P0-A：PR #40 合併前審閱

本輪從 `delivery/issues-30-19-37-29-20261008` 的 `5a54cf1e500ddb8c8f1f031ce3cd0e0282429467` 重跑。遠端 main 為 `4f783c676849a27185cd28d2410b4e76639b7357`；PR #36 仍有衝突，head 為 `544d4ed8c6516622e2f46c095351f9483a61635c`，本輪不修改其待決檔案。

已審閱 `core/monitor.py`、guard 測試、導覽逐筆索引與相關文件。`_require_fitted()` 保護 score、classify、project、summary；重新擬合開始即清除完成旗標，失敗時不沿用舊模型。未發現需先阻擋 PR #40 合併的問題。

Python 為 3.10.19，使用既有 `D:/schoolshit/專題/src/tmp/lineage_integration310/Scripts/python.exe`，沒有重建或刪除環境。資料根目錄為 `D:/schoolshit/專題/src/Lineage/data`，此次導覽覆蓋 T1/8000rpm；來源清冊列出目前 checkout 的 60 份 CSV，不宣稱已驗證另一份 90 CSV 研究資料。

| 指令（`python -m`） | 本輪結果 | 產物 |
|---|---|---|
| `tests.integration_evidence --phase candidate --data-root D:/schoolshit/專題/src/Lineage/data` | 124 passed，0 failed/error/skipped；pip check 與四個 CLI help 成功 | [validation.json](../../output/branch_integration/2026-10-09-04-08-52/validation.json)、[tests.txt](../../output/branch_integration/2026-10-09-04-08-52/tests.txt) |
| `tests.monitor_guard_evidence` | 11 項通過；兩方法 × 兩 seed × 兩階段共 8 組；score 與 PCA 最大差值均為 0，分類、summary、split 相同 | [evidence.json](../../output/monitor_guard_evidence/2026-10-09-04-09-05/evidence.json) |
| `experiments.navigation_regression --data-root D:/schoolshit/專題/src/Lineage/data --seed 42` | Mahalanobis/k-NN 各 595 筆，0 mismatch，來源 SHA 不變 | [summary.json](../../output/navigation_regression/2026-10-09-04-09-06/summary.json) |
| `tests.issue_delivery_evidence --data-root D:/schoolshit/專題/src/Lineage/data` | 16 份來源、6 工況、80 個文件目標核對成功 | [evidence.json](../../output/issue_delivery_evidence/2026-10-09-04-09-10/evidence.json) |
| `tests.guide_delivery_evidence` | 歷史 1,190 筆配對、CSV 與文件 SHA 核對成功；未重新進行瀏覽器驗收 | [evidence.json](../../output/guide_delivery_evidence/2026-10-09-04-09-11/evidence.json) |

`integration_evidence` 的 `main_before` 欄位是工具歷史常數 `64cb71d...`，不能作為本輪 base；本輪實際 base 已列於本頁及 GitHub PR。測試紀錄中的 `head` 是實際受測程式 commit。測試資料採 fixture；真實 CSV 導覽重播另列，兩者不混成研究效能結果。

AGENT.md 指定的《數位時代》寫作文章本輪無法取得全文；依 AGENT.md 明列的五項規則撰寫。正式預設、Ledoit–Wolf、k-NN factory 與 PolarMap 未變。這些驗證支持工程行為一致，不支持 fresh final、來源獨立性或準確率提高。

合併後須在實際 main commit 再測，才關閉 #29；#19、#30、#37 仍有各自缺口。後續紀錄追加於本頁，不改寫既有結果。

### P0-A 合併後

審閱與測試證據 commit `ac4c2728a2bbdee44e6f8659dfce080745661d6e` 已 push 並核對遠端 SHA。PR #40 已合併，main commit 為 `c68b7cdbb5bddf1034a9e713c8e8efd5acc61c87`。

在上述 main 重跑 `tests.integration_evidence --phase post_merge`：124 passed，0 failed/error/skipped，pip check 與四個 CLI help 成功；[主線驗證](../../output/branch_integration/2026-10-09-04-13-00/validation.json)。另重跑 guard 的 11 項測試與 8 組成功擬合等價比較，見 [guard 證據](../../output/monitor_guard_evidence/2026-10-09-04-13-12/evidence.json)。此後用獨立文件分支保存 main 驗證，不直接 push main。

## P0-B：PR #41 合併與 main 驗證

PR #41 經 `471c04b...` 更新至 PR #40 後 main，兩側文件索引保留。審閱及重跑證據 commit `ad698dcfb18a47cc85bc01b34475e75200f4622f` 已 push 並與遠端相符；合併 main 為 `ae53b92eea7dd68cd7ffefe8d5c24b3f7157428d`。

受測 main 在 Python 3.10.19 執行 `tests.integration_evidence --phase post_merge`，**162 passed、0 failed/error/skipped**，pip check、四個 CLI help 成功：[驗證](../../output/branch_integration/2026-10-09-11-32-29/validation.json)、[逐項測試](../../output/branch_integration/2026-10-09-11-32-29/tests.txt)。另以 `experiments.health_index_aggregate` 實際重算，[主線重算](../../output/exp8_health_index_aggregate/2026-10-09-11-32-45/recomputation_audit.json) 六 run、54 列、18 工況摘要、349 比對／2 差異，來源未變、無 fit。CLI 明確回傳 3（DIFFERENT），保留為差異證據，不計測試失敗或虛報全部吻合。

#25 的「有差異就回報」工程要求已滿足；歷史兩差值 `0.018513 → 0.018512` 及原始公式 UNKNOWN 保留。沒有要求降低容差，也沒有用零差異當關單條件。先發布固定證據再關單；所有歷史文件的 OPEN 敘述保留於其日期，現況由本頁與 TODO 接續。

## P0-C：主線狀態文件

回讀 #29/#25 均為 closed/completed；#22 的 closedAt 為 2026-10-06。實際讀取 `e7ee971d2ed41d5ce80e3a56f912eb4dc1c038dc:reports/issue_delivery_20261005/hard600_closeout.md`，支持 E02/E04 已執行、108 outer 評估及負面結果。TODO 更正過期 checkbox 與 OPEN 敘述，保留原 Q／solver 結果連結，不搬研究程式。

新文件 [給老師的說明](README.md) 分開列工程完成、FAILED、UNKNOWN／INCOMPLETE；索引與 health 文件不改寫 2026-10-03 歷史表。文件專用驗證入口 `tests.project_closeout_evidence` 共用原 `local_links()`，不需要 gitignored data；[5 份文件／72 個相對目標／0 失效](../../output/project_closeout_evidence/2026-10-09-11-36-02/evidence.json)。這次只驗檔案目標，沒有核對所有錨點或遠端原文。

文件修改後再跑 [完整測試與 CLI](../../output/branch_integration/2026-10-09-11-36-05/validation.json)：162 passed，0 failed/error/skipped、pip check 與四個 CLI help 成功。產物的受測 HEAD 為 `651c0a7...`，文件差異尚在工作樹，來源 SHA 由文件驗證另存；沒有將新文件 commit 誤稱成模型訓練結果。交付用獨立文件 PR，沒有直接 push main。

### P0-C 主線整合

文件 commit `88067075e65d7b0ddee0e33c781df4054ddf1f59` 已 push；PR #42 經審閱合併至 main `4d1858b830890a761af1dc550e74010db8a4b5f5`。真正 main 再跑 [162 項與 pip/CLI](../../output/branch_integration/2026-10-09-11-37-18/validation.json)，0 failed/error/skipped；[75 個相對目標](../../output/project_closeout_evidence/2026-10-09-11-37-17/evidence.json) 皆存在。此段產物於下一個獨立追蹤 PR 保存，不在 main 直接追加 commit。

## P1-A：#30 追蹤規範先提交

重新讀最新 AGENT、main、相關 issue 與 open PR，19 個議題含 closed 去重；目前只 PR #36 open，head 未變，不接手其同檔。七份新 issue 本文與三份既有 issue 補充先保存於 `issues/`。[接續判定與發布規範](README.md#已發布追蹤與去重補充) 明列 17 項狀態，沒有把 git blob 相同當反例重跑。[16 份來源及相對目標](../../output/audit_followup_evidence/2026-10-09-11-41-15/evidence.json) 可回算；仍缺 profiling 與 root escape 證據。

規範 commit `1a9c59c15ab1dc31e9017a4c24c49d86fe863a5a` 已 push；七個議題實際為 #43–#49，回讀皆 OPEN；三則去重補充與 PR #36 協調 URL 見發布清冊。提交後完整162項及pip/CLI通過，來源16檔及15相對目標通過。清冊與§2.2現況接續入庫，不覆寫原稽核表。等待本輪文件PR/main驗證後關 #30。

### P1-A 主線驗收

PR #50 的交付 head 為 `058affde052717213187fa72d98d7bdf3e607746`，已合併至 main `ebb702dae4fe2a79df44d78738d1412b0c24bd2e`。在該 main 執行 `tests.integration_evidence --phase post_merge`：[162 passed，0 failed/error/skipped，pip check 與四個 CLI help 成功](../../output/branch_integration/2026-10-09-11-46-21/validation.json)。另外驗證 [5 份文件／83 個相對目標](../../output/project_closeout_evidence/2026-10-09-11-46-19/evidence.json) 與 [16 份來源／18 個相對目標](../../output/audit_followup_evidence/2026-10-09-11-46-20/evidence.json)，失效目標均為 0。

這些產物經獨立交付分支保存並 push，再以固定 commit 連結發布 #30 完成留言。#30 的完成範圍是逐項核對、去重追蹤與第 2.2 節文件；#43–#49 的程式缺陷仍未修復，PERF 尚無 profiling，SEC-003 的 root escape 仍未確認。PR #36 未獲作者新回覆，不修改其待決檔案。

#30 已於本輪留言後設為 closed/completed，兩個未勾選項改為已勾選，API 回讀確認；[完成留言](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/30#issuecomment-6073908573)。固定證據 commit `4b258c944ec6cbb1ce636cf87825b92824407be6` 已 push，與遠端 SHA 相同。

## P1-B：環境政策第一階段

先提交並 push [環境規範](../../docs/runtime_policy.md)，commit `49fec06168bfb4eb2d921a0f37b50b812fc0d75d`。正式 CPython 3.10.x 採單一 constraints；保留既有 3.10／3.12 環境，不宣稱 main 已支持 3.14。只修改共用 logger、環境紀錄、建置入口與依賴宣告，三支 health CLI 的統一入口待 #24／PR #36。

新增環境 fixture 16 項；[178 項完整回歸、pip check、四個 CLI help](../../output/branch_integration/2026-10-09-12-02-50/validation.json) 全通過、0 failed/error/skipped；[驗證入口](../../output/runtime_policy_evidence/2026-10-09-12-02-46/evidence.json) 不讀正式資料，清冊為 0 檔。guard 另 [11 項與 8 組原版等價](../../output/monitor_guard_evidence/2026-10-09-12-03-32/evidence.json)，score／PCA 差值 0。此處受測 HEAD 為事前規範 commit，程式差異尚在 working tree，不能冒稱已封存程式 commit。

乾淨安裝第一輪在 pip 23 的 cp950 解碼失敗；新增 UTF-8 coding 宣告與回歸，失敗環境及產物保留。第二個新目錄安裝仍須驗證。POSIX 腳本在受限權限下無法啟動 Git Bash；正常權限 `bash -n` 成功，只代表語法，不代表 Linux／macOS 安裝完成。#26 尚未完成每支入口覆蓋，不關單。

程式 commit `815a548e29ee427387b33c14229feb1fe32efdb0` 已 push 並核對。最終全新環境與179項結果、兩次文件檢查及失敗原因見 [環境交付](README.md#當時環境驗收)。實驗／Web 相對基線的 diff 為空，core 只新增環境模組與 logger sidecar；不改正式105維資料或模型預設。只更新預設新 venv 的忽略規則與文件驗證範圍。交付 PR 不使用自動關 #26 的關鍵字。

PR #51 的 head `26320e7...` 已合 main `3ceea37ab1d4b52b7a414a74310eec8401b5b751`；合併後179項、pip／四個CLI、17環境fixture、版本核對、11guard與8組配對，以及7文件109目標全部成功，見 [固定產物連結與限制](README.md#當時環境驗收)。#26不關單。main差異338檔備份到D槽，逐檔解壓SHA一致、ZIP CRC檢查成功；第一次36檔不完整包保留並標FAILED，見 [機器索引](backup_manifest.json)。
