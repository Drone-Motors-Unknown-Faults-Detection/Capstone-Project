# Lineage 分支整合報告

2026-10-08，Asia/Taipei。候選程式版本：`5aaf48a8a96d92997c359fcaf709e2145f41e66d`。下列保留合併前驗收紀錄；PR #38 已於21:19合併，main `e1668cb` 的111項合併後測試通過，合併後證據見下方「合併與交付證據」。

## 完成範圍

本報告保留 2026-10-08 的整合邊界。串流抽取來源為 JW-Albert 的 PR #36／544d4ed，原共同作者為 Claude Opus 5.5；當次只納入 exp1／3／4，不含後來才合併的健康監測串流。事前配對要求及當時 localhost QA 命令見 [固定原規範](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/integration_20261008/streaming_contract.md)。現行操作以實驗手冊為準，不用這份歷史範圍判定當前功能。

已備份整合前 main、盤點全部分支、分批抽取獨立導覽與 exp1/3/4 SSE，完成真實資料配對及瀏覽器操作。沒有整批合併研究分支，沒有修改原分支或刪分支。每批 commit/push 與遠端 ref 相符。PR 必須核對最新 main、候選 head、保護與 reviews/checks 後才合併。

- [遠端備份](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/backup/main-before-integration-20261008-131410)：`64cb71d84663e1745ec54db74abad68def67e8e6`，等於 MAIN_BEFORE；不使用落後的本機 main 當備份來源。
- [全部分支矩陣](#原分支來源快照)、[執行紀錄](execution_log.md)、[機器索引](manifest.json)、[現行串流操作](../../docs/experiments/README.md#web-串流生命週期)。
- 給老師先讀 [實驗十三導覽入口](../../docs/navigation/exp13_README.md)，再看實驗總覽、白話解讀及示範腳本。老師問題的原逐項說明仍保留在[固定研究版本 teacher reply](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/b3d68f145cfa187e208e38cae74bf33f68d7c367/reports/teacher_reply_20261005/reply.md)。

## 分支最終處理決策

| 原分支 | 本輪決策 | 證據／保留原因 |
|---|---|---|
| feat/web-experiment-pages | 先前已整合，不重套 | #33 MERGED、tip 是 main 祖先 |
| feat/web-fit-notice | 先前已整合，不重套 | #34 MERGED、tip 是 main 祖先且 tree 相同 |
| refactor/exp8-health-into-experiments | 先前已整合，不倒退 | #32 MERGED；保留 experiments/health 搬家 |
| research-improvements-20260920 | 僅抽取導覽、來源規範、配對入口與 exp24 文件 | 固定 b3d68f；其餘3128檔完整tip差異含大量研究輸出、刪除、舊AGENT/health/Web。其他研究模組及guards未完成主線相依閉包驗證，留原分支；負面研究以固定連結保存，FAILED不換成正式winner |
| feat/web-streaming | 僅抽取 exp1/3/4、SSE及背景小窗 | 固定544d4ed；health_monitor與health串流旗標排除。#35索引缺陷仍OPEN。#36保留OPEN，避免把部分抽取冒稱整個PR完成 |
| backup_260806 | 歷史備份保留 | 舊Notebook，不作功能合併來源 |
| main | 被備份的整合基底 | 候選只加合格內容，透過PR更新，不force push |

研究分支其餘部分分類為「未完成主線整合驗證」，不等於全部不可用。Group DRO 是規劃，未實作／未實測；各FAILED家族、sealed protocol、引用卡、來源卡與原產物仍在固定研究提交及D槽。沒有用舊研究版本覆蓋 main 新架構，亦沒有替 Albert 完成其他 programming issue。

## 測試與實際效能差異

宣告與實際均 Python **3.10.19**；獨立環境 `D:/schoolshit/專題/src/tmp/lineage_integration310`。原環境3.12.14缺loguru，沒有把pip check當版本驗收。

| 驗證 | 實際結果 | 保存位置 |
|---|---|---|
| 整合前 main 的既有測試 | 76 passed、0 failed/errors/skipped | output/branch_integration/2026-10-08-13-21-54 |
| 導覽候選 | 98 passed、0 failed/errors/skipped | 2026-10-08-13-30-10 |
| 完整已提交候選5aaf48a | 111 passed、0 failed/errors/skipped；pip check與四CLI通過 | 2026-10-08-13-41-48 |
| 導覽逐樣本配對 | Maha/kNN各595、合計1190筆，差異0；重跑相同、來源未變 | output/navigation_regression/2026-10-08-13-42-01/summary.json |
| exp1/3/4 對原main配對 | 5組：exp1/3各Maha與kNN、exp4全部三段；原run、新run、iter_run結果全相同 | output/stream_integration_regression/2026-10-08-13-36-22/summary.json |
| 真實瀏覽器 | 導覽的build/confirm/reset/mode/dataset/reload、A/B停止；原首頁、背景小窗、三實驗完成、停止、錯誤後舊入口恢復 | [手寫導覽紀錄](browser_qa/guide_qa.md)、[手寫串流紀錄](browser_qa/stream_qa.md)；9張外部工具截圖在 [驗收附件](browser_qa/screenshots) |
| 原封存摘要 | exp6=54runs、exp8=54列可讀；SHA保留、檔案未改 | manifest.json |

Windows暫存權限造成的首次12 errors、沙箱TCP停滯ABORTED、串流首次3個test methods／5failure entries均保存，沒有刪除負面證據。串流修補只處理close例外鎖釋放及非有限速率／非物件JSON；保存路徑測試改為正確解讀ROOT相對路徑。未調模型、門檻或資料讓成績對齊。

同工況exp1本輪UI的healthy FP=7/64=10.9375%、九配置unknown拒絕100%、AUROC1，與原main相同；**差異0百分點**。這不是跨馬達fault-type分類或已解決T1跨馬達未知召回問題。A/B是CSV劇本，延遲單位是筆數；沒有物理退化真值、每小時誤報或RUL。

完整候選中 `core/`、`web/live.py`、`experiments/exp2_scale_growth.py`、`experiments/health/`、`health_monitor.py`、`pyproject.toml`、`AGENT.md` 相對MAIN_BEFORE零diff。正式105維、LW預設、q95、kNN5 factory、PolarMap與未知不fit規則不變。exp1/3/4只由run抽成iter_run，共用相同計算。

## 資料、來源與未知限制

實際唯讀資料根 `D:/schoolshit/專題/src/Lineage/data` 為**60份**clean CSV：T1/T3 ×三RPM ×十配置。完整清冊SHA `82534111c7901bfe0709637f4979976ce73445a54353e7c377e71e40b4cf2abd`，前後相同。本輪沒有重建T2，不能冒充歷史90檔指紋，也未重新執行2490次研究或完整九工況回歸。

原始session、錄製邊界、重疊窗口、清理遮罩、設備／負載等仍UNKNOWN；歷史曝光與fresh INCOMPLETE保留。T1/T2/T3為不同馬達，不能串成一顆馬達生命週期。舊main展示保留的「每點1秒」等文案屬既有設定，未由本輪資料來源證明；獨立導覽明示CSV筆數與這些限制。確認後用完整標記配置池重擬合是oracle展示，不稱arrival-only學習。

## 備份、工作區與回復

- 工作樹：`D:/schoolshit/專題/src/lineage_integration_20261008`。
- 原checkout：`D:/schoolshit/專題/src/Lineage`，仍保留dda8910，未切branch、未修改data、原status無dirty。
- 新D槽證據包：`D:/schoolshit/專題/src/lineage_integration_backups/2026-10-08-134300/validated_evidence.zip`；29entries、全包CRC通過、SHA見manifest。沒有覆蓋歷史備份。
- 抽查舊 `lineage_fault_type_artifacts/2026-09-30/fault_type_2026-09-30-23-49-34.zip`：11entries、CRC通過；其他三份共約11GB只盤點路徑／大小，未全包CRC或重訓，不能聲稱全部恢復驗證。
- Git備份僅保留tracked內容，不含data、venv、其他ignored模型、未提交修改。未發現LFS、submodule或gitlink。
- 候選無意外tracked deletions；282項舊刪除未帶入。小型log／output／截圖入Git，大型來源ZIP與raw data未stage。

需要回復時先查PR的merge SHA及父提交，用新分支做正常revert並建PR；merge commit需核對mainline-parent。不要reset/force push main到備份。備份分支可建立獨立checkout查閱；它不能代替資料或大型外部產物備份。

## 執行入口

```powershell
# 先進工作樹，啟用宣告版本環境；data根保持唯讀
python -m tests.integration_evidence --phase candidate --data-root D:/schoolshit/專題/src/Lineage/data
python -m experiments.navigation_regression --data-root D:/schoolshit/專題/src/Lineage/data --seed 42
python -m tests.stream_regression --data-root D:/schoolshit/專題/src/Lineage/data --seed 42
python -m web.guide --data-root D:/schoolshit/專題/src/Lineage/data --port 8601 --seed 42
python -m web.server --bind-address 127.0.0.1 --data-root D:/schoolshit/專題/src/Lineage/data --port 8600 --seed 42
```

工程測試需要完整Git歷史中的MAIN_BEFORE。合併後執行用 `--phase post_merge`。本輪沒有部署、對外開放QA服務、聯繫老師／作者或其他聊天室。

## 原分支來源快照

此表記錄2026-10-08盤點時的tip，behind／ahead相對整合前main，tree diff是兩tip完整差異；不能當作目前分支狀態。

| 分支 | head SHA | behind / ahead；tree diff | PR／決策 | 相依與驗證 |
|---|---|---|---|---|
| main | 64cb71d84663e1745ec54db74abad68def67e8e6 | 0 / 0；0 | 整合基準 | 完整測試、CLI、factory、Web 基準待本輪執行 |
| backup_260806 | 3a699ab81c16503fff4ba2c2b8ec566202c24d7d | 115 / 0；1327 | 歷史備份，永久保留 | 舊 Notebook 架構，不是功能候選 |
| feat/web-experiment-pages | 926da639c76e3642c5854c541fc732bb3eac6c55 | 4 / 0；12 | #33 已合併，祖先證明；不重套 | 保留 main 的實驗頁及 API |
| feat/web-fit-notice | f65ab1538c7c973085ca2674c4f601692d2ac14c | 1 / 0；0 | #34 已合併，tree 等同 main | 不重套 |
| refactor/exp8-health-into-experiments | 12c97d6adbd954c23a78bb29cc09e27be8bc13eb | 9 / 0；22 | #32 已合併，祖先證明 | 保留 experiments/health 與 main 文件規範 |
| feat/web-streaming | 544d4ed8c6516622e2f46c095351f9483a61635c | 0 / 1；15 | #36 OPEN；先驗證，必要時只抽取合格部分 | exp1/3/4/8-3 iter_run、SSE、共用鎖；#35 健康索引套到短故障池未解決；不得冒稱修復 |
| research-improvements-20260920 | b3d68f145cfa187e208e38cae74bf33f68d7c367 | 29 / 183；3128 | 部分抽取，禁止整批 merge | 舊檔刪除、health 路徑、Web 與 AGENT 都和 main 衝突；先整合可獨立驗證的導覽／工具，研究成績保留固定來源，不換正式預設 |
| backup/main-before-integration-20261008-131410 | 64cb71d84663e1745ec54db74abad68def67e8e6 | 0 / 0；0 | 本輪已驗證備份 | 不整合、不刪除 |
| integration/branches-20261008-131410 | 基準為 64cb71d84663e1745ec54db74abad68def67e8e6 | 本輪產物 | 新整合分支 | 每批限定檔案、測試、commit、push |


當時Git AGENT blob：main及Web分支為8f35a6bf14fa4747d63add1d6bc852b65bcb4784；exp8 refactor為39199ecf498aabcc9b429a50eafe1033c99eda4d；研究分支為f3b92347582bceff98ede6513f06458cdefdeac5。整合依main的experiments/health與文件規則，不回寫舊規範。當時main protection未設定、rulesets空、PR36零checks／reviews，pending不是CI成功；沒有admin bypass。[原盤點全文](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/integration_20261008/branch_inventory.md)保留查詢邊界與工作區紀錄。

## 合併與交付證據

[PR38](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/pull/38) 的候選92a9ba561a1d6eb23a86cb28cfb05e76ab2078ef合併為e1668cb48b91520d2500be643351d9c4c8d9494b。候選[validation](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/blob/17302e2ad1885ed8bb4f227401b6a4ca453e6bb4/output/branch_integration/2026-10-08-21-17-25/validation.json)與[tests](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/blob/17302e2ad1885ed8bb4f227401b6a4ca453e6bb4/output/branch_integration/2026-10-08-21-17-25/tests.txt)、合併後[validation](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/blob/17302e2ad1885ed8bb4f227401b6a4ca453e6bb4/output/branch_integration/2026-10-08-21-19-15/validation.json)與[tests](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/blob/17302e2ad1885ed8bb4f227401b6a4ca453e6bb4/output/branch_integration/2026-10-08-21-19-15/tests.txt)均保存；111項通過、0 failed/error/skipped，pip check及四CLI成功。合併後[五組原main配對](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/blob/17302e2ad1885ed8bb4f227401b6a4ca453e6bb4/output/stream_integration_regression/2026-10-08-21-19-45/summary.json)差異0。

後續證據[PR39](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/pull/39)的0edb4d8候選[validation](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/blob/17302e2ad1885ed8bb4f227401b6a4ca453e6bb4/output/branch_integration/2026-10-08-21-22-14/validation.json)同為111項；[原交付全文](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/integration_20261008/delivery.md)的合併驗收條件只適用當時版本。證據包SHA d62021a86273c6d6f69122a65f69ab2adce1015c6683407b024290e8798c6982，29 entries、CRC通過，只含合併前證據，後續結果由Git保存。

原pr_description.md與PR38已發布本文在換行正規化後逐字相同，沒有未發布的獨有內容；保留PR本文及Git歷史，不另留說明副本。

## 導覽抽取的歷史來源

2026-10-06導覽起點為研究5be4c7865a721afaab3404a5481377f39ded7abd，當時main為64cb71d、PR36 head544d4ed。研究正式版本為90 CSV、28,910筆105維，fingerprint c4145d6efcbf02d294e77e5d83fdab5636bba6d9a58b1707752dd34b836f0b2d；與本次整合實際可讀60檔版本不同。3.10.19／3.14.6是該研究環境資訊，不是本整合的雙環境驗收。

當時選獨立guide檔案以避開Albert的server／index／experiments.js及iter_run修改；重用LiveDemo、core與ScaleGrowthSession，未帶入282項來源不明刪除。這些是歷史分工，不是現行功能禁區；獨有脈絡及當時寫作來源無法讀取的限制仍可查[原範圍快照](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/navigation/exp24_範圍與分工.md)。
