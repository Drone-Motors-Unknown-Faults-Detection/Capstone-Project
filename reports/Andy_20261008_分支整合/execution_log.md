# 分支整合執行紀錄

日期：2026-10-08，Asia/Taipei。工作樹 `D:/schoolshit/專題/src/lineage_integration_20261008`。

## 盤點與備份

- 遠端 main 與新備份 `backup/main-before-integration-20261008-131410` 均為 `64cb71d84663e1745ec54db74abad68def67e8e6`；以 ls-remote 和 GitHub ref API 分別查證。未更新原 checkout 或 source branches。
- 盤點 commit `e8d08f1a6152888b6d475091f0937e271f5a5882` 已 push，遠端 SHA 相同。
- 原環境 Python 3.12.14 缺 loguru，未當作版本驗收。獨立環境 `D:/schoolshit/專題/src/tmp/lineage_integration310` 使用宣告的 Python 3.10.19，依 pyproject 八項依賴安裝；pip check 通過。未修改 pyproject 或原 venv。

## 導覽批次

先寫 `docs/experiments/exp24_experiment_navigation.md`，再從研究固定 SHA 抽取導覽與來源文件。只適配 data 根、exp24 名稱及固定 GitHub 歷史連結，不覆蓋 main 首頁、core、live.py、health 搬家或 AGENT。

實際指令（在工作樹執行，python 為上列環境的 Scripts/python.exe）：

```powershell
python -m tests.integration_evidence --phase baseline --data-root D:/schoolshit/專題/src/Lineage/data
python -m tests.integration_evidence --phase candidate --data-root D:/schoolshit/專題/src/Lineage/data
python -m experiments.navigation_regression --data-root D:/schoolshit/專題/src/Lineage/data --seed 42
python -m web.guide --data-root D:/schoolshit/專題/src/Lineage/data --port 8611 --rate 10 --seed 42
```

- 基準首次 `output/branch_integration/2026-10-08-13-20-42`：76 項中 64 通過、12 個 WinError 5 暫存權限 errors，保留原始證據。只調整測試行程 tempfile 根到該 run output，不改 production 或降低測試要求。
- 基準重跑 `2026-10-08-13-21-54`：76 passed、0 failed/errors/skipped。CLI 四入口與 pip check 通過。
- candidate `2026-10-08-13-22-14` 因沙箱本機 TCP 權限停滯；沒有完整結果，標記 ABORTED。核對自己啟動的命令與 PID 後停止自己的測試／server，沒有停止使用者行程。
- 正常權限 candidate `2026-10-08-13-24-19`：98 passed、0 failed/errors/skipped。未使用 skip 冒充驗收。
- 真實配對 `output/navigation_regression/2026-10-08-13-22-25/summary.json`：Maha 與 k-NN 各 595 筆，共 1,190 筆，score/verdict/PCA/EWMA/CUSUM/quarantine 等欄位差異 0；source_unchanged=true。兩者 75 筆故障到達後有候選；A/B 警報與原 main 相同。這是同工況工程回歸，不是新模型準確率或獨立資料研究。
- 本機實際資料為 60 檔（T1/T3 × 三 RPM × 十配置）；歷史 90 檔未完整恢復，不宣稱全部九工況驗證。完整檔案 SHA 清冊在 validation.json。
- `tests/integration_evidence.py` 子行程使用 PYTHONIOENCODING=utf-8，避免 Windows CLI 中文被錯誤解碼；首次結果不覆寫，後續另產新 run。
- 瀏覽器 QA 完成：匿名候選／確認、模式、reset、A260/B160播畢、T3新工況、reload同session暫停；手寫紀錄移至 [guide_qa.md](browser_qa/guide_qa.md)，外部工具截圖移至 [驗收附件](browser_qa/screenshots)。負面錯誤情境由 HTTP/WebSocket fixture 覆蓋，沒有冒稱 UI 點擊。已關閉測試頁、flush後只停止自己的8611服務。
- UTF-8 修補後 `2026-10-08-13-30-10` 再跑98 passed、0 failed/errors/skipped，四CLI/pip check通過；此回合仍為工程驗收。

## 待整合串流邊界

重新讀取 #36 全文、評論、reviews、checks；head 仍為 `544d4ed8c6516622e2f46c095351f9483a61635c`，無評論／review／checks。#35 仍 OPEN，health_monitor 用健康 holdout 索引取其他較短配置。本輪不接手其修法；若抽取串流，只帶 exp1/3/4，health stream 不註冊、不改 source branch 或 #35 狀態。

## 串流批次

- 導覽 commit `6ed1eae58ff9e581d39ac032d6853d26026cb83b` push及ref核對完成。
- 串流事前規範與三份手冊 commit `cb34a40587797f246bff5e2bf4ba2a32acec7576` 已push、ref相同，之後才抽取程式。
- 抽取 PR36 的 exp1/3/4 與 Web；排除 health_monitor 檔、catalog的health stream旗標及不可達windowcollector。必要適配保留 source commit與原作者，未改原分支。
- 負面 run `output/branch_integration/2026-10-08-13-35-16`：110個 test methods中3個方法未通過、5 failure entries、0 errors/skips；舊wrapper把subTest entries扣成105passed，實際107方法通過。原證據保留，後續wrapper分開記method及subTest計數。
- 確認 close例外會留下JOBS.current；新增最小失敗測試後以nested finally釋放鎖，不改實驗。另拒絕非物件參數與NaN/Inf速率。保存測試的路徑assert改為解讀原API的ROOT相對路徑，不改production保存行為。
- `2026-10-08-13-36-09`：110 passed、0 failed/errors/skipped；正常close/斷線、batch/SSE互斥及例外回歸通過。
- 安全審查拒絕server預設0.0.0.0對外監聽的QA命令，未執行。新增顯式--bind-address（保留原預設），以127.0.0.1啟動8612完成本機QA；未繞過網路保護。
- `2026-10-08-13-37-36`：111 passed、0 failed/errors/skipped；CLI四入口與pip check通過。node --check兩份JS皆exit0。
- `python -m tests.stream_regression --data-root D:/schoolshit/專題/src/Lineage/data --seed 42`：output/stream_integration_regression/2026-10-08-13-36-22/summary.json；exp1/3的Maha/kNN與exp4三子實驗共5組對原MAIN_BEFORE，run/iter_run全部相同，60檔source SHA未變。
- 瀏覽器主入口、背景小窗、三實驗完成、停止、無效次數、錯誤後批次恢復已實際操作。手寫紀錄移至 [stream_qa.md](browser_qa/stream_qa.md)；#35未關閉。

## 封存與PR前驗收

- 串流commit `5aaf48a8a96d92997c359fcaf709e2145f41e66d` 已push、ls-remote相同；保留原作者與Claude/Codex coauthors。
- 對已提交5aaf48a重跑 `2026-10-08-13-41-48`：111 passed、0 failed/errors/skipped，四CLI及pip check通過。導覽回歸重跑`2026-10-08-13-42-01`亦1190筆差異0、來源不變。
- 本輪證據ZIP `D:/schoolshit/專題/src/lineage_integration_backups/2026-10-08-134300/validated_evidence.zip`：29entries全CRC通過、SHA d62021a86273c6d6f69122a65f69ab2adce1015c6683407b024290e8798c6982。未覆蓋舊備份；包內為當時已完成證據，不假裝含尚未發生的merge。
- 抽查歷史23-49-34 ZIP：11entries全CRC通過，SHA414532b03e8dbc3a7e13b22daef2934c5771bdfbddcdcfb35f4cae80108df047。其他大型ZIP僅盤點，未全CRC。
- core、health、pyproject、AGENT、LiveDemo與exp2相對MAIN_BEFORE零diff，原exp6/exp8摘要SHA保留，無tracked deletions。原D槽checkout依舊clean/dda8910，未還原研究鏡像缺失。
- 最後ls-remote main仍64cb71d；PR前再次核對。合併後結果另存delivery，不用mergeable代替測試或已合併狀態。

## 實際 PR 合併與 main 驗收

- 報告commit `92a9ba561a1d6eb23a86cb28cfb05e76ab2078ef` 已push/ref相符。21:17執行確切HEAD全測試，PR #38在測試執行中建立與附加，但等111 passed、0 failed/errors/skipped、四CLI與pip check通過後才merge。首次讀validation時檔案尚未產生，未當成驗收成功。
- 合併前 main/backup皆64cb71d，protection回404 Branch not protected、rulesets空；PR非draft、CLEAN/MERGEABLE、零checks/reviews。沒有CI配置，不寫CI成功；沒有admin bypass或delete-branch。
- `gh pr merge 38 --merge --match-head-commit 92a9ba561a1d6eb23a86cb28cfb05e76ab2078ef` 完成，GitHub state=MERGED、merge=`e1668cb48b91520d2500be643351d9c4c8d9494b`、時間2026-10-08T13:19:08Z。
- fetch後在獨立整合worktree fast-forward origin/main，HEAD exact e1668cb。`tests.integration_evidence --phase post_merge` 產物21-19-15：Python3.10.19，111 passed/0 failed/errors/skipped、pip check/四CLI通過，資料60檔SHA不變。
- `tests.stream_regression --seed 42` 產物21-19-45：五組對原main配對相同，來源未變。再次查詢所有branches、PR #36、issue #35，來源head與OPEN狀態不變；原D槽checkout仍clean。
- 合併後證據及delivery另以文件PR提交，不改core/Web/experiments/tests實作。Git記錄與最終遠端查詢分開，避免把未發生的提交寫成已完成。

## 文件交付 PR #39

- 文件commit `0edb4d8f7daca5f191552afa8e008ceadccb6d9e` 已push，ls-remote相符；建立並附加PR #39，base e1668cb。CLEAN/MERGEABLE、非draft、checks/reviews空，保護規則仍無新增。
- 對確切0edb4d8執行完整候選驗收21-22-14：111 passed、0 failed/errors/skipped、pip check與四CLI通過、60檔SHA不變。程式tree與已驗收e1668cb相同。
- 最後加入本輪log/output及此說明後，仍只增加證據；再做無新logger產物的完整unittest驗收，核對HEAD與main，正常merge。最終merge SHA由PR #39的mergeCommit與遠端main查詢給出，不在提交內假造未存在SHA。
