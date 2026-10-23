# 實驗十三：逐步導覽與原展示的工程對照

## 主站入口與獨立模式

主站首頁的「逐步導覽」連到同來源 `/guide/`，不需新增連接埠。主站與導覽各有自己的 Hub、session、模型、rng、重任務鎖及寫檔 epoch；開啟導覽只顯示可用資料，須明確「確認所選資料 → 建立健康基準」才擬合。導覽 WS 為 `/guide/ws`，資產與文件各用 `/guide/assets/`、`/guide/docs/`，不連主站 `/ws`。

主站內導覽產物在同次 `output/web_server/{ts}/guide/sessionNNN/`；獨立 `python -m web.guide` 保留根路徑 `/ws`、`/assets/`、`/docs/` 及 `output/web_guide/{ts}/sessionNNN/`。獨立模式沿 loopback 監聽，不假設遠端使用者的 localhost 是伺服器。最後導覽連線離開會暫停並 flush；主站正常退出會關閉兩個 Hub，各自保存。強制終止或掉電不保證零遺失。

路由／來源遮蔽／重連及逐筆配對是工程驗證；桌面／窄屏的實際瀏覽器操作與組員重述資料角色仍須各別回報。沒有部署授權時只在本機驗證，不更新公開服務。

## 資料角色用量表

流程：唯讀檢查 → 明確建立健康基準 → 重播／隔離／候選 → 成群當拍 `adopt()` 自學 → 查看操作及研究細節。展示／研究是同一session的兩種顯示，不是兩個獨立實驗；首頁與獨立導覽服務則不共用session。

| 資料角色 | T1／8000rpm | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 檢查來源 | 十配置 | 3049列 | 全工況可用量 | 形狀、列數、SHA；fit=0 | 檢查不消耗資料 |
| 初始train／cal／holdout | 健康313列 | 187／62／64 | 各除以313 | 基準／門檻／健康重播 | 列級互斥 |
| 匿名異常來源 | 例S03對應5screws，答案僅此研究說明 | 212可用列 | 故障池全池 | 循環抽樣及候選 | 到達數查session，非獨立新資料 |
| 自學新增train／cal／holdout | 該次候選群（例：51筆） | 約60／20／20 | 分母是群筆數，不是212 | 重建兩類模型，類名discovered-N | 與重播列重疊；不讀完整配置池 |

來源依[唯讀計數](README.md#資料用量與分母怎麼讀)，未在此表執行新配對或宣稱候選已找到。無validation選參。畫面n／CSV須對同一session、epoch與t；暫停成功及正常關閉會flush，重連由後端恢復實際來源與統計，相關邊界由 `test_web_persistence.py` 與導覽測試核對。這些行為不保證強制終止或掉電後零遺失。

## 發表與理解檢查

口頭主軸：「用導覽解釋資料、判斷與下一步；計算沿用原monitor，沒有另一套偵測算法。成群當拍只用該群樣本自學，類名 discovered-N，導覽始終不顯示配置答案；後端為事後對照快取的truth多數須誠實說明。」工程配對相同只驗證兩入口一致，不等於模型準確。

- 切研究模式會重訓嗎？不會，只展開同session細節。
- 重連是否新的獨立測試？不是，保留同模型／資料與曝光歷史。
- 是否已證明組員看懂？需要組員親自走讀、重述，AI不能代簽。
- 組員請實際走一次檢查→基準→未知→成群自學，說出每一步資料角色；卡住的名詞與畫面請回覆#37，不另製造問答副本。

這個入口讓第一次使用的人按「檢查資料→建立健康基準→輸入未知→成群自學→觀察更新」操作，也能比較漸進／突發劇本。它重用原LiveDemo與core演算法，不新增偵測方法；工程目標是操作狀態清楚、匿名答案不提前公開、同來源／seed計算一致。

## 資料與模型何時建立

來源是現有105維clean CSV。非有限值整批拒絕、不刪列；單位與未去重見[實驗一](exp1_cold_start.md#資料與載入)。data_contract逐檔算位元組SHA，列各有限值池筆數；每池不足10筆拒絕。資料索引是concat後有限值列，不是raw-window ID；raw_session_independence=UNKNOWN、fresh_final=INCOMPLETE。九工況是motor×RPM，不是九種故障機制，三motor不能串成生命週期。

「確認所選資料」只讀規則，**不fit**；「建立健康基準」才建立LiveDemo並再次比對來源規則，初始只有healthy列洗牌60/20/20，seed42。train fit RobustScaler、LW中心／共變異數（或k-NN train庫）與顯示PCA，cal定0.95距離線。CLI method可改，legacy仍有train定線例外。unknown不fit，CycleSampler重播不代表新採集。

分數>1拒絕、>2才進隔離區；HDBSCAN候選至少25筆，節流與密度階梯見[實驗二](exp2_scale_growth.md)。介面始終遮蔽配置名，但後端候選有truth／purity快取（供首頁事後對照），不能稱後端完全無真值。成群當拍 `adopt()`：群中位分數要超過已知類校準最大值，new class只用**這一群到達樣本**重新切分／fit，類名 discovered-N；不讀完整配置池，也沒有人工確認（#287）。PCA重新fit，切頁／研究模式切換不fit。

## 實際怎麼操作

從repo根目錄依[uv環境政策](../runtime_policy.md)準備資料，啟動只綁localhost的獨立導覽：

```bash
uv run --locked python -m web.guide --help
uv run --locked python -m web.guide --data-root data --port 8601 --seed 42 --openset-method mahalanobis --method ledoit_wolf --confidence 0.95 --knn-neighbors 5
```

開http://127.0.0.1:8601。這不是原http://localhost:8600首頁；原入口仍需另外啟動web.server，兩者不共用模型session。

1. 選流程與馬達／RPM，按「確認所選資料」，查看筆數與SHA，再按「建立健康基準」。換工況須重新檢查／明確建基準，舊模型不能沿用。
2. 「開始／繼續」播放，「暫停」停止取下一列並flush紀錄；4／10／1筆每秒是播放速度，不是採樣率。最後一個瀏覽器斷線會暫停，重連不重訓。
3. 「觀看未知故障學習」選匿名來源、按「輸入所選來源」。隔離不足或群不成形就持續累積／暫停，25筆不保證有候選。成群當拍自動自學，狀態列顯示「最近自學」的筆數、動作與類名；邊界群退回。導覽不顯示配置答案。
4. 「觀看漸進／突發損壞比較」每次先「重設健康基準」，再分別播A260筆／B160筆。演算完自動暫停。EWMA與中間帶規則見[實驗三](exp3_trend.md)，時間單位是筆。
5. 「切換研究模式」展開fit／cal／holdout索引、參數與既有七問表。模式不送計算命令，沒有更換模型。

busy時拒絕重複重任務；錯誤後查看操作紀錄，明確重設或重建才能再監測。重設會fit健康、清隔離區並開新epoch；舊紀錄保留。服務重啟不會自動載入權重或續接上次session。缺資料、缺劇本配置、來源載入間SHA變動應停下查來源，不填假資料。

## 工程檢查API與命令

navigation_data_contract.run(dataset,seed=42)只讀來源、不fit；navigation_regression.run(dataset,seed=42)對照原LiveDemo與GuideHub。後者固定LW/q95/k5，分別比較Mahalanobis及k-NN，先40健康、最多600筆5screws等候第一次自學（`LiveDemo.adoption`）、自學後60筆，再重設各播A/B；`summary.json` 的 `adoption` 欄記兩入口相同的 `adopt()` 結果（schema `navigation_regression_v3`）。等候結束後先比兩入口是否都已自學（`adoption_present`）：只有一邊學到新類（任一方向）就記一筆不一致、`adoption` 為null，不再讀另一邊的欄位，回歸不會中途崩潰；兩邊都學到才比 `adoption` 內容與 fit audit，再播60筆。逐筆核對來源特徵SHA／所有同值候選列索引、score／verdict／PCA／EWMA／CUSUM／隔離等，前後核來源未變。

```bash
uv run --locked python -m experiments.navigation_data_contract --data-root data --motor T1 --rpm 8000rpm --seed 42
uv run --locked python -m experiments.navigation_regression --data-root data --seed 42
```

regression CLI固定找T1／8000rpm，不支援motor參數，需完整劇本配置；有模型fit但不做全九工況研究。API可傳dataset，沒有自動全量矩陣或續跑入口。候選未找到可記candidate_found=false，逐筆一致仍不證明學習成功；mismatches=0只表示有執行到的配對一致。比對不用 assert：`MismatchLedger` 逐項計數，每法 `mismatches` 是實際不一致數、`mismatch_details` 留前20筆（pair_id、欄位、兩邊的值），頂層 `mismatches` 為兩法合計，`source_unchanged` 是前後兩次 data_contract 的實際比較；有不一致或來源變動時 `status=FAILED`，CLI 照樣寫 `summary.json` 再以退出碼1結束，中途例外也先寫含 `error` 的 `summary.json` 再拋出（#301）。

## 輸出、讀法與限制

各main經setup_run寫logs/navigation_data_contract、logs/navigation_regression、logs/web_guide及對應output/{program}/{ts}/environment.json。規則工具寫data_contract.json；回歸寫summary.json含paired_methods／ledger／fit_audits。fit audit（`split_audit`，schema `navigation_fit_v2`）逐類標 `fit_source`：配置池類別的索引是 concat 後有限值列索引；discovered-N 的 `fit_source` 是隔離區候選群，索引是候選群矩陣內的位置，`cluster_rows` 另記每個位置的來源池列（[配置, 池列號]）——只寫進 sessionNNN 的 fit_epoch 檔與回歸輸出，導覽送給瀏覽器的狀態不含它（會洩漏匿名來源答案）。Web在sessionNNN下保存data_contract、各fit_epoch與原LiveDemo紀錄。精確writer見下方程式；不能把未保存的raw距離／最近中心／confusion matrix說成已提供。

score是無因次偏離，擴張前拒絕率與擴張後判為未知／健康／新類三種比例分開（分段看模型是否已學到新類，不看來源答案）；判為新類同實驗二`known_rate`，只數非健康已知類，被判為健康不算，也不等於自身配置分類正確。EWMA／CUSUM／latency單位按筆解讀，沒有秒級事件真值、損壞百分比或RUL。

2026-10-23在Aero（aero01）以commit `8756c51`、`--data-root data`（資料指紋`1b55b847…`）重跑回歸CLI。[navigation_regression/2026-10-23-12-48-17](../../output/navigation_regression/2026-10-23-12-48-17/summary.json)（schema `navigation_regression_v3`）：Mahalanobis與k-NN各595組配對、0筆不一致，兩法都在故障到達75筆時出現候選，同一拍 `adopt()` 學成 `discovered-1`（session第115筆、51筆、`source=cluster_only`；群中位分數85.52／44.65，已知校準最大1.119／1.081）；事後對照多數為5screws、purity 1，這兩欄只在回歸輸出與首頁事件出現。`adopted` 階段的fit audit（`navigation_fit_v2`）記 discovered-1 的 train／cal／holdout 為30／10／11、`fit_source` 為隔離區候選群、`cluster_rows` 51列；`source_unchanged=true`。舊輸出[02-04-52](../../output/navigation_regression/2026-10-23-02-04-52/summary.json)是 `46ed4d5` 的 `confirm()` 版（schema v2，沒有 `adoption`），配對數、不一致數與候選到達筆數相同，但新類當時讀5screws完整池切127／42／43。規則工具的輸出在[navigation_data_contract/2026-10-23-02-04-50](../../output/navigation_data_contract/2026-10-23-02-04-50/)（schema `navigation_data_v1` 未變，沒有重跑）。

預期匿名／busy等防護與配對數值一致；不一致要留失敗紀錄。這是工程驗收，不是部署可靠性／獨立盲測。研究分支另有未進 main 的 runner，手冊標成「分支舊編號」，不是本實驗十三。[來源索引](../navigation/exp13_來源索引.md)只連到固定研究提交。本頁不能跑那些研究。導覽的[七問表](../navigation/exp13_實驗總覽.md)由guide.js解析，維持原24列×8欄，不把本手冊重寫成另一張分數表。

## 來源與程式碼

方法文獻沿[實驗一](exp1_cold_start.md#方法來源與程式定位)、[二](exp2_scale_growth.md)、[三](exp3_trend.md)。匿名介面、SHA規則與配對是Lineage工程約定，源研究快照[b3d68f1](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/b3d68f145cfa187e208e38cae74bf33f68d7c367)，不是新演算法。

[web/guide](../../src/web/guide.py)管理狀態與遮蔽，[guide.html](../../src/web/static/guide.html)／[guide.js](../../src/web/static/guide.js)呈現；[web/live](../../src/web/live.py)共用演算；[data_contract](../../src/experiments/navigation_data_contract.py)寫來源；[regression](../../src/experiments/navigation_regression.py)寫配對；[test_navigation_guide](../../src/tests/test_navigation_guide.py)、[test_integration_navigation](../../src/tests/test_integration_navigation.py)核對工程行為。沒有實際瀏覽器操作證據時，HTTP／fixture測試不冒稱完整視覺驗收。

導覽的整合來源與檔案位置見[整合索引](../../reports/Andy_20261008_分支整合/manifest.json)，單次交付附件放在 reports/，不放進 output/。
