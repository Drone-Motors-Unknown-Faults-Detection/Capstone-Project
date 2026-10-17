# 實驗二十四：逐步導覽與原展示的工程對照

## 資料角色用量表

流程：唯讀檢查 → 明確建立健康基準 → 重播／隔離／候選 → 模擬確認 → 完整池更新 → 查看操作及研究細節。展示／研究是同一session的兩種顯示，不是兩個獨立實驗；首頁與獨立導覽服務則不共用session。

| 資料角色 | T1／8000rpm | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 檢查來源 | 十配置 | 3049列 | 全工況可用量 | 形狀、列數、SHA；fit=0 | 檢查不消耗資料 |
| 初始train／cal／holdout | 健康313列 | 187／62／64 | 各除以313 | 基準／門檻／健康重播 | 列級互斥 |
| 匿名異常來源 | 例S03對應5screws，答案僅此研究說明 | 212可用列 | 故障池全池 | 循環抽樣及候選 | 到達數查session，非獨立新資料 |
| 確認後新增train／cal／holdout | 完整5screws池 | 127／42／43 | 各除以212 | 重建兩類模型 | 可能包含之前發現列 |

來源依[唯讀計數](README.md#資料用量與分母怎麼讀)，未在此表執行新配對或宣稱候選已找到。無validation選參。畫面n／CSV須對同一session、epoch與t；主線保存／重連缺陷的修復在待審[#73](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/pull/73)，未合併前不能當main已有保證。

## 發表與理解檢查

口頭主軸：「用導覽解釋資料、判斷與下一步；計算沿用原monitor，沒有另一套偵測算法。確認前遮蔽配置答案，確認後才顯示模型更新；但後端模擬truth快取和完整池更新須誠實說明。」工程配對相同只驗證兩入口一致，不等於模型準確。

- 切研究模式會重訓嗎？不會，只展開同session細節。
- 重連是否新的獨立測試？不是，保留同模型／資料與曝光歷史。
- 是否已證明組員看懂？需要組員親自走讀、重述，AI不能代簽。
- 組員請實際走一次檢查→基準→未知→確認，說出每一步資料角色；卡住的名詞與畫面請回覆#37，不另製造問答副本。

這個入口讓第一次使用的人按「檢查資料→建立健康基準→輸入未知→確認→觀察更新」操作，也能比較漸進／突發劇本。它重用原LiveDemo與core演算法，不新增偵測方法；工程目標是操作狀態清楚、匿名答案不提前公開、同來源／seed計算一致。

## 資料與模型何時建立

來源是現有105維clean CSV，loader／每列／單位／非有限列過濾與未去重限制見[實驗一](exp1_cold_start.md#資料與載入)。data_contract逐檔算位元組SHA，列各有限值池筆數；每池不足10筆拒絕。資料索引是concat後有限值列，不是raw-window ID；raw_session_independence=UNKNOWN、fresh_final=INCOMPLETE。九工況是motor×RPM，不是九種故障機制，三motor不能串成生命週期。

「確認所選資料」只讀規則，**不fit**；「建立健康基準」才建立LiveDemo並再次比對來源規則，初始只有healthy列洗牌60/20/20，seed42。train fit RobustScaler、LW中心／共變異數（或k-NN train庫）與顯示PCA，cal定0.95距離線。CLI method可改，legacy仍有train定線例外。unknown不fit，CycleSampler重播不代表新採集。

分數>1拒絕、>2才進隔離區；HDBSCAN候選至少25筆，節流與密度階梯見[實驗二](exp2_scale_growth.md)。介面遮蔽未確認配置名，但後端候選已有truth／purity快取，不能稱後端完全無真值。按確認後模擬標註，new class使用**完整標記配置池**重新切分／fit，不是只學已到達群，也不是現場人工病因確認。PCA重新fit，切頁／研究模式切換不fit。

## 實際怎麼操作

從repo根目錄依[uv環境政策](../runtime_policy.md)準備資料，啟動只綁localhost的獨立導覽：

```bash
uv run --locked python -m web.guide --help
uv run --locked python -m web.guide --data-root data --port 8601 --seed 42 --openset-method mahalanobis --method ledoit_wolf --confidence 0.95 --knn-neighbors 5
```

開http://127.0.0.1:8601。這不是原http://localhost:8600首頁；原入口仍需另外啟動web.server，兩者不共用模型session。

1. 選流程與馬達／RPM，按「確認所選資料」，查看筆數與SHA，再按「建立健康基準」。換工況須重新檢查／明確建基準，舊模型不能沿用。
2. 「開始／繼續」播放，「暫停」停止取下一列並flush紀錄；4／10／1筆每秒是播放速度，不是採樣率。最後一個瀏覽器斷線會暫停，重連不重訓。
3. 「觀看未知故障學習」選匿名來源、按「輸入所選來源」。隔離不足或群不成形就持續累積／暫停，25筆不保證有候選。按「模擬操作員確認」只接受目前candidate ID，過期／重複確認拒絕；known群可退回，new群更新後才顯示配置。
4. 「觀看漸進／突發損壞比較」每次先「重設健康基準」，再分別播A260筆／B160筆。演算完自動暫停；比較途中不確認／更新。EWMA與中間帶規則見[實驗三](exp3_trend.md)，時間單位是筆。
5. 「切換研究模式」展開fit／cal／holdout索引、參數與既有七問表。模式不送計算命令，沒有更換模型。

busy時拒絕重複重任務；錯誤後查看操作紀錄，明確重設或重建才能再監測。重設會fit健康、清隔離區並開新epoch；舊紀錄保留。服務重啟不會自動載入權重或續接上次session。缺資料、缺劇本配置、來源載入間SHA變動應停下查來源，不填假資料。

## 工程檢查API與命令

navigation_data_contract.run(dataset,seed=42)只讀來源、不fit；navigation_regression.run(dataset,seed=42)對照原LiveDemo與GuideHub。後者固定LW/q95/k5，分別比較Mahalanobis及k-NN，先40健康、最多600筆5screws等候候選、確認後60筆，再重設各播A/B。逐筆核對來源特徵SHA／所有同值候選列索引、score／verdict／PCA／EWMA／CUSUM／隔離等，前後核來源未變。

```bash
uv run --locked python -m experiments.navigation_data_contract --data-root data --motor T1 --rpm 8000rpm --seed 42
uv run --locked python -m experiments.navigation_regression --data-root data --seed 42
```

regression CLI固定找T1／8000rpm，不支援motor參數，需完整劇本配置；有模型fit但不做全九工況研究。API可傳dataset，沒有自動全量矩陣或續跑入口。候選未找到可記candidate_found=false，逐筆一致仍不證明學習成功；mismatches=0只表示有執行到的配對一致。

## 輸出、讀法與限制

各main經setup_run寫logs/navigation_data_contract、logs/navigation_regression、logs/web_guide及對應output/{program}/{ts}/environment.json。規則工具寫data_contract.json；回歸寫summary.json含paired_methods／ledger／fit_audits。Web在sessionNNN下保存data_contract、各fit_epoch與原LiveDemo紀錄。精確writer見下方程式；不能把未保存的raw距離／最近中心／confusion matrix說成已提供。

score是無因次偏離，學會前拒絕率與學會後接受率分開；接受為任一known不等於自身配置分類正確，更不代表健康恢復。EWMA／CUSUM／latency單位按筆解讀，沒有秒級事件真值、損壞百分比或RUL。

預期匿名／busy／過期確認等防護與配對數值一致；不一致要留失敗紀錄。這是工程驗收，不是部署可靠性／獨立盲測。歷史13～23研究僅由[既有來源索引](../navigation/exp24_來源索引.md)連固定研究提交；main沒有那些runner，不宣稱本頁能跑全部研究。導覽的[七問表](../navigation/exp24_實驗總覽.md)由guide.js解析，維持原24列×8欄，不把本手冊重寫成另一張分數表。

## 來源與程式碼

方法文獻沿[實驗一](exp1_cold_start.md#方法來源與程式定位)、[二](exp2_scale_growth.md)、[三](exp3_trend.md)。匿名介面、SHA規則與配對是Lineage工程約定，源研究快照[b3d68f1](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/b3d68f145cfa187e208e38cae74bf33f68d7c367)，不是新演算法。

[web/guide](../../src/web/guide.py)管理狀態與遮蔽，[guide.html](../../src/web/static/guide.html)／[guide.js](../../src/web/static/guide.js)呈現；[web/live](../../src/web/live.py)共用演算；[data_contract](../../src/experiments/navigation_data_contract.py)寫來源；[regression](../../src/experiments/navigation_regression.py)寫配對；[test_navigation_guide](../../src/tests/test_navigation_guide.py)、[test_integration_navigation](../../src/tests/test_integration_navigation.py)核對工程行為。沒有實際瀏覽器操作證據時，HTTP／fixture測試不冒稱完整視覺驗收。

導覽的整合來源與檔案位置見[整合索引](../../reports/Andy_20261008_分支整合/manifest.json)，單次交付附件放在 reports/，不放進 output/。
