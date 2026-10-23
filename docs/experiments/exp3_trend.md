# 實驗三：漸進與突發的劇本判別

## 資料角色用量表

流程：建固定健康基準 → 劇本循環抽特徵 → 分數轉超線旗標 → EWMA累積 → 首次告警並判gradual／sudden。這是在驗證抽樣節奏，不是在預測真實磨損時間。

| 資料角色 | 馬達／轉速／類別 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| train／cal | T1／8000rpm／健康 | 187／62 | 各除以313健康池 | 所有trial共用一份基準 | 未進故障資料 |
| 健康重播來源 | 同上holdout | 64可用列 | 64/313；不是每trial消耗64筆 | 健康段及混入段抽樣 | 循環、跨trial重用 |
| 故障重播來源 | 7／6／5／4screws | 322／314／212／287 | 各配置全池100%可用 | A用7/6/5，B用4 | 未fit，但可循環重用 |
| 單trial長度 | A／B劇本 | 最多260／160 tick | 無獨立列分母；首警報提前停止 | 判別與延遲 | 不能相加成新採集數 |

數量依[唯讀分母](README.md#資料用量與分母怎麼讀)，非新trial結果；驗證／選參未使用。20trial只重複抽樣，raw錄製獨立UNKNOWN。

## 發表與理解檢查

口頭主軸：「用同一健康模型，把未知出現得慢或快的兩種CSV劇本送進來。EWMA把最近異常旗標平滑，再按中間區停留筆數判斷節奏。結果支持的是劇本辨別，沒有現場退化真值。」公式與變數見下方：EWMA的輸入是0/1旗標，α=.08控制記憶，不是振動幅值。

- 延遲10代表10秒嗎？不能；這是重播筆數，沒有原始時間戳。
- T1、T2、T3能串成慢慢老化嗎？不能，三者是不同個體。
- 為什麼同時報alarm_rate與correct_rate？沒警報和警報但判錯是不同失敗；只看已警報trial會漏掉前者。
- 組員請說出抽樣混入比例與物理故障程度的差別；實際結果及未警報情況查輸出節。

健康與故障的CSV樣本按兩種節奏混入，分數序列能不能區分「慢慢變多」與「一下變多」？它使用[實驗一](exp1_cold_start.md)的固定健康基準，不在劇本中學新類。這是抽樣劇本檢查，沒有同一顆馬達真正磨損的生命週期真值。

## 資料與模型怎麼準備

單一馬達／RPM的105維clean特徵、loader行為與單位／來源限制同[實驗一](exp1_cold_start.md#資料與載入)。健康池由seed42列洗牌60/20/20；train fit scaler、LW中心／共變異數（或k-NN參考）與顯示PCA，cal定confidence0.95門檻。legacy例外用train距離定門檻。未知配置不fit、不校準；healthy串流只抽holdout，fault串流抽各配置全池。CycleSampler洗牌、循環重用，不保證raw視窗獨立。

整次run只fit一次健康monitor，所有trial共用這個模型。trial另建TrendMonitor並重設狀態；A的rng為seed+1000+k，B為seed+2000+k（k從0開始）。因此20trial是同資料與模型下的抽樣重複，不是20顆馬達。

## 劇本與判別順序

| 劇本 | 有順序的階段（配置／抽故障機率／筆數） | 預期標籤 |
|---|---|---|
| A | 健康／0／40 → 7screws／0.25／50 → 7screws／0.7／50 → 6screws／1／60 → 5screws／1／60 | gradual |
| B | 健康／0／40 → 4screws／1／120 | sudden |

機率剩餘部分抽健康holdout，不是把兩筆特徵線性混合。onset固定40；各trial第一個alarm_now出現就停止，未必播完表中全部筆數。

每筆先算Open Set score，再交給[src/core/trend.py](../../src/core/trend.py)：
`flag=1{score>1}; ewma += 0.08*(flag-ewma)`。EWMA平滑的是異常旗標，不是原始振動或健康百分比。超過warmup10筆後，EWMA至少0.5才首次警報。當下從最近120筆歷史數EWMA落在[0.2,0.5)的筆數，transition≤12叫sudden，其餘gradual；當前警報點是在判定後才加入history。CUSUM為`max(0,previous+score-1)`，只展示、不決定kind。

這些固定規則不訓練神經網路，沒有loss／optimizer；不從T1／T2／T3推論退化速度。健康早期若已誤報也會停止，latency可能是負值，不能偷偷截成0。

## 如何執行與使用

在repo根目錄，依[環境政策](../runtime_policy.md)準備uv與資料：

```bash
uv run --locked python -m experiments.exp3_trend --help
uv run --locked python -m experiments.exp3_trend --data-root data --motor T1 --rpm 8000rpm --seed 42 --trials 40
```

CLI支援共用openset-method／method／confidence／knn-neighbors；`--trials`預設20，README與[40次結果](../../output/exp3_trend/2026-10-23-01-32-36/summary.json)用40，重現時明寫`--trials 40`；必須正整數，程式未完整驗證0或負值，可能零分母；不要當空跑成功。API為`run(pools,n_trials=20,seed=42)`，回傳trials與summary；iter_run逐事件計算，不自行持久化；每個tick事件帶監測器的`unknown`旗標（`score > 1`），Web實驗頁只抄旗標著色，不用真值也不重算。

Web啟動見[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)，切「實驗三」、選資料集／方法／seed／每劇本次數，按「▶ 執行」。「劇本 A：漸進磨損」與「劇本 B：突發外力」只播放並統計該劇本；「⏵ 邊跑邊畫」仍兩個劇本一起跑，可看各劇本首trial的分數／EWMA，其餘trial出統計。「■ 停止」會取消後續串流，不把未收到done的結果當完整。批次沒有中途停止按鈕。即時展示的A/B劇本與本頁共享SCENARIOS，但session可能已擴張，不能直接當冷啟動同一設定比較。

缺7、6、5或4screws時會失敗，應核對配置來源，不補造資料。模型未fit須重建；重跑目錄由setup_run秒級時間戳建立，不同秒分開，沒有trial續跑入口。

## 輸出怎麼看

[main](../../src/experiments/exp3_trend.py)寫`logs/exp3_trend/{ts}.log`及`output/exp3_trend/{ts}/environment.json`、`trials.csv`、`summary.json`；Web結果另存web_server的experiments子目錄。

`trials.csv`每列是劇本／trial，不是輸入視窗。expected是劇本指定標籤；verdict是程式結果；correct含未警報為false。alarm_t、latency=alarm_t−40、transition單位都是**筆**，無警報為None。`alarm_rate`以所有trial為分母；`correct_rate`同樣包含未警報；mean_latency／mean_transition只平均已警報trial，不是所有trial平均。兩劇本各自匯總，不把它們當新獨立採集。

預期A以gradual、B以sudden為主且兩者有警報；程式沒有獨立資料成功門檻。歷史完整數字見[固定原紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/experiments/exp3_trend.md#已記錄的實測)。劇本成功支持這個分數／規則辨識抽樣節奏，不支持實體故障機制、每小時誤報、真實警報秒數、RUL或損壞百分比。

## 方法來源與程式

Roberts（1959），*Control Chart Tests Based on Geometric Moving Averages*，Technometrics1(3),239–250，[DOI](https://doi.org/10.1080/00401706.1959.10489860)，支持EWMA遞迴。Page（1954），*Continuous Inspection Schemes*，Biometrika41(1/2),100–115，[DOI](https://doi.org/10.1093/biomet/41.1-2.100)；既有[固定引用查證](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/issue_delivery_20261008/citation_followup.md)仍列Page全文未取得，不能宣稱整套控制圖已原文驗證。0.08／0.2／0.5／12及劇本是Lineage操作約定，非上述論文為此資料保證的參數。

[exp3](../../src/experiments/exp3_trend.py)產生劇本、管理trial與writer；[core/trend](../../src/core/trend.py)實作旗標EWMA／CUSUM；[web/live](../../src/web/live.py)重用劇本；[test_stream_integration](../../src/tests/test_stream_integration.py)與[test_web_experiments](../../src/tests/test_web_experiments.py)檢查編排。實驗八另用SessionTrajectoryMonitor，不是同一趨勢演算法。

兩套趨勢的輸入與用途見[兩套趨勢](exp8_health.md#兩套趨勢)；EWMA／CUSUM 的引用核對與未確認項見[來源附件](../../reports/Andy_20261008_議題交付/citation_followup.md)。

串流停止、錯誤復原與模型有效狀態見[Web 串流生命週期](README.md#web-串流生命週期)及[共用模型生命週期](README.md#共用模型生命週期)。當次串流整合來源與配對證據見[固定交付紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/integration_20261008/integration_report.md)。
