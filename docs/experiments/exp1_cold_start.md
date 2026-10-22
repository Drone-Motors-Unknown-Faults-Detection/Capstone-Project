# 實驗一：冷啟動未知偵測

只拿健康資料建立警戒線，能不能把未見過的螺絲配置抓出來？這份實驗先量健康誤報，再量每個未知配置的召回與分數排序。它建立後續未知學習、趨勢與跨工況比較的健康基準；不訓練九種故障名稱的分類器。

## 先用自己的話說明

最小流程：讀105維特徵 → 只切健康池 → train建立基準 → cal決定警戒線 → healthy holdout及未知全池打分 → 分開報誤報與未知召回。先理解「健康長什麼樣」與「距離多遠才拒絕」是兩步；完整公式見[共通原理](README.md#共通原理先問模型拿什麼比較)。

示意而非實測：假設校準界線q=2.4，新資料距離d=3，s=3/2.4=1.25，所以判未知。這只表示超出健康範圍，不能回答哪顆螺絲鬆了或損壞25%。

### 資料角色用量表

以下是T1／8000rpm、seed42的切分數量，與[實驗一輸出](../../output/exp1_cold_start/2026-10-23-01-30-51/results.csv)的樣本數一致。九工況逐份數量見[分母總表](README.md#資料用量與分母怎麼讀)。

| 資料角色 | 馬達／轉速／類別 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| train | T1／8000rpm／健康 | 187 | 187/313=59.74%健康池 | scaler／中心／共變異數或近鄰參考／PCA | 與cal、holdout索引互斥 |
| calibration | 同上 | 62 | 62/313=19.81%健康池 | LW／k-NN的q95 | 不fit參考；legacy不使用此門檻來源 |
| 健康測試 | 同上 | 64 | 64/313=20.45%健康池 | 誤報率 | 列級互斥，錄製獨立UNKNOWN |
| 未知測試 | 同工況其餘九配置 | 2736 | 2736/2736=100%故障池 | 未知召回與AUROC | 沒有進入初始fit／cal |

沒有validation或超參數選擇。187+62+64=313；313+2736=3049。seed123／2026仍是相同數量、不同索引，不是新增獨立馬達。這裡只用T1一個工況，不能據此宣稱三馬達結果。

### 發表與理解檢查

口頭順序：「我們只有健康基準，先用187列建參考，再用62列設門檻，64列量誤報，2736故障列量未知拒絕。分數表示偏離，不是故障原因。結果要看本次產物，這張用量表沒有產生新研究分數。」真正結果連結與讀法見下方輸出節。

- 為什麼不用train定門檻？同一資料既估分布又驗自己會偏樂觀；LW／k-NN把cal留開，legacy對照有這個限制。
- 測試是否參與模型？本實驗列級未參與；但raw窗口／採集群組仍未知，不能推成完整獨立盲測。
- 抓到九配置是否等於分類九故障？否，這裡只學健康，拒絕它們並不知道配置名稱。
- 組員需親自重述：模型學了什麼、62列做什麼、健康誤報與未知召回各除以多少。AI只核對內容，沒有替組員宣稱已理解。

## 資料與載入

從儲存庫根目錄執行。[src/core/data.py](../../src/core/data.py) 掃描
`data/Step-*/myfeature/{Motor}/{RPM}/{Screws}/*_Group_feature_data_clean.csv`。
每次選一個馬達／轉速目錄，預設T1／8000rpm；可用組合以實際檔案為準。

每列是原始論文前處理管線留下的105維特徵向量，不是本程式接收的原始10,000Hz訊號。健康配置為`8screws`，其餘目錄名是螺絲數量／位置配置，不能直接當九種已驗證的物理故障原因。T1／T2／T3是不同馬達個體，不能串成同一顆馬達的壽命歷程。來源說明見[固定採集文件](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/afcfcc419dab3103a86a8f95601d3af85890eb38/docs/Experiments_Guide.md#L218)。

載入順序為：依配置目錄排序 → 只收正式 CSV → header 必須是已登錄的完整 105 欄順序 → 非有限值、錯欄或空配置整檔拒絕。不刪列，也不依「剛好 105 個數字欄」放行。沒有去重、原始訊號再清理或欄位單位轉換。欄名與拒絕政策見[正式特徵載入規則](../formal_feature_schema_contract.md)（#44 已實作並關閉）。單位、session、stride 與 raw window 仍是 UNKNOWN，不能由這個 loader 補出。

載入後`pools[config]`是「資料列數×105」陣列。筆數依這次載入結果與輸出`n`查核，本手冊不預填固定總數。

## 如何切分與建立基準

[OpenSetMonitor](../../src/core/monitor.py)只註冊健康配置。`make_split`用`numpy.random.default_rng(seed)`洗牌資料列；預設seed42。訓練筆數為`int(n*0.6)`，校準為`max(1,int(n*0.2))`，剩餘是holdout，比例有整數捨入。

| 用途 | 資料 | 實際動作 |
|---|---|---|
| train | 健康池約60% | fit RobustScaler、偵測器參考與顯示用PCA |
| calibration | 健康池約20% | LW或k-NN用來定距離警戒線；不fit scaler／參考樣本 |
| test／holdout | 剩餘健康列 | 計算誤報，不參與上述fit或LW／k-NN校準 |
| unknown test | 所有未註冊配置的全部列 | 只打分與算指標，不fit、不選參、不校準 |

這是同馬達、同轉速的列隨機切分，沒有獨立session或raw-window guard；不能當跨錄製盲測。少量健康列可能造成空train／holdout，不應把空集合的NaN彙總當成功。

模型學的是健康特徵的位置與散布。RobustScaler只在train學每欄中位數與四分位距；其餘集合只transform。預設Mahalanobis-Ledoit–Wolf從縮放後train估中心／收縮共變異數，距離除以calibration的0.95分位數。改k-NN則保存train近鄰索引，取平均歐氏距離；k預設5，樣本不足時取較小的有效k。多已知類時分數取各類正規化距離的最小值；本實驗初始只有健康一類。`score > 1`才判未知，等於1仍接受。

**legacy例外**：[src/core/mahalanobis.py](../../src/core/mahalanobis.py)的`legacy`刻意用train本身的距離定門檻，而非獨立calibration；不能把它寫成與LW相同的校準規則。它的類別label為-1，不適合作為多類配置辨識的成功證據。本實驗只比較score。OAS、MCD是可選共變異數方法；MCD另fit train-only PCA。顯示用2維PCA固定random_state0，不是主要偵測特徵，分數仍走105維（MCD除外）。這些統計／近鄰方法沒有epoch、loss或optimizer。

## 實驗怎麼跑與怎麼使用

先依[環境政策](../runtime_policy.md)準備Python3.10.x、uv及`uv.lock`環境，不重建既有venv；資料由原始論文產出，本repo唯讀。以下在repo根目錄執行：

```bash
uv run --locked python -m experiments.exp1_cold_start --help
uv run --locked python -m experiments.exp1_cold_start --data-root data --motor T1 --rpm 8000rpm --seed 42
uv run --locked python -m experiments.exp1_cold_start --data-root data --motor T1 --rpm 8000rpm --seed 42 --openset-method knn --knn-neighbors 5
```

兩方法比較時固定來源、馬達、RPM、seed、confidence；不要挑較好seed再報提升。`--method`預設ledoit_wolf，只控制Mahalanobis；k-NN忽略它。`--confidence`預設0.95，不能用unknown結果掃門檻再稱獨立驗證。

執行依序為健康fit → 健康holdout打分 → 每個unknown配置全池打分 → 計算逐配置及等權macro指標 → 存CSV／JSON／圖。Python API是`experiments.exp1_cold_start.run(pools, seed=42)`，回傳dict；API本身不寫CLI輸出。`iter_run`回傳fitted、scores、row、result事件，批次run把同一generator跑完。

Web實際入口為`uv run --locked python -m web.server --bind-address 127.0.0.1 --data-root data --port 8600`。開http://localhost:8600，切「實驗一」，選資料集、Open Set方法與seed，按「▶ 執行」。要逐筆看就選速度、按「⏵ 邊跑邊畫」；「■ 停止」關閉串流，不代表保存完整done結果，已在執行緒運算的一步可能仍在完成。待鎖釋放再重試。批次沒有中途取消按鈕；切頁不能當作運算取消。每次重跑重新fit，不載入長期模型。

找不到資料集時核對`--data-root`及實際RPM目錄；缺健康或維度錯誤時查來源CSV，不改欄湊105。尚未擬合的API會丟RuntimeError，須先成功fit_initial。重跑同秒可能共用setup_run目錄；不要並行覆寫，這項來源／輸出規則限制由[#46](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/46)追蹤，沒有虛構安全續跑選項。

## 輸出怎麼看

CLI writer是[main()](../../src/experiments/exp1_cold_start.py)：
`logs/exp1_cold_start/{時間戳}.log`及`output/exp1_cold_start/{時間戳}/`。
目錄含`environment.json`（setup_run記環境）、`results.csv`（逐配置）、`summary.json`（整體及model設定）、`detect_rates.png`（判未知比例）。Web由ExperimentRunner另存`output/web_server/{ts}/experiments/`的JSON，與CLI檔案不同。

| 欄位 | 讀法 |
|---|---|
| kind／n／flagged | healthy-holdout或unknown-fault、實際評估筆數、嚴格超線筆數 |
| detect_rate | 健康列是誤報率；未知列才是未知召回率，範圍0～1 |
| auroc | 未知分數能否排在健康前；0.5近隨機，1為本次樣本完全排序分離 |
| median_score | 無單位正規化偏離；不是故障百分比 |
| macro_detect_rate／macro_auroc | 各unknown配置等權平均，不按資料列數加權；資料集只有健康池時兩者為 null（#271） |
| model | train／cal／holdout數、方法、閾值與seed；不是已保存的模型權重 |

預期是未知召回／AUROC高且健康誤報低；程式沒有全域「可靠模型PASS」門檻。名義0.95分位數不保證test只有5%誤報。同工況的高分不是跨馬達已知故障分類。

## 方法來源與程式定位

- Mahalanobis（1936），*On the Generalised Distance in Statistics*，PNISI 2(1),49–55；[重印DOI](https://doi.org/10.1007/s13171-019-00164-5)。
- Ledoit與Wolf（2004），*A well-conditioned estimator for large-dimensional covariance matrices*，JMVA 88(2),365–411；[DOI](https://doi.org/10.1016/S0047-259X(03)00096-4)。
- Cover與Hart（1967），*Nearest Neighbor Pattern Classification*，IEEE TIT 13(1),21–27；[DOI](https://doi.org/10.1109/TIT.1967.1053964)。本repo的平均k距離／分位數拒絕規則是專案操作約定，不能說該論文提出本套校準管線。
- [RobustScaler官方說明](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.RobustScaler.html)。實際版本由uv.lock固定，不按網站最新版本推論現行行為。
- [src/core/runner.py](../../src/core/runner.py)負責資料定位／CLI；[src/core/openset.py](../../src/core/openset.py)是正式兩方法factory；Mahalanobis檔頭記原始論文來源，Lineage的monitor整合scaler、切分及擴張。
- [src/tests/test_openset.py](../../src/tests/test_openset.py)、[test_monitor_guard.py](../../src/tests/test_monitor_guard.py)、[test_stream_integration.py](../../src/tests/test_stream_integration.py)、[test_web_experiments.py](../../src/tests/test_web_experiments.py)核對共用方法／guard／串流與Web編排，不能證明raw來源獨立或現場準確率。

串流停止、錯誤復原與模型有效狀態見[Web 串流生命週期](README.md#web-串流生命週期)及[共用模型生命週期](README.md#共用模型生命週期)。當次串流整合來源與配對證據見[固定交付紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/integration_20261008/integration_report.md)。
