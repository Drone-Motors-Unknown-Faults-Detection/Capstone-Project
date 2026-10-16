# 實驗八逐窗監測：看健康指數與告警狀態

## 資料角色用量表

流程：只用健康建立偏離模型與指數 → 選一配置的可用列 → 逐窗映射 → session內median／EWMA → 趨勢與告警。這裡分析的是重播順序，不是已核實的退化時間；原score與平滑指數是不同欄位。

| 資料角色 | T1／8000rpm | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 健康train／cal | 健康池313列 | 187／62 | 各除以313 | detector與健康錨點 | cal共用，不進test |
| 健康重播 | config=8screws | min(max_windows,64) | 剩餘64/313可用；Web預設120則播64列 | 健康逐窗狀態 | 與fit/cal索引互斥 |
| 故障重播 | config=5screws | min(max_windows,212) | 故障池212列；Web預設播120列 | 未知逐窗狀態 | 不fit；取原列前段，非原始時間證據 |

依[唯讀用量](README.md#資料用量與分母怎麼讀)；無validation選參，未重跑監測分數。未知不是取健康holdout的64個索引，不能把兩者筆數當同一抽樣契約。兩配置各跑一次會建立不同操作狀態，不能接成同顆馬達生命週期。

## 發表與理解檢查

口頭主軸：「單窗score先換成相對健康指數，再平滑以減少跳動，最後在這次操作session內維持告警。表中每列可追到window_index，但timestamp沒有提供，因此警報延遲只能按筆，不是秒。」EWMA的輸入是健康指數，與實驗三的異常旗標不同。

- 指數一直0、趨勢stable是否健康？不是，可能故障分數都超出critical錨點而飽和。
- warning能保證再升critical嗎？現行分支缺這個轉移，不能宣稱完整升降級流程。
- 自填session ID是否補足採集證據？不能，它只隔離軟體狀態。
- 組員請分開解釋raw指數、平滑指數、is_fault及alarm_state；實際可驗證範圍見下方。

這個入口把CSV特徵逐列交給健康模型，再平滑指數、計算趨勢和告警。它展示同一操作session的狀態，不證明CSV具有連續物理時間；批次九工況指標看[benchmark](exp8_health_index_benchmark.md)。

## 資料與建立順序

選一個motor／RPM與config。來源、105維每列、缺值／未去重與未知單位見[實驗一](exp1_cold_start.md#資料與載入)。T1／T2／T3為不同個體；時間戳、raw錄製順序與視窗重疊未確認，不能把它們接成同一條退化曲線。

seed42洗牌healthy池60/20/20；不論--config選什麼，永遠只用healthy train fit RobustScaler與LW（或k-NN k5）參考，healthy cal定0.95距離線與10%／95%健康錨點。unknown不fit、不校準、不選參。健康映射細節見[benchmark](exp8_health_index_benchmark.md#資料切分與fit)；legacy可由此CLI傳入，會回到train距離與label=-1例外，不作公平正式比較。

串流healthy使用自身holdout的洗牌順序；fault使用該配置**全池原列順序**，不再套健康索引，max_windows只截前段。這些順序不是確認過的時間。每次run重新fit、建立SessionTrajectoryMonitor；沒有持久模型恢復。CalibratedHealthIndex是統計／近鄰與分位數映射，沒有epoch。

## 平滑、趨勢與告警怎麼算

有motor_id與session_id才累積該鍵的歷史，缺身份即使allow-no-identity也**不會累積時間狀態**；require_identity=True另標data_quality=insufficient。自填展示ID不是新增採集證據。

先取最近兩筆raw health加當前值的median，再EWMA alpha0.2，最多留120筆。滿5筆對平滑history線性擬合，degradation_rate=負斜率，單位為健康指數／筆；>0.005 worsening、<−0.005 recovering，其餘stable。前四筆insufficient_history。

normal下smoothed<0.5為warning候選、<0.2為critical候選，連續3筆候選才進入當前級別；warning以≥0.6連續3筆解除，critical以≥0.3連續3筆解除為warning或normal。**目前warning狀態分支沒有再升critical的轉移**，不能描述成完整雙向分級告警；這是待修程式限制，本輪不改行為。前值smoothed−當前median≥0.15連續兩筆記confirmed變點，一筆candidate；不代表物理故障事件已確認。

## 怎麼操作

repo根目錄準備[uv環境](../runtime_policy.md)與資料：

```bash
uv run --locked python -m experiments.health_monitor --help
uv run --locked python -m experiments.health_monitor --data-root data/formal_local --motor T1 --rpm 8000rpm --config 8screws --openset-method mahalanobis --method ledoit_wolf --motor-id demo-T1 --session-id demo-001 --max-windows 20 --output-mode full
uv run --locked python -m experiments.health_monitor --data-root data/formal_local --motor T1 --rpm 8000rpm --config 5screws --openset-method knn --knn-neighbors 5 --max-windows 5 --output-mode binary --allow-no-identity
```

motor／rpm必填，rpm可8000或8000rpm；config須實際存在，max_windows須正整數。API run(data_root,motor,rpm,config,...)回傳逐窗列表，iter_run先yield fitted再window；**CLI先把run收集完，才逐行印JSON**，並非計算到一筆即stdout一筆。full含全部欄位；health含指數／狀態等；binary僅is_fault。

CLI經`core.logger.setup_run("exp8_health_monitor")`初始化：日誌寫`logs/exp8_health_monitor/{ts}.log`，逐窗結果依所選output-mode寫`output/exp8_health_monitor/{ts}/windows.jsonl`，一行一窗，與stdout印出的內容逐行相同；另有setup_run寫的`environment.json`。Web依[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)啟動，8-3選工況／配置／方法／seed／筆數，按「▶ 執行」或「⏵ 邊跑邊畫」看逐窗；Web使用web-demo及資料集-config的展示session ID，不能當採集身分證明。「■ 停止」取消後續SSE，未done不算完整；完成JSON由web_server保存，重跑重新開始。

## 輸出怎麼讀

window_index從0開始；openset_score>1表示偏離已知，health_index為平滑後指數，raw_health_index為單窗映射；is_fault與is_unknown_fault不是由平滑告警決定。severity_stage是相對展示政策，不是已標定損傷。prediction_confidence=|score−1|/(1+|score−1|)，uncertainty=1−它，**不是校準機率**。

timestamp固定None，estimated_rul=None／rul_available=false。不能算每小時誤報、秒級延遲或剩餘壽命。故障指數若飽和0，持平斜率0不表示馬達健康；要分開看is_fault、raw／smoothed、alarm與history。

預期健康holdout誤報少、unknown指數較低；工程驗收為取樣正確、模式欄位與API／Web一致，不是通用可靠性門檻。缺healthy、配置不存在或工況不唯一時核對來源，不借別的motor補資料。歷史批次分離度查[固定包](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/1fa9431bb7b86959f29540d07b2b9290ab39ce42/reports/exp8_health_index_results)，不是本CLI逐窗長期驗證。

## 方法來源與程式碼

LW／近鄰來源見[實驗一](exp1_cold_start.md#方法來源與程式定位)；Roberts（1959），*Control Chart Tests Based on Geometric Moving Averages*，Technometrics1,239–250，[DOI](https://doi.org/10.1080/00401706.1959.10489860)支持EWMA遞迴。median、alpha0.2與告警政策是Lineage約定，與[實驗三](exp3_trend.md)alpha0.08不同。

[health_monitor](../../experiments/health_monitor.py)取樣／CLI；[trajectory](../../experiments/health/trajectory.py)狀態；[index](../../experiments/health/index.py)單窗模型；[schema](../../experiments/health/schema.py)模式；[web/experiments](../../web/experiments.py)編排；[test_health_monitor](../../tests/test_health_monitor.py)、[test_health_trajectory](../../tests/test_health_trajectory.py)與[test_health_schema](../../tests/test_health_schema.py)驗證工程行為，不提供實體退化真值。

共用模組與資料流見[模組資料流](../health_and_reports.md#模組資料流)；封存結果的讀法見[reports與預期成果](../health_and_reports.md#reports與預期成果)。串流停止與復原見[Web 串流生命週期](README.md#web-串流生命週期)。
