# 實驗八：把異常分數轉成相對健康指數

## 資料角色用量表

流程：train建偏離模型 → cal定距離線及兩個健康錨點 → test預測score與指數 → 看健康／未知的分離。指數`h=clip(1−(s−a)/(b−a),0,1)`中s是偏離score，a/b是健康cal的10%／95%分位；越偏離，h越低。這是Lineage量尺，不是物理損傷模型。

| 資料角色 | T1／8000rpm | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| train | 313健康列 | 187 | 187/313 | scaler及LW／k-NN參考 | 不含test |
| calibration | 同上 | 62 | 62/313 | 距離線、健康錨點 | 兩層校準共用這62列 |
| 健康test | 同上 | 64 | 64/313 | 健康指數分布、誤報 | 與train/cal互斥 |
| 未知test | 同工況fault | 2736 | 故障池100% | 指數分離、拒絕與排序 | 不fit／cal／選參 |

依[唯讀用量與六工況總表](README.md#資料用量與分母怎麼讀)查核，沒有新fit或健康指數結果；連續RNG的索引不與單工況monitor混用。本機缺T2，Web可跑部分工況，正式九工況驗收仍不足。沒有獨立validation。

## 發表與理解檢查

口頭主軸：「先量偏離，再用健康校準群的兩個位置把分數換成容易讀的0～1量尺。這裡的改善要看分離、誤報和未知召回，不能只看健康度漂亮。未知都貼0時，量尺已飽和，反而無法排序故障程度。」

- h=.2是否只剩20%壽命？不是，沒有壽命真值或物理校準。
- 兩份校準資料是否獨立？沒有，detector及映射共用同cal，須揭露依賴。
- 為何還報原score的AUROC？映射clip會丟掉排序細節，原score更能回答拒絕排序。
- 組員請解釋錨點、飽和與未知拒絕的差別；實際歷史分數仍查下方固定來源。

Open Set分數越大越偏離健康；本實驗把它轉成0～1的健康指數，方便讀圖。它衡量相對校準健康群的偏離，不能稱為損壞百分比、故障機率或剩餘壽命。多seed由[矩陣](exp8_health_index_matrix.md)保存，逐窗狀態見[監測](exp8_health_monitor.md)。

## 資料、切分與fit

來源是每個motor／RPM的105維clean池，資料列、未知單位、非有限列過濾及未去重限制見[實驗一](exp1_cold_start.md#資料與載入)。CLI預設要求九工況；--allow-partial只放寬數量，仍可能回status=completed，不能把部分工程測試當完整九格。T1／T2／T3是不同個體，不代表一條退化軌跡。

排序motor／rpm後，一個default_rng(seed=42)連續把各健康池列洗牌60/20/20。train只fit RobustScaler與LW中心／共變異數，或k=5的k-NN參考；cal只定confidence0.95距離線，並把同一cal的正規化分數交HealthIndexCalibrator。healthy holdout與其他配置全部列只predict與算指標，unknown不fit／選參／校準；legacy因train定線被拒絕。這不是獨立馬達train／cal／test。

健康錨點a是cal score第10百分位、critical錨點b是第95百分位；health=clip(1−(score−a)/(b−a),0,1)。若兩錨點相同，程式加極小間距；仍可能大量飽和。detector與健康映射**共用cal**，不是兩份獨立校準。沒有神經網路epoch。degradation_score=1−health。

## 怎麼執行與使用

在repo根目錄準備[uv環境](../runtime_policy.md)與資料：

```bash
uv run --locked python -m experiments.health_index_benchmark --help
uv run --locked python -m experiments.health_index_benchmark --data-root data/formal_local --seed 42 --openset-method mahalanobis --method ledoit_wolf --confidence 0.95
uv run --locked python -m experiments.health_index_benchmark --data-root data/formal_local --seed 42 --openset-method knn --knn-neighbors 5
```

API run(data_root,seed=42,openset_method="mahalanobis",mahalanobis_method="ledoit_wolf",confidence=0.95,knn_neighbors=5,require_nine=True)回傳九列dict。CLI經`core.logger.setup_run("exp8_health_index_benchmark")`初始化：日誌寫`logs/exp8_health_index_benchmark/{ts}.log`，完整結果寫`output/exp8_health_index_benchmark/{ts}/`的`result.json`（與API回傳的dict相同）與`rows.csv`（九列表），另有setup_run寫的`environment.json`。stdout只印status／列數／方法／seed與output_dir；結果是逐工況彙總，不含逐窗預測。

Web依[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)啟動，實驗八8-1選方法／seed後按「▶ 執行」，跑伺服器root；Web允許部分工況，須看列數。沒有中途取消、串流或續跑；8-3是另一個逐窗入口。Web由ExperimentRunner保存JSON到output/web_server/{ts}/experiments/。

缺九工況、healthy／unknown空、錨點非有限或維度不符時先查資料；不修改fault來配合校準。指紋沿用[正式版的限制](exp6_formal_benchmark.md#輸出怎麼讀)，沒有原始錄製／逐窗ID證明。

## 輸出怎麼看

rows逐工況記n_train／n_calibration／n_known_test／n_unknown_test、方法、seed、原score與健康分布。known_health_mean等是組內窗口平均；std用ddof=0。health_gap_known_minus_unknown是兩組健康平均差。health_effect_size用兩組population variance平均的平方根作分母，接近0則null，不能當經獨立採集驗證的效應量。

open_set_accuracy是score>1的健康／fault二元正確率；AUROC、AUPR、unknown recall／F1用原score，不用health反推。零分母F1設0；單真值類時排序指標不可得，但正常完整benchmark須有健康與fault。unknown_calibration_leakage=false是本流程宣告，不是全來源防洩漏稽核結果。

健康stage政策為≥0.8 healthy、≥0.5 early_warning、≥0.2 degraded、其餘critical；benchmark只報≥0.8及<0.2比例，不含逐窗stage。沒有ordinal損傷真值，rul_available=false。

預期健康平均高於unknown、health gap為正，數值在[0,1]；沒有預登錄可靠性門檻。unknown全貼0只能支持此映射超過critical錨點，無法排序不同故障嚴重度。完整歷史結果見[固定封存包](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/1fa9431bb7b86959f29540d07b2b9290ab39ce42/reports/exp8_health_index_results)，不把它當本輪重跑或多類fault-type結果。

## 方法來源與程式碼

LW與近鄰文獻見[實驗一](exp1_cold_start.md#方法來源與程式定位)；10%／95%錨點、stage與健康映射是Lineage操作約定。[benchmark](../../experiments/health_index_benchmark.py)管迴圈／摘要；[index](../../experiments/health/index.py)fit與predict；[calibration](../../experiments/health/calibration.py)映射；[evaluation](../../experiments/health/evaluation.py)指標；[test_health_benchmark](../../tests/test_health_benchmark.py)與[test_health_index](../../tests/test_health_index.py)核對known-only與輸出。Web編排在[web/experiments](../../web/experiments.py)，不另訓練模型。
