# 實驗八健康監測模組與結果讀法

本頁說明experiments/health/的輸入、計算與輸出，區分兩套趨勢程序。方法與CLI參數見 [benchmark](experiments/exp8_health_index_benchmark.md)、[matrix](experiments/exp8_health_index_matrix.md) 及 [monitor手冊](experiments/exp8_health_monitor.md)。

## 模組資料流

CalibratedHealthIndex.fit()在experiments/health/index.py只用known train擬合RobustScaler與core.openset factory偵測器；known calibration設detector門檻及健康量尺，cal標籤不得超出train已知類別。Mahalanobis-LW為預設，k-NN為可切換對照。

HealthIndexCalibrator以cal分數p10映射到1.0、預設p95映射到0.0，中間線性、兩端截斷；錨點相等時由程式保護。score > 1仍是未知判定，health只是相對校準群體的量尺。

| 模組 | 用途／輸出 |
|---|---|
| index.py | 特徵轉為score、known prediction與HealthMonitoringResult |
| calibration.py | known calibration到相對health的單調映射 |
| severity.py | health的0.8／0.5／0.2界線，沒有物理損傷真值 |
| diagnosis.py | 拒絕為unknown，已知但沒有taxonomy時為uncertain |
| schema.py | binary／health／full欄位與合法值 |
| trajectory.py | 按motor_id與session_id分開保存平滑、斜率、告警與變化點 |
| evaluation.py | 開集及軌跡／群組／順序檢查；函數存在不等於每個runner都使用 |
| config.py、interfaces.py | 設定及Protocol介面，不能據此宣稱新服務完成 |

prediction_confidence=abs(score-1)/(1+abs(score-1))是判定線距離代理，不是機率。estimated_rul=None、rul_available=False；health=0不表示壽命用盡或損壞100%。

## 兩套趨勢

| | core.trend.TrendMonitor | experiments.health.trajectory.SessionTrajectoryMonitor |
|---|---|---|
| 呼叫端 | exp3、web/live | health_monitor |
| 輸入 | 開集超線旗標 | 相對health |
| 平滑 | EWMA alpha0.08 | 3筆中位數後EWMA alpha0.2 |
| 判定 | 中間帶[0.2,0.5)、警報0.5，停留≤12筆為突發 | warning進入0.5／解除0.6、critical進入0.2／解除0.3，進出連續3筆 |
| 身分 | 不按motor分session | 按motor_id、session_id分組，缺身分回不足狀態 |

TrajectoryConfig另有history_window120、min_history5、stable_slope0.005、change_point_delta0.15／連續2筆。CSV順序或人工session代碼不證明真實採集時間，T1／T2／T3不能串成一顆馬達生命週期。

## CLI與輸出位置

```bash
uv run --locked python -m experiments.health_index_benchmark --help
uv run --locked python -m experiments.health_index_matrix --help
uv run --locked python -m experiments.health_monitor --help
```

三支health入口都經`core.logger.setup_run()`初始化：日誌寫`logs/exp8_health_index_benchmark/`、`logs/exp8_health_index_matrix/`、`logs/exp8_health_monitor/`，結果寫同名的`output/{program}/{ts}/`。matrix不給`--output-root`就寫本次output目錄；不要把`--output-root`指向封存的reports/exp8_health_index_results/。

health_monitor對healthy取未擬合holdout，其他配置依原順序取整池，與iter_run一致；這不使CSV視窗成為獨立採集，真實時間與重疊仍需證據。

## reports與預期成果

reports/exp8_health_index_results/是唯讀歷史結果：三seed×兩方法的JSON／CSV、matrix_manifest及歷史aggregate。每run九工況列不能當九顆新馬達，同工況healthy-only結果不可取代跨馬達多類研究。

experiments.health_index_aggregate只讀封存輸出重算與逐欄比較，寫新logs/output，詳見 [重算契約](experiments/exp8_health_index_aggregate.md)。不同值及不明歷史公式保留，不改舊aggregate遷就程式。

預期取得相對health、未知判定與session內趨勢，不能推論物理故障原因、損壞百分比、RUL或部署誤報保證。新邏輯放experiments/health/，入口放experiments/，輸出遵守AGENT慣例。

舊稽核、單次分數與交付紀錄見 [改寫前固定版本](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/791216cf5602c370091fa516264f1ce6aaad6ab1/docs/health_and_reports.md)，缺口由既有issues追蹤；本頁不再維護日期狀態與測試次數。
