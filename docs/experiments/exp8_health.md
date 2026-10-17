# 實驗八：相對健康指數、監測與封存重算

實驗八有四個入口，共用 `src/experiments/health/`。

| 入口 | 程式 | 做什麼 |
|---|---|---|
| 基準 8-1 | `src/experiments/health_index_benchmark.py` | 一個 seed、一種方法，把開集分數映成 0～1 |
| 矩陣 8-2 | `src/experiments/health_index_matrix.py` | 3 seed × 2 方法，保存六個 run、54 列 |
| 逐窗 8-3 | `src/experiments/health_monitor.py` | 單一工況、單一配置，逐筆看指數與告警 |
| 封存重算 | `src/experiments/health_index_aggregate.py` | 只讀舊結果，重算平均並逐欄比對 |

Web 實驗頁的 8-1、8-2、8-3 都指到本頁。8-2 讀 `reports/exp8_health_index_results/`，不讀這次新的 output 根。

## 資料能力

正式資料目前到相對健康基準、開集未知、相對健康指數、依筆數的監測與告警。有 motor×RPM 工況與 105 維 clean features，沒有可靠 timestamp、實體 motor ID、session ID、failure endpoint、維修事件、物理 fault cause 或正式嚴重度標籤。本頁不把指數說成損傷百分比、真實故障類型、真實嚴重階段或剩餘壽命。

本機用量見 [六工況總表](README.md#資料用量與分母怎麼讀)：缺 T2，不能把六工況補成封存的九工況。封存包是另一個資料版本，指紋 `81c9192476ecd1e23a17b3ed5a343dcd50cdc2e5b7e0cc5466e9162941898228`。T1／T2／T3 是不同馬達，不能串成一條壽命。

## 模組資料流

`CalibratedHealthIndex.fit()` 在 `src/experiments/health/index.py` 只用 known train 擬合 RobustScaler 與 `core.openset` factory。known calibration 設偵測門檻和健康量尺，cal 標籤不得超出 train 已知類別。預設 Mahalanobis–Ledoit–Wolf，k-NN 可切換。

`HealthIndexCalibrator` 把 cal 分數第 10 百分位映到 1.0、第 95 百分位映到 0.0，中間線性、兩端截斷。錨點相等時程式加極小間距。`score > 1` 仍是未知判定。公式 `h=clip(1−(s−a)/(b−a),0,1)`：s 是偏離分數，a／b 是上述兩個錨點。`degradation_score = 1 - health_index`。偵測器與映射共用同一份 cal。

| 模組 | 用途／輸出 |
|---|---|
| index.py | 特徵轉為 score、known prediction 與 HealthMonitoringResult |
| calibration.py | known calibration 到相對 health 的單調映射 |
| severity.py | health 的 0.8／0.5／0.2 界線，沒有物理損傷真值 |
| diagnosis.py | 拒絕為 unknown，已知但沒有 taxonomy 時為 uncertain |
| schema.py | binary／health／full 欄位與合法值 |
| trajectory.py | 按 motor_id 與 session_id 分開保存平滑、斜率、告警與變化點 |
| evaluation.py | 開集及軌跡／群組／順序檢查；函數存在不等於每個 runner 都使用 |
| config.py、interfaces.py | 設定及 Protocol 介面 |

`prediction_confidence=abs(score-1)/(1+abs(score-1))` 是判定線距離代理。`estimated_rul=None`、`rul_available=False`。health=0 不表示壽命用盡。

嚴重度政策 `severity_basis=relative_calibrated`：health `>=0.8` 為 healthy、`>=0.5` 為 early_warning、`>=0.2` 為 degraded，其餘 critical。開集拒絕輸出 `fault_type=unknown`；已知但沒有版本化 fault taxonomy 時輸出 `fault_type=uncertain`。

## 基準

流程：train 建偏離模型，cal 定距離線和兩個健康錨點，test 只 predict。T1／8000rpm 的健康池 313 列是 187 train／62 cal／64 holdout；同工況故障池 2736 列只測試。數量依 [用量總表](README.md#資料用量與分母怎麼讀)，本頁沒有新 fit。

排序 motor／rpm 後，一個 `default_rng(seed=42)` 連續把各健康池洗牌 60/20/20。train 只 fit RobustScaler 與 LW 中心／共變異數，或 k=5 的 k-NN 參考。cal 定 confidence 0.95 距離線，並把同一 cal 的正規化分數交給校準器。unknown 不 fit、不選參、不校準。legacy 因用 train 定線，基準入口拒絕。CLI 預設要求九工況；`--allow-partial` 只放寬數量。

`rows` 逐工況記筆數、方法、seed、原 score 與健康分布。組內平均的 std 用 ddof=0。`health_gap_known_minus_unknown` 是兩組健康平均差。`open_set_accuracy`、AUROC、AUPR、unknown recall／F1 用原 score。benchmark 只報 health `>=0.8` 與 `<0.2` 的比例。未知全貼 0 只說明映射超過 critical 錨點，排不出故障輕重。

CLI 經 `core.logger.setup_run("exp8_health_index_benchmark")` 初始化。日誌寫 `logs/exp8_health_index_benchmark/{ts}.log`，完整結果寫 `output/exp8_health_index_benchmark/{ts}/` 的 `result.json` 與 `rows.csv`，另有 `environment.json`。stdout 只印 status、列數、方法、seed 與 output_dir。結果是逐工況彙總，不含逐窗預測。Web 8-1 允許部分工況，完成後由 `ExperimentRunner` 寫到 `output/web_server/{ts}/experiments/`。歷史九工況結果見 [固定封存包](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/1fa9431bb7b86959f29540d07b2b9290ab39ce42/reports/exp8_health_index_results)。

封存摘要（9 工況 × seed 42／123／2026 × 兩方法，54 列；健康 train／cal 只用 `8screws`）：

| method | health gap known−unknown | Open Set accuracy | AUROC | unknown recall |
|---|---:|---:|---:|---:|
| Mahalanobis + Ledoit–Wolf | 0.583144 ± 0.073301 | 0.998849 ± 0.000887 | 1.000000 | 1.000000 |
| k-NN | 0.601657 ± 0.071637 | 0.998842 ± 0.000937 | 1.000000 | 1.000000 |

k-NN 的 health gap 高 0.018513，Mahalanobis accuracy 高 0.000007。重算時這兩欄有 0.018513 對 0.018512 的差異，見下方封存重算。event-level recall、每小時誤報、偵測延遲沒有時間軸，回報 unavailable。

```bash
uv run --locked python -m experiments.health_index_benchmark --help
uv run --locked python -m experiments.health_index_benchmark --data-root data/formal_local --seed 42 --openset-method mahalanobis --method ledoit_wolf --confidence 0.95
uv run --locked python -m experiments.health_index_benchmark --data-root data/formal_local --seed 42 --openset-method knn --knn-neighbors 5
```

API：`run(data_root, seed=42, openset_method="mahalanobis", mahalanobis_method="ledoit_wolf", confidence=0.95, knn_neighbors=5, require_nine=True)`。

## 矩陣

矩陣不另學健康量尺。預設 seeds 42／123／2026、methods mahalanobis／knn，`run_id` 如 `seed_42_mahalanobis`。每個 seed×method 呼叫一次基準，共 6 次、54 列。實驗六的 54 是 run 目錄數，這裡 54 是工況列。confidence 0.95、LW 與 k=5 沿基準預設。

CLI 經 `core.logger.setup_run("exp8_health_index_matrix")` 初始化，日誌寫 `logs/exp8_health_index_matrix/{ts}.log`。不給 `--output-root` 時結果寫本次的 `output/exp8_health_index_matrix/{ts}/`。要續跑既有輸出根才給 `--output-root`，不要指向 `reports/exp8_health_index_results/`。

```bash
uv run --locked python -m experiments.health_index_matrix --help
uv run --locked python -m experiments.health_index_matrix --data-root data/formal_local --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn
uv run --locked python -m experiments.health_index_matrix --data-root data/formal_local --output-root output/exp8_health_index_matrix/2026-10-17-10-00-00
```

API：`run_matrix(data_root, output_root, seeds=(42, 123, 2026), methods=("mahalanobis", "knn"), resume=True)`。`data_root` 與 `output_root` 都必須明確給。沒有 `run(pools)`，也沒有 Web 重跑入口。

`resume` 只查 JSON 的 status、seed、method 及 `formal_condition_count==9`。它不核對 rows 長度、唯一工況、資料指紋、參數、CSV 或來源 SHA。`--no-resume` 覆寫同名 JSON／CSV。錯誤寫入 manifest 後繼續，最後 incomplete 退出碼 2。

輸出在該次根目錄：`matrix_manifest.json`、`seed_{seed}/{method}.json` 與同名 csv。每個 run 的完成、續跑或失敗原因寫進 `logs/exp8_health_index_matrix/{ts}.log`。此模組不自動產生 `aggregate_summary.json`。Web 8-2 讀封存，不讀這個新根。

## 逐窗監測

選一個 motor／RPM 與 config。不論 config 是什麼，永遠只用健康 train 擬合、健康 cal 定 0.95 距離線與 10%／95% 錨點。健康重播用該配置 holdout 的洗牌順序；故障重播用該配置全池原列順序的前段，`max_windows` 只截斷。T1／8000rpm 健康重播最多 64 列，5screws 故障池 212 列。這些順序不是已確認的採集時間。

legacy 可由此 CLI 傳入，會回到 train 定線與 label=-1，不作公平比較。每次 `run` 重新 fit，沒有持久模型。

有 `motor_id` 與 `session_id` 才累積該鍵的歷史。缺身份時即使 `--allow-no-identity` 也不累積時間狀態；`require_identity=True` 另標 `data_quality=insufficient`。自填 ID 只隔離軟體狀態。

先取最近兩筆 raw health 加當前值的 median，再 EWMA `alpha=0.2`，最多留 120 筆。滿 5 筆對平滑歷史做線性擬合，`degradation_rate` 為負斜率，單位是健康指數／筆。大於 0.005 為 worsening，小於 −0.005 為 recovering，其餘 stable。前四筆 `insufficient_history`。

normal 下 smoothed `<0.5` 為 warning 候選、`<0.2` 為 critical 候選，連續 3 筆才進入該級。warning 以 `>=0.6` 連續 3 筆解除，critical 以 `>=0.3` 連續 3 筆解除。warning 狀態目前沒有再升 critical 的轉移。前值 smoothed 減當前 median `>=0.15` 連續兩筆記 confirmed 變點，一筆記 candidate。

## 兩套趨勢

| | `core.trend.TrendMonitor` | `experiments.health.trajectory.SessionTrajectoryMonitor` |
|---|---|---|
| 呼叫端 | exp3、`web/live` | `health_monitor` |
| 輸入 | 開集超線旗標 | 相對 health |
| 平滑 | EWMA alpha 0.08 | 3 筆中位數後 EWMA alpha 0.2 |
| 判定 | 中間帶 [0.2, 0.5)、警報 0.5，停留 ≤12 筆為突發 | warning 進入 0.5／解除 0.6、critical 進入 0.2／解除 0.3，進出連續 3 筆 |
| 身分 | 不按 motor 分 session | 按 motor_id、session_id 分組，缺身分回不足狀態 |

`TrajectoryConfig` 另有 `history_window=120`、`min_history=5`、`stable_slope=0.005`、`change_point_delta=0.15`／連續 2 筆。

```bash
uv run --locked python -m experiments.health_monitor --help
uv run --locked python -m experiments.health_monitor --data-root data/formal_local --motor T1 --rpm 8000rpm --config 8screws --openset-method mahalanobis --method ledoit_wolf --motor-id demo-T1 --session-id demo-001 --max-windows 20 --output-mode full
uv run --locked python -m experiments.health_monitor --data-root data/formal_local --motor T1 --rpm 8000rpm --config 5screws --openset-method knn --knn-neighbors 5 --max-windows 5 --output-mode binary --allow-no-identity
```

`motor`／`rpm` 必填，rpm 可寫 `8000` 或 `8000rpm`。`config` 須存在，`max_windows` 須為正整數。API `run(...)` 回傳逐窗列表，`iter_run` 先 yield fitted 再 yield window。CLI 先收集完再逐行印 JSON。`full` 含全部欄位，`health` 含指數與狀態，`binary` 只有 `is_fault`。CLI 經 `core.logger.setup_run("exp8_health_monitor")` 初始化：日誌寫 `logs/exp8_health_monitor/{ts}.log`，逐窗結果依 output-mode 寫 `output/exp8_health_monitor/{ts}/windows.jsonl`，一行一窗，與 stdout 相同，另有 `environment.json`。Web 8-3 可批次或 SSE；停止後沒有 done 就不算完整。展示用的 session ID 不能當採集身分。

`window_index` 從 0 起。`health_index` 是平滑後指數，`raw_health_index` 是單窗映射。`is_fault` 不由平滑告警決定。`timestamp` 固定 None。故障指數飽和在 0 時，斜率 0 不表示馬達健康。

## 封存重算

這個入口不載入正式資料，不 fit、不校準、不 predict。它讀 `reports/exp8_health_index_results/` 的 manifest、六個 JSON、六個 CSV、舊彙總與 README，共 16 檔，再重算七欄平均並與封存逐欄比較。

七欄：`health_gap_known_minus_unknown`、`open_set_accuracy`、`auroc`、`unknown_recall`、`unknown_f1`、`known_health_mean`、`unknown_health_mean`。全域每個 method 的 27 列等權（九工況 × 三 seed）；逐工況是 method×dataset 的三個 seed 等權。mean 為 sum/n，std 用 ddof=1。JSON 平均／std round 6。比對 JSON 的 atol=5e-7、rtol=0；CSV 的 atol=1e-12、rtol=1e-10。

退出碼 0 是 MATCH，3 是算完但 DIFFERENT，2 是來源或 schema 失敗。3 要保留差異。新矩陣根的 SHA 對不上這 16 檔時不能套用同一份鎖檔。

```bash
uv run --locked python -m experiments.health_index_aggregate --help
uv run --locked python -m experiments.health_index_aggregate --source-root reports/exp8_health_index_results --contract src/experiments/exp8_health_index_aggregate_contract.json
```

`run` 經 `setup_run` 寫 `logs/exp8_health_index_aggregate/{ts}.log` 與 `output/exp8_health_index_aggregate/{ts}/`：`aggregate_summary.json`、`aggregate_by_condition.csv`、`field_differences.csv`、`source_inventory.json`、`recomputation_audit.json`。輸出目錄與來源目錄禁止互相包含。預期六 run、54 列、18 組摘要。封存與重算有兩欄相差 0.000001（0.018513 對 0.018512），退出碼 3。

[exp8_health_index_aggregate_contract.json](../../src/experiments/exp8_health_index_aggregate_contract.json) 列出這 16 個來源檔的 SHA256（Windows CRLF 與 Git LF 各一組）、資料指紋、seed、方法與容差。`health_index_aggregate.py` 只接受這份鎖裡的雜湊，不把空白正規化後當成同一檔。

## reports與預期成果

`reports/exp8_health_index_results/` 是唯讀歷史結果：三 seed × 兩方法的 JSON／CSV、`matrix_manifest` 與歷史 aggregate。每 run 九工況列不是九顆新馬達。同工況 healthy-only 結果不取代跨馬達多類研究。

三支健康 CLI 與封存重算都經 `core.logger.setup_run()` 初始化。日誌寫 `logs/exp8_health_index_benchmark/`、`logs/exp8_health_index_matrix/`、`logs/exp8_health_monitor/`、`logs/exp8_health_index_aggregate/`，結果寫同名的 `output/{program}/{ts}/`。矩陣不給 `--output-root` 就寫本次 output 目錄；不要把 `--output-root` 指向封存的 `reports/exp8_health_index_results/`。

預期看到相對 health、未知判定與同一次 session 內的趨勢。

## 方法來源與程式碼

- P. C. Mahalanobis（1936），*On the Generalised Distance in Statistics*，Proceedings of the National Institute of Sciences of India 2(1), 49–55；重印 DOI [10.1007/s13171-019-00164-5](https://doi.org/10.1007/s13171-019-00164-5)。
- O. Ledoit、M. Wolf（2004），*A well-conditioned estimator for large-dimensional covariance matrices*，Journal of Multivariate Analysis 88(2), 365–411；DOI [10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4)。
- T. Cover、P. Hart（1967），*Nearest Neighbor Pattern Classification*，IEEE Transactions on Information Theory 13(1), 21–27；DOI [10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)。
- S. W. Roberts（1959），*Control Chart Tests Based on Geometric Moving Averages*，Technometrics 1(3), 239–250；DOI [10.1080/00401706.1959.10489860](https://doi.org/10.1080/00401706.1959.10489860)。實驗三的 EWMA alpha 0.08 是另一套，見 [實驗三](exp3_trend.md)。

10%／95% 錨點、嚴重度界線、六 run 命名與 resume 是本專案的操作約定。程式與測試：

| 角色 | 路徑 |
|---|---|
| 基準 | `src/experiments/health_index_benchmark.py`、`src/tests/test_health_benchmark.py`、`src/tests/test_health_index.py` |
| 矩陣 | `src/experiments/health_index_matrix.py`、`src/tests/test_health_matrix.py` |
| 逐窗 | `src/experiments/health_monitor.py`、`src/experiments/health/trajectory.py`、`src/tests/test_health_monitor.py`、`src/tests/test_health_trajectory.py`、`src/tests/test_health_schema.py` |
| 重算 | `src/experiments/health_index_aggregate.py`、`src/tests/test_health_index_aggregate.py`、`src/tests/exp8_aggregate_validation.py` |
| Web | `src/web/experiments.py` |

105 維載入、未知單位與未去重見 [實驗一](exp1_cold_start.md#資料與載入)。串流停止見 [Web 串流生命週期](README.md#web-串流生命週期)。
