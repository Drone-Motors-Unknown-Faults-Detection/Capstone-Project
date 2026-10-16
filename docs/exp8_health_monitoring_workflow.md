# 實驗八：健康監測工作流程

> 實驗八（exp8）的總覽。程式在 `src/experiments/health_index_benchmark.py`、`src/experiments/health_index_matrix.py`、`src/experiments/health_monitor.py` 與 `src/experiments/health/` 套件；逐支技術報告在 `docs/experiments/exp8_*.md`，全部檔案位置見 [experiments/README.md](experiments/README.md#各實驗的檔案位置)。

## 資料能力與層級

目前正式資料最高支援 Level A：相對健康基準、Open Set unknown、相對 health index、
時間監測與告警。資料有 9 個 motor×RPM 工況與 105 維 clean features，但沒有可靠
timestamp、實體 motor ID、session ID、failure endpoint、maintenance event、物理 fault
cause label 或正式 severity label。因此本版本不宣稱實體損傷百分比、真實 fault type、
真實嚴重階段或 RUL。

## 輸出定義

`health_index` 在 `[0,1]`，由 known calibration 的第 10 百分位（健康錨點）與第 95
百分位（critical 錨點）將 Open Set score 做單調映射；分數越大越偏離健康，健康度越低。
`degradation_score = 1 - health_index`，不是壽命百分比。`score > 1` 仍是未知判定。

目前 severity 是 `severity_basis=relative_calibrated`：health `>=.8` 為 healthy、`>=.5`
為 early_warning、`>=.2` 為 degraded，其餘 critical。沒有物理 severity labels，不可把
這些門檻當作真實損壞等級。

Open Set rejection 一律輸出 `fault_type=unknown`；已知但沒有版本化 fault taxonomy 時
輸出 `fault_type=uncertain`。RUL 固定為 `estimated_rul=null, rul_available=false`。

## 趨勢、change point 與告警

`SessionTrajectoryMonitor` 以 `(motor_id, session_id)` 分開保存歷史；沒有身份時回報
`data_quality=insufficient`，不串接不同馬達。它保留 raw/smoothed health，先做最近值
rolling median，再做 EWMA (`alpha=.2`)，至少 5 筆後以線性 slope 判定 stable/worsening/
recovering。warning/critical 進入與解除各需要連續 3 筆，解除門檻較寬（hysteresis）。
健康值持續下降超過 `.15` 且達 2 筆記為 `change_point_state=confirmed`；單筆為 candidate。

## 正式實驗結果

P9 已完成 9 工況 × 3 seeds（42/123/2026）× 2 methods，共 54 rows；每個 seed 已獨立
commit/push。健康 train/calibration 只使用 `8screws`，其餘設定只在 scoring 後評估。

| method | health gap known−unknown | Open Set accuracy | AUROC | unknown recall |
|---|---:|---:|---:|---:|
| Mahalanobis + Ledoit–Wolf | 0.583144 ± 0.073301 | 0.998849 ± 0.000887 | 1.000000 | 1.000000 |
| k-NN | 0.601657 ± 0.071637 | 0.998842 ± 0.000937 | 1.000000 | 1.000000 |

k-NN 的相對 health gap 高 `0.018513`，Mahalanobis accuracy 高 `0.000007`，差異很小，
不足以取代既有預設。所有 held-out unknown 在本校準設定下 health 都飽和到 `0.0`，
表示分得開，不表示已學會物理劣化排序。沒有 event labels/timebase，所以 event-level
recall、false alarms/hour、detection delay 只能回報 unavailable。

結果檔在 `reports/exp8_health_index_results/`：每個 seed 的 JSON/CSV、`matrix_manifest.json`、
`aggregate_summary.json` 與 `aggregate_by_condition.csv`。

## 方法來源

- P. C. Mahalanobis (1936), “On the Generalised Distance in Statistics,” *Proceedings of
  the National Institute of Sciences of India*, 2(1), 49–55；重印 DOI
  [10.1007/s13171-019-00164-5](https://doi.org/10.1007/s13171-019-00164-5)。
- O. Ledoit and M. Wolf (2004), “A well-conditioned estimator for large-dimensional covariance
  matrices,” *Journal of Multivariate Analysis*, 88(2), 365–411；DOI
  [10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4)。
- T. Cover and P. Hart (1967), “Nearest Neighbor Pattern Classification,” *IEEE Transactions
  on Information Theory*, 13(1), 21–27；DOI [10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)。
- S. W. Roberts (1959), “Control Chart Tests Based on Geometric Moving Averages,”
  *Technometrics*, 1(3), 239–250；DOI [10.1080/00401706.1959.10489860](https://doi.org/10.1080/00401706.1959.10489860)。

## 可直接複製的命令

```powershell
# 健康 train/cal + 單一正式 benchmark
.\venv\Scripts\python.exe -m experiments.health_index_benchmark --data-root data/formal_local --seed 42 --openset-method mahalanobis

# 完整 9×3×2 正式矩陣
.\venv\Scripts\python.exe -m experiments.health_index_matrix --data-root data/formal_local --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn

# holdout windows 的 full health/unknown/trajectory JSONL
.\venv\Scripts\python.exe -m experiments.health_monitor --data-root data/formal_local --motor T1 --rpm 8000rpm --config 1screws --openset-method mahalanobis --output-mode full --motor-id motor-001 --session-id session-001 --max-windows 20

# 舊 binary 介面
.\venv\Scripts\python.exe -m experiments.health_monitor --data-root data/formal_local --motor T1 --rpm 8000rpm --config 8screws --output-mode binary --allow-no-identity --max-windows 5
```

## 後續資料收集

每個 window 應補 `motor_id/session_id/timestamp`、RPM、load/payload、voltage、current、
temperature、ambient、ESC、maintenance event、operator-confirmed fault type、ordinal
severity 與 failure endpoint；並以 motor/session 分組切 train/validation/test，才能可信啟用
event early warning、fault-cause classifier 或 RUL。
