# 拆分大型 orchestration 的回歸規範

給要拆 `src/experiments/exp6_formal_benchmark.py:_run_benchmark` 與 `src/experiments/exp6_matrix.py:run_matrix` 的組員：先讓 `src/tests/test_orchestration_contract.py` 固定下面的行為，再每次抽一個 helper、單獨 commit。這個測試把現行程式和固定基準 commit 的同名函數放在同一份 fixture 上配對執行；任何一步讓配對不相等，那一步就要退回。

基準 commit 是 `39d3993`（PR #84 合併後的 main），用 `git show` 讀當時的原始碼執行，不另存一份副本。

## 固定的行為

| 項目 | 怎麼比 |
|---|---|
| `run()`／`main()` 與 `run_matrix()` 的參數 | `inspect.signature` 參數名、預設值與 keyword-only 位置相同 |
| 正式 benchmark 輸出 | 兩種方法各跑一次，頂層鍵集合相同；`rows` 逐列逐欄相同，只排除 `inference_seconds` |
| 錯誤 | 非法 confidence、`legacy` covariance、缺 healthy、沒有 unknown 配置：例外型別與訊息相同 |
| 隨機數消耗順序 | 追蹤每次 `make_split` 的輸入長度與回傳的 train／cal／holdout 索引，序列逐一相同 |
| 偵測器輸入 | 追蹤每次 `score_samples` 收到的矩陣與回傳分數，逐位相同 |
| 矩陣狀態 | 用假的 `run` 跑 `run_matrix`：manifest 每格的鍵、狀態、相對路徑相同；`summary.json`、`results.csv` 逐位元組相同 |
| 中斷與續跑 | 中間一格丟例外時，該格記 `failed` 並繼續其餘格，結果為 `incomplete`；重跑只重算失敗那格 |

failed-refit 由 `src/tests/test_monitor_guard.py` 涵蓋，Web 串流的事件順序由 `src/tests/test_stream_integration.py` 對照整合前的 main，本規範不重複。

## 拆分範圍與順序

1. `_condition_metrics(known_scores, unknown_scores)`：從 `_run_benchmark` 抽出八個指標與兩個中位數的計算。純函數，輸入兩組分數，輸出 dict。
2. `_condition_result(result, entry)`：從 `run_matrix` 抽出「只留目標工況那一列」的步驟。純函數，找不到或多於一列時丟出原本的 `ValueError`。

科學計算留在 `experiments/` 與 `core/`，`web/` 只做編排，不新增 detector 或改變判定。`web/live.py` 的 `tick()` 本輪不拆，要拆時先替它補一份同樣的配對測試。

## 怎麼跑

```bash
uv run --locked --extra test python -m pytest src/tests/test_orchestration_contract.py
```

測試只用 fixture 和假的 `run`，不讀正式 `data/`。
