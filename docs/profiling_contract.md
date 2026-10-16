# 效能量測契約：CSV 載入與 k-NN 評分

給要改 `src/core/data.py:load_pools` 或 `src/core/openset.py` k-NN 評分的組員：先用 `src/tests/profiling_evidence.py` 量測，依本頁事先寫好的門檻決定要不要加 CSV cache 或把逐類 `kneighbors` 批次化。門檻在量測前固定，量完不回頭改門檻。

## 量什麼

一次「run」是 `exp6_formal_benchmark` 對一個工況做的事。依序計時：

| 階段 | 對應程式 | 計時範圍 |
|---|---|---|
| `load` | `core.data.load_pools(path)` | 讀該工況全部 CSV、欄位與品質檢查 |
| `fit` | `make_split` → `RobustScaler.fit` → `create_openset_detector(...).fit` | healthy train 擬合、cal 定門檻 |
| `score` | `detector.score_samples` | healthy holdout 與全部 unknown 列 |

另量 `knn_score_by_classes`：已知類別數 1、5、10 時 k-NN 評分同一批樣本的時間，對應實驗二量尺擴張後逐類 `kneighbors` 的成本。

每個量測值保存全部重複，不只保存最好的一次：

- `warmup`（預設 1 次）不計入統計，另存。
- `repeats`（預設 5 次）逐次保存 walltime（`time.perf_counter`）與 Python 配置峰值（`tracemalloc`，只算 Python 物件與 NumPy 緩衝，不等於 RSS）。
- 統計欄位：median、min、max，以及 `load`、`fit`、`score` 在單次 run 中所佔的比例（以 median 計算）。
- 每次都記錄方法、seed、資料列數、類別數、批次大小與 reference bank 大小。
- 執行環境（Python、OS、套件、CPU 數、git HEAD）由 `setup_run` 寫的 `environment.json` 保存。

冷／熱快取：同一個 process 第一次讀檔（warmup）和之後的重複分開存。OS page cache 需要 root 權限才能清空，本工具不清，`os_page_cache` 記為 `not_controlled`。不同機器的秒數不能互相比較，只比較同一次量測裡各階段的比例。

## 資料

沒有正式 `data/` 時，工具在暫存目錄產生與正式資料同大小的合成 CSV：用已登錄的 `historical_clean_v1` 105 欄，健康 313 列，其他 9 種配置各 304 列，與 T1/8000rpm 的 3049 列同量級。合成資料只能拿來看時間比例，分數沒有意義。有正式資料時用 `--data-root` 指向它，工具只讀不寫。

## 決策門檻（量測前固定）

| 改動 | 值得做的條件 |
|---|---|
| CSV cache（以內容 SHA 為鍵） | `load` 佔單次 run median 的 **≥ 50%**，或者 `exp6_matrix` 一整個矩陣的預估載入總時間 **≥ 60 秒** |
| k-NN 批次化 | 10 類時的 `knn_score_by_classes` median **≥ 1 類時的 5 倍**，且 10 類單批評分 **≥ 0.5 秒** |

兩條都不成立就不改程式，保留量測結果作為負面結論。成立時才另開 PR，要求同輸入、同 seed 下逐筆分數與門檻在 `atol=1e-12` 內相同，並涵蓋平手、空批次、類別大小不同的 fixture；cache 還要能分辨同 size／mtime 但 bytes 不同的檔案。

`exp6_matrix` 的矩陣預估：每個 run_id 都對全部 9 個工況呼叫一次 benchmark（見[實驗六矩陣手冊](experiments/exp6_matrix.md)），所以載入次數是 9 工況 × 54 run = 486 次，預估總載入時間為 `load` median × 486。

## 怎麼跑

```bash
uv run --locked python -m tests.profiling_evidence                       # 合成資料
uv run --locked python -m tests.profiling_evidence --data-root data/formal_local/Step-1/myfeature/T1/8000rpm
uv run --locked python -m tests.profiling_evidence --repeats 10 --methods knn
```

輸出 `logs/profiling_evidence/{ts}.log` 與 `output/profiling_evidence/{ts}/`：`environment.json`、`profile.json`（全部原始量測、統計與兩條門檻的判定）。回歸測試在 `src/tests/test_profiling_evidence.py`，用很小的合成資料確認輸出結構與判定邏輯，不檢查秒數。

本工具不改模型、split 或分數，也不能用來宣稱準確率提高。
