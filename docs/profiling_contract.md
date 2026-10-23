# 效能量測規範：CSV 載入與 k-NN 評分

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
- `repeats`（預設 5 次）逐次保存 walltime（`time.perf_counter`）與 Python 配置峰值（`tracemalloc`，只算 Python 物件與 NumPy 緩衝，不等於 RSS）。兩者分開量：先在沒有 tracemalloc 的情況下計時，再重跑同一步驟量記憶體，因為 tracemalloc 會明顯拖慢純 Python 的 CSV 解析，讓 `load` 的比例偏高。
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

## CSV cache（依量測結果實作）

`core.data.PoolCache` 把 `load_pools()` 的結果存在記憶體，鍵是內容，不是路徑或 mtime：

- 鍵 = `REGISTRY_VERSION`、`require_complete`，加上工況目錄下每個子目錄名與其中每個 `*_Group_feature_data_clean.csv` 的檔名及 bytes SHA256。size、mtime 不進鍵，所以同 size、同 mtime 但 bytes 不同的檔案一定 miss。
- 不同路徑但內容相同的副本會命中同一筆，回傳的 pools 相同。快取只認內容，不把副本當成新的獨立採集；來源身分仍由 `core.provenance` 記錄。
- miss 時先算鍵、再載入、載入後重算一次鍵，兩次不同就不存，避免存到讀檔途中被改掉的內容。載入後那次重算沿用 `read_feature_csv` 對同一批 bytes 算出的 `sha256`（audit 的 `files`），不再讀第二次檔；目錄結構仍重新掃描。載入失敗不快取，下次照樣重試、照樣丟同一個錯誤。
- 命中時回傳 pools 與 audit 的深拷貝，呼叫端修改陣列不會污染快取。
- 只在記憶體、只活在呼叫端建立的那個 `PoolCache` 物件裡，不寫磁碟，因此沒有公開／私人路徑的問題。`exp6_matrix.run_matrix` 與實驗八 `health_index_matrix.run_matrix` 每次執行各建一個，傳給 benchmark 的 `run(..., pool_cache=...)`；單次 CLI 不帶 cache，行為與原本相同。

讀 bytes 算 SHA 仍要讀檔，所以命中時省下的是 CSV 解析與欄位檢查；省下多少由 `profiling_evidence` 的 `cached_load` 欄位量測。

## 怎麼跑

```bash
uv run --locked python -m tests.profiling_evidence                       # 合成資料
uv run --locked python -m tests.profiling_evidence --data-root data/formal_local/Step-1/myfeature/T1/8000rpm
uv run --locked python -m tests.profiling_evidence --repeats 10 --methods knn
```

輸出 `logs/profiling_evidence/{ts}.log` 與 `output/profiling_evidence/{ts}/`：`environment.json`、`profile.json`（全部原始量測、統計與兩條門檻的判定）。回歸測試在 `src/tests/test_profiling_evidence.py`，用很小的合成資料確認輸出結構與判定邏輯，不檢查秒數。

本工具不改模型、split 或分數，也不能用來宣稱準確率提高。
