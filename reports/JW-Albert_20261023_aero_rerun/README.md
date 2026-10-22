# Aero 全面重跑：可追溯的實驗輸出（issue #239、#242、#243、#240）

2026-10-23，JW-Albert。依 AGENT.md 鐵則 13，在 Aero 以同一個 commit、同一份資料重跑實驗一至十三，讓每份引用輸出的 `environment.json` 都記錄主機、BLAS 與資料指紋。

## 條件

- 主機：aero01，Linux 5.15.0-198 x86_64，CPython 3.10.19，numpy 連結 scipy-openblas 0.3.29。
- 程式：commit `46ed4d5`（PR #247 合併前的分支頭；與合併後的 main 只差實驗頁顯示 `eval_matched_rate` 一欄，實驗程式相同）。75 份 `environment.json` 都是 `tracked_dirty=false`、`os.host=aero01`。
- 資料：`--data-root data`，資料指紋 `source_content_v2:1b55b847…`（90 個特徵 CSV）。Aero 與 Spark 上沒有 `data/formal_local`，正式矩陣與實驗八也改讀 `data`。
- 指令：見本報告最後一節。

## 與先前引用輸出的比對

| 實驗 | 新輸出 | 先前引用的輸出 | 結果 |
|---|---|---|---|
| 實驗一 | `output/exp1_cold_start/2026-10-23-01-30-51` | `2026-10-22-21-45-32`（37ae787） | `results.csv`、`summary.json` 逐位元組相同 |
| 實驗三（40 次） | `output/exp3_trend/2026-10-23-01-32-36` | `2026-10-22-21-46-58` | 逐位元組相同 |
| 實驗四 | `output/exp4_polar_map/2026-10-23-01-32-39` | `2026-10-22-21-47-00` | 逐位元組相同 |
| 實驗五 | `output/exp5_cross_condition/2026-10-23-01-32-51` | `2026-10-22-21-47-11` | 逐位元組相同 |
| 實驗六廣度、九 | `output/exp6_osr_benchmark/2026-10-23-01-33-02` | `2026-10-22-21-47-21` | 逐位元組相同（#235 的 FPR@TPR95 改用 ROC 定義後，這批資料沒有同分，數值不變） |
| 實驗十 | `output/exp10_transfer/2026-10-23-01-34-24` | `2026-10-09-14-11-31`（無主機紀錄） | 扣掉環境欄位後相同 |
| 實驗十一 | `output/exp11_ancestor_comparison/2026-10-23-01-35-07` | `2026-10-09-14-17-27` | 扣掉環境欄位後相同 |
| 實驗十二 | `output/exp12_confusion_tsne/2026-10-23-01-36-16` | `2026-10-09-14-27-55` | 扣掉環境欄位後相同 |
| 實驗十三導覽回歸 | `output/navigation_regression/2026-10-23-01-37-17` | 總手冊 8.17（Lineage 3aa554f，本 repo 無輸出） | Mahalanobis 與 k-NN 各 595 組配對、0 筆不一致，候選確認為 5screws（同總手冊） |
| 實驗七 | `output/openset_comparison/2026-10-23-01-34-12` | 本 repo 先前沒有輸出 | 新增 |

## 讀資料根目錄不同而改變的結果

正式矩陣與實驗八先前讀 `data/formal_local`（Python 3.12.14、commit 07582ff，無主機紀錄），這次讀 `data`。`formal_local` 由階段 ZIP 物化（`docs/formal_materialization_contract.md`），與 `data` 不是同一份檔案：T1／11000 健康池在 `formal_local` 是 358 列、在 `data` 是 357 列，九工況健康 holdout 合計 573 對 572、故障 26,088 對 26,070（#242）。實驗十二的支持數 572／26,070 與這次的 `data` 一致。

表1 實驗六正式矩陣（九工況 × seed 42、123、2026，27 列等權，std 為 ddof=1）

| 方法 | 健康接受率（新，`data`） | 健康接受率（舊，`formal_local`） | 未知召回 | open-set accuracy（新） |
|---|---|---|---|---|
| Ledoit–Wolf | 94.66% ± 3.97（最低 87.50%） | 94.62% ± 4.01 | 100% | 99.886% |
| k-NN（k=5） | 94.67% ± 4.38（最低 84.38%） | 94.57% ± 4.39 | 100% | 99.887% |

輸出：`output/exp6_formal_matrix/aero_2026-10-23/`（54 次全部完成，彙總在 `aggregate/`）。兩方法差距仍在 0.01 個百分點量級，結論不變。

表2 實驗八相對健康指數矩陣（27 列）

| 方法 | 已知健康指數平均（新） | 舊（`formal_local`） | 未知健康指數 |
|---|---|---|---|
| Ledoit–Wolf | 0.583337 ± 0.073589 | 0.583144 ± 0.073301 | 全為 0 |
| k-NN | 0.600457 ± 0.072595 | 0.601657 ± 0.071637 | 全為 0 |

輸出：`output/exp8_health_index_matrix/2026-10-23-01-34-15/`。

## 實驗二：判進對應新類（#243）與健康混入（#240）

表3 實驗二九次執行（T1／8000rpm）

| seed | 設定 | 輸出 | 學到當段配置 | 被發現 | 發現延遲（筆） | 外部評估：被任一新類接受（最低／≥80% 的組數） | 外部評估：判進對應新類（最低／≥80% 的組數） | 邊界退回 | 健康窗送入／進隔離區 |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 預設 | `2026-10-23-01-30-54` | 9／9 | 9／9 | 51–75 | 53.85%／7 | 51.16%／6 | 0 | 0／0 |
| 42 | `--fixed-density` | `2026-10-23-01-31-05` | 3／9 | 7／9 | 62–402 | 0.00%／2 | 0.00%／2 | 2 | 0／0 |
| 42 | `--healthy-mix 0.3` | `2026-10-23-01-31-17` | 9／9 | 9／9 | 68–103 | 53.85%／7 | 51.16%／6 | 0 | 208／0 |
| 123 | 預設 | `2026-10-23-01-31-27` | 9／9 | 9／9 | 52–105 | 56.92%／6 | 56.92%／6 | 1 | 0／0 |
| 123 | `--fixed-density` | `2026-10-23-01-31-38` | 2／9 | 8／9 | 62–632 | 0.00%／5 | 0.00%／5 | 1 | 0／0 |
| 123 | `--healthy-mix 0.3` | `2026-10-23-01-31-49` | 9／9 | 9／9 | 73–149 | 56.92%／6 | 56.92%／6 | 1 | 231／0 |
| 2026 | 預設 | `2026-10-23-01-32-01` | 9／9 | 9／9 | 51–118 | 63.49%／8 | 63.49%／8 | 0 | 0／0 |
| 2026 | `--fixed-density` | `2026-10-23-01-32-13` | 2／9 | 7／9 | 31–696 | 0.00%／4 | 0.00%／4 | 0 | 0／0 |
| 2026 | `--healthy-mix 0.3` | `2026-10-23-01-32-23` | 9／9 | 9／9 | 71–173 | 63.49%／8 | 63.49%／8 | 0 | 306／0 |

外部評估讀 `summary.json` 的 `final_eval`（全部段結束後）；「≥80% 的組數」是九個配置中比例達 80% 的個數。每段學會當下的比例另記在 `stages.csv` 的 `eval_known_rate`／`eval_matched_rate`，兩者時點不同，不混用。

對照手冊預期：

- `matched_rate` ≤ `known_rate` 成立。預設設定三個 seed 合計，判進對應新類達 80% 的是 20／27 組，被任一新類接受達 80% 的是 21／27 組；差距只出在 seed 42 的 5screws：外部評估集有 95.35% 被某個新類接受，但只有 51.16% 判進事後多數為 5screws 的新類，其餘被事後多數不是 5screws 的新類接受（seed 42 的 4screws 段候選群混入 5screws，purity 68.33%）。表3 seed 42 的最低值 51.16% 就是這一組；7screws 兩個指標都是 53.85%。
- 健康混入 0.3：三個 seed 共送入 745 筆健康 holdout 窗，沒有一筆進入隔離區，沒有形成事後多數為 8screws 的新類，`healthy_holdout_acc` 與不混入相同；支持隔離線 2.0 擋住健康誤報。代價是發現延遲變長（延遲計數含健康筆）。
- 預設設定的實驗二數字與 `reports/JW-Albert_20261023_exp2_adopt_fix_aero/` 的 `2026-10-23-00-13-53` 等輸出相同。

## 限制

- 單一資料版本（`data`）；`formal_local` 的結果保留在舊輸出，未在 Aero／Spark 重現，因為兩台主機沒有物化來源。
- 健康混入用的是健康 holdout，段末的 `healthy_holdout_acc` 也用同一份；混入的筆只被打分、沒有進入擬合，但兩者不是互斥的資料。
- 實驗二仍是單一工況（T1／8000rpm）、三個 seed。

## 指令

在 Aero 的 `~/cc/Capstone-Project`、`PYTHONPATH=src`、`venv/bin/python -m` 依序執行：

```bash
experiments.exp1_cold_start --data-root data --motor T1 --rpm 8000rpm --seed 42
experiments.exp2_scale_growth --data-root data --seed {42,123,2026} [--fixed-density | --healthy-mix 0.3]
experiments.exp3_trend --data-root data --motor T1 --rpm 8000rpm --seed 42 --trials 40
experiments.exp4_polar_map --data-root data --motor T1 --rpm 8000rpm --seed 42 --part abc
experiments.exp5_cross_condition --data-root data --seed 42 --part abc --confidence 0.95 --method ledoit_wolf
experiments.exp6_osr_benchmark --data-root data --seed 42 --confidence 0.95
experiments.exp6_matrix --data-root data --output-root output/exp6_formal_matrix/aero_2026-10-23 --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn
experiments.aggregate_exp6 --manifest output/exp6_formal_matrix/aero_2026-10-23/matrix_manifest.json --output-root output/exp6_formal_matrix/aero_2026-10-23/aggregate
experiments.compare_openset --data-root data --motor T1 --rpm 8000rpm --seed 42 --openset-methods mahalanobis knn --method ledoit_wolf --knn-neighbors 5 --confidence 0.95
experiments.health_index_matrix --data-root data --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn
experiments.exp10_transfer --data-root data --seed 42 --confidence 0.95 --method ledoit_wolf --n-values 10,25,50
experiments.exp11_ancestor_comparison --data-root data --seeds 42,123,2026 --methods legacy,ledoit_wolf --confidence 0.95
experiments.exp12_confusion_tsne --data-root data --seed 42 --confidence 0.95 --perplexity 30 --tsne-max-per-class 200
experiments.navigation_data_contract --data-root data --motor T1 --rpm 8000rpm --seed 42
experiments.navigation_regression --data-root data --seed 42
```
