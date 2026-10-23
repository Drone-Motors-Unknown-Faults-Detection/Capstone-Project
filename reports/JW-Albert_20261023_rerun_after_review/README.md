# 第一輪程式審查修正後的 Aero 重跑（#266、#267、#268／#280、#271）

2026-10-23，JW-Albert。main `b6e2b97`（合併 #281 實驗八錨點、#282 web、#283 core 與實驗二、#284 provenance 與資料）在 Aero 以同一份資料重跑受影響的實驗，與 2026-10-23 凌晨以 `46ed4d5` 跑的 [JW-Albert_20261023_aero_rerun](../JW-Albert_20261023_aero_rerun/README.md) 逐項比對。

## 條件

- 主機：aero01，Linux x86_64，CPython 3.10.19，numpy 2.2.6 連結 scipy-openblas 0.3.29；clone 在 `~/cc/rerun-F`，`uv sync --locked --extra test`。
- 程式：commit `b6e2b97`，每份 `environment.json` 都是 `git.tracked_dirty=false`、`os.host=aero01`。
- 資料：`--data-root data`，指紋 `1b55b847…`（與 `46ed4d5` 那批相同）。Aero 沒有 `data/formal_local`，所以 `reports/exp8_health_index_results/` 與實驗六正式版的 `formal_local` 結果不重跑。
- 參數全照手冊預註冊：實驗二沒有動 `max_ticks_per_stage`（600）與密度階梯；指令見最後一節。

## 哪些實驗的數字變了

| 實驗 | 新輸出 | 比對對象（`46ed4d5`） | 結果 |
|---|---|---|---|
| 實驗一 | `output/exp1_cold_start/2026-10-23-11-56-08` | `01-30-51` | `summary.json` 相同；`results.csv` 只有 4screws 的分數中位數尾數不同（86.97517325744982 → 86.9751732574498，1 ulp），其餘逐位元組相同 |
| 實驗二（九次） | `output/exp2_scale_growth/2026-10-23-11-56-10`～`11-56-28` | `01-30-54`～`01-32-23` | 全部改變，見表3 |
| 實驗八矩陣 | `output/exp8_health_index_matrix/2026-10-23-11-56-31` | `01-34-15`（舊映射） | 健康指數平均改變，開集指標不變，見表2 |
| 實驗八基準（seed 42） | `output/exp8_health_index_benchmark/2026-10-23-11-56-38`（LW）、`11-56-41`（k-NN） | 本 repo 先前沒有 | 新增 |
| 實驗八逐窗 | `output/exp8_health_monitor/2026-10-23-11-56-44`～`11-56-47` | 本 repo 先前沒有 | 新增，見表4 |
| 實驗十一 | `output/exp11_ancestor_comparison/2026-10-23-11-56-48` | `01-35-07` | legacy 四個指標全部改變，LW 不變，見表5 |
| 實驗十二 | `output/exp12_confusion_tsne/2026-10-23-11-56-55` | `01-36-16` | 二元矩陣與 LW 矩陣不變；legacy 矩陣改變；標籤 `1screw` 改為 `1screws`，見表6 |
| 實驗三（40 次） | `output/exp3_trend/2026-10-23-11-57-25` | `01-32-36` | `summary.json`、`trials.csv` 逐位元組相同 |
| 實驗四 | `output/exp4_polar_map/2026-10-23-11-57-26` | `01-32-39` | 逐位元組相同 |
| 實驗五 | `output/exp5_cross_condition/2026-10-23-11-57-28` | `01-32-51` | 逐位元組相同 |
| 實驗七 | `output/openset_comparison/2026-10-23-11-57-32` | `01-34-12` | `summary.csv` 相同；`details.csv` 與 `summary.json` 只有同一個 4screws 中位數的 1 ulp 差 |
| 實驗十 | `output/exp10_transfer/2026-10-23-11-57-34` | `01-34-24` | 扣掉環境欄位後相同 |

實驗一與實驗七的 1 ulp 差：#283 把 `src/core/mahalanobis.py` 的三運算元 `einsum` 拆成兩個兩運算元 `einsum`，註解寫「逐位元相同」，但 T1／8000rpm 4screws 池 287 列的分數中位數差了最後一位。所有四捨五入後的數字（README 表的 86.98）不變。

## 實驗八：新錨點（#267）

錨點改為 cal q10 → 1.0、拒絕線（q95）→ 0.5、2×拒絕線 → 0.0。`open_set_accuracy`、AUROC、unknown recall 只看原分數，與舊映射那批相同。

表2 實驗八矩陣（九工況 × seed 42／123／2026，每方法 27 列等權，std 為 ddof=1）

| 方法 | 已知健康指數平均（新） | 舊映射（`01-34-15`） | 健康 holdout 指數 ≥ 0.5 的比例（新） | 未知健康指數 | open-set accuracy |
|---|---|---|---|---|---|
| Ledoit–Wolf | 0.790274 ± 0.037581 | 0.583337 ± 0.073589 | 94.66% ± 3.97 | 全為 0 | 0.998860 |
| k-NN（k=5） | 0.799017 ± 0.037106 | 0.600457 ± 0.072595 | 94.67% ± 4.38 | 全為 0 | 0.998867 |

`known_healthy_fraction` 現在等於「分數沒超過拒絕線」的比例，所以與實驗六正式矩陣的健康接受率同值；舊映射下這一欄是指數 ≥ 0.8 的比例（LW 28.49%、k-NN 30.27%），文件沒有引用過。

seed 42 基準（`11-56-38`、`11-56-41`）：九工況的 `known_health_mean` LW 0.730～0.811、k-NN 0.724～0.843，`unknown_health_mean` 全為 0、`unknown_critical_fraction` 全為 1。手冊預期 `known_healthy_fraction` 落在 0.9～1.0：LW 有兩個工況不到 0.9（T3／6000 0.875、T3／11000 0.883），k-NN 有兩個（T1／6000 0.844、T3／11000 0.867），其餘七個工況都在 0.9 以上；這一欄就是健康接受率，低於 0.9 的格也就是實驗六正式矩陣裡健康誤報超過 10% 的格。

表4 實驗八逐窗監測（T1／8000rpm、seed 42、Ledoit–Wolf、confidence 0.95、`--motor-id demo-T1`、不截斷窗數）

| 配置 | 輸出 | 窗數 | `alarm_state` | `severity_stage` | 原始指數 < 0.5 的窗 | 平滑指數最低 |
|---|---|---:|---|---|---:|---:|
| 8screws（健康 holdout） | `11-56-44` | 64 | normal 64／64 | healthy 64／64 | 7（即 `is_fault=1` 的 7 窗） | 0.667 |
| 1screws | `11-56-45` | 349 | 第 3 窗起 critical 347／347 | critical 349／349 | 349 | 0.0 |
| 5screws | `11-56-46` | 212 | 第 3 窗起 critical 210／210 | critical 212／212 | 212 | 0.0 |
| 7screws | `11-56-47` | 322 | 第 3 窗起 critical 320／320 | critical 322／322 | 322 | 0.0 |

故障配置的前兩窗是 normal，因為 critical 要連續 3 筆 `<0.2`。健康 holdout 的 7 筆超線窗原始指數掉到 0.46～0.50，平滑後都在 0.5 以上，沒有觸發 warning；`trend` 在 64 窗中 30 窗 worsening、30 窗 stable，`change_point_state` 有 4 窗 confirmed。手冊預期（normal ≥ 85%、故障第 3 窗起 warning 或 critical ≥ 85%）兩條都成立。

## 實驗二：隔離區規則一致化（#268、#280）

表3 九次執行（T1／8000rpm；「舊」是 `46ed4d5` 同設定、同 seed 的輸出）

「對上本階段配置」是 `matched_is_config` 為真的段數；「形成新類」是 `discovered` 為真的段數，它只表示該段有一個新類成形，不表示該段配置被辨識出來（[#286](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/issues/286)）；配置有沒有被認出要看 `final_eval.matched_rate`。

| seed | 設定 | 輸出 | 對上本階段配置（舊） | 形成新類（舊） | 發現延遲（筆） | 最低純度 | 外部評估 `known_rate` ≥80% 組數／最低（舊） | `matched_rate` ≥80% 組數／最低（舊） | 健康 holdout | 邊界退回（舊） | `quarantine_duplicates` | 健康窗送入／進隔離區 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 預設 | `11-56-10` | 7／9（9） | 8 次（9） | 57–621 | 100% | 7／0%（7／53.85%） | 7／0%（6／51.16%） | 89.06%–100% | 0（0） | 117 | 0／0 |
| 42 | `--fixed-density` | `11-56-12` | 4／9（3） | 6 次（9） | 31–629 | 93.50% | 3／0%（2／0%） | 3／0%（2／0%） | 95.31%–96.88% | 0（2） | 713 | 0／0 |
| 42 | `--healthy-mix 0.3` | `11-56-14` | 7／9（9） | 8 次（9） | 97–631 | 100% | 6／0%（7／53.85%） | 6／0%（6／51.16%） | 89.06%–100% | 0（0） | 56 | 487／0 |
| 123 | 預設 | `11-56-17` | 7／9（9） | 8 次（9） | 72–612 | 100% | 3／0%（6／56.92%） | 3／0%（6／56.92%） | 100% | 0（1） | 362 | 0／0 |
| 123 | `--fixed-density` | `11-56-19` | 1／9（2） | 7 次（8） | 62–641 | 94.85% | 6／0%（5／0%） | 6／0%（5／0%） | 100% | 0（1） | 740 | 0／0 |
| 123 | `--healthy-mix 0.3` | `11-56-21` | 7／9（9） | 8 次（9） | 128–612 | 100% | 5／0%（6／56.92%） | 5／0%（6／56.92%） | 100% | 0（1） | 204 | 458／0 |
| 2026 | 預設 | `11-56-24` | 7／9（9） | 8 次（9） | 52–623 | 100% | 5／0%（8／63.49%） | 5／0%（8／63.49%） | 98.44% | 0（0） | 408 | 0／0 |
| 2026 | `--fixed-density` | `11-56-26` | 5／9（2） | 7 次（8） | 31–629 | 93.50% | 4／0%（4／0%） | 4／0%（4／0%） | 98.44% | 0（0） | 801 | 0／0 |
| 2026 | `--healthy-mix 0.3` | `11-56-28` | 7／9（9） | 8 次（9） | 100–620 | 100% | 5／0%（8／63.49%） | 5／0%（8／63.49%） | 98.44% | 0（0） | 217 | 485／0 |

「發現延遲」讀 `summary.json` 的 `discovery`：某配置從開始注入到第一次成為某新類事後多數的筆數，可跨段；預設設定每個 seed 各有一個配置從未成為多數，不計入範圍。

預設設定三個 seed 的樣子一樣：4screws 段在 600 tick 內沒有形成候選（seed 2026 是 3screws 段），要到下一段的第 21／12／23 tick 才成群、事後多數是前一段的配置；於是那一段注入的配置（seed 42／123 的 3screws、seed 2026 的 2screws）整個實驗都沒有成為任何新類的多數，`final_eval` 的 `known_rate` 與 `matched_rate` 都是 0，最後 9 類而不是 10 類。其餘七段都學到當段配置，純度全部 100%，沒有同一配置被學成兩類，`rejected_boundary` 九次都是 0。seed 42 舊版那個混入 5screws 的 4screws 群（純度 68.33%）不再出現，5screws 的 `matched_rate` 從 51.16% 升到 97.67%。

對照手冊「隔離區規則一致化（issue #268）」的預期：

- 成立：seed 42 預設的 5screws `final_eval.matched_rate` 高於 51.16%（97.67%）；`--fixed-density` 下沒有同一 `matched` 被學成兩個以上新類（舊版 seed 42 的 5screws 學成三次）；`healthy_holdout_acc` 的範圍與舊版相同；`rejected_boundary` 為 0。
- 不成立：預設三個 seed 的 `matched_is_config` 段數從 9／9／9 降到 7／7／7，`discovery` 裡從未被發現的配置從 0 個變成每個 seed 各 1 個。手冊把這兩項列為「修正不成立」的訊號。
- 預期寫錯：手冊預期「預設 seed 42 的 `quarantine_duplicates` 應為 0」，實測 117（seed 123 為 362、seed 2026 為 408）。`CycleSampler` 一圈後重抽同一列，一個階段只有約 240 筆不重複列，舊版 9／9 有一部分是靠重複列把候選撐起來的（#280）。
- 延遲：七個正常學到的段，`samples_to_candidate` 落在 52～125 筆，與舊版相當；被延後的那個配置要 612～623 筆才被發現，所以 `discovery` 的範圍拉到 52–623。

`--fixed-density`：三個 seed 對上本階段配置 4／1／5 段（舊版 3／2／2），形成新類 6／7／7 次（舊版 9／8／8），`final_eval.matched_rate` 達 80% 的只有 3／6／4 組，最慢 641 筆；仍是多數段學到前一段的殘餘。`--healthy-mix 0.3`：三個 seed 共送入 487＋458＋485＝1430 筆健康窗（延後的那一段跑滿 600 tick，所以比舊版的 745 多），0 筆進入隔離區，沒有形成事後多數為 8screws 的新類；學到當段配置的段與 `final_eval` 的 0 都和不混入時落在同一個配置。

處理原則依 #280：不為了讓數字變好而調 `max_ticks_per_stage` 或密度階梯；要調預算應另立實驗、先寫預期。

## 實驗十一、十二：legacy 最近類別判讀（#266）

表5 實驗十一（九工況 × seed 42／123／2026，27 列等權）

| 協定 | 方法 | accuracy | balanced accuracy | macro-F1 | unknown F1 | 舊（`46ed4d5`） |
|---|---|---:|---:|---:|---:|---|
| A | legacy | 98.20% | 90.82% | 94.08% | 98.94% | 83.00%／16.67%／15.12%／90.71% |
| A | Ledoit–Wolf | 98.98% | 95.05% | 97.30% | 99.39% | 不變 |
| B | legacy | 98.17% | 57.12% | 61.26% | 99.07% | 97.84%／50.00%／49.46%／98.91% |
| B | Ledoit–Wolf | 99.88% | 97.14% | 98.47% | 99.94% | 不變 |

Wilcoxon（27 組 macro-F1 配對）：協定 A 的 LW−legacy 平均差 3.21 pp（舊 82.18 pp）、W=8、p 四捨五入為 0；協定 B 37.21 pp（舊 49.02 pp）、W=0。手冊預期成立：legacy 的協定 A balanced accuracy 高於 1／6，`mean_diff_pp` 比 82 pp 小。但程式設定的內部門檻 `threshold_pp=5` 在協定 A 已達不到：LW 比 legacy 的 macro-F1 只高 3.21 pp。協定 B 的 legacy balanced accuracy 57.12% 表示健康 holdout 大多仍被拒，差距來自 legacy 用 train 自身距離的 q95 定線。

表6 實驗十二 legacy 多類矩陣逐真值類（筆；LW 矩陣與二元矩陣與舊輸出相同）

| 真實已知類 | 支持數 | 正確預測原類 | 拒為 unknown | 錯分其他已知類 | 舊（`46ed4d5`） |
|---|---:|---:|---:|---:|---|
| 8screws | 572 | 464 | 108 | 0 | 0／572／0 |
| 1screws | 601 | 464 | 137 | 0 | 0／601／0 |
| 2screws | 605 | 581 | 23 | 1（→4screws） | 0／605／0 |
| 3screws | 591 | 554 | 37 | 0 | 0／591／0 |
| 4screws | 569 | 537 | 32 | 0 | 0／569／0 |
| 已知合計 | 2,938 | 2,600 | 337 | 1 | 0／2,938／0 |

已知判對 2,600／2,938＝88.50%，誤拒 337／2,938＝11.47%，互相錯分 1 筆；unknown 14,381 筆仍全部拒絕。手冊預期「legacy 矩陣的對角不再只有 unknown 一格」成立。

## 限制

- `formal_local` 不在 Aero／Spark：`reports/exp8_health_index_results/` 是舊映射、舊資料的封存，Web 8-2 封存卡讀的仍是它；`output/exp8_health_index_aggregate/` 的重算只對那份封存有效。
- 實驗二仍只有 T1／8000rpm、三個 seed、每段 600 tick。
- 實驗八逐窗只跑 T1／8000rpm 的四個配置。

## 指令

在 Aero 的 `~/cc/rerun-F`、`PYTHONPATH=src`、`venv/bin/python -m` 依序執行：

```bash
experiments.exp1_cold_start --data-root data --motor T1 --rpm 8000rpm --seed 42
experiments.exp2_scale_growth --data-root data --seed {42,123,2026} [--fixed-density | --healthy-mix 0.3]
experiments.health_index_matrix --data-root data --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn
experiments.health_index_benchmark --data-root data --seed 42 --openset-method mahalanobis --method ledoit_wolf --confidence 0.95
experiments.health_index_benchmark --data-root data --seed 42 --openset-method knn --knn-neighbors 5
experiments.health_monitor --data-root data --motor T1 --rpm 8000rpm --config {8screws,1screws,5screws,7screws} --seed 42 --openset-method mahalanobis --method ledoit_wolf --confidence 0.95 --motor-id demo-T1 --session-id demo-<config> --output-mode full
experiments.exp11_ancestor_comparison --data-root data --seeds 42,123,2026 --methods legacy,ledoit_wolf --confidence 0.95
experiments.exp12_confusion_tsne --data-root data --seed 42 --confidence 0.95 --perplexity 30 --tsne-max-per-class 200
experiments.exp3_trend --data-root data --motor T1 --rpm 8000rpm --seed 42 --trials 40
experiments.exp4_polar_map --data-root data --motor T1 --rpm 8000rpm --seed 42 --part abc
experiments.exp5_cross_condition --data-root data --seed 42 --part abc --confidence 0.95 --method ledoit_wolf
experiments.compare_openset --data-root data --motor T1 --rpm 8000rpm --seed 42 --openset-methods mahalanobis knn --method ledoit_wolf --knn-neighbors 5 --confidence 0.95
experiments.exp10_transfer --data-root data --seed 42 --confidence 0.95 --method ledoit_wolf --n-values 10,25,50
```
