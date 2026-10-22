# 實驗二：未知樣本分群與量尺擴張

## 資料角色用量表

流程：健康基準 → 超線未知隔離 → HDBSCAN找密集群 → 只用該群樣本切分並重擬合 → 事後對照真實配置。[用量依據與分母](README.md#資料用量與分母怎麼讀)為實際載入與splitter重播，不是一次新發現實驗。

| 資料角色 | 馬達／轉速／配置 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 初始train／cal／holdout | T1／8000rpm／健康 | 187／62／64 | 各除以313健康列 | 基準、門檻、健康測試 | 三用途列級互斥 |
| 發現來源 | 同工況各未知配置（例：5screws） | 每配置約80%（5screws：169列） | 配置池扣掉外部評估集 | 循環重播、隔離及分群 | 可反覆抽同列，不是169次採集 |
| 外部評估集 | 同工況各未知配置 | 每配置約20%（5screws：43列） | 配置池列數×0.2，向上取整 | 每段學會後與全部段結束後評估該配置被認成哪一類 | 與重播列互斥；不進擬合、校準或分群 |
| 自學train／cal／holdout | 該次候選群 | 約 60／20／20 | 分母是這一群的筆數，不是完整配置池 | 與健康共同重fit；holdout只含這群未用於擬合的列 | 與注入重播列重疊，不是未見配置的獨立測試 |

新類筆數等於該群，不是完整配置池的 127／42／43。類名是 `discovered-N`。候選、隔離與實際tick數須讀該次輸出，計數工具沒有執行HDBSCAN，不能預填發現延遲。無另設validation選參。

## 發表與理解檢查

口頭主軸：「先拒絕陌生資料，等特徵形成群，就只用這一群自己學一個新類。操作員不必知道現在是哪一種鬆脫。事後才對照真實配置，看這群純不純、健康有沒有退步。」示意：30筆過隔離線未必形成25筆的密集群，因此未知召回高也可能發現失敗；這不是實測延遲。

- HDBSCAN學了故障名稱嗎？沒有。它只依密度分群；模型類名是 discovered-N。truth 多數只出現在事後對照。
- 學會是否表示馬達恢復？沒有，只是模型多了一個由現場樣本組成的類。
- 更新後測試是否全新？新類 holdout 是同一群裡沒拿去擬合的列，不是另一台機或完整配置池。
- 組員請重述「拒絕、隔離、成群、自學、事後對照」五步；數字及失敗條件查下方輸出，不只背一個準確率。

系統先只認識健康，看到陌生樣本時先隔離，累積成叢後自己納入。這個實驗承接[冷啟動](exp1_cold_start.md)，檢查「發現一群」和「只用這一群還認不認得」兩件不同的事；不是沒有標註就自動知道物理故障原因。

## 資料、切分與模型

資料來自單一馬達／轉速的105維clean特徵CSV；只接受已登錄 ordered 105 欄，非法列整批拒絕、不刪列。未去重、單位與 raw window 未知，見[實驗一](exp1_cold_start.md#資料與載入)。T-code代表不同個體；螺絲目錄代表配置。初始只用健康池列隨機60/20/20，seed42；train fit RobustScaler與LW或k-NN參考，calibration定0.95分位數，健康holdout供評估。legacy的train門檻例外同實驗一。

未知注入用全配置池，不只holdout。`CycleSampler`每圈洗牌、抽完重用，rng為seed+1；600個tick不代表600次獨立採集。成群後 `add_observed` 只把這一群做 60/20/20，和既有已知類一起重新 fit scaler、detector 與 PCA。不呼叫 `add_class(config)`，因此不會把該配置的完整池拿來學。新類 holdout 仍來自同一群。

這是統計模型重擬合與近鄰索引更新，沒有epoch或optimizer。幾何由PolarMap重建，見[實驗四](exp4_polar_map.md)；方向不改Open Set拒絕結果。

## 候選如何出現與自學

1. `ScaleGrowthSession.process(x, truth)`先打分、分類。score嚴格大於1才未知；未知且嚴格大於隔離線2.0才存隔離區。1～2的邊界未知不分群。truth 只存起來供事後對照，不參與要不要學。
2. 隔離區至少25筆、距上次嘗試至少10筆新隔離樣本，才在縮放後特徵跑HDBSCAN。先(25,3)，累積75筆加試(15,3)，100筆加試(10,2)；候選仍至少25筆，噪聲-1不當新類。
3. 選足夠大的叢。HDBSCAN fit只收特徵。`_try_cluster` 會順便算出 truth 多數與純度，但 `adopt()` 不拿它當類名，也不拿它決定是否納入。
4. 邊界退回：先用目前模型重打這一群的分數。群的中位分數若不超過已知類校準資料的最大分數（`known_cal_max`），表示這群沒有比已知資料更遠，回 `rejected_boundary`，清掉這一群、不擴張。這一步不讀 truth。
5. 群至少 5 筆就 `learned`：類名 `discovered-N`，移除這一群在隔離區的列。樣本不足則 `insufficient`，不擴張。
6. 學會後清理隔離區：用重擬合後的模型重新分類隔離區剩下的列，已被任一已知類接受（分數 ≤ 1）的列移出隔離區，數量記在 `pruned`。同一配置的殘餘因此不會在下一段被學成第二個類。這一步也不讀 truth。
7. 批次與邊跑邊畫都在第一個 learned 結束該段。每段最多 600 tick。API 可設 max_ticks_per_stage，CLI 沒有這個選項。`learned` 是模型新類，`matched` 才是事後多數配置，兩者不必相同。

這不是完整配置池的 oracle 更新。首頁即時展示的 `confirm()` 仍是另一條手動確認路徑。

## 怎麼執行與操作

在repo根目錄，先備妥[uv環境](../runtime_policy.md)與data，最小指令：

```bash
uv run --locked python -m experiments.exp2_scale_growth --help
uv run --locked python -m experiments.exp2_scale_growth --data-root data --motor T1 --rpm 8000rpm --seed 42
uv run --locked python -m experiments.exp2_scale_growth --data-root data --sequence 5screws 3_14screws --openset-method knn --knn-neighbors 5
```

`run(pools, sequence=None, seed=42, max_ticks_per_stage=600, adaptive_density=True)`回傳dict、不自行存檔；`ScaleGrowthSession(pools)`供逐筆 process，成群後呼叫 `adopt()`。共用CLI支援confidence0.95、method預設ledoit_wolf；`--fixed-density` 關掉密度階梯、只用 (25,3)。隔離線須用API，CLI未暴露。

Web啟動同[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)。「實驗二」卡選資料集／方法／seed，按「▶ 執行」會跑整個自學劇本。「⏵ 邊跑邊畫」依序自動注入各配置並逐筆畫分數，成群後立刻用該群自學並繼續；「■ 停止」取消後續串流。首頁即時展示仍是另一個持續 session，確認按鈕留在那一頁，不代表實驗二還在等操作員命名。

缺配置時核對實際資料夾名（1screw／1screws可能不同），不要自行猜名稱。沒有candidate時confirm會RuntimeError。重擬合失敗沒有原子回滾：停止使用、重建健康基準，不以舊scaler與新模型混用。每次CLI建立時間戳目錄，勿同秒並行；沒有此實驗續跑旗標。

## 輸出怎麼看與判斷

[main writer](../../src/experiments/exp2_scale_growth.py)寫`logs/exp2_scale_growth/{ts}.log`、`output/exp2_scale_growth/{ts}/environment.json`、`stages.csv`、`summary.json`；Web只由runner存結果JSON到`output/web_server/{ts}/experiments/`。模型不長期保存。

- `config`是注入劇本，`learned`是模型新類，`matched`是事後多數配置；三者要一起看。
- `discovered=false`表示該段未learned；不能當作故障不存在。失敗段不一定有delay等欄。
- `samples_to_candidate`是該段到首次learned的tick；單位筆，不是秒。循環抽樣會重用列。它量的是「這段開始後多久有任一群學成」，不是「當段配置多久被發現」。
- `matched_is_config`：這段學到的群，事後多數是否就是當段注入的配置。
- `summary.json` 的 `discovery`：每個配置第一次成為某個新類事後多數的時點。`delay` 從該配置開始注入起算到那一筆，可能跨到後面的段；從未成為多數則 `discovered=false`。這才是「某配置多久被發現」。
- `cluster_purity`是事後 truth 的叢多數比例，不是模型當時知道的故障名稱。
- `own_holdout_acc`是新類自己的 holdout 有沒有被分回這個新類，`own_holdout_n`是它的筆數（群31筆時只有7筆，一筆差14.3個百分點）；`healthy_holdout_acc`是健康有沒有被保住。兩者都含拒絕造成的錯誤。
- `eval_known_rate`／`eval_n`：當段注入配置的外部評估集，學會後被任一非健康已知類接受的比例與筆數。`summary.json` 的 `final_eval` 在全部段結束後，逐配置記錄外部評估集被判為未知、健康或某個新類的比例，以及最多被判成哪個新類。
- `known_after`、`final_known`和model逐類數量記錄擴張狀態。

預期在600tick內發現配置且保持健康辨識；未預定獨立部署通過線。低純度支持混叢問題，未發現也可能是隔離／密度或資料不足，不能單憑一列確定原因。

## 密度階梯消融（adopt() 版本）

問題：成群後立刻 `adopt()` 自學時，密度階梯是否仍縮短發現延遲、且不降低候選群純度？階梯的原始消融只在 `confirm()` 用完整配置池重擬合的版本量過（commit 9116bb3）；`adopt()` 版的隔離區會留著前一段的殘餘，條件不同，要重做。

方法：
- 資料：T1／8000rpm，預設序列由輕到重注入九種未知配置，每段最多 600 tick。
- 兩組：`adaptive_density=True`（現行：(25,3)，隔離區≥75 加試 (15,3)、≥100 加試 (10,2)）與 `--fixed-density`（只用 (25,3)）。其他參數全用預設：隔離線 2.0、候選至少 25 筆、每 10 筆新隔離樣本試一次。
- seed 42、123、2026，每組各跑一次 CLI；兩組在同 seed 下的初始切分與抽樣序列相同，差別只在 HDBSCAN 的密度參數。
- 指令：

```bash
uv run --locked python -m experiments.exp2_scale_growth --seed 42
uv run --locked python -m experiments.exp2_scale_growth --seed 42 --fixed-density
```

比較的量（逐段讀 `stages.csv`）：有幾段在 600 tick 內 learned、`samples_to_candidate`、`cluster_purity`、`matched` 是否等於該段注入的 `config`、`own_holdout_acc`、`healthy_holdout_acc`。

預期：
- 階梯只在隔離區累積到 75 筆以上才介入，所以 7screws 等輕度鬆動段兩組延遲應相同；差別應出現在嚴重鬆動與複合配置段。
- 支持假設：三個 seed 合計，階梯組 learned 段數不少於固定組，介入段的延遲中位數較短，且最低純度與 `matched==config` 段數不比固定組差。
- 假設不成立：階梯組延遲沒有縮短，或純度、`matched==config` 段數明顯變差（較小的 mcs 把殘餘或混合樣本切成候選）。
- 單一工況、三個 seed，只能說明這個工況下的方向，不能外推到其他工況。

結果見 [reports/JW-Albert_20261022_exp2_density_ladder_ablation](../../reports/JW-Albert_20261022_exp2_density_ladder_ablation/README.md)，不回填本節。

## 隔離區清理、邊界退回與外部評估（issue #222）

問題：2026-10-22 三個 seed 的 `adopt()` 版輸出（`output/exp2_scale_growth/2026-10-22-21-46-48`、`22-03-36`、`22-03-57`）顯示，學到的群只有 1～3 段是當段注入配置，其餘多半是前一段留在隔離區的殘餘；4_146screws 從未成為任何新類的多數。自類 holdout 只有 6～7 筆。`adopt()` 也沒有 `confirm()` 的已知類退回。

方法（全部不讀 truth，truth 仍只進事後對照）：
- 外部評估集：每個未知配置池以 `default_rng(seed + 2)` 打亂，取 20%（向上取整）當外部評估集，重播只抽其餘 80%。健康沿用監測器的 holdout。
- 學會後清理隔離區（上節第 6 步）。
- 邊界退回（上節第 4 步），門檻 `known_cal_max` 是目前已知類校準列分數的最大值。
- 新增欄位：`matched_is_config`、`own_holdout_n`、`eval_n`、`eval_known_rate`、`pruned`；`summary.json` 新增 `discovery`、`final_eval`、`known_cal_max`。
- 其他參數不變：T1／8000rpm、預設序列、隔離線 2.0、候選至少 25 筆、密度階梯開啟、每段最多 600 tick；seed 42、123、2026 各跑一次，另跑 `--fixed-density` 對照。

```bash
uv run --locked python -m experiments.exp2_scale_growth --seed 42
uv run --locked python -m experiments.exp2_scale_growth --seed 42 --fixed-density
```

預期：
- 支持修正：三個 seed 合計，同一 `matched` 被學成兩個以上新類的次數比 2026-10-22 版少；`matched_is_config` 的段數不少於舊版（3／1／3）；`healthy_holdout_acc` 不低於舊版範圍。
- `rejected_boundary` 在這個工況應為 0 次：九種故障的分數遠高於校準最大值。出現退回時要逐群查，可能是混合群或門檻太鬆。
- 修正不成立：`matched_is_config` 段數沒有增加，或 `discovery` 裡從未被發現的配置變多（清理把當段樣本也移出隔離區）。
- 外部評估集只有每配置約 40～70 列，仍是同一馬達、同一轉速，不是跨個體驗證。

結果見 [reports/JW-Albert_20261023_exp2_adopt_fix_aero](../../reports/JW-Albert_20261023_exp2_adopt_fix_aero/README.md)（Aero）；macOS 版 [2026-10-22 報告](../../reports/JW-Albert_20261022_exp2_adopt_fix/README.md) 不作為數字來源（#224）。不回填本節。

## 判進對應新類與健康混入（issue #243、#240）

問題一（#243）：外部評估集的 `known_rate` 只要求被任一非健康已知類接受，不檢查是否判進「事後多數等於該配置」的新類，所以是認得率的上限。

問題二（#240）：批次回放每段只抽當段注入配置，健康只在段末用 holdout 檢查；隔離線 2.0 擋健康誤報這件事，在回放中沒有被直接檢驗。

方法（皆不讀 truth 做決策，truth 只進事後對照）：
- `final_eval` 每個配置與每段列加 `matched_rate`／`eval_matched_rate`：外部評估集被判進「`matched` 等於該配置」的任一新類的比例。`matched` 是學成當下候選群的事後多數。
- `--healthy-mix p`（預設 0，維持現狀）：每個 tick 以機率 p 改抽一筆健康 holdout 列送進 `process()`，rng 為 `default_rng(seed + 3)`。健康 holdout 不參與擬合與校準；段末的 `healthy_holdout_acc` 仍用同一份 holdout，兩者重疊，報告時寫明。每段記 `healthy_seen`（送進的健康筆數）與 `healthy_quarantined`（其中進入隔離區的筆數）。延遲計數含健康筆。
- 其他參數不變。Aero 上跑 seed 42、123、2026，`--healthy-mix 0` 與 `0.3` 各一次。

```bash
uv run --locked python -m experiments.exp2_scale_growth --seed 42 --healthy-mix 0.3
```

預期：
- `matched_rate` ≤ `known_rate`；兩者差距大的配置，代表被判成了別的新類。
- 混入 0.3 時，健康窗進入隔離區的比例遠低於 5%（隔離線 2.0 遠高於健康誤報的 1.01–1.08）；不形成事後多數為 8screws 的新類。
- 假設不成立：出現事後多數為 8screws 的 `discovered-N`，或 `healthy_holdout_acc` 明顯低於不混入時。
- 學到當段配置的段數可能因延遲拉長而改變；比較時看同 seed 的 0 與 0.3。

結果寫進 `reports/`，不回填本節。

## 來源與對應程式

Campello、Moulavi與Sander（2013），*Density-Based Clustering Based on Hierarchical Density Estimates*，PAKDD／LNCS7819,160–172，[DOI](https://doi.org/10.1007/978-3-642-37456-2_14)。隔離2.0、密度階梯、25筆候選與成群後只用該群自學是Lineage約定，不是論文保證；密度階梯的消融也是本專案的操作約定。LW／k-NN來源同實驗一，原始論文來源註記在Mahalanobis檔頭。

[ScaleGrowthSession與run](../../src/experiments/exp2_scale_growth.py)管理隔離／確認；[monitor](../../src/core/monitor.py)管理完整池fit；[src/web/live.py](../../src/web/live.py)管理即時session；[test_exp2_scale_growth](../../src/tests/test_exp2_scale_growth.py)、[test_monitor_guard](../../src/tests/test_monitor_guard.py)、[test_geometry](../../src/tests/test_geometry.py)、[test_web_experiments](../../src/tests/test_web_experiments.py)測狀態與編排，不證明候選物理真值、獨立採集或長期部署。
