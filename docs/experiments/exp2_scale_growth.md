# 實驗二：未知樣本分群與量尺擴張

## 資料角色用量表

流程：健康基準 → 超線未知隔離 → HDBSCAN找密集群 → 只用該群樣本切分並重擬合 → 事後對照真實配置。[用量依據與分母](README.md#資料用量與分母怎麼讀)為實際載入與splitter重播，不是一次新發現實驗。

| 資料角色 | 馬達／轉速／配置 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 初始train／cal／holdout | T1／8000rpm／健康 | 187／62／64 | 各除以313健康列 | 基準、門檻、健康測試 | 三用途列級互斥 |
| 發現來源 | 同工況5screws | 212可用列 | 212/212全池；到達數查session | 循環重播、隔離及分群 | 可反覆抽同列，不是212次採集 |
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

資料來自單一馬達／轉速的105維clean特徵CSV；來源、數值欄選取、非有限列移除、未做去重及單位／raw-window未知，見[實驗一](exp1_cold_start.md#資料與載入)。T-code代表不同個體；螺絲目錄代表配置。初始只用健康池列隨機60/20/20，seed42；train fit RobustScaler與LW或k-NN參考，calibration定0.95分位數，健康holdout供評估。legacy的train門檻例外同實驗一。

未知注入用全配置池，不只holdout。`CycleSampler`每圈洗牌、抽完重用，rng為seed+1；600個tick不代表600次獨立採集。成群後 `add_observed` 只把這一群做 60/20/20，和既有已知類一起重新 fit scaler、detector 與 PCA。不呼叫 `add_class(config)`，因此不會把該配置的完整池拿來學。新類 holdout 仍來自同一群。

這是統計模型重擬合與近鄰索引更新，沒有epoch或optimizer。幾何由PolarMap重建，見[實驗四](exp4_polar_map.md)；方向不改Open Set拒絕結果。

## 候選如何出現與自學

1. `ScaleGrowthSession.process(x, truth)`先打分、分類。score嚴格大於1才未知；未知且嚴格大於隔離線2.0才存隔離區。1～2的邊界未知不分群。truth 只存起來供事後對照，不參與要不要學。
2. 隔離區至少25筆、距上次嘗試至少10筆新隔離樣本，才在縮放後特徵跑HDBSCAN。先(25,3)，累積75筆加試(15,3)，100筆加試(10,2)；候選仍至少25筆，噪聲-1不當新類。
3. 選足夠大的叢。HDBSCAN fit只收特徵。`_try_cluster` 會順便算出 truth 多數與純度，但 `adopt()` 不拿它當類名，也不拿它決定是否納入。
4. 群至少 5 筆就 `learned`：類名 `discovered-N`，只移除這一群在隔離區的列。樣本不足則 `insufficient`，不擴張。不再因 truth 多數是已知類而退回。
5. 批次與邊跑邊畫都在第一個 learned 結束該段。每段最多 600 tick。API 可設 max_ticks_per_stage，CLI 沒有這個選項。`learned` 是模型新類，`matched` 才是事後多數配置，兩者不必相同。

這不是完整配置池的 oracle 更新。首頁即時展示的 `confirm()` 仍是另一條手動確認路徑。

## 怎麼執行與操作

在repo根目錄，先備妥[uv環境](../runtime_policy.md)與data，最小指令：

```bash
uv run --locked python -m experiments.exp2_scale_growth --help
uv run --locked python -m experiments.exp2_scale_growth --data-root data --motor T1 --rpm 8000rpm --seed 42
uv run --locked python -m experiments.exp2_scale_growth --data-root data --sequence 5screws 3_14screws --openset-method knn --knn-neighbors 5
```

`run(pools, sequence=None, seed=42, max_ticks_per_stage=600)`回傳dict、不自行存檔；`ScaleGrowthSession(pools)`供逐筆 process，成群後呼叫 `adopt()`。共用CLI支援confidence0.95、method預設ledoit_wolf；密度階梯與隔離線須用API，CLI未暴露。

Web啟動同[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)。「實驗二」卡選資料集／方法／seed，按「▶ 執行」會跑整個自學劇本。「⏵ 邊跑邊畫」依序自動注入各配置並逐筆畫分數，成群後立刻用該群自學並繼續；「■ 停止」取消後續串流。首頁即時展示仍是另一個持續 session，確認按鈕留在那一頁，不代表實驗二還在等操作員命名。

缺配置時核對實際資料夾名（1screw／1screws可能不同），不要自行猜名稱。沒有candidate時confirm會RuntimeError。重擬合失敗沒有原子回滾：停止使用、重建健康基準，不以舊scaler與新模型混用。每次CLI建立時間戳目錄，勿同秒並行；沒有此實驗續跑旗標。

## 輸出怎麼看與判斷

[main writer](../../src/experiments/exp2_scale_growth.py)寫`logs/exp2_scale_growth/{ts}.log`、`output/exp2_scale_growth/{ts}/environment.json`、`stages.csv`、`summary.json`；Web只由runner存結果JSON到`output/web_server/{ts}/experiments/`。模型不長期保存。

- `config`是注入劇本，`learned`是模型新類，`matched`是事後多數配置；三者要一起看。
- `discovered=false`表示該段未learned；不能當作故障不存在。失敗段不一定有delay等欄。
- `samples_to_candidate`是該段到首次learned的tick；單位筆，不是秒。循環抽樣會重用列。
- `cluster_purity`是事後 truth 的叢多數比例，不是模型當時知道的故障名稱。
- `own_holdout_acc`是新類自己的 holdout 有沒有被分回這個新類；`healthy_holdout_acc`是健康有沒有被保住。兩者都含拒絕造成的錯誤。
- `known_after`、`final_known`和model逐類數量記錄擴張狀態。

預期在600tick內發現配置且保持健康辨識；未預定獨立部署通過線。低純度支持混叢問題，未發現也可能是隔離／密度或資料不足，不能單憑一列確定原因。

## 來源與對應程式

Campello、Moulavi與Sander（2013），*Density-Based Clustering Based on Hierarchical Density Estimates*，PAKDD／LNCS7819,160–172，[DOI](https://doi.org/10.1007/978-3-642-37456-2_14)。隔離2.0、密度階梯、25筆候選與成群後只用該群自學是Lineage約定，不是論文保證。LW／k-NN來源同實驗一，Ancestor來源註記在Mahalanobis檔頭。

[ScaleGrowthSession與run](../../src/experiments/exp2_scale_growth.py)管理隔離／確認；[monitor](../../src/core/monitor.py)管理完整池fit；[src/web/live.py](../../src/web/live.py)管理即時session；[test_monitor_guard](../../src/tests/test_monitor_guard.py)、[test_geometry](../../src/tests/test_geometry.py)、[test_web_experiments](../../src/tests/test_web_experiments.py)測狀態與編排，不證明候選物理真值、獨立採集或長期部署。
