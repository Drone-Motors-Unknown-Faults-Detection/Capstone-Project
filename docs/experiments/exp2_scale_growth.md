# 實驗二：未知樣本分群與量尺擴張

## 資料角色用量表

流程：健康基準 → 超線未知隔離 → HDBSCAN找密集群 → 模擬確認 → 完整配置池重新切分及fit → 檢查更新後分類。[用量依據與分母](README.md#資料用量與分母怎麼讀)為實際載入與splitter重播，不是一次新發現實驗。

| 資料角色 | 馬達／轉速／配置 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 初始train／cal／holdout | T1／8000rpm／健康 | 187／62／64 | 各除以313健康列 | 基準、門檻、健康測試 | 三用途列級互斥 |
| 發現來源 | 同工況5screws | 212可用列 | 212/212全池；到達數查session | 循環重播、隔離及分群 | 可反覆抽同列，不是212次採集 |
| 確認後train／cal／holdout | 完整5screws池 | 127／42／43 | 各除以212 | 與健康共同重fit及評估 | 可能與發現時到達列重疊 |

確認後兩類合計train314、cal104、holdout107；不是只把候選群25筆拿來學。候選、隔離與實際tick數須讀該次輸出，計數工具沒有執行HDBSCAN，不能預填發現延遲。無另設validation選參。

## 發表與理解檢查

口頭主軸：「先拒絕陌生資料，等特徵形成群才提出候選。確認模擬揭露標籤，再拿整份已確認配置建立新參考。最後看它有沒有被分對、健康有沒有退步。」示意：30筆過隔離線未必形成25筆的密集群，因此未知召回高也可能發現失敗；這不是實測延遲。

- HDBSCAN學了故障名稱嗎？沒有，它依密度分群；名稱來自模擬truth多數，程式在candidate階段已有快取。
- 學會是否表示馬達恢復？沒有，只是模型新增一個已知配置。
- 更新後測試是否全新？否，完整配置池可能已用於發現，holdout不能改稱fresh。
- 組員請重述「拒絕、隔離、候選、確認、更新」五步；數字及失敗條件查下方輸出，不只背一個準確率。

系統先只認識健康，看到陌生樣本時先隔離，累積成叢後再確認是否加入已知類。這個實驗承接[冷啟動](exp1_cold_start.md)，檢查「發現候選」和「加入後認得它」兩件不同的事；不是沒有標註就自動知道物理故障原因。

## 資料、切分與模型

資料來自單一馬達／轉速的105維clean特徵CSV；來源、數值欄選取、非有限列移除、未做去重及單位／raw-window未知，見[實驗一](exp1_cold_start.md#資料與載入)。T-code代表不同個體；螺絲目錄代表配置。初始只用健康池列隨機60/20/20，seed42；train fit RobustScaler與LW或k-NN參考，calibration定0.95分位數，健康holdout供評估。legacy的train門檻例外同實驗一。

未知注入用全配置池，不只holdout。`CycleSampler`每圈洗牌、抽完重用，rng為seed+1；600個tick不代表600次獨立採集。新配置確認後，`add_class(config)`對該配置的**完整資料池**另做60/20/20，和全部已知類一起重新fit scaler、detector與PCA。已到達的未知樣本可能落入之後的train、cal或holdout；擴張後holdout不是未曝光盲測，也不是完全線上、只用已到達樣本更新。

這是統計模型重擬合與近鄰索引更新，沒有epoch或optimizer。幾何由PolarMap重建，見[實驗四](exp4_polar_map.md)；方向不改Open Set拒絕結果。

## 候選如何出現與確認

1. `ScaleGrowthSession.process(x, truth)`先打分、分類。score嚴格大於1才未知；未知且嚴格大於隔離線2.0才存隔離區。1～2的邊界未知不分群。
2. 隔離區至少25筆、距上次嘗試至少10筆新隔離樣本，才在縮放後特徵跑HDBSCAN。先(25,3)，累積75筆加試(15,3)，100筆加試(10,2)；候選仍至少25筆，噪聲-1不當新類。
3. 選足夠大的叢。HDBSCAN fit只收特徵，但**_try_cluster在產生candidate時已計算truth多數、purity及is_known**；不是到confirm才第一次讀標籤。confirm消費預先算好的majority，模擬操作員確認，不能當真實無標籤人工鑑定流程。
4. 已知多數叢回傳`rejected_known`並清除該叢；未知多數回傳`learned`、完整池重擬合、清除該配置的隔離列。剩餘其他配置可繼續累積。
5. 批次run預設按配置排序逐段注入，每段最多600tick；自動confirm，第一個learned即停該段。API可設max_ticks_per_stage，CLI沒有這個選項。候選可能來自前一段，`learned`不一定等於`config`。

標籤預算與完整池更新是程式現況；本文件不把它改寫成嚴格盲化的持續學習保證。

## 怎麼執行與操作

在repo根目錄，先備妥[uv環境](../runtime_policy.md)與data，最小指令：

```bash
uv run --locked python -m experiments.exp2_scale_growth --help
uv run --locked python -m experiments.exp2_scale_growth --data-root data --motor T1 --rpm 8000rpm --seed 42
uv run --locked python -m experiments.exp2_scale_growth --data-root data --sequence 5screws 3_14screws --openset-method knn --knn-neighbors 5
```

`run(pools, sequence=None, seed=42, max_ticks_per_stage=600)`回傳dict、不自行存檔；`ScaleGrowthSession(pools)`供逐筆process／confirm。共用CLI支援confidence0.95、method預設ledoit_wolf；密度階梯與隔離線須用API，CLI未暴露。

Web啟動同[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)。「實驗二」卡選資料集／方法／seed，按「▶ 執行」會跑整個自動確認劇本，沒有邊跑邊畫或人工停住候選的操作。要人工確認，切「即時展示」，開始播放、選故障來源，等候選卡出現再按確認；暫停與重設是session操作。兩入口不是同一操作流程，畫面預先模擬truth不能算實體診斷。

缺配置時核對實際資料夾名（1screw／1screws可能不同），不要自行猜名稱。沒有candidate時confirm會RuntimeError。重擬合失敗沒有原子回滾：停止使用、重建健康基準，不以舊scaler與新模型混用。每次CLI建立時間戳目錄，勿同秒並行；沒有此實驗續跑旗標。

## 輸出怎麼看與判斷

[main writer](../../src/experiments/exp2_scale_growth.py)寫`logs/exp2_scale_growth/{ts}.log`、`output/exp2_scale_growth/{ts}/environment.json`、`stages.csv`、`summary.json`；Web只由runner存結果JSON到`output/web_server/{ts}/experiments/`。模型不長期保存。

- `config`是注入劇本，`learned`是實際加入的多數配置；兩者要一起看。
- `discovered=false`表示該段未learned；不能當作故障不存在。失敗段不一定有delay等欄。
- `samples_to_candidate`是該段到首次learned的tick；單位筆，不是秒。循環抽樣會重用列。
- `cluster_purity`是已知truth的叢多數比例，非未知診斷準確率；`rejected_known_clusters`記退回次數。
- `own_holdout_acc`、`healthy_holdout_acc`含拒絕造成的錯誤；屬更新後、可能已曝光的CSV測試。
- `known_after`、`final_known`和model逐類數量記錄擴張狀態。

預期在600tick內發現配置且保持健康辨識；未預定獨立部署通過線。低純度支持混叢問題，未發現也可能是隔離／密度或資料不足，不能單憑一列確定原因。完整歷史結果保留在[固定版本原紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/experiments/exp2_scale_growth.md#已記錄的實測)，不抄為現行分數。

## 來源與對應程式

Campello、Moulavi與Sander（2013），*Density-Based Clustering Based on Hierarchical Density Estimates*，PAKDD／LNCS7819,160–172，[DOI](https://doi.org/10.1007/978-3-642-37456-2_14)。隔離2.0、密度階梯、25筆候選與自動confirm是Lineage約定，不是論文保證。LW／k-NN來源同實驗一，Ancestor來源註記在Mahalanobis檔頭。

[ScaleGrowthSession與run](../../src/experiments/exp2_scale_growth.py)管理隔離／確認；[monitor](../../src/core/monitor.py)管理完整池fit；[src/web/live.py](../../src/web/live.py)管理即時session；[test_monitor_guard](../../src/tests/test_monitor_guard.py)、[test_geometry](../../src/tests/test_geometry.py)、[test_web_experiments](../../src/tests/test_web_experiments.py)測狀態與編排，不證明候選物理真值、獨立採集或長期部署。
