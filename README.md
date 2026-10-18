# Capstone Project — 馬達故障診斷與持續學習系統

**冷啟動 PHM（Cold-start Prognostics and Health Management）**：只用「健康」資料起步的馬達未知故障即時偵測、健康量尺自動擴張與持續學習。

---

## 研究動機

企業導入 PHM 系統時有一個雞生蛋問題：

> 故障診斷模型需要故障資料才能訓練，但設備在導入初期是新的、不會故障——
> **收集資料的階段，PHM 系統的產值等於零。**

本專題把這個順序反過來：**系統從第一天就只用健康資料建立基準**，之後的一切
（發現未知故障、為故障命名、擴張診斷能力、判別故障發生模式）都在運轉過程中自己長出來。

原始論文（見 [與 legacy 的關係](#與-legacy-的關係)）假設一開始就認識 5 類故障；
本專題問的是：**如果一開始只認識健康呢？**

### 核心構想

1. **單錨點量尺**：只要有健康分布，「樣本到健康分布的馬氏距離」本身就是健康度量尺——
   距離小 = 健康、慢慢變大 = 退化中、突然跳大 = 出事了。不需要等第一筆故障才有產值。
2. **星狀幾何**：在特徵空間中，健康群是中心，每種故障模式是一條射線——
   **距離 = 嚴重程度、方向 = 故障種類**。新異常「更遠」不代表「同一種病更嚴重」，
   可能是另一種病；因此用 HDBSCAN 對未知樣本分群來「發現新方向」。
3. **量尺隨新群擴張**：未知樣本累積成穩定叢集 → 只用這一群樣本自學一個新類、
   整組重新擬合 → 之後同種故障直接被認出（開放集 + 持續學習）。首頁即時展示另保留
   操作員手動確認的版本。
4. **發生模式判別**：正常磨損是慢速漂移（8 顆螺絲 → 慢慢鬆），外力損傷是跳變
   （撞擊 → 8 顆直接變 4 顆）；監控異常密度的時間序列即可區分兩者。

### 系統成熟度三階段

```
階段 0（只有健康資料）：105 維特徵 + 單類馬氏距離   → 能報「異常了」（day-1 可用）
階段 1（第一種故障確認）：量尺有了第二個刻度         → 能報「哪一種、多接近」
階段 N（多種故障累積）：逐類馬氏 + 新方向發現        → 完整開放集診斷，持續擴張
```

---

## 實驗

實驗各自是獨立模組（`experiments/`），可單獨批次執行產出論文數據；
Web 即時展示（`web/`）呼叫同一份模組邏輯，兩邊數字一致。

### 實驗一：冷啟動偵測能力（`exp1_cold_start`）

**問題**：只用 8screws（Healthy）擬合健康基準，能多準地偵測九種從未見過的故障？

**方法**：Mahalanobis–Taguchi 式單類監測——RobustScaler + Ledoit–Wolf 收縮共變異數，
以獨立校準集（calibration split）的第 95 百分位距離為閾值，正規化分數 > 1 判為未知。

```bash
venv/bin/python -m experiments.exp1_cold_start --motor T1 --rpm 8000rpm
```

**實測結果**（T1/8000rpm，seed=42）：

| 注入配置 | 樣本數 | 偵測率 | AUROC | 分數中位數 |
|---|---:|---:|---:|---:|
| 8screws（健康保留集）| 64 | 10.9%（誤報）| — | 0.86 |
| 7screws | 322 | 100% | 1.0000 | 37.84 |
| 6screws | 314 | 100% | 1.0000 | 50.18 |
| 5screws | 212 | 100% | 1.0000 | 83.93 |
| 4screws | 287 | 100% | 1.0000 | 86.98 |
| 3screws | 241 | 100% | 1.0000 | 78.30 |
| 2screws | 288 | 100% | 1.0000 | 69.47 |
| 1screws | 349 | 100% | 1.0000 | 115.23 |
| 3_14screws | 351 | 100% | 1.0000 | 45.62 |
| 4_146screws | 372 | 100% | 1.0000 | 13.21 |

> 兩個值得寫進討論的觀察：① 健康誤報率 10.9% 高於名目值 5%，來源是校準集僅 62 筆的
> 有限樣本效應；② 分數中位數大致隨鬆動嚴重度上升（7→1 screws），支持「距離 ≈ 嚴重度」
> 的量尺假設，但 4_146（複合鬆動）距離明顯偏低——複合故障不在均勻鬆動的射線上。

### 實驗二：量尺擴張（`exp2_scale_growth`）

**問題**：未知故障陸續出現時，系統能否自動發現、命名並納入，同時保住既有能力？

**方法**：未知樣本進隔離區 → 累積 ≥ 25 筆後以 HDBSCAN（min_cluster_size=25,
min_samples=3）找穩定叢集 → 產生「候選新故障」→ `adopt()` 只用這一群樣本切分並重擬合，
類名為 `discovered-N`（群不到 5 筆不納入）。偵測、分群與納入全程不讀真實標籤；候選群的
真值多數與純度只寫進事後對照欄 `matched`／`cluster_purity`。採**兩段閾值**：偵測線 1.0
（判未知／警報），隔離線 2.0（分數明顯超線才參與新類發現）——已知類別的邊界誤報
只些微超線（實測 1.01–1.08）、真實故障遠在其上（最低 ≥ 10），隔離線落在空隙中，
避免誤報累積自聚成假候選。分群採**自適應密度階梯**：先用嚴格參數 (25,3)，
隔離區 ≥ 75 筆加試 (15,3)、≥ 100 筆加試 (10,2)——發散故障（嚴重鬆動、複合配置）
在固定參數下要 200~600 筆才成叢甚至失敗，放寬的只是 HDBSCAN 的密度要求，
候選門檻（叢 ≥ 25 筆）與純度不受影響。

```bash
venv/bin/python -m experiments.exp2_scale_growth --motor T1 --rpm 8000rpm
# 或指定注入順序：
venv/bin/python -m experiments.exp2_scale_growth --sequence 5screws 3_14screws 2screws
```

**實測結果**（依鬆動由輕到重全序列注入，T1/8000rpm，seed=42，自適應密度階梯）：

| 注入配置 | 發現延遲（筆）| 叢集大小 | 純度 | 自類 holdout 準確率 | 健康保持率 | 已知類別數 |
|---|---:|---:|---:|---:|---:|---:|
| 7screws | 105 | 31 | 100% | 100% | 95.3% | 2 |
| 6screws | 255 | 87 | 100% | 100% | 93.8% | 3 |
| 5screws | 205 | 25 | 100% | 93.0% | 93.8% | 4 |
| 4screws | 105 | 37 | 100% | 96.6% | 92.2% | 5 |
| 3screws | 215 | 101 | 100% | 85.7% | 92.2% | 6 |
| 2screws | 105 | 74 | 100% | 84.7% | 93.8% | 7 |
| 1screws | 65 | 32 | 100% | 90.1% | 92.2% | 8 |
| 3_14screws | 225 | 37 | 100% | 85.9% | 90.6% | 9 |
| 4_146screws | 265 | 30 | 100% | 92.0% | 93.8% | 10 |

> 三個發現：① **十種配置全部成功發現與學會**，所有叢集純度 100%——HDBSCAN
> 沒有把不同故障混成一群；② 已知類別越多，逐類閾值越擠，自類準確率從 100%
> 緩降到 ~85%，是量尺擴張的代價；③ 發散故障（嚴重鬆動與複合配置）在固定
> mcs=25 下極難成叢——實測 2screws 需 225 筆、3_14 在 600 筆內失敗；自適應
> 密度階梯把全部發現延遲壓到 65~265 筆，且不犧牲純度（消融對照見 git 歷史
> 中改動前的表格）。

### 實驗三：漸進磨損 vs 突發事件判別（`exp3_trend`）

**問題**：能否從異常出現的「時間模式」判別故障是慢速磨損還是外力損傷？

**方法**：對開集分數做逐樣本旗標（>1 = 異常），以 EWMA 平滑成異常密度；
密度越過 0.5 觸發警報，同時回看警報前的軌跡，計算密度停留在「中間帶」[0.2, 0.5)
的時間——漸進退化會在中間帶徘徊很久，突發跳變只是快速穿過（≤ 12 筆判突發）。
另計 CUSUM（Page, 1954）作為偏移能量的展示統計量。

兩個劇本（`SCENARIOS`，Web 介面共用同一份定義）：

- **劇本 A（漸進磨損）**：健康 → 25% 混入 7screws → 70% → 6screws → 5screws
  （模擬螺絲逐漸鬆脫的早期間歇性症狀與惡化）
- **劇本 B（突發外力）**：健康 → 直接切換 4screws（模擬異物撞擊的狀態跳變）

```bash
venv/bin/python -m experiments.exp3_trend --motor T1 --rpm 8000rpm --trials 40
```

**實測結果**（40 次隨機重複，T1/8000rpm）：

| 劇本 | 警報率 | 判別正確率 | 平均警報延遲 | 平均中間帶停留 |
|---|---:|---:|---:|---:|
| A：漸進磨損 | 100% | 98% | 49.8 筆 | 37.9 筆 |
| B：突發外力 | 100% | 98% | 7.7 筆 | 6.7 筆 |

> 中間帶停留時間的兩個分布（B：5–13 筆、A：12–60 筆）僅在 12–14 有少量重疊，
> 98% 的判別正確率反映的是真實的分布交界，非參數硬湊。

### 實驗四：極座標健康地圖（`exp4_polar_map`）

**問題**：把「星狀幾何」（健康為中心、故障各成射線）做成可量測機制——新異常能否
自動判別是「已知故障家族的更嚴重版本」（同方向更遠）還是「真正的新故障模式」
（新方向）？嚴重度能否從距離回復？

**方法**（`src/core/geometry.py`）：以健康類 Ledoit–Wolf precision 的 Cholesky 白化
特徵空間，健康中心為原點——半徑 = 馬氏距離（嚴重度原始量）、每個已知故障類
= 一條射線；逐樣本算**方向熟悉度**（對各射線的最大 cos，連續分數）。

```bash
venv/bin/python -m experiments.exp4_polar_map --motor T1 --rpm 8000rpm
```

**實測結果**（T1/8000rpm，seed=42）：

- **(a) 射線結構**：中段均勻鬆動（6~2screws）共享一束緊密射線（族內平均 cos
  0.92）；4_146 是明確的獨立方向（vs 均勻族 0.49）；3_14 介於其間（0.76）——
  不對稱鬆動在方向上仍屬「鬆動家族」。星狀假設**部分成立**，且結構可量測。
- **(b) 方向熟悉度**（家族成員 vs 4_146 新模式的 AUROC）：

  | 冷啟動後第一個學到的故障 | AUROC | 意義 |
  |---|---:|---|
  | 7screws（最輕）| **0.997** | 輕度錨點幾乎完美分辨「同家族惡化」vs「新模式」|
  | 5screws（中度）| **1.000** | 中段錨點最佳 |
  | 7s + 1s 兩端 | 0.889 | 極端錨點拉低判別力 |
  | 1screws（最重）| 0.146（反轉）| **從最嚴重端當錨點會誤判**——新模式反而看起來更熟悉 |

  對冷啟動敘事的意義：真實部署最先遇到的是輕度故障，而**輕度錨點正是最好的
  方向羅盤**——系統在只認識一種輕故障時，就能近乎完美地預判「後續惡化」與
  「全新故障」該走不同處理路徑。
- **(c) 嚴重度可回復性**（錨點 = 7s 與 1s，對 6~2screws 排序）：純半徑（馬氏距離）
  Spearman **ρ=0.877**、主束投影 0.808、最佳射線投影 0.194——嚴重度**粗粒度可
  回復**（輕重兩端明確分離），細粒度受特徵空間非單調摺疊限制（5s>4s>3s 距離倒置，
  與 exp1 分數中位數一致）。

Web 端同步整合：即時極座標健康地圖（圓心=健康、✕=已知故障射線端點、點的角度
偏離=方向陌生度）＋狀態卡逐筆方向熟悉度。

### 實驗五：跨工況冷啟動泛化（`exp5_cross_condition`）

**問題**：在某工況（壽命期 × 轉速）建的健康基準能否跨工況共用？老化造成多大漂移？

```bash
venv/bin/python -m experiments.exp5_cross_condition
```

**實測結果**（9 組資料集，seed=42）：

- **(a) 9×9 遷移矩陣**：自身 AUROC 全部 1.0；跨工況平均 0.893、**最低僅 0.10**——
  健康基準不可任意搬移（同轉速跨壽命 0.921、同壽命跨轉速 0.928）。
- **(b) 部署策略**：逐工況基準（健康接受 97.4%／故障偵測 100%）**優於**全混合
  （96.8%／99.3%）與同轉速混合（94.9%／100%）——以數據支持本系統
  「每個工況各自冷啟動」的架構選擇。
- **(c) 壽命期漂移**：T1（新機）基準下，T2/T3 的「健康」資料 **100% 被判未知**
  （分數中位 14~86）——「老機的健康 ≠ 新機的健康」。註：T1/T2/T3 為不同馬達
  個體，此位移為個體差＋老化的疊加，無法分離；結論不變——健康基準必須
  **逐機台**建立，而這正是冷啟動方法的主場。

### 實驗六：開集偵測方法基準比較（`exp6_osr_benchmark`）

**問題**：主線的逐類 Ledoit–Wolf 馬氏距離相對其他單類/異常偵測方法是否合理？

**方法**（`src/core/detectors.py`）：七種方法統一資料切分與校準哲學（校準集 95 百分位
= 門檻），9 組資料集各測一輪取平均——LW 馬氏（主線）、legacy 全域馬氏（論文版
對照）、OC-SVM、Isolation Forest、LOF、kNN 距離、PCA 重建誤差。

```bash
venv/bin/python -m experiments.exp6_osr_benchmark
```

**實測結果**：**七種方法 AUROC 全部 1.0000**——在 105 維特徵上偵測本身不是瓶頸，
方法差異全在校準品質（健康誤報率）：kNN 6.0% ≲ PCA 6.4% ≲ LOF 6.7% ≲
**主線 LW 馬氏 7.1%** < OC-SVM 7.4% < iForest 8.4% ≪ **legacy 83.1%（崩潰）**。
三個結論：① 特徵工程是主角、偵測器選擇是配角；② legacy 以訓練距離自校準的
做法在高維＋單類＋小樣本的冷啟動情境失效——**獨立校準集是生死線**；
③ 主線續用 LW 馬氏合理：誤報僅差最佳者 1 個百分點，卻保有參數化模型才有的
幾何地圖（實驗四）、逐類擴張與可解釋性。

### 實驗七：Open Set 方法切換與單工況公平比較（`compare_openset`）

程式 `src/experiments/compare_openset.py`，技術報告 [docs/experiments/exp7_compare_openset.md](docs/experiments/exp7_compare_openset.md)。
所有實驗及 Web server 都可用同一個 CLI 介面切換 detector：

```bash
# 既有方法（預設）：逐類 Ledoit–Wolf Mahalanobis
venv/bin/python -m experiments.exp1_cold_start --openset-method mahalanobis --method ledoit_wolf

# 新方法：逐類 k-NN 平均歐氏距離
venv/bin/python -m experiments.exp1_cold_start --openset-method knn --knn-neighbors 5

# 兩者使用相同 seed、60/20/20 split、known/unknown 定義與指標
venv/bin/python -m experiments.compare_openset --openset-methods mahalanobis knn   # 實驗七
```

`k-NN` 不需要分類 logits 或神經網路 embedding，直接使用本專案既有、經
`RobustScaler` 處理的 105 維特徵。對每個已知類別計算「查詢樣本到該類 training
split 的 k 個最近鄰平均歐氏距離」，再除以該類 calibration split 距離的第 95
百分位；跨類取最小值，正規化分數嚴格大於 1 判為 unknown。分數越大代表越不像
任何已知類別。此方法的鄰近性基礎源自
[Cover & Hart (1967)](https://doi.org/10.1109/TIT.1967.1053964)，以近鄰距離作異常分數
的依據可追溯至
[Ramaswamy, Rastogi & Shim (2000)](https://doi.org/10.1145/342009.335437)。

公平比較輸出到 `logs/openset_comparison/` 與 `output/openset_comparison/`，包含
`summary.csv/json`、逐配置 `details.csv` 與比較圖；unknown 為正類，紀錄 AUROC、
AUPR、FPR@95TPR、Open Set accuracy、known-class accuracy、unknown precision／
recall／F1、正規化 threshold、每類原始 threshold、校準策略、split 與 seed。
unknown/test 樣本不參與 threshold 選擇。

實驗六正式版矩陣（`exp6_formal_benchmark` + `exp6_matrix` + `aggregate_exp6`）以 9 工況 ×
3 seed（42/123/2026）比較 Mahalanobis 與 k-NN，並記錄資料指紋、commit 等 provenance：

```bash
venv/bin/python -m experiments.exp6_matrix
```

結果見 `output/exp6_formal_matrix/aggregate/aggregate.md`；兩方法 AUROC 皆 1.0、Open Set
accuracy 差異 < 0.00001。

### 實驗八：相對健康指數與健康監測（`health_index_*`、`health_monitor`）

程式 `src/experiments/health_index_benchmark.py`、`src/experiments/health_index_matrix.py`、
`src/experiments/health_monitor.py`，邏輯在 `src/experiments/health/`。把開集分數依 `8screws` 校準分數的
第 10／95 百分位映成 0～1 的相對健康指數，再分四段嚴重度、逐 session 追蹤趨勢告警。

```bash
uv run --locked python -m experiments.health_index_matrix --data-root data/formal_local --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn
```

9 工況 × 3 seed × 2 方法的封存摘要：known−unknown 健康差距 0.583（Mahalanobis）／0.602（k-NN），
AUROC 皆 1.0；所有 unknown 的健康指數都是 0.0，分得開但排不出輕重。現有資料沒有馬達
編號、時間戳與真實損傷標記，不能把它當成損壞百分比或 RUL。結果在
`reports/exp8_health_index_results/`，操作見
[docs/experiments/exp8_health.md](docs/experiments/exp8_health.md)。

### 實驗說明

名詞、畫面與各實驗在問什麼：
[docs/experiments/README.md](docs/experiments/README.md)。

---

## 原創貢獻總覽

相對於學長論文（假設一開始就有 5 類標記故障、批次訓練），本專題的原創內容：

| # | 貢獻 | 證據 |
|---|---|---|
| 1 | **冷啟動框架**：健康-only 起步的開集診斷與持續學習迴圈 | exp1（九故障偵測 100%）、exp2（十類全自動學會）|
| 2 | **兩段閾值**：偵測線 1.0／隔離線 2.0 分離「警報」與「新類發現」 | 健康誤報 1.01–1.08 vs 故障 ≥10 的空隙實測 |
| 3 | **自適應密度階梯**：發散故障的發現延遲 600+→265 筆內 | exp2 消融（git 歷史保留改動前表格）|
| 4 | **候選群自學與操作員在環**：實驗二成群即自學、不讀真值；首頁展示保留人工確認＋退回機制 | exp2 事後對照欄；假候選（健康誤報聚叢）實測案例與修復 |
| 5 | **中間帶停留時間**：漸進 vs 突發的變化點判別統計量 | exp3（雙劇本各 98%）|
| 6 | **極座標健康地圖**：方向=故障類型、半徑=嚴重度；錨點效應發現 | exp4（輕錨 AUROC 0.997、重錨反轉 0.146）|
| 7 | **跨工況泛化分析**：9×9 遷移矩陣、部署策略比較、老化漂移量化 | exp5（逐工況策略最優；老機健康 100% 判未知）|
| 8 | **OSR 基準與校準診斷**：legacy 自校準在冷啟動失效的證據 | exp6（83.1% vs 7.1% 誤報）|
| 9 | **非線性 AutoEncoder 偵測器**：瓶頸重建誤差，健康誤報比既有七種方法都低 | exp9（AUROC 1.0、健康誤報 5.8%，八法最低）|
| 10 | **跨工況遷移學習定量化**：目標端少量健康資料時，中心平移 vs 從零冷啟動的具體落差 | exp10（N=50 時健康接受率 9.3% vs 87.9%，支持逐工況冷啟動）|
| 11 | **原始論文協定 Accuracy/F1 對照**：Ledoit–Wolf 相對 legacy 的提升遠超專題 5~10% 門檻 | exp11（Balanced Accuracy +78.38 pp、macro-F1 +82.18 pp，Wilcoxon p≈0）|
| 12 | **開集混淆矩陣與 t-SNE**：把「legacy 無法做多類判別」的結論變成可見的畫面 | exp12（legacy 開集混淆矩陣對角線全 0，ledoit_wolf 已知類別互不混淆）|

另附即時展示系統（web/）與可重現實驗框架（seed 決定論、logs/output 全留檔）。

---

## 即時展示介面（Web）

發表用的儀表板。頁首可切換兩種頁面：

- **即時展示**：把資料集當成「感測器串流」逐筆播放，現場演示實驗一～四的完整劇情（下方功能清單）。
  新類納入走操作員手動確認（`confirm()`），跟實驗頁實驗二的自學不同。
- **實驗一～實驗十二**：每個實驗一頁（實驗九只有說明、沒有執行按鈕），填參數（資料集、Open Set 方法、seed 等）按「執行」，
  伺服器呼叫該實驗模組的 `run()`，結果畫成表格、熱圖與線圖。實驗六、八另外顯示已提交的
  正式矩陣結果（`output/exp6_formal_matrix/aggregate/`、`reports/exp8_health_index_results/`），
  唯讀、不從網頁重跑。每次執行的完整結果存到 `output/web_server/{ts}/experiments/{實驗}_{時間}.json`。
  實驗在另一個執行緒池跑，一次一個；跑實驗時即時展示照常串流。每次按「執行」都會從頭擬合該實驗的
  模型（統計擬合，不用 GPU）；模型跑完即丟、不保存，同 seed 重跑結果相同。每頁會寫出這次擬合哪些模型、約需幾秒。

```bash
./scripts/opt/run_web.sh                      # 預設 T1/8000rpm、http://localhost:8600
./scripts/opt/run_web.sh --port 8600 --motor T1 --rpm 8000rpm --rate 4
```

功能：

- **開集分數串流圖**：每點一筆樣本，虛線 = 未知判定線（1.0），顏色 = 系統判定
  （綠 健康／藍 已知故障／紅 未知），事件以垂直色線標記
- **即時狀態卡**：實際注入（僅展示者可見）vs 系統判定、開集分數、健康度儀表
- **未知故障處理流程 stepper**：監測 → 異常隔離累積（n/25）→ HDBSCAN 分群 →
  操作員確認 → 擴張重訓，跟著真實狀態亮燈
- **候選卡**：叢集大小／純度／**樣本蒐集區間**（區分舊帳與現行串流）／揭示真實身分，
  「✔ 操作員確認」一鍵納入重訓（亞秒完成）；若揭示後發現叢集其實是**已知類別的
  邊界樣本**（如健康誤報累積自聚成叢），按鈕轉為「✋ 操作員退回」——清出隔離區、量尺不變
- **分群進度回饋**：流程列顯示「已試分群 N 次」，HDBSCAN 連續失敗每 5 次發事件說明
  樣本發散、密度門檻將自動放寬——發散故障累積期不會看起來像當機
- **健康量尺**：已知類別 chips，隨確認即時長大
- **變化點分析卡**：EWMA 異常密度、CUSUM、漸進／突發判別結果
- **PCA 特徵空間投影**：已知類別中心（✕）與最近樣本的即時散佈
- **偵測統計**：依「學會前／學會後」拆開呈現——學會前判未知 = 偵測成功（偵測率），
  學會後判已知 = 認出成功（認出率）。認出率天生 ≈ 90–95%，因為校準閾值取第 95
  百分位，已知樣本本來就有少數落在線外
- **控制**：注入來源下拉（十種螺絲配置）、劇本 A／B 一鍵播放、速率 1–10 筆/秒、
  資料集切換（九組 motor×rpm）、重置回階段 0
- **session 持久化**：每次伺服器執行自動保存到 `output/web_server/{ts}/`（實驗頁的結果在其下 `experiments/`）——
  `samples_epoch{N}.csv`（逐筆：t／真實注入／分數／判定／EWMA／CUSUM…）與
  `model_epoch{N}_{K}classes.json`（每次擬合後的逐類閾值與樣本數快照；重置一次算
  一個 epoch）。批次實驗的 `summary.json` 也同樣附上模型快照

**發表演示腳本建議**：

```
1. 開場：階段 0，只認識健康 → 串流健康資料，分數貼地
2. 注入 5screws → 分數暴衝、判定「未知」、變化點警報
3. 隔離區累積到 25+ → HDBSCAN 候選出現 → 按「操作員確認」
4. 量尺擴張為 2 類 → 再注入 5screws → 系統直接認出（藍色）
5. 劇本 A 完整播放 → 警報判「漸進退化」
6. 重置 → 劇本 B → 警報判「突發事件」，對比收尾
```

---

## 目錄結構

```
Lineage/
├── core/                        # 共用零件
│   ├── data.py                  #   資料池載入、60/20/20 切分、循環抽樣
│   ├── mahalanobis.py           #   開集馬氏偵測器（複製自論文版程式碼，含來源註記）
│   ├── openset.py               #   detector 共用介面、工廠與 k-NN 距離拒絕法
│   ├── monitor.py               #   OpenSetMonitor：健康冷啟動 + add_class 量尺擴張
│   ├── geometry.py              #   PolarMap：白化空間、故障射線、方向熟悉度、嚴重度
│   ├── detectors.py             #   實驗六的七種偵測器（統一 fit/score 介面）
│   ├── formal_data.py           #   正式原始資料 → Lineage 資料格式的 materialization
│   ├── trend.py                 #   TrendMonitor：EWMA 異常密度 + CUSUM + 漸進/突發判別
│   ├── logger.py                #   loguru 執行期日誌 + logs/ + output/ 慣例
│   └── runner.py                #   實驗 CLI 共用參數與 JSON 儲存
├── experiments/                 # 實驗模組：各自可獨立執行（python -m experiments.expN_*）
│   ├── exp1_cold_start.py       #   實驗一：冷啟動偵測能力
│   ├── exp2_scale_growth.py     #   實驗二：量尺擴張（含 ScaleGrowthSession 狀態機）
│   ├── exp3_trend.py            #   實驗三：漸進 vs 突發（含 SCENARIOS 劇本定義）
│   ├── exp4_polar_map.py        #   實驗四：極座標健康地圖（射線結構/方向熟悉度/嚴重度）
│   ├── exp5_cross_condition.py  #   實驗五：跨工況泛化（9×9 矩陣/策略/老化漂移）
│   ├── exp6_osr_benchmark.py    #   實驗六：OSR 方法基準比較（七種偵測器）
│   ├── exp6_formal_benchmark.py #   實驗六正式版：Mahalanobis vs k-NN（含 provenance）
│   ├── exp6_matrix.py           #   實驗六：正式版 9 工況 × 3 seed × 2 方法矩陣
│   ├── aggregate_exp6.py        #   實驗六：正式矩陣彙整
│   ├── compare_openset.py       #   實驗七：單工況相同 split 比較 Mahalanobis 與 k-NN
│   ├── health_index_*.py        #   實驗八：健康指數 benchmark／matrix
│   ├── health_monitor.py        #   實驗八：健康監測 CLI
│   └── health/                  #   實驗八的套件：健康指數、校準、嚴重度、趨勢/告警、軌跡、診斷（見 docs/experiments/exp8_health.md）
├── reports/                     # 實驗八結果 exp8_health_index_results/（health_index_matrix 寫入）
├── tests/                       # unittest（venv/bin/python -m unittest discover -s tests -t .）
├── web/                         # 即時展示與實驗頁（只做編排與視覺化，不含實驗邏輯）
│   ├── live.py                  #   LiveDemo：把三個實驗模組串成互動串流
│   ├── server.py                #   Tornado + WebSocket 伺服器
│   └── static/index.html        #   單檔儀表板（原生 JS，無外部依賴）
├── docs/                        # 論文版技術文件快照 + Lineage 時期研究文件（見 docs/README.md）
├── data/                        # 特徵資料（由論文版管線產出，本專案唯讀；git 忽略）
│   └── Step-*/myfeature/{Motor}/{RPM}/{Screws}/*_Group_feature_data_clean.csv
├── logs/                        # 每次執行的 loguru 日誌（納入版控）
├── output/                      # 每次執行的結果檔與圖表（納入版控）
├── scripts/quantity/            # run_pytest.sh、run_ruff.sh（提交前檢查）
├── scripts/opt/                 # run_web.sh、build_uv.sh（展示與環境）
└── pyproject.toml               # 正式 Python 3.10.x；版本訂死，完整依賴樹鎖在 uv.lock
```

---

## 環境與執行

```bash
./scripts/opt/build_uv.sh --python python3.10 --venv venv  # 新目錄才可建立，既有 venv 不刪除
./scripts/opt/build_uv.sh --legacy --venv venv-legacy     # legacy extras 未完整鎖版，非正式驗證環境
```

正式範圍、更新流程與未完成的 health CLI 覆蓋見 [執行環境規則](docs/runtime_policy.md)。3.14 歷史研究不代表本輪 main 安裝相容。

所有指令都在**專案根目錄**執行，直接用 `venv/bin/python`，不需 activate：

```bash
venv/bin/python -m experiments.exp1_cold_start --motor T1 --rpm 8000rpm
venv/bin/python -m experiments.exp2_scale_growth
venv/bin/python -m experiments.exp3_trend --trials 40
./scripts/opt/run_web.sh
```

### 日誌與輸出慣例（沿用論文版專案習慣，改以 loguru 實作）

每次執行自動建立、以程式名與時間戳隔離，皆納入版控：

```
logs/{program}/{YYYY-MM-DD-HH-MM-SS}.log      # 完整執行日誌
output/{program}/{YYYY-MM-DD-HH-MM-SS}/       # results.csv / summary.json / *.png
```

呼叫 `setup_run()` 的入口另存 `environment.json`，記錄 Python、OS、套件、Git HEAD 與 constraints SHA；不記錄 token 或使用者絕對路徑。尚未統一的三支 health CLI 與未初始化的直接 API 呼叫不包含在這項保證。

---

## 資料集

沿用學長論文的實驗資料：以**拆除馬達固定螺絲**模擬不同程度的機械鬆動故障，
感測三軸振動、電流與溫差，每筆樣本為 1 秒訊號（10 kHz）萃取的 **105 維特徵向量**
（15 統計量 × 5 通道 + 三軸各 10 個轉頻諧波能量；由論文版 Step 1–2 管線產出）。

| 螺絲配置 | 本專題角色 |
|---|---|
| 8screws | **健康基準**（唯一的初始已知）|
| 7 → 1 screws | 未知故障（鬆動由輕到重，天然的嚴重度階梯）|
| 3_14 / 4_146 screws | 未知故障（複合不均勻鬆動，「不同方向」的壞法）|

本專案只讀 `*_Group_feature_data_clean.csv`（逐列 IQR scale=1.5 過濾版，
與論文 Step 4/5 相同輸入）。已知類別以 60/20/20 切為訓練／校準／保留集；
校準集只用已知資料決定閾值，未知資料絕不參與擬合。

---

## 方法與引用文獻

| 模組 / 環節 | 方法 | 主要文獻 |
|---|---|---|
| PHM 總體框架 | 條件監測 → 診斷 → 預後的分層流程 | ISO 13374；Jardine, Lin & Banjevic (2006) *MSSP*；Lee et al. (2014) *MSSP*；Lei et al. (2018) *MSSP* |
| 階段 0 健康基準 | 健康分布 + 馬氏距離的單類監測 | Taguchi & Jugulum (2002) *The Mahalanobis–Taguchi Strategy*；綜述：Pimentel et al. (2014) *Signal Processing*；對照法：Schölkopf et al. (2001) OC-SVM、Tax & Duin (2004) SVDD |
| 開集判定 | 逐類高斯 + 馬氏距離、校準分位數閾值 | 學長論文（本資料集與基礎流程）；K. Lee et al. (2018) *NeurIPS*（深度特徵逐類馬氏）；開集理論：Scheirer et al. (2013) *TPAMI*、Bendale & Boult (2016) *CVPR* OpenMax |
| 共變異數估計 | Ledoit–Wolf 收縮（高維小樣本可逆、良態）| Ledoit & Wolf (2004) *J. Multivariate Analysis*；替代：Chen et al. (2010) OAS |
| 非參數開集對照 | 逐類 k-NN 平均歐氏距離 + known-only 校準分位數 | [Cover & Hart (1967)](https://doi.org/10.1109/TIT.1967.1053964) *IEEE Transactions on Information Theory*；[Ramaswamy, Rastogi & Shim (2000)](https://doi.org/10.1145/342009.335437) *SIGMOD* |
| 新故障分群 | HDBSCAN（自動叢集數、雜訊點標記）| Campello, Moulavi & Sander (2013) |
| 變化點偵測 | EWMA超線旗標比例；CUSUM只供展示，判別門檻由專案設定 | Roberts (1959) *Technometrics*；Page (1954) *Biometrika*；[原文與實作對照](reports/Andy_20261008_議題交付/citation_followup.md) |
| 持續學習模擬 | 實驗二只用候選群樣本自學；首頁展示的操作員確認仍用完整配置池重擬合（oracle）。兩者都未驗證消除遺忘 | EWC／iCaRL僅為背景對照，未實作：[引用核對](reports/Andy_20261008_議題交付/citation_followup.md) |
| 衰退外插（未實作）| 需要同個體歷程與失效終點，現有資料不支援RUL | 明確背景候選：Gebraeel et al. (2005), *Residual-life distributions from component degradation signals: A Bayesian approach*；Wang et al. (2008), *A Similarity-Based Prognostics Approach for Remaining Useful Life Estimation of Engineered Systems*；[DOI、全文狀態與舊引用限制](reports/Andy_20261008_議題交付/citation_followup.md) |

---

## 與 legacy 的關係

論文版程式碼（Step 1–6 管線、128 個 notebook、scripts、執行紀錄與論文全文）
已整批移回 [原始論文](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor)
repo 保存——issue 討論串（#1–#60，缺陷成因與修正驗證）也在該處；
本 repo 的 git 歷史仍完整保留搬移前的所有版本，可隨時回溯。
技術文件快照（Step 1–6 說明 + 三份 Lineage 時期研究文件）保留於本 repo 的
[docs/](docs/README.md)，每份都有歷史標記。

- 本專案**繼承**：資料集與 105 維特徵工程（直接讀論文版管線產出的 clean CSV）、
  逐類 Ledoit–Wolf 馬氏開集偵測器（[docs/Mahalanobis_Improvement.md](docs/Mahalanobis_Improvement.md)
  的採用結論）、HDBSCAN 參數、logs/output 慣例。
- 本專案**翻轉**：起點從「已知 5 類」改為「只知健康」；HDBSCAN 從「判定器」
  改為「新方向發現器」；判定粒度從叢集改為逐樣本。
- **規範**：新程式碼一律不依賴論文版程式碼——需要的零件以「複製 + 來源註記」帶入
  （見 `src/core/mahalanobis.py` 檔頭）。

---

## 限制與未來工作

給老師的實驗七問、負面結果與操作流程另見 [exp13 導覽入口](docs/navigation/exp13_README.md)。獨立導覽用 `python -m web.guide --data-root data --port 8601 --seed 42` 啟動；原實驗頁與即時展示仍由 `python -m web.server` 提供。模式切換不改模型；確認後是既有完整配置池重擬合。歷史跨馬達弱結果、來源 UNKNOWN 與 fresh INCOMPLETE 不因整合而消失。

1. **螺絲數是離散的故障代理**：真實磨損是連續過程，量尺結論外推需保留。
2. **「前期資料保證健康」的實務前提**：需要 burn-in 期；不同轉速的健康基準需分開建
   （本專案依 motor×rpm 分資料集正是為此）。
3. **健康誤報率受校準集大小影響**（實測 10.9% vs 名目 5%）：可用更多健康資料或
   conformal 校準改善。
4. **發散故障需要較長累積**：嚴重鬆動與複合配置在特徵空間發散，即使有自適應密度
   階梯，發現延遲（225~265 筆）仍明顯高於輕度鬆動（65~105 筆）——更快的新方向
   發現（如線上式分群）是可改進點。
5. **衰退曲線外插未實作**：T1/T2/T3 是不同馬達個體，個體與老化影響混雜；
   缺同個體連續歷程及失效終點，不能當作 run-to-failure 模板或估計RUL。
6. **單一測試台、單一故障機制**：結論外推到其他 PHM 任務需再驗證。
