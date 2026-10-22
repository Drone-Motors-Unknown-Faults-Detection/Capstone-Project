# 實驗十一：原始論文配置協定的Accuracy與F1對照

## 資料角色用量表

流程：固定A/B類別角色 → 每known類切分 → 建全域legacy或逐類LW → known holdout與unknown全池分類 → 分別報accuracy、balanced accuracy與macro-F1。它與老師healthy＋五fault known／四unknown不同；A是healthy＋四fault known／五unknown。

| 資料角色 | T1／8000rpm、A協定 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| known train | 8／1／2／3／4共1478列 | 884 | 各類floor(.6n)再合計；884/1478 | scaler及分布 | 與本類cal/test互斥 |
| known cal | 同上 | 293 | 各類floor(.2n)再合計 | LW各類q95；legacy例外 | legacy門檻不讀此組 |
| known test | 同上 | 301 | 1478−884−293 | 配置分類，含被拒錯誤 | 列級互斥 |
| unknown test | 5／6／7／3_14／4_146 | 1571 | 1571/1571故障子池 | 合併unknown分類 | 未fit／cal |
| B協定 | 只known健康 | 187／62／64，unknown2736 | 健康分母313；unknown全池 | 冷啟動分類API對照 | 不與A當新獨立資料相加 |

依[用量產物](README.md#資料用量與分母怎麼讀)，三seed只是同資料不同索引。A評估分母為301+1571=1872，不是全部3049；已用known train/cal不再進分類測試。沒有validation選參或新final test。

## 發表與理解檢查

口頭主軸：「先區分看過的配置與保留未知，再比較輸出分類。必須注意legacy把label固定為-1，現行classify無法返回known名稱；所以大分差混合了分類介面、逐類結構與校準差異，不能報成單改LW的純提升。」結果及5pp內部門檻見下方，未自動產生可靠性PASS。

- 全判unknown能否有高accuracy？未知佔多數時可以，所以還要看各known召回與balanced accuracy。
- 是否完整重跑原始論文？不是，只重建指定類別角色，未重建原notebook資料協定。
- 5pp達標是否已可部署？仍受不公平基線、列級切分與缺獨立資料限制。
- 組員請說出A有幾個known faulty、unknown分母及legacy介面問題，勿與老師的另一協定混用。

本實驗在Lineage現行monitor中比較legacy全域分布與逐類LW，分開「認得部分配置」及「只認健康」兩種情境。它重建指定類別角色，不是完整重跑原始論文 notebook、原始資料切分或公平隔離單一共變異數變更的實驗。

## 資料、類別與切分

105維clean特徵來源／每列／未知單位／未去重見[實驗一](exp1_cold_start.md#資料與載入)。逐工況載入，沒有強制九工況；實際n_datasets與類別coverage須查資料。T1／T2／T3是不同個體，但本實驗**每工況內列隨機切分**，不作跨馬達專用train／cal／test。

| 協定 | known | unknown評估 |
|---|---|---|
| A | healthy 8screws＋1／2／3／4螺絲配置 | 5／6／7／3_14／4_146與現有池的交集 |
| B | healthy 8screws | 所有其他現有配置 |

1screw／1screws依候選順序只取一種，缺類會被交集略過，程式沒有保證一定五known／五unknown；A不在候選清單的額外配置也不會自動評估。這與老師healthy＋五known faulty／四unknown的另一協定不同，不能互稱已完成。

每seed42／123／2026、每method legacy／ledoit_wolf分別從健康fit_initial，再依known順序add_class；各known列洗牌60/20/20。scaler只fit全部known train，LW逐類中心／共變異數只fit train、各cal定0.95線。unknown不fit／校準／選參；已知各holdout與unknown全池只評估。沒有神經網路epoch或共同winner選擇。

**legacy的比較例外很大**：忽略類別標籤、只建全域分布，用train距離定線，distribution label=-1。monitor.classify把-1當拒絕，已知名稱無法返回，即使score≤1也可能輸出unknown。因此大幅多類F1差距同時包含分類介面／逐類結構／校準來源差異，不能稱「只把協方差改LW就提升這麼多」，也不是可正常分類的原始論文基線證明。

## 如何執行

repo根目錄依[環境政策](../runtime_policy.md)準備uv與資料：

```bash
uv run --locked python -m experiments.exp11_ancestor_comparison --help
uv run --locked python -m experiments.exp11_ancestor_comparison --data-root data --seeds 42,123,2026 --methods legacy,ledoit_wolf --confidence 0.95
```

seeds／methods是逗號字串。API run(data_root,seeds=(42,123,2026),confidence=0.95,methods=("legacy","ledoit_wolf"))回傳dict；build_ancestor_monitor(pools,seed,confidence,method)供[實驗十二](exp12_confusion_tsne.md)共用。Web「實驗十一」只用畫面上的一個 seed，兩種方法與協定 A／B 都跑；沒有逐筆串流、續跑或取消，每次重建模型。CLI 預設三個 seed。非法method、缺healthy／未知、空holdout可能失敗，先核對來源與coverage，不補造類別。九工況齊全且三 seed 時 9×3×2×2＝108列，不是108次獨立採集。

## 指標與輸出

main經setup_run寫logs/exp11_ancestor_comparison/{ts}.log、output/exp11_ancestor_comparison/{ts}/environment.json、summary.json與ancestor_comparison.png，不存逐樣本預測或模型權重。

已知holdout真值保留配置名；所有unknown真值合併成unknown；classify None也合併unknown。accuracy是總正確比例；balanced_accuracy是各真值類別recall平均；macro_f1是各類F1平均，零分母設0；unknown_f1只把unknown當正類。fault比例高時「全判unknown」仍可能有高accuracy／unknown F1，須看known召回，不能只報好看的數字。

rows先round4，summary按protocol×method等工況-seed平均；Wilcoxon只對dataset×seed的**macro-F1**配對，不是對balanced_accuracy檢定，輸出mean_diff_pp、W與p（round6，0不表示機率真為0）。同motor多RPM／seed共用來源，27pairs不滿足27個獨立馬達，p值只能作此依賴資料下描述，不能宣稱motor母體顯著保證。

原約定主指標balanced_accuracy與macro-F1，LW−legacy絕對提升≥5個百分點（pp）；threshold_pp只是輸出設定，程式未自動判PASS。滿足這個內部門檻仍受legacy介面問題與列切分限制。完整分數見 [output/exp11_ancestor_comparison/2026-10-23-01-35-07](../../output/exp11_ancestor_comparison/2026-10-23-01-35-07/summary.json)（Aero、commit `46ed4d5`、seed 42／123／2026；與 2026-10-09 的舊輸出扣掉環境欄位後相同）。不能取代[#9](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/9)跨馬達研究、fresh final或可靠fault-type驗證。

## 來源與程式碼

馬氏／LW與原始論文沿用來源見[實驗一](exp1_cold_start.md#方法來源與程式定位)。Wilcoxon（1945），*Individual Comparisons by Ranking Methods*，Biometrics Bulletin1,80–83，[DOI](https://doi.org/10.2307/3001968)。類別角色、5pp與兩協定是Lineage操作約定，不由文獻保證成效。

[exp11](../../src/experiments/exp11_ancestor_comparison.py)建monitor／評估／writer；[monitor](../../src/core/monitor.py)分割與classify；[mahalanobis](../../src/core/mahalanobis.py)兩分布實作；[test_exp11_ancestor_comparison](../../src/tests/test_exp11_ancestor_comparison.py)核對known集合與unknown不add_class。測試不證明原始論文完整重現或錄製獨立。
