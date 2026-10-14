# 實驗六：健康-only異常偵測器基準

## 資料角色用量表

流程：逐工況讀資料 → 同一健康切分交八方法 → 各自fit／cal → 同一test打分 → 比較排序與誤報。距離、邊界、隔離樹、密度及重建各抓不同偏離方式；單看模型名稱不能推定哪個會贏。

| 資料角色 | 本機六工況 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 健康train | 各工況健康池合計1877 | 1124 | 逐池floor(.6H)後加總 | scaler及模型 | 同工況八方法重用 |
| 健康cal | 同上 | 373 | 逐池floor(.2H)後加總 | q95；legacy例外 | 不與holdout重疊 |
| 健康test | 同上 | 380 | 各池剩餘；380/1877 | 健康誤報 | 非八倍獨立樣本 |
| 未知test | 其餘配置全池 | 17176 | 17176/17176 | 拒絕與AUROC | 多方法重用，不fit |

依[用量總表](README.md#資料用量與分母怎麼讀)，沒有新訓練。跨工況共用連續RNG，這張表只共用數量，不能把逐模型重設seed的cold_start索引摘要套成exp6索引。六工況×八方法為48列／seed，不是九工況完成。AE內部分驗證另見[實驗九](exp9_autoencoder.md)。

## 發表與理解檢查

口頭主軸：「用同一份健康訓練、校準與測試，比較八種偏離分數。先看AUROC排序，再看固定線的健康誤報與未知召回。legacy用train定線，與其他法並非完全同校準契約。」結果來源見下方，沒有共同部署PASS。

- 為什麼不只挑AUROC最高？排序好不代表固定門檻誤報低；方法還有不同校準限制。
- AE是在學故障嗎？沒有，它只學重建健康特徵。
- 此k-NN等於主線factory嗎？不是同一實作，固定5鄰居且無有效k縮減；正式兩法請看6-2或實驗七。
- 組員請用「拿什麼比較」解釋距離／重建／密度，不必背完整參數表，但要知道legacy例外。

健康基準該用橢球距離、近鄰、樹、邊界，還是重建誤差？本實驗把同一工況的健康切分交給各方法，比較誤報與未知配置偵測。它不是healthy＋五種known配置的多類分類研究；正式factory兩方法比較見[正式版](exp6_formal_benchmark.md)。

## 資料與fit順序

data-root依[loader](../../core/data.py)掃描單一motor／RPM的105維clean特徵池，沒有強制九工況。每列／單位／移除非有限列／未去重及來源限制見[實驗一](exp1_cold_start.md#資料與載入)。數量以rows與實際來源為準。三顆馬達是不同個體，列隨機切分不能證明錄製獨立。

run以一個default_rng(seed=42)連續處理所有工況，健康池每次列洗牌60/20/20；改變掃描集合可能改變後續切分。同工況所有方法共用同一train／cal／holdout。每個方法獨立用train fit RobustScaler和參考模型，cal只定0.95分位數門檻，healthy holdout與所有fault配置全池只評估，unknown不選參或校準。legacy是例外：用train本身距離定線。

[core/detectors.py](../../core/detectors.py)的ALL_DETECTORS目前有**八個**方法，早期「七種」是尚未加入AE的名稱：

| 方法ID | train學什麼／原分數 | 影響重現的設定 |
|---|---|---|
| maha_ledoit_wolf | 健康中心與LW共變異數／馬氏距離 | cal分位數 |
| maha_legacy | 經驗共變異數＋pinv／馬氏距離 | train分位數，含既有ridge |
| ocsvm | 健康邊界／負decision_function | nu=0.05、gamma=scale |
| iforest | 隨機隔離樹／負score_samples | 200樹、random_state=seed |
| lof | 健康局部密度／負decision_function | 20鄰居、novelty=True |
| knn_dist | train近鄰庫／5鄰居平均歐氏距離 | 固定5，不是正式factory的有效k縮減 |
| pca_recon | 保留95%變異的PCA／重建L2誤差 | full SVD、train-only |
| mlp_autoencoder | 105→64→16→64→105重建／L2誤差 | 詳見[實驗九](exp9_autoencoder.md) |

一般_Base把raw score除cal分位數；門檻≤0時先依cal最小值平移，避免負分母。LW／legacy走自己的馬氏score實作。統一score>1判未知，不表示所有方法門檻來源完全一致。kNN train少於5或LOF資料太少可能失敗；沒有靜默補樣本策略。

## 怎麼執行與使用

從repo根目錄依[uv環境政策](../runtime_policy.md)準備資料與鎖版環境：

```bash
uv run --locked python -m experiments.exp6_osr_benchmark --help
uv run --locked python -m experiments.exp6_osr_benchmark --data-root data --seed 42 --confidence 0.95
```

API run(data_root="data",seed=42,confidence=0.95)回傳dict。CLI沒有選單一detector或motor的參數；每次跑全部方法，含最多2000迭代的AE，可能較慢。缺healthy／fault、工況空或健康切分太小時應核對資料，不把缺格平均成完整九工況。

Web依[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)啟動，選「實驗六：多方法」、seed，按「▶ 執行」。跑整個伺服器data-root，沒有邊跑邊畫、續跑或中途取消；不要用切頁當取消。每次重新fit，不保存可恢復的模型。

## 輸出與指標

main經setup_run寫logs/exp6_osr_benchmark/{ts}.log、output/exp6_osr_benchmark/{ts}/environment.json、results.csv、summary.json及osr_benchmark.png。Web另存web_server的experiments子目錄。results每列是一工況×方法，非獨立馬達數。

auroc衡量unknown分數是否較高；healthy_fp為健康score>1比例；fault_detect為所有fault合併後score>1比例，不是各配置等權macro。fpr_at_tpr95用**評估fault分數第5百分位**當診斷線，健康嚴格大於該線的比例；這不是部署用門檻，也不回寫模型。summary先對各列round4，再跨工況等權平均；auroc_std／healthy_fp_std用ddof=0，排序依AUROC再FPR。不是信賴區間或自動部署winner。

圖的AUROC座標從0.9開始，低分可能不顯眼，必須看JSON。預期未知排序高、健康誤報低；沒有預先固定可靠性成功門檻。完整歷史數字見[固定紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/experiments/exp6_osr_benchmark.md)。同工況健康-only高分不能推論跨馬達known fault-type成功。

## 方法來源與程式碼

馬氏距離／LW來源見[實驗一](exp1_cold_start.md#方法來源與程式定位)。其餘保留方法出處：

- Schölkopf等（2001），*Estimating the Support of a High-Dimensional Distribution*，Neural Computation13,1443–1471，[DOI](https://doi.org/10.1162/089976601750264965)。
- Liu、Ting、Zhou（2008），*Isolation Forest*，ICDM,413–422，[DOI](https://doi.org/10.1109/ICDM.2008.17)。
- Breunig等（2000），*LOF: Identifying Density-Based Local Outliers*，SIGMOD,93–104，[DOI](https://doi.org/10.1145/335191.335388)。
- Ramaswamy、Rastogi、Shim（2000），*Efficient Algorithms for Mining Outliers from Large Data Sets*，SIGMOD,427–438，[DOI](https://doi.org/10.1145/342009.335437)。本repo用平均k距離，不能稱與論文的第k距離定義完全相同。
- Jolliffe（2002），*Principal Component Analysis*，Springer，[DOI](https://doi.org/10.1007/b98835)。重建分數／cal正規化是專案組合。

[runner](../../experiments/exp6_osr_benchmark.py)負責公平切分、指標、writer；[detectors](../../core/detectors.py)實作模型；[web/experiments](../../web/experiments.py)只編排；[test_exp9_autoencoder](../../tests/test_exp9_autoencoder.py)、[test_web_experiments](../../tests/test_web_experiments.py)檢查元件與入口。正式factory的k-NN與此knn_dist不是同一類別實作，不直接混用結果。
