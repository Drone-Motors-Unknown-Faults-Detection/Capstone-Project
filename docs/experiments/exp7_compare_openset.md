# 實驗七：單工況Mahalanobis與k-NN對照

## 資料角色用量表

流程：同一單工況與seed → 兩個monitor各自從相同健康split建立參考 → 各自cal定線 → 同test配對。LW與k-NN原理見[共通原理](README.md#共通原理先問模型拿什麼比較)，不引入新的分類器。

| 資料角色 | T1／8000rpm | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| train／cal | 313健康列 | 187／62 | 各除以313 | 各方法參考與門檻 | 同方法用途互斥；方法間刻意相同 |
| 健康test | 同上 | 64 | 64/313 | 健康接受／誤報 | 不參與fit／cal |
| unknown test | 九fault配置 | 2736 | 故障全池100% | 未知拒絕與排序 | 兩方法共用，不算5472獨立樣本 |

依[唯讀用量](README.md#資料用量與分母怎麼讀)，非新比較分數。沒有validation選winner；相同seed重跑不是獨立採集。跨馬達專用train／cal／test是另一研究協定。

## 發表與理解檢查

口頭主軸：「固定同一資料與切分，只換偏離量法；先比較健康誤報，再比較未知召回、AUROC與耗時。差值只能歸因於這個條件下的方法，不代表新馬達或所有故障都改善。」

- 已知accuracy相同是否兩detector重複？這裡known只有healthy，且同資料；要看score與拒絕結果，不把它混成fault-type分類。
- 為什麼要同seed？控制列切分；三seed只是切分敏感度，不是三次獨立錄製。
- k大一點一定好嗎？本輪不調k，固定5並保存有效k；密度與尺度不同可能失效。
- 組員請用一句話說清兩法參考資料相同、分數定義不同，再讀下方實際輸出。

想先在一個馬達／RPM看兩種Open Set方法的差異，用這個入口即可；九工況與多seed比較看[實驗六正式版](exp6_formal_benchmark.md)。這裡只認識healthy，不訓練九種故障名稱分類器。

## 資料與建立順序

資料是單工況的105維clean特徵，每列、單位、清理與未去重限制見[實驗一](exp1_cold_start.md#資料與載入)。同來源／seed下，每方法各建OpenSetMonitor，健康列洗牌60/20/20一致；seed預設42。RobustScaler及參考模型只fit train；healthy cal定confidence0.95門檻；healthy holdout與其他配置全池只評估。unknown不選參、不fit、不校準。不同motor是不同個體，這不是跨馬達群組切分。

mahalanobis用LW中心／共變異數；knn保存train庫、平均歐氏距離，k=5且資料少時縮減；score>1判未知，等於1接受。--method可選legacy／ledoit_wolf／oas／mcd，只影響馬氏方法。legacy用train距離的confidence分位數定線（不讀cal），合併分布的內部標籤是-2（`LEGACY_POOLED_LABEL`）；分數≤1的列取縮放空間最近的已知類train中心，所以classify對單一已知類回healthy，>1才是-1未知（`core.mahalanobis._labels_from`，#266）。這時known_class_accuracy就是healthy holdout分數≤1的比例；門檻量在train上，holdout常超線，數字偏低不是介面限制。該列的calibration_source記「known-only 60% train split (legacy: threshold on training distances)」，fairness欄的known-only calibration宣告不適用此例外。正式矩陣拒絕legacy。顯示PCA train-only，MCD另有train-only降維；沒有神經網路epoch。

## 怎麼執行與使用

在repo根目錄準備[uv環境](../runtime_policy.md)與資料：

```bash
uv run --locked python -m experiments.compare_openset --help
uv run --locked python -m experiments.compare_openset --data-root data --motor T1 --rpm 8000rpm --seed 42 --openset-methods mahalanobis knn --method ledoit_wolf --knn-neighbors 5 --confidence 0.95
```

--openset-methods可只留一法；不是單數的--openset-method。API run(pools,methods=("mahalanobis","knn"),seed=42,confidence=0.95,mahalanobis_method="ledoit_wolf",knn_neighbors=5)回傳results／details／fairness，不保存模型或檔案。

Web依[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)啟動，選「實驗七」、資料集／seed按「▶ 執行」，一次對照兩法。沒有串流、續跑或取消；重跑重新fit。找不到工況／healthy、unknown空或資料太少，先查來源，不用其他工況補空集合。

## 輸出與指標

main經setup_run寫logs/openset_comparison/{ts}.log、output/openset_comparison/{ts}/environment.json、summary.csv、details.csv、summary.json與method_comparison.png。Web另寫web_server的experiments JSON。

results的auroc／aupr_unknown_positive是unknown正類排序／平均精確率；AUPR與open_set_accuracy受fault比例影響。unknown_recall／precision／f1_unknown按score>1算，零分母為0；known_class_accuracy按classify是否回healthy算。details的healthy列unknown_recall其實是健康誤報率，fault列才是未知召回率，並列n與無因次median_score。

fpr_at_95_tpr取ROC上TPR≥0.95的最小FPR，是事後評估點，不修改門檻；實驗六廣度、正式版與本入口都呼叫`core.openset.fpr_at_tpr`（#235），定義相同。class_thresholds是原距離線，正規化線為1。

預期兩法產生有限分數，unknown高於healthy且誤報低；沒有通用成功數字。不按這張圖自動換預設；同工況好分數不代表跨馬達、fresh final或真實損壞辨識。九工況三seed的比較見[實驗六矩陣](exp6_matrix.md#輸出怎麼看)，不是本單工況實測。

## 實測結果

2026-10-23在Aero（aero01）以commit `46ed4d5`、`--data-root data`（資料指紋`1b55b847…`）跑上方第二行指令，輸出[output/openset_comparison/2026-10-23-01-34-12](../../output/openset_comparison/2026-10-23-01-34-12/summary.json)。T1／8000rpm、seed42，健康test 64列、unknown test 2736列：

| 方法 | AUROC | AUPR | FPR@95TPR | open_set_accuracy | known_class_accuracy | 健康holdout誤報 | unknown_recall | unknown_precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mahalanobis（LW） | 1.0 | 1.0 | 0.0 | 99.75% | 89.06% | 7／64（10.94%） | 100% | 99.745% |
| k-NN（k=5） | 1.0 | 1.0 | 0.0 | 99.96% | 98.44% | 1／64（1.56%） | 100% | 99.963% |

健康誤報取自`details`的8screws列（`unknown_recall`在健康列就是誤報率）。兩法排序都完全分開；差別只在固定線1上的健康誤報。這是單一工況、單一seed、64個健康窗，k-NN少6個誤報不足以換預設；九工況×三seed下兩法健康接受率是94.66%對94.67%。Mahalanobis的7／64和[實驗一](exp1_cold_start.md)同工況同seed的健康誤報一致。

## 方法來源與程式碼

LW／馬氏／近鄰見[實驗一](exp1_cold_start.md#方法來源與程式定位)。Chen等（2010），*Shrinkage Algorithms for MMSE Covariance Estimation*，IEEE TSP58,5016–5029，[DOI](https://doi.org/10.1109/TSP.2010.2053029)對應OAS；Rousseeuw與Van Driessen（1999），*A Fast Algorithm for the Minimum Covariance Determinant Estimator*，Technometrics41,212–223，[DOI](https://doi.org/10.1080/00401706.1999.10485670)對應MCD。切分與正規化為Lineage操作約定。

[compare_openset](../../src/experiments/compare_openset.py)評估與writer；[monitor](../../src/core/monitor.py)建基準；[openset](../../src/core/openset.py)共用factory；[web/experiments](../../src/web/experiments.py)編排；[test_openset](../../src/tests/test_openset.py)及[test_web_experiments](../../src/tests/test_web_experiments.py)驗證入口，不證明採集獨立。
