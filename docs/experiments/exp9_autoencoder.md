# 實驗九：用非線性重建誤差偵測未知

## 資料角色用量表

流程：健康外層切分 → train縮放 → MLP壓縮再重建健康 → 內部validation提早停止 → 外層cal定誤差線 → 未知與健康test比較。輸入輸出都是105維，瓶頸16維不是把正式CSV改成16欄。

| 資料角色 | T1／8000rpm | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 外層train | 健康池 | 187 | 187/313 | fit scaler，供MLP內切 | scaler見過全部內部validation |
| 內部validation | 上述187列的一部分 | 約10%，實際IDs未保存 | 分母187，不是全部3049列 | R²選提早停止權重 | 與外層cal不同；前處理非完全隔離 |
| 外層cal／健康test | 同健康池 | 62／64 | 各除以313 | q95誤差／健康誤報 | 不進外層scaler或MLP |
| unknown test | 九fault配置 | 2736 | 故障全池100% | 重建異常排序與拒絕 | 不選epoch或瓶頸 |

依[唯讀用量](README.md#資料用量與分母怎麼讀)，沒有本輪AE訓練。`e=||z−fθ(z)||₂`是推論重建L2誤差；訓練的half squared error及正則化不同，不能混稱同一loss。θ只由健康train學，cal不反向傳播。

## 發表與理解檢查

口頭主軸：「把健康特徵壓縮後重建，陌生資料如果重建不回來就有高誤差。是否比PCA好要用相同外層資料比較，不能因為有神經網路就說更準。最大2000輪只是上限，實際停止輪數未保存。」

- 為什麼可能失效？未知也可能被網路重建得好；健康跨域差異也可能重建得差。
- 內部validation完全獨立嗎？不是，scaler先看整個外層train；外層test仍未參與。
- 可從summary知道最佳epoch嗎？不能，程式未保存訓練曲線／權重。
- 組員請說明輸入、瓶頸、重建誤差、外層門檻四件事；架構及官方預設列於下方，不必死背。

AutoEncoder先學習把健康特徵壓縮，再重建原輸入；新資料重建得差時給高異常分數。本實驗檢查非線性重建是否比PCA重建有用，不以神經網路名稱推定會更準。它是[實驗六八方法](exp6_osr_benchmark.md)中的mlp_autoencoder，不另建資料協定或runner。

## 資料與三種資料角色

每工況105維clean特徵；來源、每列／單位、非有限列移除與未去重限制見[實驗一](exp1_cold_start.md#資料與載入)。default_rng(seed=42)連續對各工況健康池列洗牌60/20/20，八方法共用切分。

| 用途 | 實際學習／評估 |
|---|---|
| 外層healthy train | fit RobustScaler；縮放後交MLP，另由MLP保留10%作early-stopping validation |
| healthy cal | 只算重建誤差0.95分位數門檻，不選網路 |
| healthy holdout | 評估健康誤報 |
| 其他配置全池 | 只評估unknown，不fit、校準或選參 |

scaler學整份外層train，**包含MLP內部validation**，因此內部validation不是完全隔離的前處理選擇驗證；外層cal／holdout未進scaler。T1／T2／T3為不同個體，但這裡各自同工況列切分，沒有raw錄製獨立保證。

## 網路如何訓練

[MLPAutoencoderDet](../../core/detectors.py)呼叫sklearn MLPRegressor，把X當輸入也當目標，105→64→16→64→105，隱藏層ReLU、輸出線性。程式指定random_state=seed、max_iter=2000、early_stopping=True；其他沿uv.lock固定sklearn1.7.2的[官方預設](https://scikit-learn.org/1.7/modules/generated/sklearn.neural_network.MLPRegressor.html)：

loss=squared_error（實作half squared error），含alpha=0.0001的L2；optimizer=Adam，learning_rate_init=0.001；batch_size=auto（min(200,n_samples)），每epoch洗牌。內部validation_fraction=0.1，以重建R²改善選最佳權重；tol=1e-4、n_iter_no_change=10，不改善則提早停，最多2000epoch，不是保證跑滿。沒有用fault挑epoch／瓶頸。

raw_score為縮放後X與重建X的L2差，再除外層cal誤差分位數；score>1判未知。它不是逐類fault分類器，也不替換PolarMap幾何。程式未保存loss curve、最佳epoch或模型checkpoint，summary不能提供不存在的訓練軌跡。

## 如何執行與使用

repo根目錄準備[uv環境](../runtime_policy.md)及資料，無需TensorFlow或GPU：

```bash
uv run --locked python -m experiments.exp6_osr_benchmark --help
uv run --locked python -m experiments.exp6_osr_benchmark --data-root data --seed 42 --confidence 0.95
```

CLI會跑全部八方法，不只AE；沒有experiments.exp9_autoencoder模組、獨立--epochs或--detector參數。API可用MLPAutoencoderDet(confidence=0.95,seed=42).fit(X_train,X_cal).score(X_test)，請自行維持資料用途；完整比較用exp6.run(data_root,seed,confidence)。

Web選實驗六多方法、seed後按「▶ 執行」可看AE列；沒有獨立實驗九執行卡、續訓、串流或中途停止。依[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)啟動伺服器。健康樣本少可能使early-stopping R²不可估或ConvergenceWarning；警告不是已收斂證明，不增加test資料幫它訓練。缺fault／工況／105欄先查來源。

## 輸出、預期與限制

writer沿exp6 main，logs/exp6_osr_benchmark/{ts}.log與output/exp6_osr_benchmark/{ts}/environment.json、results.csv、summary.json、osr_benchmark.png，方法ID為mlp_autoencoder；沒有另寫exp9模型。Web另保存web_server的experiments JSON。

AUROC、健康誤報、fault_detect、事後FPR@TPR95與等工況mean／ddof0的定義見[實驗六](exp6_osr_benchmark.md#輸出與指標)。要支持AE優勢，需相同輸入／split／cal政策下排序與誤報的配對證據；某seed略優不是泛化成功。誤報高可能來自重建、校準或分布差異，不能單憑誤報確診過擬合。程式無通用PASS門檻。

完整歷史分數查[固定實測包](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/1fa9431bb7b86959f29540d07b2b9290ab39ce42/output/exp6_osr_benchmark/2026-10-09-14-07-53)。健康-only同工況分離不能回答known fault-type、跨馬達部署、fresh final、物理嚴重度或RUL。

## 來源與程式碼

Sakurada與Yairi（2014），*Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction*，MLSDA,4–11，[DOI](https://doi.org/10.1145/2689746.2689747)提供重建異常偵測方法背景。網路大小、ReLU與cal分位數是Lineage設定，非逐項重現該論文。

[core/detectors](../../core/detectors.py)建網路與分數；[exp6 runner](../../experiments/exp6_osr_benchmark.py)做比較／writer；[test_exp9_autoencoder](../../tests/test_exp9_autoencoder.py)檢查成員、重現與校準邊界；[web/experiments](../../web/experiments.py)編排。未另新增Ancestor神經網路依賴。
