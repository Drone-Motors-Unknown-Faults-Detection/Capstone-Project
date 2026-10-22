# 實驗十：少量目標健康資料能不能幫助跨工況

## 資料角色用量表

流程：source健康基準 → target抽健康support → 平移或重新建基準 → 剩餘target健康與全部fault評估。shift只搬中心；沒有重新學跨域共變異數，也沒有深度遷移。

| 資料角色 | source T1／8000rpm → target T3／8000rpm，N=25 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| source train／cal | source健康313列 | 187／62 | 各除以313 | 原模型／原門檻 | 不用target fault |
| target support | target健康272列 | 25 | 25/272=9.19% | adapted算median shift | 與同N target eval互斥 |
| scratch內切 | 上述25列 | 15 train／5 cal／5不用 | 各除以25 | 從零fit／門檻 | 5列support holdout未評估 |
| adapted／scratch健康test | target剩餘健康 | 247 | 247/272=90.81% | 同N同pair比較 | 兩策略刻意共用 |
| direct健康test | target完整健康 | 272 | 272/272 | N=0直接搬用 | 包含上述support，不與另兩策略逐筆相同 |
| unknown test | target全部fault | 2953 | 故障池100% | 各策略拒絕／排序 | 各策略重用 |

數量依[唯讀用量](README.md#資料用量與分母怎麼讀)，未新fit；N=10/50等另存`transfer_support`。六工況有30個有向配對，不是九工況72對。無validation選N或shift；不同N的support不是巢狀子集。

## 發表與理解檢查

口頭主軸：「用少量已知健康的target資料估中心偏移，再比較保留source模型與從零建立。相同N的adapted和scratch可配對，direct用了完整健康池，不能把所有差值都歸因於遷移。」`shift=median(target support)−median(source train)`，對target減shift再送source模型；此公式是Lineage約定，不是完整CORAL。

- 為什麼要保留target健康不參與support？否則會用適應過的資料評自己。
- 平移能修好所有域差異嗎？不能，形狀／相關性變化沒有被對齊。
- 30個配對是否30個獨立馬達？只有本機兩個馬達、各三RPM，重複評估不是新個體。
- 組員請先指出source、support與eval，再談結果，並主動說明direct不完全配對。

來源工況的健康基準可能不適合另一馬達或RPM。本實驗比較直接搬用、以少量目標健康資料平移、只用這些資料重新建基準；它沒有更新正式預設，也沒有神經網路遷移訓練。

## 資料與配對順序

從data-root掃描單工況105維clean池，來源／單位／未去重與raw限制見[實驗一](exp1_cold_start.md#資料與載入)。工況數d以實際掃描為準，程式沒有強制九格；九格才有9×8＝72個有向src→dst配對。不同motor是不同個體，不是老化前後。

每src用healthy列洗牌60/20/20、seed42，RobustScaler／LW中心共變異數只fit src train，src cal定confidence0.95線。預設method=ledoit_wolf；legacy例外用train定線。沒有k-NN CLI選項。dst fault全池只評估，不用來選N或調shift。

| 策略 | 目標端用途 | 實際評估集合 |
|---|---|---|
| direct，N=0 | 不適應來源模型 | dst全部healthy與fault |
| adapted，N=10／25／50 | dst健康列洗牌取N筆support；shift=median(support)−median(src train) | 剩餘healthy及fault都減shift，仍用src模型／src門檻 |
| scratch，同N | support內再60/20/20；train fit scaler／LW、cal定線；support的holdout不用 | 與同N adapted相同的剩餘healthy、同fault全池 |

_draw_support的support／eval索引不交集。單一default_rng(seed)隨src、dst、N遞進；不同N的support**不是巢狀子集**。direct只跑N=0完整目標健康池，**沒有在每個N扣除support重新算direct**；不能宣稱它與adapted是逐樣本完全配對。adapted和scratch在同pair／N才共用eval。source fit只用健康，unknown不fit／cal／選參。

N=10的scratch為6train、2cal、2不用的holdout，估計不穩與門檻不確定須保留。相同dst在不同src重複出現，72pairs不能當72個獨立馬達或採集。

## 怎麼使用

在repo根目錄依[環境政策](../runtime_policy.md)準備uv與資料：

```bash
uv run --locked python -m experiments.exp10_transfer --help
uv run --locked python -m experiments.exp10_transfer --data-root data --seed 42 --confidence 0.95 --method ledoit_wolf --n-values 10,25,50
```

API run(data_root="data",seed=42,confidence=0.95,method="ledoit_wolf",n_values=(10,25,50))回傳dict。CLI的N為逗號分隔，沒有專用train motor。Web在「實驗十」選 seed 按「▶ 執行」，掃描整個 data-root，N 仍是 10／25／50；不是實驗五按鈕會附帶跑。沒有逐筆串流、續跑或中途取消，重跑重新fit。

每N須足夠建立train／cal且小於dst健康池長度；程式未完整驗證N≤0、N≥池大小，可能空集合或NaN／AUROC失敗，不能當成功。資料缺healthy／fault或工況不足先核對coverage，不造資料。多pairs會重複fit scratch，成本高於單工況，避免為讀手冊重跑。

## 輸出怎麼讀

main經setup_run寫logs/exp10_transfer/{ts}.log及output/exp10_transfer/{ts}/environment.json、summary.json、transfer_curve.png。rows鍵src／dst／n／strategy，healthy_accept是健康score≤1比例、fault_detect是fault score>1比例，AUROC是兩者排序；非known配置分類率。每row先round4，再按strategy×N等pair平均，不按樣本加權；沒有SD或逐樣本ID。

圖只畫AUROC曲線，須同時看JSON健康接受率；AUROC高但誤報高代表排序好、原警戒線仍不適合，不能直接部署。預期同N下adapted比scratch有額外價值才支持搬用來源模型；若無優勢如實記錄，沒有事先通用成功數字。direct與N>0的健康eval不同，跨N差值含樣本組成變動。

seed 42 的完整結果見 [output/exp10_transfer/2026-10-23-01-34-24](../../output/exp10_transfer/2026-10-23-01-34-24/summary.json)（Aero、commit `46ed4d5`、資料指紋 `1b55b847…`；與 2026-10-09 無主機紀錄的舊輸出扣掉環境欄位後相同）。本流程只支持現有工況的小樣本探索，不證明新session／廣泛motor泛化、固定5%誤報或RUL。

## 方法來源與程式碼

Sun、Feng、Saenko（2016），*Return of Frustratingly Easy Domain Adaptation*，AAAI30,2058–2065，[原文](https://arxiv.org/abs/1511.05547)提出CORAL二階相關對齊；本程式**只平移median，沒有共變異數白化／重著色，不是完整CORAL實作**，shift公式為Lineage操作約定。LW來源見[實驗一](exp1_cold_start.md#方法來源與程式定位)。

[exp10](../../src/experiments/exp10_transfer.py)管support／eval、策略、指標與writer；[monitor](../../src/core/monitor.py)建基準；[test_exp10_transfer](../../src/tests/test_exp10_transfer.py)檢查結構與索引不重疊。資料處理沿原始論文特徵來源，方法整合在Lineage，不依賴原始論文可執行碼。
