# 實驗六正式版：同條件比較Mahalanobis與k-NN

## 資料角色用量表

流程：讀實際工況 → 健康切分 → train建立正式factory模型 → cal定q95 → holdout與未知全池評估。方法與[共通原理](README.md#共通原理先問模型拿什麼比較)相同；這是健康／未知二元比較，不是六個known配置的分類器。

| 資料角色 | T1／8000rpm | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 健康train／cal／test | 健康池313列 | 187／62／64 | 各除以313 | 模型／q95／誤報 | 列級互斥 |
| 未知test | 九fault配置 | 2736 | 故障池100% | 拒絕與排序 | 不進fit／cal |
| 全root合計 | 九工況 | 1689／560／572，未知26070 | 逐池取整後加總 | 可用量查核 | 方法／seed會重用 |

來源為[用量產物](README.md#資料用量與分母怎麼讀)，不是新benchmark分數。此runner連續使用RNG；相同數量不表示與單工況monitor有相同索引。CLI正式coverage要求與Web的require_nine=False不同；現行`data`九工況齊全；少於九工況時不能當完整PASS。沒有validation選方法；unknown不調門檻。

## 發表與理解檢查

口頭主軸：「LW用健康分布的橢球距離，k-NN看離健康參考點有多遠；兩者都用相同健康校準比例定線。每種方法、seed與工況的輸出要對上，資料缺格就保留限制。」只改選單不代表封存矩陣也重跑。

- healthy-only是否能預測具名fault？不能，指標是健康接受或未知拒絕。
- 正式分位數能保證5%誤報嗎？不能，有限校準列與跨域會改test分布。
- 本次指紋能證明錄製獨立嗎？不能；main的舊fingerprint限制與待審#60要分開。
- 組員請解釋n_known_test與n_unknown_test的分母，再讀實際輸出節，不用一個accuracy涵蓋所有問題。

這支runner只用健康建立基準，透過同一個core.openset factory比較兩種偵測器，保留設定與工況指標。它與[八方法廣度比較](exp6_osr_benchmark.md)分開；不是多類故障配置分類，也不修改正式預設。

這份手冊給執行實驗六正式版的組員，回答資料如何切分、如何建立模型、如何執行與解讀輸出。`docs/experiments/README.md`、`docs/README.md` 及 `src/web/experiments.py:CATALOG[exp6_formal].doc` 引用本頁；runner 不讀取 Markdown，也不從本頁取得模型參數。完整欄名與通用載入拒絕政策另見[正式特徵規則](../formal_feature_schema_contract.md)，本頁只說明實驗六如何使用該規則。

## 資料、切分與模型

從data-root掃描105維clean特徵。CLI要求九個不同motor／RPM組合，API require_nine=False可作部分工程測試；九個數量通過不等於已驗證指定三馬達×三RPM全格、來源或視窗獨立。列的物理語意、未知單位與loader限制見[實驗一](exp1_cold_start.md#資料與載入)。

工況按motor／rpm字串排序，default_rng(seed=42)連續洗牌各健康池；約60%train、20%cal、20%holdout。同來源、集合順序與seed才有相同列切分。RobustScaler只fit健康train。LW只用縮放train估中心／共變異數；k-NN保存同一train庫，預設k=5，資料少時縮減有效k。各法cal距離0.95分位數定線，score>1判未知。所有其他配置全池只打分／算指標，不fit、不校準、不選參。legacy因用train距離定線而明確拒絕。沒有神經網路epoch。

每工況先fit→校準→一次健康與fault打分→二元指標→記錄來源與參數。known_accuracy是**健康接受比例**，不是已知故障配置分類率。T1／T2／T3是不同個體；本實驗各自同工況建模，並非專用三馬達train／cal／test研究。

## 操作

在repo根目錄依[環境政策](../runtime_policy.md)準備uv、資料：

```bash
uv run --locked python -m experiments.exp6_formal_benchmark --help
uv run --locked python -m experiments.exp6_formal_benchmark --data-root data --seed 42 --openset-method mahalanobis --method ledoit_wolf --confidence 0.95
uv run --locked python -m experiments.exp6_formal_benchmark --data-root data --seed 42 --openset-method knn --knn-neighbors 5 --confidence 0.95
```

API run(data_root,seed=42,confidence=0.95,openset_method="mahalanobis",mahalanobis_method="ledoit_wolf",knn_neighbors=5,require_nine=True)回傳dict，並保存來源與環境紀錄；API不寫CLI的results.csv與圖。confidence須在0與1間；缺九工況、healthy、unknown或非有限score明確失敗，先查來源，不借test調門檻。

### 輸入規則

正式 `require_nine=True` 入口另驗每工況完整10個配置，cold-start與顯式partial入口允許健康資料或較少配置，但記錄缺類。ordered105維header、finite/nonempty與alias政策見 [正式特徵規則](../formal_feature_schema_contract.md)。不改健康train／cal／holdout、q95、模型、指標或既有數值；不是新的研究比較。預期非法資料明確拒絕、合法來源逐元素與切分不變，達不到即工程驗收失敗。

來源指紋另依 [內容來源規則](../content_provenance_contract.md) 版本化為 `source_content_v2`；每次讀取實際CSV bytes，舊stat／manifest指紋僅明確legacy讀取。formal API記環境與來源，模型／閾值／切分不變；預期同stat異bytes可區分、同內容副本不冒充獨立採集。

Web啟動見[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)，選「實驗六正式版」、方法／seed後「▶ 執行」。Web用require_nine=False，因此看見成功不代表正式九工況完整；須看formal_condition_count。沒有逐筆串流、取消或續跑。多seed持久化由[矩陣](exp6_matrix.md)負責，不用手動覆寫歷史目錄。

## 輸出怎麼讀

main經setup_run寫logs/exp6_formal_benchmark/{ts}.log及output/exp6_formal_benchmark/{ts}/environment.json、summary.json、results.csv、osr_benchmark.png。Web另寫web_server的experiments JSON。

`run()` 不呼叫 setup_run、不動 loguru（#269）：呼叫端給 `evidence_dir` 時才寫 provenance.json 與 environment.json（CLI 傳自己的 output 目錄並以 `reuse_environment=True` 沿用 setup_run 剛寫的 environment.json；[矩陣](exp6_matrix.md)傳每格的 `runs/{run_id}/`，不傳旗標，重跑時重新收集並覆寫舊的 environment.json，#285），`provenance_record_id` 記該目錄最後兩節；Web 不傳，磁碟上不留目錄，source 與 environment 只放在回傳值。資料來源只在執行前後各 capture_source 一次，`dataset_fingerprint` 沿用執行前那次。失敗紀錄保留 error_type，不公開任意例外內容；私人解析根只在被 Git 忽略的 `.lineage_private/`。

| 欄位 | 含義 |
|---|---|
| n_train／n_calibration／n_known_test／n_unknown_test | 每用途實際列數，不是錄製批次 |
| known_accuracy／open_set_accuracy | 健康接受率／健康對fault二元正確率；後者受類別比例影響 |
| auroc／aupr_unknown_positive | unknown為正類的排序／平均精確率；AUPR須同時看正類比例 |
| unknown_precision／unknown_recall／unknown_f1 | 判未知的精確率／召回／調和平均，零分母按sklearn設0 |
| fpr_at_tpr95 | `core.openset.fpr_at_tpr`：unknown為正類，取ROC上TPR≥0.95的最小FPR（#235，實驗六廣度、正式版與實驗七共用）；事後評估點，不是已校準部署線 |
| raw_thresholds／threshold_strategy | cal的原距離線與政策，正規化threshold固定1 |
| inference_seconds／peak_memory_mb | 兩次打分的耗秒，不含fit；記憶體未量測為null |

dataset_fingerprint 預設為 `source_content_v2`，對排序的相對檔案ID與實際CSV內容SHA清冊取雜湊；有manifest時另驗清冊與已宣告輸出SHA，前後來源變動拒絕完成。相同內容副本不代表獨立採集。明確指定 `legacy_fingerprint_v1` 才使用舊stat／manifest指紋，不回寫舊結果。schema 3摘要包含 source、environment、provenance_record_id 與有效 config；未保存逐樣本預測或模型checkpoint。

預期完整九列、有限指標且unknown排序高／健康誤報低；沒有通用可靠模型PASS門檻。不把不同資料指紋或seed直接當改善。

2026-10-23 在 Aero 以 commit `46ed4d5`、`--data-root data`（資料指紋 `1b55b847…`）跑完 54 格：健康接受率（known_accuracy）27 個工況-seed 列等權平均，LW 94.66% ± 3.97（最低 87.50%，T3／6000 seed 42）、k-NN 94.67% ± 4.38（最低 84.38%，T1／6000 seed 42），AUROC 與未知召回都是 1，見 [aggregate.md](../../output/exp6_formal_matrix/aero_2026-10-23/aggregate/aggregate.md)。舊版 94.62% ± 4.01／94.57% ± 4.39 是 `data/formal_local` 的歷史結果（commit 07582ff、Python 3.12.14、無主機紀錄，[舊 aggregate](../../output/exp6_formal_matrix/aggregate/aggregate.md)）；`formal_local` 由階段 ZIP 物化，不在 Aero／Spark 上，兩份資料的列數差異見 [#242](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/issues/242)。不保證5%現場誤報、錄製獨立或fresh final。

## 方法來源與程式碼

馬氏距離、LW與近鄰來源見[實驗一](exp1_cold_start.md#方法來源與程式定位)；平均k距離／分位數正規化是Lineage操作約定。原始論文提供馬氏核心來源，Lineage整合factory與正式比較。

[runner](../../src/experiments/exp6_formal_benchmark.py)負責流程／指標／writer；[factory](../../src/core/openset.py)建立兩法；[test_exp6_benchmark](../../src/tests/test_exp6_benchmark.py)檢查known-only與指標；[test_openset](../../src/tests/test_openset.py)驗證參考與校準分離；[web/experiments](../../src/web/experiments.py)編排。這些測試不證明raw來源鏈完整。
