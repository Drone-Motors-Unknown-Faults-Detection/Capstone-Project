# 實驗技術報告

本目錄有13個現行實驗編號（1～13）、16份方法與操作手冊。實驗六的基準、正式比較、矩陣與彙總各有手冊。實驗八在 [exp8_health.md](exp8_health.md)。實驗九用實驗六的 runner。各手冊說明資料、擬合／校準／測試、CLI／API／Web、輸出與限制。逐項參數以各手冊及程式為準。

載入筆數以本次資料與輸出為準。手冊不替缺少的採集紀錄填值。

## 名詞

| 名詞 | 意思 |
|---|---|
| 樣本 | 一段運轉濃縮成的 105 維特徵列 |
| 健康基準 | 只用健康 train 建立的統計模型 |
| 開集分數 | 離健康多遠。大於 1 判未知 |
| 隔離區 | 分數大於 2 的未知樣本才累積，湊滿後嘗試分群 |
| 量尺 | 目前認識的類別。一開始只有健康 |

健康資料預設切成 train／cal／holdout 約 60／20／20。故障列用來考，確認之前不用來擬合。預設 seed 42；實驗六、八的矩陣另用 123 與 2026。

## 即時展示畫面

`./scripts/opt/run_web.sh` 打開首頁。頁首「實驗一」到「實驗十二」是實驗頁籤。實驗九只說明，執行在實驗六。

| 畫面 | 對應 | 看什麼 |
|---|---|---|
| 開集分數串流圖 | 實驗一 | 分數相對判定線 1.0 |
| 未知故障處理與候選卡 | 實驗二 | 隔離累積、分群、確認後量尺增加 |
| 變化點分析卡 | 實驗三 | EWMA；警報時判漸進或突發 |
| 極座標健康地圖 | 實驗四 | 方向與半徑 |
| 偵測統計表 | 實驗一／二 | 學會前看拒絕，學會後看接受 |
| 資料集下拉 | 實驗五 | 各工況各自冷啟動 |

實驗六、八頁面下方的封存卡只讀已提交結果，不隨上方新選項重跑。

## 常見問題

校準集約 62 筆時，名義 5% 的健康誤報會到約 10.9%。隔離線設在分數 2.0，邊界誤報不進新類發現。確認那一刻才揭示配置標籤。同一 seed、同一指令重跑，隨機切分相同；每次仍會在 `logs/` 與 `output/` 留檔。

| 編號 | 報告 | 程式 | 在問什麼 |
|---|---|---|---|
| 實驗一 | [exp1_cold_start](exp1_cold_start.md) | `src/experiments/exp1_cold_start.py` | 只用 `8screws` 建基準，九種故障抓不抓得到 |
| 實驗二 | [exp2_scale_growth](exp2_scale_growth.md) | `src/experiments/exp2_scale_growth.py` | 未知樣本能否分群、成群後只用該群自學擴張量尺 |
| 實驗三 | [exp3_trend](exp3_trend.md) | `src/experiments/exp3_trend.py` | 漸進鬆脫與突發切換能否從分數序列分開 |
| 實驗四 | [exp4_polar_map](exp4_polar_map.md) | `src/experiments/exp4_polar_map.py` | 故障方向與半徑能不能對上配置 |
| 實驗五 | [exp5_cross_condition](exp5_cross_condition.md) | `src/experiments/exp5_cross_condition.py` | 某一工況的健康基準能否搬到別的工況 |
| 實驗六 | [exp6_osr_benchmark](exp6_osr_benchmark.md) | `src/experiments/exp6_osr_benchmark.py` | 八種偵測器的排序與誤報；legacy校準方式須另解讀 |
| 實驗六 | [exp6_formal_benchmark](exp6_formal_benchmark.md) | `src/experiments/exp6_formal_benchmark.py` | 主線 Mahalanobis 與 k-NN 的單次正式比較 |
| 實驗六 | [exp6_matrix](exp6_matrix.md) | `src/experiments/exp6_matrix.py` | 上面那次比較的 9×3×2 可續跑矩陣 |
| 實驗六 | [exp6_aggregate](exp6_aggregate.md) | `src/experiments/aggregate_exp6.py` | 把 exp6 矩陣已完成的 summary 彙總，缺檔就停 |
| 實驗七 | [exp7_compare_openset](exp7_compare_openset.md) | `src/experiments/compare_openset.py` | 單一工況上、同一 split 的 Mahalanobis 對 k-NN |
| 實驗八 | [exp8_health](exp8_health.md) | `src/experiments/health_index_benchmark.py`、`health_index_matrix.py`、`health_monitor.py`、`health_index_aggregate.py` | 相對健康指數、54 列矩陣、逐窗告警、封存重算 |
| 實驗九 | [exp9_autoencoder](exp9_autoencoder.md) | `src/core/detectors.py`（`MLPAutoencoderDet`） | 非線性 AutoEncoder 重建誤差，能不能比線性 PCA 重建更分得開健康與故障 |
| 實驗十 | [exp10_transfer](exp10_transfer.md) | `src/experiments/exp10_transfer.py` | 目標工況少量健康資料，中心平移遷移划不划算、還是從零冷啟動就好 |
| 實驗十一 | [exp11_ancestor_comparison](exp11_ancestor_comparison.md) | `src/experiments/exp11_ancestor_comparison.py` | 原始論文協定下 legacy vs Ledoit–Wolf 的 Accuracy/Balanced Accuracy/macro-F1 對照 |
| 實驗十二 | [exp12_confusion_tsne](exp12_confusion_tsne.md) | `src/experiments/exp12_confusion_tsne.py` | 開集混淆矩陣與 t-SNE 視覺化，模型區分正常與故障的效果看不看得出來 |
| 實驗十三 | [exp13_experiment_navigation](exp13_experiment_navigation.md) | `src/experiments/navigation_data_contract.py`、`navigation_regression.py` | 獨立導覽與原主線計算是否一致；整理歷史七問，不新增研究成績 |

實驗七比較同條件開集方法；實驗八提供健康量尺與趨勢。

實驗九～十二依序檢查非線性重建、跨工況平移、原始論文協定對照及可視化。實驗十的direct與adapted／scratch不完全共用測試健康列；實驗十一的legacy分類label固定為-1，與逐類LW的分差混合了分類API差異，不能直接解讀成共變異數估計帶來的改善。實驗十二的t-SNE使用包含known fit列的完整資料池，只作圖像診斷。各手冊列出實際流程與固定產物；入口存在不等於獨立驗收通過。

## 先分清頁面與研究問題

首頁 `/` 的「即時展示」持有一個持續存在的模型；`/#exp1`～`/#exp8` 是重新執行實驗的頁籤，不是獨立伺服器路由。選頁籤本身不代表完成研究。下表的 ID 是 `web.experiments.CATALOG` 與 API 使用的識別字；同一編號的子卡片不重複計算研究成果。

| 頁面／入口 | runner／主要手冊 | 主問題與實際用途 | 資料角色與呈現限制 |
|---|---|---|---|
| 即時展示 `/` | `web.live.LiveDemo`；[導覽手冊](exp13_experiment_navigation.md) | 串起偵測、隔離、候選、模擬確認及模型更新 | 初始只 fit 健康；確認後使用完整配置池。首頁具名來源是展示真值，不是模型猜出的故障名稱 |
| `/#exp1`，ID `exp1` | `exp1_cold_start.run/iter_run`；[實驗一](exp1_cold_start.md) | 健康基準會誤報多少、能拒絕多少未知配置？ | 同工況健康 train/cal/holdout；故障全池只測試，沒有訓練故障分類器 |
| `/#exp2`，ID `exp2` | `exp2_scale_growth.run`／`iter_playback`；[實驗二](exp2_scale_growth.md) | 候選群能不能只用自身樣本變成新類？ | 邊跑邊畫與批次都在群出現時自學。類名是模型新類；配置名只做事後對照。不用完整配置池 |
| `/#exp3`，ID `exp3` | `exp3_trend.run/iter_run`；[實驗三](exp3_trend.md) | 合成的漸進／突發順序能否辨別？ | 頁上可只播劇本 A 或 B。「邊跑邊畫」兩個劇本一起跑。延遲單位是重播筆數，不是現場秒數或壽命 |
| `/#exp4`，ID `exp4` | `exp4_polar_map.run/iter_run`；[實驗四](exp4_polar_map.md) | 已確認配置的方向與半徑如何呈現？ | 幾何可解釋圖；不是經物理標定的故障程度。Web 沒有 Open Set 方法選單，走 runner 固定設定 |
| `/#exp5`，ID `exp5` | `exp5_cross_condition.run`；[實驗五](exp5_cross_condition.md) | 一個工況的健康基準能否搬到另一工況？ | 跑實際掃到的工況；先看 source→target，不能只報總平均或預設一定九組 |
| `/#exp6`，ID `exp6` | `exp6_osr_benchmark.run`；[廣度比較](exp6_osr_benchmark.md)、[實驗九](exp9_autoencoder.md) | 八種單類方法的排序與誤報有何差異？ | 包含 AutoEncoder；legacy 使用不同門檻來源。不是八種獨立切分實驗 |
| `/#exp6`，ID `exp6_formal` | `exp6_formal_benchmark.run`；[正式比較](exp6_formal_benchmark.md) | 固定一個方法與 seed，檢查各工況二元識別 | Web 允許少於九工況，必須看實際 coverage；不是已知五配置／未知四配置研究 |
| 同頁 6-3 封存卡 | `/api/committed/exp6_formal_matrix`；[矩陣](exp6_matrix.md)、[彙總](exp6_aggregate.md) | 唯讀查看保存的配對矩陣 | 不受上方新選項控制；不能把封存分數當成本次執行 |
| `/#exp7`，ID `exp7` | `compare_openset.run`；[實驗七](exp7_compare_openset.md) | 同工況同 split，LW 與 k-NN 差多少？ | healthy-only，兩方法共用資料角色；不是跨馬達 fault-type 分類 |
| `/#exp8`，ID `exp8` | `health_index_benchmark.run`；[基準](exp8_health.md#基準) | 偏離分數轉成相對量尺後，兩群相隔多遠？ | 分位數由健康校準列建立；0～1 是相對刻度 |
| 同頁 8-2 封存卡 | `/api/committed/exp8_health_index_results`；[矩陣](exp8_health.md#矩陣)、[封存重算](exp8_health.md#封存重算) | 唯讀查看保存的 54 列 | 歷史彙總的兩欄差異見重算產物；頁面不重跑 54 列 |
| `/#exp8`，ID `exp8_monitor` | `health_monitor.run/iter_run`；[逐窗監測](exp8_health.md#逐窗監測) | 選定配置的相對指數與告警怎麼走？ | unknown 取其全池再截 max_windows；順序未證實為原始時間 |
| `/#exp9`，ID `exp9` | 無獨立 runner；[實驗九](exp9_autoencoder.md) | 非線性重建是否比線性重建分得開？ | 頁籤只說明。權重在實驗六的 mlp_autoencoder，由健康 train 建立 |
| `/#exp10`，ID `exp10` | `exp10_transfer.run`；[實驗十](exp10_transfer.md) | 少量目標健康資料的中心平移有沒有用？ | 網頁用所選 seed 跑全部掃描工況。direct、adapted、scratch 的資料角色要逐項看 |
| `/#exp11`，ID `exp11` | `exp11_ancestor_comparison.run`；[實驗十一](exp11_ancestor_comparison.md) | 同一原始論文-style 協定的分類 API 有何差別？ | 網頁只跑所選的一個 seed。legacy 的 -1 label 會影響分類分數，不能全歸因於 LW |
| `/#exp12`，ID `exp12` | `exp12_confusion_tsne.run`；[實驗十二](exp12_confusion_tsne.md) | 混淆矩陣與投影能看出哪些錯誤？ | 網頁畫出混淆矩陣與 t-SNE。t-SNE 混入 fit 列只作診斷，不是獨立測試 |
| 獨立導覽 `/assets/guide.html` | `web.guide`，另啟服務；[實驗十三](exp13_experiment_navigation.md) | 帶操作員走過同一串流流程，並顯示研究細節 | 展示／研究模式共用 session，不新增實驗。主首頁沒有直連這個獨立服務的入口 |

呼叫鏈：實驗卡片的選項 → `POST /api/experiments/{id}/run` 或 `GET /api/experiments/{id}/stream` → `ExperimentRunner._clean/_dispatch` → 上表 runner → `jsonable` → `experiments.js` 的 renderer。封存卡另走 `/api/committed/{name}`；導覽走自己的 `/ws`。`exp1`、`exp2`、`exp3`、`exp4`、`exp8_monitor` 支援 SSE。實驗二的邊跑邊畫在候選出現時自行納入，不再等確認。實驗五到八的批次卡，以及實驗十到十二，按下執行後出表或圖，沒有逐筆 session。實驗九有頁籤，沒有獨立執行。

講解時先選正確問題：判未知比例回答「拒絕」，`predict_known_class` 才涉及「已知分類」，HDBSCAN 候選回答「形成群」，確認後更新回答「學進模型」，EWMA 回答「序列變化」。不把這些都叫準確率。成功標準與未通過條件見各手冊；現行 runner 沒有共同的部署可靠性 PASS。

## 資料用量與分母怎麼讀

現行 `data` 是九工況、90 個特徵 CSV、28,891 個有效105維特徵列，資料指紋 `source_content_v2:1b55b847…`。下表取自 2026-10-23 在 Aero 以 commit `46ed4d5` 跑的[實驗六正式矩陣](../../output/exp6_formal_matrix/aero_2026-10-23/matrix_manifest.json)：「全部有效列」是各 run `input_schema_audit` 十個檔的列數加總（rejected／discarded 皆 0），H 的三個角色與 U 是 seed 42 那次切分的 `n_train`／`n_calibration`／`n_known_test`／`n_unknown_test`；seed 123、2026 的數量相同。

| 工況 | 全部有效列 | 健康池 H | 健康 train／H | 健康 cal／H | 健康 holdout／H | 故障全池 U |
|---|---:|---:|---:|---:|---:|---:|
| T1／11000rpm | 3349 | 357 | 214／357 | 71／357 | 72／357 | 2992 |
| T1／6000rpm | 3354 | 320 | 192／320 | 64／320 | 64／320 | 3034 |
| T1／8000rpm | 3049 | 313 | 187／313 | 62／313 | 64／313 | 2736 |
| T2／11000rpm | 3417 | 253 | 151／253 | 50／253 | 52／253 | 3164 |
| T2／6000rpm | 3186 | 324 | 194／324 | 64／324 | 66／324 | 2862 |
| T2／8000rpm | 3249 | 368 | 220／368 | 73／368 | 75／368 | 2881 |
| T3／11000rpm | 3127 | 300 | 180／300 | 60／300 | 60／300 | 2827 |
| T3／6000rpm | 2939 | 314 | 188／314 | 62／314 | 64／314 | 2625 |
| T3／8000rpm | 3221 | 272 | 163／272 | 54／272 | 55／272 | 2949 |
| 合計 | 28891 | 2821 | 1689 | 560 | 572 | 26070 |

這份 `data` 不是研究分支 90 個 CSV／28,910 列的資料版本，也不是 `data/formal_local`：`formal_local` 由階段 ZIP 物化，T1／11000rpm 健康池是 358 列，九工況健康 holdout 合計 573、故障 26,088（[#242](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/issues/242)）。2026-10-14 的[唯讀用量產物](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/blob/17302e2ad1885ed8bb4f227401b6a4ca453e6bb4/output/experiment_data_usage_evidence/2026-10-14-19-36-34/usage.json)是當時某台本機沒有 T2 的六工況版本（60 個 CSV、19,053 列），只作歷史紀錄，數量不能和上表混用。

H的三個角色互斥、合計H；U另列，不把未知全池混進「20%測試」。例如T1／8000rpm是健康池187+62+64=313，再加2736未知列；不是全部3049列的60/20/20。三個seeds為42／123／2026，數量相同但索引摘要不同，不能累加成三倍獨立資料。

要重新計數或看多類原始論文 A、逐配置數量、監測截取上限、遷移support與exp5策略重疊，在 Aero 或 Spark 跑下面的唯讀工具（只呼叫既有loader與`make_split`，沒有fit、選參或算新分數）；資料版本不同時重新計數，不複製本表：

```bash
uv run --locked python -m tests.experiment_data_usage_evidence --data-root data --seeds 42,123,2026
```

writer是[src/tests/experiment_data_usage_evidence.py](../../src/tests/experiment_data_usage_evidence.py)，經setup_run保存`logs/experiment_data_usage_evidence/{ts}.log`、`output/experiment_data_usage_evidence/{ts}/environment.json`與`usage.json`；fixture見[test_experiment_data_usage_evidence.py](../../src/tests/test_experiment_data_usage_evidence.py)。來源讀取前後逐檔SHA一致；raw/session獨立性與fresh資格保留UNKNOWN。特徵列索引不能恢復清理前視窗、刪點遮罩或採集時間。

## 共通原理：先問模型拿什麼比較

RobustScaler先把每欄減去健康train的中位數，再除以train四分位距；其作用是讓大單位欄位不直接壓過小單位欄位。常數欄的零尺度由套件安全處理，不手算除零。用test學縮放量會讓基準偷看評估資料，因此其餘角色只transform。[官方定義](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.RobustScaler.html)對應`src/core/monitor.py:_refit`。

馬氏距離把新向量和健康中心比較，並考慮各欄的共同變動。縮放後輸入z、train中心μ、精度矩陣P（共變異數的逆）給出`d(z)=sqrt((z−μ)ᵀP(z−μ))`。LW把經驗共變異數S向等向目標收縮：`Σ=(1−α)S+α·trace(S)/p·I`；p是特徵維度，α由train估計，不是本輪調的係數。這讓少量高維樣本的逆矩陣較穩，不能保證跨馬達分得開。[Ledoit–Wolf公式及原論文](https://scikit-learn.org/1.7/modules/generated/sklearn.covariance.LedoitWolf.html)；距離實作在`src/core/mahalanobis.py:_distances`。

k-NN改成與該類train參考點比鄰近程度：`d_k(z)=mean(||z−z_j||₂)`，j為最近k個參考點。cal列只能查鄰居，不加入參考庫。LW／k-NN各以cal距離的95%分位數q正規化，`s=d/max(q,eps)`；多類取最小s，大於1才拒絕。**legacy例外用train自身定q**。分數沒有物理單位，1是經校準的界線，不是機率。名義95%只描述校準規則，不保證新馬達固定5%誤報。

講解指標時說分母：未知召回=拒絕的未知／全部未知，健康誤報=拒絕的健康／全部健康，已知分類正確率=猜對配置／已知測試列。AUROC問的是未知分數有沒有排在健康前，與固定門檻的召回不同。新類別被接受不等於設備修好；健康指數也不等於剩餘壽命。

## 共用模型生命週期

`core.monitor.OpenSetMonitor` 建構後尚未擬合。`score()`、`classify()`、`project()`、`summary()` 都要求完整成功的擬合，否則丟出 `RuntimeError` 並要求成功呼叫 `fit_initial()`。健康基準只用 healthy train 擬合 scaler、detector 與 PCA，known calibration 設門檻；未知資料不參與擬合與校準。

`fit_initial()` 與 `_refit()` 開始即撤銷有效旗標，所有模型及摘要狀態完成後才恢復。重擬合失敗時禁止混用前次模型與新 scaler；操作者須明確重新建立健康基準。`add_class()` 使用已確認配置的完整資料池重新擬合，沒有原子回滾保證；失敗後已改動的 known／splits 不能當作有效模型。無效或重複配置若在註冊前被拒，原模型仍可使用。

`holdout(config)` 只讀既有 splits，不使用推論 guard，未註冊配置保留 `KeyError`。成功擬合後，Mahalanobis-LW／k-NN 均以正規化分數大於 1 拒絕；PolarMap 幾何不受本生命週期說明改動。回歸入口為 `src/tests/test_monitor_guard.py`、`src/tests/test_openset.py` 與 `src/tests/test_geometry.py`。

### 共用設定解析

所有 Open Set 入口的方法參數都經 [`core.openset.resolve_openset_config()`](../../src/core/openset.py) 解析一次，`create_openset_detector()` 與 `OpenSetMonitor` 也走同一個函數。CLI（`core.runner.add_openset_args` 解析後交給 `run()`）、程式 `run()` 與 Web catalog 呼叫的是同一段解析，因此同一組參數得到同一份設定。回傳的 resolved config（`version="openset_config_v1"`）寫進各入口的輸出：

| 欄位 | 內容 |
|---|---|
| `method_id` | 穩定方法 ID：`mahalanobis/ledoit_wolf`、`mahalanobis/oas`、`knn/k5` 等 |
| `openset_method`、`requested_method` | 正規化後的方法名與呼叫端原本傳入的字串（例如 `maha` → `mahalanobis`） |
| `effective` | 實際生效的參數：`confidence`、`threshold=1.0`，以及 Mahalanobis 的 `mahalanobis_method` 或 k-NN 的 `knn_neighbors` |
| `ignored` | 有傳入但這個方法不使用的參數，附原因；k-NN 會列 `mahalanobis_method`，Mahalanobis 會列 `knn_neighbors` |

輸出位置：`OpenSetMonitor.summary()["resolved_config"]`（實驗一、二、三、四、七與 Web 的 `model` 都帶這份）、實驗六正式版與實驗八 benchmark 結果的頂層 `resolved_config`。

非法值在擬合前就以 `OpenSetConfigError`（`ValueError` 子類）拒絕，訊息固定以參數名開頭，不退回 library 預設值：未知方法、`confidence` 不在 (0, 1) 或不是有限數、`knn_neighbors` 不是 ≥1 的整數（`5.7`、`True` 都拒絕）、`mahalanobis_method` 不在 `legacy／ledoit_wolf／oas／mcd`。用 k-NN 時 covariance 選項不影響模型，只記在 `ignored`。特徵維度不符由 detector 的 `fit()`／`score_samples()` 檢查，訊息帶實際與預期維度。

預設值不變：Mahalanobis-LW、confidence 0.95、k=5；分數、門檻與事件順序不變。回歸測試在 `src/tests/test_openset.py`。

## Web 串流生命週期

### 問答結果的閱讀順序與安全呈現

實驗十二沿用`summarize_matrices()`核算，Web只組裝「來源／時間、三句內摘要、二元表、多類比較、必要限制」，逐類、t-SNE、資料角色、原矩陣與安全來源物件另列詳情。文字版與可選`assistant_presentation_v1`使用同一份資料，不交由模型重算。表格含合法Markdown分隔列；特殊字元以文字跳脫，完整標籤與來源另保留。原核算函式只補出既有比率函式可表示的漏判率／未知誤接受率欄位供排版，計數與公式不變。

來源與設定使用有標題的列表。明確offset的時間固定顯示為Asia/Taipei（UTC+08:00）；原UTC字串、機器欄位、完整路徑及SHA留於技術詳情，缺少原SHA不另算一個冒充。時區不明、缺開始、時間矛盾及資料不完整預設可見；時間檢查通過只描述起訖與量測，不代表實驗可靠。時間品質判定仍只使用`run_timing.timing_fields()`。

可信結果的mode／origin與來源識別必須一致，才可使用可選presentation。舊客戶端仍讀可讀Markdown；無presentation或資料非法不換來源。前端以DOM建立details／summary與表格，模型HTML不經這條路徑。完整回覆後一次呈現詳情，工具進度不串長箭頭，依據逐項列出。HTTP環境複製失敗時顯示可選取的完整文字，不假稱已複製。這些是排版與操作契約，不新增實驗編號、改模型或改資料。

`python -m tests.assistant_presentation_evidence --help`為8787專用呈現驗收入口，唯讀指定新／舊JSON，核對OCI版本、實際程式／前端雜湊、六次決定論問答與實際SSE→JS；不重跑t-SNE或其他研究。另加`--method-probe`才執行一次有界一般方法問答。經`setup_run()`寫入`logs/assistant_presentation_evidence/`及`output/assistant_presentation_evidence/`，保留失敗，不把HTTP／Node結果當作瀏覽器畫面驗收。

原始驗收產物含問答與來源資訊，對外發布須先確認授權。`python -m tests.presentation_public_evidence --help`只讀取已完成產物，以白名單匯出測試數量、狀態與公開程式版本到`output/assistant_presentation_public_evidence/{時間戳}/public_summary.json`；不複製問答、伺服器路徑、來源metadata或容器識別。只獲授權公開去敏摘要時，不提交原始產物或整個執行目錄。

### 執行紀錄的起訖與耗時

`src/web/experiments.py`在參數驗證完成、worker進入實驗執行邊界時擷取一次`started_at`；完成結果轉換後擷取`finished_at`。新時間為UTC、ISO 8601毫秒格式，含`+00:00`。`seconds`由`time.perf_counter()`量測並四捨五入至兩位，不以系統時鐘相減；排除參數驗證、worker排隊、存檔及HTTP傳輸。批次包含資料載入、實驗與JSON可序列化轉換；串流從generator首次執行開始，包含frame轉換、消費端播放節流及等待，屬該次串流經過時間，不是純模型計算時間。`timing_scope`分別為`batch_execution`與`stream_elapsed_including_playback`，`timing_version`為`web_run_timing_v1`。

起訖由runner傳到payload、原始JSON、HTTP／done、PlaybackTracker、問答工具與來源摘要；讀檔或問答時不重建實驗起點。`updated_at`只表示紀錄更新，不代表實驗開始／完成。頁面直接顯示保存的ISO時間，不暗中轉成瀏覽器或Docker當地時間。

「何時開始／完成、花多久」直接從同一紀錄產生`mode=record_timing`、`origin=record`的SSE正文，不呼叫模型猜時間；指定存檔仍通過既有保存根目錄與實驗身分檢查。前端不把這份程式文字存為模型歷史。實驗十二其他結果摘要仍維持`exp12_standard_answer_v2`。

舊檔沒有`started_at`就保留缺值，顯示「開始時間：原檔未記錄」；naive時間標示「原檔未提供時區」。不由檔名、父目錄、mtime或完成時間減耗時推算開始。保留有效結果及原始完成時間；格式錯誤、負／非有限耗時、起訖順序或量測不一致均提示，不覆寫歷史JSON。時間一致性比較依毫秒與兩位耗時精度容許0.011秒，舊秒級時間容許1.01秒；不同時區資訊無法比較時明說。極短執行可落在同一顯示時間，時鐘回撥仍保留實際擷取值與單調耗時，不偽裝正常。

時間僅為執行紀錄，不是資料採集或raw window時間；不改資料切分、指標或模型。契約由`src/tests/test_web_run_clock.py`、`test_run_timing.py`及既有runner／playback／HTTP／前端測試驗證。

`python -m tests.execution_timing_evidence --help`提供8787專用驗收入口：唯讀SSH核對指定舊檔、透過HTTP建立一次實驗一、比對原始JSON與記憶體／存檔時間問答，並執行實際SSE到前端JS測試。再次驗收使用`--new-saved`沿用新紀錄，不重複fit。程式經`setup_run`寫`logs/execution_timing_evidence/{ts}.log`與`output/execution_timing_evidence/{ts}/`；HTTP與Node結果不能當實際瀏覽器操作。它拒絕正式站URL及非標準保存路徑，SSH沿用已授權的使用者專用金鑰，不修改登入設定。

`web.experiments.CATALOG` 的 stream 旗標決定可播放的實驗：exp1、exp3、exp4、exp8_monitor。既有 `run()` 批次與 `iter_run()` 逐步計算共用實驗邏輯；展示編排不自行計算另一套指標。使用 `uv run --locked python -m web.server --bind-address 127.0.0.1 --data-root data --port 8600` 啟動本機展示，再在實驗卡片選批次執行或「邊跑邊畫」。

`GET /api/experiments/{id}/stream` 接受 JSON 物件 params 與有限 rate；速率限制在 1～1000 筆／秒，只控制播放節奏。SSE 事件從 start 開始，再推送 scores／tick／window 等實驗事件，完成才發 done。done 的 payload 已由 runner 存入 `output/web_server/{時間戳}/experiments/`，帶 `streamed: true`；停止或失敗沒有 done 就不得當作完整結果。

批次與 SSE 使用同一執行鎖，另一工作執行中即拒絕。瀏覽器停止或斷線時中止後續播放；已交給 executor 的一次計算可能仍需完成。finally 關閉 generator，即使 close 丟例外也釋放鎖。完成、取消或錯誤後，待伺服器釋放鎖再重試；不能以重新點擊掩蓋中斷。回歸入口為 `src/tests/test_stream_integration.py`、`src/tests/test_health_monitor.py`。

前端呈現規則另由 `src/tests/test_web_experiment_alignment.py` 呼叫 Node.js 執行真正的 `experiments.js`，驗證舊設定結果標示、空資料、失敗後重試與欄位映射；執行測試的 PATH 需有 Node.js。這是合成 DOM 工程測試，不能代替實際瀏覽器操作或組員理解驗收。頁面顯示保存結果的時間與參數；選項改變但尚未重跑時，下方明標舊設定結果，封存矩陣不隨新選項變動。

獨立導覽另用 [UI 操作腳本](../navigation/exp13_UI操作腳本.md) 的 WebSocket session：重連不自動播放，模型失敗須明確重設或重建。兩個入口均為 CSV 重播，不保證實體時間、原始視窗獨立或部署誤報率。現行 server／guide 的部分錯誤會把例外文字傳給 client，可能包含私人路徑；這項限制仍在，不能宣稱全部錯誤回應已去敏。[#28](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/28) 已關閉（NOT_PLANNED）：系統只在封閉實驗室網路使用，不處理；要部署到非封閉網路時另開 issue。

## 各實驗的檔案位置

實驗八重算補充：`src/experiments/health_index_aggregate.py`、`src/tests/test_health_index_aggregate.py`、`src/experiments/exp8_health_index_aggregate_contract.json`；紀錄寫入 `logs/exp8_health_index_aggregate/` 與 `output/exp8_health_index_aggregate/`，不經 web 層。

一個實驗的程式常分散在 `src/experiments/`、`src/core/`、`src/experiments/health/`、`src/web/`、`src/tests/` 與 `reports/`。下表列出入口與相關位置；健康監測的封存結果讀法見[reports與預期成果](exp8_health.md#reports與預期成果)，各手冊的程式碼節說明用途。

| 編號 | 入口 | 邏輯 | 測試 | Web | 已提交的紀錄與結果 | 其他文件 |
|---|---|---|---|---|---|---|
| 實驗一 | `src/experiments/exp1_cold_start.py` | `src/core/monitor.py`、`src/core/openset.py`、`src/core/mahalanobis.py`、`src/core/data.py` | 無專屬（`src/tests/test_openset.py` 測共用偵測器） | `src/web/live.py`、`src/web/static/index.html` 分數串流圖；`src/web/experiments.py` 實驗頁 | `logs/`、`output/exp1_cold_start/` | 本頁「名詞」 |
| 實驗二 | `src/experiments/exp2_scale_growth.py` | `ScaleGrowthSession`、`src/core/monitor.py`、`src/core/geometry.py` | `src/tests/test_exp2_scale_growth.py` | `src/web/live.py`、`src/web/server.py`、`src/web/static/index.html` 候選卡；`src/web/experiments.py` 實驗頁 | `logs/`、`output/exp2_scale_growth/`；`output/web_server/` | 本頁「名詞」 |
| 實驗三 | `src/experiments/exp3_trend.py` | `src/core/trend.py` | 無專屬 | `src/web/live.py`、`src/web/static/index.html` 變化點卡；`src/web/experiments.py` 實驗頁 | `logs/`、`output/exp3_trend/` | 本頁「即時展示畫面」 |
| 實驗四 | `src/experiments/exp4_polar_map.py` | `src/core/geometry.py`、`src/core/mahalanobis.py` | `src/tests/test_geometry.py` | `src/web/live.py`、`src/web/static/index.html` 極座標地圖；`src/web/experiments.py` 實驗頁 | `logs/`、`output/exp4_polar_map/` | 本頁「即時展示畫面」 |
| 實驗五 | `src/experiments/exp5_cross_condition.py` | `src/core/data.py`、`src/core/monitor.py` | 無專屬 | `src/web/static/index.html` 資料集下拉；`src/web/experiments.py` 實驗頁 | `logs/`、`output/exp5_cross_condition/` | 本頁「即時展示畫面」 |
| 實驗六 | `src/experiments/exp6_osr_benchmark.py`、`exp6_formal_benchmark.py`、`exp6_matrix.py`、`aggregate_exp6.py` | `src/core/detectors.py`、`src/core/openset.py`、`src/core/formal_data.py` | `src/tests/test_exp6_benchmark.py`、`test_exp6_matrix.py`、`test_aggregate_exp6.py`、`test_formal_data.py`、`test_openset.py` | `src/web/experiments.py` 實驗頁 | `logs/`、`output/exp6_osr_benchmark/`；`output/exp6_formal_matrix/` | [exp6_osr_benchmark.md](exp6_osr_benchmark.md) |
| 實驗七 | `src/experiments/compare_openset.py` | `src/core/monitor.py`、`src/core/openset.py`、`src/core/mahalanobis.py` | `src/tests/test_openset.py` | `src/web/experiments.py` 實驗頁 | 無（輸出目錄名是 `openset_comparison`） | [exp7_compare_openset.md](exp7_compare_openset.md) |
| 實驗八 | `src/experiments/health_index_benchmark.py`、`health_index_matrix.py`、`health_monitor.py`、`health_index_aggregate.py` | `src/experiments/health/` 全部模組、`src/core/openset.py` | `src/tests/test_health_*.py` | `src/web/experiments.py` 實驗頁 | `reports/exp8_health_index_results/` | [exp8_health.md](exp8_health.md) |
| 實驗九 | 無獨立入口（`src/core/detectors.py` 新增偵測器，透過 `src/experiments/exp6_osr_benchmark.py` 執行） | `src/core/detectors.py`（`MLPAutoencoderDet`） | `src/tests/test_exp9_autoencoder.py` | `src/web/experiments.py` 說明卡，執行在實驗六 | `logs/exp6_osr_benchmark/`、`output/exp6_osr_benchmark/`（與既有 exp6 輸出合併） | 無 |
| 實驗十 | `src/experiments/exp10_transfer.py` | `src/core/monitor.py`、`src/core/data.py` | `src/tests/test_exp10_transfer.py` | `src/web/experiments.py` 實驗頁 | `logs/exp10_transfer/`、`output/exp10_transfer/` | 無 |
| 實驗十一 | `src/experiments/exp11_ancestor_comparison.py` | `src/core/monitor.py`、`src/core/mahalanobis.py`、`src/core/data.py` | `src/tests/test_exp11_ancestor_comparison.py` | `src/web/experiments.py` 實驗頁 | `logs/exp11_ancestor_comparison/`、`output/exp11_ancestor_comparison/` | `docs/Mahalanobis_Improvement.md`（歷史對照） |
| 實驗十二 | `src/experiments/exp12_confusion_tsne.py` | `src/experiments/exp11_ancestor_comparison.py`（匯入 `build_ancestor_monitor`）、`src/core/monitor.py`、`src/core/data.py` | `src/tests/test_exp12_confusion_tsne.py` | `src/web/experiments.py` 實驗頁 | `logs/exp12_confusion_tsne/`、`output/exp12_confusion_tsne/` | 無 |
| 實驗十三 | `src/experiments/navigation_data_contract.py`、`navigation_regression.py` | 沿用 `src/core/monitor.py`、`src/core/openset.py`、`src/core/geometry.py`、exp2/3 與 `src/web/live.py` | `src/tests/test_navigation_guide.py`、`test_integration_navigation.py` | 獨立 `src/web/guide.py`、`guide.html/js/css` | `logs/navigation_*`、`output/navigation_*`、`logs/web_guide`、`output/web_guide` | `docs/navigation/exp13_*.md`；`reports/Andy_20261008_分支整合` |

Web 頁首可切換即時展示、實驗一～十二；實驗九只說明。實驗十三是獨立導覽。實驗頁的目錄與參數在 `src/web/experiments.py` 的 `CATALOG`，畫面在 `src/web/static/experiments.js`，每次執行的結果存到 `output/web_server/{ts}/experiments/`。不屬於任何編號實驗的檔案：`src/web/server.py` 與 `output/web_server/` 是展示本身；`logs/session_analysis/`、`output/session_analysis/`（2026-08-26）是一次 Web session 的事後分析圖，產生它的腳本不在 repo。

共同資料入口：`core.data` 掃描 `data/Step-*/myfeature/{Motor}/{RPM}/{Screws}/*_Group_feature_data_clean.csv`，讀105維特徵；健康類是 `8screws`。一般健康基準由`make_split`洗牌資料列後分train/calibration/holdout約60/20/20，正規化分數`> 1`判未知，seed預設42。多類已知配置、legacy門檻、持續學習確認與實驗四的幾何校準各有不同資料用途，請查對應手冊；不能用這段摘要替它們保證沒有洩漏。T1／T2／T3是不同馬達個體，raw錄製／視窗獨立性與單位仍須來源證據。
