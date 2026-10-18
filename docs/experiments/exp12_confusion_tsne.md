# 實驗十二：把錯分位置與特徵分布畫出來

## 資料角色用量表

流程：按既有基準產生分類 → 真值×預測計數 → 另抽特徵做t-SNE → 診斷錯分。混淆矩陣回答哪類錯，t-SNE回答這批點在低維如何鄰近，兩者都不是新的部署分類器。

| 資料角色 | T1／8000rpm示例 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 二元基準train／cal | 健康池313 | 187／62 | 各除以313 | LW及門檻 | 不含二元test |
| 二元矩陣test | 健康holdout＋fault全池 | 64+2736=2800 | 每真值行另算recall | 健康／故障計數 | 工況累加為樣本加權 |
| 多類矩陣test | A協定known holdout＋unknown | 301+1571=1872 | 沿[實驗十一](exp11_ancestor_comparison.md#資料角色用量表) | 六標籤計數 | legacy介面限制保留 |
| t-SNE作圖 | 十配置各最多200 | 2000 | min(每池n,200)，不是模型test比例 | 事後低維嵌入 | 從完整池抽，包含known fit/cal列 |

依[唯讀用量](README.md#資料用量與分母怎麼讀)，未重跑t-SNE或分類。unknown只進作圖，不回饋detector；但圖含fit資料，不能稱獨立holdout圖。多seed／多張圖仍重用來源。

## 發表與理解檢查

口頭主軸：「先看真值行有多少被分到哪個預測欄，找過度拒絕與漏報；再用t-SNE輔助觀察特徵重疊。低維圖可扭曲群距，各工況座標也不能直接對照，因此不能只靠漂亮分群宣布可靠。」t-SNE使高、低維鄰近分布較接近，原論文與程式設定見下方。

- 圖上的三角是否全部錯？不是，未知被正確拒絕也是三角。
- 行總數不同怎麼比較？每類recall除自己真值行總數；空行不適用，不填100%。
- 為何不拿群距當嚴重度？嵌入無物理標定，也未保證全域距離。
- 組員請讀一格誤報、一格漏報，並說清t-SNE包含fit資料的限制；真實矩陣仍查保存JSON。

總準確率看不出哪類被拒絕。這個實驗畫二元及開集多類混淆矩陣，再用t-SNE看特徵鄰近關係；圖片用來診斷，不訓練新的故障分類器或調整門檻。

## 三部分如何用資料

使用各工況105維clean特徵，來源／每列／單位／未去重與raw限制見[實驗一](exp1_cold_start.md#資料與載入)。沒有強制九工況；T1／T2／T3是不同馬達，不能把全域表當同顆馬達時間序列。

a：每工況healthy列按seed42切60/20/20，train fit RobustScaler與LW、cal定confidence0.95線；healthy holdout與fault全池score，≤1叫healthy、>1叫fault。各工況結果累加成2×2**原始筆數**矩陣，樣本多的工況權重較大。

b：重用[實驗十一](exp11_ancestor_comparison.md)build_ancestor_monitor，known為8／1／2／3／4，unknown為5／6／7／3_14／4_146交集，分別LW與legacy。known各holdout、unknown全池classify後合併unknown，1screw／1screws顯示名統一成1screw，形成固定六標籤矩陣。缺類不代表完整coverage；legacy label=-1使classify無法返回known，不能將全部拒絕解讀成單純協方差差異。unknown未進detector fit／cal。

c：每工況另建LW的A協定monitor，按default_rng(seed)每配置抽最多200列（不放回），**從完整池抽，包含known train／cal／holdout**。用該monitor的train-only scaler transform，再讓t-SNE在抽出的known＋unknown樣本上fit二維嵌入。unknown只參與事後視覺化、不回饋偵測器；這張圖不是獨立holdout準確率。

## t-SNE怎麼建立

TSNE(random_state=seed,init="pca")，perplexity預設30，effective_perplexity=min(要求值,max(5,抽樣總數//4))；其他參數沿uv.lock固定sklearn。它最佳化高／低維鄰近分布的KL差異，沒有本專案另設神經網路epoch／loss或用test挑圖。資料極少時這個cap仍可能不小於總樣本而失敗；不補造點。

顏色表示配置，圓點known_correct、叉known_wrong、三角unknown。unknown配置若被正確拒絕也畫三角，**三角不是錯誤標記**。各子圖各自fit，座標、群距與方向不能跨圖比較，沒有物理單位或損傷意義。PolarMap用馬氏白化幾何，不是把t-SNE或PCA座標直接當嚴重度量尺。

## 怎麼操作

repo根目錄依[uv環境政策](../runtime_policy.md)準備資料：

```bash
uv run --locked python -m experiments.exp12_confusion_tsne --help
uv run --locked python -m experiments.exp12_confusion_tsne --data-root data --seed 42 --confidence 0.95 --perplexity 30 --tsne-max-per-class 200
```

API run(data_root,seed=42,confidence=0.95,perplexity=30,tsne_max_per_class=200)回傳矩陣及內部_tsne_by_dataset。max_per_class只降低圖的成本，不改混淆矩陣樣本。Web「實驗十二」選 seed 按「▶ 執行」，畫出混淆矩陣與各工況 t-SNE；沒有逐筆串流、取消或續跑，重新執行會重fit。缺配置／105欄／資料太少先核對來源，不能調出最好看的seed作研究成績。

## 看輸出

main經setup_run寫logs/exp12_confusion_tsne/{ts}.log、output/exp12_confusion_tsne/{ts}/environment.json、summary.json、confusion_matrices.png及tsne_grid.png。summary含labels／matrix原始數字；t-SNE座標**不落地**，只留圖。網格固定3×3，超過九工況時圖片只呈前九格，不能以圖確認完整coverage。

每一橫列（row）是真值，每一直欄（column）是預測。healthy→fault是健康誤報；fault→healthy是漏報；known→unknown是誤拒；unknown→known是未拒絕；unknown→unknown是正確拒絕，但沒有辨識具體未知配置身分。全域筆數不能當各motor等權平均。

### 計數核算與 AI 摘要契約

以原始整數計數矩陣 `M`、明確的任務和標籤語意核算，按標籤找 healthy／fault／unknown，不假定 unknown 在最後。橫列和 `sum(M[i, :])` 是該真實類的支持數，對角 `M[i, i]` 是正確數；總量是所有格子的和。

| 指標 | 分子 | 分母 |
| --- | --- | --- |
| 二元健康誤報率 | healthy→fault | healthy 真值支持數 |
| 二元故障召回率 | fault→fault | fault 真值支持數 |
| 已知判對率 | 所有 known 對角和 | known 真值支持總數 |
| 已知誤拒率 | 所有 known→unknown | known 真值支持總數 |
| 已知互相錯分數 | 所有 known→其他 known 的非對角和 | 計數，不用近似百分比代替 |
| 未知正確拒絕率 | unknown→unknown | unknown 真值支持數 |
| 未知未拒數 | unknown→任何 known | 計數 |

必須滿足 `known_total = known_correct + known_rejected + known_misclassified`。已知判對率的分母包含被拒樣本，不能換成已接受量；二元與多類屬不同流程，不能互換支持數。分母為零時回傳 null／不適用，不填0%或100%；缺unknown標籤時未知指標為不適用。矩陣須為與唯一labels等長的方陣，值必須為非負整數，布林值、浮點／normalized比例與不合法shape明確拒絕，不能將比例冒充樣本數。僅有整數陣列仍無法證明來源確實是計數，呼叫端須明示 `counts` 語意。

統計純函式放`src/core/confusion_summary.py`；Web只呼叫並呈現。舊保存JSON可唯讀推導，不回寫原檔，也不採信保存的衍生摘要取代原始矩陣。摘要優先保留實驗／存檔來源、設定、分子分母與限制；預算不足則整段標示不完整，不截出缺分母的百分比。`get_run_status`及`get_run_events`提供相同的事後核算，來源文件僅說明方法，不是模型特徵資料來源。AI可以解釋程式核算，不能沿用對話中的錯答或其他實驗的誤報常數。

聊天視窗另顯示「程式核算」純文字區塊：SSE的`facts`來自工具artifact，與模型`delta`分開，也不當成模型回答寫回對話歷史。存檔來源顯示實際讀取路徑、原檔SHA與設定；檔名／實驗ID或參數／結果設定不一致時，不提供可信數值。這保障計數和呈現來源，不保證LLM每句解釋都正確；mock與Node DOM測試也不等於真實模型或瀏覽器驗收。

legacy介面限制必須另核對來源版本：`fit()`固定label=-1、`classify()`將負標籤轉None，再由實驗記unknown。只憑矩陣全部unknown不足以確認該原因；缺版本證據時標為未核實。任何版本都不能把unknown對角的正確拒絕說成零判對。本核算不改detector、校準或t-SNE，也不能只由誤拒率斷言邊界過嚴、用同批test調線、或宣稱跨馬達泛化。

預期圖能揭露錯誤來源，低分／重疊也照留；沒有「圖上分群漂亮」成功門檻。t-SNE群聚不證明高維泛化或排序嚴重度，不能用它支持部署、fresh final或RUL。圖與數字在 [exp12 輸出](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/1fa9431bb7b86959f29540d07b2b9290ab39ce42/output/exp12_confusion_tsne/2026-10-09-14-27-55)。

## 方法來源與程式碼

van der Maaten與Hinton（2008），*Visualizing Data using t-SNE*，JMLR9,2579–2605，[原文](https://jmlr.org/papers/v9/vandermaaten08a.html)。perplexity30與配置標記是Lineage設定，不是原文對此資料的保證。混淆矩陣為計數工具；LW／legacy來源見[實驗十一](exp11_ancestor_comparison.md#來源與程式碼)。

[exp12](../../src/experiments/exp12_confusion_tsne.py)負責取樣／矩陣／嵌入／writer；[exp11](../../src/experiments/exp11_ancestor_comparison.py)供known協定；[core/monitor](../../src/core/monitor.py)fit與classify；[test_exp12_confusion_tsne](../../src/tests/test_exp12_confusion_tsne.py)檢查形狀／行和／有限嵌入，不能證明視覺推論正確。

計數核算延伸沿用本實驗編號；新增的core純函式及其`src/tests/test_confusion_summary.py`測試不另占新實驗。資料鏈使用`src/web/playback.py`、`src/web/assistant.py`、`src/web/static/ai_chat.js`，對應測試為`src/tests/test_playback.py`、`src/tests/test_assistant.py`、`src/tests/test_ai_chat_frontend.py`及其`ai_chat_frontend.cjs`；不新增人工output報告，驗證證據依`src/tests/pytest_evidence.py`的writer保存。
