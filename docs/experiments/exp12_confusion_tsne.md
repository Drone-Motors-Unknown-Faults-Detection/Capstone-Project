# 實驗十二：把錯分位置與特徵分布畫出來

## 資料角色用量表

流程：按既有基準產生分類 → 真值×預測計數 → 另抽特徵做t-SNE → 診斷錯分。混淆矩陣回答哪類錯，t-SNE回答這批點在低維如何鄰近，兩者都不是新的部署分類器。

| 資料角色 | T1／8000rpm示例 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 二元基準train／cal | 健康池313 | 187／62 | 各除以313 | LW及門檻 | 不含二元test |
| 二元矩陣test | 健康holdout＋fault全池 | 64+2736=2800 | 每真值行另算recall | 健康／故障計數 | 工況累加為樣本加權 |
| 多類矩陣test | A協定known holdout＋unknown | 301+1571=1872 | 沿[實驗十一](exp11_ancestor_comparison.md#資料角色用量表) | 六標籤計數 | legacy介面限制保留 |
| t-SNE作圖 | 十配置各最多200 | 2000 | min(每池n,200)，不是模型test比例 | 事後低維嵌入 | 從完整池抽，包含known fit/cal列 |

依[唯讀用量](README.md#資料用量與分母怎麼讀)，未重跑t-SNE或分類。unknown參與開集測試與事後作圖，不參與detector擬合／校準；圖含fit資料，不能稱獨立holdout圖。多seed／多張圖仍重用來源。

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

顏色表示真實配置，形狀表示模型輸出：已知且判對為圓點known_correct；pred為None為三角unknown，包含已知誤拒及未知正確拒絕；其餘錯誤接受為叉known_wrong，包含未知被判成已知。**三角表示模型拒為unknown，不限定真實未知配置，也不是一律判錯。** 每工況每配置最多抽200筆，十配置完整時每工況最多2000筆；各池不足時取min(n,200)，不把上限當當次實際數量。各子圖各自fit，座標、群距與方向不能跨圖比較，沒有物理單位或損傷意義。PolarMap用馬氏白化幾何，不是把t-SNE或PCA座標直接當嚴重度量尺。

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

一般模型問答若查到矩陣工具，聊天視窗可另顯示工具artifact的`facts`純文字區塊，不當成模型回答寫回對話歷史。實驗十二結果模式則以下節的程式正文為準，不重複顯示facts。存檔來源顯示實際讀取路徑、原檔SHA與設定；檔名／實驗ID或參數／結果設定不一致時，不提供可信數值。一般模型問答的自由解釋仍須核對；mock與Node DOM測試不等於真實模型或瀏覽器驗收。

### 結果摘要與追問

實驗十二的結果摘要、錯誤前提追問與同一紀錄的延續問題，主要答案直接由已驗證矩陣產生，標示「結果摘要由程式依矩陣產生」。此模式不呼叫LLM，不先串出自由評論再補正確框；主要答案與來源一次固定，不另顯示重複的facts正文。一般原理、操作與方法問答仍走既有模型工具流程。

逐已知類列出支持數、判對、誤拒、互錯與判對率；`8screws`明示為健康。fault→healthy是故障漏判，unknown對角是未知正確拒絕，不能推成健康沒有誤報。legacy只描述實際計數與來源版本可支持的限制，不推論原方法失效；不猜逐類範圍、分類優劣或另一套CLI輸出目錄。

`POST /api/assistant/ask`保留`question`與`history`，可另帶`context`（experiment、mode與saved）。明確指定存檔必須讀該檔，無法定位就說明不足，不能改拿最新一筆。保存根目錄外、路徑穿越、symlink逃逸與同名多筆未消歧均拒絕；舊客戶端仍可用實驗編號和問句辨識結果模式。明確詢問新結果時不沿用上一筆來源。

結果模式以`mode` frame聲明非生成來源，再用既有`delta`傳送程式產生的完整正文，維持舊客戶端可讀。前端將其標為程式摘要，不假稱模型回覆或在模型離線時顯示虛假失敗；後續只傳遞結構化來源context，程式正文不偽裝成LLM歷史。缺矩陣、來源不一致、未完成或預算不足時，回覆明確不足，不初始化模型或從舊錯答補數字。舊payload只有二元或多類矩陣時，可呈現已有核算，但必須點出缺少的部分並將摘要狀態標為`incomplete`，不能當完整實驗十二結果。

普通CI以已知歷史矩陣及使用者錯答片段做回歸，故意讓模型產生錯誤多段delta，驗證它們在結果模式沒有被呼叫或顯示。這些fixture不是新網站原檔或新研究成績；真實模型、瀏覽器與公開部署版本須另行驗收，CI通過不等於公開站已更新。

批次分類由模型產生預測，再用事後真值計算指標；「事後評估」不表示沒有模型判斷。已知判對率包含8screws健康類。資料池總量、train、cal、holdout分開列；T1／8000rpm健康池313的示例是187／62／64，示例不當成其他工況用量。角色數量重用core.data.make_split核算，不以模型心算。當次t-SNE數量從保存的各工況config_labels／verdicts／embedding長度核對；缺少時寫未知，不重跑或用2000補值。

一般模型工具問答每請求最多3輪工具、6次工具要求、4次模型呼叫；相同工具與正規化參數限本請求快取。工具文字預算16000字，每次模型輸出最多1600 tokens，不自動重試。重複查詢、工具預算或接近時間上限時，停止工具並保留一次禁工具整理；整理仍要求工具或逾時時，只呈現已取得資料與未回答範圍。總時間預算90秒，每次模型最多25秒，保留25秒給最後整理。這些是應用層約定，不修改共享模型配置，也不只提高LangGraph步數上限。工具超限、模型逾時與連線錯誤分開提示，斷線會取消本輪工作。一般模型答案在該輪完整回覆且確認沒有工具要求後才送正文，工具前言不先串出。

部署核對可讀`/api/assistant/status`的`app.git_head`、`tracked_dirty`與`answer_schema`；只提供版本識別，不公開環境變數、套件清冊或私人路徑。映像沒有Git資訊時版本為null，不能拿schema或某一支JS含有handler就宣稱整套部署已核實。HTML的script版本參數與內容雜湊仍需另比對。

legacy介面限制必須另核對來源版本：`fit()`固定label=-1、`classify()`將負標籤轉None，再由實驗記unknown。只憑矩陣全部unknown不足以確認該原因；缺版本證據時標為未核實。任何版本都不能把unknown對角的正確拒絕說成零判對。本核算不改detector、校準或t-SNE，也不能只由誤拒率斷言邊界過嚴、用同批test調線、或宣稱跨馬達泛化。

預期圖能揭露錯誤來源，低分／重疊也照留；沒有「圖上分群漂亮」成功門檻。t-SNE群聚不證明高維泛化或排序嚴重度，不能用它支持部署、fresh final或RUL。圖與數字在 [exp12 輸出](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/1fa9431bb7b86959f29540d07b2b9290ab39ce42/output/exp12_confusion_tsne/2026-10-09-14-27-55)。

## 方法來源與程式碼

van der Maaten與Hinton（2008），*Visualizing Data using t-SNE*，JMLR9,2579–2605，[原文](https://jmlr.org/papers/v9/vandermaaten08a.html)。perplexity30與配置標記是Lineage設定，不是原文對此資料的保證。混淆矩陣為計數工具；LW／legacy來源見[實驗十一](exp11_ancestor_comparison.md#來源與程式碼)。

[exp12](../../src/experiments/exp12_confusion_tsne.py)負責取樣／矩陣／嵌入／writer；[exp11](../../src/experiments/exp11_ancestor_comparison.py)供known協定；[core/monitor](../../src/core/monitor.py)fit與classify；[test_exp12_confusion_tsne](../../src/tests/test_exp12_confusion_tsne.py)檢查形狀／行和／有限嵌入，不能證明視覺推論正確。

計數核算延伸沿用本實驗編號；新增的core純函式及其`src/tests/test_confusion_summary.py`測試不另占新實驗。資料鏈使用`src/web/playback.py`、`src/web/assistant.py`、`src/web/static/ai_chat.js`，對應測試為`src/tests/test_playback.py`、`src/tests/test_assistant.py`、`src/tests/test_ai_chat_frontend.py`及其`ai_chat_frontend.cjs`；不新增人工output報告，驗證證據依`src/tests/pytest_evidence.py`的writer保存。

抽樣與資料角色核算另見`src/tests/test_exp12_semantics.py`；有限工具編排在`src/web/assistant_budget.py`，對應`src/tests/test_assistant_budget.py`。圖步數錯誤說明見[LangGraph官方文件](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)；本專案預算為應用層設計，不是官方對模型語意正確的保證。
