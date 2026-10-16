# 實驗五：健康基準換到另一工況會怎樣

## 資料角色用量表

流程：各source建健康基準 → 套各target → 分開看同馬達換RPM、同RPM換馬達 → 檢查策略及個體分布。它揭露基準能搬多遠；不同個體與設備條件混雜，不能據此量化老化。

| 資料角色 | 範圍 | 有效數量示例 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| source train／cal | T1／8000rpm健康 | 187／62 | 各除以313 | scaler／LW／門檻 | 同source holdout互斥 |
| 自身健康test | T1／8000rpm | 64 | 64/313 | 對角線接受率 | 列級互斥 |
| 跨工況健康／fault test | T3／8000rpm | 272／2953 | 目標健康／fault各全池100% | 非對角線評估 | 同target會被多source重用 |
| 策略b的outer test | T1／8000rpm，seed42 | 64 | 64/313 | 三策略共用評估 | per_condition其中42列進train、9列進cal |

最後一列是[唯讀索引重播](../../output/experiment_data_usage_evidence/2026-10-14-19-36-34/usage.json)的可確認用途重疊，沒有改split或重跑分數。不同seed的數字另存`exp5_strategy_overlap`。本機六工況只能形成6×6=36格矩陣，不是完整9×9；配置與列數見[總表](README.md#資料用量與分母怎麼讀)。

## 發表與理解檢查

口頭主軸：「先把source的健康範圍搬去target，觀察健康會不會被誤報、故障排序是否變差。逐格呈現才能看出哪種轉移失敗。策略b有共同測試列進per_condition建模的缺口，因此不能據它選部署方案。」沒有超參數選擇或新的final test。

- 9工況等於9顆馬達嗎？不是，三馬達各三轉速；本機只有兩馬達各三轉速。
- healthy_accept與fault_detect哪個是準確率？兩個分母不同，要分開報，均非配置分類率。
- T3分數較高是否證明老化？不能排除馬達個體、安裝等差異。
- 組員請指出source與target，並說出策略b目前為何不能公平比較；實際歷史分數仍連下方固定產物。

同一工況建立的健康範圍，換馬達或轉速後可能把正常差異當故障。這個實驗比較跨工況分數與三種基準策略，也檢查T1與T2／T3的差異；它沒有同一顆馬達的長期退化真值。

## 資料與共同處理

[runner](../../experiments/exp5_cross_condition.py)掃描data-root下的Motor／RPM，通常預期T1／T2／T3×6000／8000／11000rpm，但程式**沒有強制九工況齊全**。實際coverage看輸出keys，不拿「9組」補缺件。每個配置池是105維clean特徵，資料列、loader與未知單位／raw視窗限制見[實驗一](exp1_cold_start.md#資料與載入)；沒有去重或重建錄製順序。

每次模型只用healthy建立OpenSetMonitor：列洗牌60/20/20，RobustScaler fit健康train，LW估計中心／共變異數，cal的confidence0.95分位數定線，score>1判異常。CLI可改confidence或馬氏method；legacy改用train距離定線，不能混稱校準政策相同。此實驗沒有k-NN選項，沒有神經網路訓練。

## 三部分的資料用途

a：每個source工況健康池fit一個模型，逐一套用所有target工況。同source健康用自己的holdout，跨source健康用target**完整健康池**；fault用target全部fault配置。每格計算healthy_accept、fault_detect與AUROC，分同工況、跨工況、同RPM跨motor、同motor跨RPM作等格權重平均。跨motor是個體轉移，不是同個體時間前後。

b：用seed+7先對每個工況建立外層健康split，取outer holdout作共同評估池，比較per_condition、global與per_rpm。後兩者把對應工況的outer train+cal混合，再內切60/20/20 fit。**per_condition卻對完整健康池用seed重新內切**，並未排除outer holdout；共同評估列可能進入它的train或cal。因此此部分不能宣稱三策略都是相同無洩漏holdout。這是現行程式缺口，本手冊揭露但不修改計算。

c：每個RPM以T1健康池fit，對T1自身holdout、T2／T3完整健康池算分數；缺T1或對照motor會略過該列。文件支持三者為不同馬達個體，T1描述新、T2／T3描述老，但使用時數、安裝與負載等未確認。個體與老化混雜，不能把跨個體分數差歸因為磨損量。程式及UI的aging字樣是歷史命名，不是已驗證生命週期分析。

三部分unknown都只在評估可見，不fit健康模型；既有特徵清理是否跨raw視窗使用資訊仍未確認。a、b、c也沒有未曝光final test主張。

## 怎麼執行

從repo根目錄依[環境政策](../runtime_policy.md)準備uv與正式特徵：

```bash
uv run --locked python -m experiments.exp5_cross_condition --help
uv run --locked python -m experiments.exp5_cross_condition --data-root data --seed 42 --part abc --confidence 0.95 --method ledoit_wolf
```

--part選abc中的部分；API為run(data_root="data",seed=42,parts="abc",confidence=0.95,method="ledoit_wolf")。它載入整個root，不接受單一pools、motor或rpm，也沒有openset-method。資料空、缺healthy或fault可能在堆疊／彙總失敗，應確認coverage與105維，不借其他馬達假補。

Web啟動見[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)，實驗五選seed後按「▶ 執行」，處理伺服器的整個data-root、跑三部分，不以頁面單一資料集下拉選單限制範圍。此卡沒有邊跑邊畫／中途停止與續跑入口；失敗後修正前提，再另跑一次。這個操作不變更現場模型或資料檔。

## 看輸出與判斷限制

main寫logs/exp5_cross_condition/{ts}.log、output/exp5_cross_condition/{ts}/environment.json與summary.json；a+c齊全才寫cross_condition.png。Web寫web_server的experiments子目錄。cross_condition中的train／test是模型來源／評估工況，不代表三顆馬達分成專用train／cal／test的另一套研究協定。

healthy_accept是健康score≤1比例，fault_detect是fault score>1比例，AUROC是故障分數高於健康的排序指標。不是九種故障配置分類準確率。各格先round4，再等格平均，不按視窗數加權；少工況或缺故障改變平均對象。健康score、p95等為無因次正規化量，不是故障嚴重度或時數。

預期跨工況健康接受率可能下降，分工況基準是否改善需配合coverage與公平holdout判讀。程式沒有正式成功門檻。b目前用途重疊使策略比較受限，不能據它推薦部署；c只描述現有不同馬達分布。舊完整數字見[固定紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/experiments/exp5_cross_condition.md#已記錄的實測)。

## 方法來源與程式碼

馬氏距離、LW、RobustScaler來源見[實驗一](exp1_cold_start.md#方法來源與程式定位)。三種混合策略、配對工況與統計分組是Lineage操作約定，不是由某篇論文保證的域適應方法。

[exp5](../../experiments/exp5_cross_condition.py)負責載入、三部分、彙總與writer；[core/data](../../core/data.py)切分；[core/monitor](../../core/monitor.py)建健康基準；[web/experiments](../../web/experiments.py)批次編排；[test_web_experiments](../../tests/test_web_experiments.py)與[test_monitor_guard](../../tests/test_monitor_guard.py)檢查入口／基準。這些測試不消除b的外層／內層holdout重疊。
