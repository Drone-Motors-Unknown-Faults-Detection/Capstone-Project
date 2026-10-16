# 實驗四：用極座標描述偏離方向

## 資料角色用量表

流程：已確認配置建模型 → 健康白化 → 配置中心形成射線 → 方向相似／半徑／排序分析。白化把健康常見的共同變動縮小；方向是偏向哪類特徵變化，半徑是離健康多遠，兩者不等同故障機制與損壞量。

| 資料角色 | T1／8000rpm範圍 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| a全類train／cal／holdout | 十配置共3049列 | 1825／605／619 | 每類各自60/20/剩餘後加總 | 模型與全類射線 | holdout另用作方向tau，非獨立方向測試 |
| c錨點健康、7、1的train／cal／holdout | 三池313+322+349=984 | 589／195／200 | 逐池取整，不直接乘984 | 健康白化及兩方向錨點 | fault holdout校準tau |
| c排序評估 | 七種均勻配置全池 | 2013 | 2013/2013 | 逐樣本Spearman | 包含錨點fit／cal列 |
| b各known場景 | 依KNOWN_SETS逐場景 | 隨known池變動 | [逐配置有效列](README.md#資料用量與分母怎麼讀)代入make_split | 其餘配置全池方向評估 | 場景重用資料，不能累加樣本 |

以上是載入列數與切分取整查核，沒有重新fit幾何或產生新cos分數。a沒有未知測試；b/c不是相同資料角色，不能合成一個準確率。無validation選參。

## 發表與理解檢查

口頭主軸：「在健康白化座標看資料往哪個方向偏移；先用已知配置中心建立射線，再看其他配置的方向與距離。三部分分別問結構、方向與指定排序，不能由一張漂亮圖證明可靠分類。」`cos=內積/(兩向量長度)`只量夾角；severity是相對錨點投影，無物理損傷單位。

- radius=2等於兩倍損壞嗎？沒有這種標定，只是特徵空間相對距離。
- holdout完全沒被用過嗎？方向tau用了fault holdout，排序圖也含已用錨點列。
- 換k-NN會改PolarMap嗎？核心幾何固定LW；本實驗CLI/Web本來就沒有k-NN選項。
- 組員請分開說明score、cos與severity；實際假說支持程度看下方a/b/c輸出，不把圖當驗收。

偵測器回答「是否超出已知範圍」，PolarMap另外回答「往哪個已知配置方向偏離」。本實驗檢查方向、幾何關係與配置排序；它不診斷真實故障原因，也沒有把分數標定成損壞百分比。

## 資料與建立順序

使用一個馬達／RPM的105維clean特徵。每列意義、非有限值移除、未去重、單位與raw視窗來源限制見[實驗一](exp1_cold_start.md#資料與載入)。T1／T2／T3是不同個體，不能串成生命週期。

[exp4](../../experiments/exp4_polar_map.py)依各部分的known清單建立OpenSetMonitor：healthy先列洗牌60/20/20，再逐一加入known配置並重新fit。seed預設42；RobustScaler只fit當次所有known train；所有known的cal用來校準LW距離。unknown不參與這些fit。顯示PCA不取代105維判定。

[PolarMap](../../core/geometry.py)另建LW馬氏幾何，使用已縮放的healthy train中心及共變異數，把其他known train中心轉成射線。每條射線的方向接受線tau，卻使用該known類別**holdout**餘弦的第5百分位。此holdout已參與方向校準，不能同時宣稱為未碰過的獨立方向測試。此工具沒有epoch、loss或optimizer。

## 三部分實際比較什麼

| 部分 | 建模／評估資料 | 回答的問題與限制 |
|---|---|---|
| a 幾何 | 所有fault配置加入known；比較類別中心射線 | 均勻鬆動與複合配置方向是否接近；沒有unknown測試 |
| b 方向 | known依序為7、1、7+1、5、3_14配置；其餘配置全池計算最大餘弦 | 其他均勻配置為正組、4_146為負組的方向AUROC；3_14另列，不是全部fault的開集AUROC |
| c 排序 | 7與1配置當方向錨點；7至1的均勻配置全池評估 | 距離或投影是否隨指定配置順序增加；包含錨點已用於fit／cal的樣本 |

表中7／1等指7screws／1screws，不是編碼成數值特徵。程式硬列UNIFORM與COMPOUND名單，未全面兼容1screw別名；缺配置可能KeyError，應核對資料名稱，不改名冒充來源。

c部分比較radius、best_ray、bundle三種數值。radius以嚴重錨點距離作除數；best_ray沿選中射線投影；bundle沿均勻配置平均方向投影。Spearman相關使用**逐樣本**分數與重複的配置rank（7至1列為0至6），不是只對七個median計算。rank是本專案排序約定，沒有實測磨損量。投影可為負或超過1。

新樣本先轉到健康白化空間，再取與已知射線最大餘弦；超過該射線tau回same_ray，否則new_direction。只有healthy時沒有射線，best_ray／severity可能None。這個verdict與Open Set score>1分開解讀，不能用方向相似取代異常偵測。

## 怎麼跑與怎麼用

在repo根目錄依[環境政策](../runtime_policy.md)準備uv與資料：

```bash
uv run --locked python -m experiments.exp4_polar_map --help
uv run --locked python -m experiments.exp4_polar_map --data-root data --motor T1 --rpm 8000rpm --seed 42 --part abc
```

--part可選a、b、c的組合。此CLI固定LW／confidence0.95，**沒有**openset-method、knn-neighbors或method選項；不要照其他實驗的參數直接貼入。API為run(pools,seed=42,parts="abc")，iter_run依part_start／part／result送事件。run回傳dict，不自行寫檔。

Web啟動見[實驗一](exp1_cold_start.md#實驗怎麼跑與怎麼使用)。選實驗四、資料集與seed，按「▶ 執行」看三部分；「⏵ 邊跑邊畫」按部分回傳，並非逐筆感測串流。「■ 停止」取消後續串流，未收到done不算完整；批次沒有中途停止入口。重跑會重新fit，沒有模型恢復功能。只跑b時不產生組合圖；圖需要a與c都存在。

## 輸出與判讀

main經setup_run寫logs/exp4_polar_map/{ts}.log與output/exp4_polar_map/{ts}/，含environment.json、summary.json；a+c齊全時另有polar_map.png。Web另寫web_server的experiments子目錄。geometry列中心／射線夾角關係，direction列餘弦分布與方向AUROC，severity列配置median與逐樣本Spearman。餘弦、正規化距離、rank相關都是無因次量。

預期均勻配置方向接近、複合配置較不同，且配置rank與severity正相關；程式沒有預登錄通用成功門檻。結果反向或不同工況不一致時，這些假說不受支持，不能只挑好看的投影。完整舊數字見[固定紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/experiments/exp4_polar_map.md#已記錄的實測)；本手冊不把它重列成新成績。

## 方法來源與程式碼

LW與馬氏距離來源見[實驗一](exp1_cold_start.md#方法來源與程式定位)。Spearman（1904），*The Proof and Measurement of Association between Two Things*，American Journal of Psychology15(1),72–101，[DOI](https://doi.org/10.2307/1412159)。健康白化、射線、tau與三種severity組合是Lineage操作方法，不宣稱上述論文提出整個PolarMap。

[exp4](../../experiments/exp4_polar_map.py)負責三部分與writer；[core/geometry](../../core/geometry.py)實作射線；[core/monitor](../../core/monitor.py)供known切分與scaler；[web/experiments](../../web/experiments.py)編排。驗證入口為[test_geometry](../../tests/test_geometry.py)、[test_stream_integration](../../tests/test_stream_integration.py)、[test_web_experiments](../../tests/test_web_experiments.py)。視覺幾何測試不能證明獨立馬達泛化或損壞標定。

串流停止、錯誤復原與模型有效狀態見[Web 串流生命週期](README.md#web-串流生命週期)及[共用模型生命週期](README.md#共用模型生命週期)。當次串流整合來源與配對證據見[固定交付紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/integration_20261008/integration_report.md)。
