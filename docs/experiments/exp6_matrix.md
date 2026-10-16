# 實驗六矩陣：保存多工況、seed與方法的執行狀態

## 資料角色用量表

流程：先列預期鍵 → 呼叫同一正式benchmark → 保存目標工況那列 → 保留失敗／續跑狀態。這是實驗六的排程器，不是新增模型。

| 資料角色 | 範圍 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 單工況train／cal／test | 例T1／8000rpm健康 | 187／62／64 | 各除以313 | 沿用正式benchmark | 每方法、seed重用 |
| 同工況unknown | 其餘配置 | 2736 | 全故障池100% | 沿用未知拒絕 | 不fit／cal |
| 預期矩陣 | 9工況×3seed×2方法 | 54個run鍵 | 是執行格，非資料列 | 排程coverage | 每run還會重算其他工況 |
| 本機可用來源 | [六工況](README.md#資料用量與分母怎麼讀) | 19053列 | 不是九工況資料 | 唯讀前提查核 | 缺T2，不足此正式矩陣 |

沒有validation選參；不把54格乘上列數當獨立樣本。本輪沒有重跑矩陣，完整正式資料不足仍須保留缺件，不能用allow-partial借道湊PASS。

## 發表與理解檢查

口頭主軸：「先固定工況、方法與seed清單，把每格的來源和狀態保存，讓失敗可追查。54格全部完成只代表工程coverage；是否好用要看未知召回、健康誤報與來源獨立性。」

- 六個方法／seed執行是否等於六顆馬達？不是，資料重用。
- resume是否驗完所有來源？沒有，現行快取防護有限，見下方逐項契約。
- 六工況能否關完整九工況驗收？不能，必須列缺T2三轉速。
- 組員請分開講run鍵、資料列與獨立採集；歷史分數只查固定輸出，不把排程完成當研究通過。

這個工具把[正式benchmark](exp6_formal_benchmark.md)放進可續跑矩陣，保留完成與失敗格；不增加偵測方法，也不做共同模型選擇。後續平均由[彙總工具](exp6_aggregate.md)處理。

## 迴圈、資料與fit

expected_run_matrix要求掃描九工況，預設seed為42／123／2026、方法為mahalanobis／knn，產生9×3×2＝54個run_id，格式T1_8000rpm_seed42_knn。這是54個工況×seed×方法記錄，不是54顆獨立馬達。資料列105維、healthy切60/20/20、train-only scaler與參考、known-only cal與unknown全池評估，全部沿用正式benchmark；原始時間／session／單位未知，不能靠seed建立獨立採集。

實作有一個成本差異：**每個run_id都呼叫一次涵蓋全部九工況的benchmark，再只取目標工況那一列保存**。因此有重複計算，54列不是只fit54次的承諾。參數固定benchmark預設LW、confidence0.95、k=5；matrix CLI沒有改這三者的選項，不從fault結果調參。

## 操作與續跑風險

在repo根目錄準備[uv環境](../runtime_policy.md)與完整資料，先選一個尚未使用的輸出根：

```bash
uv run --locked python -m experiments.exp6_matrix --help
uv run --locked python -m experiments.exp6_matrix --data-root data/formal_local --output-root output/exp6_formal_matrix/manual_run_001 --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn
```

manual_run_001只是範例，已存在就換新名稱，不能覆寫封存。API實際為run_matrix(data_root,output_root,seeds=(42,123,2026),methods=("mahalanobis","knn"),resume=True)，沒有獨立run(pools)介面或Web重跑按鈕。網頁只能讀既有結果。

同命令、同根再跑預設resume：_is_complete查summary狀態、方法、seed、單列motor／rpm及資料fingerprint，通過就跳過。**不驗證results.csv、run.log、模型、全部指標或CSV內容SHA**，也沒有完整參數lock；不能把resumed視為整包證據已驗證。--no-resume會重算並覆寫同run目錄，不用在封存上。缺九工況或非法方法在開跑前拒絕；個別run失敗保存error並繼續，最後退出碼2，不把失敗靜默刪除。

## 輸出怎麼看

此模組自己的_atomic_json／_write_run_csv寫output-root，**尚未呼叫setup_run**，也沒有標準logs/exp6_matrix/{ts}.log；這是與AGENT慣例的缺口，不補寫不存在的日誌。

- matrix_manifest.json：預期清單、每格狀態、耗時、摘要相對路徑；開始時會重寫manifest。
- runs/{run_id}/summary.json：正式benchmark設定、fingerprint、單目標工況列。
- runs/{run_id}/results.csv：相同單列的表格。
- runs/{run_id}/run.log：開始、狀態與錯誤；duration_seconds含整次九工況運算，不等於該列單獨推論耗時。

completed只表示本工具54格完成，unknown召回／健康接受等解釋見[正式版](exp6_formal_benchmark.md#輸出怎麼讀)。三seed的切分敏感度不能當motor母體信賴區間。預期54完成、0失敗且各列可追溯；來源／參數完整防護仍不足時，不宣稱研究可靠性通過。完整舊包見[固定矩陣](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/1fa9431bb7b86959f29540d07b2b9290ab39ce42/output/exp6_formal_matrix)。

## 方法來源與程式碼

統計方法文獻見[正式benchmark](exp6_formal_benchmark.md#方法來源與程式碼)，run_id、54格與resume為Lineage操作約定。[exp6_matrix](../../src/experiments/exp6_matrix.py)負責排程與writer；[benchmark](../../src/experiments/exp6_formal_benchmark.py)負責fit／校準／指標；[test_exp6_matrix](../../src/tests/test_exp6_matrix.py)檢查矩陣與續跑。[web/experiments](../../src/web/experiments.py)的唯讀結果入口不會啟動此工具。
