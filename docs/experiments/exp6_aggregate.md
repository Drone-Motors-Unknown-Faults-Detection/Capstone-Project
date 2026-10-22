# 實驗六彙總：讀取矩陣摘要，不重新訓練

## 資料角色用量表

流程：讀manifest → 核對摘要 → 依工況／方法／seed分組 → 算平均、標準差及配對差 → 寫新彙總。這一步沒有學模型，也沒有取得新測試資料。

| 資料角色 | 範圍 | 實際／預期數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---|---|---|---|
| 原始特徵train／cal／test | 不載入 | 不適用 | 不適用 | 沿用原run的資料角色，見[正式版](exp6_formal_benchmark.md) | 不能由平均還原逐樣本 |
| 彙總輸入 | manifest指向摘要 | 預期54個run，實際以manifest為準 | 每method27個工況-seed列等權 | method總平均 | 同馬達多RPM與seed依賴 |
| 逐工況／方法 | 同dataset、method | 預期3seed | 每seed權重1/3 | 觀察切分變動 | 不新增獨立motor |
| 配對比較 | 同dataset、seed兩方法 | 預期27對 | 同鍵相減；不是列加總 | k-NN−LW | 共享test來源 |

本輪沒有新重算分數，原矩陣數量／缺件按下方來源規則判定；[本機資料用量](README.md#資料用量與分母怎麼讀)不能拿來替歷史54格補來源。

## 發表與理解檢查

口頭主軸：「這裡只回答保存的結果如何彙總。每格等權平均，會與按樣本數加權不同；標準差描述格子的變動，不是新馬達泛化的信賴保證。」`mean=Σx/n`；`std=sqrt(Σ(x−mean)²/(n−1))`，x是保存指標而非原始特徵，n是該組列數，單列依程式約定std=0。

- 為什麼同資料還要配對？同鍵兩法相減能控制工況／seed差異，但不能消除來源依賴。
- FPR差值為正是否更好？否，誤報越低越好；knn_better_pairs歷史欄名不能照字面讀。
- 平均吻合是否證明分類正確？沒有逐樣本核對與獨立性證據時不能這樣推論。
- 組員請先說平均分母與方向，再講分差，勿只報最高數字。

已完成的工況×seed×方法結果，如何放在一起比較？本工具只讀[矩陣](exp6_matrix.md)manifest指向的JSON，計算平均、標準差與配對差。沒有載入105維資料、fit模型、重設門檻或新增測試樣本。

## 輸入與驗證邊界

load_matrix先檢查矩陣層級，再對每格呼叫[矩陣手冊](exp6_matrix.md#完成判定規範)的`validate_run`：

1. manifest status=completed，runs長度等於expected_runs。
2. run_id不重複，且等於`{motor}_{rpm}_seed{seed}_{method}`；seed、method屬於manifest的seeds、methods。
3. (工況, seed, method)集合等於「manifest出現的工況 × seeds × methods」，缺格或多格都列出。
4. 每格的summary.json、results.csv、run.log符合完成判定規範，manifest記的dataset_fingerprint等於summary的值，八個指標為有限值。
5. 所有格的dataset_fingerprint相同。

任何一條不符就失敗，錯誤訊息逐格列出全部原因，不補零、不跳過。dataset_fingerprint本身的限制見[正式版](exp6_formal_benchmark.md#輸出怎麼讀)。

要看不完整矩陣時，明確加`--allow-partial`（API為`aggregate(manifest, allow_partial=True)`）：不符的格放進`rejected_runs`並附原因，其餘照算，status寫`PARTIAL`，預設寫到`aggregate_partial/`。PARTIAL結果不能當完整54格結果引用；不同fingerprint仍不混算。

## 怎麼算

八欄是known_accuracy、open_set_accuracy、auroc、aupr_unknown_positive、fpr_at_tpr95、unknown_precision、unknown_recall、unknown_f1，意義見[正式版](exp6_formal_benchmark.md#輸出怎麼讀)。known只指healthy，不是known fault type。

| 產物分組 | 權重與計算 |
|---|---|
| condition_summary | dataset×method，對保存的seed列等權平均 |
| macro_summary | method，所有工況×seed列等權，不按視窗數加權 |
| paired_differences | dataset×seed匹配，k-NN減Mahalanobis；兩方列數須匹配 |

mean用保存的已round6指標；std用ddof=1，單列std設0；輸出再round6。seed不是獨立馬達，這些std只描述保存列的變動。paired的knn_better_pairs欄實際計算「差值>0的筆數」；對FPR等越低越好的指標，名稱**不代表k-NN較好**，須反向解讀。其他正差也不是統計顯著性。

## 怎麼使用

在repo根目錄依[環境政策](../runtime_policy.md)準備uv，不需要data目錄，但需要完整結果包：

```bash
uv run --locked python -m experiments.aggregate_exp6 --help
uv run --locked python -m experiments.aggregate_exp6 --manifest output/exp6_formal_matrix/aero_2026-10-23/matrix_manifest.json --output-root output/exp6_formal_matrix/aero_2026-10-23/recompute_001
```

選未使用的recompute_001；預設會寫manifest旁的aggregate/（PARTIAL寫aggregate_partial/），**會覆寫同名檔**，不要對封存直接使用預設。API aggregate(manifest_path)只算dict；write_aggregate(manifest_path,output_root)寫檔；main提供CLI，沒有run(pools)、Web重算按鈕或續跑機制。Web唯讀讀原正式aggregate，新根不會自動部署到頁面。

## 輸出、判斷與限制

writer是[write_aggregate](../../src/experiments/aggregate_exp6.py)，寫五檔：aggregate.json、condition_summary.csv、macro_summary.csv、paired_differences.csv與aggregate.md。aggregate.md雖是Markdown，確由程式生成，不是手寫報告。main經`setup_run("exp6_aggregate", make_output=False)`寫`logs/exp6_aggregate/{ts}.log`，執行環境記在該log裡。

預期同來源重算數字一致，paired鍵對齊，缺件明確失敗；工況、方法、seed都依名稱排序，同一輸入的輸出逐位元組相同。它只能核對摘要算術，不證明逐樣本預測、錄製獨立、fresh test或最佳部署方法。未知正類比例不同時不直接比accuracy／AUPR；不同指紋／split／cal策略不可混為方法提升。現行彙總是 [output/exp6_formal_matrix/aero_2026-10-23/aggregate/](../../output/exp6_formal_matrix/aero_2026-10-23/aggregate/aggregate.md)（讀 `data`，指紋 `1b55b847…`）；`output/exp6_formal_matrix/aggregate/` 是 `data/formal_local`（指紋 `81c9…`）的歷史彙總。兩者指紋不同，不混算。

## 依據與程式碼

平均、ddof與配對方向是Lineage統計約定，沒有新模型；原演算法文獻見[正式benchmark](exp6_formal_benchmark.md#方法來源與程式碼)。[aggregate_exp6](../../src/experiments/aggregate_exp6.py)讀取、驗證與寫檔；[test_aggregate_exp6](../../src/tests/test_aggregate_exp6.py)提供不依賴data的fixtures；[web/experiments](../../src/web/experiments.py)只讀正式結果。程式元件測試不替代來源完整性查核。
