# 實驗八：唯讀重算封存健康指數摘要

## 資料角色用量表

流程：先驗來源SHA／唯一鍵 → 讀固定六run → 按事前定義重算 → 與封存逐欄比 → 差異原樣保留。沒有任何train、calibration、validation或test的新使用。

| 資料角色 | 範圍 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 唯讀來源 | 固定manifest、JSON/CSV及舊彙總 | 16檔 | 檔案數，不是樣本數 | SHA及schema契約 | 不修改來源 |
| 摘要輸入 | 2法×3seed×9工況 | 54列 | 每方法27列等權 | 全域mean/std | 同motor/RPM/seed資料依賴 |
| 逐工況／方法 | 18組 | 每組3列 | seed各1/3 | 條件摘要 | 不能當三個新session |
| 原始模型資料 | 不載入 | 不適用 | 原數量讀來源JSON | 不重新fit或predict | 無逐窗ID，獨立性UNKNOWN |

此封存來源與[本機六工況用量](README.md#資料用量與分母怎麼讀)版本不同，不能混換SHA或用本機缺件填它。原彙總的ddof／round順序UNKNOWN，兩欄差異已有固定證據，不能移除失敗以求MATCH。

## 發表與理解檢查

口頭主軸：「把舊結果算回來，確認分母、權重與四捨五入。完成重算不等於全部相同；DIFFERENT是需要解釋的差異，不是模型突然變差，也不是要修改封存。」平均與ddof1公式見[實驗六彙總](exp6_aggregate.md#發表與理解檢查)，容差依本工具固定契約。

- 為什麼不按樣本數加權？契約採工況-seed列等權；換權重會回答不同問題。
- stdout退出3算失敗嗎？表示算完但與歷史不同；必須保留差異，不能改為0掩蓋。
- SHA相同是否證明資料採集獨立？只證明來源位元組相符，不提供採集事實。
- 組員請區分「算完」「數字吻合」「模型可靠」三種判斷。

想確認舊平均是否算得回來，用這個工具讀封存輸出，不重新訓練模型。它比較固定六run、54列的七個指標，保存新舊值與差異；不把算術吻合當成新資料或模型可靠性證明。

## 來源契約與讀取

唯讀來源為reports/exp8_health_index_results/的manifest、六JSON、六CSV、舊彙總與README，共16檔。[契約](exp8_health_index_aggregate_contract.json)列Windows CRLF及固定Git LF兩組**完整位元組SHA256**，只接受已核實hash，不一般化正規化空白或改來源。fingerprint為81c9192476ecd1e23a17b3ed5a343dcd50cdc2e5b7e0cc5466e9162941898228，不能換成後續fault-type資料指紋。

先核對檔案集合／SHA，再核schema1、seed42／123／2026、mahalanobis-LW與knn-k5、confidence0.95、run與列唯一鍵、T1／T2／T3×三RPM、每run九列、樣本數一致及CSV與JSON對應。缺件、額外來源、重複鍵、非有限值、欄位缺失或錯指紋一律失敗，沒有PARTIAL模式。正規矩陣樣本數須為正整數，不用零分母假造結果。

原模型的105維、健康train／cal／holdout與unknown全池含義見[benchmark](exp8_health_index_benchmark.md)。本工具**不載入正式資料、不fit scaler／模型、不校準、不predict**。JSON只存unknown配置數，沒有配置名稱、逐窗ID或raw時間，不能還原來源獨立性。三顆motor是不同個體，不是連續退化。

## 固定的算術規則

七欄：health_gap_known_minus_unknown、open_set_accuracy、auroc、unknown_recall、unknown_f1、known_health_mean、unknown_health_mean。語意見[benchmark](exp8_health_index_benchmark.md#輸出怎麼看)，不從組平均反推逐窗預測。

全域每method的27列等權（九工況×三seed）；逐工況method×dataset的三seed等權。沒有樣本加權或方法選擇。mean為sum/n，std用ddof=1；排序method／dataset字串升冪，CSV index從0起。JSON平均／std round6；方法差先取未round平均的k-NN−Mahalanobis，再round6。CSV保留double。

比對JSON atol=5e-7、rtol=0；CSV atol=1e-12、rtol=1e-10；ID／欄集合精確比對。NaN／Inf拒絕；health_effect_size為null僅在兩組std都是0時允許，且不加入七欄平均。三seed的std不是獨立馬達母體信賴區間。

舊README支持「9工況×3seed平均」，原彙總生成程式未保存；ddof與round順序仍UNKNOWN。本工具規則明確，不代表已證實原作者用了同一公式。相符與不符都保留，不為通過修改封存或放寬容差。

## 怎麼使用

repo根目錄準備[uv環境](../runtime_policy.md)，不需要data：

```bash
uv run --locked python -m experiments.health_index_aggregate --help
uv run --locked python -m experiments.health_index_aggregate --source-root reports/exp8_health_index_results --contract docs/experiments/exp8_health_index_aggregate_contract.json
```

API run(source_root,contract_path)會保存結果；main退出碼0表示MATCH，3表示重算完成但DIFFERENT，2表示來源／schema等失敗。**3不是模型訓練失敗，也不能改成0掩蓋差異**。新矩陣根不符合這份固定來源SHA，不能直接套契約；另需事先建立來源契約。沒有Web重算／續跑入口，Web8-2仍讀舊封存。

## 輸出與預期

run經setup_run寫logs/exp8_health_index_aggregate/{ts}.log與output/exp8_health_index_aggregate/{ts}/：
aggregate_summary.json、aggregate_by_condition.csv、field_differences.csv、source_inventory.json、recomputation_audit.json與環境檔；失敗時failure.json。輸出與來源目錄禁止互相包含，重算前後再次核對16檔未變。

field_differences列key、old、new、difference、allowed_tolerance與matches；audit分status=COMPLETED與comparison_status=MATCH／DIFFERENT，並記SHA、commit、Python、套件與命令。預期六run、54列、18組摘要、每欄皆有判定；允許揭露未解差異，不能據此宣稱準確率提升。完整既有驗證與兩個round差異見[固定交付紀錄](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/experiments/exp8_health_index_aggregate_delivery.md)，不在此手冊重抄分數或單次工作紀錄。

## 方法來源與程式碼

平均、ddof、容差是Lineage重算約定，原演算法文獻見[benchmark](exp8_health_index_benchmark.md#方法來源與程式碼)。[aggregate](../../src/experiments/health_index_aggregate.py)實作驗證／重算／writer；[契約](exp8_health_index_aggregate_contract.json)鎖來源；[test_health_index_aggregate](../../src/tests/test_health_index_aggregate.py)測完整與失敗fixtures；[exp8_aggregate_validation](../../src/tests/exp8_aggregate_validation.py)保存測試證據。正式資料、模型、歷史摘要與Web邏輯均不改動。
