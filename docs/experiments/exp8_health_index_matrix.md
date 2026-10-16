# 實驗八矩陣：保存六個run、54列健康指數結果

## 資料角色用量表

流程：固定3seed×2方法 → 每run呼叫一次九工況健康指數benchmark → 保存每run九列 → 另交彙總工具。這個排程器不另外學健康量尺，也不挑共同winner。

| 資料角色 | 範圍 | 數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---|---|---|---|
| train／cal／test | 每工況健康池 | 例T1／8000rpm為187／62／64 | 各除以313 | 沿用benchmark | cal同時定線與映射 |
| unknown test | 同工況fault全池 | 例2736 | 故障池100% | 同方法／seed評估 | 方法與seed重用 |
| 正式矩陣 | 九工況×三seed×兩法 | 6run、54列 | run與工況列是不同單位 | coverage及分數保存 | 不等於54顆馬達 |
| 本機可用資料 | [六工況](README.md#資料用量與分母怎麼讀) | 19053有效列 | 非原九工況封存版本 | 唯讀前提查核 | 不能補成完整54列 |

本輪沒有重跑矩陣或改resume。無validation選參；資料用量以實際root為準，不能把封存列數套進本機缺T2的來源。

## 發表與理解檢查

口頭主軸：「同一健康指數流程用固定三seed與兩方法保存，讓切分變動可查。矩陣只是結果管理，模型是否有用仍看benchmark指標與來源條件。」

- 為何實驗六是54run而這裡6run？兩個runner的排程粒度不同，這裡一run已含九工況。
- 看Web8-2是否剛重跑？沒有，它讀封存，與新output根分開。
- 六run完整是否獨立驗證？不是，來源重用且原始session仍未知。
- 組員請說清run、row與sample三層；用量和狀態不足就列限制，不把resume當來源完整查核。

這個工具重複[健康指數benchmark](exp8_health_index_benchmark.md)，讓每種方法的seed切分變動可查。它沒有新的健康模型或自動選擇勝者；六個run各含九工況，與實驗六的54個run目錄不同。

## 資料與執行順序

預設seeds為42／123／2026，methods為mahalanobis／knn；run_id是seed_42_mahalanobis。每次benchmark自己掃描／檢查九工況、按motor／rpm排序，健康列60/20/20：train fit scaler與參考；cal定距離線及健康錨點；healthy holdout與所有fault全池評估。105維／unknown不fit／原始來源限制見[benchmark](exp8_health_index_benchmark.md)。九工況是馬達×RPM，不是九種物理故障。重複seed不是獨立採集。

矩陣每seed×method呼叫一次benchmark，因此共6次、54列。confidence0.95、LW與k=5來自benchmark預設，matrix CLI沒有另外調這些值。只有來源與迴圈集合一致時方法之間切分才一致。

## 怎麼跑與續跑

在repo根目錄依[uv環境政策](../runtime_policy.md)準備正式資料。CLI經`core.logger.setup_run("exp8_health_index_matrix")`初始化，日誌寫`logs/exp8_health_index_matrix/{ts}.log`；不給`--output-root`時結果寫本次的`output/exp8_health_index_matrix/{ts}/`。要續跑既有輸出根才給`--output-root`，不要指向`reports/exp8_health_index_results/`封存：

```bash
uv run --locked python -m experiments.health_index_matrix --help
uv run --locked python -m experiments.health_index_matrix --data-root data/formal_local --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn
uv run --locked python -m experiments.health_index_matrix --data-root data/formal_local --output-root output/exp8_health_index_matrix/2026-10-17-10-00-00   # 續跑上一次的輸出根
```

API run_matrix(data_root,output_root,seeds=(42,123,2026),methods=("mahalanobis","knn"),resume=True)；data_root與output_root都必須明確給，沒有預設寫入位置。沒有run(pools)，也沒有Web重跑入口。

resume只查JSON status、seed、method及formal_condition_count==9；**沒核對rows實際長度／唯一工況、當前資料fingerprint、參數、CSV或來源SHA**。即使data-root已缺件，快取條件通過仍可能全部跳過；因此不能說resume已重新驗證九工況完整。--no-resume會覆寫同名JSON／CSV；不在封存目錄使用。錯誤寫manifest後繼續，最後incomplete退出碼2。

## 輸出與讀法

[run_matrix](../../src/experiments/health_index_matrix.py)以_atomic_json與pandas直接寫output-root：
matrix_manifest.json、seed_{seed}/{method}.json及同名.csv。manifest記expected_runs=6、每run狀態／相對路徑，JSON保存benchmark九列設定與指標；CSV是rows表。每個run的完成、續跑或失敗原因都寫進`logs/exp8_health_index_matrix/{ts}.log`。此模組不自動產生aggregate_summary.json。

健康指數與各指標見[benchmark](exp8_health_index_benchmark.md#輸出怎麼看)。預期六run完成、54列齊全且same-source同seed一致；這是工程coverage，不是獨立馬達信賴保證。每次重跑先保留失敗原因，不混入其他fingerprint的平均。

[唯讀彙總工具](exp8_health_index_aggregate.md)針對既有封存的固定契約重算，不能把新run根直接當成已接受的歷史來源。Web實驗八8-2讀reports/exp8_health_index_results/，**不自動讀本次新output根**；用於查歷史，不代表剛執行這個matrix。

## 來源與程式碼

方法文獻見[benchmark](exp8_health_index_benchmark.md#方法來源與程式碼)，六run命名與resume是Lineage操作約定。[matrix](../../src/experiments/health_index_matrix.py)排程與writer；[test_health_matrix](../../src/tests/test_health_matrix.py)檢查矩陣；[web/experiments](../../src/web/experiments.py)的_load_exp8_results讀封存。原完整結果見[固定封存包](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/tree/1fa9431bb7b86959f29540d07b2b9290ab39ce42/reports/exp8_health_index_results)，不是新資料驗證。
