# 實驗六矩陣：保存多工況、seed與方法的執行狀態

## 資料角色用量表

流程：先列預期鍵 → 呼叫同一正式benchmark → 保存目標工況那列 → 保留失敗／續跑狀態。這是實驗六的排程器，不是新增模型。

| 資料角色 | 範圍 | 有效數量 | 比例與分母 | 用途 | 重疊 |
|---|---|---:|---|---|---|
| 單工況train／cal／test | 例T1／8000rpm健康 | 187／62／64 | 各除以313 | 沿用正式benchmark | 每方法、seed重用 |
| 同工況unknown | 其餘配置 | 2736 | 全故障池100% | 沿用未知拒絕 | 不fit／cal |
| 預期矩陣 | 9工況×3seed×2方法 | 54個run鍵 | 是執行格，非資料列 | 排程coverage | 每run還會重算其他工況 |
| 現行來源 | [九工況](README.md#資料用量與分母怎麼讀) | 28891列 | `data`，指紋`1b55b847…` | 唯讀前提查核 | 與`data/formal_local`不是同一份檔案（[#242](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/issues/242)） |

沒有validation選參；不把54格乘上列數當獨立樣本。缺工況時保留缺件，不能用allow-partial借道湊PASS。

## 發表與理解檢查

口頭主軸：「先固定工況、方法與seed清單，把每格的來源和狀態保存，讓失敗可追查。54格全部完成只代表工程coverage；是否好用要看未知召回、健康誤報與來源獨立性。」

- 六個方法／seed執行是否等於六顆馬達？不是，資料重用。
- resume是否驗完所有來源？它逐格核對三個必備檔、身分與資料指紋，見下方「完成判定規範」；仍不證明原始錄製獨立。
- 少於九工況能否關完整九工況驗收？不能，必須列出缺的工況。
- 組員請分開講run鍵、資料列與獨立採集；歷史分數只查固定輸出，不把排程完成當研究通過。

這個工具把[正式benchmark](exp6_formal_benchmark.md)放進可續跑矩陣，保留完成與失敗格；不增加偵測方法，也不做共同模型選擇。後續平均由[彙總工具](exp6_aggregate.md)處理。

## 迴圈、資料與fit

expected_run_matrix要求掃描九工況，預設seed為42／123／2026、方法為mahalanobis／knn，產生9×3×2＝54個run_id，格式T1_8000rpm_seed42_knn。這是54個工況×seed×方法記錄，不是54顆獨立馬達。資料列105維、healthy切60/20/20、train-only scaler與參考、known-only cal與unknown全池評估，全部沿用正式benchmark；原始時間／session／單位未知，不能靠seed建立獨立採集。

實作有一個成本差異：**每個run_id都呼叫一次涵蓋全部九工況的benchmark，再只取目標工況那一列保存**。因此有重複計算，54列不是只fit54次的承諾。同一次run_matrix內，CSV解析由[PoolCache](../profiling_contract.md#csv-cache依量測結果實作)以內容SHA快取，重複的只剩fit與評分；快取不改變split或分數。參數固定benchmark預設LW、confidence0.95、k=5；matrix CLI沒有改這三者的選項，不從fault結果調參。

## 操作與續跑風險

在repo根目錄準備[uv環境](../runtime_policy.md)與完整資料，先選一個尚未使用的輸出根：

```bash
uv run --locked python -m experiments.exp6_matrix --help
uv run --locked python -m experiments.exp6_matrix --data-root data --output-root output/exp6_formal_matrix/manual_run_001 --seed 42 --seed 123 --seed 2026 --method mahalanobis --method knn
```

manual_run_001只是範例，已存在就換新名稱，不能覆寫封存。API實際為run_matrix(data_root,output_root,seeds=(42,123,2026),methods=("mahalanobis","knn"),resume=True)，沒有獨立run(pools)介面或Web重跑按鈕。網頁只能讀既有結果。

同命令、同根再跑預設 resume：先唯讀核對新契約及所有完成格，通過後才寫 manifest。完成格缺件、hash／設定／版本不符一律拒絕，不偷偷重算覆寫。同契約的 pending／running／failed 格可重試，先將既有三檔保存到該格 attempts/ 保留失敗。--no-resume 只用新根；已有 manifest 或非空目錄不得覆寫。schema 1 歷史矩陣只讀，不能補欄位後原根續跑。

### 新證據契約

新 manifest 為 schema 2，contract.version 為 formal_matrix_provenance_v2：

| 固定欄位 | 核對內容 |
|---|---|
| source | 起始內容清冊、fingerprint_version／data_version、manifest SHA；不含絕對根 |
| environment／execution | 既有 collector 的實際來源、Python、套件、host、BLAS、Git HEAD／dirty、uv.lock SHA，另記流程程式檔 SHA；續跑核對數值環境與版本，主機名僅追溯 |
| method_configs | 共用 resolver 的逐法實值，及105維、train-only RobustScaler、健康60／20／20與unknown只測試；LW與k-NN各核對自己的合法設定 |
| run_keys | 工況、seed、方法的預期唯一鍵；新呼叫不能暗改coverage |

contract_sha256 是固定內容的 canonical JSON SHA，不含進度、時間或自身 hash。summary 的 matrix_provenance 綁此 contract、資料版本及逐法設定；summary／CSV／log 全部寫妥才保存三檔原始 bytes SHA。沿用格重讀 bytes；每次 benchmark 回傳及結束前重查起始來源，任何格不得改寫固定 fingerprint。中途變動保留 failed／incomplete。Git或鎖版證據不足時不得宣稱新契約通過；舊欄位不補造。這些驗證不消耗模型RNG，不重跑正式54格。

### 完成判定規範

唯一鍵：`run_id = {motor}_{rpm}_seed{seed}_{method}`，目錄固定為`runs/{run_id}/`；manifest裡的路徑一律用`/`分隔，舊封存的`\`路徑讀取時換成`/`。一格算completed要同時滿足：

| 必備產物 | 要求 |
|---|---|
| `summary.json` | status=completed；method、seed與鍵相同；恰一列、該列status=completed且motor／rpm／method／seed與鍵相同；有dataset_fingerprint，且等於本次資料指紋（續跑）或manifest該格記錄（彙總） |
| `results.csv` | 恰一列；dataset、motor、rpm、seed、method、status與八個指標和summary那一列相同 |
| `run.log` | 有`status=completed`；condition、seed、method三行與鍵相同 |

[`validate_run`](../../src/experiments/exp6_matrix.py)把每項不符寫成一條原因（例如`results.csv auroc: summary=0.91 csv=0.92`），`_is_complete`就是「原因清單為空」；[彙總](exp6_aggregate.md)用同一個函數。缺九工況或非法方法在開跑前拒絕；個別run失敗保存error並繼續，最後退出碼2，不把失敗靜默刪除。

## 輸出怎麼看

main經`setup_run("exp6_formal_matrix", make_output=False, data_root=args.data_root)`寫`logs/exp6_formal_matrix/{ts}.log`（含執行環境與每格狀態）；結果仍由_atomic_json／_write_run_csv寫output-root，讓同一根可以續跑。

- matrix_manifest.json：固定契約、預期清單、每格狀態、耗時、摘要相對路徑與三檔SHA；已有根先唯讀核對，不先重寫。
- runs/{run_id}/summary.json：正式benchmark設定、fingerprint、單目標工況列；同目錄的 provenance.json／environment.json 是該格 `run(..., evidence_dir=runs/{run_id})` 寫的來源與環境紀錄（#269）。
- runs/{run_id}/results.csv：相同單列的表格。
- runs/{run_id}/run.log：開始、狀態與錯誤；duration_seconds含整次九工況運算，不等於該列單獨推論耗時。

completed只表示本工具54格完成，unknown召回／健康接受等解釋見[正式版](exp6_formal_benchmark.md#輸出怎麼讀)。三seed的切分敏感度不能當motor母體信賴區間。預期54完成、0失敗且各列可追溯；來源／參數完整防護仍不足時，不宣稱研究可靠性通過。

2026-10-23 在 Aero 以 commit `46ed4d5`、`--data-root data`（資料指紋 `1b55b847…`）跑完 54 格（[matrix_manifest.json](../../output/exp6_formal_matrix/aero_2026-10-23/matrix_manifest.json)，54 completed、0 failed）：健康接受率（known_accuracy）27 個工況-seed 列等權平均，LW 94.66% ± 3.97（最低 87.50%，T3／6000 seed 42）、k-NN 94.67% ± 4.38（最低 84.38%，T1／6000 seed 42），AUROC 與未知召回都是 1，見 [aggregate.md](../../output/exp6_formal_matrix/aero_2026-10-23/aggregate/aggregate.md)。舊版 94.62% ± 4.01／94.57% ± 4.39 是 `data/formal_local` 的歷史結果（commit 07582ff、Python 3.12.14、無主機紀錄，[舊 aggregate](../../output/exp6_formal_matrix/aggregate/aggregate.md)）；`formal_local` 由階段 ZIP 物化，不在 Aero／Spark 上，兩份資料的列數差異見 [#242](https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/issues/242)。

## 方法來源與程式碼

統計方法文獻見[正式benchmark](exp6_formal_benchmark.md#方法來源與程式碼)，run_id、54格與resume為Lineage操作約定。[exp6_matrix](../../src/experiments/exp6_matrix.py)負責排程與writer；[benchmark](../../src/experiments/exp6_formal_benchmark.py)負責fit／校準／指標；[test_exp6_matrix](../../src/tests/test_exp6_matrix.py)用fixture檢查完成判定的每條拒絕原因與續跑。[web/experiments](../../src/web/experiments.py)的唯讀結果入口不會啟動此工具。
