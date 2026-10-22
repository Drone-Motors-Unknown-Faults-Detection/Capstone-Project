# 內容SHA與portable／private來源規則

給維護資料匯入與正式實驗入口的組員：本頁說明 `src/core/provenance.py` 如何辨識內容、驗證 manifest，以及哪些資訊留在本機。正式欄位檢查見[載入規則](formal_feature_schema_contract.md)，模型與切分見[實驗六手冊](experiments/exp6_formal_benchmark.md)。

## 版本與雜湊規則

`source_content_v2`：對實際正式CSV bytes逐檔SHA256，以排序的POSIX相對ID＋內容SHA清冊作canonical JSON（sort_keys=True，UTF-8，compact separators），整體SHA作新資料指紋。size／mtime不進內容身分。無manifest仍檢查真實bytes；同內容異root副本可得到相同指紋，但不視為獨立錄製。同dataset內副本列為alias，不自行去重或當新馬達。
明確拒絕空清冊、無法讀取、損壞manifest／未知schema、不匹配已宣告輸出SHA／檔案清冊；前後來源變動拒絕完成。既有stat與manifest指紋只透過明確legacy入口讀取並標 `legacy_fingerprint_v1`；不回寫歷史輸出，不把舊指紋與新版本混為同資料ID。
物化新manifest `formal_materialization_v3`：來源根改固定portable ID，output為根內相對ID；每筆明記output SHA（歷史source_sha256實為衍生CSV bytes的欄位保留相容），archive內容SHA與member ID另存。採集session／時間／原始同步仍UNKNOWN。v1／v2可唯讀讀取；其絕對path只在本機解析成相對清冊，不複製到公共摘要。

## 公開與私人資料

公開只保存schema、相對ID、SHA、data version、合法schema版本與設定；不序列化解析後絕對root、不蒐集token／任意環境變數。私人sidecar放 `.lineage_private/`（新增忽略規則），保存source／output解析路徑；不stage、不上傳Actions。這是本機明文追溯檔，不聲稱加密或可抵抗擁有本機權限的人。
只有記憶體pools的其他API不會由此取得檔案來源證明；不能補造檔案SHA或把array內容指紋當raw錄製身分。
環境只能重用 `core.runtime_environment.collect_environment`／`setup_run`，不新增第二套collector。formal API回傳實際環境與來源；formal CLI／物化run使用既有setup_run保存環境、portable source及resolved已傳入設定，失敗保存error type與已知來源狀態，不公開例外payload。

## 入口與支援範圍

### 資料池來源上下文

`load_pool_source(data_root, dataset_path)` 重用 `capture_source()` 與 `load_pools()`，核對載入前後清冊及 parser 實際讀取 bytes 的 SHA；可傳既有 `PoolCache`，命中也必須核對內容。`PoolSource` 只供行程內傳遞，不能直接序列化；公開資料由 `source_for_pools()` 取得。它核對清冊與池陣列未改、磁碟仍對應同一版本，私人根目錄僅留原有 sidecar。

`pool_source_v1` 的 `row_mapping` 每段描述 configuration、portable source ID、CSV SHA、pool 的半開區間及 CSV 資料列的零起算半開區間；不含標頭，未刪列、不改拼接次序。這些是 clean CSV 列，不是原 DAQ 時間或 raw-window。陣列 SHA 只做行程綁定，不替代檔案 SHA 或採集身分。清冊或陣列被改、載入中變動、sidecar 寫入失敗均拒絕；空來源的 API 明記 `UNVERIFIED_MEMORY`、版本及物理獨立性 UNKNOWN。

`core.runner.execution_context()` 重用既有環境 collector 與 Open Set resolver；`run_dataset()` 必須在入口 `setup_run(..., unique=True)` 後呼叫，載入／執行失敗仍保存去敏 `source.json`，不輸出任意例外 payload。

環境資料根必須與實際來源相同：`execution_context()` 先驗證 `PoolSource`，再把行程內的 `root` 傳給 `collect_environment(data_root=...)`；root API 直接傳實際的 `data_root`。CLI 的 `setup_run()` 明確帶 `args.data_root`，`run_dataset()` 與有輸出目錄的 root runner 也將同一實際根的環境寫回該次 `environment.json`。這只更新本次新建紀錄，不回寫歷史產物。公開環境只保留內容指紋，不序列化 `PoolSource` 或絕對根。

純記憶體 pools 呼叫 `collect_environment(include_data=False)`：`environment.data.status=UNKNOWN`、指紋為空並標 `reason=UNVERIFIED_MEMORY`，不查工作目錄或命令列中的另一份 `data` 來背書。一般環境查核入口仍保留既有命令列預設；此選項僅供已知沒有可信檔案來源的呼叫者明確關閉資料蒐集。

程式根與資料根分開：collector 預設由載入的模組定位程式樹，Git 與 uv.lock 不隨 CWD 改變；明確 root 查詢沿原契約。`program_source` 的 program_location_v1 區分預設／明確定位與 Git／lock 未知原因，不公開絕對根。無 Git 匯出或已安裝套件不借其他 repo 的版本；輸出位置仍由 setup_run 相對工作目錄建立。matrix／aggregate 的既有程式根查詢與版本拒絕保留，新程式內容 SHA 不得補回舊契約後原根續跑，細節見[環境操作](runtime_policy.md#執行環境紀錄)。

| 適用入口 | 來源現況與驗收邊界 |
| --- | --- |
| provenance／formal_data／exp6_formal 的 run/main | 已有內容來源、private 分離與環境；沿用既有規則 |
| exp1／exp2／exp3 CLI | 先 setup_run(unique=True)，再 run_dataset；環境與成功／失敗 source.json 屬同一 run |
| exp1／exp2／exp3 run／iterator API | 可傳 load_pool_source 取得的 source_context；無上下文明記 UNVERIFIED_MEMORY；結果 provenance 綁定環境與實值設定 |
| Web exp1／exp2／exp3 批次與串流 | 共用 core 的內容快取與來源；回傳前再驗，來源／計算失敗保存 FAILED，不回覆 done 成功 |
| exp4／exp7 CLI、run API；exp4 iterator；Web 相同入口 | 重用 PoolSource 與 run_dataset；exp7 各方法的結果各保存有效設定與 provenance；純記憶體 API 保留 UNVERIFIED_MEMORY |
| exp5／廣度 exp6／exp10／exp11／exp12 run、CLI、Web 批次 | 共用 run_root_sources；逐工況 load_pool_source，前後核對全根版本及各池；CLI 同 run 保存 source.json 成功／失敗。結果參數記實際資料角色規則；列映射只到 clean CSV，不保證獨立錄製 |
| exp8 health_index_benchmark run／CLI／Web 批次 | 同一來源核對器支援內容快取；fit_audit 保存各工況健康 train／cal／holdout 列及未知評估配置，與 row_mapping 合讀 |
| exp8 health_monitor run／iterator／CLI／Web | 共用 PoolSource；fitted 與逐窗結果保存來源及環境，耗盡 iterator 前再驗版本；provenance.health_calibration 保存既有 model.metadata() 的實際健康校準錨點、分位數與樣本數，CLI 各 output-mode 的 metadata.json 皆保留，不改健康映射。CLI 載入失敗仍保存同 run FAILED。window_index 是回放順序，原列位置另由 source_pool_index 指向 clean CSV |
| matrix／aggregate | 新矩陣採 formal_matrix_provenance_v2 固定來源、執行身分及逐法設定，彙總綁實際父檔 bytes SHA；舊 schema 只讀並標 LEGACY_UNVERIFIED，不允許補欄位後原根續跑 |
| exp9 | 無獨立 runner，沿廣度 exp6 的支援範圍 |
| exp13／首頁／guide | 有自己的契約／落盤，但不由此推論所有執行已具有完整 source_content_v2 鏈 |

本表未全部驗收前 #46 保持開啟；來源工程不讓 #9 的 fresh／採集獨立性自動通過。

支援範圍依上表；其他記憶體 pools API 與尚未整合的 runner 不由本規則保證來源 sidecar；缺口見 [#46](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/46)。Open Set 入口的 resolved config 由 `core.openset.resolve_openset_config()`（`openset_config_v1`）產生並寫進各入口輸出（見 [實驗總覽的共用設定解析](experiments/README.md)），但單有設定紀錄不代表內容來源清冊完整。

```bash
uv run --locked python -m core.provenance --data-root data/formal_local
uv run --locked --extra test python -m pytest src/tests/test_provenance.py
```

唯讀核對入口經 setup_run("source_provenance", unique=True) 寫 logs/source_provenance/{ts}.log、output/source_provenance/{ts}/environment.json 與 source.json。成功的 VERIFIED 只證實所讀內容與適用manifest；raw/session與fresh test仍UNKNOWN。失敗保存error_type並拋出例外，不輸出假成功。

## 預期成果與驗證

### 矩陣與彙總的父產物

新 matrix manifest schema 2 的 `contract.version=formal_matrix_provenance_v2`，保存起始 source_content_v2 清冊、data_version、環境、程式檔 SHA、Git HEAD／dirty、uv.lock SHA、逐方法有效設定及預期 run 鍵。`contract_sha256` 只雜湊這份不含自身 hash 的固定內容；完成格 summary 綁同一 contract。三個必備產物的 `artifact_sha256` 是完整原始 bytes，不是排序後 JSON。

新彙總 schema 2 的 `provenance` 保存實際讀取 manifest 的 bytes SHA、接受格的三檔 SHA、portable run ID、彙總執行環境及來源驗證狀態，均不參與八項科學平均。completed／PARTIAL 描述工程 coverage，VERIFIED／LEGACY_UNVERIFIED 描述證據契約；checksum 不是作者簽章，也不證明採集或視窗獨立。缺原始採集事實繼續 UNKNOWN。

fixture涵蓋同stat不同bytes、同內容副本、不同root、無manifest、legacy讀取、損壞／不符manifest、private不進Git公開清冊、來源讀取失敗及無Git。預期同stat異bytes能區分、相同相對ID與內容的副本指紋一致、錯manifest拒絕，來源紀錄不改fixture的score／threshold／classify。實際平台支援與跳過案例以當次受測SHA及測試紀錄為準。
入口遵循run/main、setup_run及logs/output，工程規則不占exp9–12。

`setup_run(..., unique=True)` 同秒採 `_001` 等尾碼並以exclusive建立保留，避免來源紀錄覆寫；既有呼叫預設不變，不保證所有runner無碰撞。這項只影響新run的產物ID，不消耗模型rng。formal summary版本3，dataset_root記實際根的名稱（例如 `data` 或 `formal_local`），真實解析根只留sidecar；名稱不是獨立採集身分。
傳入參數與有效config一併記錄，不代表統一resolved設定schema已完成。standalone `materialize()` 回傳portable v3 manifest；只有 `core.formal_data.run/main` 額外保存本機sidecar，直接API呼叫者需自行保留其來源根。private sidecar失敗會使run失敗，但已物化的檔案可能保留，沿用非整批交易限制。
