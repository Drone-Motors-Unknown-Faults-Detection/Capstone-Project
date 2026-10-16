# 內容SHA與portable／private來源契約

給維護資料匯入與正式實驗入口的組員：本頁說明 `core/provenance.py` 如何辨識內容、驗證 manifest，以及哪些資訊留在本機。正式欄位檢查見[載入契約](formal_feature_schema_contract.md)，模型與切分見[實驗六手冊](experiments/exp6_formal_benchmark.md)。原交付與失敗紀錄另存[固定版本證據](../reports/Andy_20261009_正式入口來源/交付紀錄.md)，不在本頁維護單次測試數。

## 版本與雜湊規則

`source_content_v2`：對實際正式CSV bytes逐檔SHA256，以排序的POSIX相對ID＋內容SHA清冊作canonical JSON（sort_keys=True，UTF-8，compact separators），整體SHA作新資料指紋。size／mtime不進內容身分。無manifest仍檢查真實bytes；同內容異root副本可得到相同指紋，但不視為獨立錄製。同dataset內副本列為alias，不自行去重或當新馬達。
明確拒絕空清冊、無法讀取、損壞manifest／未知schema、不匹配已宣告輸出SHA／檔案清冊；前後來源變動拒絕完成。既有stat與manifest指紋只透過明確legacy入口讀取並標 `legacy_fingerprint_v1`；不回寫歷史輸出，不把舊指紋與新版本混為同資料ID。
物化新manifest `formal_materialization_v3`：來源根改固定portable ID，output為根內相對ID；每筆明記output SHA（歷史source_sha256實為衍生CSV bytes的欄位保留相容），archive內容SHA與member ID另存。採集session／時間／原始同步仍UNKNOWN。v1／v2可唯讀讀取；其絕對path只在本機解析成相對清冊，不複製到公共摘要。

## 公開與私人資料

公開只保存schema、相對ID、SHA、data version、合法schema版本與設定；不序列化解析後絕對root、不蒐集token／任意環境變數。私人sidecar放 `.lineage_private/`（新增忽略規則），保存source／output解析路徑；不stage、不上傳Actions。這是本機明文追溯檔，不聲稱加密或可抵抗擁有本機權限的人。
只有記憶體pools的其他API不會由此取得檔案來源證明；不能補造檔案SHA或把array內容指紋當raw錄製身分。
環境只能重用 `core.runtime_environment.collect_environment`／`setup_run`，不新增第二套collector。formal API回傳實際環境與來源；formal CLI／物化run使用既有setup_run保存環境、portable source及resolved已傳入設定，失敗保存error type與已知來源狀態，不公開例外payload。

## 入口與支援範圍

支援 `core.provenance.run/main`、`core.formal_data.run/main` 與 `experiments.exp6_formal_benchmark.run/main`。其他記憶體pools API、多數runner與統一resolved設定不由本契約保證；缺口見 [#46](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/46) 與 [#47](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/47)。

```bash
uv run --locked python -m core.provenance --data-root data/formal_local
uv run --locked --extra test python -m pytest tests/test_provenance.py
```

唯讀核對入口經 setup_run("source_provenance", unique=True) 寫 logs/source_provenance/{ts}.log、output/source_provenance/{ts}/environment.json 與 source.json。成功的 VERIFIED 只證實所讀內容與適用manifest；raw/session與fresh test仍UNKNOWN。失敗保存error_type並拋出例外，不輸出假成功。

## 預期成果與驗證

fixture涵蓋同stat不同bytes、同內容副本、不同root、無manifest、legacy讀取、損壞／不符manifest、private不進Git公開清冊、來源讀取失敗及無Git。預期同stat異bytes能區分、相同相對ID與內容的副本指紋一致、錯manifest拒絕，來源紀錄不改fixture的score／threshold／classify。實際平台支援與跳過案例以當次受測SHA及測試紀錄為準。
入口遵循run/main、setup_run及logs/output，工程契約不占exp9–12。

`setup_run(..., unique=True)` 同秒採 `_001` 等尾碼並以exclusive建立保留，避免來源紀錄覆寫；既有呼叫預設不變，不保證所有runner無碰撞。這項只影響新run的產物ID，不消耗模型rng。formal summary版本3，dataset_root固定 `formal_features` ID，真實解析根只留sidecar。
傳入參數與有效config一併記錄，不代表統一resolved設定schema已完成。standalone `materialize()` 回傳portable v3 manifest；只有 `core.formal_data.run/main` 額外保存本機sidecar，直接API呼叫者需自行保留其來源根。private sidecar失敗會使run失敗，但已物化的檔案可能保留，沿用非整批交易限制。
