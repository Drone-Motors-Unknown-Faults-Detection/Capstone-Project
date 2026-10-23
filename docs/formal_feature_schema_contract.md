# 正式105維特徵載入規則

給資料匯入與 loader 維護者：本頁說明合法欄位、拒絕原因與完整配置檢查。`src/core/data.py:load_pools` 使用 `src/core/feature_schema.py:SCHEMAS` 驗證 CSV。實驗六的模型、切分、指標與執行方式見[正式比較手冊](experiments/exp6_formal_benchmark.md)。

## 合法輸入

完整 ordered columns 在 `src/core/feature_schema.py` 的 `SCHEMAS`。只接受兩個完整版本，不以數值欄數、欄名前綴或大小寫猜測相容，不排序／重排欄。

- `historical_clean_v1`：對照 `Step2_Feature_Extraction_8000_2.py` 的 `feature_name`。三個 `current` 小寫與 Y／Z FFT 的 X 後綴是原始命名，欄名本身不能證明訊號取自哪個軸。
- `adapter_generated_v1`：對照 `src/core/formal_data.py:FEATURE_NAMES`。三個 Current 為大寫，FFT 後綴依 Y／Z 軸命名。合法舊檔不改名、不重產。

105個位置依序為 Current15、Vibration_X25、Vibration_Y25、Vibration_Z25、Delta_T15；各通道先15統計量，振動再10倍頻幅值。完整計算差異已由研究分支 `b3d68f1` 的 `reports/fault_type_openset/provenance_followup.md` 記錄，本項不重做公式研究、不修歷史命名。
採樣率、物理單位、安裝軸向、負載與 session／raw-window獨立性仍 UNKNOWN；schema通過僅證格式，不證量測一致。

## 拒絕政策與用途分層

缺／多／重複／錯序欄、任意 f0…f104、混合兩版本欄名皆拒絕；非數值、NaN／Inf、空檔／空列／空配置皆拒絕，不靜默刪列。允許 UTF-8 BOM 作編碼標記。錯誤提供 reason、portable file ID、實際 rows、拒絕 rows 與總計；不把私人絕對路徑寫入摘要。
配置只接受現有10類與 `1screw`→`1screws` alias。池的 key 是正規名（`1screw` 目錄讀成 `1screws`），原目錄名留在 audit 的 `configuration_dirs` 與各檔案的 file_id，檔案順序不變（#271）；同工況兩個 alias 同時存在時拒絕，避免把重複來源當兩類。忽略沒有正式CSV的非配置輔助目錄；合法配置目錄存在卻沒正式CSV則拒絕。
`load_pools(..., require_complete=False)` 保持 cold-start healthy-only 可用；`validate_pool_coverage(..., require_complete=True)` 另驗10個canonical類別。正式 exp6 在 `require_nine=True` 時同時要求每工況完整配置；明確 partial入口仍回報缺類。不能用 healthy-only 通過完整正式矩陣。工況數另由既有9工況檢查負責，不把類別數當工況數。
檔案逐一先驗header／row width，再沿用 pandas 原解析與 float矩陣，不更改有限合法值、列順序或 make_split 隨機數消耗。標記在來源相對檔名與原始 row index，既有來源檔SHA／樣本身分規則不改。

## 操作與驗證範圍

`src/tests/test_feature_schema.py` 以暫存 fixture 檢查兩套 header、拒絕案例、alias、coverage、重現與 CLI 錯誤，不依賴 ignored data。預期非法輸入整批拒絕、合法數值與列順序不變；格式通過不等於採集獨立性通過。`src/tests/feature_schema_evidence.py` 是唯讀載入與切分的證據入口，經 setup_run 寫 logs/output，不擬合模型。

`rejected_rows` 指已查明的違規列數；錯header時整檔列都違規，數值問題則另列零起算的 `rejected_row_indices`。遇到錯誤整個載入失敗、不回傳合法子集；其他尚未讀檔列數不能由此計數推定。無法讀取的來源不補造列數。`discarded_rows=0` 表示沒有清洗／刪點；不等於整批輸入已被接受。
CLI：`uv run --locked python -m core.data --dataset <RPM目錄>`；完整配置加 `--require-complete`。API失敗拋 `FeatureSchemaError`，可由 `audit={}` 取得portable拒絕紀錄；CLI保存 `output/formal_schema_audit/*/schema_audit.json` 並exit 1。同目錄的 `environment.json` 以 `core.data.dataset_data_root(--dataset)`（RPM目錄往上找到 `Step-*` 的上一層）當資料根算指紋。回歸檢查：`uv run --locked --extra test python -m pytest src/tests/test_feature_schema.py`。
