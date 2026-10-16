# 正式 archive 物化操作與安全契約

core/formal_data.py 從既有階段ZIP取得105維特徵：Stage1／3複製clean特徵，Stage2依既有五通道流程轉換。本頁說明用途與拒絕條件，不授權重新覆寫正式資料。

## 入口與結果

API為 materialize(source_root, output_root, *, stages, conditions, force=False)；run(...)加上日誌。先看CLI：

```bash
uv run --locked python -m core.formal_data --help
```

需要另建資料版本時，明確提供唯讀來源與新的目的目錄，例如：

```bash
uv run --locked python -m core.formal_data --source-root SOURCE --output-root NEW_DEST --stage 2 --condition T2/8000rpm/8screws
```

SOURCE、NEW_DEST是呼叫者替換的路徑；不要直接使用既存正式data作目的地。--force是覆寫選項，不應用於封存正式資料。

特徵及formal_materialization_manifest.json寫到指定目的地。此版本寫 formal_materialization_v3：根以固定ID表示，各檔 output 為相對ID、另記 output_sha256；保留 archive SHA、通道形狀、清理規則及同步未知狀態。v1／v2 舊檔唯讀相容，不回寫；內容核對與公開／私人邊界見[內容來源契約](content_provenance_contract.md)。

run()／CLI另經setup_run("formal_materialization", unique=True)保存logs、environment與去除私人路徑的summary，解析根保存於本機 `.lineage_private/` sidecar。直接materialize()不初始化該執行紀錄或保存sidecar，呼叫者須保留來源上下文。sidecar失敗會使run失敗，但已物化檔案可能保留，不宣稱整批交易。

## 路徑與寫入

ZIP各層拒絕空節、.、..、反斜線、絕對路徑、drive／UNC、冒號與symlink。MOTOR／RPM／CONFIG須符合程式允許值及既有1screw alias。目的檔resolve後須在解析後output_root內，根下沿途symlink即使指向根內也拒絕；output_root不得在來源根內。

全部來源讀取、驗證及計算後才開始寫入；來源SHA前後不符即拒絕。個別檔案用同目錄唯一暫存與os.replace，中斷清除該檔暫存。多檔提交途中磁碟失敗可能保留已完成檔案，但不發布完成manifest；不是整批交易或敵對多使用者隔離。

## 通道契約

Stage2的equal_window_counts_v1要求五通道窗口數、每窗口樣本數皆相等且非空，不同直接拒絕，不裁成最短通道。同形狀只代表計算相容；缺時間戳或clock bridge時alignment_status=UNKNOWN，不能宣稱物理同步。

預期成功結果是版本化特徵及轉換紀錄，不是新的獨立採集。既有FFT、統計量及IQR不由本頁修改。

## 驗證與歷史

```bash
uv run --locked python -m pytest tests/test_formal_materialization_safety.py -q
```

fixture涵蓋惡意路徑、symlink、通道差異、拒絕前零寫入、中斷清理、來源不變與舊版數值等價；缺權限而跳過的案例列未驗證。歷史失敗與修正見 [改寫前固定版本](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/791216cf5602c370091fa516264f1ce6aaad6ab1/docs/formal_materialization_contract.md)，原logs/output保留。
