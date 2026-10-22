# AGENT.md — Agent 指引文件

本文件為 AI Agent（含 Claude Code 等自動化工具）提供操作本專案的背景知識、慣例與行為準則。

---

## 專案定位

本 repo 目前的主線是 **冷啟動 PHM 畢業專題**：只用健康資料（8screws）建立基準，
即時偵測未知故障、以 HDBSCAN 發現新故障類型、成群後只用該群樣本自學一個新類、擴張「健康量尺」
（開放集 + 持續學習），並以變化點分析判別故障是漸進磨損還是突發事件。
實驗二（批次與實驗頁）走 `ScaleGrowthSession.adopt()` 自學；首頁即時展示與獨立導覽保留
`confirm()` 手動確認，兩條路徑見「已知行為與陷阱」第 1 條。
完整說明、實驗結果與引用文獻見 [README.md](README.md)。

原論文版程式碼（Step 1–6 管線、128 個 notebook 與論文全文）已整批移回
[原始論文](https://github.com/Drone-Motors-Unknown-Faults-Detection/Original-Paper) repo
（本機路徑 `~/Ancestor`）保存；本 repo 的 git 歷史仍保留搬移前的所有版本。
技術文件快照（含三份 Lineage 時期研究文件）於 2026-09-17 放回本 repo 的 `docs/`。

---

## 目錄結構

```
src/core/        共用零件：data / feature_schema / mahalanobis / monitor / openset / geometry / detectors /
                 formal_data / provenance / trend / logger / runner / runtime_environment
reports/         單次實驗結果／單次說明／更新紀錄，每篇一個資料夾，命名
                 `<Author Name>_<YYYYMMDD>_<Report Subject>`；見「docs／reports／note 分工」
note/            個人記事，非專案全局變動；每人一個資料夾（例如 note/JW-Albert/）
src/experiments/ 實驗模組（exp1 冷啟動 exp1_cold_start.py、exp2 量尺擴張 exp2_scale_growth.py、
                 exp3 趨勢 exp3_trend.py、exp4 極座標 exp4_polar_map.py、
                 exp5 跨工況 exp5_cross_condition.py、
                 exp6 OSR 基準＋正式矩陣＋彙整（exp6_osr_benchmark.py / exp6_formal_benchmark.py /
                 exp6_matrix.py / aggregate_exp6.py）、exp7 compare_openset.py、
                 exp8 health_index_* / health_monitor.py（邏輯在 src/experiments/health/：
                 健康指數、校準、嚴重度、趨勢/告警、軌跡與診斷，見 docs/experiments/exp8_health.md）、
                 exp9 非線性 AutoEncoder 偵測器（程式在 src/core/detectors.py，複用 exp6_osr_benchmark.py）、
                 exp10 跨工況遷移學習 exp10_transfer.py、
                 exp11 原始論文協定對照 exp11_ancestor_comparison.py、
                 exp12 混淆矩陣與 t-SNE 視覺化 exp12_confusion_tsne.py、
                 exp13 導覽一致性 navigation_data_contract.py／navigation_regression.py）；
                 總覽見 docs/experiments/README.md；技術報告在 docs/experiments/
src/web/         即時展示與實驗頁（live.py 串流編排、experiments.py 實驗頁目錄與執行、
                 server.py Tornado+WS+HTTP API、playback.py 實驗頁執行紀錄、
                 guide.py 獨立導覽（static/guide.html + guide.js + guide.css）、
                 static/index.html + experiments.js + ai_chat.js + markdown.js、
                 assistant.py AI 問答（LangGraph agent 連本地模型，知識庫在 web/knowledge/*.md））
src/tests/       pytest 單元測試；conftest.py 讓每個測試在自己的暫存目錄執行，
                 測試中呼叫 setup_run() 不會寫進 repo 的 logs/ 與 output/。
                 需要留存證據的驗證腳本（*_evidence.py 等）以 CLI 執行時
                 一樣經 core.logger.setup_run() 寫 logs/ 與 output/，不得改成手寫報告
docs/            實驗技術報告在 docs/experiments/（每個 src/experiments/*.py 一份 .md）
                 ＋論文版技術文件快照、Lineage 研究文件及 navigation 操作導覽；
                 單次交付與整合紀錄放 reports/。
                 歷史快照見 docs/README.md；快照裡的程式路徑不對應現行架構，勿據以改碼
data/            特徵資料（git 忽略；由論文版管線產出，本專案唯讀）
logs/ output/    每次執行的日誌與結果（納入版控）。output/ 只能是程式碼寫出的檔案，見鐵則 9
docker/          compose.yaml（include 下面兩個）、vllm.compose.yaml（NGC vLLM + Qwen3，AI 問答模型）、
                 app.compose.yaml + Dockerfile（展示伺服器映像，data/ 唯讀掛入、logs/ output/ 可寫）、.env.example
uv.lock          完整依賴樹的唯一鎖版來源（含間接依賴）；pyproject.toml 只宣告直接依賴的訂死版本
scripts/quantity/ 提交前檢查：run_pytest.sh（鐵則 10）、run_ruff.sh（鐵則 12）；
                 CI 的獨立 Pytest／Ruff step 也呼叫這兩支
scripts/opt/     其餘 shell：run_web.sh 啟動展示伺服器；run_llm.sh 啟動 AI 問答用本地模型（vLLM／llama.cpp）；
                 restart_docker.sh 先 compose down 再 up -d --build（all／app／vllm）；build_uv.sh 以 uv sync --locked
                 建立新環境，版本訂死在 pyproject.toml，完整依賴樹鎖在 uv.lock，不刪除既有目錄
```

---

## 鐵則（違反會破壞專案架構）

1. **不依賴論文版程式碼。** 需要其零件時，以「複製 + 檔頭註記來源」帶入
   （範例：`src/core/mahalanobis.py`）。論文版程式碼在原始論文 repo，視為唯讀封存。
2. **實驗邏輯只住在 `src/experiments/` 與 `src/core/`。** `src/web/` 只做編排與視覺化——
   任何判定、分群、統計邏輯若出現在 web 層就是放錯位置。這是為了讓論文批次執行
   與即時展示共用同一份邏輯、數字一致。
3. **每個實驗模組維持雙介面**：`run(pools, ...) -> dict`（程式化 API，web 與論文都呼叫）
   ＋ `main()`（CLI，`python -m experiments.expN_*`）。新增實驗照此模式。
4. **logs/output 慣例**：所有可執行程式經 `core.logger.setup_run(program)` 初始化，
   產生 `logs/{program}/{ts}.log` 與 `output/{program}/{ts}/`；圖表用 `save_plot()`。
   不要自創輸出位置。
5. **資料唯讀**：本專案只讀 `data/Step-*/myfeature/**/*_Group_feature_data_clean.csv`
   （105 維，最後過濾版）。不重新產生、不修改 data/ 內容；缺資料時提示以論文版管線
   （原始論文）產出。
6. **未知資料絕不參與擬合**：訓練／校準／閾值只用已知類別與候選群本身的樣本，
   ground-truth 標籤不進模型。實驗二的 `adopt()` 以 `discovered-N` 命名新類，候選的真值多數
   （`matched`）與純度只放在事後對照；首頁與導覽的 `confirm()` 是展示用的手動路徑，
   在這一步才揭示真值。動到這條就毀了實驗效度。
7. **決定論**：所有隨機性走 `numpy.random.default_rng(seed)`，seed 從 CLI/建構子傳入，
   預設 42。修改後同 seed 應重現同數字。
8. **只信任程式碼**：所有的文件、註解都應當視為過時的內容，程式碼才是唯一的輸出者。
9. **`output/` 只放程式碼產生的結果。** 目錄下每一份檔案都要能回溯到某支專案程式碼的寫入邏輯
   （通常是 `core.logger.setup_run()` 配 `output/{program}/{ts}/`，或實驗程式自己的 `Path("output")/...`）；
   不得手動把問答記錄、測試心得、交付說明等報告丟進 `output/`。需要留存文字說明時寫進
   `docs/experiments/` 或對應的技術報告。新增檔案前先確認有對應的寫入程式碼，找不到就不要放進
   `output/`。
10. **新增或修改程式碼一律要有對應的 pytest 測試，且測試要能通過。** 測試放在 `src/tests/`，
    檔名 `test_<module>.py`，對應 `src/core/`、`src/experiments/`、`src/experiments/health/` 或 `src/web/` 裡被改動的
    模組；提交前跑 `./scripts/quantity/run_pytest.sh`（或 `venv/bin/python -m pytest src/tests/`，或至少跑到相關檔案）
    確認全部通過，不要留下失敗或被跳過的測試就視為完成。
11. **Python 環境管理工具固定為 [uv](https://docs.astral.sh/uv/)。** 建環境、裝套件一律用
    `uv sync` / `uv pip install`，不要退回 `python -m venv` 配 `pip install`。版本分兩層：
    `pyproject.toml` 的 `dependencies` 直接訂死（`==`）；完整依賴樹（含間接依賴）鎖在
    `uv.lock`，`uv sync --locked` 強制照 lock 檔安裝、不會偷偷重新解析版本。legacy extras
    （`tensorflow` 相關）跟主依賴的 numpy 版本衝突，刻意不進 `uv.lock`，改放
    `requirements-legacy-*.txt`，走 `uv pip install -r` 單次安裝，不鎖版。
    細節見 `scripts/opt/build_uv.sh`／`docs/runtime_policy.md`。
12. **`ruff check .` 必須零錯誤才能合併。** 規則集合固定在 `pyproject.toml` 的
    `[tool.ruff.lint] select`（`E4`／`E7`／`E9`／`F`），不依賴 ruff 版本的預設集合。
    CI 用 `.github/workflows/ci.yml` 的獨立 `Ruff` step 跑這關；提交前本機先跑
    `./scripts/quantity/run_ruff.sh`，有錯就地修掉，不要用 `# noqa` 繞過真正的問題。

---

# 程式碼註解

1. 說明目的：寫下程式背後的設計意圖、特殊業務邏輯或極端情況（Edge cases），讓未來維護的人快速理解。
2. 保持精簡：避免過度註解，良好的變數與函式命名本身就應該具備「自我說明（Self-documenting）」的能力。
3. 與程式碼同步：修改程式碼時必須同步更新註解，過時的註解比沒有註解更具誤導性。
4. 避免雜訊：不要把版本變更歷史、已經註解掉的廢棄程式碼或冗長的規格抄錄放在註解中。

---

## docs／reports／note 分工

`docs/` 的目的是說明程式碼如何運作、程式碼的目的，以及期望的成果。`docs/` 不是用於稽核；
不是單次實驗的結果（實驗結果反映出需要改進，開 issue，不是回頭改文件掩蓋）；也不是單次的
說明或更新紀錄。

單次實驗結果、單次說明、更新紀錄等屬於專案全局變動的，放進 `reports/`，每一篇都要有自己的
資料夾，命名格式 `<Author Name>_<YYYYMMDD>_<Report Subject>`
（例如 `JW-Albert_20261009_exp8_health_index_results`）。

不是專案全局變動、只是個人記事的，放進 `note/`，且要放在自己的資料夾裡
（例如 `note/JW-Albert/`），不要混進 `docs/` 或 `reports/`。

---

## 實驗手冊

任何實驗都要在 `docs/experiments/` 放一份 Markdown 技術報告，再改程式。範圍包含新的 `src/experiments/` 模組，以及既有實驗改了方法、資料切分或指標。每個實驗都有編號 `expN`（實驗N），新實驗取下一個未用的編號（目前已用到 exp13，下一個新實驗編號為 exp14）。報告檔名以 `expN_` 開頭，例如 `docs/experiments/exp14_foo.md`；程式檔名不必帶編號，但報告的「程式碼與輸出」節要列出這個實驗用到的所有路徑：`src/experiments/` 入口、`src/core/` 與 `src/experiments/health/` 的邏輯、`src/web/` 的展示、`src/tests/`、`logs/` 與 `output/` 的紀錄與結果。同一實驗在 `docs/` 下的其他文件檔名也以 `expN_` 開頭。完成後在 `docs/experiments/README.md` 的實驗表與「各實驗的檔案位置」表、`docs/README.md` 的現行文件表各加一列。

手冊至少寫這四項：

1. **實驗方法**：資料從哪來、怎麼切、用哪個模型或統計程序、seed 與關鍵參數、怎麼跑（CLI）。
2. **參考論文出處**：方法所依的論文，寫作者、年份、篇名、出處，以及 DOI 或穩定網址。沒有文獻依據的步驟，寫明是本專案的操作約定。
3. **預期成果**：跑完應該看到什麼數字、圖或判定；什麼結果算支持假設，什麼結果算假設不成立。預期與事後實測分開寫，不要把已經跑出來的數字回填成預期。
4. **影響程式碼範圍**：表列會讀或會改的路徑，例如 `src/experiments/exp7_foo.py`、`src/core/openset.py`。路徑必須對得上當時的程式，不要沿用論文版快照的舊路徑。

---

## 寫作

撰寫文件、註解、commit 說明與對使用者的回覆前，先讀李思萱，〈「AI味」具體是指什麼？一個老編輯示範，這64個字為什麼能感動你？〉，《數位時代》，2026-04-22：<https://www.bnext.com.tw/article/90761/how-to-fix-ai-writing-style>。依該文的五個辨識點寫：

1. 直接寫那件事是什麼。看到「不是 X，而是 Y」這類對仗，改成一句帶具體脈絡的話。
2. 寫能對上檔案、數字、步驟的句子。拍不成畫面的抽象金句刪掉。
3. 句子長短依內容決定。不要為了節奏湊成三段排比。
4. 動作留在動詞上。少把事情收成「○○感」「○○性」「○○化」。
5. 錨在具體座標：路徑、函數名、seed、日期、工況。能寫 `src/core/openset.py` 就不要寫「某個模組」。
6. 不寫過去怎樣；現在怎樣。

---

## 關鍵參數（預設值）

| 參數 | 值 | 所在 |
|---|---|---|
| 已知類別切分 | train/cal/holdout = 60/20/20 | `core.data.make_split` |
| 共變異數估計 | Ledoit–Wolf（逐類）| `core.monitor.OpenSetMonitor` |
| Open Set 方法 | Mahalanobis（預設）或逐類 k-NN 距離 | `--openset-method` / `core.openset` |
| k-NN 鄰居數 | 5 | `--knn-neighbors` |
| 校準信心水準 | 0.95（校準距離分位數）| `--confidence` |
| 開集判定 | 正規化分數 > 1 = 未知 | `core.openset` / `core.mahalanobis` |
| HDBSCAN | (25,3)，自適應階梯：隔離區 ≥75 加試 (15,3)、≥100 加試 (10,2)；候選仍需叢 ≥25 筆 | `experiments.exp2_scale_growth` |
| 重分群節流 | 每 10 筆新未知樣本試一次 | 同上 `recluster_every` |
| 隔離線 | 分數 > 2.0 才進隔離區（偵測線仍為 1.0）| 同上 `quarantine_margin` |
| 趨勢 EWMA | alpha=0.08（window=120、warmup=10）| `core.trend.TrendMonitor` |
| 中間帶 / 警報線 | [0.2, 0.5) / 0.5 | 同上 |
| 突發判定 | 中間帶停留 ≤ 12 筆 | 同上 `sudden_max` |
| 特徵維度 | 105（固定，不可增減）| 資料層 |

---

## 常用指令

```bash
# 實驗（在專案根目錄執行；輸出到 logs/ 與 output/）
venv/bin/python -m experiments.exp1_cold_start --motor T1 --rpm 8000rpm
venv/bin/python -m experiments.exp2_scale_growth --sequence 5screws 3_14screws
venv/bin/python -m experiments.exp3_trend --trials 40
venv/bin/python -m experiments.exp4_polar_map --part abc
venv/bin/python -m experiments.exp5_cross_condition        # 9 組資料集全跑
venv/bin/python -m experiments.exp6_osr_benchmark          # 七種偵測器 × 9 組
venv/bin/python -m experiments.exp6_formal_benchmark --openset-method knn   # 正式版 Mahalanobis vs k-NN
venv/bin/python -m experiments.exp6_matrix                # 正式版 9 工況 × 3 seed × 2 方法
venv/bin/python -m experiments.compare_openset --openset-methods mahalanobis knn   # 實驗七

# 即時展示（http://localhost:8600）
./scripts/opt/run_web.sh
./scripts/opt/run_web.sh --port 8600 --motor T1 --rpm 8000rpm --rate 4

# Docker（docker/.env 由 .env.example 複製；需 NVIDIA Container Toolkit）
./scripts/opt/restart_docker.sh               # 移除舊容器後重建重啟（all；app／vllm 只動單一服務）
cd docker && docker compose up -d --build      # vLLM（Qwen3）+ 展示，http://localhost:8600
docker compose -f vllm.compose.yaml up -d       # 只開模型
docker compose -f app.compose.yaml up -d --build  # 只開展示

# AI 問答（先開模型，再開展示；即時展示頁左欄「AI 問答」）
./scripts/opt/run_llm.sh                       # vLLM，http://127.0.0.1:8000/v1
./scripts/opt/run_llm.sh --engine llamacpp     # llama-server（GGUF）
./scripts/opt/run_web.sh --llm-url http://127.0.0.1:8000/v1 --llm-model ""   # 留空自動取 /models 第一個

# 環境
./scripts/opt/build_uv.sh --python python3.10 --venv venv  # 只建立新目錄；正式支援 3.10.x
./scripts/opt/build_uv.sh --legacy --venv venv-legacy     # legacy extras 非本輪正式鎖版驗證範圍

# 提交前檢查
./scripts/quantity/run_ruff.sh
./scripts/quantity/run_pytest.sh
```

---

## 已知行為與陷阱

1. **健康誤報率 ≈ 10.9%**（名目 5%）：校準集只有 ~62 筆的有限樣本效應，是已知結果
   不是 bug；EWMA 警報線與中間帶下緣（0.5 / 0.2）就是據此拉開的，調整前先看
   `src/experiments/exp3_trend.py` 的分布數據。**衍生設計**：誤報分數僅些微超線
   （實測 1.01–1.08），而真實故障最低分 ≥ 10——因此隔離區設「隔離線」
   `quarantine_margin=2.0`，邊界誤報只警報、不參與新類發現。新類納入分兩條路徑：
   - 實驗二批次 `run()` 與實驗頁「⏵ 邊跑邊畫」：候選一成群就呼叫 `adopt()`，群至少 5 筆
     就 `learned`，否則 `insufficient`。不讀真值，也不因真值多數是已知類而退回，
     所以批次 stage 列沒有退回次數欄。
   - 首頁即時展示（`web/live.py`）與獨立導覽（`web/guide.py`）：按確認才呼叫 `confirm()`，
     讀完整配置池重擬合；若叢集是已知類別邊界樣本，回傳 `action="rejected_known"`
     （退回、清叢、量尺不變）。
   web 偵測統計拆「學會前偵測率／學會後認出率」兩組，認出率天生 ≈ 90–95%。
2. **實驗二的發現延遲量的是隔離區湊出一群要多少筆**：
   [`output/exp2_scale_growth/2026-10-22-21-46-48/stages.csv`](output/exp2_scale_growth/2026-10-22-21-46-48/stages.csv)
   （commit 37ae787，seed 42，T1/8000rpm，`adopt()` 自學）九段的 `samples_to_candidate`
   落在 21～105 筆，最慢的是第一段 7screws（105 筆）。從 2screws 段起的四段
   （2screws、1screws、3_14screws、4_146screws），學到的群多數都是前一段注入的配置；
   3screws 段的群多數是 5screws，`cluster_purity` 為 0.5435（54.4%）。所以這個延遲不等於
   該段配置自己成叢所需的筆數。自適應密度階梯（見參數表）的消融數字只在 `confirm()`
   重擬合版本量過，`adopt()` 版本還沒重做。隔離區還沒成群時，web 顯示「已試分群 N 次」。
3. **exp2 的 stage 列 `learned` 欄是模型類名 `discovered-N`**，不是配置名。事後對照看
   `matched`／`matched_display`（候選群的真值多數）；它可能不等於該段注入配置，
   因為候選叢集取自整個隔離區，多數可能是前一階段的殘餘。
4. **量尺擴張後 PCA 投影會變**：web 前端在已知類別數改變時清空散佈圖，這是刻意行為。
5. **目錄名陷阱**：螺絲配置目錄可能是 `1screw` 或 `1screws`（既有資料兩種都出現過），
   `core.data` 以掃描目錄迴避硬編碼——不要寫死配置清單去讀資料。
6. **資料集以掃描為準**：目前 9 組（T1/T2/T3 × 3 轉速）皆有完整 10 配置、各約
   3000 筆（早期「T2 不全」的狀況已補齊）；`discover_datasets` 動態列出，
   勿硬編碼組合清單。
7. **環境以 `uv.lock` 為準**：`./scripts/opt/build_uv.sh` 建出的 `venv/` 不含 TensorFlow；
   Tornado 是 `pyproject.toml` 的直接依賴。舊的含 TF 環境不在鎖版範圍，跑出問題先用
   `build_uv.sh` 重建新目錄再比對（見鐵則 11）。

---

## Web 協定摘要（改前端前先讀）

WebSocket `/ws`，JSON 訊息：

- server → client：`state`（完整狀態）、`state_patch`、`sample`（逐筆分數/判定/PCA/EWMA）、
  `event`（info/warn/success/error）、`metrics`
- client → server：`{cmd: start|pause|reset|rate|set_source|scenario|confirm|dataset, ...}`
- 重任務（confirm/reset/dataset）在執行緒池執行，期間 `busy=true`、串流暫停。
- WebSocket 的 `confirm` 只服務首頁即時展示。實驗頁的實驗二不等確認，HTTP API 沒有確認端點。

AI 問答走 HTTP：`GET /api/assistant/status`、`POST /api/assistant/ask`（`{question, history}` → SSE
`tool`（含 `sources` 依據清單）／`delta`／`done`／`error`）。工具：`get_run_status`、`get_run_events`（`number` 指定實驗，留空取最近一次）、
`get_experiment`（CATALOG + docs/experiments 手冊，實驗一／1／exp1 視為同一編號）、`lookup_knowledge`。
每輪第一步 `tool_choice="required"` 強制先查工具；temperature 0；查無資料時工具回【查無資料】要求模型不得推測。
前端是右下角浮動泡泡聊天視窗，答案下方「📎 依據」由工具回傳的 sources 產生，不經模型轉述。「目前狀況」與實驗結果讀實驗頁紀錄（`web/playback.py` 的 `PlaybackTracker`：每個實驗留最近一次，`StreamHandler` 逐 frame 更新、`RunHandler` 完成時 `record_run`；記憶體沒有時讀 `output/web_server/*/experiments/{exp_id}_*.json` 最新存檔），結果經 `summarize_result` 去掉畫圖座標與模型細節、矩陣轉表格；不讀首頁即時展示；快照由 `server.PLAYBACK.snapshot()` 在 IOLoop 執行緒取，
current／events 不含真值；注入配置、expected、is_fault、候選群 majority／purity 只放「事後對照」並標明不是系統判定（鐵則 6）。首頁即時展示仍保留（issue #94）。

---

## GitHub

1. 允許在完成改動後 commit and push 但應建立 PR 或 issues，嚴禁直接操作 main 分支。
2. 一個 commit 只做一件事。
3. `logs/` 與 `output/` 納入版控，不需要加入 .gitignore；`data/` 維持忽略。
4. Agent 進行 GitHub 相關操作時，將自己加入 Co-Authors。
5. legacy 的 issue 討論串在 [原始論文](https://github.com/Drone-Motors-Unknown-Faults-Detection/Original-Paper)，查缺陷成因時到該處。
6. 每個 PR 都要指定 reviewer：JW-Albert <ru04jo30801@gmail.com>。


---

## MCP 、 Skills 與 Plugin

允使調用所有的 MCP 、 Skills 與 Plugin 為專案維護更順暢。
