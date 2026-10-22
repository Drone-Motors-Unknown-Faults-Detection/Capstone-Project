# Python 環境操作

正式安裝要求 CPython 3.10.x（見 `pyproject.toml`）。環境管理使用 uv，直接依賴版本訂死於 pyproject.toml，完整依賴樹由 `uv.lock` 鎖定。Python 3.14 與 legacy 擴充不在此正式驗證範圍。

## 安裝與檢查

先安裝 uv，指定實際 Python executable，目的環境目錄必須不存在。入口拒絕刪除或覆寫既有環境；失敗目錄保留供診斷。

```bash
./scripts/opt/build_uv.sh --python python3.10
uv run --no-project --no-sync --python venv/bin/python python -m pytest src/tests -ra --tb=short
uv run --no-project --no-sync --python venv/bin/python python -m ruff check .
uv pip check --python venv/bin/python
```

`scripts/opt/build_uv.sh` 先回到專案根目錄，設定 `UV_PROJECT_ENVIRONMENT`，執行 `uv sync --python <Python> --extra test --locked`。test extras 包含 pytest、ruff；lock 與專案宣告不一致時失敗，不在安裝途中重新解析版本。目標環境不需安裝 pip／setuptools／wheel，依賴檢查使用 `uv pip check`。

`--legacy` 在標準環境安裝後，用 `uv pip install -r requirements-legacy-linux.txt` 單次解析，不受 uv.lock 管理；TensorFlow 與 numpy 的相容性須另驗。沒有 macOS 或 Windows 安裝入口。

若使用預設 `venv` 目錄，也可執行 `./scripts/quantity/run_pytest.sh` 與 `./scripts/quantity/run_ruff.sh`。兩者先回到專案根目錄，再呼叫 `venv/bin/python`。

## 執行環境紀錄

`core.runtime_environment.collect_environment()` 收集 Python、OS、架構、主機名稱、numpy 連結的 BLAS／LAPACK（`numeric`）、套件版本、Git HEAD／tracked dirty 與鎖版 SHA。`constraints_sha256` 沿用欄位名稱，現行值取自 uv.lock；不保存環境變數、token、使用者名稱或絕對 executable／資料路徑。Git 無法讀取時記 UNKNOWN。

`core.logger.setup_run()` 將紀錄寫入該 run 的 `environment.json`；`make_output=False` 則寫 log。安裝入口呼叫 `python -m core.runtime_environment`，寫入 `logs/environment_install/` 與 `output/environment_install/`。健康 CLI（`experiments.health_index_benchmark`、`health_index_matrix`、`health_monitor`、`health_index_aggregate`）都經 `setup_run` 寫 sidecar；直接呼叫、未經 `setup_run` 初始化的 API 不會自動保存 sidecar。

## 執行主機

提交前檢查（pytest、ruff）與所有實驗 CLI（會寫 `logs/`、`output/` 的程式）只在 Aero 或 Spark 執行（AGENT.md 鐵則 13）。兩台都是 Linux，numpy 連結 scipy-openblas 0.3.29：Aero 為 x86_64、Spark 為 aarch64。實驗二 seed 42 在兩台的 `stages.csv` 逐位元組相同；`summary.json` 裡的門檻、PCA 座標等連續值只在第 13 位有效數字之後不同（`output/exp2_scale_growth/2026-10-23-00-05-08` 對 `2026-10-23-00-05-22`）。macOS 的 numpy 連結 Apple Accelerate，同 seed、同套件版本，實驗二從第 4 段起分岔（#224），所以本機 Mac 的結果不作為文件或報告的數字來源。`environment.json` 的 `os.host` 與 `numeric.blas` 用來核對一份輸出是在哪台、用哪個 BLAS 跑的。

前端測試（`test_ai_chat_frontend.py`、`test_guide_frontend.py`、`test_markdown_frontend.py`、`test_assistant_presentation.py`、`test_result_presentation.py`、`test_web_experiment_alignment.py`，以及 `test_assistant.py` 的 HTTP→前端案例）以 `shutil.which("node")` 找 Node.js，缺少時判失敗、不當通過。兩台主機以使用者權限安裝 Node.js 官方 LTS tarball，不用系統套件：

```bash
V=v24.22.1; A=x64            # Aero 用 x64，Spark 用 arm64
F=node-$V-linux-$A.tar.xz
curl -fsSLO https://nodejs.org/dist/$V/$F
curl -fsSLO https://nodejs.org/dist/$V/SHASUMS256.txt
grep " $F$" SHASUMS256.txt | sha256sum -c -
mkdir -p ~/.local/lib/nodejs ~/.local/bin
tar -xJf $F -C ~/.local/lib/nodejs
for b in node npm npx; do ln -sfn ~/.local/lib/nodejs/node-$V-linux-$A/bin/$b ~/.local/bin/$b; done
```

`ssh aero '<指令>'` 這類非互動 shell 的 `PATH` 不含 `~/.local/bin`（uv 也裝在這裡），提交前檢查要帶上：

```bash
ssh aero 'cd ~/cc/Capstone-Project && PATH=$HOME/.local/bin:$PATH ./scripts/quantity/run_pytest.sh'
```

升級 Node 時改 `V` 重跑上面步驟，再跑完整 pytest；舊版目錄留在 `~/.local/lib/nodejs/` 不影響新連結。

## 更新與支援邊界

版本變更須另驗依賴、完整 pytest、ruff、CLI 與數值回歸後交付 PR，不原地升級研究環境。現行 [CI](ci_contract.md) 只跑 Ubuntu。skip 須分開回報。安裝成功也不證明資料獨立或模型可靠。

安裝成功不代替當前環境的 pytest、ruff 與 CLI 檢查。
