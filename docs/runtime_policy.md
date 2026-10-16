# Python 環境操作

正式安裝要求 CPython 3.10.x（見 `pyproject.toml`）。環境管理使用 uv，直接依賴版本訂死於 pyproject.toml，完整依賴樹由 `uv.lock` 鎖定。Python 3.14 與 legacy 擴充不在此正式驗證範圍。

## 安裝與檢查

先安裝 uv，指定實際 Python executable，目的環境目錄必須不存在。入口拒絕刪除或覆寫既有環境；失敗目錄保留供診斷。

```bash
./build_uv.sh --python python3.10 --venv .venv310
uv run --no-project --no-sync --python .venv310/bin/python python -m pytest tests -ra --tb=short
uv run --no-project --no-sync --python .venv310/bin/python python -m ruff check .
uv pip check --python .venv310/bin/python
```

Windows PowerShell 7：

```powershell
./build_uv.ps1 -Python 'C:/path/to/python3.10.exe' -VenvDir '.venv310'
uv run --no-project --no-sync --python .venv310/Scripts/python.exe python -m pytest tests -ra --tb=short
uv run --no-project --no-sync --python .venv310/Scripts/python.exe python -m ruff check .
uv pip check --python .venv310/Scripts/python.exe
```

`build_uv.sh`／`build_uv.ps1` 設定 `UV_PROJECT_ENVIRONMENT`，執行 `uv sync --python <Python> --extra test --locked`。test extras 包含 pytest、ruff；lock 與專案宣告不一致時失敗，不在安裝途中重新解析版本。目標環境不需安裝 pip／setuptools／wheel，依賴檢查使用 `uv pip check`。

`build_uv_mac.sh` 轉呼叫 legacy-mac 分支。legacy 清單位於 `requirements-legacy-linux.txt`／`requirements-legacy-mac.txt`，標準環境安裝後再用 `uv pip install -r` 單次解析，不受 uv.lock 管理；TensorFlow 與 numpy 的相容性及平台支援須另驗。

若使用預設 `venv` 目錄，也可執行 `./run_pytest.sh` 與 `./run_ruff.sh`。兩者固定呼叫 `venv/bin/python`，不會自動尋找 .venv310 或 Windows Scripts 路徑。

## 執行環境紀錄

`core.runtime_environment.collect_environment()` 收集 Python、OS、架構、套件版本、Git HEAD／tracked dirty 與鎖版 SHA。`constraints_sha256` 沿用欄位名稱，現行值取自 uv.lock；不保存環境變數、token、使用者名稱或絕對 executable／資料路徑。Git 無法讀取時記 UNKNOWN。

`core.logger.setup_run()` 將紀錄寫入該 run 的 `environment.json`；`make_output=False` 則寫 log。安裝入口呼叫 `python -m core.runtime_environment`，寫入 `logs/environment_install/` 與 `output/environment_install/`。直接呼叫未初始化的 API，或未使用 setup_run 的入口，沒有自動保存 sidecar 的保證；缺口見 [#24](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/24)。

## 更新與支援邊界

版本變更須另驗依賴、完整 pytest、ruff、CLI 與數值回歸後交付 PR，不原地升級研究環境。現行 [CI](ci_contract.md) 僅跑 Ubuntu，不能由其成功宣稱 Windows／macOS 全部已驗證；skip 須分開回報。安裝成功也不證明資料獨立或模型可靠。

安裝失敗、版本決策與當時實測見 [環境驗證紀錄](../reports/Andy_20261009_環境驗證紀錄/runtime_policy.md)，可對照 [搬遷前固定版本](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/runtime_policy.md)。歷史結果不能代替當前環境的檢查。
