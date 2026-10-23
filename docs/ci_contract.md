# CI 使用與輸出規則

`.github/workflows/ci.yml` 與 `src/tests/ci_evidence.py` 的操作、輸出及限制如下。

## 執行流程

push／pull_request 觸發 Ubuntu、CPython 3.10 的 contracts job。checkout 取完整 Git 歷史供固定舊版回歸使用；測試不依賴被忽略的正式 data。workflow 用 uv sync --locked 建立 venv，依 uv.lock 安裝專案與 test extras，再執行獨立步驟：

```bash
./scripts/quantity/run_pytest.sh -ra --tb=short
./scripts/quantity/run_ruff.sh
venv/bin/python -m tests.ci_evidence
```

`tests.ci_evidence` 只執行 uv pip check、七個 CLI 的 --help，以及選定文件的相對目標檢查，不在其中重跑 pytest／ruff。CLI 清單為 exp1、exp2、exp3、exp4、compare_openset、exp6_formal_benchmark、web.server，尚未涵蓋所有實驗。正式資料不能拿來補 fixture。本機先依 [環境操作](runtime_policy.md) 安裝 test 依賴。

## 輸出與判定

setup_run("ci_evidence") 寫入 logs/ci_evidence/{ts}.log、output/ci_evidence/{ts}/environment.json 及 public_summary.json。摘要保存受測 HEAD／dirty、版本、命令退出碼、測試數、失敗 ID 與文件失效目標；不公開任意例外 payload、token 或私人絕對路徑。

任何命令非零或文件目標失效，CLI 回傳非零。public_summary 的 PASS 只涵蓋該入口，不證明獨立 pytest／ruff 步驟通過；須另看 Actions 的兩個步驟。pytest 的 skip 不會自動使 pytest 退出碼失敗，缺權限案例須列未驗證，不寫成全部完成。

Actions 成功或失敗都上傳白名單 public_summary.json，保存14天。artifact 可能含 checkout 內先前已版控摘要，必須核對每份 environment.git.head，不能把舊摘要當成當次測試。

## 支援邊界與預期成果

現行 CI 只有 Ubuntu。安裝入口是 `scripts/opt/build_uv.sh`，沒有 Windows／macOS 腳本。workflow 只有 contents:read，不用 secrets、不部署。WebSocket 與實驗／AI 問答 HTTP 端點的 Origin 測試確認只接受同站或 `--allow-origin` 允許的來源（#265、#301）；Host 測試確認只接受綁定位址、localhost／127.0.0.1／[::1]、本機 IP 與 `--allow-host` 名稱，其他 Host 一律 403，擋 DNS rebinding（#304）。實驗執行、邊跑邊畫與 AI 問答這類會改變狀態的端點，請求若 Origin、Referer、Sec-Fetch-Site 三種來源標頭都沒有也回 403（#304 F5-02）；curl 與 `*_evidence.py` 腳本要自己帶同站 `Origin`。預設 `--bind-address 0.0.0.0` 仍在：[#28](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/28) 已關閉（NOT_PLANNED），系統只在封閉實驗室網路使用。

預期成果是可重現的工程測試及明確失敗訊號，不是模型準確率或部署可靠性證明。
