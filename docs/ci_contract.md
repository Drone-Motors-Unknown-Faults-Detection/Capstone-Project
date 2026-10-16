# CI 使用與輸出契約

本頁說明 `.github/workflows/ci.yml` 與 `tests/ci_evidence.py` 的操作、輸出及限制。各次 Actions 的受測 SHA 與結果另外保存，本頁不維護單次驗收報告。

## 執行流程

push／pull_request 觸發 Ubuntu、CPython 3.10 的 contracts job。checkout 取完整 Git 歷史供固定舊版回歸使用；測試不依賴被忽略的正式 data。workflow 用 uv sync --locked 建立 venv，依 uv.lock 安裝專案與 test extras，再執行獨立步驟：

```bash
./run_pytest.sh -ra --tb=short
./run_ruff.sh
venv/bin/python -m tests.ci_evidence
```

`tests.ci_evidence` 只執行 uv pip check、七個 CLI 的 --help，以及選定文件的相對目標檢查，不在其中重跑 pytest／ruff。CLI 清單為 exp1、exp2、exp3、exp4、compare_openset、exp6_formal_benchmark、web.server，尚未涵蓋所有實驗。正式資料不能拿來補 fixture。本機先依 [環境操作](runtime_policy.md) 安裝 test 依賴。

## 輸出與判定

setup_run("ci_evidence") 寫入 logs/ci_evidence/{ts}.log、output/ci_evidence/{ts}/environment.json 及 public_summary.json。摘要保存受測 HEAD／dirty、版本、命令退出碼、測試數、失敗 ID 與文件失效目標；不公開任意例外 payload、token 或私人絕對路徑。

任何命令非零或文件目標失效，CLI 回傳非零。public_summary 的 PASS 只涵蓋該入口，不證明獨立 pytest／ruff 步驟通過；須另看 Actions 的兩個步驟。pytest 的 skip 不會自動使 pytest 退出碼失敗，缺權限案例須列未驗證，不寫成全部完成。

Actions 成功或失敗都上傳白名單 public_summary.json，保存14天。artifact 可能含 checkout 內先前已版控摘要，必須核對每份 environment.git.head，不能把舊摘要當成當次測試。

## 支援邊界與預期成果

現行 CI 只有 Ubuntu，不含 Windows／macOS；Windows 安裝入口仍保留，需另附實機測試。workflow 只有 contents:read，不用 secrets、不部署。WebSocket 任意 Origin 測試描述既有不安全行為，缺口仍由 [#28](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/28) 追蹤。

預期成果是可重現的工程測試及明確失敗訊號，不是模型準確率或部署可靠性證明。歷史紅燈／綠燈與整合紀錄見 [CI 驗證紀錄](../reports/Andy_20261009_CI驗證紀錄/ci_contract.md)，可對照 [搬遷前固定版本](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/791216cf5602c370091fa516264f1ce6aaad6ab1/docs/ci_contract.md)，相關 logs/output 保留。
