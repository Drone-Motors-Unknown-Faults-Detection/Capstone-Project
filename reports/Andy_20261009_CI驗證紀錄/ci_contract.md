> #65 的CI歷史紀錄。作者／日期依 Git 原作：05zhi（Andy），2026-10-09；JW-Albert 的後續修改一併保留。來源為 [固定版本 791216c](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/791216cf5602c370091fa516264f1ce6aaad6ab1/docs/ci_contract.md)。下方保留當時敘述、日期標題、成功與失敗；相對 output 連結僅隨搬遷調整。這些舊命令、政策及進度不代表現行狀態，現在請依 [操作說明](../../docs/ci_contract.md)；本次沒有重跑歷史實驗。

# #27：無正式資料的 CI 契約

日期：2026-10-09。基線：main `0282209959183bf7b8654115b175a94aafa26f35`。
已重讀該版本 AGENT.md、#27 本文及留言、開放 PR。公開實作仍只有 PR #36；不修改其待決 Web／health 檔案，也不能確認作者未推送的工作。

## 範圍與事前驗收

本輪新增 workflow、fixture 與工程證據入口，不改任何模型或正式資料。
Windows／Ubuntu 使用 CPython 3.10、新 venv、同一份 runtime-constraints.txt；macOS 尚未驗證。
每個 job 執行 pip check、`python -m unittest discover -s tests -t .`、exp1～3 與既有重要 CLI help、文件相對目標檢查。
既有測試會用 `git show` 核對歷史程式，因此 checkout 保留完整歷史；這是 Git 依賴，非 ignored data 依賴。

新增 fixture 直接測 core.data 的掃描／多檔載入／合法 1screw 異名、exp1／3 的既有事件與資料角色、exp2 的確認前後狀態，以及真正本機 WebSocket 的 start／pause／reset／rate／錯誤 JSON／未知指令。
目前未知指令只回 state、任意 Origin 可連線，均記為現狀特徵化；**不構成安全通過**，#28 保持待修。
測試只寫 TemporaryDirectory，不讀正式 data。exp1～3 CLI 以 fixture 走載入與輸出；其結果只驗工程契約，不作研究成績。

CI permissions 只有 contents:read，使用 push／pull_request，不使用 pull_request_target、不讀 secrets、不部署服務。
官方 Actions 固定完整 commit SHA；pip cache 由 OS／Python／constraints SHA 區分，只快取下載套件。
成功與失敗都只上傳經去敏的測試摘要，不上傳正式 CSV、模型、全部 logs/output 或 private sidecar。
測試例外須保留類型、測試 ID、失敗狀態；私人路徑和任意例外 payload 不公開。

在獨立測試分支先提交故意失敗 fixture，等真實 Actions 紅燈；下一 commit 移除，再等綠燈。
故障 fixture 不進正式 PR。保存兩次 SHA／run URL／job 結果，不用本機 log 代替。
若權限或服務阻擋，交付 PARTIAL/BLOCKED 與原始原因；不降低條件。
本輪不得自行合併 main；即使候選 CI 全通過，#27 仍待 main 整合後驗收。

## 基線與接續紀錄

工作樹為 `lineage_integration_20261008`，由乾淨 checkout 的最新 origin/main 建立 `delivery/ci-contracts-20261009`。
另一個 Lineage checkout 為 main `dda8910`，本輪唯讀，沒有 reset 或覆寫。
Python 沿用已安裝的 CPython 3.10.19 鎖版環境，不重新建置既有 venv。
既有 integration_evidence 的 `baseline` 模式只讀歷史 `64cb71d` 測試清單，本次實跑 76 項，不能當作最新全套；因此另用 discover 模式保存本輪完整基線。
原有約束、來源 ZIP、歷史結果、先前關閉的 #25／#29／#30 均不變。

## 官方設定依據

- [GitHub：Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)：設定 Python 與測試步驟。
- [GitHub：token 最小權限](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token)：顯式限制 permissions。
- [actions/setup-python](https://github.com/actions/setup-python)：python-version、pip cache-dependency-path；實際使用已存在的 v6 SHA。
- [Python 3.10 venv](https://docs.python.org/3.10/library/venv.html)：每個 job 建立新環境。
- [pip constraints](https://pip.pypa.io/en/stable/user_guide/#constraints-files)：constraints 控制版本，不自行安裝所有項目。
- [actions/upload-artifact](https://github.com/actions/upload-artifact)：只上傳指定去敏檔案。

2026-10-09 已查上述官方頁與 GitHub API 的 tag SHA。AGENT 要求的數位時代文章本次開啟回 Internal Error，未聲稱讀取全文；遵守 AGENT 已明列的五項具體寫作要求。

## 影響檔案

新增 `.github/workflows/ci.yml`、`tests/ci_evidence.py`、`tests/test_ci_contracts.py`；既有 fixture 視真正跨平台失敗作最小修補。
證據沿用 `logs/ci_evidence/`、`output/ci_evidence/`，手寫結果追加本文件。
本項沒有改實驗方法／切分／指標，不占 exp9～12。

## 實測與公開交付

事前契約 `d85cf83`；實作 `d2647f3`；[PR #53](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/pull/53)。
本輪基線 discover 實跑 179 項全通過；新增 14 項後本機 CPython 3.10.19 共 193 項全通過，pip check、七個 CLI help、8 份文件相對目標零失效。
第一次 fixture 關閉順序曾在測試後造成 WebSocket on_close 存取已復原的 HUB，雖 unittest 仍 OK，不能忽略背景例外；已修測試自身的斷線等待與 teardown 次序，再跑四個 WS 測試與全部193項。沒有修改 Web 實作。

- [候選 push 綠燈](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/actions/runs/37891076062)：`d2647f3`，Windows／Ubuntu 均成功。
- [候選 PR 綠燈](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/actions/runs/37891111826)：相同候選的 pull_request 事件，兩平台成功。
- [故意失敗紅燈](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/actions/runs/37891086977)：獨立分支 `8d804ace5b3568a61b965018e7ddbce4237821ea`，兩平台只因 `test_intentional_failure` 失敗；194項中新增故障fixture確實令exit非零。
- [下一 commit 移除後綠燈](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/actions/runs/37891379134)：`42476c3739a0a0bc83b58ce7c391ab65fd6d6db4`，兩平台成功。該分支沒有 PR，故障檔不在正式交付。

初版 workflow 只在失敗上傳摘要，因此初次成功run的逐項計數未保存為artifact；後續改成成功／失敗都只保存白名單摘要並在log列skip。不可為舊綠燈補造artifact。
遠端證據入口為 `python -m tests.ci_remote_evidence --run-id RUN_ID`；輸出 `output/ci_remote_evidence/*/remote_runs.json`，保留真實 head／run／job／artifact與摘要。
#27 候選工程條件已交付；尚未合 main，issue 維持 OPEN，macOS 未驗證。模型／準確率變動為0。

## main 整合後驗收（2026-10-09）

上述「尚未合 main」是候選交付當下狀態。遠端後續合併 PR #53 為 `68e4211324b1c9c547cfb59d0a827482e019c9a2`，PR #54 為 `22a253b14eb5df14058800f5a086bf4d7afeae58`；代理未執行合併。
[main 真實 CI](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/actions/runs/37891990633) 的兩平台均成功，完整 suite 為210項：Windows CPython 3.10.11 通過210項、skip 0；Ubuntu CPython 3.10.22 通過207項、skip 3（Windows 專用案例）。pip check、七個 CLI 與文件目標檢查均成功。正式政策支援 3.10.x，不把本機3.10.19寫成遠端實際版本。
證據位於 `output/ci_remote_evidence/2026-10-09-14-12-42/remote_runs.json`。artifact 同時含先前已版控摘要，必須以 `environment.git.head=22a253b14eb5df14058800f5a086bf4d7afeae58` 辨識本次 main 結果，不能把舊193項摘要再算一次。
本機相同程式樹 `85734ae` 的再驗為210項、207通過、3項因 symlink 權限 skip、0失敗；原始限制保留。main 測試已實際覆蓋 Windows symlink。
#27 的列明工程驗收與 main 整合已符合，可依授權關閉；任意 Origin 的現況仍留 #28，未宣稱 Web 安全完成。

## 改用 pytest、CI 只跑 Ubuntu（2026-10-10）

部署主機固定 Ubuntu，`.github/workflows/ci.yml` 的 `strategy.matrix.os` 移除 `windows-latest`，`contracts` job 直接 `runs-on: ubuntu-latest`；job 名稱、`upload-artifact` 的 `name` 去掉 `${{ matrix.os }}` 插值，固定寫 `ubuntu-latest`。原本因雙平台存在的 `if [ "$RUNNER_OS" = 'Windows' ]` 分支一併移除，建環境步驟只留 POSIX 路徑。Windows 本機安裝入口 `build_uv.ps1` 與 `docs/runtime_policy.md` 的 Windows 段落不受影響，只是不再由 CI 驗證。

`tests/ci_evidence.py` 的測試步驟由 `python -m unittest discover -s tests -t .` 改成 `python -m pytest tests -ra --tb=short`；`execute()` 的輸出解析新增 pytest 摘要列（`X passed, Y failed, Z skipped, W error(s)`）與 `^FAILED `/`^ERROR ` 失敗 ID 的解析，原本 unittest 格式（`Ran N tests in`／`FAILED (failures=…)`）的解析保留不動，`tests/test_ci_contracts.py::EvidencePrivacyTests` 兩項既有測試不必改。`pyproject.toml` 新增 `[project.optional-dependencies] test = ["pytest>=9.1,<10"]`；`runtime-constraints.txt` 固定 `pytest==9.1.1` 與其在 3.10 下的直接依賴（`pluggy`、`iniconfig`、`pygments`、`exceptiongroup`、`tomli`、`typing-extensions`）。CI 的建環境步驟在安裝完 `.` 之後另外 `pip install -c runtime-constraints.txt pytest`。

本機 `venv/bin/python -m tests.ci_evidence` 實跑：pytest 蒐集 233 項、232 passed（含 subtests）、skipped 3（Windows 專用案例，pytest 下用 `pytest.mark.skip`／`unittest.skipIf` 效果相同）、1 failed——`tests/test_runtime_environment.py::RuntimeEnvironmentTests::test_active_dependency_closure_is_pinned`，原因是本機既有 venv（AGENT.md 已知「超集」venv）裝的 `cloudpickle` 缺 `importlib.metadata` 紀錄，屬本機舊 venv 殘留、非這次改動造成；全新 `build_uv.sh` 建的環境會重新安裝 `cloudpickle` 並帶正確 metadata，預期不重現。這項失敗與本輪 pytest／Ubuntu-only 改動無關，本文件只如實記錄，不據此宣稱 CI 綠燈，仍待下一次真實 Actions 執行確認。

