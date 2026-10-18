# 給老師：2026-10-09 主線交付與剩餘問題

本文記錄截至main 3ceea37的2026-10-09交付；狀態與環境政策只適用此版本。現行安裝以[環境政策](../../docs/runtime_policy.md)為準，最新issue狀態由GitHub回讀。

## 本輪已確認

| 項目 | 程式／PR | main 與 issue | 可以支持的結論 |
|---|---|---|---|
| 未擬合防護 | PR #40；core/monitor.py | main `c68b7cdbb5bddf1034a9e713c8e8efd5acc61c87`；#29 completed | 沒有擬合或重擬合失敗時，推論有明確 RuntimeError；成功擬合分數與原版相同 |
| 健康指數封存彙總 | PR #41；experiments/health_index_aggregate.py | main `ae53b92eea7dd68cd7ffefe8d5c24b3f7157428d`；#25 completed | 六 run／54 列可重算；349 欄中 347 吻合、2 精度差異明列 |
| hard600 subset/all-train | 研究分支既有 E02/E04，非本輪新訓練 | #22 於 2026-10-06 關閉；研究程式未整批合入 main | 固定 seed0 的 fault accuracy 下降 3.507／3.655 個百分點；未找到可靠提升 |
| 稽核追蹤 | PR #50；七個新 issue 與三則去重補充 | main `ebb702d` 驗證後 #30 completed | 17 項現況、限制與負責議題可追溯；缺陷未因此修復 |
| 環境工程第一階段 | PR #51；Python 3.10.x、單一 constraints、非破壞安裝與環境 sidecar | main `3ceea37`；#26 仍 OPEN | 乾淨 Windows 環境可安裝、版本可回讀；三支 health CLI 覆蓋尚未完成 |

主線最後一次 Python 3.10.19：179 passed，0 failed/error/skipped，pip check 與四個 CLI help 成功；本輪新增17項環境 fixture。guard 另有 11 項與 8 組等價比較；先前導覽 Mahalanobis/k-NN 各 595 筆、0 mismatch，本階段未重新跑正式資料導覽。詳見 [命令、commit 與產物](execution_log.md)、[當時環境驗收](#當時環境驗收)。本階段沒有新的準確率結果。

## 仍未通過或無法確認

- **FAILED**：hard600 沒有可靠提升；不能把最佳化收斂當分類改善。其 [固定研究證據](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/e7ee971d2ed41d5ce80e3a56f912eb4dc1c038dc/reports/issue_delivery_20261005/hard600_closeout.md) 與負面結果保留。
- **DIFFERENT**：兩個健康指數方法差值欄為舊 0.018513、新 0.018512；容差 0.0000005。先四捨五入可解釋舊值，原生成程式缺失，歷史操作 **UNKNOWN**。#25 要求回報差異，不要求零差異，因此可關工程 issue。
- **UNKNOWN／INCOMPLETE**：raw/session/run、原始視窗／刪點／同步證據、獨立新採集與 fresh final 仍不足。T1/T2/T3 是三顆馬達，不能拼成一顆退化生命週期。#9 不關閉。
- #19 尚未核實全部原文引用；#30 已 completed，追蹤已發布 [#43–#49 與去重補充](#已發布追蹤與去重補充)，各缺陷仍 open；#26 依賴政策已合 main，每支入口覆蓋未完成；#27 CI 尚未完成。#13–16 須另核完整協定與驗收。
- #24/#28/#35/#37 原首頁入口與 Albert PR #36 有同檔重疊，本輪未修改 health_monitor.py、server.py、index.html、experiments.js，未合併 PR #36。

七問背景與示範操作在 [實驗十三導覽](../../docs/navigation/exp13_README.md)；[實驗八完整重算說明](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/experiments/exp8_health_index_aggregate_delivery.md) 保存手冊、SHA、舊值／新值與限制。本輪證據僅支持工程交付與現有資料比較，不支持部署可靠度、RUL 或 fresh final。

## 文件驗證入口

`python -m tests.project_closeout_evidence` 只檢查本輪選定文件的相對檔案目標及來源 SHA，輸出到 `logs/project_closeout_evidence/`、`output/project_closeout_evidence/`；不載入正式資料，不驗證所有歷史錨點或遠端論文全文。完整回歸使用 `python -m tests.integration_evidence --phase candidate --data-root <既有唯讀資料根>`。

## 已發布追蹤與去重補充

當時基線4d1858b830890a761af1dc550e74010db8a4b5f5，對照4f783c6的科學／載入／Web／環境檔案未變；沿用17項反例，不冒稱全部重跑。逐項原判定見[前次交付](../Andy_20261008_議題交付/delivery.md#當次17項稽核與分組)及[固定發布全文](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/project_closeout_20261009/audit_followups.md)。來源SHA／函數跨度由tests.audit_followup_evidence保存；只有選定文件目標重新驗證。

追蹤規範1a9c59c15ab1dc31e9017a4c24c49d86fe863a5a已推送；七個issue及三則補充均已發布、未指派組員。各缺陷的驗收獨立於#30關單。

| 分組 | 實際追蹤 | 缺口 |
|---|---|---|
| R1 | [#43](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/43) | exp6 矩陣與彙總完整性 |
| R2 | [#44](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/44) | 正式105維資料規範 |
| R3 | [#45](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/45) | containment／不等長通道 |
| R4 | [#46](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/46) | 內容SHA與portable metadata |
| R7 | [#47](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/47) | resolved config／schema |
| R8 | [#48](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/48) | profiling後決定優化 |
| R10 | [#49](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/49) | 小步拆分orchestration |

不新建重複議題：R5 [補入 #27](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/27#issuecomment-6073814615)，R6 [補入 #28](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/28#issuecomment-6073814870)，R9 [補入 #26](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/26#issuecomment-6073815098)。每項包含最小重現、範圍與驗收；各缺陷仍未完成。


[當次PR36協調留言](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/pull/36#issuecomment-6073815390)請作者決定重整或關閉，不替作者決定。PR50合併與main驗證見[執行紀錄](execution_log.md)；[162項完整結果](../../output/branch_integration/2026-10-09-11-42-53/validation.json)、[16來源／15相對目標](../../output/audit_followup_evidence/2026-10-09-11-42-52/evidence.json)保存當時驗收範圍。

## 當時環境驗收

PR51第一階段使用3.10.x、直接依賴上下限及runtime-constraints.txt；這是該版本政策，不是現行uv.lock安裝說明。Windows PowerShell7及POSIX入口只建新目錄，既有環境保留；setup_run保存去敏environment.json，直接run API及三支health CLI覆蓋仍不完整。事前規範49fec06168bfb4eb2d921a0f37b50b812fc0d75d、實作815a548e29ee427387b33c14229feb1fe32efdb0均已推送。

第一次cp950安裝失敗、pip私有API失敗及cloudpickle間接依賴更正留在[原環境交付全文](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/project_closeout_20261009/runtime_delivery.md)及其固定logs/output。最終全新安裝[12-08-35證據](../../output/runtime_policy_evidence/2026-10-09-12-08-35/evidence.json)全部exit0、版本0差異；實際新環境[sidecar](../../output/environment_install/2026-10-09-12-14-19/environment.json)不與外層runner混用。

最終環境[179項完整fixture](../../output/branch_integration/2026-10-09-12-15-20/validation.json)、[版本核對](../../output/runtime_policy_evidence/2026-10-09-12-15-16/evidence.json)成功。文件檢查先因sidecar秒數筆誤[失敗](../../output/project_closeout_evidence/2026-10-09-12-15-57/evidence.json)，更正後[104目標／0失效](../../output/project_closeout_evidence/2026-10-09-12-16-11/evidence.json)，失敗證據不刪。

交付head26320e7e8b4935a14ccbdbdeb223453cd5e61d34合併為3ceea37ab1d4b52b7a414a74310eec8401b5b751；真正main再跑[179項](../../output/branch_integration/2026-10-09-12-19-13/validation.json)、[版本／17fixture](../../output/runtime_policy_evidence/2026-10-09-12-19-09/evidence.json)、[11guard與8組配對](../../output/monitor_guard_evidence/2026-10-09-12-19-41/evidence.json)、[7文件109目標](../../output/project_closeout_evidence/2026-10-09-12-19-43/evidence.json)均成功；sidecar HEAD為受測main、tracked_dirty=false。runner內phase仍叫candidate，以實際HEAD判定版本，不改封存欄位。

當時只驗Windows／3.10.19；Linux／macOS、legacy未完整驗收，#26保持OPEN。備份與第一次不完整失敗包由[backup manifest](backup_manifest.json)定位，未搬動ZIP或改歷史SHA。
