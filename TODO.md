# TODO

專題目標（未知故障辨識、性能指標、視覺化）對照現況後的待辦清單，加上 repo 其他 open issues。2026-10-02 建立，細節見各 issue。
依 [AGENT.md](AGENT.md)「實驗手冊」規定，每一項都要先在 `docs/experiments/` 寫手冊、在 `docs/README.md` 加一列，再改程式。

2026-10-09 主線收尾：PR #40/#41 已合併並在 main 重測，#29/#25 已 completed；[#22 固定研究證據](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/e7ee971d2ed41d5ce80e3a56f912eb4dc1c038dc/reports/issue_delivery_20261005/hard600_closeout.md) 已確認，有限研究交付完成，未得到可靠提升。見 [主線驗收紀錄](reports/Andy_20261009_主線交付/execution_log.md)。其他 issue 不能由這兩個 PR 推定完成。

## 1. 未知故障辨識與遷移學習

已有：只用健康資料的冷啟動偵測（`exp1_cold_start`）、HDBSCAN 新故障發現（`exp2_scale_growth`）、七種單類偵測器比較（`exp6_osr_benchmark`，含 PCA 重建 `src/core/detectors.py:127`）。

- [x] [#13](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/13) 實驗九：非線性 AutoEncoder 偵測器
  - [x] 手冊 `docs/experiments/exp9_autoencoder.md`
  - [x] `src/core/detectors.py` 新增 `MLPAutoencoderDet`（sklearn `MLPRegressor`，不加依賴），加進 `ALL_DETECTORS`
  - [x] `exp6_osr_benchmark` 9 工況比較；`src/tests/test_exp9_autoencoder.py` 加 seed 重現檢查
  - 實測：AUROC 全頂滿 1.0，健康誤報 5.8%，八種方法裡最低，但沒有展現非線性優勢（線性方法已經打滿 AUROC）
- [x] [#14](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/14) 實驗十：遷移學習
  - [x] 手冊 `docs/experiments/exp10_transfer.md`：72 組 `(src,dst)` 配對、目標端健康樣本數 0/10/25/50
  - [x] 中心平移（本專案操作約定，CORAL 簡化版）＋目標端從零冷啟動兩種方法
  - [x] 對照：直接搬移（AUROC 0.893）vs 中心平移（AUROC 0.999+，健康接受僅 9.3%）vs 從零冷啟動（AUROC 1.0，健康接受 87.9%）
  - 結論：中心平移修好排序、沒修好校準；目標端有 10 筆以上健康資料時，從零冷啟動全面優於遷移，支持逐工況冷啟動架構

## 2. 性能指標：相對原始論文提升 5%~10%

已有：`docs/Mahalanobis_Improvement.md`（歷史文件，腳本在原始論文）開集 Accuracy 0.9509 → 0.9864（+3.55 pp，未達標）、Balanced Accuracy +6.51 pp、F1 未記錄。

- [x] [#15](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/15) 實驗十一：原始論文對照實驗
  - [x] 手冊 `docs/experiments/exp11_ancestor_comparison.md`，先寫定主指標與達標門檻（Balanced Accuracy、macro-F1；絕對提升 ≥5 pp）
  - [x] legacy 邏輯沿用既有 `src/core/mahalanobis.py`（已依鐵則 1 複製並註記來源，未新增）
  - [x] 原始論文協定（已知 8/1/2/3/4screws、未知 5/6/7/3_14/4_146screws）＋冷啟動協定
  - [x] 9 工況 × 3 seed（42/123/2026），輸出 Accuracy、Balanced Accuracy、macro-F1、unknown F1、Wilcoxon 配對檢定
  - [x] 手冊「注意」段落寫明切分方式，不和 [#9](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/9) 的 campaign-group 結果混用
  - [x] 結果寫進 README「原創貢獻總覽」
  - 實測：Protocol A（原始論文協定）Balanced Accuracy 0.1667→0.9505（+78.38pp）、macro-F1 0.1512→0.9730（+82.18pp），Wilcoxon 27 對配對 p≈0，遠超 5pp 門檻。根因：legacy 用單一全域共變異數，`predict_known_class` 幾乎找不到已知類別，結構性無法做多類判別，不只是校準鬆緊的差異

## 3. 視覺化：混淆矩陣與 t-SNE

已有：PCA 投影（`src/core/monitor.py:90`、Web）、`polar_map.png`、`detect_rates.png`、`cross_condition.png`、`osr_benchmark.png`。

- [x] [#16](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/16) 實驗十二：混淆矩陣與 t-SNE
  - [x] 手冊 `docs/experiments/exp12_confusion_tsne.md`
  - [x] 混淆矩陣：健康 vs 故障（exp1 邏輯重做）、開集多類（原始論文協定 5 已知 + unknown）
  - [x] t-SNE：`monitor.scaler` 縮放空間、顏色 = 螺絲配置、標記 = 系統判定，9 工況 3×3 網格
  - [x] 經 `core.logger.setup_run` / `save_plot()` 輸出；README「原創貢獻總覽」引用
  - [x] #15 完成後加 legacy vs LW 混淆矩陣並排圖
  - 實測：二元混淆矩陣健康誤報 7.87%、故障偵測 100%；開集多類混淆矩陣 legacy 的 5 個已知類別 100% 全部判成 unknown（對角線全 0），ledoit_wolf 已知類別之間完全沒有互相混淆——把 #15 的數字變成可見的畫面

## 4. 資料來源與分支文件

兩項都屬於 `research-improvements-20260920` 分支；#11 文件已在研究分支完成，尚未併入 main。#9 的資料來源與可靠度缺口仍開啟。2026-10-09 exp9–exp12（#13–#16）的 PR 不處理 #9：待辦清單裡的實體馬達/session/run 身分、每類獨立 test 擷取群組兩項，研究分支自己的文件已記成「UNKNOWN／未通過」，需要新硬體採集才能解決；跨階段 105 維特徵語意稽核與環境驗證理論上可以單靠寫程式/文件完成，但審查標準和「新增四個實驗」不同，留給未來一個專門處理 #9 的 PR，不混在一起。

- [ ] [#9](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/9) 故障配置評估：獨立資料來源與可靠度未解
  - 現況：campaign-group 切分下 N=5 已知準確率 25.744%，unknown AUROC 0.528 / 0.524（Mahalanobis / k-NN），未證明能拒絕未見配置
  - [ ] 核對實體馬達 / session / run 身分與原始訊號區間、stride
  - [ ] 每個評估配置至少兩個獨立測試擷取群組（不拿 calibration 頂替 test，不把 RPM 檔案當獨立馬達）
  - [ ] 稽核跨階段 105 維特徵語意與感測器方向（Stage2 為重建）
  - [ ] 預先登錄新的 train/validation 比較與新的 final test，不回頭調這次的 test
  - [ ] 另外驗證宣告的 Python 3.10 環境（本次以 3.14.6 執行）
- [x] [#11](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/11) `research-improvements-20260920` 分支文件
  - [x] 在 `docs/` 新增一份 md 說明這個分支做了什麼
  - [x] 更新 `README.md`、`docs/README.md`
  - 已push研究分支 commit `2f15c797d2adb43b0103629783c7dd4baba03138`；見 [分支說明](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/2f15c797d2adb43b0103629783c7dd4baba03138/docs/research_improvements_20260920.md)。

## 5. Andy：持續研究與分支文件（2026-10-02）

工作分支：`research-improvements-20260920`。依 main 的 AGENT.md（blob `f3b92347582bceff98ede6513f06458cdefdeac5`）先寫實驗手冊，再修改方法。相關來源問題保留在 [#9](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/9)。

- [x] [#22](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/22) 持續研究：有限比較與失敗分析已交付（2026-10-06 closed；研究程式未整批合入 main）
  - [x] 完成 [#11](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/11)：分支總覽、README 與 docs 索引；補充同步 raw schema 的實際用途與限制。
  - [x] 補齊現行 continuous_research 各模組手冊；已有程式的手冊註明補寫日期，後續改動先寫手冊。
  - [x] 核對已封存 Q01–Q08 配對評估：72 格預定；12 格因訓練最佳化未收斂而 INCOMPLETE，不補零、不換參數填回。
  - [x] 重推論、重算指標，核對來源 SHA、fit/cal/test IDs 與健康總誤報；保存全部成功與失敗格。
  - [x] 封存本批結果、雙 Python 測試與 D 槽備份；記錄相對 C02/C17/C24/D01 的配對差異。
  - [x] 依未收斂／跨馬達失敗證據，先登錄下一批有限改編，再實作與消融；保留 formal105、LW、k-NN factory 與 PolarMap 正式預設。
  - 已有 28,910 筆資料全部有 test 曝露歷史。新比較皆為 exploratory；來源未知與獨立 test groups 不足不改成 PASS。此項不依賴重新採集，也不宣稱可靠模型已完成。
  - 2026-10-03接續：Q 60格／576,144筆重推論與配對報告完成，六完整方法均未通過可靠性門檻；原12格INCOMPLETE保留。
  - train-only solver診斷27fits：hard150 3/9、smooth150 5/9、hard600 9/9收斂；27loss/gradient重算及原Q三成功weights exact。兩Python各396tests/37CLI/pip check PASS。
  - [x] hard600 subset/all-train k-NN outer 已實際執行 E02/E04 並封存；seed0 三 motor 等權 fault accuracy 分別下降 3.507／3.655 個百分點，T1 unknown recall 仍為 0%。[固定收尾證據](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/e7ee971d2ed41d5ce80e3a56f912eb4dc1c038dc/reports/issue_delivery_20261005/hard600_closeout.md)。
  - 原 Q／solver 結果保留：[Q報告](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/8bd6d14042e61a40548e37e989a1f2db2d7a4fbb/reports/continuous_research/final_findings.md)、[solver報告](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/8bd6d14042e61a40548e37e989a1f2db2d7a4fbb/reports/continuous_research/solver_findings.md)。#22 已因有限研究交付關閉；沒有通過可靠性規則，不關閉 #9，不整批合併研究分支。

## 6. 文件：`src/experiments/health/` 與 `reports/`

- [x] [#21](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/21) 釐清 health 與 reports 內容
  - [x] `docs/experiments/exp8_health.md`：`src/experiments/health/` 各模組與呼叫端、`src/core/trend.py` 與 `src/experiments/health/trajectory.py` 兩套趨勢邏輯的差別、`reports/` 各子目錄的來源與現況
  - [x] 更新 `docs/README.md`、`README.md`、`AGENT.md` 目錄說明
  - [x] 2026-10-03 `health/` 搬到 `src/experiments/health/`；`reports/` 只保留 `exp8_health_index_results/`，稽核紀錄刪除（可從 commit `80bdf54` 取回）
  - 文件記下的不一致已另開 issue：[#24](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/24) health CLI 未走 `setup_run`、[#25](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/25) 健康指數彙總檔沒有產生程式、[#26](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/26)–[#29](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/29) 抽查仍成立的稽核發現、[#30](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/30) 重驗其餘 17 項。
- [x] 實驗編號：`compare_openset` 定為實驗七、健康指數三支定為實驗八；規劃中的 #13–#16 順延為實驗九～十二。相關文件改名為 `expN_` 開頭，`docs/experiments/README.md` 加各實驗檔案位置表。程式檔名不變。

## 7. 主線工程驗收（2026-10-09）

- [x] [#29](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/29)：PR #40 已合併 `c68b7cdbb5bddf1034a9e713c8e8efd5acc61c87`，main 124 項與 11 項 guard 驗證通過；未擬合／失敗重擬合有明確 RuntimeError，成功擬合分數及投影不變。
- [x] [#25](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/25)：PR #41 已合併 `ae53b92eea7dd68cd7ffefe8d5c24b3f7157428d`，main 162 項通過；六 run／54 列可重算，349 比對有 2 精度差異，原公式 UNKNOWN。依「有差異回報」驗收關單，未改歷史彙總。
- [x] [#30](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues/30)：PR #50 已合 main `ebb702d`，主線 162 項測試與文件驗證通過後 completed；七個去重追蹤 #43–#49 已發布，R5/R6/R9 已補 #27/#28/#26；[發布證據](reports/Andy_20261009_主線交付/README.md#已發布追蹤與去重補充)。這些後續程式修補仍未完成。
- #19 尚缺原參考表／合法全文；#9 仍缺獨立新資料與採集證據；#24/#28/#35/#37 原首頁部分等待 Albert PR #36 定案。測試通過不消除這些限制。
