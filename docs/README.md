# docs/ 文件索引

先找操作說明：[安裝與版本紀錄](runtime_policy.md)、[CI執行與輸出](ci_contract.md)、[archive物化安全契約](formal_materialization_contract.md)、[健康監測模組](health_and_reports.md)、[導覽操作](navigation/exp24_README.md)。各頁說明程式用途、操作、預期輸出與限制。

當前缺口與交付狀態請查 [GitHub issues](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues) 及各 PR 的受測 SHA，不把歷史測試次數當成最新 main 驗收。

內容指紋、portable manifest 與本機 private sidecar 的操作見[內容來源契約](content_provenance_contract.md)；它只涵蓋正式入口，不代表所有 runner 都有完整來源紀錄。

匯入105維 CSV 前先讀[正式特徵載入契約](formal_feature_schema_contract.md)與[完整欄名清冊](formal_feature_schema.json)；清冊供維護者與測試比對，runtime 規則在 `core/feature_schema.py`。

## 歷史交付附件（不作現行操作手冊）

以下連到 reports/ 的原交付與失敗紀錄，只適用其記載日期、來源及受測版本。

- [整合來源與舊驗收](../reports/Andy_20261008_分支整合/integration_report.md)、[歷史manifest](../reports/Andy_20261008_分支整合/manifest.json)。
- [當時引用核對](../reports/Andy_20261008_議題交付/citation_followup.md)、[導覽驗收](../reports/Andy_20261008_議題交付/guide_acceptance.md)、[未擬合舊回歸](../reports/Andy_20261008_議題交付/execution_log.md#未擬合防護的原驗收邊界)。現行操作見[共用模型生命週期](experiments/README.md#共用模型生命週期)。
- [當時給老師的說明](../reports/Andy_20261009_主線交付/README.md)、[當時執行紀錄](../reports/Andy_20261009_主線交付/execution_log.md)。
- [封存彙總重算的原交付說明](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/1fa9431bb7b86959f29540d07b2b9290ab39ce42/docs/experiments/exp8_health_index_aggregate_delivery.md)保存該次差異與驗收；現行操作見下表重算手冊。
- [正式入口來源的原交付與失敗紀錄](../reports/Andy_20261009_正式入口來源/交付紀錄.md)，只適用其中固定版本。

老師要看程式如何操作，先用[導覽入口與示範腳本](navigation/exp24_README.md)；歷史研究分數只在固定來源下解讀，不當成新增盲測結果。

> 實驗的白話說明在 [實驗說明手冊](Experiments_Guide.md)。
> 對程式的技術報告在 [experiments/](experiments/README.md)。
> 下面的 Step 1–6、Tensorflow 與三份 Lineage 研究筆記是論文版管線的歷史快照。

歷史快照對應本 repo git 歷史 `dda8910`（2026-09-17 放回）。那些文件裡的 Step 1–6
程式、notebook 與 `scripts/` 現存於
[Ancestor repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor)，
**不對應**目前的 `core/ experiments/ web/`；每份快照頂部都有歷史標記。
新專案的說明見根目錄 [README.md](../README.md)。

## 現行實驗技術報告

[experiments/README.md](experiments/README.md)列出13個現行實驗編號（1～12及24）、19份方法／操作手冊及程式位置。實驗六、八的矩陣與彙總共用原編號；實驗九共用實驗六runner。每份手冊說明資料與用途切分、模型建立、CLI／API／Web操作、輸出及限制。[Experiments_Guide.md](Experiments_Guide.md)提供總覽，實際參數與例外以對應手冊及程式為準。

| 編號 | 文件 | 程式 |
|---|---|---|
| 實驗一 | [exp1_cold_start](experiments/exp1_cold_start.md) | `experiments/exp1_cold_start.py` |
| 實驗二 | [exp2_scale_growth](experiments/exp2_scale_growth.md) | `experiments/exp2_scale_growth.py` |
| 實驗三 | [exp3_trend](experiments/exp3_trend.md) | `experiments/exp3_trend.py` |
| 實驗四 | [exp4_polar_map](experiments/exp4_polar_map.md) | `experiments/exp4_polar_map.py` |
| 實驗五 | [exp5_cross_condition](experiments/exp5_cross_condition.md) | `experiments/exp5_cross_condition.py` |
| 實驗六 | [exp6_osr_benchmark](experiments/exp6_osr_benchmark.md) | `experiments/exp6_osr_benchmark.py` |
| 實驗六 | [exp6_formal_benchmark](experiments/exp6_formal_benchmark.md) | `experiments/exp6_formal_benchmark.py` |
| 實驗六 | [exp6_matrix](experiments/exp6_matrix.md) | `experiments/exp6_matrix.py` |
| 實驗六 | [exp6_aggregate](experiments/exp6_aggregate.md) | `experiments/aggregate_exp6.py` |
| 實驗七 | [exp7_compare_openset](experiments/exp7_compare_openset.md) | `experiments/compare_openset.py` |
| 實驗八 | [exp8_health_index_benchmark](experiments/exp8_health_index_benchmark.md) | `experiments/health_index_benchmark.py` |
| 實驗八 | [exp8_health_index_matrix](experiments/exp8_health_index_matrix.md) | `experiments/health_index_matrix.py` |
| 實驗八 | [封存彙總重算](experiments/exp8_health_index_aggregate.md) | `experiments/health_index_aggregate.py`；來源完整性、54 列重算定義及差異輸出 |
| 實驗八 | [exp8_health_monitor](experiments/exp8_health_monitor.md) | `experiments/health_monitor.py` |
| 實驗九 | [exp9_autoencoder](experiments/exp9_autoencoder.md) | `core/detectors.py`（`MLPAutoencoderDet`，`experiments/exp6_osr_benchmark.py` 自動納入） |
| 實驗十 | [exp10_transfer](experiments/exp10_transfer.md) | `experiments/exp10_transfer.py` |
| 實驗十一 | [exp11_ancestor_comparison](experiments/exp11_ancestor_comparison.md) | `experiments/exp11_ancestor_comparison.py` |
| 實驗十二 | [exp12_confusion_tsne](experiments/exp12_confusion_tsne.md) | `experiments/exp12_confusion_tsne.py` |
| 實驗二十四 | [exp24_experiment_navigation](experiments/exp24_experiment_navigation.md) | `experiments/navigation_data_contract.py`、`experiments/navigation_regression.py`；獨立 `web.guide` |

給老師的七問整理與示範順序：[exp24 導覽入口](navigation/exp24_README.md)。研究摘要連到固定研究提交，未將未合入主線的方法當成 main 功能。

## 現行健康監測工作流程

| 文件 | 內容 |
|---|---|
| [exp8_health_monitoring_workflow](exp8_health_monitoring_workflow.md) | 實驗八資料能力、Health Index、趨勢／告警、矩陣方法、限制與 CLI |
| [health_and_reports](health_and_reports.md) | `experiments/health/` 各模組、兩套趨勢的差別、CLI輸出位置與封存結果讀法 |

## 論文版管線文件（程式碼在 Ancestor）

| 文件 | 內容 | 與新專案的關係 |
|---|---|---|
| [Step1_Data_Preprocessing](Step1_Data_Preprocessing.md) | 原始訊號清洗、IQR、切段 | 新專案不重跑，只讀其下游產物 |
| [Step2_Feature_Extraction](Step2_Feature_Extraction.md) | **105 維特徵定義**（15 統計量 × 5 通道 + FFT 諧波）| **直接沿用**——新專案的輸入格式即此，`*_clean.csv` 的由來 |
| [Step3_Model_Training](Step3_Model_Training.md) | CNN/ResNet/VGG16 訓練與遷移學習 | 新專案捨深度模型（見 Model_Choice 結論）|
| [Step4_Unknown_Detection](Step4_Unknown_Detection.md) | HDBSCAN + 馬氏距離 95 百分位未知偵測 | 判定機制的前身；新專案改逐類 Ledoit–Wolf、逐樣本判定 |
| [Step5_Random_Sampling_Detection](Step5_Random_Sampling_Detection.md) | 隨機組合驗證 | 概念由 exp2 隨機注入取代 |
| [Step6_Model_Retraining](Step6_Model_Retraining.md) | 5 類 → 10 類重訓練 | 概念由 exp2 量尺擴張（全資料重擬合）取代 |
| [Tensorflow](Tensorflow.md) | 論文版環境的 TF/GPU 指南 | 新專案不需 TF/GPU |

## Lineage 時期研究文件（結論已進入新專案）

| 文件 | 內容 | 對新專案的貢獻 |
|---|---|---|
| [Mahalanobis_Improvement](Mahalanobis_Improvement.md) | 歷史深層特徵的 Ledoit–Wolf / OAS / MCD 比較 | `core/mahalanobis.py` 採用收縮估計的背景；不是現行105維跨馬達多類成績 |
| [Model_Choice_Analysis](Model_Choice_Analysis.md) | 45 模型實測：閉集準確率飽和、開集分離度才是關鍵 | 「換模型／調參無益，該動特徵與判定機制」——新專案改用 105 維手工特徵冷啟動的背景 |
| [OpenSet_Recognition](OpenSet_Recognition.md) | OpenMax、能量分數等五種 OSR 替代方案評估 | 新專案的未來方向清單（README「限制與未來工作」）|

> 注意：文內的相對連結（如 `logs/claude/Result.md`、`scripts/*.py`、`../README.md` 的舊錨點）
> 指向撰寫當時的 repo 佈局，於現行佈局中可能失效——以 Ancestor repo 與 git 歷史為準。
