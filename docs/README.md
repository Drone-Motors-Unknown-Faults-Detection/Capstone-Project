# docs/ 文件索引

組員學習與發表請先讀[畢業專題總手冊](畢業專題總手冊.md)：在同一份正文說明 Ancestor 基礎、資料角色、各方法與實驗、Web、工程、未解工作及有答案的發表練習；研究分支、歷史結果與現行功能分開標示。真人走讀與試講回饋尚待確認。

先找操作說明：[安裝與版本紀錄](runtime_policy.md)、[CI執行與輸出](ci_contract.md)、[archive物化安全規則](formal_materialization_contract.md)、[實驗八健康監測](experiments/exp8_health.md)、[導覽操作](navigation/exp13_README.md)。各頁說明程式用途、操作、預期輸出與限制。

拆 `exp6_formal_benchmark._run_benchmark` 或 `exp6_matrix.run_matrix` 前，先讀[orchestration 回歸契約](refactoring_contract.md)。

當前缺口與交付狀態請查 [GitHub issues](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/issues) 及各 PR 的受測 SHA，不把歷史測試次數當成最新 main 驗收。

內容指紋、portable manifest 與本機 private sidecar 的操作見[內容來源規則](content_provenance_contract.md)；它只涵蓋正式入口，不代表所有 runner 都有完整來源紀錄。

匯入105維 CSV 前先讀[正式特徵載入規則](formal_feature_schema_contract.md)。欄名在 `src/core/feature_schema.py` 的 `SCHEMAS`。

老師要看程式如何操作，先用[導覽入口與示範腳本](navigation/exp13_README.md)。

> 實驗總覽與名詞在 [experiments/README.md](experiments/README.md)。
> Step 1–6、Tensorflow 與三份研究筆記描述 Ancestor 的論文版管線。程式在 [Ancestor](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor)。
> 新專案的說明見根目錄 [README.md](../README.md)。

## 現行實驗技術報告

[experiments/README.md](experiments/README.md)列出13個現行實驗編號（1～13）、16份方法／操作手冊及程式位置。實驗六的基準、正式比較、矩陣與彙總各有手冊。實驗八在 [exp8_health](experiments/exp8_health.md)。實驗九用實驗六的 runner。每份手冊說明資料與用途切分、模型建立、CLI／API／Web操作、輸出及限制。實際參數與例外以對應手冊及程式為準。

| 編號 | 文件 | 程式 |
|---|---|---|
| 實驗一 | [exp1_cold_start](experiments/exp1_cold_start.md) | `src/experiments/exp1_cold_start.py` |
| 實驗二 | [exp2_scale_growth](experiments/exp2_scale_growth.md) | `src/experiments/exp2_scale_growth.py` |
| 實驗三 | [exp3_trend](experiments/exp3_trend.md) | `src/experiments/exp3_trend.py` |
| 實驗四 | [exp4_polar_map](experiments/exp4_polar_map.md) | `src/experiments/exp4_polar_map.py` |
| 實驗五 | [exp5_cross_condition](experiments/exp5_cross_condition.md) | `src/experiments/exp5_cross_condition.py` |
| 實驗六 | [exp6_osr_benchmark](experiments/exp6_osr_benchmark.md) | `src/experiments/exp6_osr_benchmark.py` |
| 實驗六 | [exp6_formal_benchmark](experiments/exp6_formal_benchmark.md) | `src/experiments/exp6_formal_benchmark.py` |
| 實驗六 | [exp6_matrix](experiments/exp6_matrix.md) | `src/experiments/exp6_matrix.py` |
| 實驗六 | [exp6_aggregate](experiments/exp6_aggregate.md) | `src/experiments/aggregate_exp6.py` |
| 實驗七 | [exp7_compare_openset](experiments/exp7_compare_openset.md) | `src/experiments/compare_openset.py` |
| 實驗八 | [exp8_health](experiments/exp8_health.md) | `src/experiments/health_index_benchmark.py`、`health_index_matrix.py`、`health_monitor.py`、`health_index_aggregate.py` |
| 實驗九 | [exp9_autoencoder](experiments/exp9_autoencoder.md) | `src/core/detectors.py`（`MLPAutoencoderDet`，`src/experiments/exp6_osr_benchmark.py` 自動納入） |
| 實驗十 | [exp10_transfer](experiments/exp10_transfer.md) | `src/experiments/exp10_transfer.py` |
| 實驗十一 | [exp11_ancestor_comparison](experiments/exp11_ancestor_comparison.md) | `src/experiments/exp11_ancestor_comparison.py` |
| 實驗十二 | [exp12_confusion_tsne](experiments/exp12_confusion_tsne.md) | `src/experiments/exp12_confusion_tsne.py` |
| 實驗十三 | [exp13_experiment_navigation](experiments/exp13_experiment_navigation.md) | `src/experiments/navigation_data_contract.py`、`src/experiments/navigation_regression.py`；獨立 `web.guide` |

給老師的七問整理與示範順序：[exp13 導覽入口](navigation/exp13_README.md)。研究摘要連到固定研究提交，未將未合入主線的方法當成 main 功能。

## 現行健康監測

實驗八的資料能力、模組、兩套趨勢、矩陣、逐窗監測、封存重算與 Ubuntu 指令都在 [exp8_health.md](experiments/exp8_health.md)。

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
| [Mahalanobis_Improvement](Mahalanobis_Improvement.md) | 歷史深層特徵的 Ledoit–Wolf / OAS / MCD 比較 | `src/core/mahalanobis.py` 採用收縮估計的背景；不是現行105維跨馬達多類成績 |
| [Model_Choice_Analysis](Model_Choice_Analysis.md) | 45 模型實測：閉集準確率飽和、開集分離度才是關鍵 | 「換模型／調參無益，該動特徵與判定機制」——新專案改用 105 維手工特徵冷啟動的背景 |
| [OpenSet_Recognition](OpenSet_Recognition.md) | OpenMax、能量分數等五種 OSR 替代方案評估 | 新專案的未來方向清單（README「限制與未來工作」）|

> 注意：文內的相對連結（如 `logs/claude/Result.md`、`scripts/*.py`、`../README.md` 的舊錨點）
> 指向撰寫當時的 repo 佈局，於現行佈局中可能失效——以 Ancestor repo 與 git 歷史為準。
