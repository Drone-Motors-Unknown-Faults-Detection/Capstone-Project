> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `core/ experiments/ web/` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# 馬氏距離改善實驗：收縮與穩健共變異數

## 結論

在現有 18 個可執行模型（T1、T3；CNN、ResNet、VGG16；三種轉速）上，**逐類別 Ledoit–Wolf 收縮共變異數**是本輪最佳方法：

- 開集準確率：`0.9509 → 0.9864`（+3.55 個百分點）
- Balanced Accuracy：`0.9022 → 0.9672`（+6.51 個百分點）
- AUROC：`0.9884 → 0.9987`
- FPR@TPR95：`0.0543 → 0.0055`
- 已知樣本接受率：`0.8291 → 0.9385`
- 未知故障召回率：`0.9753 → 0.9960`
- 18/18 個模型的開集準確率都高於舊方法。

因此 `MahalanobisOpenSetDetector` 的預設方法設為 `ledoit_wolf`。舊方法仍可用 `method="legacy"` 明確選取，供回歸比較與重現論文流程。

## 原方法及改善設計

Step 4/5 原本將五個已知類別合併為單一分布，以樣本共變異數、`pinv`、固定 `1e-6 I` 計算距離，再用訓練距離第 95 百分位當閾值。這會把語意不同的五類故障當成一個高斯分布；在 176、128、48 維深層特徵下，經驗共變異數也容易病態或奇異。

改善版的共同設計為：

1. 每個已知類別分別估計中心與共變異數。
2. 閾值只用獨立的已知 calibration split 決定，未知測試資料不得參與。
3. 每個樣本計算到各已知類的距離，並以該類閾值正規化。
4. 開集分數為 `min(distance_c / threshold_c)`；大於 1 判為未知。
5. HDBSCAN 後續只應負責將已判定的未知樣本分群，不再決定樣本是否未知。

## 測試的三種演算法及來源

### 1. Ledoit–Wolf 收縮共變異數（採用）

Olivier Ledoit 與 Michael Wolf 在 2004 年 *Journal of Multivariate Analysis* 論文〈[A well-conditioned estimator for large-dimensional covariance matrices](https://doi.org/10.1016/S0047-259X(03)00096-4)〉提出。方法將樣本共變異數與縮放後的單位矩陣作最佳線性收縮，在高維下得到可逆、條件良好的估計。

### 2. Oracle Approximating Shrinkage（OAS）

Yilun Chen、Ami Wiesel、Yonina C. Eldar、Alfred O. Hero 在 2010 年 *IEEE Transactions on Signal Processing* 第 58 卷第 10 期發表〈[Shrinkage Algorithms for MMSE Covariance Estimation](https://doi.org/10.1109/TSP.2010.2053029)〉，提出 OAS，以閉式解近似 oracle 收縮係數，針對 Gaussian、小樣本高維共變異數估計。

OAS 結果幾乎等同 Ledoit–Wolf，但平均開集準確率與 AUROC 略低，因此保留為可選方法、不設為預設值。

### 3. Minimum Covariance Determinant（MCD，不採用）

Peter J. Rousseeuw 在 1985 年 *Mathematical Statistics and Applications* 的〈[Multivariate Estimation with High Breakdown Point](https://doi.org/10.1007/978-94-009-5438-0_20)〉提出 MCD；Rousseeuw 與 Katrien Van Driessen 後來在 1999 年 *Technometrics* 發表 FastMCD。MCD 從共變異數行列式最小的樣本子集估計位置與散布，對離群值具高穩健性。

本專案單一類別的樣本數可能少於深層特徵維度，測試時先以只在訓練資料上擬合的 PCA 降至可估計維度，再套用 MCD。結果顯示它犧牲太多未知召回率，不適合目前資料。

## 實驗設計

```bash
venv/Scripts/python scripts/benchmark_mahalanobis.py --max-per-class 300
```

資料來自「馬達研究」備份的 `myfeature/*_clean.csv` 與 `.keras` 模型，放入被 Git 忽略的 `Lineage/data/` 後執行。

| 項目 | 範圍 |
|------|------|
| 可用模型 | 18 |
| T1 | Model 01–09 |
| T3 | Model 19–27 |
| 架構 | CNN、ResNet、VGG16 |
| 轉速 | 6000、8000、11000 RPM |
| 方法 | Legacy、Ledoit–Wolf、OAS、MCD |
| 每個螺絲配置取樣上限 | 300 |

階段 2 備份只有 `csv.zip`，沒有 `myfeature` 與 `.keras`，因此 Model 10–18 未納入；本報告不將這 9 組宣稱為已測。

已知類別 `8/1/2/3/4screws` 以 `random_state=42` 分層切為 60% 訓練、20% 校準、20% 測試。未知類別 `5/6/7/3_14/4_146screws` 只用於最後評估，不參與訓練、PCA 或閾值選擇。

Legacy 為忠實重現，保留訓練 HDBSCAN inlier 過濾，並用訓練距離本身的第 95 百分位；三種改善方法使用獨立 calibration split。

## 完整平均結果

| 方法 | AUROC | FPR@TPR95 | 開集準確率 | Balanced Acc. | 已知接受率 | 未知召回率 |
|------|------:|----------:|-----------:|--------------:|-----------:|-----------:|
| Legacy | 0.988395 | 0.054339 | 0.950947 | 0.902173 | 0.829058 | 0.975287 |
| **Ledoit–Wolf** | **0.998721** | **0.005455** | **0.986447** | **0.967246** | **0.938479** | **0.996013** |
| OAS | 0.998694 | 0.005643 | 0.986061 | 0.966326 | 0.936756 | 0.995895 |
| MCD | 0.975240 | 0.141648 | 0.922726 | 0.929553 | 0.939873 | 0.919234 |

閉集分類準確率平均為 `0.999249`，四種方法完全相同，因為本輪沒有重訓分類模型。表中的「準確率改善」指已知／未知二元開集判定準確率。

### 配對檢定

以 18 個模型為配對樣本，對 Ledoit–Wolf 與 Legacy 做單尾 Wilcoxon signed-rank test：

| 指標 | p-value |
|------|--------:|
| 開集準確率提高 | 3.81e-6 |
| Balanced Accuracy 提高 | 3.81e-6 |
| AUROC 提高 | 3.27e-4 |
| FPR@TPR95 降低 | 7.37e-4 |
| 已知接受率提高 | 3.81e-6 |
| 未知召回率提高 | 8.98e-3 |

## 可重現產物

正式全量結果位於：

- `logs/mahalanobis_benchmark/2026-08-06-13-56-50.log`
- `output/mahalanobis_benchmark/2026-08-06-13-56-50/results.csv`
- `output/mahalanobis_benchmark/2026-08-06-13-56-50/summary.csv`
- `output/mahalanobis_benchmark/2026-08-06-13-56-50/summary.json`
- `output/mahalanobis_benchmark/2026-08-06-13-56-50/method_comparison.png`

先前兩次輸出失敗與一次成功的 Model 01 冒煙測試也保留，以追蹤實驗建置過程。

## 變更紀錄

| Commit | 內容 |
|--------|------|
| `eae1258` | 新增四種偵測模式與單元測試 |
| `078e621` | 修正 Legacy 閾值，使 baseline 忠實對應現行 Notebook |
| `9737b79` | 新增真實資料 benchmark、CSV 指標與圖表 |
| `3cddc69` | 修正 Windows 無頭執行並保存冒煙測試紀錄 |
| `a16caa2` | 依 18 模型結果選定 Ledoit–Wolf 為預設方法 |

## 限制與下一步

1. Model 10–18 尚缺 Step-2 特徵與模型，補齊後應用相同指令重跑。
2. 本輪只比較馬氏距離的共變異數估計與逐類校準，尚未與 OpenMax、Energy Score 等不同開集分數比較。
3. 正式取代 54 個 Step 4/5 Notebook 前，應先把共同流程改由本模組呼叫，避免再次複製判定邏輯。
4. HDBSCAN 應保留作為未知樣本分群／新類發現工具，而不是已知／未知判定器。
