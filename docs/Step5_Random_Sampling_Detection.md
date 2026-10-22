> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `src/core`、`src/experiments`、`src/web` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# Step 5：隨機取樣偵測 (Random Sampling Detection)

## 概述

Step 5 是 Step 4 偵測機制的延伸驗證，透過**隨機抽樣**方式從未知故障類型中組合測試批次，評估偵測演算法在不同故障組合情境下的穩健性與泛化能力。

---

## 目標

- 以隨機方式組合 1 ~ 5 種未知故障類型進行測試
- 驗證馬氏距離閾值判決在多元情境下的一致性
- 提供統計上更全面的偵測效能評估
- 輸出「New Faulty N」標籤以對接 Step 6 重訓練

---

## Notebook 一覽

共 **27 個 Jupyter Notebook**（`Step5_Model_1_Random_Detecting.ipynb` ~ `Step5_Model_27_Random_Detecting.ipynb`），對應 Step 3 / Step 4 的 27 個模型配置。

---

## 與 Step 4 的差異

| 面向 | Step 4 | Step 5 |
|------|--------|--------|
| 批次組合方式 | 依序逐步引入（1→2→3→4→5 種） | **隨機**抽取 1~5 種未知類型 |
| 測試確定性 | 固定順序，結果可重現 | 每次執行結果不同 |
| 目的 | 驗證單步漸進偵測能力 | 驗證隨機情境下的泛化能力 |
| 輸出標籤 | Unknown 1 ~ 5 | **New Faulty N**（對接重訓練）|

---

## 未知故障標籤對應

| 螺絲配置 | Step 4 標籤 | Step 5 標籤 |
|----------|------------|------------|
| 5 screws | Unknown 1 | New Faulty 1 |
| 6 screws | Unknown 2 | New Faulty 2 |
| 7 screws | Unknown 3 | New Faulty 3 |
| 3_14 screws | Unknown 4 | New Faulty 4 |
| 4_146 screws | Unknown 5 | New Faulty 5 |

---

## 隨機抽樣機制

```python
import random

# 所有可能的未知類型清單
unknown_screws = ['5screws', '6screws', '7screws', '4_146screws', '3_14screws']

# 隨機決定本次測試引入幾種未知類型（1 ~ 5）
n_unknown = random.randint(1, 5)

# 隨機抽取指定數量的未知類型
selected_unknowns = random.sample(unknown_screws, n_unknown)
```

每次執行 Notebook，均會產生不同的未知故障組合，確保評估的廣泛性。

---

## 偵測函數

Step 5 提供兩個核心評估函數：

### `evaluate_unknown_batch()`

直接回傳馬氏距離判決結果：

```python
def evaluate_unknown_batch(batch_features, threshold, cluster_model, VI, global_center):
    # 萃取特徵 → 叢集 → 計算馬氏距離 → 判決
    distances = compute_mahalanobis(batch_features, VI, global_center)
    results = ['Unknown' if d > threshold else 'Known' for d in distances]
    return results
```

### `evaluate_unknown_batch2()`

將偵測到的未知故障映射為 "New Faulty N" 格式：

```python
def evaluate_unknown_batch2(batch_features, screw_type, threshold, ...):
    results = evaluate_unknown_batch(batch_features, threshold, ...)
    # 已知：保留原故障標籤
    # 未知：映射為 New Faulty N（依 screw_type 對應）
    labeled_results = map_to_new_faulty(results, screw_type)
    return labeled_results
```

---

## 偵測流程

```
載入 Step 3 CNN 模型與 Step 4 HDBSCAN 叢集模型
    ↓
隨機抽取 1~5 種未知螺絲配置
    ↓
組合測試批次（含已知 + 隨機選定的未知）
    ↓
萃取 CNN 中間層特徵
    ↓
RobustScaler 標準化
    ↓
計算馬氏距離（相對於訓練資料分佈中心）
    ↓
與閾值（95th percentile）比較
    ↓
已知故障 → 輸出對應已知類別名稱
未知故障 → 輸出 "New Faulty N"
    ↓
記錄偵測結果與準確率
```

---

## 評估指標

| 指標 | 說明 |
|------|------|
| 已知故障識別率 | 已知樣本被正確歸類為已知類別的比率 |
| 未知故障偵測率 | 未知樣本被判定為 Unknown 的比率 |
| 誤報率（False Positive） | 已知樣本被誤判為 Unknown 的比率 |
| 漏報率（False Negative） | 未知樣本被誤判為已知類別的比率 |

---

## 視覺化輸出

- 隨機組合的批次構成說明（文字輸出）
- 馬氏距離長條圖（含閾值線）
- 各樣本的判決結果統計表

---

## 技術依賴

```python
import random
import hdbscan
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import matplotlib.pyplot as plt
```

---

## 注意事項

- 每次執行結果因隨機性不同，建議多次執行取平均評估
- 閾值沿用 Step 4 計算的 95th percentile，無需重新訓練
- `evaluate_unknown_batch2()` 的 "New Faulty N" 標籤設計為對接 Step 6 的資料準備
- 若需結果可重現，可設定 `random.seed()` 固定隨機種子
