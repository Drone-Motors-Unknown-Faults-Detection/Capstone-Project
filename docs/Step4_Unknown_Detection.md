> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `core/ experiments/ web/` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# Step 4：未知故障偵測 (Unknown Fault Detection)

## 概述

Step 4 利用 Step 3 訓練好的 CNN 模型提取深度特徵，結合 HDBSCAN 無監督叢集演算法與馬氏距離（Mahalanobis Distance）統計閾值，偵測並識別模型從未見過的「未知故障」類型。

---

## 目標

- 偵測不屬於 5 類已知故障的新型故障
- 利用叢集分析理解未知故障的分佈結構
- 以馬氏距離設定統計判決邊界，區分已知 / 未知狀態
- 為 Step 6 的重訓練提供未知故障樣本

---

## Notebook 一覽

共 **27 個 Jupyter Notebook**（`Step4_Model_1_Detecting.ipynb` ~ `Step4_Model_27_Detecting.ipynb`），對應 Step 3 的 27 個模型編號。

---

## 已知 vs 未知類別定義

| 狀態 | 螺絲配置 | 類型 |
|------|----------|------|
| Healthy | 8 screws | 已知（Known）|
| Faulty 1 | 1 screw | 已知（Known）|
| Faulty 2 | 2 screws | 已知（Known）|
| Faulty 3 | 3 screws | 已知（Known）|
| Faulty 4 | 4 screws | 已知（Known）|
| Unknown 1 | 5 screws | **未知（Unknown）**|
| Unknown 2 | 6 screws | **未知（Unknown）**|
| Unknown 3 | 7 screws | **未知（Unknown）**|
| Unknown 4 | 3_14 screws | **未知（Unknown）**|
| Unknown 5 | 4_146 screws | **未知（Unknown）**|

---

## 偵測方法

### 整體流程

```
載入 Step 3 CNN 模型
    ↓
萃取中間層（Flatten 層）特徵
    ↓
RobustScaler 特徵標準化
    ↓
HDBSCAN 叢集分析（僅使用已知訓練資料）
    ↓
計算各叢集中心的馬氏距離
    ↓
設定閾值（訓練資料距離分布的第 95 百分位數）
    ↓
對新批次資料進行判決：
  距離 > 閾值 → 未知故障
  距離 ≤ 閾值 → 已知故障（對應 LABEL_ORDER）
    ↓
視覺化結果（水平長條圖）
```

---

## 關鍵演算法

### 1. 深度特徵萃取

使用 Keras 功能 API 擷取 Flatten 層輸出：

```python
feature_extractor = keras.Model(
    inputs=model.input,
    outputs=get_feature_layer(model).output   # 依架構自動取 flatten 或 GAP
)
features = feature_extractor.predict(X)
```

### 2. HDBSCAN 叢集

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=25,
    min_samples=3,
    cluster_selection_method='eom'  # 或 'leaf'
)
labels = clusterer.fit_predict(scaled_features)
```

HDBSCAN 特性：
- 自動決定叢集數量
- 雜訊點標記為 `-1`
- 對高維稀疏資料有良好表現

### 3. 馬氏距離閾值判決

```python
from scipy.spatial.distance import mahalanobis

# 計算訓練資料叢集中心間距離
VI = np.linalg.inv(np.cov(train_features.T))
distances = [mahalanobis(center, global_center, VI) for center in cluster_centers]

# 設定閾值
threshold = np.percentile(distances, 95)

# 判決新批次資料
for new_center in new_cluster_centers:
    dist = mahalanobis(new_center, global_center, VI)
    if dist > threshold:
        label = "Unknown"
    else:
        label = LABEL_ORDER[closest_known_cluster]
```

---

## 資料取樣策略

| 參數 | 值 | 說明 |
|------|----|------|
| 每類上限 | 60 樣本 | 避免單一類別主導分析 |
| 最大組合數 | 450 樣本 | 防止批次不平衡 |

---

## 測試策略

Step 4 採漸進式測試：

1. **單一未知類型：** 一次引入 1 種未知螺絲配置
2. **兩種未知組合：** 同時引入 2 種未知類型
3. **三種、四種、五種：** 逐步增加至 5 種未知類型全部混入

這種設計驗證演算法在不同複雜度情境下的偵測穩健性。

---

## 輸出與視覺化

- **水平長條圖：** 各叢集中心距離 vs 馬氏距離閾值（虛線）
- **顏色編碼：** 依已知/未知故障類型區分
- **決策結果：** 每個叢集的判定標籤（已知故障名稱或 "Unknown"）

---

## 技術依賴

```python
import hdbscan
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import matplotlib.pyplot as plt
```

---

## 注意事項

- HDBSCAN 結果具有隨機性，可能影響叢集邊界
- 馬氏距離需要協方差矩陣可逆，若特徵高度共線需先降維
- 閾值 95% 為經驗設定，可依實際需求調整
- 第 95 百分位閾值在訓練資料少時可能過於寬鬆，需根據資料規模評估
- 此步驟的判決結果為 Step 5 隨機測試和 Step 6 重訓練提供依據
