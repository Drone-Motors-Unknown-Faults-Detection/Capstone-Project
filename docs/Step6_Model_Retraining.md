> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `src/core`、`src/experiments`、`src/web` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# Step 6：模型重訓練 (Model Retraining)

## 概述

Step 6 是整個持續學習迴圈的最後一步，將未知故障資料納入訓練集，對模型進行增量重訓練，使模型擴充從 5 類分類器升級為 **10 類分類器**，完成一次完整的「故障知識更新」週期。

> **實作說明：** 目前 Step 6 直接從 `myfeature/` 讀取已標記的未知故障特徵（ground truth），而非接入 Step 4/5 的偵測輸出，以評估模型知識擴充的效能上界。完整的閉環流程（自動接收 Step 4/5 偵測結果）為未來工作方向。

---

## 目標

- 將 5 種未知故障資料整合進訓練集
- 重訓練 CNN 使其具備識別 10 種故障類型的能力
- 驗證新類別的分類準確率
- 建立下一輪偵測循環的基礎模型

---

## Notebook 一覽

共 **27 個 Jupyter Notebook**（`Step6_Model_1_Retrain.ipynb` ~ `Step6_Model_27_Retrain.ipynb`），對應前序步驟的 27 個模型配置。

---

## 分類類別擴充

### 重訓練前（Step 3 模型）：5 類

| 標籤 | 螺絲配置 | 狀態 |
|------|----------|------|
| 0 | 8 screws | Healthy（健康）|
| 1 | 1 screw | Faulty 1 |
| 2 | 2 screws | Faulty 2 |
| 3 | 3 screws | Faulty 3 |
| 4 | 4 screws | Faulty 4 |

### 重訓練後（Step 6 模型）：10 類

| 標籤 | 螺絲配置 | 狀態 |
|------|----------|------|
| 0 | 8 screws | Healthy |
| 1 | 1 screw | Faulty 1 |
| 2 | 2 screws | Faulty 2 |
| 3 | 3 screws | Faulty 3 |
| 4 | 4 screws | Faulty 4 |
| 5 | 5 screws | **New Faulty 1**（新加入）|
| 6 | 6 screws | **New Faulty 2**（新加入）|
| 7 | 7 screws | **New Faulty 3**（新加入）|
| 8 | 3_14 screws | **New Faulty 4**（新加入）|
| 9 | 4_146 screws | **New Faulty 5**（新加入）|

---

## 模型架構變更

Step 6 的 CNN 架構與 Step 3 **完全相同**，唯一差異在於輸出層：

```
... (前 9 層相同) ...
    ↓
Dense(16, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(10, Softmax)   ← 從 5 類擴充為 10 類
```

---

## 訓練資料組合

```
訓練資料 = 原始已知資料（5 類）+ 新增未知資料（5 類）
```

| 資料來源 | 類別數 | 說明 |
|----------|--------|------|
| Step 1/2 原始資料 | 5 類 | 8、1、2、3、4 screws |
| 未知故障特徵（ground truth） | 5 類 | 5、6、7、3_14、4_146 screws |

兩組資料合併後重新進行 80/20 訓練測試切割。

---

## 訓練配置

與 Step 3 相同：

| 參數 | 值 |
|------|----|
| 優化器 | Adam（learning_rate = 1e-4）|
| 損失函數 | Sparse Categorical Crossentropy |
| 訓練週期 | 最多 100 epochs |
| 批次大小 | 32 |
| 訓練/測試比例 | 80 / 20 |
| 早停策略 | EarlyStopping（patience = 10）|

---

## 訓練流程

```
載入合併後的 10 類資料集
    ↓
資料標準化（與 Step 3 相同處理）
    ↓
訓練/測試集切割（80/20）
    ↓
建構 10 類 CNN 模型
    ↓
模型訓練（fit with EarlyStopping）
    ↓
評估：混淆矩陣、學習曲線
    ↓
儲存重訓練後的模型（.keras）
    ↓
（可選）建立新的 Flatten 層萃取器供下一輪偵測使用
```

---

## 輸出

| 輸出物 | 說明 |
|--------|------|
| `{ARCH}_{LETTER}{RPM}_retrained.keras` | 重訓練後的 10 類分類模型（例：`CNN_C8000_retrained.keras`）|
| 混淆矩陣圖（10×10）| 全類別測試集預測結果 |
| 學習曲線圖 | 訓練過程可視化 |

---

## 模型表現

- 測試集準確率：通常達到 **100%**
- 新加入的 5 類故障可被準確識別
- 原有 5 類故障分類能力不退化

---

## 持續學習迴圈

Step 6 完成後，系統可進入下一輪循環：

```
Step 3（5 類模型）
    ↓
Steps 4-5（偵測未知）
    ↓
Step 6（重訓練 → 10 類模型）
    ↓
Steps 4-5（以 10 類模型偵測更多未知）
    ↓
Step 6（繼續擴充 → 15 類模型）
    ↓
...（持續迭代）
```

此架構實現了**開放集（Open-Set）故障診斷**的持續學習能力。

---

## 技術依賴

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 注意事項

- 重訓練從頭開始（非 Fine-tuning），確保新舊類別之間無偏差
- 若採用增量學習（Fine-tuning），需注意災難性遺忘（Catastrophic Forgetting）問題
- 10 類混淆矩陣的解讀需注意新舊類別的邊界區分
- 重訓練模型應作為下一輪 Step 4/5 的基礎模型，確保版本對應正確
