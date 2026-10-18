> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `core/ experiments/ web/` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# Step 3：模型訓練 (Model Training)

## 概述

Step 3 使用 Step 2 萃取的 105 維特徵向量，訓練一維卷積神經網路（1D CNN），建立馬達故障類型的分類模型，可辨識 5 種已知故障狀態。訓練完成的模型同時作為 Step 4/5 特徵萃取的基礎。

---

## 目標

- 以三種架構（CNN / ResNet / VGG16）分別訓練，區分已知的 5 種故障類別
- 儲存訓練好的模型供 Step 4 偵測使用
- 建立中間層特徵萃取器用於異常偵測
- 評估模型在測試集上的表現（混淆矩陣、學習曲線）

---

## Notebook 一覽

| 模型範圍 | 類型 | 說明 |
|----------|------|------|
| Model 01 ~ 09 | 基礎版本 | 從頭訓練（僅 01~03 是真正的 baseline，04~09 由 8000rpm 遷移）|
| Model 10 ~ 27（OneStage）| OneStage 變體 | 載入基礎模型、凍結前半層、以不同馬達資料 Fine-tune |
| Model 10 ~ 27（TwoStage）| TwoStage 變體 | 兩階段遷移學習（6000/11000rpm 組才是真正的兩階段）|

共 **45 個 Jupyter Notebook**：`Step3_Model_01.ipynb` ~ `Step3_Model_09.ipynb`（基礎）、`Step3_Model_10_OneStage.ipynb` ~ `Step3_Model_27_OneStage.ipynb`（OneStage）、`Step3_Model_10_TwoStage.ipynb` ~ `Step3_Model_27_TwoStage.ipynb`（TwoStage）

編號與「馬達時期 × 轉速 × 架構」的完整對照見 [README 的模型編號系統](../README.md#模型編號系統)。

---

## 分類目標

| 類別標籤 | 螺絲配置 | 狀態說明 |
|----------|----------|----------|
| 0 | 8 screws | 健康（Healthy）|
| 1 | 1 screw | 故障 1（Faulty 1）|
| 2 | 2 screws | 故障 2（Faulty 2）|
| 3 | 3 screws | 故障 3（Faulty 3）|
| 4 | 4 screws | 故障 4（Faulty 4）|

---

## 模型架構

每個組合都會以 CNN / ResNet / VGG16 三種架構各訓練一次（論文稱 CNN_11 Layers / CNN_Res / CNN_VGG），三者輸入輸出與訓練設定一致，只有中間結構不同。以下為 CNN 基準架構。

輸入：`(105, 1)`（105 維特徵向量，reshape 為 1D 序列）

```
Input (N, 105, 1)
    ↓
Conv1D(16 filters, kernel=3, padding='same', ReLU)    → (N, 105, 16)
    ↓
Conv1D(16 filters, kernel=3, ReLU) → MaxPooling1D(2)  → (N, 51, 16)
    ↓
Conv1D(16 filters, kernel=3, ReLU) → MaxPooling1D(2)  → (N, 24, 16)
    ↓
Conv1D(16 filters, kernel=3, ReLU) → MaxPooling1D(2)  → (N, 11, 16)
    ↓
Flatten                                                → (N, 176)  ← 特徵萃取點
    ↓
Dense(16, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(5, Softmax)   ← Step 3 輸出 5 類；Step 6 重訓練時改為 10 類
```

> CNN 的 `Flatten` 層輸出為 **176 維向量**，作為 Step 4/5 HDBSCAN 叢集的輸入特徵空間。
>
> **三種架構的特徵層並不相同**——ResNet 使用 Global Average Pooling，沒有 Flatten 層：
>
> | 架構 | 特徵層 | 維度 |
> |------|--------|------|
> | CNN | `flatten` | 176 |
> | ResNet | `global_average_pooling1d` | 128 |
> | VGG16 | `flatten` | 48 |
>
> Step 4/5 以 `scripts/model_utils.get_feature_layer(model)` 自動取得對應層。

### 模型建構程式碼

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape=(105, 1), num_classes=5):
    input_layer = tf.keras.Input(shape=input_shape)

    x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = Conv1D(16, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(16, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(16, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax', name='Screw_Number_Output')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

## 訓練配置

| 參數 | 值 |
|------|----|
| 優化器 | Adam（learning_rate = 1e-4）|
| 損失函數 | Sparse Categorical Crossentropy |
| 訓練週期上限 | 100 epochs |
| 批次大小 | 32 |
| 訓練/測試比例 | 80 / 20（stratify=True 分層抽樣）|
| 早停策略 | EarlyStopping（patience=10，monitor='val_loss'，restore_best_weights=True）|
| 標準化方法 | RobustScaler（以訓練集 fit，測試集 transform）|

---

## 訓練流程

```
1. [Bootstrap]  初始化 logger（建立 logs/ 與 output/ 目錄）
2. [設定]       匯入套件 + GPU 初始化（scripts.gpu_utils）
3. [載入資料]   從 myfeature/ 讀取 *_Group_feature_data.csv
4. [標籤編碼]   螺絲字串 → 整數（'8screws'→0, '1screw'→1, ...）
5. [分割資料]   train_test_split（test_size=0.2, stratify=y）
6. [標準化]     RobustScaler fit on X_train → transform X_train & X_test
7. [形狀轉換]   (N, 105) → (N, 105, 1) 以符合 Conv1D 輸入格式
8. [建模]       build_cnn_model(input_shape=(105,1), num_classes=5)
9. [訓練]       model.fit() + EarlyStopping callback
10.[評估]       混淆矩陣 + 準確率/損失曲線 → 自動儲存至 output/
11.[儲存]       model.save('data/Step-*/model/CNN_*.keras')
```

### 資料前處理程式碼

```python
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 標準化（僅以訓練集擬合）
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# reshape 為 Conv1D 格式
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)
```

---

## 模型輸出

| 輸出物 | 儲存位置 | 說明 |
|--------|----------|------|
| `CNN_*.keras` | `data/Step-*/model/` | 完整 5 類分類模型 |
| 混淆矩陣圖 | `output/{run}/plot_*.png` | 測試集各類別預測結果 |
| 學習曲線圖 | `output/{run}/plot_*.png` | Accuracy 與 Loss 隨 epoch 變化 |

---

## OneStage vs TwoStage 遷移學習

| 類型 | 訓練方式 | 說明 |
|------|----------|------|
| 基礎版（Model 1-9） | 從頭訓練 | 使用單一馬達資料完整訓練 |
| OneStage（Model 10-27）| Fine-tune | 載入基礎版模型，凍結前 50% 層，以第二組馬達資料繼續訓練 |
| TwoStage（Model 10-27）| 二次 Fine-tune | 在 OneStage 基礎上再凍結並以第三組馬達資料 Fine-tune |

```python
# OneStage / TwoStage 遷移學習核心程式碼
model = load_model('CNN_base.keras')

freeze_count = len(model.layers) // 2
for layer in model.layers[:freeze_count]:
    layer.trainable = False
for layer in model.layers[freeze_count:]:
    layer.trainable = True

# 凍結後必須重新 compile 才會生效
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(X_new, y_new, ...)
```

---

## 模型表現

- 測試集準確率：通常達到 **100%**（5 類已知故障邊界明確）
- 早停機制有效防止過擬合，實際通常在 20–50 epochs 提前結束
- Dropout(0.3) 增強模型泛化能力

---

## Logger 與輸出

每個 Notebook 頂部包含自動維護的 bootstrap cell：

```python
# --- logging bootstrap (auto-added) ---
import importlib
from scripts.notebook_bootstrap import bootstrap
_logger_mod = importlib.reload(_logger_mod)
save_plot = _logger_mod.save_plot
setup_logger = _logger_mod.setup_logger
tee_std_to_file = _logger_mod.tee_std_to_file

LOG, RUN_PATHS = setup_logger('notebook', console=False)
_tee_ctx = tee_std_to_file(RUN_PATHS.log_file)
_tee_ctx.__enter__()
import atexit
atexit.register(_tee_ctx.__exit__, None, None, None)
# ... (自動儲存圖表與 GPU 資源釋放邏輯略)
# --- end logging bootstrap ---
```

`plt.show()` 呼叫時自動將圖表存至 `output/{run_name}/plot_000.png`、`plot_001.png` 等。stdout/stderr 同步寫入日誌檔。

---

## 技術依賴

```python
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.gpu_utils import device_scope, DEVICE
```

---

## 注意事項

- **模型對應：** `Step3_Model N.ipynb` 訓練出的模型必須配合 `Step4_Model N__Detecting.ipynb` 使用，不可交叉混用
- **中間層特徵：** 特徵萃取層是 Step 4/5 的核心（CNN 176 維、ResNet 128 維、VGG16 48 維），Step 6 重訓練後的新模型也需保持對應架構
- **Step 6 輸出層：** 重訓練時將最後一層改為 `Dense(10, softmax)`，其他層結構完全相同
- **遷移學習版本：** OneStage/TwoStage 依賴對應的基礎版模型，執行前須確認來源模型已存在
- **特徵檔案命名：** Step 2 輸出的特徵檔案為 `*_Group_feature_data.csv`，Step 3 直接讀取此檔案
