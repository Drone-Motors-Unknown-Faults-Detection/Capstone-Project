> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `src/core`、`src/experiments`、`src/web` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# TensorFlow 使用指南 — Lineage 專案

本文件說明 `gpu_utils.py` 的運作原理、如何在 TensorFlow 中使用 GPU 進行訓練，以及本專案中所使用的 TensorFlow / Keras 語法與模型架構。

---

## 目錄

1. [gpu_utils 模組說明](#1-gpu_utils-模組說明)
2. [TensorFlow GPU 訓練機制](#2-tensorflow-gpu-訓練機制)
3. [Keras 模型建構語法](#3-keras-模型建構語法)
4. [模型訓練語法](#4-模型訓練語法)
5. [中間層特徵萃取](#5-中間層特徵萃取)
6. [遷移學習模式](#6-遷移學習模式)
7. [模型儲存與載入](#7-模型儲存與載入)
8. [常見 API 速查表](#8-常見-api-速查表)
9. [Logger 模組說明](#9-logger-模組說明)

---

## 1. gpu_utils 模組說明

### 模組位置

```
gpu_utils.py
```

### 模組用途

`gpu_utils` 是本專案的 GPU 統一管理模組。所有 TensorFlow Notebook 在啟動時匯入此模組，即可自動完成 GPU 偵測、記憶體設定，並提供一致的裝置切換介面。

### 完整原始碼解析

```python
import tensorflow as tf


def _configure() -> tuple[str, list]:
    gpus = tf.config.list_physical_devices('GPU')   # 列出所有實體 GPU
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # 動態記憶體成長
            names = [gpu.name for gpu in tf.config.experimental.get_visible_devices('GPU')]
            print(f"[gpu_utils] GPU x{len(gpus)} 已啟用：{names}")
            return '/GPU:0', gpus
        except RuntimeError as e:
            # set_memory_growth 必須在 session 建立前呼叫，否則會拋出 RuntimeError
            print(f"[gpu_utils] GPU 初始化失敗（{e}），回退至 CPU")
    print("[gpu_utils] 未偵測到 GPU，使用 CPU")
    return '/CPU:0', []


DEVICE, _GPUS = _configure()   # 模組載入時立即執行一次


def device_scope():
    """回傳 tf.device context manager，自動對應 GPU 或 CPU。"""
    return tf.device(DEVICE)


def gpu_count() -> int:
    return len(_GPUS)


def is_gpu() -> bool:
    return len(_GPUS) > 0
```

### 各函式說明

| 函式 / 變數 | 型別 | 說明 |
|---|---|---|
| `DEVICE` | `str` | 全域裝置字串，`'/GPU:0'` 或 `'/CPU:0'` |
| `_GPUS` | `list` | TensorFlow `PhysicalDevice` 物件列表 |
| `device_scope()` | `contextmanager` | 回傳 `tf.device(DEVICE)` context manager |
| `gpu_count()` | `int` | 可用 GPU 數量 |
| `is_gpu()` | `bool` | 是否有可用 GPU |

### 動態記憶體成長（Memory Growth）

```python
tf.config.experimental.set_memory_growth(gpu, True)
```

**為何需要？**  
TensorFlow 預設在程式啟動時佔用 GPU 的全部 VRAM。啟用動態記憶體成長後，TensorFlow 只在需要時才申請 VRAM，避免與同機其他程式（如另一個 Jupyter kernel）衝突。

**限制：** 此設定必須在任何 TensorFlow 運算開始前呼叫，否則會拋出 `RuntimeError`。`gpu_utils` 在模組載入時即執行，因此只要在 `import tensorflow as tf` 之前或之後立即 `import gpu_utils` 即可。

### 在 Notebook 中的標準用法

```python
# ── 匯入 ──────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from scripts.gpu_utils import device_scope, DEVICE   # 匯入裝置介面
from scripts.logger import setup_logger, save_plot   # 匯入日誌模組

# ── Logger 初始化（Notebook 頂部 bootstrap cell）──────────────────
LOG, RUN_PATHS = setup_logger(__file__)      # 建立 logs/ 與 output/ 目錄
# 圖表自動儲存已由 bootstrap cell 掛載到 plt.show()

# ── 訓練 ──────────────────────────────────────────────────────────
print(f"[Training] Device: {DEVICE}")        # 確認使用的裝置
with device_scope():                          # 將以下運算固定到指定裝置
    model = build_cnn_model((105, 1), num_classes)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10,
                                  restore_best_weights=True)]
    )
```

---

## 2. TensorFlow GPU 訓練機制

### TensorFlow 裝置分配原則

TensorFlow 使用 **裝置字串（device string）** 指定運算位置：

| 字串 | 說明 |
|---|---|
| `'/GPU:0'` | 第一張 GPU（索引從 0 開始）|
| `'/GPU:1'` | 第二張 GPU |
| `'/CPU:0'` | CPU（當無 GPU 時的回退）|

`tf.device()` 是一個 context manager，區塊內建立的所有 `tf.Variable` 與張量運算都會被分配到指定裝置：

```python
with tf.device('/GPU:0'):
    # 以下所有運算都在 GPU:0 執行
    model = build_model(...)
    model.fit(...)
```

### GPU 訓練的三個必要條件

1. **CUDA 環境：** 需安裝與 TensorFlow 版本相容的 CUDA Toolkit 及 cuDNN。
2. **記憶體設定：** 在第一次 TF 運算前呼叫 `set_memory_growth()`（`gpu_utils` 已處理）。
3. **裝置指定：** 使用 `with device_scope():` 或讓 TF 自動分配（TF 2.x 會自動偏好 GPU）。

### 確認 GPU 是否正確啟用

執行以下程式碼確認 TensorFlow 能辨識 GPU：

```python
import tensorflow as tf
print("TF 版本：", tf.__version__)
print("GPU 清單：", tf.config.list_physical_devices('GPU'))
print("CUDA 可用：", tf.test.is_built_with_cuda())
```

或直接匯入 `gpu_utils` 觀察輸出訊息：

```python
from scripts.gpu_utils import DEVICE, gpu_count
# [gpu_utils] GPU x1 已啟用：['/physical_device:GPU:0']
print("訓練裝置：", DEVICE)   # /GPU:0
print("GPU 數量：", gpu_count())   # 1
```

### 訓練時的裝置確認訊息

Notebook 在 `with device_scope():` 前會印出：

```
[gpu_utils] GPU x1 已啟用：['/physical_device:GPU:0']
[Training] Device: /GPU:0
```

若印出 `[gpu_utils] 未偵測到 GPU，使用 CPU`，則表示環境無 GPU，訓練將自動回退到 CPU，不影響正確性，只影響速度。

---

## 3. Keras 模型建構語法

### Functional API vs Sequential API

本專案使用 **Functional API**（函式式 API）建構模型，而非 `Sequential`。Functional API 允許多輸入、多輸出，也方便後續萃取中間層特徵。

```python
# Functional API（本專案使用）
input_layer = tf.keras.Input(shape=(105, 1))
x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(input_layer)
# ...
model = Model(inputs=input_layer, outputs=output)

# Sequential API（未使用，僅供對比）
model = tf.keras.Sequential([
    Conv1D(16, kernel_size=3, activation='relu', padding='same', input_shape=(105, 1)),
    # ...
])
```

### 標準 CNN 模型（Step 3 / Step 6 共用）

```python
def build_cnn_model(input_shape, num_classes):
    input_layer = tf.keras.Input(shape=input_shape)           # (105, 1)

    # ── 卷積區塊 ──────────────────────────────────────────────
    x = Conv1D(16, kernel_size=3, activation='relu',
               padding='same')(input_layer)                    # 輸出: (105, 16)

    x = Conv1D(16, kernel_size=3, activation='relu')(x)        # 輸出: (103, 16)
    x = MaxPooling1D(pool_size=2)(x)                           # 輸出: (51,  16)

    x = Conv1D(16, kernel_size=3, activation='relu')(x)        # 輸出: (49,  16)
    x = MaxPooling1D(pool_size=2)(x)                           # 輸出: (24,  16)

    x = Conv1D(16, kernel_size=3, activation='relu')(x)        # 輸出: (22,  16)
    x = MaxPooling1D(pool_size=2)(x)                           # 輸出: (11,  16) ← 最後 MaxPool

    # ── 分類區塊 ──────────────────────────────────────────────
    x = Flatten()(x)                                           # 輸出: (176,)  ← 特徵萃取點
    x = Dense(16, activation='relu')(x)                        # 輸出: (16,)
    x = Dropout(0.3)(x)                                        # 訓練時隨機丟棄 30%

    output = Dense(num_classes, activation='softmax',
                   name="Screw_Number_Output")(x)              # Step3: 5 類；Step6: 10 類

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 各層說明

| 層 | 參數 | 功能 |
|---|---|---|
| `tf.keras.Input` | `shape=(105, 1)` | 定義輸入形狀，不含 batch 維度 |
| `Conv1D` | `filters=16, kernel_size=3` | 1D 卷積，提取局部時序特徵 |
| `Conv1D(padding='same')` | 第一層專用 | 保持輸出長度與輸入相同 |
| `MaxPooling1D` | `pool_size=2` | 降採樣，壓縮空間維度 50% |
| `Flatten` | — | 展平為 1D 向量（CNN 為 176 維），作為特徵萃取點 |
| `Dense` | `units=16, activation='relu'` | 全連接分類層 |
| `Dropout` | `rate=0.3` | 正規化，防止過擬合 |
| `Dense(softmax)` | `units=num_classes` | 輸出各類別的機率分布 |

### 啟動函式選擇

| 位置 | 啟動函式 | 理由 |
|---|---|---|
| 卷積層、全連接隱藏層 | `relu` | 梯度不消失，計算快 |
| 輸出層（多分類）| `softmax` | 輸出所有類別機率總和為 1 |

### 損失函式選擇

```python
loss='sparse_categorical_crossentropy'
```

使用 `sparse_categorical_crossentropy` 而非 `categorical_crossentropy`，原因是標籤為整數（0, 1, 2, ...），不需要先 `to_categorical()` 轉 one-hot 編碼。

---

## 4. 模型訓練語法

### 資料前處理與形狀轉換

```python
from sklearn.preprocessing import RobustScaler

# 標準化（RobustScaler 對離群值較穩健）
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)   # 僅用訓練資料擬合
X_test  = scaler.transform(X_test)        # 測試資料只做轉換，不重新擬合

# 形狀轉換：(N, 105) → (N, 105, 1) 以符合 Conv1D 輸入要求
X_train_r = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_r  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)
```

> **為何 Conv1D 需要 3D 輸入？**  
> `Conv1D` 預期輸入為 `(batch_size, steps, channels)`。本專案的 105 維特徵對應 `steps=105`，`channels=1`（單通道時序訊號）。

### 訓練呼叫

```python
history = model.fit(
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1),   # 訓練資料
    y_train_screws,                                             # 整數標籤
    validation_data=(
        X_test.reshape(X_test.shape[0], X_test.shape[1], 1),  # 驗證資料
        y_test_screws
    ),
    epochs=100,                                                 # 最多訓練 100 輪
    batch_size=32,                                              # 每批次 32 筆
    callbacks=[
        EarlyStopping(
            monitor='val_loss',          # 監控驗證損失
            patience=10,                 # 連續 10 輪無改善則停止
            restore_best_weights=True    # 還原最佳權重
        )
    ]
)
```

### EarlyStopping 說明

```python
EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

| 參數 | 值 | 說明 |
|---|---|---|
| `monitor` | `'val_loss'` | 監控驗證損失，而非訓練損失 |
| `patience` | `10` | 容忍 10 個 epoch 無改善 |
| `restore_best_weights` | `True` | 訓練結束後自動回溯至最佳 epoch |

**效果：** 避免過擬合，實際訓練通常在 20–50 epochs 內提前停止，而非跑滿 100 epochs。

### 訓練歷程視覺化

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.legend()

plt.tight_layout()
plt.show()
```

### 清除 Session

```python
tf.keras.backend.clear_session()
```

在 Notebook 中每次重訓前呼叫，釋放先前的計算圖與 GPU 記憶體，避免殘留狀態影響下一次訓練。

---

## 5. 中間層特徵萃取

### 用途

Step 4 / Step 5 需要將輸入資料投影到模型學習到的特徵空間，再用 HDBSCAN 叢集分析偵測未知故障。萃取點依架構而異：CNN 與 VGG16 取 **`Flatten` 層輸出**（176 / 48 維），ResNet 取 **`global_average_pooling1d`**（128 維）。程式中以 `scripts.model_utils.get_feature_layer(model)` 自動判別。

### 萃取模型建構

```python
from tensorflow.keras.models import Model, load_model

# 載入已訓練模型（compile=False 跳過重建最佳化器，節省時間）
cnn = load_model(MODEL_PATH, compile=False)

# 建立子模型：輸入不變，輸出改為最後一個 MaxPooling1D 層之後的 Flatten
feat_model = Model(
    inputs=cnn.input,
    outputs=Flatten()(cnn.get_layer("max_pooling1d").output)
    # 等價於：outputs=cnn.get_layer("flatten").output
)
```

> **為何從 `max_pooling1d` 而非 `flatten` 取出？**  
> 部分 Notebook 的 Flatten 層沒有命名，因此透過前一層（最後一個 MaxPooling1D）的輸出再接一個新 `Flatten()` 來萃取，效果相同。

### 批次推論

```python
# 訓練集特徵（用於建立馬氏距離基準）
X_tr_f = feat_model.predict(
    X_tr_s.reshape(-1, X_tr_s.shape[1], 1),
    verbose=0
)   # 輸出 shape: (N_train, 176)

# 測試集特徵（用於偵測未知故障）
X_te_f = feat_model.predict(
    X_te_s.reshape(-1, X_te_s.shape[1], 1),
    verbose=0
)   # 輸出 shape: (N_test, 176)
```

### 與 HDBSCAN / 馬氏距離的銜接

```python
import hdbscan
import numpy as np

# HDBSCAN 叢集（在中間層特徵空間）
clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=3)
cluster_labels = clusterer.fit_predict(X_te_f)

# 馬氏距離（偵測是否為未知故障）
mean = X_tr_f.mean(axis=0)
cov  = np.cov(X_tr_f.T)
inv_cov = np.linalg.pinv(cov)   # 偽逆以處理奇異矩陣

diff = X_te_f - mean
distances = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
threshold = np.percentile(distances_train, 95)   # 95 百分位閾值
unknown_mask = distances > threshold
```

---

## 6. 遷移學習模式

### 三種模型類型對應

| 模型編號 | 訓練方式 | 說明 |
|---|---|---|
| 1 ~ 9（基礎版）| 從頭訓練 | 使用單一馬達資料訓練完整模型 |
| 10 ~ 18（OneStage）| 遷移學習 | 載入基礎版模型，凍結 50% 層，以不同馬達資料 Fine-tune |
| 19 ~ 27（TwoStage）| 兩階段遷移 | 先 OneStage，再凍結並以第三組馬達資料 Fine-tune |

### 遷移學習程式碼

```python
# 載入預訓練模型
model = load_model(os.path.join(modelDirectory, 'CNN_C8000.keras'))

# 計算並凍結前 50% 的層
num_layers    = len(model.layers)
freeze_count  = num_layers // 2

for layer in model.layers[:freeze_count]:
    layer.trainable = False   # 凍結：權重不更新

for layer in model.layers[freeze_count:]:
    layer.trainable = True    # 解凍：繼續訓練

# 凍結後必須重新 compile，使 trainable 設定生效
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune（相同的 fit 呼叫）
history = model.fit(...)
```

> **凍結後為何要重新 compile？**  
> Keras 在 `compile()` 時確定哪些變數需要梯度。若不重新 compile，`layer.trainable = False` 的修改不會反映到優化器中。

### Step 6 重訓練（從頭訓練，非 Fine-tune）

```python
# 重建全新的 10 類模型（不載入舊權重）
model = build_cnn_model((105, 1), num_classes=10)

# 合併 5 類已知資料 + 5 類未知資料
X_all = np.vstack([X_known, X_unknown])
y_all = np.hstack([y_known, y_unknown])

history = model.fit(X_all, y_all, ...)
model.save('*_retrained.keras')
```

Step 6 選擇從頭重訓（而非 Fine-tuning）的原因：確保新舊 10 個類別的決策邊界公平學習，避免對原有 5 類別的偏差。

---

## 7. 模型儲存與載入

### 儲存模型

```python
# .keras 格式（Keras 原生，推薦）
model.save('CNN_C8000.keras')

# 儲存至指定路徑
model.save(os.path.join(rootDir, 'data', 'Step-1', 'model', 'CNN_C8000.keras'))
```

`.keras` 格式會儲存：
- 模型架構（層定義）
- 訓練好的權重
- 編譯設定（優化器、損失函式）

### 載入模型

```python
from tensorflow.keras.models import load_model

# 完整載入（含最佳化器狀態，可繼續訓練）
model = load_model('CNN_C8000.keras')

# 僅載入架構與權重（跳過最佳化器，用於推論或特徵萃取）
model = load_model('CNN_C8000.keras', compile=False)
```

---

## 8. 常見 API 速查表

### 匯入語句（本專案標準）

```python
import tensorflow as tf
from tensorflow.keras.models    import Model, load_model
from tensorflow.keras.layers    import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping
from scripts.gpu_utils import device_scope, DEVICE
from scripts.logger import setup_logger, save_plot
```

> `BatchNormalization` 與 `Sequential` 在本專案中未使用；模型均以 Functional API 建構。

### 層 API 一覽

| 層 | 匯入路徑 | 本專案用法 |
|---|---|---|
| `Input` | `tf.keras.layers.Input` | `shape=(105, 1)` |
| `Conv1D` | `tf.keras.layers.Conv1D` | `filters=16, kernel_size=3` |
| `MaxPooling1D` | `tf.keras.layers.MaxPooling1D` | `pool_size=2` |
| `Flatten` | `tf.keras.layers.Flatten` | 展平至 176 維（CNN；ResNet 改用 GAP 至 128 維）|
| `Dense` | `tf.keras.layers.Dense` | `units=16/5/10` |
| `Dropout` | `tf.keras.layers.Dropout` | `rate=0.3` |

### 訓練相關 API

| API | 說明 |
|---|---|
| `model.compile(optimizer, loss, metrics)` | 設定訓練配置 |
| `model.fit(X, y, validation_data, epochs, batch_size, callbacks)` | 執行訓練 |
| `model.predict(X, verbose=0)` | 批次推論 |
| `model.evaluate(X, y)` | 計算損失與指標 |
| `model.summary()` | 印出架構摘要 |
| `tf.keras.backend.clear_session()` | 清除計算圖與記憶體 |

### gpu_utils API

| API | 回傳值 | 說明 |
|---|---|---|
| `from scripts.gpu_utils import DEVICE` | `str` | `'/GPU:0'` 或 `'/CPU:0'` |
| `device_scope()` | `contextmanager` | `with device_scope():` 包裹訓練區塊 |
| `gpu_count()` | `int` | 可用 GPU 數量 |
| `is_gpu()` | `bool` | 是否有 GPU |

---

## 9. Logger 模組說明

### 模組位置

```
logger.py
```

### 主要元件

| 元件 | 型別 | 說明 |
|------|------|------|
| `RunPaths` | dataclass | 每次執行的路徑集合（logs_dir、output_dir、log_file）|
| `SimpleFileLogger` | class | 輕量級文字日誌（info / warning / error）|
| `setup_logger()` | function | 初始化目錄與 logger，回傳 `(log, paths)` |
| `save_plot()` | function | 自動儲存 matplotlib 圖表（遞增編號）|
| `_TeeToFileStream` | class | 同時寫入終端機與日誌檔的串流包裝器 |
| `tee_std_to_file()` | context manager | 將 stdout/stderr 導向 `_TeeToFileStream` |
| `redirect_std_to_logger()` | context manager | **向後相容的 no-op**，不執行任何動作 |

### 使用方式

```python
from scripts.logger import setup_logger, save_plot

# 初始化（建立 logs/ 與 output/ 目錄）
log, run_paths = setup_logger(__file__)

# 記錄訊息
log.info("開始訓練，epochs={}", 100)
log.warning("資料量不足：{}", len(X))
log.error("模型載入失敗")

# 儲存圖表（通常透過 plt.show() 自動觸發）
save_plot(plt, log, run_paths)
# → output/{run_name}/plot_000.png, plot_001.png, ...
```

### 日誌格式

```
2026-05-07 13:51:09 | INFO  | Logger initialized
2026-08-05 15:30:16 | INFO  | log_file=logs/Step3_Model_01/2026-08-05-15-30-16.log
2026-05-07 13:51:09 | INFO  | output_dir=output/program_2026-05-07-13-51-09/
```

### `_TeeToFileStream` 的 Keras 輸出過濾

`_TeeToFileStream` 在寫入日誌檔時過濾 Keras 訓練輸出的雜訊：

| 行類型 | 處理方式 |
|--------|----------|
| `Epoch x/y` 標頭 | 保留 |
| 最後一個 step 的 loss/accuracy 摘要 | 保留 |
| 中間進度條（含 `━━━━━━━━`）| 丟棄 |
| ANSI 控制碼（顏色字元）| 移除 |
| 純進度比例（如 `42/75`）| 丟棄 |

### 注意事項

- `redirect_std_to_logger()` 目前為 no-op（向後相容保留），呼叫後不影響 stdout/stderr
- 若需要將輸出導向日誌檔，使用 `tee_std_to_file(run_paths.log_file)` context manager
- Notebook 的 bootstrap cell 使用 `redirect_std_to_logger`（目前效果等同無操作），圖表仍通過 `plt.show()` 掛載自動儲存

---

## 附錄：本專案 CNN 架構形狀流

```
輸入                       (N, 105, 1)
Conv1D(16, k=3, same)  →  (N, 105, 16)
Conv1D(16, k=3)        →  (N, 103, 16)
MaxPooling1D(2)        →  (N,  51, 16)
Conv1D(16, k=3)        →  (N,  49, 16)
MaxPooling1D(2)        →  (N,  24, 16)
Conv1D(16, k=3)        →  (N,  22, 16)
MaxPooling1D(2)        →  (N,  11, 16)  ← Step4/5 特徵萃取點（CNN 接 Flatten = 176 維）
Flatten                →  (N, 176)
Dense(16, relu)        →  (N,  16)
Dropout(0.3)           →  (N,  16)
Dense(5, softmax)      →  (N,   5)      ← Step3（已知 5 類）
Dense(10, softmax)     →  (N,  10)      ← Step6（擴充 10 類）
```
