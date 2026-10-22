> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `src/core`、`src/experiments`、`src/web` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# Step 1：資料預處理 (Data Preprocessing)

## 概述

Step 1 是整個馬達故障診斷流程的起點，負責從原始量測訊號中進行資料讀取、清洗、視覺化與儲存，為後續特徵萃取建立乾淨的資料基礎。

---

## 目標

- 讀取不同轉速下的馬達原始訊號（振動、電流、溫度）
- 使用 IQR 方法移除離群值（scale = 3.0）
- 將連續訊號切割成固定長度片段
- 儲存整理後的資料供 Step 2 使用
- 視覺化各類訊號以輔助確認資料品質

---

## 腳本一覽

| 類型 | 轉速 | 腳本數量 | 說明 |
|------|------|----------|------|
| `figure` 系列 | 6000 / 8000 / 11000 RPM | 6 支（各轉速 2 支）| 繪製原始訊號波形圖 |
| `save` 系列 | 6000 / 8000 / 11000 RPM | 9 支（各轉速 3 支）| 處理並儲存清洗後的 CSV 資料 |

命名規則：

```
Step1_Data_Preprocessing_save_{RPM}_{1|2|3}.py
Step1_Data_Preprocessing_figure_{RPM}_{1|2}.py
```

例如：`Step1_Data_Preprocessing_save_6000_1.py`、`Step1_Data_Preprocessing_figure_8000_1.py`

---

## 資料結構

### 輸入

原始資料存放於多層目錄結構中：

```
data/
└── Step-{1|2|3}/
    └── {Motor}/              # 馬達代碼，例如 T1
        └── {RPM}/            # 6000rpm / 8000rpm / 11000rpm
            └── {Screws}/     # 8screws / 1screw / 2screws / ... / 4_146screws
                └── *.csv     # Tab 分隔，標頭位於第 23 行（header=22）
```

### 馬達類型與螺絲配置

| 螺絲配置 | 說明 |
|----------|------|
| 8 screws | 正常狀態（Healthy）|
| 1 screw | 故障 1（注意：目錄名為 `1screw`，無 s）|
| 2 ~ 7 screws | 不同程度的人為故障 |
| 3_14、4_146 | 特殊複合故障配置（非均勻鬆動）|

螺絲數量越少，代表馬達固定程度越低，模擬鬆脫或故障情境。

---

## 訊號讀取與欄位處理

### 原始 CSV 欄位

原始資料包含以下欄位（依馬達型號略有不同）：

| 欄位名稱 | 說明 |
|----------|------|
| `Acceleration_X` | X 軸振動加速度 |
| `Acceleration_Y` | Y 軸振動加速度 |
| `Acceleration_Z` | Z 軸振動加速度 |
| `Current` | 馬達電流 |
| `Temp_C` | 馬達溫度（馬達殼體）|
| `Temp_room` | 環境室溫 |
| `X_Value`、`Comment` | 時間戳或說明欄位（讀取後直接捨棄）|

### 衍生欄位

```python
df['Delta_T'] = df['Temp_C'] - df['Temp_room']
```

`Delta_T`（溫度差）為計算欄位，代表馬達殼體溫度與環境溫度的差值，比單一溫度更能反映馬達的發熱狀態。

### 最終輸出的 5 個訊號欄位

讀取並計算 `Delta_T` 後，從原始欄位中選取以下 **5 個訊號**作為後續處理對象：

```python
signal_columns = ['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'Current', 'Delta_T']
```

`Temp_C` 與 `Temp_room` 僅用於計算 `Delta_T`，不獨立輸出。

---

## 處理流程

```
原始 CSV 讀取（Tab 分隔，header=22）
    ↓
捨棄無關欄位（X_Value、Comment 等）
    ↓
計算 Delta_T = Temp_C - Temp_room
    ↓
選取 5 個訊號欄位
    ↓
逐欄 IQR 離群值移除（scale = 3.0）
    ↓
訊號切割（每段 10,000 點）
    ↓
水平拼接（多段並排儲存）
    ↓
輸出至 data/Step-{1|2|3}/csv/{Motor}/{RPM}/{Screws}/
```

### IQR 離群值移除

```python
def iqr(series, scale=3.0):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - scale * IQR
    upper = Q3 + scale * IQR
    return series[(series >= lower) & (series <= upper)]
```

`scale=3.0` 表示移除超出 3 倍四分位距範圍的資料點，相較常見的 `1.5` 倍設定更寬鬆，保留振動訊號中的極端但合理的峰值。

---

## 輸出

- **目錄：** `data/Step-{1|2|3}/csv/{Motor}/{RPM}/{Screws}/`
- **格式：** CSV，每欄為一個 1 秒片段，每列為一個取樣點
- **每欄資料量：** 10,000 列（= 1 秒 @ 10 kHz）
- **欄數：** 依可用資料長度決定（片段數量）
- **命名：** `{Motor}_{SignalName}_data.csv`，例如 `T1_Acceleration_X_data.csv`

每個螺絲配置輸出 5 個 CSV 檔案，對應 5 個訊號欄位。

---

## 視覺化內容（figure 系列）

figure 腳本產生以下圖表，用於人工確認資料品質：

- 三軸振動訊號波形（X / Y / Z 軸）
- 馬達溫度與室溫時序圖
- 溫度差（Delta_T）曲線
- 電流訊號波形

這些圖表識別異常區段（訊號中斷、感測器雜訊、異常峰值等），確認資料可用性後再進行 Step 2。

---

## Logger 日誌

所有 `save` 系列腳本頂部包含自動加入的 logger bootstrap，執行時會：

- 建立 `logs/{腳本名稱}/{時間戳}.log` 日誌檔
- 建立 `output/{腳本名稱}_{時間戳}/` 輸出目錄
- 自動儲存 matplotlib 圖表至 `output/` 目錄

日誌記錄每個螺絲配置的資料處理進度與儲存路徑。

---

## 技術依賴

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted    # 自然排序，確保檔案以正確順序讀取
```

---

## 注意事項

- 原始 CSV 為 **Tab 分隔**，讀取時需指定 `sep='\t'`，並以 `header=22` 跳過前 21 行說明文字
- 切割後的每段固定 **10,000 個資料點**（= 1 秒 @ Fs=10,000 Hz）
- IQR 過濾使用 **scale=3.0**（非常見的 1.5），保留較寬的資料範圍
- `1screw` 目錄名稱沒有複數 `s`，與其他 `{N}screws` 不同，程式碼中已特別處理
- 所有腳本以**專案根目錄**為執行基準，使用 `os.getcwd()` 取得根路徑
