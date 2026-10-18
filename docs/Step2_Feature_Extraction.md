> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `core/ experiments/ web/` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# Step 2：特徵萃取 (Feature Extraction)

## 概述

Step 2 從 Step 1 處理後的訊號片段中萃取時域與頻域特徵，將原始波形轉換為 **105 維固定特徵向量**，作為後續機器學習模型的輸入。

---

## 目標

- 計算振動、電流、溫度訊號的統計特徵（時域）
- 對振動訊號執行 FFT，萃取諧波頻率幅值（頻域）
- 輸出結構化特徵矩陣供 Step 3 模型訓練使用

---

## 腳本一覽

| 腳本 | 轉速 | 說明 |
|------|------|------|
| `Step2_Feature_Extraction_6000_1.py` | 6000 RPM | 馬達批次 1 |
| `Step2_Feature_Extraction_6000_2.py` | 6000 RPM | 馬達批次 2 |
| `Step2_Feature_Extraction_6000_3.py` | 6000 RPM | 馬達批次 3 |
| `Step2_Feature_Extraction_8000_1.py` | 8000 RPM | 馬達批次 1 |
| `Step2_Feature_Extraction_8000_2.py` | 8000 RPM | 馬達批次 2 |
| `Step2_Feature_Extraction_8000_3.py` | 8000 RPM | 馬達批次 3 |
| `Step2_Feature_Extraction_11000_1.py` | 11000 RPM | 馬達批次 1 |
| `Step2_Feature_Extraction_11000_2.py` | 11000 RPM | 馬達批次 2 |
| `Step2_Feature_Extraction_11000_3.py` | 11000 RPM | 馬達批次 3 |

共 **9 支腳本**，每個轉速 3 支，對應不同的馬達個體或資料配置。

---

## 特徵向量結構（105 維）

| 訊號 | 統計特徵 | FFT 特徵 | 小計 | 維度範圍 |
|------|----------|----------|------|----------|
| 電流（Current） | 15 | 0 | 15 | 1 ~ 15 |
| 振動 X 軸 | 15 | 10 | 25 | 16 ~ 40 |
| 振動 Y 軸 | 15 | 10 | 25 | 41 ~ 65 |
| 振動 Z 軸 | 15 | 10 | 25 | 66 ~ 90 |
| 溫度差（Delta_T） | 15 | 0 | 15 | 91 ~ 105 |
| **合計** | **75** | **30** | **105** | |

此維度順序固定，所有步驟（Step 3 ~ Step 6）均依賴此格式，**不可更動**。

---

## 特徵定義

### 時域統計特徵（每訊號 15 個）

| 序號 | 特徵名稱 | 說明 |
|------|----------|------|
| 1 | RMS | 均方根值 $\sqrt{\frac{1}{N}\sum x_i^2}$ |
| 2 | Mean | 算術平均值 $\bar{x}$ |
| 3 | Kurtosis | 峰態係數（偵測衝擊特性）$\frac{\mu_4}{\sigma^4}$ |
| 4 | Std Dev | 標準差 $\sigma$ |
| 5 | Skewness | 偏態係數 $\frac{\mu_3}{\sigma^3}$ |
| 6 | Peak-to-Peak | 峰值差 $x_{max} - x_{min}$ |
| 7 | Crest Indicator | 波峰指標 $\frac{x_{max}}{RMS}$ |
| 8 | Clearance Indicator | 餘隙指標 $\frac{x_{max}}{(\frac{1}{N}\sum\sqrt{|x_i|})^2}$ |
| 9 | Shape Indicator | 波形指標 $\frac{RMS}{\frac{1}{N}\sum|x_i|}$ |
| 10 | Impulse Indicator | 衝擊指標 $\frac{x_{max}}{\frac{1}{N}\sum|x_i|}$ |
| 11 | Max | 最大值 $x_{max}$ |
| 12 | Min | 最小值 $x_{min}$ |
| 13 | MSA | Mean Square Amplitude $\frac{1}{N}\sum x_i^2$ |
| 14 | Variance | 變異數 $\sigma^2$ |
| 15 | Mean Amplitude | 平均振幅 $\frac{1}{N}\sum|x_i|$ |

### 頻域 FFT 特徵（振動三軸各 10 個）

對振動訊號進行 FFT，在基頻的 1 ~ 10 倍諧波頻率處各萃取一個幅值：

| 轉速 | 基頻 | 諧波頻率範圍 | 換算依據 |
|------|------|-------------|----------|
| 6000 RPM | 100 Hz | 100 ~ 1000 Hz | 6000 / 60 = 100 |
| 8000 RPM | 133 Hz | 133 ~ 1330 Hz | 8000 / 60 ≈ 133 |
| 11000 RPM | 183 Hz | 183 ~ 1830 Hz | 11000 / 60 ≈ 183 |

FFT 參數：

```python
Fs = 10000      # 取樣率（固定，不可更改）
rawdata = 10000 # 每段點數（= 1 秒）
dF = Fs / rawdata   # 頻率解析度 = 1 Hz

# 萃取第 i 倍諧波的幅值
for i in range(1, 11):
    target_freq = base_freq * i
    idx = np.argmin(np.abs(freq - target_freq))
    feature = fft_amplitude[idx]
```

---

## 處理流程

```
讀取 data/Step-{1|2|3}/csv/{Motor}/{RPM}/{Screws}/ 資料
    ↓
逐欄讀取（每欄 = 1 個 10,000 點片段）
    ↓
RobustScaler 標準化（fit_transform 每個片段）
    ↓
計算 15 項時域統計特徵（5 種訊號各 15 個 = 75 個）
    ↓
對三軸振動訊號執行 FFT
    ↓
提取 10 個諧波幅值（三軸各 10 個 = 30 個）
    ↓
組合成 105 維特徵向量，附加螺絲標籤
    ↓
移除含 NaN 的列（dropna）
    ↓
依螺絲配置合併輸出 feature_data.csv
```

### 標準化說明

在計算統計特徵前，先對原始訊號套用 `RobustScaler`：

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
segment_scaled = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
```

`RobustScaler` 使用中位數與四分位距（IQR）進行標準化，對訊號中殘留的離群值具強健性，不受極端振幅污染統計特徵的計算結果。

---

## 輸出

### 目錄結構

```
data/Step-{1|2|3}/myfeature/{Motor}/{RPM}/{Screws}/
├── {Motor}_Group_feature_data.csv        # 萃取結果（移除含 NaN 的列後）
└── {Motor}_Group_feature_data_clean.csv  # 再以 IQR 逐列過濾離群值
```

### 檔案格式

- **每列**：一個 1 秒訊號片段的 105 維特徵向量
- **最後一欄**：螺絲配置標籤（字串，如 `8screws`、`1screw`）
- **欄位順序**：固定為 Current(15) → Vib_X(25) → Vib_Y(25) → Vib_Z(25) → Delta_T(15)

### 兩種輸出檔案的差異

| 檔案 | 說明 | 讀取者 |
|------|------|--------|
| `*_Group_feature_data.csv` | 萃取結果，已移除含 NaN 的列 | **Step 3 / Step 6**（訓練）|
| `*_Group_feature_data_clean.csv` | 再以 IQR（scale=1.5）逐列過濾，任一維超界即捨棄整列 | **Step 4 / Step 5**（叢集與距離判定）|

兩者不可混用。Step 4/5 的 HDBSCAN 與馬氏距離對離群值敏感，因此另備乾淨版本；
Step 3/6 的訓練則使用未過濾的完整資料。實測過濾後約保留 54% 的列。

> 早期版本另有 `*_raw.csv`，現已廢除——它沒有任何讀取端，且曾造成
> baseline（T1/8000rpm）長期讀到過期特徵檔。

---

## 程式碼關鍵片段

### FFT 特徵萃取

```python
from scipy.fftpack import fft

def extract_fft_features(signal, base_freq, Fs=10000, num_harmonics=10):
    N = len(signal)
    freq = np.fft.fftfreq(N, d=1/Fs)
    fft_vals = np.abs(fft(signal)) / N    # 正規化幅度

    features = []
    for i in range(1, num_harmonics + 1):
        target_freq = base_freq * i
        idx = np.argmin(np.abs(freq - target_freq))
        features.append(fft_vals[idx])
    return features
```

### 統計特徵計算

```python
from scipy.stats import kurtosis, skew

def extract_stat_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    mean_amp = np.mean(np.abs(signal))
    peak = np.max(np.abs(signal))

    return [
        rms,                                        # 1. RMS
        np.mean(signal),                            # 2. Mean
        kurtosis(signal),                           # 3. Kurtosis
        np.std(signal),                             # 4. Std
        skew(signal),                               # 5. Skewness
        np.max(signal) - np.min(signal),            # 6. Peak-to-Peak
        peak / rms,                                 # 7. Crest Indicator
        peak / (np.mean(np.sqrt(np.abs(signal)))**2),  # 8. Clearance Indicator
        rms / mean_amp,                             # 9. Shape Indicator
        peak / mean_amp,                            # 10. Impulse Indicator
        np.max(signal),                             # 11. Max
        np.min(signal),                             # 12. Min
        np.mean(signal**2),                         # 13. MSA
        np.var(signal),                             # 14. Variance
        mean_amp,                                   # 15. Mean Amplitude
    ]
```

---

## 技術依賴

```python
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import RobustScaler
from natsort import natsorted
```

---

## 注意事項

- **FFT 基頻**：各轉速的基頻 = RPM / 60，不同腳本硬編碼對應值，新增轉速時需正確設定
- **取樣率固定**：`Fs = 10000 Hz`，所有腳本共用，與 Step 1 一致，**不可修改**
- **NaN 處理**：以 `dropna()` 直接移除含 NaN 的列，不進行插補（NaN 通常來自除零或常數訊號）
- **欄位順序固定**：105 維特徵的欄位順序不可變動，否則 Step 3 模型輸入不匹配
- **使用 `_clean.csv`**：Step 3 以後的步驟均使用乾淨版本，使用 `_data.csv` 可能包含雜訊特徵
