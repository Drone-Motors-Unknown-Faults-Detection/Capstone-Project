> 📌 **歷史文件**：本文撰於論文版（Step 1–6）管線時期。文中提及的程式路徑、notebook 與執行紀錄以 [原始論文 repo](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor) 與本 repo 的 git 歷史為準，並不對應目前的 `src/core`、`src/experiments`、`src/web` 架構；新專案說明見 [README](../README.md)、文件索引見 [docs/README.md](README.md)。

# 以開放集辨識取代馬氏距離閾值

## 概述

目前 Step 4/5 以「HDBSCAN 叢集中心到訓練分布的馬氏距離是否超過第 95 百分位數」來判定未知故障。本文拆解該機制的三個結構性弱點，並整理五種可替換的開放集辨識（Open-Set Recognition, OSR）方法，逐一標註它們接進本專案需要動到什麼。

評估日期：2026-08-05
對應程式：`Step4_Model_*_Detecting.ipynb`、`Step5_Model_*_Random_Detecting.ipynb`（各 27 個）

---

## 現況機制

Step 4 的判定邏輯集中在兩個函式：

```python
def mahalanobis_stats(X, conf):
    mu  = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-6)
    d   = np.sqrt(((X - mu) @ inv * (X - mu)).sum(axis=1))
    thr = np.percentile(d, conf * 100)          # conf = 0.95
    return mu, inv, thr

# 叢集中心距離 > thr → New Fault，否則歸入最近的已知類別
cl = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=3,
                     cluster_selection_method="leaf").fit(features)
```

特徵空間取自分類網路的中間層。實測三種架構取到的維度並不相同：

| 架構 | 特徵層 | 維度 |
|------|--------|------|
| CNN | `flatten` | 176 |
| ResNet | `global_average_pooling1d` | 128 |
| VGG16 | `flatten` | 48 |

### 三個結構性弱點

**1. 單一全域共變異數**

`mahalanobis_stats` 對全部 5 個已知類別合起來估一個 `mu` 與 `cov`。當各類在特徵空間中形成分離的團塊時，這個橢球會涵蓋類與類之間的空隙——落在空隙裡的樣本距離很小，卻不屬於任何已知類別。

**2. 百分位數不是機率**

第 95 百分位數只描述訓練集自身的距離分布，對「未見過的資料落在這個距離的機率」沒有任何陳述。閾值因此無法跨轉速、跨馬達移植——實測 27 組模型的閾值散布在 **7.55 ~ 18.40**，平均 13.00。

**3. 判定粒度是叢集，不是樣本**

目前比對的是 HDBSCAN 叢集中心到分布的距離。單一樣本無法判定，且 HDBSCAN 標為 `-1` 的離群點被直接丟棄——而那些點正是最可能屬於新故障的樣本。

> 文獻普遍指出馬氏距離隱含**多變量高斯假設**，而深度網路的中間層表徵經常違反這個假設；同時它把有方向性的故障特徵壓縮成單一純量，對細粒度的嚴重程度分級不利。

---

## 五種可替換的方法

以下依**保留現有管線的程度**排序——最上面的幾乎不動網路，最下面的需要改訓練流程。這個順序有實際意義：本專案已有 45 個訓練完成的模型，能沿用它們的方法成本低得多。

### 1. OpenMax：以極值理論校正 Softmax

*Bendale & Boult, CVPR 2016 — Towards Open Set Deep Networks*

**機制**　取倒數第二層的激活向量（activation vector），為每個類別計算平均激活向量（MAV），再以 Weibull 分布擬合距離的尾端。推論時用擬合結果重新分配 Softmax 的機率質量，多出一個「未知」類別的機率。

**為何比現況強**　極值理論（EVT）專門描述尾端行為，不需要假設整體分布是高斯；判定逐樣本進行，不必先做叢集；而且每個類別各有自己的 MAV 與尾端模型，避開了單一全域共變異數的問題。

**接進本專案**　可直接套用在已訓練好的封閉集模型上，**不需重新訓練、不需改架構**。用現成的 `scripts/model_utils.get_feature_layer(model)` 取激活向量即可。需新增的只有 Weibull 擬合（`libmr` 或 `scipy.stats.weibull_min`）。

**代價**　需調的超參數從 1 個（百分位數）變成 2 個（尾端長度、α）。

**改動幅度**　僅換掉判定層。

### 2. 能量分數（Energy Score）

*Liu et al., NeurIPS 2020 — Energy-based Out-of-distribution Detection*

**機制**　直接由 logits 計算自由能 `E(x) = −logsumexp(logits)`，以能量高低區分分布內外。論文指出能量分數在理論上對齊輸入的機率密度，而 Softmax 信心分數是有偏的評分函數，容易過度自信。

**實測表現**　在 CIFAR-10 預訓練的 WideResNet 上，能量分數相對 Softmax 信心分數把平均 FPR（TPR 95% 時）降低 **18.03%**。

**接進本專案**　改動最小的一個——只要拿模型輸出的 logits 算一行 `logsumexp`。不需要特徵層、不需要叢集、不需要共變異數矩陣，也就完全繞開了目前 `np.linalg.pinv` 加 `1e-6` 擾動那段數值上不穩的處理。

**代價**　仍需要一個閾值。而且它**不是無條件優於馬氏距離**——在部分基準上馬氏距離反而更好（曾有一組報告為 FPR 73.60% 對 54.64%），值得在本專案資料上實測比較。

**改動幅度**　僅換掉判定層。

### 3. 折衷路線：保留馬氏距離，改用 EVT 擬合尾端

*開放集故障診斷文獻的常見作法*

**機制**　計算每個正確分類樣本到其所屬類別中心的馬氏距離，再以 Weibull 分布擬合這些距離的尾端，用擬合出的機率取代硬性的百分位數切點。

**為何值得考慮**　這是對現況改動最小、卻同時解掉三個弱點中兩個的路線：類別中心改為**逐類**估計（解掉全域共變異數），閾值改為**機率**（解掉百分位數不可移植）。判定粒度也自然變回逐樣本。

**接進本專案**　`mahalanobis_stats()` 改成對每個類別各算一組 `(mu, inv)`，末尾的 `np.percentile` 換成 `weibull_min.fit`。HDBSCAN 可保留作為新故障的分群用途，而不再承擔判定責任。

**代價**　每類需足夠樣本才能穩定估共變異數；105 維特徵下要留意奇異性（現況已用 `pinv` 迴避）。

**改動幅度**　重寫判定函式。

### 4. 證據深度學習（Evidential Deep Learning）

*Advanced Engineering Informatics, 2025 — SAD-DEF*

**機制**　讓網路直接輸出「證據量」而非機率，由此同時得到分類結果與不確定度。半監督對抗判別加上深度證據融合（SAD-DEF）在已知類別診斷的同時做不確定度估計，以高不確定度標示未知故障。

**為何適合這個題目**　這條路線把「未知偵測」從**後處理**提升為**模型本身的能力**。對本專案這種要持續把新故障納入訓練的迴圈（Step 6），不確定度是決定「哪些樣本值得標註並納入下一輪」的天然依據。

**接進本專案**　需要改損失函數與輸出層，45 個模型全部要重訓。以實測的訓練成本（45 個 notebook 約 22 分鐘）而言，重訓本身不是障礙。

**代價**　與現有 45 個已訓練模型不相容；Step 4/5 的整段邏輯需重寫。

**改動幅度**　改訓練目標，全數重訓。

### 5. 原型學習 + 不確定度校正

*2024–2025 — OWFD-UCPM 類、增量新類發現*

**機制**　以每個類別的原型（prototype）表示已知類，配合不確定度校正判定開放集；相關工作再以密度峰值聚類（DPC）或線上叢集辨識新故障，並搭配增量更新策略把新類別納入模型。

**為何值得注意**　這是**唯一與本專案現有架構同構**的一類方法——Step 4（偵測）→ Step 5（驗證）→ Step 6（重訓練納入新類）的迴圈，正是這條文獻線在做的事。差別在於它們用原型與不確定度取代了叢集中心與距離閾值。

**接進本專案**　可視為 Step 4–6 的整體重構，HDBSCAN 的角色由「判定」轉為「新類發現」。也是最貼近論文既有敘事的升級路線。

**代價**　工程量最大；需要重新設計三個步驟之間的介面。

**改動幅度**　Step 4–6 重構。

---

## 對照

| 方法 | 需重訓 | 逐樣本判定 | 免高斯假設 | 閾值可移植 | 新增依賴 |
|------|--------|-----------|-----------|-----------|---------|
| 現況：HDBSCAN + 馬氏 95% | 否 | 否 | 否 | 否 | — |
| 能量分數 | 否 | 是 | 是 | 部分 | 無 |
| 馬氏 + Weibull 尾端 | 否 | 是 | 尾端免除 | 是 | `scipy.stats` |
| OpenMax | 否 | 是 | 是 | 是 | `libmr` 或 `scipy` |
| 證據深度學習 | 是 | 是 | 是 | 是 | 自訂損失 |
| 原型 + 不確定度校正 | 是 | 是 | 是 | 是 | Step 4–6 重構 |

---

## 建議路線

本專案已有 45 個訓練完成、準確率 0.9883–1.0000 的模型，因此值得**先走不重訓的路線**，把「換掉判定機制」與「換掉模型」兩件事分開驗證。

1. **先做基準對照。** 在同一批特徵上同時算三種分數——現況的馬氏距離、能量分數、OpenMax 的未知機率——輸出 AUROC 與 FPR@TPR95。三者都能從既有模型直接算出，一個 notebook 就能跑完 27 組。

2. **用 5–7screws 當已知的「假未知」驗證。** 本專案的類別語意天然分級（螺絲越少鬆動越嚴重），可以把某一類暫時排除於訓練外，量測各方法把它認出來的能力，而不必依賴人工標註。

3. **依結果再決定是否重訓。** 若不重訓的方法已足夠，就停在 OpenMax 或馬氏＋Weibull；若對細粒度嚴重程度仍不理想，再往證據深度學習走。

> 實作上建議把評分函式抽成 `scripts/openset.py`，與現有的 `scripts/model_utils.py`、`scripts/train_guard.py` 一致——54 個 Step 4/5 notebook 高度重複，任何寫在 notebook 裡的判定邏輯都會變成 54 份副本。

---

## 參考來源

- Bendale, A. & Boult, T. — [Towards Open Set Deep Networks](https://www.semanticscholar.org/paper/Towards-Open-Set-Deep-Networks-Bendale-Boult/d094fb0af5bc6a26fa9c27d638c4a3a0725d8b5c), CVPR 2016（[官方程式碼 OSDN](https://github.com/abhijitbendale/OSDN)）
- Liu, W. et al. — [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759), NeurIPS 2020（[論文 PDF](https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)、[程式碼](https://github.com/weitliu/energy_ood)）
- [MetaMax: Improved Open-Set Deep Neural Networks via Weibull Calibration](https://arxiv.org/html/2211.10872v2) — OpenMax 的後續改良
- [A new open set fault diagnosis method based on adversarial discrimination and deep evidential fusion under limited labeled samples](https://www.sciencedirect.com/science/article/abs/pii/S1474034625009097), Advanced Engineering Informatics 2025
- [Research on Open-Set Recognition Methods for Rolling Bearing Fault Diagnosis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12115238/)
- [Compound fault recognition and diagnosis of rolling bearing in open-set-recognition setting](https://www.sciencedirect.com/science/article/abs/pii/S0263224124020177), Measurement 2024
- [A novel incremental method for bearing fault diagnosis that continuously incorporates unknown fault types](https://www.sciencedirect.com/science/article/abs/pii/S0888327024004229), MSSP 2024
- [OWFD-UCPM: An open-world fault diagnosis scheme based on uncertainty calibration and prototype management](https://www.researchgate.net/publication/377416125_OWFD-UCPM_An_open-world_fault_diagnosis_scheme_based_on_uncertainty_calibration_and_prototype_management)
- [Fine-Grained Open-Set Fault Diagnosis via Metric-Guided Time-Frequency Configuration Selection and Class-Specific Autoencoders](https://arxiv.org/html/2607.13368) — 對馬氏距離侷限的討論來源
- [Unknown-class recognition adversarial network for open set domain adaptation fault diagnosis of rotating machinery](https://link.springer.com/article/10.1007/s10845-024-02395-2), J. Intelligent Manufacturing 2024
- [Diagnostics for Mechanical Systems with Unknown Fault Modes: A Novel Open Set Recognition Approach](https://papers.phmsociety.org/index.php/phme/article/view/5015), PHM Society European Conference

> 文中的量化數字（FPR 降低 18.03%、閾值 7.55–18.40、特徵維度 176/128/48、模型準確率）分別來自上列論文與 2026-08-05 全流程執行的實測紀錄（見 [原始論文 logs/claude/Result.md](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor/blob/1ef4a891ae02dc95910f747a5163bb2edd608f28/logs/claude/Result.md)）。跨基準的比較請以原論文的實驗設定為準——能量分數與馬氏距離的優劣會隨資料集與骨幹網路而變。
