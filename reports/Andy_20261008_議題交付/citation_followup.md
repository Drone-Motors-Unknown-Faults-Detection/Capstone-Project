# #19：引用與實作對照（2026-10-08）

## 範圍與判定

沿用固定 `e7ee971d2ed41d5ce80e3a56f912eb4dc1c038dc` 的 `reports/citation_audit_20261005/README.md`、`content_notes.md`，以及研究提交 `b3d68f145cfa187e208e38cae74bf33f68d7c367` 的117條目錄。本輪沒有重建目錄、重新訓練或把取得PDF等同全文主張通過。exp24只繼承exp1／2／3／7的文獻；匿名、SHA、配對、操作員步驟是專案工程約定。

下表分開判定書目、所讀內容與實作。查證日期均為2026-10-08；全文連結只供閱讀，不把未確認再散布權利的PDF提交進repo。出版社失敗／付費頁不繞過。

| 條目與完整書目 | 來源／全文狀態與實際支持位置 | 實作對照及判定 |
|---|---|---|
| S. W. Roberts (1959), Control Chart Tests Based on Geometric Moving Averages, Technometrics 1(3):239–250；DOI 10.1080/00401706.1959.10489860 | [CMU原文典藏](https://www.stat.cmu.edu/technometrics/59-69/VOL-01-03/v0103239.pdf)，本輪讀p.239–242，p.240 §2式(1)為幾何平均遞迴，初值為中心線 | 已核實遞迴對應。`core/trend.py:57–61`把輸入改為開集超線旗標、初值0；alpha .08、.2/.5帶、warmup10、12筆判別是專案設計，不能引用原文保證故障或誤報效能。 |
| E. S. Page (1954), Continuous Inspection Schemes, Biometrika 41(1–2):100–115；DOI 10.1093/biomet/41.1-2.100 | [出版社DOI](https://doi.org/10.1093/biomet/41.1-2.100)、[NDL書目](https://ndlsearch.ndl.go.jp/en/books/R100000136-I1361137046437729152)核實書目；本輪未取得原文全文。[NIST官方CUSUM](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm)只作獨立實作參照 | `core/trend.py:61`只有正向累積與歸零；`kind`依EWMA，不依CUSUM控制界限。書目已核實；原文逐式對應尚無法核實，不宣稱完整Page程序。 |
| Nagi Z. Gebraeel, Mark A. Lawley, Rong Li, Jennifer K. Ryan (2005), Residual-life distributions from component degradation signals: A Bayesian approach, IIE Transactions 37(6):543–557；DOI 10.1080/07408170590929018 | [出版社書目](https://www.tandfonline.com/doi/abs/10.1080/07408170590929018)、[作者出版清單](https://sites.gatech.edu/nagi/publications/)。搜尋可核書目，直接開頁失敗；未取得全文，無支持頁碼 | 補為明確背景候選。舊裸引Gebraeel2005沒有題名，不能保證這就是原指涉。main沒有此Bayesian RUL實作，不能支持三馬達拼成生命週期。 |
| Tianyi Wang, Jianbo Yu, David Siegel, Jay Lee (2008), A Similarity-Based Prognostics Approach for Remaining Useful Life Estimation of Engineered Systems, PHM 2008；DOI 10.1109/PHM.2008.4711421 | [IEEE書目](https://ieeexplore.ieee.org/document/4711421)、[作者上傳原文](https://www.researchgate.net/profile/Jay-Lee-27/publication/269167324_A_Similarity-Based_Prognostics_Approach_for_Remaining_Useful_Life_Estimation_of_Engineered_Systems/links/548349f50cf2f5dd63a91127/A-Similarity-Based-Prognostics-Approach-for-Remaining-Useful-Life-Estimation-of-Engineered-Systems.pdf)。本輪讀p.1摘要與§I，使用同個體歷程及失效歷程資料 | 此候選原文局部已核實；舊裸引Wang2008映射仍尚無法核實。沒有實作RUL，現有T1/T2/T3不是run-to-failure模板。 |
| Ole Ledoit & Michael Wolf (2004), A well-conditioned estimator for large-dimensional covariance matrices, J. Multivariate Analysis 88(2):365–411；DOI 10.1016/S0047-259X(03)00096-4 | [作者摘要](https://www.ledoit.net/ole1_abstract.htm)、[作者版本清單](https://ledoit.net/research.htm)。本輪核作者摘要與書目；2001稿不得冒充2004出版全文 | `core/mahalanobis.py:82`用sklearn LedoitWolf。收縮背景只有摘要支持，本輪未重做library逐式證明；不改正式LW預設，也不推導未知識別保證。 |
| Thomas M. Cover & Peter E. Hart (1967), Nearest Neighbor Pattern Classification, IEEE Transactions on Information Theory 13(1):21–27；DOI 10.1109/TIT.1967.1053964 | [作者機構原文](https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf)，本輪讀p.21–22，主題為最近鄰分類及漸近分類誤差 | `core/openset.py:108,142,162`使用逐類平均鄰居距離／已知校準分位數。原文不等同此開集規則，不能把其分類界限轉成未知召回保證。 |
| Ricardo J. G. B. Campello, Davoud Moulavi, Joerg Sander (2013), Density-Based Clustering Based on Hierarchical Density Estimates, PAKDD, LNCS7819:160–172；DOI 10.1007/978-3-642-37456-2_14 | [Springer原始書目與摘要](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)；全文需權限，本輪只有摘要 | exp2用hdbscan library；密度階層背景只有摘要支持，25/3→15/3→10/2、隔離線2、10筆節流及確認均屬專案設計，未宣稱論文證明這些值。 |
| James Kirkpatrick et al. (2017), Overcoming catastrophic forgetting in neural networks, PNAS；DOI 10.1073/pnas.1611835114 | [作者預印本摘要](https://arxiv.org/abs/1612.00796)，本輪未全文核對 | EWC以神經網路參數重要度限制改動；main全池重擬合未實作EWC。只有摘要支持背景；刪除「迴避災難性遺忘」保證。 |
| Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert (2017), iCaRL: Incremental Classifier and Representation Learning, CVPR | [作者預印本摘要](https://arxiv.org/abs/1611.07725)，本輪未全文核對 | main手工特徵＋完整配置池重擬合，不是iCaRL表徵學習／exemplar流程。只有摘要支持背景，不能宣稱方法相同。 |

exp7的OAS／MCD、README其他PHM綜述維持歷史目錄判定；本輪exp24未增加它們的效能主張，沒有重新標成已全文核實。Ye & Xie (2015)裸引也沒有足夠題名映射，本輪從未來工作表移至此剩餘清單，不猜配。

## 原始論文的原始引用仍缺頁

唯讀固定 `1ef4a891ae02dc95910f747a5163bb2edd608f28`：[論文全文.md](https://github.com/Drone-Motors-Unknown-Faults-Detection/Ancestor/blob/1ef4a891ae02dc95910f747a5163bb2edd608f28/論文全文.md)，blob `328d0243b12302104f9b32f7b67020bbaad4758c`；html blob `528cefa72d020892268b0ff359855b05bba3d5f6`。轉錄的參考文獻節只指向「論文原稿p.89以後」。本輪沒有取得那幾頁，不能把指向頁碼寫成已讀支持位置，也不能推測裸引對應作者。若要完成這部分，只需原稿完整參考表及相關引用段落；不需要重新提供馬達資料。

## 已修正與待發布草稿

README改為EWMA旗標＋CUSUM展示；全配置池重擬合明列為oracle模擬，不保證消除遺忘；RUL未實作，T1/T2/T3是不同個體，不能串成退化軌跡。exp3手冊補原文與專案設計差異。未改演算法、資料、門檻或準確率。

給#19的草稿：本輪補查exp24繼承引用與未解決項，修正持續學習與RUL過度主張。Roberts遞迴已對原文；Page全文、Gebraeel候選全文／舊指涉、Wang舊指涉、Ye與原始論文參考表仍未核實。#19應保留未完成，不建議關閉。此草稿未發布。

## 2026-10-23：新增引用差量與現行程式對照

### 掃描範圍

本節核對 main `ad26b1d1fced2bedc8ede123d020f1412cfe845c`、文獻 PR #211 的 `52a433970da77d9e6e5a18a1877550c36bcce555`、固定研究版 `b3d68f145cfa187e208e38cae74bf33f68d7c367`。Original-Paper 遠端 main 仍為 `1ef4a891ae02dc95910f747a5163bb2edd608f28`，與原稽核版本相同；沒有新的原稿 PDF 或參考表。

- main：`git ls-files` 篩出 240 個追蹤的 `.md/.py/.js/.cjs/.sh/.toml/.yml/.yaml/.txt/.rst/.mmd/.html`，用 `rg --no-ignore -n -i` 擷取 DOI、arXiv／學術網址及作者年份線索；另讀正式報告與方法使用處。共 74 行 DOI／arXiv 線索、31 個 DOI 候選；候選含書目、重印與重複入口，不是 31 篇新論文。
- PR #211：唯讀其唯一報告全文與引用 [1]～[16]；與最新 main 的封面、誌謝、摘要及圖說分開，不以尚未合併的文獻冒稱 main 已有正文。
- 固定研究版：在 README、AGENT、docs、core、experiments、web、reports 的同類追蹤文字檔共 356 個中，`git grep` 得 232 行方法／學術網址線索，並對原清冊已掃的 `e58e37307af4f10655f05c672c84623867c98c6f` 做差量。新增導航及掃描工具的 URL 不另計方法；B26／B27／B34～B36 等既有內容卡沿用，沒有重標全文已核實。
- 排除 data、logs、output、private sidecar、權重、ZIP、圖片、PDF／Word 二進位及 notebook outputs。main 沒有追蹤的 notebook；Original-Paper 的 notebook source 與轉錄沿固定版既有清冊，不重新執行。這是上述文字範圍的差量，不是全 Git 歷史、所有 PDF 或所有論文全文驗收。

原 117 條清冊及 2026-10-17 Andy 的人工「存在性查核」保持原判定。下方新增 ID 接續舊目錄最大的 P125，只供本 follow-up 定位，不覆寫固定 `paper_catalog.json`，也不將 117 加上新行數當成全文閱讀數。

### 新來源與原文取得程度

| ID／出現位置 | 書目與本輪取得程度 | 主張、原文支持與程式對照 |
| --- | --- | --- |
| P126；總手冊第6章、成果報告 AE 列、PR #211 §2.4.4／[15] | Hinton、Salakhutdinov（2006），*Reducing the Dimensionality of Data with Neural Networks*，Science 313(5786):504–507，DOI 10.1126/science.1127647。[作者稿](https://www.cs.toronto.edu/~hinton/absps/science.pdf)可讀；本輪核 p.504–505 的重建與分層預訓練說明，PDF 開頭混有前一篇文章末段，未將它當 AE 內容 | 僅支持瓶頸重建背景。[MLPAutoencoderDet](../../src/core/detectors.py)為 `(64,16,64)`、ReLU、`MLPRegressor(X,X)`，不是原作 RBM 預訓練；內部早停與外部 cal 分開。有限背景相符，未宣稱重現原作或首先發明 AE。 |
| P127；[實驗九來源](../../docs/experiments/exp9_autoencoder.md#來源與程式碼) | Mayu Sakurada、Takehisa Yairi（2014），*Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction*，MLSDA 2014，4–11，2 December 2014，DOI 10.1145/2689746.2689747。[Crossref 出版者登錄](https://api.crossref.org/works/10.1145%2F2689746.2689747)已直接取得題名、作者、會議、頁次與日期；ACM 頁取得失敗，全文未讀 | 書目 VERIFIED；內容 UNVERIFIED。手冊只列重建異常偵測背景，不支持本案網路大小、ReLU、q95 或實測成績。不能由 Crossref 登錄升為逐式重現，仍需合法全文核對。 |
| P128；總手冊末章、PR #211 §2.4.2／[12] | McInnes、Healy、Astels（2017），*hdbscan: Hierarchical density based clustering*，JOSS 2(11),205，21 March 2017，DOI 10.21105/joss.00205。[原刊及兩頁軟體文](https://joss.theoj.org/papers/10.21105/joss.00205)已讀 Summary 與 References | 軟體文支持密度階層／穩定群背景，明指 Campello 2013 為方法來源，沿 S07，不合併年份。[exp2](../../src/experiments/exp2_scale_growth.py)隔離線2、密度階梯、候選筆數、邊界退回及 prune 是專案規則；沒有病因或未知召回保證。 |
| P129；[實驗十一來源](../../docs/experiments/exp11_ancestor_comparison.md#來源與程式碼) | Frank Wilcoxon（1945），*Individual Comparisons by Ranking Methods*，Biometrics Bulletin 1(6):80–83，December 1945，DOI 10.2307/3001968。[原刊目錄](https://www.jstor.org/stable/i350470)核實書目；文章頁未取得全文 | 原文逐式查核未完成。[exp11 `_paired_wilcoxon`](../../src/experiments/exp11_ancestor_comparison.py)使用 [SciPy 1.15.3 `wilcoxon`](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.wilcoxon.html)；官方 Notes 要求成對差值的獨立結構。相同馬達／資料反覆 seed 不因此成為獨立個體，不將 p 值當跨馬達母體保證。未改原 p 值或歷史輸出。 |
| P130；PR #211 §2.2／[2] | Chen 等（2024），*Fault diagnosis of drone motors driven by current signal data with few samples*，Measurement Science and Technology 35(8),086202，DOI 10.1088/1361-6501/ad3d00。[出版者入口](https://iopscience.iop.org/article/10.1088/1361-6501/ad3d00)本輪受 robots 限制；沿 PR #211 的 2026-10-22 出版頁／可得方法段核對紀錄，不稱本輪重新取得全文 | 2.2 限定為電流／少樣本 BLSCNN 與本案特徵、配置及評估角色不同；沒有實作 BLSCNN，也不把對方分類成績與本案 unknown recall 直接排名。全文取得與資料細節仍需依合法原文複驗。 |
| P131；PR #211 §2.3／[5] | Bendale、Boult（2015），*Towards Open World Recognition*，CVPR:1893–1902，DOI 10.1109/CVPR.2015.7298799。[作者 v1](https://arxiv.org/html/1412.5687v1)已核 Introduction／Open World 定義與 NNO；2014 預印本與 2015 會議分開 | 支持未知拒絕與納入新類的問題界線。本案未實作 NNO；`adopt()` 與 `confirm()` 各自的資料來源見下節，不能直接繼承該論文 open-space-risk 理論條件。 |
| P132；總手冊 VGG 段與原始論文架構背景 | Simonyan、Zisserman，*Very Deep Convolutional Networks for Large-Scale Image Recognition*，[作者 v6](https://arxiv.org/abs/1409.1556)：首版2014，修訂2015；本輪核作者書目／摘要，不是全文逐層核對 | 摘要的 3×3 影像卷積與深度不能證明原始論文 1D notebook 的結構與成績。只保留背景；原始程式需逐 notebook 核層數，未以原作 ImageNet 成績背書。 |
| P133；總手冊 ResNet 段與原始論文架構背景 | He、Zhang、Ren、Sun，*Deep Residual Learning for Image Recognition*，[作者 v1](https://arxiv.org/abs/1512.03385)首版2015，正式 CVPR2016；本輪核書目／摘要 | 支持殘差架構背景，不能把原影像網路與本案 105 維 1D 改編當成同一實驗。本輪未重現或重新訓練原始論文網路。 |
| P134；PR #211 [9]／[16] | scikit-learn developers，[OAS 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.covariance.OAS.html)與 [MLPRegressor 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.neural_network.MLPRegressor.html)，軟體 API 頁而非新論文。MLP 本輪重讀 Parameters／validation score；OAS 公式差異沿 PR #211 已核紀錄 | 使用版本1.7.2。MLP 早停以內部驗證分數控制迭代，外部 known cal 才定拒絕線；OAS API 省略 Eq.(23) 的小項，不能稱逐式相同。兩個網頁在正式報告各有條目，此 ID 只作軟體族對照，不虛增論文篇數。 |

CORAL 的新 arXiv URL 沿用舊 P077：[AAAI 原刊](https://ojs.aaai.org/index.php/AAAI/article/view/10306)確認 Sun、Feng、Saenko、2016、DOI 10.1609/aaai.v30i1.10306；本輪讀摘要的二階統計對齊。[exp10](../../src/experiments/exp10_transfer.py)只有目標支援集與來源 train 的 median 平移，沒有白化／重著色；現行手冊已明寫不是完整 CORAL。S14 的 SVDD 補 DOI 10.1023/B:MACH.0000008084.60811.49，仍是同一來源，不因新增 DOI 另配 ID。Gebraeel／Wang 候選 DOI 分別放回 S08／S09 的「待確認指涉」，不是新證據證明原裸引唯一答案。

### 現行主張與保留限制

main@ad26b1d 的 exp2 `adopt()` 只用候選群到達列建 `discovered-N`，以 known cal 最大分數做邊界退回，更新後清除已被解釋殘餘，另留20%配置列做外部評估；`matched`／`cluster_purity` 僅事後對照。首頁及 guide 的 `confirm()` 仍讀完整標記配置池。這兩條程式路徑均不等同 EWC、iCaRL 或 NNO，也沒有消除遺忘／跨馬達可靠性保證。上方2026-10-08的全文池描述屬該版紀錄，不套用到現行批次 `adopt()`。

B02／B03／B10／B15、P027／P033／P040、S07／S14 以原 ID 對照 PR #211 的方法節；1936 Mahalanobis、1901 Pearson、2013 HDBSCAN 方法原文及 Page 1954 的全文取得限制仍保留。JOSS 軟體文可讀不代表已取得2013全文，2019重印 DOI 不代表1936首次出版。

本輪沒有發現需要改演算法才能配合引用的理由。**#19仍未符合「所有內容相符」**：需原稿p.89後完整參考表與相關段落、原撰寫者確認 S08／S09／S16 裸引或明確重新選文，以及受限全文的合法取得。新增差量的有限背景可供後續正文使用；不能把本表、書目存在、模型已執行或 CI 成功當成全部內容已核實。沒有公開 PDF、私人路徑、原始 log 或新增研究成績。
