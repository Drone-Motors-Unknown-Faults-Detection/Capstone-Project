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
