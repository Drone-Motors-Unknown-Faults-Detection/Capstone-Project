# 實驗總覽

先看這張表，再看[來源、逐批與分母](exp13_來源索引.md)及[白話解讀](exp13_結果白話解讀.md)。表中「已驗證」只限該資料／協議；沒有新盲測或現場保證。三顆馬達×三RPM是九工況，九個faulty是螺絲配置，不是九個已確認故障原因。

| 實驗名稱 | 想解決什麼問題？ | 為什麼需要做？ | 使用哪些資料？ | 訓練、校準、測試如何分？ | 使用什麼演算法？ | 成功標準是什麼？ | 實際結果與限制是什麼？ |
|---|---|---|---|---|---|---|---|
| 冷啟動 exp1〔S1〕 | 只有健康，能否拒絕異常？ | 新設備常無已知故障 | 舊T1/8000，105維，健康313、九配置2736筆，seed42 | 健康187 train／62 cal／64 holdout；faulty全池只測試 | 穩健縮放＋LW馬氏距離，cal第95百分位 | 健康誤報低且未知召回高；沒有事前跨馬達達標值 | 同工況拒絕與健康誤報均有封存結果；不支持九類識別或固定5%誤報保證，見S1。 |
| 歷史深層特徵共變異數〔S14〕 | 舊神經模型的開集規則能否改善？ | 先保存早期LW採用依據，避免套在新資料 | 歷史T1/T3×CNN/ResNet/VGG16×三RPM，18模型；深層176/128/48維，每配置最多300筆，非本輪formal105模型 | healthy＋四known faulty分層60/20/20、random_state42；另外五unknown只測試；legacy用train閾值、改善法獨立cal | Legacy／LW／OAS／PCA後MCD；分類器未重訓 | 當時方法比較；未事前宣告跨motor可靠性標準 | 歷史文件回報；本次未逐筆重算，不把深層特徵二元成績套到formal105多類研究，見S14。 |
| 開集六方法冷啟動〔S5〕 | 哪種拒絕規則較合適？ | 高排序分數不代表門檻好用 | P1正式105維，3 seeds，固定immutable test | healthy-only train/cal、相同budget及test；資料ID見P1 | Maha-LW、kNN、OC-SVM、IF、LOF、PCA重建 | 多指標與健康誤報並列，未宣告部署達標 | 固定門檻召回與排序指標可分歧，不能排成通用最佳；各方法完整數值見S5。 |
| LW及校準消融〔S5〕 | 共變異數與門檻有何影響？ | 105維小樣本會使估計不穩 | 相同P1三seed、健康＋未知舊資料 | healthy train、專用known cal；val/test只評估 | legacy／LW×q90/95/97.5/99及conformal | 相同用途配對，保留誤報和召回；無普遍5%規則 | 共變異數與校準都有配對記錄；流程同時改動的差異不全歸因LW，見S5。 |
| 同工況Mahalanobis／kNN〔S4〕 | 兩種正式detector如何不同？ | factory要可公平切換 | 九工況、105維及既有seeds | 各工況known60/20/20；相同sample roles | 逐類LW距離／逐類5近鄰平均距離，q95 | 同split、相同分數方向及來源 | 同工況配對輸出可追溯；不能替代跨馬達成績，見S4。 |
| 跨馬達全類別組合〔S6〕 | healthy＋5 known能否拒絕另外4？ | 回答老師未知類別驗證 | formal90CSV、28910筆105維；N5全部126組 | 三documented motor folds；舊validation/cal共用與跨折selector依賴保留 | 線性分類＋factory Maha／kNN | coverage與獨立群要求；全部INCOMPLETE | 全126組已保存；分類與未知排序偏弱，切分及來源限制保留，見S6。 |
| N sweep與Protocol B〔S6〕 | N改變、unknown角色輪換如何？ | 不只挑一組有利類別 | 同formal90；N1–8及N9 closed-set、B unknown rotation | 既有class manifests；各N的組合數不同 | 同線性分類、兩detector | 依原protocol及coverage，不把支援當跑完 | 既有各N與B產物已保存；不同N覆蓋不同，程式支援不等實際全組跑完，見S6。 |
| 固定方法／專用cal〔S7〕 | 排除共同選模型及val/cal共用？ | 修正可確認的評估依賴 | 同formal90、五known／四unknown、seeds0/1/2 | T1train/T2cal/T3test，T2/T3/T1，T3/T1/T2；selection空，unused unknown不fit | 105／75×linear×Maha／kNN | IDs隔離、無selector；fresh/group資格另外判 | 選擇IDs為空、校準專用；相同fit/cal時預測可不變，資料仍已曝光，見S7。 |
| 105與75特徵〔S7〕 | 電流溫度資訊是否幫忙？ | 跨馬達可能受尺度／機台影響 | 同五known／四unknown、三motor，105 vs振動75 | 同S7、train-only scaler，不挑test最佳seed | 相同linear與paired factory detectors | 原固定研究規則，不自動改正式105 | 固定比較有局部改善，弱motor未知仍漏判；不自動替換正式105維，見S7。 |
| 分類器／表示法初輪〔S8〕 | 線性、RBF、樹及表示能否分類？ | 先測簡單候選，避免盲目深模型 | formal105與train-only表示 | 歷史train/val選法，test已曝光；跨折共同selector限制見S7 | linear／RBF SVM／樹、PCA等已登錄候選 | macro-F1／balanced accuracy選法；限制明列 | 歷史選法有跨折依賴；保留限制，不把共同winner當獨立final，見S8。 |
| C／D文獻方法〔S9〕 | 分類與拒絕可否各取所長？ | 兩件事需不同表示 | 同formal五known／四unknown、三motor | 固定候選／no-selection；train與cal用途分離 | LDA、SVM、線性、特徵及RPM策略；D解耦 | 保留每類／每工況／unknown非劣保護 | 解耦可改善部分平均指標，但弱配置與motor未解；沒有可靠winner，見S9。 |
| 機制改編〔S10〕 | 形狀／幅值與近鄰有何缺口？ | 把失敗原因變成可反駁假說 | 同105衍生已鎖表示，無新raw | train-only變換、known cal、固定test | 幅值比值、prototype與距離改編 | 原可靠性規則與配對非劣 | 有限改編未通過完整規則；逐方法改編來源、參數與失敗見S10。 |
| Q與solver〔S11〕 | 局部距離是否受收斂限制？ | 先分工程失敗與模型弱 | 同正式data；solver只train | Q外層用途固定；solver沒有test | 局部度量、hard/smooth損失 | 收斂與outer成績分開 | Q有未完成格；solver收斂不等外層分類改善，見S11。 |
| E–J原型／距離〔S11〕 | 距離、原型與RPM拒絕能否改善？ | 不只重調threshold | 同28910筆，三fold×三seed | 固定train/cal/test；不重用unknown選參 | E收斂kNN、F Fisher、G判別原型、H L1、I RPM原型、J聯合拒絕 | 對照C24 fault+2pp、C17 F1+0.02、每工況健康FP≤10%、每類及每motor未知recall≥10%等 | 各批已保存完成／失敗記錄；有限改編全未通過完整規則，見S11。 |
| K–M不變核／風險／自挑戰〔S11〕 | 保留形狀、跨RPM與去易特徵能否幫忙？ | 測有原始文獻的有限改編 | 同資料、表示均train-only | 同鎖定三fold與seeds0/1/2 | 三軸排列核、REx、表格RSC改編 | 相同完整可靠性規則，不挑最好的項目 | 各批完整結果與弱組保護失敗保留；排列非任意旋轉，步數不等收斂，見S11。 |
| 新配置學習 exp2〔S2〕 | 累積未知成群後能否自學辨識？ | 未知名稱不由人工指定 | 舊T1/8000，九配置依序循環CSV | 初始healthy-only；未知只累積；成群即adopt()只用該群60/20/20重擬合，類名discovered-N | HDBSCAN階梯25/3→15/3→10/2；叢至少25；隔離分數>2 | 找候選、邊界退回、保留healthy；事後才對照真值 | 形成新類不等於對上該段配置；看matched_is_config，見S2。 |
| 到達樣本-only研究〔S12〕 | 不借未到達全池能否保留能力？ | 主線 `adopt()` 只用群內樣本，但沒有記憶預算與固定test | P1 formal、budget10/25/50/100、三seeds四arrival orders | arrival buffer與immutable test分離；full_pool_oracle另列 | historical replay、random／balanced buffer、new-class-only及oracle | 舊類保持、新類、成本與忘記分開 | arrival-only與full_pool_oracle分開；不能互換成績，見S12。 |
| 趨勢 exp3〔S3〕 | 能否分出兩種劇本節奏？ | 警報時間形態與故障名稱不同 | 同工況CSV重排與混入劇本，無真實連續時間 | healthy fit/cal；劇本窗口測試 | EWMA α.08、[.2,.5)中間帶、≤12筆突發；CUSUM展示 | 原手冊期待A/B分開；未固定現場壽命標準 | 兩批劇本結果分開保存；筆數延遲不等真實磨損或碰撞辨識，見S3。 |
| 跨工況 exp5〔S4〕 | 換轉速／馬達基準可否搬移？ | 現場不總是同工況 | 三motor×三RPM | 源工況健康fit/cal→其他工況評估 | 健康距離與接受率矩陣 | 同機台與跨個體分列；無共同部署contract | 源與目標工況矩陣已保存；同工況結果不能替代跨個體可靠性，見S4。 |
| PolarMap exp4〔S4〕 | 方向與距離怎樣呈現？ | 未知方向不等於同種故障更嚴重 | 已知健康／配置105維 | 基準fit資料及顯示用樣本分離看原手冊 | 健康白化、方向投影、極座標幾何 | 圖形與計算一致；沒有病因／嚴重度truth驗收 | 支持幾何視覺化；方向不是病因，半徑不是損壞百分比，見S4。 |
| 健康指數 exp8〔S12〕 | 能否連續描述偏離與告警？ | 二元拒絕以外需要可讀量尺 | 九工況、三seed、兩detectors，54結果列 | 各工況healthy split；非跨motorS7 | 既有health index、校準、平滑與告警 | 原exp8手冊；物理嚴重度缺真值 | 同工況healthy-only與跨motor多類資料指紋不同，不直接排名或推論RUL，見S12。 |
| 場景重播／同步raw〔S12〕 | 噪音、掉點與新來源怎樣受控進入？ | 工程健全性與科學效度分開 | 舊formal重播／synthetic fixture；fresh入口尚無合格新實測 | replay與fresh gates、來源／曝光ledger分開 | 既有場景重播與同步特徵管線 | 同步／schema／曝光守門；非重播當fresh | 工程入口與fixture不等真實fresh final；同步／單位／原始間隔仍需來源證據，見S12。 |
| 獨立性／引用稽核〔S13〕 | 評估是否洩漏、來源是否可信？ | 可跑不代表可信 | formal90、舊manifests／predictions／文獻清冊 | ID交集、fit來源、alias、raw位置及曝光查核 | checksums、metadata證據分級與final guards | 不降門檻；未知採集留UNKNOWN | 可核對ID／SHA與重複；重複為零不證明視窗獨立，原始採集仍UNKNOWN，見S13。 |
| Group DRO等規劃〔S11〕 | 未來是否可提高最弱工況？ | 現有規則未通過 | 本輪無新資料／模型 | 尚未實作 | 已知group loss計畫 | 尚未評估 | 規劃與實測分開；未實作項目不得稱有效或失敗，見S11。 |

主表只有七個問題加實驗名稱。完整指標／來源在附錄，不新增第八問。百分點和AUROC不同；同一批窗口反覆用不增加獨立motor數。原105／linear／LW、kNN factory及PolarMap均未因總覽改動。
