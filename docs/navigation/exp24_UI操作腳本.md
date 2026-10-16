# UI操作腳本

獨立入口 web.guide 重用 LiveDemo／Hub；原 web.server 展示仍可使用。啟動方式見[導覽入口](exp24_README.md#啟動與操作)。

首頁只有三個主要操作：冷啟動、未知學習、漸進／突發比較。三條流程共用一session；切頁不新增worker。展示／研究切換只改畫面，不送重設或擬合指令。研究頁另連實驗總覽、來源及原研究入口。

| 步驟ID／前置 | 按什麼 | 資料角色 | 呼叫／事件 | 畫面 | 判斷原因 | 可信度／限制 | 下一步／錯誤與復原 |
|---|---|---|---|---|---|---|---|
| U0未選資料 | 三入口之一 | catalog motor/RPM | state | 三入口、資料選單 | 尚無模型 | CSV重播，不連硬體 | 選資料；空catalog阻止 |
| U1待建基準 | 確認所選資料／建立健康基準 | healthy；顯示105維、各匿名配置筆數、CSV SHA、seed | inspect檢查；build→busy→state擬合 | 建基準中，其他操作停用 | 檔案載入／既有fit | 驗證105數值欄、過濾非有限列，各池至少10筆；不證明物理語意一致 | 失敗顯示error、重選或重試 |
| U2可輸入 | 開始健康重播 | healthy holdout；train/cal不重播 | start／sample | 分數主圖、normalized threshold=1 | core.openset score與classify | 本工況展示；非跨motor可靠性 | 暫停／改匿名訊號來源 |
| U3監測／累積 | 輸入匿名異常來源 | 未知配置全池，從未fit | source→tick | 未知/已知文字、隔離數、分群嘗試 | >1拒絕、>2隔離；核心HDBSCAN | unknown不等已確認原因；感測品質未驗證 | 無候選等待，沒有強制確認 |
| U4待確認 | 模擬操作員確認 | 候選ID、size；答案尚不顯示 | confirm(candidate_id)→busy | 更新中 | 現有confirm揭示配置／判斷退回 | 後端已有truth快取；僅UI遮蔽 | stale/重複確認拒絕；已知群退回 |
| U5已更新 | 繼續監測 | 原新配置成已知，從holdout重播 | existing confirm／state | known類別、新模型、前後接受率 | 完整標記配置池重擬合 | 接受為已知不等自身分類正確、不等健康恢復 | 失敗暫停，明確重建；不可自動重試fit |
| T1基準存在 | 漸進A／突發B | SCENARIOS固定窗口抽樣 | scenario | EWMA主圖、警報種類／筆數 | alpha .08、帶.2–.5、≤12筆突發 | 合成順序，非實際磨損時間 | A260/B160筆完成自動暫停；切劇本先重設以作獨立展示 |
| R任意狀態 | 研究模式 | 同一session | 無計算命令 | 參數、來源、split indices、模型threshold、原始事件摘要與實驗表 | 後端既有summary | 未提供raw距離/最近中心/confusion顯示未計算，不編造 | 返回展示保留t/epoch/模型 |
| P監測中 | 暫停／繼續 | 不換順序 | pause/start | 暫停文字 | 不再呼叫tick | 取消代表暫停，不是完整run | 繼續沿同sampler |
| E斷線／error | 重連 | 既有session | websocket open | 自動暫停、現存state | 最後一個client離線停止tick | 已送出重任務不中止，完成後保留狀態 | 重連不fit、不自動start |
| X重看／換motor/RPM | 明確重建／重設 | 新基準重新fit | build/reset | epoch／session更新 | 防止舊model沿用新工況 | 舊log不覆蓋，新session目錄 | busy拒絕重複，手動建立 |

候選名只使用群組ID；匿名異常來源可按catalog順序指定，但不把資料集答案當系統預測。操作者確認是模擬truth揭示，沒有自由文字標註或合併群操作。已知配置在確認後可顯示原代碼，不使用缺真值的嚴重度名稱。

每屏五項：步驟、資料、判斷、原因、限制。score是normalized分數；raw距離／最近中心目前LiveDemo事件未提供，研究模式明示缺項。現有未校準health百分比不傳到新頁；顯示分數和閾值，不寫損壞百分比。sample事件的score沿用既有三位小數，顯示1.000可能來自略高於1的真值；核心用未四捨五入的分數判定，不以畫面數字重新判斷。

程式核對補充：cluster_attempts是連續分群失敗次數，成功即歸零，不能當總嘗試。新頁使用精確名稱；候選t_range是重播筆數索引，不是raw時間。inspect先唯讀檢查，build才擬合。未完成模型時不可播放劇本；有錯誤時先明確重建／重設。

速度只調 tick 間隔，不改樣本順序或門檻。沒有候選就等待或暫停，不降低門檻。操作示例以 T1/8000rpm、seed42、LW/q95 為條件；實際筆數以資料檢查與 session 紀錄為準。這些操作說明不構成新一次模型評估。

### 暫停與保存

樣本以 session 目錄、epoch、t 識別。`LiveDemo.tick()` 同步完成預測與 CSV 寫入後，才送出 sample；每 15 筆 flush 一次。`Hub.pause()` 停止新 tick，再 flush 尾批，成功後才回覆暫停。最後一個 WebSocket 離線會保存並保留模型，重連不重訓。正常停止 IOLoop 或 Ctrl+C 經 `serve()` 等背景工作完成後關檔。

原首頁換馬達／轉速時，共用該次執行的輸出目錄；新 `LiveDemo` 跳過已有 CSV、模型或 metadata 的 epoch，以排他建立 CSV 保留舊檔。導覽頁重建則使用新的 session 子目錄。兩個入口都不以切換資料集為由覆寫已保存紀錄。

暫停 state 內含同一 t 的 `metrics` 與 `persistence`，前端立即套用，不等下一個 15 筆批次。統計分為學會前／學會後的已處理筆數，各來源兩欄合計應等於 session 的 t；隔離數是其中符合隔離條件的樣本，事件數包含控制與警報，不等於樣本數。導覽頁忽略舊 session／epoch／t 與舊連線的統計，不能讓延遲事件把最終值倒退。

`persistence.written_t` 是已交給 CSV writer 的最後索引，`flushed_t` 是已完成 flush 的索引；未設定輸出目錄時 `enabled=false`，不能宣稱已保存。保存失敗會留下錯誤、停止串流並阻止直接恢復。重複暫停／關閉不追加樣本。此契約保證正常暫停／關閉後可重新開檔讀取，不包含強制終止、作業系統故障或掉電的零遺失保證；沒有執行 fsync。

### 來源恢復

重連以後端 state 的 session／epoch、馬達／轉速、source 與方法為準。選單尚未套用的值另標「待套用」，按「輸入所選來源」才會改後端。來源不在清單時顯示不可用選項，不自動切回健康資料；舊 WebSocket 的事件不更新新連線。每次 flush 同時寫 `session_epochNN.json`，記錄實際配置代碼、工況、方法與已保存索引；此檔只供伺服器端追溯，導覽 UI 仍遮蔽未知配置答案。前端回歸測試由 pytest 呼叫 Node.js 執行真正的 `guide.js`，需要 PATH 中有 Node.js。

空清單時不能建立基準。`data_root_missing` 表示目錄不存在，`no_datasets` 表示沒有掃到可用資料集；兩者皆需管理者核對 `--data-root` 與唯讀正式 CSV、重啟後再重新整理頁面。頁面沒有重新掃描伺服器清單的按鈕。已有清單但檢查失敗時，可改選工況或按「確認所選資料」重試；讀取權限、零筆／不足筆數或缺少健康配置不能靠啟用開始按鈕繞過。載入中請等待；已建基準且 t=0 只表示尚未重播。
