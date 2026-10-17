# Lineage 導覽入口

給老師先讀：[實驗總覽](exp13_實驗總覽.md) → [結果白話解讀](exp13_結果白話解讀.md) → [研究流程圖](exp13_研究流程圖.md) → [示範腳本](exp13_示範腳本.md)。數值由[來源索引](exp13_來源索引.md)查閱。

## 啟動與操作

依[環境政策](../runtime_policy.md)準備 uv 與已鎖版環境，在 checkout 根目錄執行：

```bash
uv run --locked python -m web.guide --data-root data --port 8601 --seed 42
```

開啟 http://127.0.0.1:8601 。資料根目錄可改成實際唯讀位置；空資料目錄顯示「缺少資料」，不能建立模型。先選 motor／RPM，按「確認所選資料」，核對筆數與 SHA 後才建立健康基準。預設 Mahalanobis–LW、q95；k-NN 對照在啟動時加 --openset-method knn，仍使用 core.openset factory。更換方法需明確重啟服務，不暗改現有 session。

三個入口為冷啟動、未知學習、漸進／突發比較，共用同一 session。詳見[UI 操作腳本](exp13_UI操作腳本.md)。研究模式展開參數及七問表，切換模式不重擬合、不改計算。原 python -m web.server／8600 展示及 PolarMap 仍可使用；本入口不取代它們。

## 復原與限制

暫停後可沿原順序繼續；重設或換工況會明確重建。斷線後不自動播放。更新失敗先停止，按錯誤狀態明確重建，不連續重送確認；[共用模型生命週期](../experiments/README.md#共用模型生命週期)說明重訓失敗的限制。

輸入是 CSV 重播，沒有連接實體馬達。健康-only 展示與跨馬達多類研究分開；學前未知拒絕、學後已知接受、自身配置分類 accuracy 的分母不同。沒有物理損壞百分比、RUL 或部署保證。歷史人工驗收見[固定導覽報告](https://github.com/Drone-Motors-Unknown-Faults-Detection/Lineage/blob/b3d68f145cfa187e208e38cae74bf33f68d7c367/reports/navigation_20261007/final_report.md)，不代表目前所有環境已重新驗收。
