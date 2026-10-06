# ws/hetero-proxy — JOURNAL

Append-only。每條：日期、角色、做了什麼、卡在哪、留給下個 session 的話。
需要 boss 裁決的事項標 **「需 boss 裁決」**。

---

## 2026-09-23 · boss · 建立

- 建立 workstream，寫入 STATUS 第 1 節。
- 1a 的三組配置與判讀邏輯已寫在 STATUS；`PHASE0_FINDINGS.md` 2.4 標為推論的那句
  就是 1a 要驗證的。
- 留給 worker：Phase 0 的兩個腳本要複製進本 ws 再改，不要改 `experiments/` 底下的原檔。

## 2026-10-06 · boss 兼 worker · 教授新方向；文獻與開源調查

- **教授新方向**：PCIe 也要能用 Flux 的邏輯做通訊 / 計算重疊，配合 NVLink 環境；節點外的不同廠牌 PCIe 晶片要能協同。由本 ws 承接（與第 1 節目標一致）。
- **調查**：三路平行（repo 程式碼、論文、開源），子代理擷取原文核對；FlexLink、ConCCL、FlagCX 三項由我親自抽查原文。報告 `reports/20261006_survey_pcie_overlap_hetero.md`。
- **關鍵發現**：
  - Flux 的重疊機制與傳輸無關（tile 等 GPU 記憶體旗標）→ 等級 1 的切入點。
  - NVIDIA 之間的 PCIe + Flux 重疊已存在；NVLink 節點內的 PCIe 額外頻寬上限低（FlexLink +26%、D-001）。
  - 研究空白：不同廠牌 PCIe 晶片以 tile 級重疊加入 NVLink 域；同機跨廠牌 PCIe P2P 無人量過；無開源跨廠牌共用 tile 旗標。
- **卡在哪**：方向解讀（連外部晶片的通道 vs 節點內額外頻寬）與晶片型號 / 軟體堆疊需教授確認。
- **留給下個 session**：確認方向後，先做 1a / 1b，再做「能否從 Flux 外部寫 AG-GEMM 旗標」的技術探針。等級 0 / 1 只需 css-host-158 單機。
