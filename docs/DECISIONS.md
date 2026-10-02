# DECISIONS.md — 決策日誌

**Append-only。boss session 專屬。** 每條決策附：日期、決策、依據（指向原始記錄）、
審查狀態、什麼條件下重開。編號不重用。

---

## D-001 · 2026-09-23 · 取消 A100 NVLink+PCIe 雙通道（Phase 1 / Phase 2）

**決策**：不在 A100 上實作 NVLink 與 PCIe 主機中轉並行的雙通道傳輸。原規劃的
Phase 1（Flux 內雙通道排程）與 Phase 2（調優）取消。

**依據**（`docs/PHASE0_FINDINGS.md` 第 1 節；原始資料 `experiments/2026-09-23-phase0-*/`）：
1. NVLink 217.66 GB/s/卡 vs staging 5.3 GB/s/卡，理論上限 ≈ +2.6%。
2. 實測 α 掃描每一點皆淨損失且單調，有無 switch-aware ring 皆同（D_CORRECTION D.2、D.6）。
3. 在 N=49152，AG 快 17% 只換到端到端 0.9%（FLUX_BASELINE F.5）。

**審查狀態**：Phase 0 經四輪自我修正，但**尚未經獨立 auditor 審查**。本決策先行生效，
因為三條依據方向一致且第二條是直接量測；若 auditor 推翻其中任一條，重開。

**重開條件**：出現兩條路徑頻寬在同一數量級的情境（無 NVLink 機器、PCIe-only 晶片、
跨節點）。這不是重開 A100 雙通道，而是異質計算本身——見 `ws/hetero-proxy`。

## D-002 · 2026-09-23 · 撤回 Phase 0 原始報告的多項數字

**決策**：`experiments/2026-09-23-phase0-*/README.md`（初版）與 `CDE_REPORT.md` 的
D、E 段及雙向表格全數撤回，以 `docs/PHASE0_STATUS.md` 第 3 節的撤回表為準。撤回的
腳本保留供稽核，不刪除。

**依據**：`PHASE0_STATUS.md` 第 3 節（16 條撤回主張，各附原因與替代來源）。

**審查狀態**：撤回本身即修正的產物；替代數字尚未經獨立審查（見 D-001）。

## D-003 · 2026-09-23 · 設立三個 workstream

**決策**：建立 `ws/diag-overlap`（最高優先）、`ws/hetero-proxy`（第二，可並行）、
`ws/cost-model`（第三，依賴前兩者）。目標與成功標準寫在各自 STATUS.md 第 1 節。

**依據**：
- diag-overlap：N=4096 overlap efficiency ≈ 2%，總時間距完全串行 1.3%（FLUX_BASELINE F.4）。
  這是最大已知改善空間，也是異質計算的前置問題。
- hetero-proxy：專案最終目標；Phase 0 的 PCIe 特性是其設計輸入（PHASE0_FINDINGS 第 3 節）。
- cost-model：把量測整合為可預測模型，用 diag-overlap 實測驗證。

**審查狀態**：不適用（組織決策）。

## D-004 · 2026-09-23 · 確立協作制度與量測準則

**決策**：採用 `CLAUDE.md` 所述之 boss / worker / auditor 三角色制度、收工程序、
檔案架構，以及第 5 節的十條量測準則與報告規範。**任何要進本日誌的技術結論，之前必須
經過至少一次獨立 auditor 審查**（D-001、D-002 為制度建立前的例外，已註明）。

**依據**：Phase 0 四輪修正的共同失敗模式——基準拿錯、證據在自己輸出裡未被讀。
準則逐條對應：基準獨立（缺陷 1）、錨點對照（缺陷 2）、自我一致與效率 > 100% 停
（缺陷 3）、差值交錯（缺陷 4）。

**審查狀態**：不適用（制度決策）。

## D-005 · 2026-09-23 · 這台機器上 Flux AG 預設模式改為 All2All + use_read

**決策**：後續所有 Flux AG 量測與開發，`AllGatherOption.mode = All2All`、
`use_read = True`。不再使用 Ring2D 作為基準。

**依據**：E_CORRECTION E.1 — All2All pull 在 M=1024–16384 全程最快，比 Ring2D 快
17–20%、比 NCCL 快 13–22%；Ring2D 走 `copy_ring_push_2d_pcie`，為 PCIe 拓撲設計，
在 NVSwitch 上無益。FLUX_BASELINE F.5 — 端到端在 N=4096 差 10.7%。

**審查狀態**：未經獨立審查，但為低風險的配置選擇，且所有配置皆經 bitwise 驗證。

**重開條件**：換機器（非 NVSwitch 拓撲）時重新掃描模式。

## D-006 · 2026-09-29 · 新增 ws/fusion-dispatch 並列為最高優先

**決策**：
1. 建立 `ws/fusion-dispatch`：量出推論時「每層 × 每個 M」Flux fused 與非 fused 路徑的勝負地圖，
   評估何時不開 Flux 比開好、以及用哪種決策器。優先於 `ws/diag-overlap`。
2. **量測證據顯示值得之前，不改 Flux `src/`**（包含 A\* arm 所需的 pybind 改動）。
   先用不需 rebuild 的方法（斷崖對、A vs D 分解）判斷。

**依據**：
- 與教授討論後的方向調整（使用者，2026-09-29）。
- 論文 Section 6 / Fig. 14、Section 5.2 / Fig. 17：極小 m 時 Flux 可能慢於不 overlap 基準。
- Phase 0 沒有任何 M < 1024 的數字，也沒有獨立的 cuBLAS GEMM 時間，無法從既有資料設計決策器
  （`ws/fusion-dispatch/reports/20260929_experiment_design.md` 第 2 節）。

**審查狀態**：不適用（組織決策）。

**重開條件**：E1 顯示在 [8, 16384] 內所有層都沒有切換點且 π_on 的 regret < 1% → 決策器無價值，
fusion-dispatch 結案，diag-overlap 回到最高優先。

## D-007 · 2026-09-30 · fusion-dispatch 第二階段：決策器與 Flux 弱點優化

**決策**：
1. 核准 `ws/fusion-dispatch/reports/20260930_plan_dispatcher.md`，順序 F0 → F1 → F2 / F3 → F4（使用者，2026-09-30）。
2. **目標推論框架：vLLM 0.8.5.post1**（由 boss session 選定，使用者授權）。理由：
   - 支援 torch 2.6.0 的最後一版，與本機 Flux build 用的 PyPI torch 2.6.0+cu124（CXX11 ABI=0）相容，不需重編 Flux；
   - 最常用的服務框架；
   - 論文 decoding 的不 overlap 基準即 vLLM。
   安裝在獨立 venv，不動 Flux 的 pixi 環境。
3. 因為 vLLM 預設的 TP 是「每卡完整 token + row-parallel 後 AllReduce」**[推論，F4 安裝後讀原始碼確認]**，
   決策器的 off 選項加入「TP + AllReduce」路徑；比較單位為整個 block。
4. D-006 延伸：計劃的關卡 G2 / G3 通過後，允許修改 Flux `src/`。改前打 git tag，改後做 bitwise 驗證。

**依據**：E1–E3（`ws/fusion-dispatch/reports/20260930_e1_e3_dispatch_map.md`）；
環境檢查（2026-09-30：pixi 的 torch 2.6.0+cu124 來自 PyPI，`_GLIBCXX_USE_CXX11_ABI=False`；PyPI 可連線）。

**審查狀態**：不適用（組織決策）。E1–E3 的技術結論仍待 auditor（D-004）。

**重開條件**：vLLM 0.8.5 在本機無法與 Flux 共存；或 F0.4 顯示 Flux 無法進 CUDA graph，而目標部署必須用 graph。

