# ws/hetero-proxy — STATUS

**本 workstream 唯一權威。** 第 1 節由 boss 寫入，worker 不改；第 2 節起由 worker 維護。

建立：2026-09-23（boss）
最後更新：2026-10-09（boss 兼 worker，後續計劃 H0–H5）
狀態：**調查完成，實作未開始**

---

## 0. 快速跟上（新 session 先讀這節；每次收工更新）

最後更新：2026-10-06

**教授新方向（2026-10-06）**：讓 PCIe 也能用 Flux 的邏輯做通訊 / 計算重疊，配合 NVLink 相連的運算環境；
節點外多一顆只有 PCIe 的不同廠牌晶片也要能協同運作、用 PCIe 溝通。這正是本 workstream 第 1 節的目標，故由本 ws 承接。

**調查結論**（`reports/20261006_survey_pcie_overlap_hetero.md`）：
1. Flux 的重疊只靠 GEMM tile 在 GPU 記憶體裡等旗標（`include/flux/cuda/system_barrier.hpp:38-86`），跟傳輸方式無關 → 任何能寫旗標的傳輸都能驅動重疊。
2. NVIDIA GPU 之間走 PCIe 的 Flux 重疊已存在（Ring1D / Ring2D、PCIe 版 GEMM+RS；A100 PCIe op 1.20–3.25×），不是新貢獻。
3. 在 NVLink 節點內把 PCIe 當額外頻寬：上限低（FlexLink 在 8×H800 只 +26%、只有集合通訊），與 D-001 一致 → 不作主線。
4. **沒人做過**：PCIe 掛的不同廠牌晶片以 tile 級重疊加入 NVLink 域；沒有論文量過同機 NVIDIA ↔ 他廠晶片 PCIe P2P；沒有開源專案跨廠牌共用 tile 旗標。
5. 可借用：copy engine 搬資料（ConCCL：SM 21% vs DMA 72% 理想加速）、主機中轉 + 主機記憶體旗標（ThunderEP、LLMQ）、
   tile 計數觸發傳輸（FlashOverlap，已在昇騰上跑過）、主機代理執行緒（MSCCL++ PortChannel）、跨廠牌傳輸（FlagCX，12 家 CCL）。

**建議路線**（報告第 4 節）：1a / 1b 量測 → 技術探針（能否從 Flux 外部寫 AG-GEMM 的旗標）→ 等級 0（算子級中轉）→
等級 1（chunk 級中轉 + 旗標驅動 GEMM tile）→ NPU 型代理 → 工作切分 → 晶片到手後等級 2。

**實作進度（2026-10-09 使用者核准「開始所有實作」；第一個加速器後端 = CPU，Python 原型）**：
- [x] **H0 工作切分**（`reports/20261009_h0_work_partition.md`）：TP 成員排除（PCIe 每層 0.8–9.2× 域的整層時間，與晶片快慢無關）；
  **pipeline 段可行**，吞吐量上限 +s/8，邊界傳輸佔段時間 0.6–2.4%；平衡時單次延遲約 2×。→ 示範做成「GPU 段 → 晶片段 → GPU 段」雙向 chunk 串流。
- [ ] I1 介面規格　[ ] I2 Bridge + CPU 後端　[ ] I3 chunk 重疊　[ ] I4 A100 代理後端　[ ] I5 NPU 型模擬

**後續計劃（2026-10-09 提案，H0–H5；原則：最便宜的步驟先排除最大的風險，每步產出下一步的基準，預測先登記再量）**：
- **H0 工作切分（不用 GPU）**：晶片當 TP 成員 / pipeline 段 / attention-KV 卸載 / draft 模型，各自的 PCIe 流量與預期效益。
  **為什麼先做 [推論，粗估]**：Qwen2.5-32B TP=4 若晶片是 TP 成員，每層要經 PCIe 收 31.5 MB（M=1024）/ 125.8 MB（M=4096），
  單向最佳 1.42 / 5.67 ms、同時收送 4.87 / 19.48 ms，**不少於** 4 張 A100 算一層的 1.39 / 5.03 ms → 再怎麼重疊都會拖慢。
  當 pipeline 段只在邊界收送 21 / 84 MB（0.94 / 3.78 ms），攤在多層上。角色決定 Flux 式重疊要做在哪種通訊上。
  如何驗證：H0 用成本模型完整估算；H2 的等級 0 實測。
- **H1 三個前提（GPU，小）**：(a) 代理隔離——關 peer access + 主機中轉，`nvidia-smi nvlink -gt d` 計數差值為 0（10-09 確認免 sudo 可讀）；
  (b) 旗標探針——從 Flux 外部寫 AG-GEMM 等待的旗標；(c) 1a / 1b PCIe 特性量測。
- **H2 等級 0**：在 H0 選定的角色上整塊中轉後才算，結果正確 + 量時間 → 等級 1 的基準。
- **H3 等級 1**：FlexLink 式傳輸（雙緩衝、copy engine、GPU 輪詢、單調計數器）接 Flux tile 旗標，資料一落地 GEMM 就開算；反方向用 FlashOverlap 式 tile 計數觸發。
- **H4 NPU 型代理**：固定順序 + 軟體 pipeline（國產晶片多半是 NPU 型，A100 代理太樂觀）。
- **H5 晶片到手**：等級 2 直連可行性；FlagCX 接廠牌 runtime。

**待使用者 / 教授確認**：「PCIe 加入通訊通道」是指連外部晶片的通道（建議主線），還是 NVLink 之外的額外頻寬；晶片的型號與軟體堆疊。

**環境注意**：css-host-159 目前被同帳號的 `sglang::server` 佔用；本 ws 的等級 0 / 1 只需要 css-host-158 單機（以一張關閉 peer access 的 A100 代理晶片）。

---

## 1. 目標與成功標準（boss）

### 問題

Phase 3 的起點：為「PCIe-only 晶片 + A100 NVLink 域」的異質計算建立後端抽象層，
並以 A100 代理尚未到手的晶片。設計輸入是 `docs/PHASE0_FINDINGS.md` 第 3 節的八條規則。

### 步驟

**1. 先補 Phase 0 兩項量測**（各自出一份 `reports/`，數字進 `params.json`）：

- **1a. 雙向競爭位置驗證。** Phase 0 只量到「同一張 GPU 同時 H2D + D2H 時 H2D 掉到
  6.46 GB/s」；競爭發生在哪一層是**推論**。設計：
  - GPU0 只 H2D + GPU2 只 D2H（不同 switch）同時進行；
  - 對照 GPU0 只 H2D + GPU1 只 D2H（同 switch）；
  - 對照 GPU0 單獨 H2D。
  判讀：不同 switch 不降、同 switch 降 → 競爭在 switch 上行；兩者皆不降 → 只在單卡
  端點；兩者皆降 → 主機側。
  工具：`experiments/2026-09-23-phase0-*/scripts/bidirectional_bandwidth_v2.py` 已有
  `--direction h2d|d2h`，但目前**對所有 device 套同一方向**；需改成 per-device 方向
  （小改：把 `direction` 變成 dict）。沿用其逐 copy event + 重疊窗口方法。
- **1b. staging 機制固定開銷。** α=0 但啟用 staging 機制（建立 stream、event、
  pinned buffer、雙緩衝 handshake）相對純 NVLink 的時間增加。工具：`dual_path_v3.py`
  加一個 `--staging-overhead-only` 模式（走完 handshake 但 staging_elems=0 或極小）。
  這個數字是等級 0/1 後端的固定成本項。

**2. 定義後端介面**：註冊 buffer、非同步收送 comm tile、設定 / 等待 signal、
執行計算單位、回報能力。

**3. 能力分級**：
- 等級 0：算子級同步、經 host。
- 等級 1：tile 級 flag、經 host。
- 等級 2：PCIe P2P + kernel 內等待。
- 另一維度——延遲隱藏方式：**動態**（類 GPU，硬體排程）或**靜態**（類 NPU，需嚴格
  順序 + 軟體 pipeline + 時間餘裕）。

**4. A100 代理**：**不 enable peer access**，跨邊界一律顯式 D2H + H2D；用 NVLink 流量
計數器確認代理流量為零（否則量到的是 NVLink）。NPU 行為模擬：少量 persistent CTA +
顯式雙緩衝。

**5. 配置**：
- 同 NUMA：TP=2（GPU0–1），閘道 GPU1，代理 GPU2。
- 跨 NUMA：TP=4（GPU0–3），閘道 GPU3，代理 GPU4。

**先把等級 0 的完整流程跑通，再往上加。**

### 成功標準

- 1a、1b 各有 report + params.json 條目，1a 明確判定競爭位置。
- 等級 0 端到端跑通：AG+GEMM 一個 TP 域 + 一個代理裝置，結果 allclose，
  NVLink 計數器確認代理流量為零，時間有交錯 ECT 量測。

### 設計約束（來自 PHASE0_FINDINGS 第 3 節）

閘道 GPU 避免同時收送；生產者消費者跨 switch 組（0/1、2/3、4/5、6/7）；pinned buffer
綁來源 GPU 的 NUMA node；chunk ≥ 16 MiB；等級 0 建模用競爭態 6.46 GB/s，不用 nominal。

---

## 2. 目前成立（worker 維護）

（尚無。）

## 3. 撤回表（worker 維護）

| 撤回主張 | 出處 | 原因 | 替代 |
| --- | --- | --- | --- |
| （無） | | | |

## 4. 下一步（worker 維護）

**2026-10-06 更新**（依調查報告第 4 節；原 1a / 1b 仍是第一步）：
0. 與教授確認方向解讀（見第 0 節「待確認」）。
1. 1a、1b（如下）。
2. 技術探針：從 Flux 外部（H2D stream 上的 `cuStreamWriteValue32` / 主機執行緒）寫入 AG-GEMM kernel 等待的旗標，確認等級 1 對 Flux 的改動量。
3. 等級 0 → 等級 1 → NPU 型代理 → 工作切分（先用成本模型估，避開跨裝置頻寬的「死區」，arXiv 2602.09721）。

原有：
1. 1a：改 `bidirectional_bandwidth_v2.py` 為 per-device 方向，複製到本 ws 的 `scripts/`
   （保留原檔不動），跑三組配置各 20 repeats。
2. 1b：`dual_path_v3.py` 加 overhead-only 模式，同樣複製到本 ws。

## 5. 產出到 common 的數字

（尚無。1a → `pcie.contention_locus`；1b → `staging.fixed_overhead_ms`。）
