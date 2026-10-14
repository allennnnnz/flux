# ws/hetero-proxy — STATUS

**本 workstream 唯一權威。** 第 1 節由 boss 寫入，worker 不改；第 2 節起由 worker 維護。

建立：2026-09-23（boss）
最後更新：2026-10-14（boss 兼 worker：I1–I4 完成；E2E 計劃定案）
狀態：**介面原型與 I3 / I4 量測完成；下一步：真實模型端到端（A / B / C）**

---

## 0. 快速跟上（新 session 先讀這節；每次收工更新）

最後更新：2026-10-14

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

**最新（2026-10-14）—— 一段話跟上**：做出了「任何加速器都能接進 NVIDIA 環境」的介面原型 hxb（`scripts/hxb/`）：
雙方在**主機記憶體**交換資料（中轉 slot），用主機記憶體裡的**單調計數器**通知彼此；GPU 端用 copy engine 搬、用 stream memory op 等 / 寫計數器
（ctypes 直接呼叫 libcuda，全部事先排進 stream，不需主機執行緒）；接收端 GPU 用每塊一個旗標的 Triton GEMM（Flux 的 tile 等旗標）。
兩個不同的後端都接上且正確：CPU（獨立程序扮晶片）與 A100 代理（GPU2，NVLink 計數器增量 0）。
交棒切塊重疊實測：level 0 → level 1 快 **2.3–2.8×**（模擬計算 / 真實計算）。使用者要求下一步用**真實模型端到端**驗證（見下）。

**E2E 計劃（2026-10-14 與使用者討論定案）**：Qwen2.5-32B-Instruct 真實權重（下載約 65 GB），自寫精簡推論程式（vLLM 未安裝，也不支援異質分段），
先固定批次（prefill / decode / 吞吐量），再 ShareGPT（快取裡已有）。三組比較：
- **A**：8 卡 NVLink、預設 NCCL（基準）。
- **B**：8 卡 + CPU 經 PCIe 參與每次前向（CPU 做最後的 RMSNorm；資料 GPU → PCIe → CPU → PCIe → GPU，hxb 切塊重疊）→ 證明「PCIe 能與 NVLink 環境協同」，
  生成文字必須與 A 完全一致；量通道代價（預期 B 比 A 慢約 0.5–2%，推論）。
  不把整層給 CPU：CPU fp32 0.7–2 TFLOPS（I2 v2 B）vs A100 236 TFLOPS → CPU 算 1 層（prefill 4096 字約 2 s、decode 約 10–20 ms）就比 8 卡跑完整模型
  （約 0.18 s / 11.5 ms，fusion-dispatch G4）慢 [推論；下載權重後實測一層驗證]。
- **C**：8 卡 NVLink + PCIe 同時傳（FlexLink 式：TP all-reduce 的一部分經主機中轉走 PCIe）→ 檢驗 PCIe 通道能否加速。
  **預期在本機最多約 +0.5%**（推論：8 卡同時收送時 PCIe 每卡約 6 GB/s〔Phase 0 5.87〕對 NVLink 約 200 GB/s，通訊約佔 prefill 兩成；decode 受延遲限制不受益；
  FlexLink 8×H800 AllReduce 只 +2%）。使用者知悉、仍要做：沒加速本身也是結論（NVLink 很強的機器上 PCIe 適合接外部晶片，不適合分擔 GPU 間通訊）。
  先量純頻寬，再接模型。
- 步驟：(1) 權重 + 推論程式 + 與 HF 輸出一致 + 量 A；(2) B；(3) C；(4) ShareGPT。每步預測先登記、守衛、交錯、多輪中位數。
- **步驟 1 完成（10-14）**：推論程式與 HF 逐字一致（4×64）；CPU 一層 prefill 4096 字 2.05 s、decode B=32 109 ms（推論得證）；A（eager）prefill 87 / 532 / 1056 ms，
  decode 每字約 48 ms 且不隨批次變（受 Python 發指令限制）。
- **追加決定（10-14 使用者）**：(1) decode 的 A、B、C 都用 CUDA graph（同一用法）；(2) B 兩種位置都做——**B-final**（CPU 做最後的 RMSNorm，
  只需各序列最後一個位置，每次傳幾 KB）與 **B-mid**（CPU 做第 33 層的輸入 RMSNorm，全部 token 經 PCIe 切塊往返，回到 GPU0 後經 NVLink 廣播給 8 卡）。
  推論：B-mid prefill 慢約 3–6%、decode 影響小；B-final 接近 A。

**E2E 結果（2026-10-14，guard 全部 CLEAN，尚未審查）**：
- **B：PCIe 確實能與 NVLink 環境協同**（`results/e2e/bench_v3/`，v3 @ 77717fc，20 輪）：每個字都經過 GPU → PCIe → CPU → PCIe → NVLink 廣播，
  生成文字與 A 一致（Bfinal 4/4 題全同；Bmid 3/4 全同、第 4 題在 HF top-2 logit 完全平手處翻轉，`check/hf_margin_v3_vsA.json`）。
  代價：decode 每步 0.3–1.0 ms（B 減同樣拆 graph 的 Asplit 對照）；prefill Bfinal +0.1–0.7%、Bmid8 +2.5–4.1%、Bmid1 +4.4–5.3%；
  切 8 塊藏掉交棒時間的 22–46%。CPU 不增加算力 → 不可能加速（預期之內）。
  拆 graph 本身讓 decode 快 0.1–6.5%（AsplitM vs A）：本推論程式的 decode 有可觀的 graph 啟動開銷（A b1 每字 22 ms，b32 43 ms；G4 最佳約 11.5 ms）。
- **C：本機上 PCIe 通道無法加速 GPU 間通訊**（`results/e2e/c_bw/`）：all-reduce 分任何比例 f > 0 走主機中轉 PCIe，在 0.6–160 MiB 全部比只走 NCCL 慢（+63% 以上）。
  上限推算：NCCL 160 MiB algbw 約 102 GB/s，PCIe 每卡同時收送約 6 GB/s → 最多分擔 all-reduce 的 5.6%；all-reduce 約佔 prefill 23% → 端到端最多約 1.3%（推論，主機加總不計時間）；
  decode 訊息太小，PCIe 固定延遲已超過 NCCL 全程。事前登記的預測成立。是否把 C 接進模型、或先優化主機路徑，待使用者決定。

**實作進度（I1–I4 完成；細節與數字見第 2 節）**：
- [x] H0 工作切分（`reports/20261009_h0_work_partition.md`，推論）：TP 成員排除；pipeline 段可行，吞吐量上限 +s/8。
- [x] I1 介面規格（`reports/20261009_i1_backend_interface.md`）：6 個基本動作 + 語意規則 + Bridge 四條硬規則。
- [x] I2 CPU 後端 + Bridge + 特性量測 v2（`results/i2_characterize_v2/`，CLEAN）。
- [x] I3 交棒切塊重疊（`results/i3_overlap/`，CLEAN，200 輪）——分析報告未寫。
- [x] I4 A100 代理後端（`results/i4_proxy/`，CLEAN，200 輪）——分析報告未寫。
- [ ] I5 NPU 型模擬：腳本 `scripts/i5_npu_sim_v1.py` 已寫、未跑（E2E 優先）。
- [ ] I3 / I4 報告；auditor 審查（進 DECISIONS 前必需）。

**以下為 2026-10-09 的原計劃 H0–H5（保留供追溯；已由上面的實作與 E2E 計劃取代）**：
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

**待使用者 / 教授確認**：晶片的型號與軟體堆疊。（「通道」的兩種解讀已由 E2E 計劃的 B〔接外部晶片〕與 C〔額外頻寬〕同時涵蓋，2026-10-14。）

**環境注意**：css-host-159 曾被同帳號的 `sglang::server` 佔用（10-06），其 openclaw-gateway 仍停著，待使用者決定何時還原；本 ws 全部只用 css-host-158。
執行 hxb：`env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml python ...`（不經 `launch.sh`）。

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

全部 2026-10-09 量測、css-host-158、exclusive_guard_v2 CLEAN；**尚未經 auditor 審查**。

1. **介面可接不同後端**：CPU 後端語意測試 9/9（`scripts/test_hxb_backend_v1.py`：依序、釋放語意、單調、跨佇列依賴、運算正確、靜態裝置），
   A100 代理 7/7（T8 為 CPU 專屬）。測試抓到並修正：DMA worker 缺 sfence（NT store 未排序）、測試自身用 `value()+1` 推目標值的競爭。
2. **特性量測 v2 與錨點一致**（`results/i2_characterize_v2/`，D 段只計兩方向同時在飛的窗口）：D2H 24.07、H2D 21.7–21.9、同卡雙向 H2D 5.95–7.3 GB/s
   （錨點 6.46）、GPU0 送 + GPU2 收 23.7 / 21.3 GB/s。CPU 裝置 DMA（4 執行緒、GPU 剛寫入的資料）兩方向合計 20–23 GB/s；
   DMA 持續運作時 CPU 計算慢 2.1–2.8×（C 段）。F 段：GPU 流量不會拖慢裝置 DMA（n ≥ 8 背景流量覆蓋完整；n ≤ 4 未覆蓋窗口，不採計）。
3. **I3 交棒切塊重疊**（`results/i3_overlap/rounds.csv`，全部 200 輪中位數，`scripts/i3_overlap_v1.py` @ 3431a0b）：
   emu4（裝置計算 4 ms）13.83 → 5.88 ms（n=8，×2.35，可省等待的 81%）；emu0 10.08 → 6.33（×1.59）；cpu64 真實計算最多 ×1.3、n 大反而變慢。
   事前登記：P1（emu 在 15% 內）emu4 n ≤ 8 過（+2–14%）、n ≥ 16 不過；emu0 只有 n ≤ 2 過。P2（cpu 落在樂觀–悲觀區間）大多不過或邊界。
   P3 前半過（emu4 ≥ 2×）、後半不過（同卡收送沒有比較慢：瓶頸在裝置端、兩方向實際同時傳的時間短）。P4 過（最佳 n 為模型所選或相鄰）。
   模型漏掉的：每塊固定開銷（n=32 時主機 enqueue 13–15 ms，> pass 時間）；link-bound 時裝置 DMA 在 pipeline 內慢約 33%，原因未隔離
   （「冷 slot」假說已否定，`results/i3_overlap/diag/`）。時脈過濾：GPU0 有 2/3 輪次在 1155 MHz，故以全部輪次中位數為準。
4. **I4 A100 代理**（`results/i4_proxy/`，@ f210b89）：pemu4_sep 12.64 → 4.54 ms（n=16，×2.78，可省的 86%）；preal_sep 9.60 → 4.19（n=32，×2.29）。
   P1 過：NVLink Tx+Rx 增量 GPU0 / 2 / 4 皆 **0 KiB**。P2 **不過**：分開佇列比共用佇列快（預測相反）。P3 過（rel err ≤ 5.7e-3）。
   P4 過但有附註：代理的模擬計算校準偏差，設定 4 ms 實為約 3.3 ms，與 CPU emu4 的比較不完全對等。

## 3. 撤回表（worker 維護）

| 撤回主張 | 出處 | 原因 | 替代 |
| --- | --- | --- | --- |
| 無 cuda-python → GPU 旗標改以 copy engine 搬 4 bytes 寫入 | JOURNAL 2026-10-09「開始實作」 | ctypes 直接呼叫 libcuda `cuStreamWriteValue/WaitValue*_v2` 可用（2026-10-09 探針，`scripts/hxb/cudamem.py`） | 旗標與計數器一律用 stream memory op |
| I2 v1 A 段裝置 DMA 44–52 GB/s | `results/i2_characterize/`（v1） | 每次拷貝同一塊快取熱的緩衝區；pipeline 內是 GPU 剛寫入的冷資料，約一半 | `scripts/i2_characterize_v2.py` A 段（v1 腳本已加 WITHDRAWN） |
| I2 run1 C 段「計算與拷貝競爭 1.6–3.6×」 | `results/i2_characterize/run1_ht_sibling_layout/` | 控制執行緒放在計算核心的 HT 兄弟核，混入自旋干擾 | v2 C 段 2.1–2.8×（專屬控制核心） |

## 4. 下一步（worker 維護）

**2026-10-14**（取代 10-06 的清單；1a / 1b 已由 I2 特性量測 D 段涵蓋：同卡雙向 H2D 5.95–7.3 GB/s、GPU0 送 + GPU2 收無競爭）：
1. E2E 步驟 1：下載 Qwen2.5-32B-Instruct；精簡 TP8 推論程式（NCCL）；輸出與 HF 一致；實測 CPU 算一層的時間（驗證第 0 節的推論）；量 A。
2. E2E 步驟 2：B（CPU 做 final RMSNorm，經 hxb）；文字與 A 一致；A / B 交錯量測。
3. E2E 步驟 3：C 先量純頻寬（NVLink-only vs NVLink + PCIe 中轉分流），再接模型。
4. E2E 步驟 4：ShareGPT。
5. 寫 I3 / I4 分析報告；I5 視需要再跑；安排 auditor。

## 5. 產出到 common 的數字

（尚無。1a → `pcie.contention_locus`；1b → `staging.fixed_overhead_ms`。）
