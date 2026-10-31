Supersedes: 無（補充 `20261030_topic_search.md`、`20261029_lit_moe_hybrid.md`）

# 「兩個方法互補」的選題組合（2026-10-31）

使用者要求：找兩個方法，A 的弱點由 B 補、B 的弱點由 A 補，合成一個論文題目。
選題條件（記憶檔 topic-preferences）：端到端要有明顯提升；不能只是改設定；不做 KV；只用 css-host-158；約 10 週，workshop 或中等會議即可。

本報告沒有做 GPU 量測。文獻多數只讀了摘要或二手摘要，個別標明。

## 1. 結論

- 單卡上「兩法合一」的組合幾乎都已被做完。剩下有空間的，都要靠本機的「多卡 NVLink＋PCIe 裝置」環境。
- **推薦組合一**：SwiftSpec（草稿與目標模型分開、非同步）× DuoDecoding（草稿放異質裝置），合成「推測解碼的草稿交給 PCIe 裝置，NVLink GPU 全部給目標模型」。
- 組合二（KTransformers × MoE-Lightning 流水線）看似互補，但端到端上限推算約 1.1×（第 3 節），不推薦當主線。

## 2. 組合一：PCIe 裝置當草稿機

| | SwiftSpec（arXiv 2506.11309，ByteDance／UChicago） | DuoDecoding（arXiv 2503.00784，EMNLP'25） |
|---|---|---|
| 做法 | 草稿樹與目標模型放在不同 GPU 群、各自 TP、非同步，互不等待 | 草稿模型放 CPU、目標放 GPU，同時跑；草稿長度 = 兩邊速度比 |
| 成果（摘要） | 比既有推測解碼平均快 1.75×；8 張 Hopper 跑 Llama3-70B 348 tok/s | 生成延遲最多快 2.61× |
| 弱點 | 草稿佔用 NVLink 域裡的 GPU；兩群之間假設 NVLink | CPU 慢，草稿長度受限；目標只有單卡、沒有 TP |

**合起來**：草稿放在只有 PCIe 的裝置上（將來是國產晶片；現在用「只准走 PCIe 的 A100 代理」，也就是 hxb 的 A100 後端；CPU 當弱裝置的對照），目標模型用 NVLink 域的 GPU 做 TP。兩邊非同步；每步交接（草稿 token、必要時 hidden state，KB 級）用 hxb 的主機記憶體計數器，GPU 端自己等，不經 CPU 執行緒（hxb 量過往返約 11 µs，`scripts/hxb/cudamem.py`）。

**為什麼適合本專案**：
- 晶片不在每個字的關鍵路徑上，正好修掉 E2E B 的根本問題：晶片串在路線上，所以沒有加速（STATUS P1 背景）。
- P1 測過的角色（attention 外放、MoE 專家、P/D）沒有「草稿機」這一項，是新角色。
- 符合教授的方向：只能走 PCIe 的晶片和 NVLink 叢集協同。

**端到端預估（推論，未量）**：
- 對「不用推測解碼」：2–3×（文獻常見的小批次數字）。
- 對「草稿和目標放在同一批 GPU」的推測解碼（審稿會要的公平基準）：1.2–1.35×。假設草稿時間約佔每步 2–3 成，非同步後被藏掉。
- 如何驗證：見第 5 節。

**相近工作與差別**：
- PEARL、PipeSpec、AMUSD：非同步或平行推測解碼，都在 GPU 上。
- HeteroReason（arXiv 2609.28717，2026-09）：FPGA 做草稿、GPU 驗證，延遲 1.01–1.42×。屬於推理步驟層級（PRM），只讀了摘要。
- 我們的差別：PCIe-only 異質草稿機 × NVLink TP 目標 × GPU 端訊號交接；並用「同一張卡走 NVLink（SwiftSpec 式）vs 只走 PCIe」量出 PCIe 的代價。

**風險**：
- 推測解碼很熱門。
- 批次大時增益變小。
- 工程量：引擎要支援一次驗證多個 token。

**本機條件**：
- Qwen3.6-27B 已在本機（52 GB，`Qwen3_5ForConditionalGeneration`，線性注意力＋全注意力混合，**自帶 1 層 MTP**：index 有 15 個 `mtp.*` 權重），可當「目標＋草稿」，但需要 SGLang 支援。
- 或用自家引擎（`k2_engine_v2`，Qwen2.5-32B TP8 CUDA graph）配 Qwen2.5-0.5B 當草稿（約 1 GB 要下載）。
- CPU 當草稿機太慢（推論：Qwen3.6 的 LM head 約 248k × 5120 × 2 B ≈ 2.5 GB，CPU 每猜一個字約 10 ms，目標一步約 10 ms 級），只能當弱裝置的對照。

## 3. 組合二：KTransformers × MoE-Lightning（不推薦當主線）

- KTransformers：多卡 TP、線上服務，但每層 GPU 和 CPU 輪流等。
- MoE-Lightning（arXiv 2411.11217）：CPU／GPU／I/O 三方流水線（CGOPipe）＋階層 roofline 模型，但為離線吞吐量設計，在 1–4 張 T4 上評估。
- **上限推算**：
  - LMSYS 部落格的數字：單一請求時 8 卡只比 1 卡快 26%。設 T = C + G/p，代入 T1/T8 = 1.26，得 G ≈ 0.31 C，所以 8 卡時 CPU 段約佔 decode 的 96%（推論，忽略通訊；他們是 AMX CPU＋L20，本機 CPU 更慢，佔比只會更高）。
  - Kimi 有 384 個專家、每個 token 選 8 個，decode 時每個專家通常只分到 1–2 個 token。這是 M0 的 1 token 區（1.10×）。
  - 套到 96% 上，整體約 1.09×；流水線在 GPU 段只佔約 4% 時也藏不了什麼。
- 已有類似做法：SpecMoEOff（2508.21706，兩個 micro-batch 輪流）、2606.10493（雙批次交錯，1–2 張 5090）、vLLM RFC #33869（DBO：搬專家與 CPU 計算同時做，2026-07 以「not planned」關閉；後續 RFC #38256 / PR #37190 狀態未查）。
- → M 在 decode 的端到端上限有限。除非 M1 顯示 int4 在本機 CPU 上嚴重受計算限制。

## 4. 已被做掉或已評估的組合

| 組合 | 為何不選 |
|---|---|
| 推測解碼 × MoE 卸載 | 至少 5 篇：SpecMoEOff（2508.21706）、SP-MoE（2510.10302）、SpecMoE（2604.10152）、Specter（2511.14102）、MoE-SpAc（2603.09983），單卡為主 |
| CPU 計算 × 搬專家同時做（單卡） | Apex（2610.23006）、vLLM RFC #33869 |
| T × F（offload 干擾 × 融合決策） | 10-30 已評估，上限約 1.1× |
| 改 Flux 自己的排程（FS） | 10-30 原型 P1 依規則不成立 |

## 5. 組合一的驗證（建議，待使用者同意；跑之前登記預測與判定規則）

1. 基準：在本機量「草稿與目標同在一批 GPU」的推測解碼，草稿佔每一步的時間比例。這個比例就是能藏掉的上限。
2. 量草稿放在 PCIe 裝置上時每猜一個 token 的時間：A100 代理只准走 PCIe；CPU 為對照。另量每步交接延遲。
3. 用 (接受長度, 草稿／驗證時間比, 交接延遲) 的簡單模型算端到端預估。
4. 判定規則（建議）：對同 GPU 推測解碼預估 ≥ 1.2× 才投入。

## 來源

- SwiftSpec：https://arxiv.org/abs/2506.11309 ；DuoDecoding：https://arxiv.org/abs/2503.00784 ；Dovetail：https://arxiv.org/abs/2412.18934
- HeteroReason：https://arxiv.org/abs/2609.28717 （只讀摘要）；NebulaSD：https://arxiv.org/abs/2609.29364 （未讀）
- MoE-Lightning：https://arxiv.org/abs/2411.11217 ；MoE-Lens：https://arxiv.org/abs/2504.09345 ；CoX-MoE：https://arxiv.org/abs/2605.17889
- SpecMoEOff：https://arxiv.org/abs/2508.21706 ；SP-MoE：https://arxiv.org/abs/2510.10302 ；SpecMoE：https://arxiv.org/abs/2604.10152 ；Specter：https://arxiv.org/abs/2511.14102 ；MoE-SpAc：https://arxiv.org/abs/2603.09983
- 2606.10493：https://arxiv.org/abs/2606.10493 ；vLLM RFC：https://github.com/vllm-project/vllm/issues/33869
- SpecBundle / SpecForge：https://lmsys.org/blog/2025-12-23-spec-bundle-phase-1
