# ws/hetero-proxy — STATUS

**本 workstream 唯一權威。** 第 1 節由 boss 寫入，worker 不改；第 2 節起由 worker 維護。

建立：2026-09-23（boss）
最後更新：2026-10-30（boss 兼 worker：選題重找——通訊優化、推論或訓練、10 週論文；環境實驗 T1 / T1b / T2＋三路文獻檢查）
狀態：**選題待使用者決定。推薦 T：訓練時主機↔GPU 搬運讓 NVLink 集合通訊慢 2–3 倍（T1 實測）→ 干擾模型＋協定選擇＋搬運節拍器；備案 F：fusion-dispatch 成文；MoE 冷門專家（M0 通過）降為第三**

---

## 0. 快速跟上（新 session 先讀這節；每次收工更新）

最後更新：2026-10-30

**最新（2026-10-30 下午）—— 題目 T 的驗證（使用者：先確認值得投入再規劃）**（計劃 `reports/20261030_t_validation_plan.md`、結果 `reports/20261030_t_validation_result.md`、數字 `results/topic/analysis.txt`）
- **結論：有條件值得**——現象真實且新（最新 NCCL 2.32.3 仍 ×1.9–2.8；ChunkFlow 2026 §2.3 明寫 NVLink 機器上不會干擾）、改設定修不掉（通道數沒用、限速更糟、LL128 大訊息慢 23–29%）；
  但真實訓練步驟（V3：TP8＋SP、Qwen2.5-32B 層、梯度已對 fp32 參考）裡 naive 只慢 1–6%，峰值 24%（延遲更新梯度 offload、4096 token × 1 micro-batch，搬運 ≈ 計算），能拿回的整步時間 3–8%。
- 事前規則：1 成立（24% / 11%）、2 成立、3 部分（6 個 ≥ 5% 情境中 4 個收回 61–81%；最大損失那格只有 LL128 拿回 8%，窗口與混合排程都更糟）、4 成立。
- 預測錯的：V1a「25% 限速仍剩一半減速」（實際約 1/10–1/5）、V1b「通道多較不敏感」（相反）、V3「activation offload 損失 10–25%」（6.1% / 2.1%）。
- **待使用者決定**：做 T（特性量測為主、整步 3–8%）、改做 F、或 F＋T 合併（成本模型加干擾項）。
- **10-30 新選項 FS：改 Flux 自己的排程**（使用者問：除了決策器，能否改 Flux 的排程邏輯）。用既有資料算（`ws/fusion-dispatch/results/final_ag_points.csv`，gpu 模式，M ≥ 512，40 點）：
  Flux 融合只做到可能重疊的**中位數 40%**；融合時間是理想值 max(Flux AG 單獨, GEMM 單獨) 的**中位數 1.30×、最多 1.72×**（以較快的 GEMM 算：1.37× / 1.89×）；
  M ≤ 2048（chunked prefill 的大小）與 N 小的層（L-QKV）最差；有幾點融合比「先 AG 再 GEMM」還慢（G-FC1 M=3072：2.73 vs 2.21 ms；G-QKV M=2048：1.30 vs 1.19）。
  原始碼：A100 的 All2All＋use_read 是單一 stream **依序**拉 7 份（rank+1、rank+2…，每份切 SPLIT 塊各設旗標，`src/coll/ths_op/all_gather_op.cc:533-575`）；
  GEMM tile 也從自己那份開始依序輪轉（`src/ag_gemm/sm80_all_gather_gemm_threadblock_swizzle.hpp:113-127`）→ 順序已對齊；
  推論的原因：M 小時一份資料的 tile 數 < 一波（M=1024、n=6144、tile 128×256：每份 24 個 tile vs 108 個 SM）→ 第一波就要等 4–5 份；靜態 stream-K 讓 CTA 卡在還沒到的 tile；登錄表選到 PCIe 的 config（陷阱表）。
  可改：依「資料是否已到」動態分派 tile、依 (形狀, M) 選 tile 大小／切法、第一份資料切更細、通訊受限的層把 AG 提早到前一個運算。`AGKernel.profiling()` 有 pybind（`src/pybind/ag_gemm.cc:90`）可先不重編就看「只換 config」能拿回多少；`build.sh --arch 80` 可只編 sm80。
  推論的上限：運算層級中位數 1.2–1.3×、chunked prefill 大小最多 1.5–1.7×；prefill 端到端約 1.15–1.3×（未量）。文獻威脅：TileLink、Triton-distributed、FlashOverlap、PyTorch async-TP（未查）。此方向即從未開始的 `ws/diag-overlap`。
- 10-30 使用者問目前有哪些選項 → 回覆總表：M（待驗證、潛在提升最大）、T（已驗證、最多 1.09×）、F（最多約 1.1×）、F＋T、新想法 O（PCIe 受限的 offload 訓練：壓縮搬運量；
  由 V3 gimm 4096 推上限約 1.3×，推論，未查文獻）；排除：KV（教授）、跨節點（只用 158）、改設定類、已擠的題目。
- **使用者 10-30（驗證後）**：要「較明顯的效能提升」→ T 的上限（整步最多 1.09×；單次通訊可從 ×2–3 變慢恢復；activation offload 的前向 32.8 → 28.5 ms，快 13%）不夠當主賣點。
  回覆建議：明顯提升要選「基準本身被通訊卡住」的問題 → M（MoE 專家放主機、KTransformers TP8 只用 4 條上行線中的 1 條）；M0 的專家部分對單線設計 1.08×（1 token）/ 1.17×（4）/ 1.61×（16），
  批次更大時 KT 由 CPU 算冷專家（受計算限制），多線搬到 GPU 的上限是頻寬比約 4×（推論，未量；KT 的 layerwise 門檻是否也用在 decode 批次待讀原始碼）。下一步待使用者同意：M 的驗證（另建 venv 裝 SGLang＋kt-kernel，量 KT TP8 在 Kimi-K2.6 的時間組成，事前登記 ≥ 1.3× 的判定規則）。
- 環境：NCCL 2.32.3 裝在 `~/opt/nccl-2.32.3`，在 pixi 環境內以 `LD_PRELOAD` 使用（`torch.cuda.nccl.version()` 只報編譯版本，實際版本要用 ctypes `ncclGetVersion`）。

**之前（2026-10-30 上午）—— 選題重找（使用者：通訊優化，推論或訓練皆可，10 週內一篇架構完整的論文）**（`reports/20261030_topic_search.md`）
- 拓撲：每個 PCIe 交換器 = 2 張 GPU＋1 張雙埠 100 GbE 網卡、一條 Gen4 x16 上行。**159 目前被使用者帳號的 SGLang 服務（DP4×TP2）佔滿 8 卡，且少一張網卡（rail 5/6）** → 跨節點題目環境風險高。
- **T1**（`scripts/topic/t1_bigcoll_pcie_v1.py`、`results/topic/t1_v1_*`，30 輪打亂、guard CLEAN、eager）：8 卡背景 GPU→主機搬運讓 16 MiB reduce-scatter / all-gather 慢 **×2.9–3.0**、32 MiB ×1.9–2.1、64 MiB ×1.4，128 MiB 以上 ≤ ×1.08；主機→GPU 小得多（≤ ×1.5）。搬運本身不受影響。
  強制協定：16 MiB 以上預設 = Simple；LL128 鈍感得多（16 MiB ×1.2–1.3）→ 有 d2h 時 RS 16 MiB LL128 比 Simple 快 2.2×、AR 32 MiB 快 1.75×，128 MiB Simple 仍較好——**最好的協定隨背景搬運翻轉**。
- **T1b**（`results/topic/t1b_v1_bg0`、`_bg0_1`）：**只有同一交換器的 GPU0＋GPU1 在搬，就幾乎等於 8 卡都在搬**（RS 16 MiB ×2.88 vs ×3.02）；只有 GPU0 搬也有 ×1.84。
- **T2**（`scripts/topic/t2_rdma_paths_v1.sh`、`results/topic/t2_v1`，單機 RoCE 迴環）：網卡讀 GPU 47.9 Gb/s、8 個 QP 也 48.4（不是 QP 數的問題）；網卡讀主機滿速 97.4，但同交換器兩張 GPU 往主機搬時掉到 30.5 → 「經主機中轉繞過讀取上限」不是免費的（而且 MVAPICH / UCX / NCCL 已有），不推薦。
- 錨點：單卡與交換器上限和 M0 相符；**未解**：8 卡同時搬 101–104 GB/s vs M0 85.4（+18–22%），NCCL 無搬運頻寬比 nccl-tests 高 9–12%（`results/topic/analysis.txt`）。
- 文獻（三路獨立搜尋，報告附錄）：同一機制只見於 NVIDIA/nccl #2218（V100/H200，−8–11%，已關沒修）與 PyTorch 論壇（H100），**沒有論文**；ChunkFlow 是純 PCIe 機器。
  跨節點的 GDR 讀繞路、HSDP／壓縮、rail 感知、RL 權重同步、TP decode all-reduce 都已擠或已做。fusion-dispatch 仍有空間（TokenWeave 用固定門檻、FlashOverlap 不會退回不融合）。
- **建議**：主題目 T（訓練時 offload 干擾的量測＋排程器，單機即可、10 週計劃見報告 5.5，第 1 週先在真實 FSDP2 / activation offload 訓練步驟裡量干擾占比，≥ 10% 才繼續）；備案 F；M 第三。
- **T3（使用者問能否結合 Flux 的切割；`scripts/topic/t3_flux_vs_hostcopy_v1.py`、`results/topic/t3_v1/`，Qwen2.5-32B 層形狀、TP8、4096 token，30 輪、guard CLEAN）**：
  大 GEMM 層 Flux 融合把變慢的通訊吸收掉（gate_up ×1.11、down ×1.07；NCCL＋mm ×1.23 / ×1.36）；小 GEMM 層 Flux 反而更慘（qkv ×2.12 vs NCCL ×1.65）；只有 GEMM ×1.00–1.09。
  融合優勢隨搬運改變（o_proj 1.42× → 2.02×、qkv 1.37× → 1.06×）→ T 的架構改為「每層決定融合與否＋搬運放在會被吸收的窗口＋用 Flux 式訊號切塊放行搬運＋不融合時選協定」，同時用上 fusion-dispatch 的成本模型（報告 5.7）。
- **待使用者決定**：選 T / F / M；T 的框架與模型大小。**使用者 10-30：只用 158**（跨節點與 159 不列入）。
- T 的 offload 做法（對使用者的說明，推論待第 1 週驗證）：主情境 = 單機全參數微調 Qwen2.5-32B（本機有權重），TP8＋序列平行（每層集合通訊 [序列, 5120] bf16：序列 4096 → 40 MiB、2048 → 20 MiB，落在 T1 受害最重的區間），
  Adam 狀態放主機（梯度 GPU→主機、更新後權重主機→GPU），可選 activation offload；自寫 PyTorch 2.6 訓練迴圈（沿用 e2e 的 TP8 程式，每筆搬運自己控制），FSDP2 `CPUOffloadPolicy` 當現成框架對照（但它每層的集合通訊 ≥ 數百 MB，預期受害小）。

**之前（2026-10-29）—— 選題：MoE 專家放在 PCIe 裝置上，8 張 NVLink GPU 一起用**（`reports/20261029_lit_moe_hybrid.md`）
- 為什麼換：使用者認為「關掉 NCCL 一個設定」不算貢獻；背景載入權重（X4 的主案例）細算後問題太小——每卡 8 GB 以 9–12 GB/s 約 0.7–0.9 秒就搬完（推論，用 K1 頻寬算），
  而且 4 條上行線都滿時經 NVLink 轉送不會多出頻寬（10-28 回覆中「一半 PCIe、一半 NVLink 交換」的說法錯誤）。排程要有意義，PCIe 上必須有**持續、在關鍵路徑上**的流量 → MoE 冷門專家放在 PCIe 裝置（現在是 CPU，將來是國產晶片）。
- 文獻：1 張 GPU＋CPU 已很擠（Fiddler、KTransformers SOSP'25、HybriMoE、DALI、LayerScope、Adaptive Expert Assignment…）；GPU＋記憶體端運算裝置（MoNDE、DynaNDE）最像國產晶片的角色，但都是單一加速器。
  **最接近本機環境且開源的是 KTransformers 併入 SGLang 的 8 卡版本**——讀原始碼（sglang `d213124`、ktransformers `c987331`）確認：
  只有 GPU0（tp_rank 0）和 CPU 交換資料（`kt_ep_wrapper.py` 第 299、322、361–383 行）；GPU 等 CPU 用 `cudaLaunchHostFunc` 回呼（`cpuinfer.h` 第 96、121 行）；專家分配靜態；8 卡在單一請求時只比 1 卡快 26%（LMSYS 部落格）。
- 可改良處（全為推論）：H1 每個插槽和自己 NUMA 的 GPU 交換、用上 4 條上行線、CPU 部分和併入既有的 NVLink all-reduce；H2 回呼換成 hxb 計數器；H3 TP 下每卡只搬專家的 1/8，多線同時搬 → 「裝置算 vs 搬到 GPU」分界點下移；
  H4 用 X 系列知識避免搬運拖慢 NCCL；H5 慢的一端換成任何 PCIe 裝置（hxb）。
- **改良主軸（推論，報告第 6 節）**：8 卡經 4 條上行線各搬專家的 1/8（約 90 GB/s vs 單線 19），和 CPU 同時分工算冷門專家 → 估約 1.9×（單卡單線只有 1.19×）；先做 1–2 天的否證實驗：CPU 算＋8 卡搬同時進行，量兩邊速度與 DRAM 互搶。
- **M0 結果（2026-10-29，`results/moehy/m0_v1/analysis.txt`，`scripts/moehy/m0_split_v1.py`，30 輪打亂、guard CLEAN）——題目可做**：
  - 主機→GPU 的頻寬上限：單卡 21.3、同一交換器兩卡合計 25.2、同一插槽兩交換器 41.9、8 卡 85.4 GB/s（每卡 10.7）→ 限制依序是卡、交換器上行線（約 25）、插槽（約 42）。
  - CPU 單獨算（兩插槽、bf16）：DeepSeek-V3 專家形狀 1 token 每插槽 138–140 GB/s。
  - **同時做的增益**（CPU＋8 卡搬，相對只有 CPU）：1 token 1.10×、4 token **1.24×**、16 token 2.03×（16 token 時這個 CPU kernel 受計算限制，偏高）；
    只用 GPU0 一條線（= 單卡系統）只有 1.02 / 1.06 / 1.26×；4 卡（每交換器一張）拿到 8 卡增益的約 9 成。Qwen3-235B 形狀相近（1.14 / 1.28 / 2.08×）。
  - **互搶吃掉一大半增益**：理想（各自單獨的速度相加）1.31 / 1.40× → 實際 1.10 / 1.24×；CPU 慢 14%、搬運慢 22%（1 token），
    但插槽總讀取只有 152–154 GB/s，低於讀取錨點 172 → 不是頻寬被吃滿（推論：互相拉高記憶體延遲；如何驗證見 analysis.txt F1）。
  - 預測：M1、M2、M5 成立；M3 搬運只慢 ≤10% 不成立（22%）；M4 1 token 不成立（1.10 < 1.15）、4 token 成立。判定規則（g8 4 token ≥ 1.20 且比單線多 ≥ 0.10）**達成**（1.238，p10 1.213；差 0.177）。
  - 對論文的意義：(1) 多出來的上行線是多卡伺服器獨有的資源，單卡系統（Apex、KTransformers 單卡）拿不到；(2) 用單獨速度估算的分工（Apex 的做法）會高估 GPU 那邊，排程器要把互搶算進去；
    (3) 增益隨 CPU 端越受計算限制越大 → Kimi-K2.6 的 int4（本機無 AVX512_BF16 / AMX）可能落在 1 與 16 token 之間（推論，待量 kt-kernel）。
  - 真實模型：本機已有 **Kimi-K2.6**（`~/.cache/huggingface`，555 GB，DeepSeek-V3 架構，384 專家、專家 7168×2048 = M0 的形狀，int4 group 32；2026-05-02 由同帳號下載，只讀使用）；
    8×80 GB 放進去幾乎沒有 KV 空間 → 需要把部分專家放 CPU，是真實的動機。KTransformers 支援 Kimi-K2 系列，RAWINT4 可在本機 CPU（AVX512BW / VNNI）跑。
  - Apex（2610.23006）全文確認：1 張 A6000＋1 顆 Xeon 8358P、單條 PCIe；沒有多線、NUMA、互搶。
- **下一步（待使用者同意）**：M1 = kt-kernel RAWINT4 在本機量 Kimi-K2.6 專家的 CPU 速度（決定落在 1.1× 還是 2× 區間）；M2 = 另建 venv 裝 SGLang＋kt-kernel，TP8 跑 Kimi-K2.6，量每層時間組成（rank 0 / rank 1 的交換、回呼、CPU 計算、其他卡等待；本機 Xeon 8358 無 AMX，有 AVX-512）＋回呼 vs 計數器微基準。


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

**最新（2026-10-28）—— 新題目 X 系列：PCIe 搬運對 TP 推論的干擾與 Flux 式訊號驅動排程**
（教授 2026-10 建議不做 KV cache 題目：太多人在做。K1 / K2 的結果保留當背景；K4.1 的發現與 KV 無關，成為新題目的起點。）
- **要講的故事**：大家以為 NVLink 機器上 TP 的通訊走 NVLink、PCIe 上做什麼都不影響它。K4.1 量到：背景大量 PCIe 搬運讓 TP 的 decode 慢 45–145%，
  原因不是 kernel 啟動（X1 撤回該推論）。X1d：主因是 NCCL 預設（`NCCL_GRAPH_MIXING_SUPPORT=1`）在 graph 內每個操作前後加的 event 等待／記錄節點，記錄節點被背景搬運拖慢 4–5 倍；
  關掉後 LL / LL128 的干擾從 +410–594% 降到 +3–5%、decode 一步 +197–202% → +2.3–2.5%。Simple 協定每段有系統層級屏障（×2.7，機制已見於 NVIDIA/nccl #2218），關掉仍 +203%。
  K4.1 的 Flux 式放行（約 +3%）要和「關 mixing、不限速」重新比較（待做，K4.1 是在 mixing=1 下量的）。
  KV 只是會在背景搬資料的情境之一；主案例改為**背景載入權重**（非同步 RL 的權重同步、服務中途換模型 / 載入 LoRA）。
- **研究問題**：Q1 現象有多大、在哪些條件下出現；Q2 原因是什麼（kernel 啟動被延遲？NVLink 被拖慢？kernel 本身變慢？）；
  Q3 Flux 式排程和簡單做法（限速、低優先權 stream、只在 decode 步之間搬、停下來等）比，「別人慢多少 vs 搬多快」的取捨曲線誰最好；Q4 真實服務裡整體效果。
- **步驟**（每步先登記預測，guard + tmux，graph + 打亂交錯）：
  - **X1 機制確認**（2–3 天）：CUDA graph 內跑合成工作——N 個極小 kernel（N 變化）、只有 NCCL 小 all-reduce、只有 decode 大小的 GEMM、混合——背景加 PCIe 搬運；
    用 GPU 時鐘（`%globaltimer`）在每個 kernel 開頭打時間戳，直接在 GPU 上量 kernel 之間的空檔（不靠 nsys：8 卡 torchrun 下 nsys 兩次寫不出報告）。
    另做單卡（無 NCCL）、只有其他卡在搬（是本卡還是整台主機的問題）。
  - **X2 普遍性**（約 2 天）：方向、塊大小、總速率、幾張卡同時搬、NUMA 本地 / 遠端、批次大小、TP8 / TP4 / TP2。
  - **X3 排程器**（3–4 天）：把 K4.1 的放行機制做成通用元件（每層多個放行點、依量到的時間窗決定塊大小、速率旋鈕），與限速、低優先權 stream、步間搬運、停下來等比較，畫取捨曲線。
  - **X4 案例研究**（約 1 週）：在 K2 服務迴圈（`k2_engine_v2`）上，一邊服務、一邊在背景載入新權重（每卡約 8 GB）；比較各排程對使用者首字 / 每字間隔與載入完成時間的影響；KV 搬回當第二個案例。
  - **X5 文獻回顧與寫作**：相近工作 ChunkFlow（arXiv 2605.11335：PCIe-only 機器、diffusion、預取與集合通訊搶同一條 PCIe）、ConCCL / DMA collectives（AMD）、NCCL 2.28 copy-engine collectives；
    我們的差異：NVLink 機器（通訊不走 PCIe）、TP 推論小 all-reduce 的量化與 LL 也受害、GPU 端訊號驅動的放行（屏障機制本身已見於 NVIDIA/nccl #2218，不是我們的發現）。
- 已有基礎：`results/kvtier/k4/`（interference_v2、place_v1、diag_v1_nsys/gaps.txt）、`results/e2e/c_diag`、`scripts/kvtier/k4_place_v1.py`（graph 內計數器放行）、`k2_engine_v2.py`。

- **X1 結果（2026-10-28，`results/xpcie/`，guard 全 CLEAN）——干擾的真正機制是「系統層級記憶體屏障」，不是 kernel 啟動延遲**：
  - `x1_v1`（`scripts/xpcie/x1_dispatch_v1.py`，CUDA graph、100 輪打亂；每個 kernel 開頭用 `%globaltimer` 打時間戳）：背景 PCIe 搬運時，
    **極小 kernel 的啟動間隔完全不變**（0.93 µs）、decode 大小的 GEMM 只 +1–3%，**NCCL 小 all-reduce 搬回 +69%、存出 +594%**；只有其他 GPU 在搬也 +81%
    （all-reduce 等最慢的那張卡）；模擬 decode 一步 +57% / +202%（與 K4.1 一致）。
    → K4.1 / STATUS 先前的推論「GPU 啟動 kernel 被延遲」**撤回**（那是逐個發 kernel + nsys 的現象）；預測 P1–P3 不成立、P4 成立。
  - `x1b_v1`（`x1b_latency_v1.py`，CUDA C 小 kernel、GPU0↔GPU1）：NVLink 互傳小旗標來回 2.6 → 5.2 µs（搬回）/ 11.2 µs（存出）；GPU 讀主機記憶體 1.3 → 3.2 / 4.7 µs；讀本卡記憶體不變。
  - `x1c_v1`（`x1c_fence_v1.py`）：**同樣的 NVLink 來回，寫入後用系統層級屏障（`__threadfence_system`）：2.6 → 5.5 / 12.0 µs；改用 GPU 層級屏障或不加屏障：1.1 µs，完全不受影響**。
    → NVLink 本身不受 PCIe 影響；被拖慢的是系統層級的記憶體排序（推論：它要等 PCIe 上排隊的資料；如何進一步驗證：看 NCCL 各協定在哪裡用系統層級的排序）。
  - `x1_proto_{LL,LL128,Simple}`（`NCCL_PROTO`，30 輪）：三種協定都受影響（存出時 +591% / +410% / +212%）→ 不能用 NCCL 設定避開。
  - 下一步：看 NCCL 原始碼找系統層級排序在哪；寫一個只用 GPU 層級同步（旗標與資料同一次寫入，LL 式）的 all-reduce，測是否不受 PCIe 影響——若是，得到不必限制搬運的第二種解法。
  - **原始碼與文獻（2026-10-28，不佔 GPU）——修正上面兩點的範圍**：
    - **機制不是新發現**：NVIDIA/nccl issue #2218（2026-06，V100 / H200，256 MiB all-gather 背景 D2H 時頻寬 −8–10%）的提問者已發現「系統層級屏障在背景 D2H 時延遲大增，
      改成 GPU 層級屏障影響大減」；NVIDIA 工程師回覆 sys 範圍的 membar 要確保「所有記憶體流量在系統層級的 POC 可見」，所以貴且對無關流量敏感，並認為 LL / LL128 較不受影響；
      2026-08 無修正關閉。更早：#1023（2023，維護者 sjeaugey：「走 NVLink 的話看不出 D2H 為何會干擾」）、PyTorch dev-discuss（H100，D2H offload 拖慢 all-gather）。
      → 「找到機制」不能當貢獻；可主張的是：TP 推論小 all-reduce 的量化（+69% / +594%，遠大於 #2218 的大訊息 −10%）、LL 也同樣受害（與 NVIDIA 的預期相反）、共用交換器的跨卡傳播、解法。
    - **NCCL LL 的受害原因還沒找到**：本 repo 附的 NCCL 2.19.3 原始碼（`3rdparty/nccl`，非實際執行的 2.21.5）：Simple 每送一段前 `fence_acq_rel_sys()`（`prims_simple.h:182`）→ 與 x1c 一致；
      LL128 在 sm80 用 GPU 層級 `__threadfence()`（`prims_ll128.h:84-89`，sys 只在 sm90+）；LL 完全沒有屏障（旗標在資料裡，`prims_ll.h:133`）；graph 模式工作描述在 GPU 記憶體（`enqueue.cc` persistent）。
      但 LL 實測 +70% / +591%。→ 「NCCL 變慢 = 系統層級屏障變慢」對 Simple 成立、對 LL / LL128 **未證實**（推論候選：kernel 結束時硬體做的系統層級排序、或其他經 PCIe 的存取）。
      如何驗證（X1d）：graph 內連發「寫一個字到對端 GPU 就結束」的極小 kernel，看背景搬運時是否變慢；背景改成 NVLink P2P 搬運（不走 PCIe）看屏障是否仍變慢；
      背景放在同交換器（GPU1）與不同交換器（GPU2）比較；改背景塊大小看是否隨排隊量變化；確認 2.21.5 原始碼與 2.19.3 是否不同。
      → **X1d 已回答（見下）**：LL 不是被屏障拖慢，是 graph 內 NCCL 加的 event 節點；2.21.5 與 2.19.3 的屏障位置相同（LL128 在 sm80 的 GPU 層級、Simple 的 `prims_simple.h:179/367`）。

- **X1d 結果（2026-10-28，`results/xpcie/x1d_v1` 100 輪 + `x1d_rep1–4` 各 20 輪（5 個獨立程序）、`x1e_*`、`x1_nomix*`、`x1_mix{1,0}_rep`；guard 全 CLEAN、時脈 1410 MHz 無降頻、背景在 ≥ 95% 的案例中撐到受測 kernel 結束）**：
  腳本 `scripts/xpcie/x1d_scope_v1.py`（單一程序控制 GPU0/1/2，CUDA C kernel 內用 `%globaltimer` 量）、`x1e_graphdump_v1.py`、`run_x1d_v1.sh`、`run_x1d_rep_v1.sh`；表格 `results/report_x1_20261028/x1d_scope.csv`、`x1_nomix.csv`（`export_x1d_v1.py`）。
  - **NCCL 在 graph 內的包法**（`x1e_v1/*.dot`、NCCL 2.21.5 `misc/strongstream.cc:181-229`）：每個 all-reduce kernel 前一個 EVENT_WAIT、後一個 EVENT_RECORD 節點（同一個 serialEvent），
    來自預設 `NCCL_GRAPH_MIXING_SUPPORT=1`；NCCL log 確認 16×5120 bf16 用 Ring + LL（5 blocks × 512 threads）。設成 0 後 graph 只剩 kernel 節點（`x1e_nomix`）。
  - **單卡受測項（GPU0，ns／項，5 程序一致）**：
    | 受測 | 沒搬 | GPU0 存出 | GPU0 搬回 | 只有 GPU1 存出（同交換器） | 只有 GPU2 存出（不同交換器） | GPU0 經 NVLink 搬給 GPU2 |
    | --- | --- | --- | --- | --- | --- | --- |
    | 極小 kernel＋event 記錄節點（`tiny_evrec`） | 1918–2246 | 9760–9814（×4.3–5.1） | — | — | — | — |
    | event 等待＋kernel＋event 記錄（`tiny_ev`，NCCL 的包法） | 1914–2296 | 9797–9864（×4.3–5.1） | ×2.1–2.5 | ×1.3–1.4 | ×1.0 | ×1.8–2.1 |
    | 寫對端後系統層級屏障（`fs_sys`，單獨） | 2457–2470 | ×2.64–2.67 | ×1.48 | ×2.31–2.32 | ×1.00–1.01 | ×1.81–2.15 |
    | 寫對端後 GPU 層級屏障（`fs_gpu`） | 352–357 | ×1.00 | ×1.00 | ×1.00 | ×1.00 | ×1.08 |
    | 極小 kernel（只寫本卡 / 寫對端不加屏障） | 957 / 1090–1095 | ×1.00 | ×1.00 | ×1.00 | ×1.00 | ×1.00 / ×1.03 |
    | LL 式 16 KiB 互傳（無屏障，`llpp`） | 5688–6002 | ×0.93–1.00 | ×0.99–1.04 | ×0.99–1.04 | ×0.99–1.00 | ×0.95–1.06 |
    背景頻寬：PCIe 存出 23.3 GB/s、搬回 21.1、GPU1/GPU2 存出 19.5–19.6、64 KiB 小塊 12.9、NVLink 274 GB/s（錨點：PCIe 單向 ~22、單對 peer copy 270 → 皆在 20% 內）。
  - **NCCL 端（8 卡，`x1_dispatch_v1.py`，GPU0 每項間隔 / rank-max 時間）**：
    | 協定 | mixing | all-reduce 沒搬 | 全部搬回 | 全部存出 | decode 一步 沒搬 | 全部搬回 | 全部存出 |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | LL（預設） | 1（`x1_v1` / `x1_mix1_rep`） | 31.9 µs | +69% / +38% | +594% / +513% | 204.5 µs | +57% / +55% | +202% / +197% |
    | LL（預設） | 0（`x1_nomix` / `x1_mix0_rep`） | 23.6 µs | +4.3% / +3.4% | +5.1% / +4.8% | 155.0 µs | +3.2% / +3.2% | +2.5% / +2.3% |
    | LL128 | 1 → 0 | 45.8 → 36.8 µs | +93% → +1.9% | +410% → +3.2% | 237.0 → 188.0 µs | +47% → +2.2% | +170% → +1.7% |
    | Simple | 1 → 0 | 134.6 → 126.7 µs | +113% → +70% | +212% → +203% | 415.2 → 362.1 µs | +70% → +49% | +123% → +127% |
    → 關 mixing **沒有背景搬運時也快 24–26%**（all-reduce 31.9 → 23.6 µs、decode 一步 204.5 → 155.0 µs；event 節點本身的開銷，與 PyTorch 論壇、TensorFold #597 的回報一致）。
    殘留：只有其他 GPU 搬回 +17%（rank-max 時間；GPU0 每項間隔 +4.1 µs）。
  - **事前預測**：D1 成立（同交換器 ×2.31、不同交換器 ×1.00）；**D2 不成立**（系統層級屏障也被本卡的 NVLink 搬運拖慢 ×1.8–2.2，不是 PCIe 獨有；但讀主機記憶體不受 NVLink 搬運影響）；
    D3 成立（64 KiB 小塊 ×1.46 < 4 MiB ×2.66，但頻寬也較低 12.9 vs 23.3 GB/s，兩者混在一起）；D4 成立（LL 資料路徑不受影響）；**D5 不成立**（kernel 寫完對端就結束不會變慢：沒有隱藏的 kernel 結束排序）；
    D6 大致成立（`fs_gpu` ≤ ×1.08；`pp_sys` 與 x1c 差 ≤ 9%）**但** `hostread` 在本卡存出時與 x1b 不符，見下；D7 成立；D8 成立。
  - **自我一致性發現（不得丟棄）**：「GPU0 讀主機記憶體＋GPU0 自己存出」在不同程序是**雙峰**：1.70 µs（x1b、x1d_smoke、rep2、rep3）或 4.86–5.00 µs（x1d_v1、rep1、rep4），
    同一程序內 100 輪都很穩定（x1d_v1 全距 4906–5092）。其他格子跨程序一致（≤ 3%，`fs_sys|nvl_g0` 例外 ×1.81–2.15）。原因未知（推論候選：主機緩衝區的實體頁位置、copy engine 指派）；
    如何驗證：同一程序內重新配置主機緩衝區、或固定 copy engine 再量。不影響 NCCL 結論（NCCL 端兩次獨立程序一致）。
  - **還沒解釋的**：8 卡 NCCL 在 mixing=1、全部存出時每次 all-reduce 多 157–182 µs，遠大於單卡 event 記錄節點多的 7.5 µs；放大的原因未知
    （推論候選：8 張卡的 event 等待／記錄串成鏈、每卡等最慢的一張；或記錄節點要等正在搬的那一塊搬完——4 MiB ÷ 23 GB/s ≈ 180 µs 只是巧合也可能）。如何驗證：x1 背景改塊大小（1 / 4 / 16 MiB）看多出的時間是否跟著變。
  - **意義**：(1) 對 LL / LL128（decode 大小的 all-reduce）有**零成本解法**：graph 專用 communicator + `NCCL_GRAPH_MIXING_SUPPORT=0`（限制：同一 communicator 不能混用 graph 與非 graph 的 NCCL 呼叫，
    NCCL 文件 env.html）；(2) Simple（大訊息、prefill 大小）、自己用系統層級旗標的 all-reduce（推論：vLLM custom all-reduce 用 `ld.acquire.sys` / `st.release.sys`，待查證）、
    以及不能關 mixing 的框架，仍需要排程類解法 → X3 的比較對象要加入「關 mixing」；(3) 新意：event 節點本身的開銷已有人回報，**被 PCIe 搬運放大 6 倍**與上述解法在初步搜尋中沒找到。
  - 下一步候選（待使用者確認）：K4.1 在 mixing=0 下重量（Flux 式放行還有沒有必要）；查 vLLM / SGLang 的 TP all-reduce 實際走哪條路、是否用系統層級旗標並實測；塊大小實驗解釋 182 µs 的放大；Simple 協定大訊息（prefill）的干擾與解法。

**最新（2026-10-16）—— CPU 當 KV 外存（K 系列）**：使用者決定先在本機 CPU 上做出**真正的加速**（晶片短期拿不到），選 P1 的 T3（KV 外存），
同意「計劃好就開始做」。
- 情境：多輪對話，GPU 放不下所有對話的 KV。現有做法（vLLM / SGLang 預設）：被擠掉的對話回來時整段重算 prefill。
  本做法：KV 存到 CPU 記憶體，回來時經 PCIe **一層一層搬回、邊搬邊算**（第 i 層一到就算第 i 層，Flux 的「資料到了就開算」以層為單位）。
- 借鑑：LMCache / SGLang HiCache 的逐層重疊與寫入時機（馬上存 / 用過才存 / 擠掉才存）、LMCache 的分塊索引、Cake（ICML 2025）的「GPU 從頭重算、PCIe 從尾搬回、中間會合」。
  我們自己的部分：裝置主動送（真晶片的記憶體 GPU 讀不到）、避開與 TP all-reduce 的干擾（C 診斷：PCIe 流量讓 NCCL 慢 1.7–5×）、介面不綁 NVIDIA。
- **K0+K1**（`scripts/kvtier/kvtier_v1.py`，拉取模式：GPU 自己從釘住的主機記憶體搬，每卡存自己那 1 個 KV head，主機記憶體放在該卡的 NUMA 節點）：
  單一請求首字時間，五條路徑交錯量——理想（KV 還在 GPU，下限）/ 重算（基準）/ 先搬完再算 / 逐層邊搬邊算 / Cake；外加純搬回、純存出的每卡頻寬（錨點 10–12 GB/s）。
  正確性：真實 ShareGPT 多輪對話（`scripts/kvtier/hf_reference_kv_v1.py` 挑歷史約 2k / 4k / 8k token 的三段），每條路徑前先把 GPU 上的 KV 蓋成雜訊；
  搬回路徑必須與 KV 留在 GPU 的路徑**逐位元相同**，所有路徑與 HF 對照。事前預測 K1–K8 寫在腳本開頭（跑之前）。
- **K2**（K1 之後）：工作負載層級——ShareGPT 多人多輪、GPU 的 KV 容量受限：量「重算佔 GPU 時間的比例」（= 加速上限）與實際首字時間 / 吞吐量；
  事先定的判斷：重算佔比 < 10% → 改走 MoE 專家外放路線。GPU 本身約可放 200 萬 token 的 KV（每卡 32 KiB / token），要嘛模擬很多使用者，要嘛限制 KV 容量（論文常用，需寫明）。
- **K3**：裝置主動送（模擬真晶片：KV 在裝置記憶體、經 hxb 中繼區 + 計數器送進 GPU），與 K1 的差距 = 真晶片路徑的代價。
- **K1 結果（2026-10-16，`results/kvtier/bench_v5/analysis.txt`，`kvtier_v5.py`，200 輪交錯、rank-max 中位數、逐案例時脈過濾，guard CLEAN〔1 筆 SELF_CPU 警告〕，尚未審查）**——
  單一請求、新一輪 128 token、所有計時路徑都是 CUDA graph：

  | 歷史 P | 重算（基準） | 理想（KV 還在 GPU） | **搬回、邊搬邊算** | 先搬完再算 | 純搬運（每卡） | 加速（保守：重算換算到 1410 MHz） |
  | --- | --- | --- | --- | --- | --- | --- |
  | 1024 | 77.7 ms | 20.1 | **21.6** | 24.0 | 3.95 ms（8.5 GB/s） | 3.6×（3.6×） |
  | 4096 | 234.0 | 20.8 | **23.5** | 34.4 | 14.0（9.6） | 9.9×（9.7×） |
  | 8192 | 450.4 | 21.4 | **28.4** | 47.3 | 26.4（10.2） | 15.9×（15.2×） |
  | 16384 | 920.5 | 22.4 | **53.5** | 73.7 | 51.7（10.4） | 17.2×（16.1×） |
  | 32000 | 1958.5 | 24.3 | **100.9** | 121.9 | 98.3（10.7） | 19.4×（17.8×） |

  - 邊搬邊算 = max(純搬運, 理想) + 1.5–2.7 ms：8k 以下搬運完全藏在計算底下（只比 KV 沒被擠掉多 1.5–7 ms）；16k 以上由 PCIe 決定。
  - 搬運頻寬與錨點一致（c_diag 8 卡同時 H2D 10.1–12.2 GB/s）：GPU0–3 每卡 11.3–12.4 GB/s、GPU4–7 9.3–10.7（主機記憶體已放在各自 NUMA 節點，差異仍在 → 推論是 GPU4–7 這側的 PCIe 拓撲，非 NUMA；如何驗證：nvbandwidth 逐卡 H2D）。存出 9.9–11.0。
  - **Cake 在這台機器上全面變慢**（+36% ~ +126%）：搬一個 token 的 KV 比重算它快約 15 倍，讓 GPU 分擔只是多跑一趟、並與 PCIe 互相干擾。適用於搬運比重算慢的情況（較慢的鏈路、多人同時搬）。
  - 正確性（`results/kvtier/check_v5`）：3 段真實 ShareGPT 對話（歷史 1639 / 3590 / 8354），搬回路徑與 KV 留在 GPU 的路徑 logits 與 64 個生成字**逐位元相同**；graph 與 eager 逐位元相同；與 HF 不同處都在 HF top-2 差 0 或 0.125 的位置。
  - 事前預測：G3（邊搬邊算 ≤ max+3 ms）、G4（加速倍數下限）、G6（Cake 不會更快）成立；G1（理想 8–20 ms，實測 20.1–24.3）、G2（P=1024 重算錄 graph 後應快 15–45 ms，實測 0.5 ms——1152 token 的重算本來就受 GPU 限制）、
    G5（差 1.6–3.3 ms，正是 G3 量到的邊搬邊算額外開銷）、G7（P=1024 搬運 8.5 GB/s < 9，小拷貝的固定開銷）不成立。
  - 引擎開銷的教訓：eager 的新一輪 128 token 要 49.5 ms（graph 20.1 ms），v1 / v2 的 smoke 數字量到的是 Python 發 kernel 的速度。
  - **這是單一請求的結果，不是系統層級的加速**：整體能快多少取決於多少請求會被擠掉、搬運會不會拖慢別人的 decode → K2。
- **K2 計劃（2026-10-16 使用者選完整版、不裝 vLLM 參考）**：多人多輪服務迴圈，GPU 只給固定數量的對話空間（縮小版，需寫明），使用者多於空間 → 系統自己擠掉最久沒說話的人。
  使用者 = 真實 ShareGPT 多輪對話（接回的完整對話），送出 → 等回答完（長度固定為資料集該輪回答長度，上限 512）→ 思考時間 → 下一句。
  三種做法同一套使用者與排程、各跑兩次交錯：重算（基準）/ CPU 外存（每輪結束順手存到 CPU、回來時逐層搬回邊搬邊算）/ 理想（空間無限）。
  量：首字等待時間（中位數 / 90% / 99%，按 沒被擠掉 / 搬回 / 重算 分開）、每字生成時間（搬運對他人的干擾）、吞吐量、重算佔 GPU 時間比例（< 10% → 改走 MoE）。
  程式：(1) decode kernel 依「第幾個人在第幾格」的對照表讀寫 KV；(2) 長度不固定的 prefill 也錄 graph（新的一句補齊到幾種長度、歷史長度從 GPU 記憶體讀）；
  (3) 排程器（rank 0 決定、廣播給 8 卡）；(4) 小規模試跑定參數 → 登記預測 → 正式跑。預估 2–3 天。
- **K2 試跑（`results/kvtier/k2/pilot0/`，各 90 秒，96 人 / 24 格 / 思考 30 秒，guard CLEAN）**：首字中位數 理想 29.0 / CPU 外存 30.5 / 重算 53.2 ms（90%：48.8 / 48.7 / 106.0）；
  吞吐量 705 / 704 / 697 字/秒；重算佔 GPU 時間 11.5%（過 10% 門檻）；背景存 KV 時他人 decode 不變慢（−2%）。engine 正確性 `results/kvtier/k2/check_v1`：搬回 = 留在 GPU 逐位元相同。
- **K2 事前預測（2026-10-16，正式量測前寫下）[推論]**：兩種負載，各 3 種做法 × 2 次交錯（理想、外存、重算、重算、外存、理想），暖身 180 s、量 600 s。
  輕負載 L（96 人、24 格）：L1 外存首字中位數在理想的 10% 內、90% 線在 15% 內；L2 重算首字中位數 ≥ 外存 1.5×、90% 線 ≥ 1.8×；L3 三者吞吐量差在 ±3% 內（使用者多在思考，GPU 有餘裕）；
  L4 重算佔 GPU 時間 10–20%；L5 背景存 KV 時 decode 每步在 ±3% 內；L6 每字間隔 99% 線 重算 ≥ 外存 1.5×。
  重負載 H（192 人、32 格、批次上限 48）：H1 重算佔 GPU 時間 25–45%；H2 外存吞吐量比重算高 ≥ 10%；H3 外存首字中位數比重算快 ≥ 2×；H4 外存首字中位數在理想的 25% 內；H5 背景存 KV 時 decode 每步在 ±5% 內。
- **K2 輕負載正式結果 L（`results/kvtier/k2/loadL/analysis.txt`，96 人 / 24 格 / 思考 30 s，每種做法 2 次 × 600 s 交錯，6 次 guard 全 CLEAN〔1 筆 SELF_CPU 警告〕，兩次之間差 < 2%）**：

  | | 首字中位數 | 90% | 99% | 吞吐量 | 每字間隔 99% | 重算佔 GPU |
  | --- | --- | --- | --- | --- | --- | --- |
  | 理想（不會被擠掉） | 29.4 ms | 46.4 | 100.1 | 705 字/s | 38.4 ms | — |
  | **CPU 外存** | **30.8** | **47.2** | **99.5** | 704 | **40.3** | — |
  | 重算 | 66.2 | 123.4 | 222.0 | 693 | 106.3 | 12.3% |

  - 外存比重算：首字中位數快 2.15×、90% 線 2.62×、99% 線 2.23×；最差的每字間隔 2.6× 好（重算時 prefill 卡住所有正在生成的人）；吞吐量只多 1.7%（使用者多在思考、GPU 有餘裕）。
  - 回來的人 63% 被擠掉過（2124 / 3362）；搬回的首字 31.2 ms ≈ 沒被擠掉的 29.9 ms（歷史中位數約 690 token）。
  - 背景存 KV 時他人 decode 每步 −1.2 ~ −2.0%（17 種批次大小一致**變快**，原因未明；推論：有存 KV 的是「剛有人講完」的那一步，批次組成不同；如何驗證：在隨機步插入假的存出、比較同一批次）。至少可確定沒有變慢。
  - 事前預測 L1–L6 全部成立。
  - 限制：ShareGPT 對話短（搬回的歷史中位數 690 token），絕對差距約 35 ms；GPU 容量是縮小版。
- **K2 重負載正式結果 H（`results/kvtier/k2/loadH/analysis.txt`、`decompose.txt`，192 人 / 32 格 / 批次上限 48，每種 2 次 × 600 s 交錯，6 次 guard 全 CLEAN〔3 筆 SELF_CPU 警告〕，兩次之間差 < 2%）**：

  | | 首字中位數 | 90% | 99% | 吞吐量 | 完成輪數 | 每字間隔 99% | 重算佔 GPU |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 理想（192 格） | 31.1 ms | 57.9 | 108.5 | 1377 字/s | 3325 | 44.3 ms | — |
  | **CPU 外存** | **35.2** | **474** | **1410** | **1370** | **3308** | **45.7** | — |
  | 重算 | 1239 | 3168 | 4456 | 1275 | 3071 | 129.2 | 28.0% |

  - 外存比重算：首字中位數快 35×、90% 線 6.7×、99% 線 3.2×；吞吐量 +7.5%（完成輪數 +7.7%）；最差每字間隔 2.8× 好。
  - **拆解（decompose.txt）**：外存每次 prefill（含搬回）中位數 20 ms，尾端是**排隊等格子**（等待 90% 線 0.43–0.50 s、99% 線 1.4 s）：32 格常被**正在生成**的人佔滿（22% 的步批次 = 32），沒有閒置的人可擠。
    → **CPU 外存只能幫閒置對話騰空間；正在生成的對話的 KV 必須在 GPU 上**——這是本方法的本質限制，剩下的差距是 GPU 記憶體對同時生成人數的上限。
    重算：prefill 中位數 66 ms，佔 GPU 28% → 每輪變長 → 格子滿（83% 的步 = 32）→ 等待中位數 1.2 s（連鎖惡化）。
  - 背景存 KV 時他人 decode 每步 −1.8 ~ −2.8%（同輕負載方向，原因未明；推論：存 KV 的那一步前 CPU 多做了取字，GPU 空檔多一點；如何驗證：在隨機步插入假的存出）。
  - 事前預測：H1（重算 25–45%）、H3（首字中位數 ≥ 2×）、H4（外存中位數在理想 25% 內，+13%）、H5（干擾 ±5%）成立；**H2（吞吐量 ≥ +10%）不成立：+7.5%**。H4 只約定中位數——外存的 90% 線是理想的 8×，須一併報告。
  - 限制：格子是固定長度（每格 4608 token，對話中位數約 1500），等同 GPU 容量的縮小版；用分頁式 KV 的系統同樣記憶體可放更多對話。
- **K4 計劃：方向一「避開 NVLink 的搬運排程」（2026-10-17 使用者同意；理由：Flux 指出 TP 的 all-reduce 在每層必經路線上）**：
  K2 的排程是「搬回時其他人停下來等」；長對話（16k–32k）搬回 50–100 ms，所有正在生成的人一起卡住。改成背景搬回，搬運會和他人 decode 的 NVLink all-reduce 重疊
  （c_diag：重疊的 10–80 MiB all-reduce 慢約 1.7×，但 decode 的 all-reduce 只有 0.1–0.3 MiB）。
  K4.1 量干擾（`scripts/kvtier/k4_interference_v1.py`，事前預測 I1–I4 在腳本開頭）→ K4.2 背景搬回 → K4.3 Flux 式錯開（decode graph 每層寫旗標、搬運引擎看旗標才搬一塊，只在 NVLink 空閒時搬）
  → K4.4 長對話（文件問答：第一句 8k–24k 的長文件 + 後續短問題）比較三種排程。K4.1 干擾 < 3% 則 K4.3 不需要。
- **K4.1 結果（2026-10-17，guard 全 CLEAN）——PCIe 搬運確實拖慢他人的 decode，而且「什麼時候搬」決定傷害大小**：
  - `results/kvtier/k4/interference_v2/`（CUDA graph、打亂順序、200 輪；32 條硬體佇列重跑結果相同 → 不是佇列共用）：背景全速搬時，decode 一步 搬回（H2D）**+45–48%**、
    存出（D2H）約 **+145%**（D2H 時運算比背景搬運還晚結束，被濾掉，數字取全部輪次）；prefill 2048 字 +6.7%、新的一句 128 字 +10%；搬運本身 10.6–12.4 GB/s。
    事前預測 I1（decode +0–10%）、I3（D2H 與 H2D 差 ≤ 5 點）**不成立**；I2（prefill +5–30%）、I4（搬運 9–12 GB/s）成立。
  - `results/kvtier/k4/place_v1/`（`k4_place_v1.py`，decode graph 每層放一個計數器 +1，搬運引擎等到計數器才搬一塊；100 輪打亂）：
    同樣的搬運量（每步 64 塊 × 0.5 MB），**在最大的矩陣乘法開始時放行：+2.8–3.7%**（含計數器本身 1.2–1.4%）；**在 all-reduce 開始時放行：+5.3–11.9%**；
    全速搬：+45–145%。**塊放大到 1 MB（超過矩陣乘法的時間窗）優勢消失**（+10–13%，與 all-reduce 放行相當）→ 傷害來自搬運壓在 all-reduce 上。
    換算每搬 1 MB 對他人的拖慢：錯開比全速好約 3.5×（H2D）到 6.6×（D2H）；代價是搬運速度 2 GB/s（全速 11.4）。預測 R1、R3 成立；R2 只在 D2H 成立。
  - 機制（`results/kvtier/k4/diag_v1_nsys/gaps.txt`，GPU0 的 CUPTI kernel 時間，逐個發 kernel + nsys，只作佐證）：有背景搬運時，**相鄰兩個 all-reduce kernel 之間的空檔 2.5 → 23.9 µs（約 10×）**；
    kernel 本身 H2D 時不變（56.7 → 55.6 µs）、D2H 時 +31%。→ 主要是 GPU 啟動下一個 kernel 被延遲；decode 一步有上千個小 kernel，所以受害最深，prefill 的 kernel 少而大幾乎不受影響。
    推論：啟動 kernel 要經 PCIe 讀指令／參數，被大量搬運塞住（D2H 更嚴重：讀取請求排在大量寫入之後）。Flux 式放行有效，是因為搬運只發生在一個長 kernel 執行期間。
  - 待解的矛盾：K2 的背景存 KV（每輪約 8 MB）沒有拖慢 decode（−2%），與這裡的 +145% 不一致；推論是量太小、只壓到少數幾層。
  - 撤回：逐個發 kernel 的分解實驗 `k4_diag_v2` / `v3` 的結論（基準在 11.4 / 12.9 / 17.0 ms 間漂移，只加計數器反而快 24%），已加 WITHDRAWN；`k4_diag_v1` 只當定性參考。
- **過程中修掉的問題**（細節見 JOURNAL 10-16）：ShareGPT 檔案把對話切成約 2k 一段（HF 參考 v1 / v2 撤回 → v3 接回）；decode graph 暖身蓋掉 KV 位置 0（kvtier v1 撤回；影響 E2E v3–v5 check，見撤回表）；
  eager 開銷主導（v2 bench 由 v3 取代）；錄 graph 時 all-reduce 輸出不能重用 → OOM（v3 撤回）；graph 與 eager 的 NCCL all-reduce 共用同一個 communicator 交錯幾百次後卡死（v4 撤回 → v5 給 graph 獨立 communicator，同設定 3 輪除錯跑完）。

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

**異質角色初步測試 P1（2026-10-15，`results/hetero_roles/p1/analysis.txt`，guard CLEAN）**——使用者要求「先初步測試調查到的方法，有用再延伸」：
- 背景：目前只確認「跑通」、沒有加速；根本原因是 B 把晶片串在每個字的必經路線上（GPU 等晶片），晶片再快也不會加速；主流做法都是讓晶片與 GPU **同時做不同工作**。
- T1 attention 外放（Lamina 式）：CPU 讀 KV 10–77 GB/s（頻寬上限 172 GB/s），一張 A100 0.84–1.77 TB/s → 在 8 卡旁只能分擔 0.1–0.5%（即使用滿頻寬 1.2%）→ **對低頻寬裝置無用**（修正 10-14 回覆中「HTX301 適合 attention」的說法：attention 吃頻寬，Lamina 用的是 4 TB/s 的 H20）。
- T2 MoE 專家外放（Fiddler / KTransformers 式）：權重不在 GPU 時，專家直接在裝置上算比搬到 GPU 再算快 **5.3–7.3×**（每專家 1–4 個 token，decode 常見）、16 token 時 1.8–2.0×；64 token 起搬到 GPU 較快 → **有用**，適用「模型放不進 GPU」的情境。
- T3 KV cache 存放於大記憶體裝置（推估，未做端到端）：續聊時從主機搬回 8k 前綴的 KV 約 27 ms vs 重算 prefill 約 398 ms（~15×）→ 有用（用記憶體，不用算力）。
- T4 晶片做 decode（P/D）：CPU 替身只多約 1.8% decode 能力 → 此替身下無用；實際效益取決於晶片頻寬。
- 量測過程抓到兩個基準錯誤並修正：專家權重重複使用而留在 L3（讀出「221 GB/s」超過 DRAM 錨點）→ 改輪流使用 ≥1.5 GB 的專家池；頻寬 kernel 受加法延遲限制（82→172 GB/s）。

**E2E 結果（2026-10-14，guard 全部 CLEAN，尚未審查）**：
- **B：PCIe 確實能與 NVLink 環境協同**（`results/e2e/bench_v3/`，v3 @ 77717fc，20 輪）：每個字都經過 GPU → PCIe → CPU → PCIe → NVLink 廣播，
  生成文字與 A 一致（Bfinal 4/4 題全同；Bmid 3/4 全同、第 4 題在 HF top-2 logit 完全平手處翻轉，`check/hf_margin_v3_vsA.json`）。
  代價：decode 每步 0.3–1.0 ms（B 減同樣拆 graph 的 Asplit 對照）；prefill Bfinal +0.1–0.7%、Bmid8 +2.5–4.1%、Bmid1 +4.4–5.3%；
  切 8 塊藏掉交棒時間的 22–46%。CPU 不增加算力 → 不可能加速（預期之內）。
  拆 graph 本身讓 decode 快 0.1–6.5%（AsplitM vs A）：本推論程式的 decode 有可觀的 graph 啟動開銷（A b1 每字 22 ms，b32 43 ms；G4 最佳約 11.5 ms）。
- **C：本機上 PCIe 通道無法加速 GPU 間通訊**（`results/e2e/c_bw/`）：all-reduce 分任何比例 f > 0 走主機中轉 PCIe，在 0.6–160 MiB 全部比只走 NCCL 慢（+63% 以上）。
  上限推算：NCCL 160 MiB algbw 約 102 GB/s，PCIe 每卡同時收送約 6 GB/s → 最多分擔 all-reduce 的 5.6%；all-reduce 約佔 prefill 23% → 端到端最多約 1.3%（推論，主機加總不計時間）；
  decode 訊息太小，PCIe 固定延遲已超過 NCCL 全程。事前登記的預測成立。使用者（10-14）不接受就此結論：要有足夠數據說明為何效果不佳，並嘗試讓 NVLink 與 PCIe 並存的排程（認為善加排程應能加速）。
  **已確認的 C 計劃**：(1) 拆開計時找原因（8 卡同時 D2H / H2D 的每卡頻寬、CPU 加總、同步、對 NCCL 的干擾）；(2) 試三種改良：主機端 C 加總 + 切塊流水線、
  GPU 自己加總（主機只轉送）、收送分時段避開同卡雙向掉速；(3) 純頻寬真的贏過 NVLink-only 才接進模型量端到端。事先告知的上限：all-reduce −5–10%、prefill 端到端約 −1–2%（推論）。
- **C 診斷結果（10-14，`results/e2e/c_diag/`）**：PCIe 流量讓同一張 GPU 的 NCCL all-reduce 慢 1.7–5×（nsys 證實是干擾而非排隊）；8 卡同時使用時 PCIe 每卡約 10 GB/s、
  同時收送時 H2D 2.3–2.9 GB/s；C 版主機加總 19 GB/s。→「與 NCCL 同時傳」受干擾限制；下一步排程方向待與使用者確認。
- **推論程式 v5（10-14）**：decode b1 / b32 / b64 = 14.8 / 16.4 / 18.1 ms（v3：22.1 / 43.0 / 61.2）；B v5 交棒代價 decode 0.23–0.77 ms / 步、prefill Bmid8 +3.6%。
- **順序（10-14 使用者）**：先加快推論程式（nsys：decode 每步約 3850 個 kernel，逐元素小 kernel 佔 GPU 時間約一半，`results/e2e/nsys_decode/analysis.txt`
  → v4 用 Triton 融合 kernel）→ C 的分析與改良 → ShareGPT。

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
| E2E v3–v5 正確性檢查的細節：「en_explain 只同 HF 5/64、因 HF top-2 差 0.625」與「Bmid 只在 code 第 18 字與 A 不同」 | `results/e2e/check/check_v3.json`–`check_v5.json`、§0 E2E 結果 | `Engine._capture` 錄 decode graph 前暖身 3 次、`pos` 緩衝區為 0 → 蓋掉 KV 第 0 個位置；check 模式每題每種設定第一次 decode 都會碰到（2026-10-16 由 KV 外存檢查 `results/kvtier/check_v1` 發現）。bench 計時不受影響（graph 在計時前建好）。en_explain 第 5 字的分歧是這個 bug 造成的，不是平手 | `scripts/e2e/qwen_tp_v6.py` 重驗（`results/e2e/check_v6/`，guard CLEAN）：A vs HF en_explain **50/64**、其餘 3 題 64/64；**Bfinal 與 A 4/4 題逐字相同**；Bmid 2/4 相同，另 2 題在近似平手處分歧（code 第 18 字 HF top-2 差 0.000；en_explain 第 50 字差 0.375，此處 Bmid 與 HF 相同、A 選了次高）（`hf_margin_v6_vsA.json`、`hf_margin_v6_vsHF.json`）。結論「B 的結果與 A 一致到數值誤差」不變 |
| P1 T3「8k 前綴搬回約 27 ms vs 重算約 398 ms（~15×）」 | §0 P1、`results/hetero_roles/p1/analysis.txt` | 推估，重算用的是 8×1024 批次（非單一 8k 序列），也沒算新一輪本身的時間 | K 系列實測（`results/kvtier/bench_v3/analysis.txt`） |
| kvtier v1 / v2 的首字時間（smoke 3 輪：理想 50–57 ms、Cake 比邊搬邊算慢） | `results/kvtier/bench_smoke/` | 逐個發 kernel 的 Python 開銷（約 45 ms）主導 128 token 的新一輪，重算也背同樣開銷 → 量到的是引擎開銷而非記憶體分層；事前預測 K3（理想 10–30 ms）因此不成立 | `scripts/kvtier/kvtier_v3.py`（全部計時路徑錄成 CUDA graph，hxb 計數器在 graph 內逐層把關） |
| X1「干擾機制是系統層級記憶體屏障」（套用到全部 NCCL） | §0 X1 結果、報告頁「原因」卡片（2026-10-28 第 3 版） | LL / LL128 沒有系統層級屏障；X1d：關 `NCCL_GRAPH_MIXING_SUPPORT` 後干擾消失（+594% → +5%） | NCCL LL / LL128：graph 內 event 記錄節點被搬運拖慢；Simple：每段的系統層級屏障（X1d） |
| 「背景搬運對 prefill／大訊息集合通訊影響小（約 +7%）」（10-29 選題時的說法，以 K4.1 prefill 2048 字 +6.7% 為據） | §0 10-29「(3)」、JOURNAL 10-29 | 只對 ≥ 128 MiB 或主機→GPU 成立；T1：GPU→主機搬運時 16–64 MiB 的 RS / AG / AR 慢 ×1.4–3.0（eager、30 輪、guard CLEAN） | `results/topic/analysis.txt`：影響依大小、協定、方向而定 |
| x1c / 說明「系統屏障慢是因為等 PCIe 排隊」（PCIe 獨有） | §0 X1 x1c 推論、對使用者的說明 | X1d：本卡經 NVLink 的 copy engine 搬運也讓系統屏障慢 ×1.8–2.2（D2 不成立） | 系統屏障被本卡任何 copy engine 流量與同交換器鄰卡的 PCIe 流量拖慢（X1d） |

## 4. 下一步（worker 維護）

**2026-10-14**（取代 10-06 的清單；1a / 1b 已由 I2 特性量測 D 段涵蓋：同卡雙向 H2D 5.95–7.3 GB/s、GPU0 送 + GPU2 收無競爭）：
1. E2E 步驟 1：下載 Qwen2.5-32B-Instruct；精簡 TP8 推論程式（NCCL）；輸出與 HF 一致；實測 CPU 算一層的時間（驗證第 0 節的推論）；量 A。
2. E2E 步驟 2：B（CPU 做 final RMSNorm，經 hxb）；文字與 A 一致；A / B 交錯量測。
3. E2E 步驟 3：C 先量純頻寬（NVLink-only vs NVLink + PCIe 中轉分流），再接模型。
4. E2E 步驟 4：ShareGPT。
5. 寫 I3 / I4 分析報告；I5 視需要再跑；安排 auditor。

## 5. 產出到 common 的數字

（尚無。1a → `pcie.contention_locus`；1b → `staging.fixed_overhead_ms`。）
