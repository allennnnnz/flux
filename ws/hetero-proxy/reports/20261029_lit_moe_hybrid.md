Supersedes: （無）

# MoE 專家「GPU＋PCIe 裝置」混合推論：文獻與開源實作調查

日期：2026-10-29（boss 兼 worker）
緣起：X1d 之後，使用者認為「關掉 NCCL 一個設定」不算貢獻，要找排程相關的論文題目。背景權重載入（題目 A）細算後問題太小：每卡 8 GB 以 9–12 GB/s 約 0.7–0.9 秒（推論，用 K1 頻寬算），所以改看「PCIe 上有持續、在關鍵路徑上的流量」的情境：MoE 冷門專家放在 PCIe 裝置（現在是 CPU，將來是國產晶片）上計算。
使用者要求：「查一下別人有沒有在做，可以直接從人家做法中去改善」。

本報告未做任何 GPU 量測。文獻摘要只讀了摘要頁的，會標明；原始碼部分的結論都附檔案與行號。

## 1. 誰在做什麼

| 類別 | 代表 | 硬體 | 決定什麼 | 與我們的關係 |
|---|---|---|---|---|
| 1 張 GPU＋CPU | Fiddler（arXiv 2402.07033）、MoE-Lightning、**KTransformers（SOSP'25）**、HybriMoE（2504.05897）、DAOP（DATE'25）、PreScope / LayerScope（2509.23638，ICS'26）、DALI（2602.03495）、Adaptive Expert Assignment（2610.23006，2026-10）、2512.16473 | 單張 GPU（A100 40GB / 4080 / 消費級），雙路 Xeon | 每個專家在 CPU 算，還是把權重搬到 GPU 算（依 token 數）；預取；GPU 上的專家快取 | **最擠**。P1 T2 的結論（1–4 token 時在裝置算快 5.3–7.3×）他們都已用過 |
| GPU＋記憶體端運算裝置（NDP） | MoNDE（2405.18832，CXL 記憶體裝置算冷門專家）、PIMoE、CXL GPU–NDP（2512.04476）、DATE'26 南京大學、DynaNDE（2609.00407，NPU＋NDP，2026-08） | 多為單一 GPU / NPU（DynaNDE 摘要未寫硬體） | 熱門專家搬到 GPU、冷門專家在裝置內算；動態排程 | **最像「國產晶片算冷門專家」**，但都是單一加速器，多為模擬（推論，未讀全文） |
| 多張 GPU＋CPU | **KTransformers 併入 SGLang（LMSYS 2025-10-22）**：TP8＋CPU 專家；MoE-Infinity（2401.14361）：EP 時專家依 NUMA 分給該節點的 GPU；2606.10493：2 張 5090＋雙路 EPYC | 8×L20＋雙路 Xeon Gold 6454S；2×5090 | 熱門專家固定放 GPU（`--kt-num-gpu-experts`），其餘在 CPU | **最接近我們的環境，而且開源，是直接改良的對象**（第 2 節） |
| 拓撲 / PCIe 感知 | CAP（2607.05116）：依通訊拓撲放專家（只有 GPU）；Predictable LLM Serving（2508.20274）：避免把延遲敏感的租戶放在忙碌的 PCIe 交換器下；Memory Offloading w/ SLO（2502.08182）：同一條 PCIe 上的 GPU 互搶；OptPipe（訓練，同一交換器的卡不同時卸載）；Harvest（2602.00328）：把專家權重放在鄰卡閒置記憶體，經高速互連取 | 各異 | 放置與時機 | 「同一交換器的卡會互搶」已有人用在排程；**沒找到**把它用在多 GPU＋PCIe 裝置的 MoE 上的 |

沒找到的東西（2026-10-29 搜尋）：
- 多張 NVLink GPU＋一個 PCIe 裝置，每張 GPU 經自己的 PCIe 線和裝置交換資料（使用全部上行線），並考慮同一交換器互搶。
- 在 MoE 混合推論中，把 CPU 回呼換成 GPU 端計數器。
- 處理「PCIe 搬運拖慢 NCCL」的 MoE 混合推論系統（X 系列）。

## 2. KTransformers＋SGLang 的 8 卡做法（讀原始碼）

版本：sglang `d213124`（2026-10-30 +0800）、ktransformers `c987331`（2026-10-28 +0800），sparse clone 在 scratchpad，未提交。

1. **只有 GPU0 和 CPU 交換資料。** `python/sglang/srt/layers/moe/kt_ep_wrapper.py`：
   - `submit()` 第 299 行、`sync()` 第 322 行：`if self.tp_rank != 0 ... return`。
   - `apply()` 第 361–383 行：只有 rank 0 送出 CPU 工作，也只有 rank 0 把 CPU 結果加回自己的輸出；其他 7 張卡的 CPU 部分是 0，由後面的 all-reduce 合起來。
   - 所以每一層 MoE 的 CPU 往返都只走 GPU0 的 PCIe 線。在本機，這條線和 GPU1 共用。其他 7 張卡在下一個 all-reduce 等 GPU0。
2. **GPU 等 CPU 用的是 CPU 回呼。** `kt-kernel/cpu_backend/cpuinfer.h`：
   - 第 96 行，送出工作：`cudaLaunchHostFunc(stream, func, args)`。
   - 第 121 行，等結果：`cudaLaunchHostFunc(stream, &sync_, args)`。`sync_` 在回呼裡阻塞，等 CPU 的工作佇列做完，這段時間 stream 停住。
   - 論文第 4 節：把 submit / sync 包成 host function，整個 decode 一個字可以放進一個 CUDA graph，「up to 1.23×」。
3. **專家怎麼分是靜態的。** `kt_ep_wrapper.py` 的 `mask_cpu_expert_ids`（第 99 行）：編號 ≥ `num_gpu_experts` 的專家在 CPU 算，其他在 GPU；沒有每一步依 token 數改變（這個 wrapper 內）。
4. **CPU 端的 NUMA 做法：** 每個專家的權重切成兩半，每個插槽一半，各自在本地算，再做一次 reduce-scatter（論文第 4 節，「up to 1.63×」）。合起來的結果仍然只經 GPU0 回到 GPU。
5. **已支援多家加速器當「GPU 端」**：`kt-kernel/cpu_backend/vendors/` 有 `ascend_npu.h`、`musa.h`（摩爾線程）、`maca.h`（沐曦）、`hip.h`。但「慢的那一端」一律是 CPU。
6. **8 卡效益**（LMSYS 部落格，8×L20、DeepSeek-V3 int4）：只有 1 個請求時，8 卡比 1 卡只快 26%；8 個請求同時時快 264%。ShareGPT 平均每字間隔 431.61 ms，P99 1935.13 ms。

## 3. 可以改良的地方（全部是推論，每條附驗證方法）

| # | 他們的做法 | 改良 | 預期影響在哪 | 如何驗證 |
|---|---|---|---|---|
| H1 | 只有 GPU0 和 CPU 交換 | 每個插槽和自己 NUMA 的 GPU 交換（GPU0–3 ↔ 插槽 0，GPU4–7 ↔ 插槽 1），4 條上行線都用上。CPU 的部分和直接交給已經存在的 NVLink all-reduce 加總，省掉 CPU 端的 reduce-scatter 和跨插槽傳送 | prefill（每層活化值大：4096 字 × 7168 × 2 B ≈ 58 MB，單線 23 GB/s 約 2.5 ms 一個方向）；decode 的活化值只有 KB 級，影響小 | 在 rank 0 和 rank 1 量每層的時間線：D2H、回呼、CPU 計算、H2D、其他卡等待 |
| H2 | CPU 回呼（`cudaLaunchHostFunc`）讓 GPU 等 CPU | 改用 hxb 的主機記憶體計數器：GPU 用 stream memory op 等，不經驅動的回呼執行緒（`scripts/hxb/`，已在 E2E B 驗證過正確） | decode：每層 2 次 × MoE 層數；能省多少取決於回呼延遲 | 微基準：本機的回呼來回 vs `cuStreamWaitValue` 來回（單卡，幾分鐘） |
| H3 | 專家分配靜態；要搬權重時（別的系統）只走一條線 | TP 下每張卡只需要每個專家的 1/8。8 張卡各經自己的線搬自己那份，總頻寬約為單線的 4 倍（本機 4 條上行線），「在裝置算 vs 搬到 GPU 算」的分界點會從 P1 的約 64 token 往下移 | prefill、大批次 | 用 K1 的頻寬算分界點，再實測；同一交換器的兩張卡搬一半時是否真的只剩一半頻寬（X 系列待做） |
| H4 | 沒有處理 PCIe 搬運對 NCCL 的干擾 | 用 X 系列的知識：`NCCL_GRAPH_MIXING_SUPPORT=0`、搬運用 GPU 端計數器放行（K4.1）、避免和 Simple 協定大 all-reduce 重疊 | 一旦加入 H3 的搬運，decode 會被拖慢 | 在 H3 的搬運下量 decode（mixing 開／關） |
| H5 | 「慢的一端」只能是 CPU | 把慢的一端換成任何 PCIe 裝置（hxb 介面已經接過 CPU 和 A100 代理），對應最終的國產晶片 | 題目的異質性 | hxb 換後端，與 CPU 版逐位元比對 |

H1–H4 合起來就是一個排程問題：每一層、每一個專家，決定在哪裡算、資料走哪條 PCIe 線、什麼時候放行搬運。

## 4. 風險

- KTransformers 團隊仍在開發，部落格說 SGLang 版是「proof-of-concept」，之後可能自己修掉 H1。要盯他們的 issue / PR。
- 本機 CPU 是 2× Xeon Platinum 8358（Ice Lake，有 AVX-512，**沒有 AMX**）。kt-kernel 文件說有 AVX2 / AVX-512 路徑，但支援哪些量化格式要實裝確認。CPU 專家會比論文的 AMX 數字慢。
- 模型：要有「GPU 放不下、所以有專家在外面」的理由。主機記憶體 1.5 TB 夠；磁碟只剩 672 GB。
  - DeepSeek-V3 bf16 約 1.3 TB，放不下。
  - Qwen3-235B-A22B bf16 約 470 GB，放得進 8×80 GB，只能用「GPU 記憶體預算」人為限制（論文常見做法，但要說明理由）。
- SGLang 新版大概需要比 pixi 環境（PyTorch 2.6）新的 PyTorch，要另建 venv（推論，安裝時確認）。

## 5. 建議的第一步（2–3 天，待使用者同意）

1. 另建 venv 安裝 SGLang＋kt-kernel，確認在 Ice Lake 上能跑 TP8＋CPU 專家（先用小的 MoE，例如 Qwen3-30B-A3B，把 `kt-num-gpu-experts` 調小，強迫部分專家在 CPU）。
2. 量 rank 0 和 rank 1 每層 MoE 的時間組成（GPU 時間戳，guard＋tmux）：D2H、回呼、CPU 計算、H2D、其他卡等待。這一步量化 H1 和 H2 有多少空間。
3. 單卡微基準：`cudaLaunchHostFunc` 往返 vs hxb 計數器往返。

如果第 2 步顯示 CPU 計算佔掉幾乎全部時間（交換和回呼不到 10%），H1 / H2 的空間就小，題目要轉往 H3（多線搬權重的分界點與排程）。

## 來源

- KTransformers SOSP'25 全文：https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
- KTransformers × SGLang：https://www.lmsys.org/blog/2025-10-22-KTransformers ；https://github.com/sgl-project/sglang/issues/11425
- HybriMoE：https://arxiv.org/abs/2504.05897 ；DALI：https://arxiv.org/abs/2602.03495 ；PreScope / LayerScope：https://arxiv.org/abs/2509.23638 ；DAOP：https://arxiv.org/abs/2501.10375
- Adaptive Expert Assignment：https://arxiv.org/abs/2610.23006 （只讀摘要）；2512.16473：https://arxiv.org/abs/2512.16473
- MoNDE：https://arxiv.org/abs/2405.18832 ；DynaNDE：https://arxiv.org/abs/2609.00407 （只讀摘要）；CXL GPU–NDP：https://arxiv.org/abs/2512.04476
- MoE-Infinity：https://arxiv.org/abs/2401.14361 ；2606.10493：https://www.emergentmind.com/papers/2606.10493 （只讀摘要頁）
- CAP：https://arxiv.org/abs/2607.05116 （只讀摘要）；Predictable LLM Serving：https://arxiv.org/abs/2508.20274 ；Memory Offloading w/ SLO：https://arxiv.org/abs/2502.08182 ；Harvest：https://arxiv.org/abs/2602.00328 （只讀摘要）

## 6. 改良的主軸（2026-10-29 補，回答「打算怎麼改良」；全為推論）

**核心**：8 卡時 GPU 大多在等 CPU 算冷門專家。用 4 條上行線同時把一部分冷門專家搬進 GPU 算，讓「CPU 算」和「8 卡搬來算」同時進行、同時做完。

估算（P1 T2 `results/hetero_roles/p1/analysis.txt` 是單卡單線，約 19 GB/s：Mixtral 專家 352 MB 搬＋算 18.2 ms；CPU 算 4 token 3.45 ms）：
- 8 卡各搬 1/8、4 條線約 90 GB/s（K1：8 卡同時每卡 9–12 GB/s）→ 搬＋算約 18.2 / 4.7 ≈ 3.9 ms，和 CPU 算差不多。
- 兩邊分工的處理速度：1/3.45 + 1/3.9 ≈ 0.55 個專家/ms，只用 CPU 是 0.29 → 約 1.9×。單卡單線只有 1/3.45 + 1/18.2 → 約 1.19×。
  → 多出來的上行線是 8 卡伺服器才有的資源；KTransformers 只經 GPU0 一條線，用不到。
- 「裝置算 vs 搬到 GPU」分界從 P1 的每專家約 32 token 降到約 4–8 token。
- 未計：主機 DRAM 被 CPU 計算和 DMA 同時讀的互搶、每塊 1/8 變小後的固定開銷、搬運拖慢 NCCL（X 系列）。

組成：
1. 排程器：每層依 router 結果（每個冷門專家的 token 數）決定 CPU 算或 8 卡搬來算，讓兩邊同時結束。時間模型要包含同一交換器兩卡互搶、DRAM 互搶。
2. 權重擺放與 TP 切法一致：專家中間維切 8 份；第 0–3 份放插槽 0 的記憶體，第 4–7 份放插槽 1。CPU 算時各插槽算自己那半（同 KTransformers）；GPU k 搬第 k 份，都在本地 NUMA。
3. H1＋H2：每個插槽和本地 GPU 交換；回呼換 hxb 計數器。
4. H4：`NCCL_GRAPH_MIXING_SUPPORT=0`；搬運用 GPU 端計數器放行；活化值交換優先於權重搬運。

**最便宜的否證實驗（1–2 天，不需裝 SGLang）**：同一時間 (a) CPU 兩個插槽算 k 個專家，(b) 8 卡各搬 1/8 的另外 m 個專家。量兩邊各自的速度和互搶程度，看約 1.9× 是否成立。
不成立（例如 DRAM 互搶吃掉大半），主軸就退回 H1 / H2。
