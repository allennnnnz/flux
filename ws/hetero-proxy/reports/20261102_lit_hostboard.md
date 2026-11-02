Supersedes: 無（補充 reports/20261102_pflux_plan.md §9）

> 由 hetero-proxy session 派出的文獻子代理撰寫（2026-11-02），未經 auditor。原始下載檔留在該 session 的 scratchpad，未提交。

# 經主機「上傳一次、大家下載」all-gather 的前人做法（文獻調查，2026-11-02）

範圍：只讀文獻與原始碼，沒有跑任何 GPU 程式。標記規則：
- 「推論」＝我從原文/原始碼推出來、原文沒直接寫的。
- 「未確認」＝沒能從全文或原始碼核實的。
- 引文盡量照原文（英文），附章節或檔案行號。

讀過的原始材料（存在 scratchpad，供複查）：
- ThunderEP HTML 全文：`.../scratchpad/lit/dl_thunderep/thunderep.txt`（arXiv 2609.40093v1）
- LLMQ HTML 全文 + repo：`.../scratchpad/lit/dl_llmq/llmq.txt`、`.../dl_llmq_repo/llmq`（commit befeb03，2026-08-31）
- NCCL 原始碼：最新 v2.32.3（`.../nccl_src/nccl`，commit 3e46c08，2026-10-08）；舊版 shm.cc v2.21.5 / v2.30.3 / v2.30.4 / v2.30.7（`.../nccl_old/`）
- Demystifying NCCL（arXiv 2507.04786v3）、Fire-Flyer AI-HPC / HFReduce（arXiv 2408.14158v2）、NSDI'24 rPCIeBench（Hou et al.）
- MPICH `release_gather.h`（main，2026-10-28）、Open MPI `coll_xhc_bcast.c`、`coll_accelerator_allgather.c`（main）
- nvbandwidth repo（commit 82fc4e8）、nvbandwidth issue #53、NVIDIA 論壇 thread 352186

本機事實（唯讀檢查，非 GPU）：CPU = 2× Intel Xeon Platinum 8358（Ice Lake-SP）；每兩張 GPU 共一個 PCIe 交換器（`nvidia-smi topo -m` 中 NIC 為 PXB）；pixi 環境的 NCCL = **2.21.5**；`/dev/gdrdrv` 存在且 world rw，`libgdrapi.so.2.5` 已安裝；`EnablePCIERelaxedOrderingMode: 0`。

---

## 1. ThunderEP（arXiv 2609.40093v1，2026-09-30，SNU Jaejin Lee 組）

**全文已讀。** 注意：我們 plan §9 寫「ThunderEP 做 MoE 的 all-to-all」**不正確**——它把 dispatch/combine 寫成 **All-Gather + Reduce-Scatter**（§2.1 "An alternative is an All-Gather and Reduce-Scatter formulation"；§4.5 "ThunderEP therefore retains All-Gather and Reduce-Scatter and leaves efficient routing-aware All-to-All communication on consumer GPUs to future work"）。也就是說，它做的正是 dense AG/RS，跟我們 TP 的 AG/RS 是同一種集合操作。

### 1.1 主機記憶體的用法：write-once / read-many（不是轉手）
- §4.2："each GPU writes its own shard once into host memory buffer, and then, every GPU reads the remaining N-1 shards back directly. No shard is ever forwarded between GPUs, so the pinned host memory serves as a single broadcast medium that each GPU writes to once and reads from as needed."
- 流量：ring 每卡 2(N−1)S，single-step 每卡 N·S；比值 2(N−1)/N（N=6 時 10S→6S，§5.2）。N=8 時比值 1.75（推論，代公式）。
- RS（combine）：兩種都實作（Fig. 6）。GPU reduction：每卡上傳 N−1 塊、目的卡下載 N−1 塊自己加，流量 2(N−1)S，與 NCCL ring 相同。Partial CPU reduction：流量 N·S，但被否決——§4.5 "the aggregate PCIe bandwidth exceeds the available host DRAM bandwidth, leaving little bandwidth for CPU reduction. The resulting CPU reduction overhead exceeds the time saved by reducing PCIe traffic. We therefore adopt GPU reduction"。

### 1.2 同步
- 旗標放 pinned host（§4.1 "The completion flag buffer must therefore reside in pinned host memory"）。
- 旗標與資料分開、單一寫者、不用 atomic、少量 thread 輪詢（§4.4）："One thread per remote sender polls its flag with uncached loads and pauses between attempts, while the remaining threads wait on a word in GPU memory. Each flag has a single writer, so no atomic operation is required." → 輪詢者是 **GPU kernel 的 thread**（文中 "threads … GPU memory"），**沒有提到 cuStreamWaitValue**（未確認 DMA 路徑是否另用 stream memory op）。
- per-sender 釋放（Fig. 10）："a GPU begins reading from each sender as soon as its completion flag is available"。
- 主機緩衝區雙緩衝 + 接收方回 credit，credit 用 **PCIe 寫**而非讀（§4.4）："Receivers send these credits through PCIe writes. Writes do not wait for a response, whereas reads do and saturate earlier"。
- prefill 的計算重疊：§4.5 "The host submits transfers and expert computation on each GPU's token activations to separate CUDA streams and connects their dependencies with CUDA events." 粒度＝「每個來源 GPU 的整份 shard」，不是 tile。
- CUDA graph 陷阱（§4.5）："A captured cudaMemcpyAsync operation for a DMA transfer reuses the address recorded at capture time, so alternating buffers would require the host to update the graph before every replay. ThunderEP therefore performs the data transfers of the decode phase on the SMs"。

### 1.3 Copy engine vs SM
- prefill 走 DMA（§4.3 "we instead issue transfers on the GPU's DMA engines"，API 舉例 cudaMemcpyAsync，§2.2）；decode 走 SM（因 CUDA graph，見上）。reduction 留在 SM。
- 兩條 stream：§4.3 "placing outgoing DMA transfers on one stream and incoming transfers on a second … a single DMA queue is deep enough to saturate its direction on its own, so two streams are enough."
- 消融（Table 3，Qwen3-30B-A3B，RTX 5090）：DMA 取代 SM 搬運 prefill 再 ×1.07；§5.4 "DMA and SM transfers provide similar bandwidth when executed alone."

### 1.4 Chunk 大小
- §4.3 把傳輸切塊，讓「上傳第 k 塊」與「下載第 k−1 塊」重疊（Fig. 8b）。**沒給數字**："we use the smallest chunk count that keeps both lanes busy through the steady state." → chunk 大小 未確認。

### 1.5 雙向不對稱 / 交換器共用
- **沒有觀察到、也沒有模型化「同卡 H2D 被 D2H 拖慢」**。相反地，它**主動讓兩方向重疊**來吃滿 full duplex（§4.3 "A PCIe link is full-duplex, so it can carry a send on the upstream lane and a receive on the downstream lane at once"），並把 AG 的頻寬優勢部分歸功於此（§5.2 "ThunderEP overlaps D2H and H2D transfers on separate DMA streams and uses both directions of the full-duplex PCIe link. NCCL instead serializes the two directions within each ring channel."）。
- 它對 NCCL 的說法（§4.3）："a barrier within each communication channel forces the send of an internal chunk to complete before its receive begins. The two directions thus alternate chunk by chunk"——這是作者的詮釋，我沒在 NCCL 原始碼逐行核對（未確認）。
- 有量 **host bridge 共用**（§4.1，RTX 5090 Gen5）："a GPU communicating alone reaches 56.5 GB/s … When a second GPU under the same PCIe host bridge communicates concurrently, the bandwidth of each GPU falls to 35.5 GB/s."

### 1.6 硬體、基準、數字
- 硬體（§5.1, Table 1）：單節點 6× RTX 4090（Gen4 x16）/ 6× RTX 5090（Gen5 x16）；1× AMD EPYC 9124（16 核，Genoa）；8× DDR5-4800；三個 host bridge 各掛 2 卡；單一 NUMA；無 P2P；driver 580.65.06。
- 基準：NCCL v2.30.7（集合操作層級）；vLLM v0.27.1、SGLang v0.5.18、Megatron-Core v0.19.2（端到端）。每點 5 次平均。
- AG（dispatch，Fig. 11，§5.2）：1KB–1MB 平均延遲加速 2.30×（4090）/ 2.65×（5090）。1 GB：**26.5 vs 20.8 GB/s（4090）、35.3 vs 23.0 GB/s（5090）**（約 1.27× / 1.53×，比值為我計算）。這裡的 GB/s 是 algbw 還是 busbw，原文沒說（未確認）。
- RS（combine，Fig. 12）：小訊息 1.91× / 2.14×；1 GB 幾乎打平：21.0 vs 21.3（4090）、24.5 vs 23.2（5090）。
- 摘要寫 dispatch 2.00×、combine 1.53× 平均；怎麼平均出來的，正文無法對回（未確認）。
- single-step 演算法本身（§5.4，每卡 8 MB）：N=4 時 1.26×、N=6 時 1.58×；N=2 打平。
- 端到端（RTX 5090，§5.3）：prefill 對 vLLM 平均 1.42× / 1.33× / 1.22×（三模型），最高 1.66×；decode 1.17× / 1.12× / 1.16×，最高 1.26×。消融（Table 3）：演算法+同步 1.15×，+DMA 1.23×，+DMA 重疊 1.47×（prefill）。對「自行加了 async 重疊的 vLLM」仍有 1.24×。
- 值得注意：在它的平台上 **NCCL SHM 的 AG 1 GB 能跑到 20.8–23.0 GB/s**；我們這台 NCCL 只走 PCIe 時 busbw 7.4–7.6（我們自己的量測）。兩邊 NCCL 的差距可能正來自我們平台的同卡雙向崩落（推論；驗證：在本機用 nsys 看 NCCL SHM ring 每卡的 H2D/D2H 是否同時在飛、各自速率）。

---

## 2. LLMQ（Schultheis & Alistarh，IST Austria；arXiv 2512.15306，2025-12-17；CPAL 2026, PMLR v328）

存在且相關，但它不是我們這種「主機公告欄 + 旗標」設計。

- **主機當共用媒介**：§3.2 "Weight caching on host"："on recent consumer devices, the GPUs cannot communicate directly … placing weights in host memory actually reduces the required communication: They need to be sent from GPU to host during the first forward pass after each optimizer step, but the backward pass, and all subsequent forward passes for gradient accumulation, can use the values cached on the host." → 是 write-once / read-many（跨 GPU 也跨時間重用）。
  - 原始碼佐證：`src/models/llama_weights.cpp` `WeightsMgrSharded` 建構子中 `quant_alloc = mOffloadQuants ? options.offload_alloc() : ON_DEVICE`（約 L727–731）；`offload_alloc()` 是 `PINNED` 或 `WRITE_CMB`（`src/models/llama_model.h:73-75`；`src/utilities/allocator.cpp:61-66` 用 `cudaMallocHost` / `cudaHostAlloc(WriteCombined|Mapped)`）。AG 實作 `NCCLCommunicatorThreads::gather_weight`（`src/utilities/comm.cpp:536-548`）：先在 CPU thread 之間交換各 shard 指標，再對每個 shard 發 `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, stream())`。shard 在主機時，這等於每卡各自從同一份主機 shard 下載（推論，依程式碼）。shard 在 GPU 上、又沒有 P2P 時，跨卡 D2D 由 CUDA driver 內部經主機中轉——每個讀者各觸發一次中轉，就不是 write-once（推論；driver 中轉細節未確認）。
- **同步**：單一 process、一卡一個 CPU thread；`std::barrier` + 交換 cudaEvent + `cudaStreamWaitEvent`（`comm.cpp:570-578`、`604-612`）。**沒有主機旗標、沒有 cuStreamWaitValue、沒有 GPU 端輪詢。**
- **Copy engine**：§3.2 "cudaMemcpy-based communication"："When running all-gather … and reduce-scatter … with nccl, we observed that the PCIe link utilization was quite low. On the other hand, memory transfers handled by the GPUs copy-engines, as initiated by cudaMemcpy, do achieve much better utilization, and do not require reserving any SMs." RS 拆三段：CE 搬、SM 加（Fig. 1）。
- **Chunk**：以 transformer block 為單位、雙緩衝（`mWork.Blocks` 2 份，`llama_weights.cpp` 約 L713-718）；沒有更細的 chunk。
- **雙向不對稱**：沒提。另一個相關觀察（§3.1 Offloading）：GPU 直接讀 pinned host（zero-copy）在遊戲卡（5060Ti、4090）上 PCIe 利用率很差，在 L40S 上很好；雙緩衝則相反。
- **數字**（Table 5，14B 模型，tokens/s）：4×4090 FP8：NCCL 4.3k → 全用 memcpy 7.8k（1.81×）；BF16 2.9k → 5.2k（1.79×）。4×L40S（有 P2P）：差異很小（FP8 9.5k → 9.9k）。（v1 版本，標明 work in progress，正式版數字可能不同。）
- 另一個坑（§3.2 "Multi-threaded multi-GPU and deadlocks"）：多 thread + NCCL collective 會死鎖，靠 CPU 端 barrier 解。

---

## 3. NCCL SHM transport

### 3.1 資料怎麼走（最新 v2.32.3，`src/transport/shm.cc`）
- 選用時機：transport 順序 `p2p, shm, net, collNet`（`src/transport.cc:15-20`），P2P 不行才落到 SHM；`shmCanConnect` 只要求同 host、同 `/dev/shm`（shm.cc:61-84）。
- FIFO 放哪：`NCCL_SHM_LOCALITY` 預設 2＝**接收端**（shm.cc:56 `NCCL_PARAM(ShmLocality, "SHM_LOCALITY", SHM_RECV_SIDE); // 1 is sender-size, 2 is receiver-size`）。接收端在 setup 時把各 protocol 的 buffer 加進自己要配的大小（shm.cc:134-136），由**自己 rank 的 proxy thread** 配置（shm.cc:141-143 → `shmRecvProxySetup` L269 → `ncclShmAllocateShareableBuffer`）。cuMem 路徑用 `ncclCuMemHostAlloc`，位置是 `CU_MEM_LOCATION_TYPE_HOST_NUMA`、id＝**呼叫者 GPU 的 HOST_NUMA_ID**（`src/include/alloc.h:72-83`）→ FIFO 在接收端 GPU 的 NUMA。
  - 我們的 pixi 環境是 NCCL 2.21.5：接收端用 `ncclShmOpen` 建 `/dev/shm` 段（v2.21.5 shm.cc:110-114）；實體頁落在哪個 NUMA 取決於 first-touch（推論，未確認）。
- 誰搬：預設（不開 CE 模式）**GPU kernel 的 SM 用 load/store 直接讀寫映射進來的主機 FIFO**；送端寫 `conn.buffs`（接收端的主機 buffer）並更新 `tail`，收端讀並更新 `head`（shm.cc:163-170、188-195）。v2.32.3 SHM 不註冊任何 proxy progress（transport 表 shm.cc:467-472，最後一欄 proxyProgress 為 NULL）。Demystifying NCCL §IV-A 也這樣描述："In SHM mode, one GPU's controlling process writes data to a shared memory segment, which is then read by the other GPU's process."
- `NCCL_SHM_USE_CUDA_MEMCPY=1`（**v2.21.5 ~ v2.30.4 有；v2.30.7 起已移除**——我逐版比對 shm.cc：v2.30.4 仍有、v2.30.7 / v2.31.2 / v2.32.3 沒有）。v2.21.5 行號：
  - 參數：shm.cc:41-42 `SHM_USE_CUDA_MEMCPY`、`SHM_MEMCPY_MODE` 預設 `SHM_SEND_SIDE`（"1 is sender-side, 2 is receiver-side, 3 is both"）。
  - 只有 SIMPLE protocol 走 CE（shm.cc:317、376 "Only Simple uses cudaMemcpy"）。
  - kernel 仍在 SM 上跑，先把資料搬到 GPU 上的 `devFifo`（`ncclCudaCalloc(&proxyInfo->devFifo…)` shm.cc:236、254）。
  - 送端 proxy CPU thread：輪詢 GPU 寫的 tail（"Check GPU has sent everything"，shm.cc:326）→ `cudaMemcpyAsync(shmFifo ← devFifo, DeviceToHost)`（shm.cc:329）→ 記 event、輪詢 event → 寫 SHM 裡的 tail（"Notify SHM"，shm.cc:342）。
  - 收端 proxy CPU thread：輪詢 SHM tail（"Check data is ready in SHM"，shm.cc:385）→ `cudaMemcpyAsync(devFifo ← shmFifo, HostToDevice)`（shm.cc:388）→ event 完成後寫 GPU 看的 tail（"Notify GPU"，shm.cc:399）。
  - 也就是說 CE 模式下 proxy thread 是「CPU 輪詢 + 發 cudaMemcpyAsync + 輪詢 event」的中介，仍是逐跳 ring。

### 3.2 Ring all-gather 是否每跳都經主機轉手？是。
`src/device/all_gather.h`（v2.32.3）`runRing`：
- L51-60 "step 0: push data to next GPU"（`directSend` / `directCopySend`）
- L62-67 "k-2 steps: copy to next GPU"：`prims.directRecvCopyDirectSend(offset, offset, nelem)`——從上一卡的 FIFO 讀、寫進輸出、再寫進下一卡的 FIFO。
- L73-74 最後 `directRecv`。
SHM 下 FIFO 都在主機，所以中間 N−2 步每卡**同一步內既讀主機（H2D 方向）又寫主機（D2H 方向）**，且 NCCL_STEPS 個 slot 管線化，兩方向同時在飛。每卡總量上傳 (N−1)S、下載 (N−1)S（與 ThunderEP §4.2 的 2(N−1)S 一致）。

### 3.3 NCCL 有沒有把主機記憶體當廣播媒介的演算法？沒有。
- 演算法清單：`src/init.cc:57-58` `{"Tree","Ring","CollNetDirect","CollNetChain","NVLS","NVLSTree","PAT"}`；AllGather 可用 `RING, COLLNET_DIRECT, NVLS, PAT`（`src/device/generate.py:105`）。NVLS 是 NVSwitch multicast，不是主機。
- 最接近的是 **CE collectives**（`src/ce_coll.cc`，AllGather/AlltoAll/Scatter/Gather，CUDA ≥12.5）：資料用 `cudaMemcpyBatchAsync`（`NCCL_CE_CHUNK_SIZE` 預設 8 MiB，L23），同步用 seq number——每 rank 用 `cudaMemcpyAsync` 把自己的 ready/complete 4-byte 值寫到對端（L282-292），本地用 `cuStreamBatchMemOp` 的 `CU_STREAM_MEM_OP_WAIT_VALUE_32` / `WAIT_VALUE_EQ` 等其他每個 rank（L294-302）；graph capture 時改等常數 `GRAPH_SYNC_VALUE=1`。**這和我們計畫的「CE 搬 + stream wait value 等計數器」是同一套機制**，但它明確排除我們的情況（L195-205）："symmetric support is not enabled"、"host-backed cuMem segments are not supported"、"not all ranks have NVLink connectivity" 都會跳過。

---

## 4. 其他「寫一次、多人讀」的共享記憶體集合通訊

**(a) MPICH CH4 POSIX release_gather（Jain et al., Intel, SC18 "Framework for Scalable Intra-Node Collective Operations using Shared Memory"）。** 共享記憶體裡有一個多 cell 的 bcast buffer（cell 輪用，`release_state % bcast_num_cells`）。`src/mpid/ch4/shm/posix/release_gather/release_gather.h:180-183`："Release step … top-down step in the release_tree. Parent notifies the children to go, once it arrives. In case of Bcast, root places the data in shm bcast buffer before notifying the children. Children copy the data out of shm buffer when notified by the parent"。旗標是單調遞增的 64-bit `release_state`，用 release-store / acquire-load（同檔 L20-26、L284、L311）；反向的 gather step（L356-358，tree 由下往上）讓 root 知道大家讀完、cell 可以重用。→ 「寫一次 + 單調計數器 + 讀完回報才能覆寫」三件事都在這裡。論文全文沒讀到（只核對了程式碼與 SC18 摘要頁）。

**(b) Open MPI coll/xhc（XPMEM Hierarchical Collectives，FORTH）。** 拓撲感知的階層式節點內集合通訊。bcast 的 leader 有兩種模式：XPMEM 單次拷貝（讀者直接讀 leader 的使用者 buffer），或 CICO（leader 先拷進自己的共享 CICO buffer，讀者從那裡拷出）。讀者輪詢一個**位元組計數器** `data_ready`，可用量 ≥ 一個 `chunk_size` 就開始拷，邊到邊拷（`ompi/mca/coll/xhc/coll_xhc_bcast.c:292-345`："Check if data's ready to be copied (without blocking)"）；拷完用 seq 號回 ack，leader 收齊 ack 才能重用 CICO buffer（`xhc_bcast_ack`，L362-400；L451-454 "Safe to alter the CICO buffer without checking any flags … the rank has gathered acks that these copies have completed"）。這和我們「每塊一個計數器、讀者等計數器再拷」幾乎同構，只是讀者是 CPU。main 分支的 xhc 只有 bcast / allreduce / barrier / reduce，沒有獨立的 allgather 檔案。

**(c) GPU 無 P2P：HFReduce（Fire-Flyer AI-HPC，DeepSeek，SC24，arXiv 2408.14158）。** 8× A100 **PCIe** + 2× EPYC Rome/Milan，GPU 直連 CPU、無交換器，其中 GPU5/6 共一個 root port（Fig. 4）。allreduce：各 GPU D2H 一次（大訊息用 cudaMemcpyAsync、小訊息用 GDRCopy）→ CPU 用 SIMD 加總 → 跨節點 → H2D 一次（§IV-A）。§IV-B1："for each unit of data, HFReduce only requires one D2H and one H2D data transfer"，相對 NCCL ring 的 (2n−1)/n。**H2D 用 GDRCopy 由 CPU 推（寫進 GPU BAR）**：§IV-A "utilizing GDRCopy to write data to four GPUs within the same NUMA node, effectively reducing reads from host memory by threefold compared to MemCpyAsync … GDRCopy can read data from host memory and temporarily cache it in the CPU caches"——這是「讀一次、CPU 推給四卡」的廣播變體，讓 H2D 變成 CPU 端的 posted write，不靠 GPU copy engine 去讀主機。NUMA：§IV-D1 "D2H destination memory is interleaved across two NUMA nodes for maximum bandwidth"。瓶頸（§IV-D3）：理論 ~12 GB/s，實測略高於 8 GB/s；"In AMD EPYC Rome and Milan CPUs, the maximum bandwidth from the Root Complex Port to the CPU's internal bus is about 37.5GB/s … when two GPUs transfer data concurrently, the bandwidth is limited to around 37GB/s. Furthermore, if bidirectional data transfer occurs simultaneously, this bandwidth decreases even further."。對 NCCL：6.3–8.1 vs 1.6–4.8 GB/s（Fig. 7a，跨節點 186 MiB）。

**(d) 次要例子。**
- LLMQ 的主機權重快取（見 §2）。
- Open MPI coll/accelerator（`ompi/mca/coll/accelerator/coll_accelerator_allgather.c`）：訊息小於門檻時，把 device sendbuf D2H 到私有主機 buffer，跑主機端 allgather，再整份 H2D 回去（"allgather for device buffers through temp CPU buffer"）。每 rank 只 D2H 一次，但中間的 gather 是 CPU 拷貝，GPU 不直接讀共享 buffer。
- MVAPICH2(-GDR)：ESPM2'15（Queen's 大學）的二手描述說它多數 GPU 集合操作先把資料 stage 到主機 buffer，或用 CUDA IPC 收到 GPU 共享 buffer。原始設計論文沒找到（未確認）。
- Open MPI coll/sm（舊的共享記憶體 fan-out bcast）、coll/smdirect：沒有核對原始碼（未確認）。
- TCCL（Kim, Ryu, Lee，ASPLOS'24，與 ThunderEP 同一組）：PCIe GPU 叢集的路徑搜尋，處理經主機 / NUMA 的 bounce-buffer 擁塞。全文沒讀到（未確認細節）。

---

## 5. 同卡雙向時 H2D 崩落：有沒有人報告過？怎麼解釋？

**與我們同型平台的第三方報告（非同儕審查）：**
- **NVIDIA/nvbandwidth issue #53**（2025-11-21 開，2026-05-18 仍 open）＋同內容 NVIDIA 論壇 thread 352186。平台：**8× A100-SXM4-80GB，雙路 Intel Xeon Platinum 8358**（作者留言），每兩張 GPU 共一個 PCIe 交換器，GPU0-3 → NUMA0（CPU 0-31,64-95）——**和我們這台是同一種平台**（同 CPU 型號、同 NUMA 對應、同樣兩卡一個交換器）。nvbandwidth v0.8、driver 560.35.05、CUDA 12.6。
  - CE 單向：H2D ~24.8、D2H ~25.9 GB/s/卡。
  - CE 雙向：D2H ~24.2，**H2D ~11.3–12.0**（SUM 94.74）。
  - 開啟 relaxed ordering 後 H2D 11.66–11.76，沒改善；MRRS 4096。
  - **SM 雙向：D2H 21.47–21.58、H2D 18.06–21.55（多數 ~21.5）**——SM copy 幾乎沒有不對稱。
  - 回覆：Robert_Crovella 表示 "There might be some drop … but I wouldn't expect it to be ~50%"；貢獻者 deepakcu 建議開 RO，並說 "It is hard to tell whether asymmetry is expected for the hardware configuration."。**無根因、無修正。**
  - 我們的數字（plan §9 引 params.json）：只下載 22.01 → 同時上傳時 6.46 GB/s（nvbandwidth 獨立量 8.88）。比 #53 的 11.3 還低；差在 driver（615 vs 560）、nvbandwidth 版本、BIOS 或量法，未知。
- **nvbandwidth 官方 README** 的範例輸出（平台未標明）：`host_to_device_memcpy_ce` ~26.0，`host_to_device_bidirectional_memcpy_ce` 18.37–19.59 GB/s，也就是官方範例本身就掉約 25–30%。README 說明 CE 雙向測法："Stream 0 (measured stream) performs writes to the device, while the interfering stream in the opposite direction produces reads"；SM 雙向是「奇偶 warp 各搬一個方向」，頻寬用整個 kernel 時間算（所以 SM 數字無法分開兩個方向，解讀時要注意）。

**論文：**
- Fire-Flyer / HFReduce（SC24，§IV-D3，A100 PCIe + EPYC Rome）：定性說 "if bidirectional data transfer occurs simultaneously, this bandwidth decreases even further"——但那是**兩卡共用 root port** 的情境，沒有拆出同卡 H2D/D2H、也沒數字。
- ThunderEP（§4.4）：定性說讀比寫更早飽和（"Writes do not wait for a response, whereas reads do and saturate earlier"），但在它的平台（RTX 4090/5090 + EPYC Genoa）**沒有回報同卡雙向崩落**，反而靠同時上下傳得到好處。
- NSDI'24 rPCIeBench（Hou et al.，FPGA 端點，§4.6）：**反例**——"routable PCIe incurs no communication interference among concurrent reverse PCIe flows over one physical path"（量的是反向流量下延遲敏感流的延遲：1.6→1.8 µs）。同篇 §4.5："the bandwidth partition among concurrent PCIe flows mainly depends on the ratio of their outstanding bytes"。
- **我沒找到任何同儕審查論文量化「A100 同卡 H2D 被 D2H 壓到 1/2~1/3」。**（van Werkhoven et al. CCGrid'14 有建 CPU-GPU 傳輸的效能模型，是否涵蓋雙向重疊——未確認。）

**可能的解釋（全部是推論，沒有人證實）：**
1. CE 的 H2D 是 GPU 發出的**讀請求（non-posted）**，資料以 completion 回來；D2H 是 GPU 發出的 **posted write**。兩者的請求 TLP 都在同一張卡的上行方向。PCIe 排序規則裡「讀請求不得超越先前的 posted write」（廠商文件的 ordering 表，例：Efinix PCIe controller UG；PCIe 規格原文我沒核對）→ 讀請求排在大量 D2H 寫後面，讀延遲上升；讀頻寬 ≈ 在途位元組 / 延遲（與 rPCIeBench §4.5 的「按在途位元組分配」一致），CE 在途讀請求數有限，所以 H2D 掉。RO 只放寬寫/completion 的排序、不讓讀請求越過寫，這和 #53「開 RO 沒用」相符。
2. #53 作者的假說：root complex 仲裁偏向上行 posted write。
3. SM copy 在 #53 幾乎對稱，代表瓶頸可能在 **CE 的讀取管線**（例如在途請求上限）而不是鏈路本身（推論）——SM 的 load 也是 non-posted 讀，但可以有大量在途請求。
- 如何驗證：本機跑 `nvbandwidth -t host_to_device_bidirectional_memcpy_sm` 與 `_ce` 對照（幾秒鐘）；再試「D2H 用 SM store、H2D 用 CE」與「H2D 用 GDRCopy CPU 推、D2H 用 CE」兩種混搭，看 H2D 是否回到 ~20 GB/s。全部包在 `exclusive_guard.py` 內。

---

## 對我們的設計的啟示

- **「上傳一次、大家下載」＋ CE ＋ 主機旗標 ＋ 收到就算，這組合本身不是新的。** ThunderEP §4.2–4.5 已經在無 P2P 的 PCIe 機器上，對 dense AG/RS 做了一模一樣的 single-step 廣播（每卡寫一次、其他卡直接讀）、DMA 搬運、per-sender 主機旗標、到一份算一份；NCCL 的 CE collectives 用的也是同一套「seq 號 + `CU_STREAM_MEM_OP_WAIT_VALUE_32`」同步（只限 NVLink）；CPU 端的 MPICH release_gather、Open MPI XHC 早就是「單一共享 buffer + 單調計數器 + 讀完 ack 才覆寫」。plan §9 說 ThunderEP 做 all-to-all 要更正：它做的就是 AG/RS。可能的新意要放在：(1) TP dense GEMM 的 **tile 級**旗標融合（ThunderEP 的粒度是「每個來源 GPU 的整份 shard」，用 host 端 CUDA event 串）；(2) 不需主機 thread 的 stream memory op 等待；(3) **繞開 A100 HGX 同卡雙向崩落的排程**——前人沒有人這樣做。
- **不要照抄 ThunderEP 的「上下傳兩條 stream 同時跑」。** 它的 AG 頻寬優勢有一部分來自讓 D2H 與 H2D 重疊（§5.2），那是在 Genoa + RTX 平台、沒有崩落的前提下；在我們這種平台（#53 是同 CPU、同拓撲的獨立第三方報告）會正好落進 H2D 掉到 1/2–1/3 的區域。AG 照 plan「先傳自己那份、再下載」，並用 nsys 確認同一張卡的 D2H 與 H2D 沒有重疊。RS 與跨層重疊天生是雙向，這部分先量兩個便宜的替代：SM 驅動的搬運（#53 中 SM 雙向約 21.5/21.5 GB/s），以及 GDRCopy 由 CPU 推 H2D（HFReduce 的做法；本機 `/dev/gdrdrv` 可讀寫、`libgdrapi.so.2.5` 已裝）。兩者都是推論，要先量；SM 搬運會和 GEMM 搶 SM，LLMQ 也報告遊戲卡上 zero-copy 讀主機效率很差。
- **可以照抄的工程細節：** 旗標與資料分開、單一寫者、不用 atomic（ThunderEP §4.4）；主機緩衝區雙緩衝，讀完的 credit 用 **PCIe 寫**回報、不用讀（§4.4）；XHC 用位元組計數器邊到邊拷，MPICH 用單調 `release_state` 輪用多個 cell，都避開「每輪重設旗標」。CUDA graph：ThunderEP 因為 capture 會把 `cudaMemcpyAsync` 的位址寫死、無法輪換 buffer，decode 只好改用 SM 搬（§4.5）；NCCL CE 在 capture 時改等常數 `GRAPH_SYNC_VALUE`。我們若要 graph 化，旗標值與 buffer 位址都要設計成固定的。
- **RS 的 CPU 加總（plan 選項 b）前人都不看好。** ThunderEP 實作後否決（§4.5：主機 DRAM 頻寬被 DMA 吃掉）；HFReduce 實測只到理論值的 2/3，原因是 root port 共用與雙向（§IV-D3）。這和我們 M0 量到的互搶一致，可以把 (b) 的優先序往後排。
- **NUMA 放置沒有現成答案。** NCCL 預設把 FIFO 放在**接收端** NUMA（`SHM_LOCALITY=2`，`alloc.h:72-83`）；HFReduce 把 D2H 目的地在兩個 NUMA 之間 interleave。「寫一次、八卡讀」下一定有一半讀者跨插槽，plan 裡「每個插槽各放一份」是合理的待量選項，前人沒有直接可抄的數字。
- **預期增益的錨點：** ThunderEP 在 N=6、1 GB 時 AG 比 NCCL 快 1.27–1.53×，小訊息 2.3–2.65×；RS 大訊息打平。N=8 的流量比是 1.75×（公式推論）。我們的 NCCL 基準（busbw 7.5）比它們的（20.8–23.0）低得多，所以這些倍數不能直接搬過來，plan 預估的 1.4×（10.7 vs 7.6，兩個不同實驗）仍要在 V2 以同一路徑實測。
