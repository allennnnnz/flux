Supersedes: （無）

# 調查：PCIe 通道上的 Flux 式重疊，以及 NVLink 域 + PCIe 異質晶片的協同

- 日期：2026-10-06。性質：文獻與開源調查（不含量測）。尚未經 auditor 審查。
- 起因：教授指派新方向——讓 PCIe 也能用 Flux 的邏輯做通訊 / 計算重疊，能配合 NVLink 相連的運算環境；
  節點外多一顆只有 PCIe 的不同廠牌運算晶片，也要能協同運作並用 PCIe 溝通。
- 方法：三路平行調查（repo 內程式碼、論文、開源專案），由子代理擷取原文核對；本報告撰寫者另抽查了三項關鍵說法。
  標記：**[已抽查]** = 本報告撰寫者親自擷取原文確認；未標 = 調查代理依擷取的摘要 / README 核對；**[未驗證]** = 無一手來源。

## 0. 結論（白話）

1. **Flux 的重疊機制本身跟傳輸方式無關。** GEMM 的每個 tile 只是在 GPU 記憶體裡等一個旗標變成 1
   （`include/flux/cuda/system_barrier.hpp:38-86`，`src/ag_gemm/sm80_all_gather_gemm.hpp:922-940`）。
   任何能在資料到齊後寫那個旗標的東西——copy engine、主機執行緒、別的 DMA 引擎——都能驅動重疊。這是接上 PCIe 與異質晶片的切入點。
2. **「NVIDIA GPU 之間走 PCIe + Flux 重疊」已經存在，不是新東西。** Flux 原生支援 PCIe P2P（Ring1D / Ring2D、PCIe 版 GEMM+RS），
   論文在 A100 PCIe 叢集上 op 層級 1.20–3.25×；本 repo 的 `docs/performance.md` 記錄 L20（PCIe）AG+GEMM 5.9 ms vs torch+NCCL 16.9 ms。
3. **把 PCIe 當成 NVLink 之外的「額外頻寬」，上限很低。** 最接近的 FlexLink 在 8×H800 上把 NVLink、PCIe、網卡一起用，
   AllReduce 只快 26%、AllGather 27%，而且只有集合通訊、沒有跟 GEMM 重疊 **[已抽查]**。這與 Phase 0 的結論一致
   （PCIe 與 NVLink 頻寬差 41 倍、加不起來，D-001 決定不做雙通道）。
4. **真正沒人做過的是：PCIe 掛的不同廠牌晶片，以 tile 級重疊加入 NVLink 域的計算。**
   - 所有跨廠牌系統（HetCCL、FlagCX、H2、HetHub、Mooncake）都由主機協調、經網卡或主機記憶體中轉，沒有 GEMM 融合。
   - 沒有論文量過「同一台機器裡 NVIDIA GPU ↔ 其他廠牌晶片直接走 PCIe」。
   - 沒有開源專案讓兩個廠牌共用 tile 旗標。
   → 這就是本方向的研究貢獻所在。
5. **現成可借用的設計都有**：用 copy engine 搬資料（ConCCL）、經主機中轉 + 主機記憶體旗標（ThunderEP、LLMQ）、
   tile 計數觸發任意通訊（FlashOverlap）、核心觸發 + 主機代理執行緒搬資料（MSCCL++ PortChannel）、跨廠牌傳輸（FlagCX）。

## 1. repo 內：Flux 已有什麼、缺什麼

| 已有 | 位置 |
| --- | --- |
| AllGather 模式：All2All（NVLink）、Ring1D（PCIe、單 NUMA）、Ring2D（PCIe、跨 NUMA，硬寫 2 個 NUMA × 4 卡） | `src/coll/ths_op/all_gather_types.h:23-65`；`all_gather_op.cc:376-707` |
| 拓撲偵測只看 NUMA 與 NVLink（不偵測 PCIe switch、不量頻寬） | `src/ths_op/topo_utils.cc:140-235` |
| PCIe 版 GEMM+RS：epilogue 寫本地、per-tile 旗標、獨立高優先權 rs_stream、不用原子操作 | `src/gemm_rs/ths_op/gemm_reduce_scatter.cc:225-316`、`epilogue_evt.hpp:265-417` |
| 不需 P2P 原子操作的 barrier；零拷貝主機緩衝（`FLUX_RS_USE_SHM`，SM 直接寫主機記憶體） | `src/cuda/cudaipc_barrier_all.cu:66-93`；`flux_shm.cc:560-577` |
| 跨節點：`AGKernelInterNode` 跨節點用 NCCL send/recv，再以 copy + `CUStreamWriteValue` 旗標在節點內轉送 | `all_gather_gemm_op_internode.cc:452-762` |

| 缺少 | 影響 |
| --- | --- |
| 沒有「D2H + H2D 中轉」的傳輸方式；緩衝區配置假設一定有 IPC / NVSHMEM 對端指標；沒有檢查 P2P 能力 | 不能 P2P 的裝置完全不支援 |
| 全部是 CUDA / CUTLASS；只接受 sm80/89/90 與特定 SM 數（`src/cuda/op_registry.cu:39-51`） | 不同廠牌晶片無法直接加入 |
| 沒有後端介面（註冊緩衝、送 tile、設 / 等旗標、回報能力）、沒有主機代理寫旗標的路徑 | hetero-proxy 規劃的能力等級 0 / 1 都還不存在 |
| PCIe 環上的跨節點重疊被 `nccl_event` 串行化（`:343-346`） | PCIe 跨邊界時重疊打折 |

Phase 0 的 PCIe 量測與設計規則（`docs/PHASE0_FINDINGS.md` 第 2.3–2.5、3 節）仍適用：單卡單向 21.8 / 23.9 GB/s；
同一張卡同時收送時 H2D 掉到 6.46 GB/s；中轉最佳 22.2 GB/s（同 NUMA、不同 switch）；chunk ≥ 16 MiB；
GPU 兩兩共用 switch 上行（0/1、2/3、4/5、6/7）。

## 2. 論文

### A. PCIe-only 伺服器上的重疊

| 工作 | 機制 | 硬體 / 效益 | 程式碼 |
| --- | --- | --- | --- |
| **Flux**，arXiv 2406.06858 | GEMM tile 化 + 融合 AG/RS、per-tile 等待；PCIe 上用 cudaMemcpy pull/push、環狀順序 | A100 PCIe：op 1.20–3.25×、prefill 最多 2.06× | github.com/bytedance/flux |
| **FlashOverlap**，EuroSys'26，2504.19519 | GEMM 累加 tile 完成計數，湊滿一組就觸發一般 NCCL 呼叫；不綁通訊函式庫 | RTX 4090（PCIe、跨 NUMA）最多 1.65×；也在 A800、**昇騰 910B** 上跑過 | github.com/infinigence/FlashOverlap |
| **ThunderEP**，2609.40093 | P2P 被關閉的消費級 GPU：經主機 bounce buffer、cudaMemcpyAsync（DMA 引擎）、旗標放主機 pinned 記憶體、雙 stream 用滿 PCIe 兩個方向 | RTX 4090/5090：dispatch 2.00×、combine 1.53× | 未找到 |
| **LLMQ**，2512.15306 | copy engine + 主機中轉的集合通訊，與下一層 backward 重疊 | 4×RTX 4090：7.8k vs NCCL 4.3k tok/s | github.com/IST-DASLab/llmq |
| **ConCCL**，2412.14335 **[已抽查]**；DMA-Latte，MICRO'26，2511.06605 | 計算與通訊同時跑時互相干擾；通訊改走 DMA 引擎 | 用 SM 傳只有理想加速的 21%，DMA 引擎 72%（最多 1.67×）；AMD MI300X | — |
| ISO，2409.11155；RoundPipe，2604.27085 | 序列內重疊；消費級 GPU 的 pipeline 排程 | RTX 4090 | RoundPipe：github.com/ITcarrot/RoundPipe |

NVLink 上的重疊工作（TileLink、Triton-distributed、Comet、TokenWeave、Domino、NanoFlow、T3、Centauri、Lancet）只作背景。

### B. 節點內 NVLink + PCIe 一起用

| 工作 | 機制 | 效益 / 限制 |
| --- | --- | --- |
| **FlexLink**，2510.15882 **[已抽查]** | NCCL 替代品，NVLink + PCIe + RDMA 網卡同時傳；PCIe 路徑為主機中轉雙緩衝、4 MB chunk、`cuStreamWait/WriteValue32` 同步 | 8×H800：AllReduce +26%、AllGather +27%，2–22% 流量移出 NVLink；**不與 GEMM 重疊** |
| Blink，MLSys'20 | spanning tree 同時走 PCIe 與 NVLink | 舊 DGX 上最多 8× |
| 多路徑節點內傳輸，2604.22228；MMA，2512.16056 | 點對點訊息拆到 NVLink / PCIe / 主機；主機到 GPU 借用其他 GPU 的 PCIe 再經 NVLink 轉送 | 頻寬最多 2.95× / 4.62×；不與計算重疊 |

### C. 跨廠牌 / 異質加速器協同

| 工作 | 裝置之間怎麼通訊 | 重疊 | 硬體 |
| --- | --- | --- | --- |
| **HetCCL（BAAI 等）**，2605.31000 | 主機代理驅動 RDMA：device→RDMA buffer→網卡→目的端；複用各廠牌 CCL 的 reduction | 拷貝與 RDMA pipeline 重疊（非 GEMM 融合） | A800 + 3 家匿名廠牌（GPGPU 與 ASIC） |
| **FlagCX**（BAAI，開源） | 12 家廠牌 CCL 後端，Homo / Hetero 模式，device-buffer IPC / RDMA **[已抽查]** | 主機協調 | NVIDIA、AMD、多家國產晶片 |
| H2（上海 AI Lab），2505.17548 | 裝置直連 RDMA（DiComm）；每個 pipeline stage 只用一種晶片 | P2P 與 backward 分段重疊 | 4 種匿名晶片、千卡以上 |
| HetCCL（Moreh / SNU），2601.22585 | NVIDIA + AMD，GPUDirect / DirectGMA RDMA；節點內依拓撲 P2P 或共享記憶體中轉 | — | 最多 1.48× |
| AMD + NVIDIA 聯合訓練，2602.18007；HetHub，2405.16256 | 只在 pipeline 邊界跨廠牌、經 RDMA 網卡；統一通訊器 + 平行規劃 | — | H200 + MI325X；768 卡 |
| MegaScale-Infer，2504.02263；HeterMoE，2504.03871 | attention 與 FFN / expert 放在不同類型裝置 | ping-pong / zebra pipeline 藏通訊 | 不同 GPU 型號 |
| HexGen / HexGen-2 / Helix / Metis / Cephalo / Poplar | 異質 GPU 的切分規劃器 | 只有規劃，沒有新傳輸 | 多半只有 NVIDIA |

### D. 把部分計算卸載到較弱 / 不同的裝置

Lamina（2405.01814，attention 放 H20、其餘放 H100，RoCE，交錯 pipeline）、Adrenaline（2503.20552）、NEO（MLSys'25，CPU attention）、
KTransformers（SOSP'25）、Mirror-SD（2510.13161，draft 模型在 NPU、目標模型在 GPU 平行跑，2.8–5.8×）。
**警告**：2602.09721 指出跨裝置頻寬有限時，attention / FFN 拆分會落入「死區」——正是 PCIe 掛單顆晶片的情況。

### E. 跨廠牌 PCIe P2P 的系統條件

- Linux P2PDMA：兩個裝置要在同一個 root port 下或白名單 host bridge 後；預設擋跨階層路由；ACS 影響路由。
- NVIDIA GPUDirect RDMA 文件：第三方 PCIe 裝置可經 `nvidia_p2p_get_pages` 或 `cuMemGetHandleForAddressRange(DMA_BUF_FD)` 取得 GPU 記憶體；
  要同一個 root complex、IOMMU 非 1:1 轉換時要關；跨 CPU socket 的路徑「嚴重受限或不可靠」。
- NVIDIA open-gpu-kernel-modules discussion #1046（2026）：x86 上 dma-buf P2P 給任意第三方裝置仍被驅動擋住，有人移除檢查才讓 FPGA 能用。
- **沒有任何論文量過同機 NVIDIA GPU ↔ 其他廠牌加速器的 PCIe P2P。** 所有跨廠牌系統都用 RDMA 網卡或主機中轉。

## 3. 開源專案

| 名稱 | 授權 | 最近活動 | 能用在哪 | 跨廠牌 |
| --- | --- | --- | --- | --- |
| **Flux**（本 fork） | Apache-2.0 | 2025-08 | NVIDIA 側的融合 kernel 與旗標協定（已有 PCIe 路徑） | 否 |
| **FlagCX** | Apache-2.0 | 2026-10 | 跨廠牌傳輸：12 家 CCL、Hetero 模式、Torch plugin、Device / CCL / Net adaptor plugin | **是** |
| **Triton-distributed** | MIT | 2026-09 | Flux 式 AG+GEMM / GEMM+RS 的 Triton 版，已移植到沐曦（MetaX）、昇騰分支 | 多後端，但未在同一作業混用 |
| **MSCCL++** | MIT | 2026-10 | PortChannel + 主機 ProxyService：kernel 觸發、CPU 執行緒搬資料、再回寫旗標——可直接對應「GPU tile → PCIe → 外部晶片」 | 代理可接任意 runtime |
| **FlashOverlap** | Apache-2.0 | 2026-01 | GEMM 發 tile 訊號 + 計數 kernel 擋住任意主機發起的傳輸 | 通訊端可替換 |
| Mooncake TE / NIXL / UCX | Apache-2.0 / Apache-2.0 / BSD-3 | 2026 | 多廠牌資料搬運（Mooncake 已有 MACA、MUSA、昇騰、HIP、FlagCX transport） | 部分 |
| NVSHMEM / NCCL | Apache-2.0 | 2026 | 只限 NVIDIA；NCCL NET plugin 只能搬 NVIDIA 端緩衝，外部晶片不能當 NCCL rank；NVSHMEM 在純 PCIe P2P 上原子操作需 IB 或 UCX | 否 |
| PyTorch SymmetricMemory / async-TP | BSD | 2026 | 要求所有裝置兩兩 NVLink、只限節點內 | 否 |

**沒有任何開源專案提供**：跨廠牌的融合重疊（共用 tile 旗標）、跨廠牌經 PCIe 的訊號順序與原子操作定義、使用者空間可用的 GPU ↔ 外部加速器 PCIe P2P、
把同一個 TP GEMM 不對稱地切給 NVLink 域與 PCIe 晶片並重疊的排程器。

## 4. 研究定位與建議路線

**定位**：在 NVLink TP 域與一顆 PCIe 連接、不同廠牌的晶片之間，做 **Flux 式 tile 級重疊**——遠端分片經 PCIe（先主機中轉、日後 P2P）送到，
每個 chunk 到齊就寫旗標、GEMM tile 立刻開算；並用 hetero-proxy 的能力分級，讓同一套排程同時適用 GPU 型（動態排程）與 NPU 型（靜態 pipeline）裝置。
第 2、3 節顯示這個交集目前沒有論文或開源專案覆蓋。

**建議路線**（以 A100 代理晶片；與 `ws/hetero-proxy/STATUS.md` 第 1 節的能力分級一致）：

| 步驟 | 內容 | 借用 |
| --- | --- | --- |
| 1 | 補 Phase 0 兩項量測：PCIe 競爭位置（1a）、中轉機制固定開銷（1b） | 原計畫 |
| 2 | **技術探針**：能否從 Flux 外部（Python / 主機執行緒 / H2D stream 的 `cuStreamWriteValue32`）寫入 AG-GEMM kernel 等待的旗標？這決定等級 1 要改多少 Flux | Flux `all_gather_op.cc:513-530`、`sm80_all_gather_gemm.hpp:922-940` |
| 3 | **等級 0**：一個 NVLink TP 域 + 一張關閉 peer access 的代理 A100，算子級 D2H + H2D，AG+GEMM 結果正確，NVLink 計數器確認代理流量為零，交錯量測 | Phase 0 規則 7 |
| 4 | **等級 1**：chunk 級中轉（copy engine、雙緩衝、≥16 MiB），每個 chunk 落地後寫旗標，A100 端的 GEMM tile 等旗標開算；反方向（A100 產出 → 晶片）用 FlashOverlap 式的 tile 計數觸發傳輸 | ThunderEP / LLMQ / FlexLink 的中轉設計；FlashOverlap；ConCCL（用 DMA 不用 SM） |
| 5 | **NPU 型代理**：代理端改成少量 persistent CTA + 嚴格順序 + 顯式雙緩衝，驗證靜態型裝置也能跟上 | hetero-proxy 規劃 |
| 6 | **工作切分**：單顆晶片放什麼（attention / KV、draft 模型、expert、不對稱 TP 分片），先用成本模型估，避開 2602.09721 的死區 | Lamina、Mirror-SD、MegaScale-Infer |
| 7 | **晶片到手後**：等級 2（dma-buf / P2PDMA 直連）的可行性；以 FlagCX adaptor 接廠牌 runtime | FlagCX、MSCCL++ ProxyService |

**不建議作為主線**：在 A100 節點內把 PCIe 當成 NVLink 的額外頻寬。上限約數個百分比（PCIe 約 22 GB/s vs NVLink 單對約 270 GB/s；
FlexLink 在 H800 上集合通訊只 +26%），且 A100 之間的 P2P 會走 NVLink，PCIe 通道只能主機中轉。Phase 0（D-001）已有同樣結論。

## 5. 待確認

- 教授的「PCIe 加入通訊通道」是指 (a) 連到外部晶片的通道（本報告建議的主線），還是 (b) 節點內 NVLink 之外的額外頻寬？若是 (b)，需先說明上限。
- 那顆國產晶片的型號 / 軟體堆疊（是否有 Triton 後端、廠牌 CCL、dma-buf 匯出 / 匯入能力）決定步驟 7 與工作切分。
- 未驗證項目：Hybe 的拓撲、FastDecode 與 Cephalo 的發表處、FlagCX 的頻寬宣稱、HetCCL（2605.31000）程式碼連結。

## 附錄 A：FlexLink 細節（2026-10-09 讀全文，arxiv.org/html/2510.15882 **[已抽查]**）

- **定位**：NCCL API 相容的直接替代品；目前只支援 AllReduce、AllGather（AllToAll 列為未來工作）。程式碼未公開（文中說約 500 行 Python + 3,500 行 C++/CUDA）。
- **PCIe 路徑 = 經主機中轉**：NVLink 相連的 GPU 之間做 P2P 會自動走 NVLink，所以要刻意用 PCIe 只能 GPU → pinned 主機記憶體 → GPU。
  - 雙緩衝 pipeline：拆成 producer D2H、H2D consumer 兩段，各一塊 pinned buffer，一個 chunk 上傳時另一個在下載；chunk 4 MB（PCIe 與 RDMA 都是）。
  - 用 copy engine（`cudaMemcpyAsync`），不用 SM。
  - 同步：`cuStreamWaitValue32` / `cuStreamWriteValue32`，GPU 直接輪詢記憶體旗標；**單調遞增計數器**——第 i 輪 producer 等 `semEmpty == i`、寫資料、把對方 `semFull` 設為 `i+1`，不必重設旗標，也不會讀到舊值。
- **網卡路徑**：節點內 RDMA loopback，以 NVSHMEM 的 CPU 發起 API 實作；作者自承「次佳，需要再優化」。
- **三路同時跑、資料切塊分配**：兩階段負載平衡。初始化約 10 秒反覆量測、調到三路同時完成；執行時看最近約 10 次呼叫，最慢與最快差距超過門檻就移一小塊給最快的。
  結果 PCIe 約 10–14%、網卡 4–10%、其餘 NVLink。環狀演算法如何按路徑拆分，文中未詳述。
- **測試平台與效益**：8×H800（NVLink 被限到 400 GB/s）、PCIe 5.0 x16、ConnectX-6。AllReduce 2 卡 256 MB +26%（PCIe 12% + RDMA 9%）；
  AllGather 4 卡 +27%、8 卡 +24%；**8 卡 AllReduce 只有 +2%**（環狀 2(N−1)=14 步，慢路徑的延遲被放大）。
- **限制（作者自述）**：協調用的 kernel 會佔 SM；網卡與主機流量共用同一條 PCIe 上行（H800 上合計被限在一個 PCIe 介面 128 GB/s）；PCIe 被其他工作佔用時效益下降；
  不跟計算重疊（只做頻寬加總）。
- **對本 ws 的意義**：
  - PCIe 路徑的設計（主機中轉雙緩衝 + copy engine + GPU 輪詢旗標 + 單調計數器）可直接作為等級 1 的傳輸層；Phase 0 的 staging 量測用的是同一類機制（`cuStreamWrite/WaitValue32` 雙緩衝，16 MiB 最佳）。
  - 它沒有 Flux 式的「chunk 到就開算」，那是本 ws 要補的部分。
  - 在本機 A100 上把 PCIe 當額外頻寬，效益會比 H800 更小 **[推論]**：H800 是 PCIe 5.0 配限速 NVLink；本機 PCIe 4.0 中轉最佳 22.2 GB/s、NVLink 單對約 270 GB/s，若三路同時完成，PCIe 最多分到約 8%。
    如何驗證：在本機實作 PCIe 中轉路徑與 NVLink 並行跑 AllGather，量合計頻寬。
