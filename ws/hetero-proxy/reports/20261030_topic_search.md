Supersedes: 無（補充 `20261029_lit_moe_hybrid.md` 的選題；MoE 題目改列為候選之一）

# 選題重找：這台機器上適合 10 週寫成論文的「通訊優化」題目（2026-10-30）

使用者要求：通訊優化，推論或訓練都可以；10 週內產出一篇架構完整的論文（不必頂會）；先看拓撲、可以做一點環境實驗。

---

## 1. 結論（先講）

**推薦題目：訓練時「主機↔GPU 搬運」把 NVLink 集合通訊拖慢 2–3 倍——量清楚、找出原因、做一個會避開的排程器。**

- 今天量到的（第 3 節）：8 卡之中只要**同一個 PCIe 交換器上的 2 張卡**把資料往主機記憶體搬（offload），
  8 卡一起做的 16 MiB reduce-scatter / all-gather 就慢 **2.8–3.0 倍**、32 MiB 慢約 2 倍、64 MiB 慢 1.4 倍。
  這個大小正是訓練時 FSDP / ZeRO 的分桶、TP＋序列平行的 activation 通訊會用到的。
- NCCL 不知道背景有搬運：沒搬運時它選的協定（Simple）最快，有搬運時改用 LL128 反而快 1.75–2.2 倍；128 MiB 以上又是 Simple 好。
  → 「什麼時候搬、搬多快、通訊用哪種協定」需要一起決定，這就是排程器的工作。
- 已有人碰到同一個機制但**沒有論文**：NVIDIA/nccl #2218（V100 / H200，背景 D2H 讓 all_gather 掉 8–11%，維護者只建議改用 LL/LL128）、
  PyTorch 論壇 2024（H100，非同步 D2H 讓 all-gather 從 2.7 ms 變 4.4–13 ms，未解決）。最接近的論文 ChunkFlow 是**純 PCIe 機器**上的推論。
- 為什麼適合：**只要一台機器**（不依賴 159）；能直接接到既有的工具（X 系列的 GPU 時間戳、hxb 計數器、Flux 式放行）；
  和長期目標一致——將來那顆只有 PCIe 的國產晶片，它的流量就是「一個交換器上的 PCIe 搬運」，T1b 顯示這已足以拖慢整個 8 卡集合通訊。

**備案：把 fusion-dispatch（決策器）寫成論文。** 實驗八成已做完，文獻上仍有空間（第 5 節），10 週最穩；缺點是單機上簡單規則幾乎一樣好，賣點要放在跨節點。

**現行的 MoE 冷門專家題目**降為第三：KTransformers 已有逐層多線搬運（10-30 補充），而且它的搬運方向是主機→GPU，今天量到這個方向的干擾小得多。

需要使用者決定的事列在第 8 節。

---

## 2. 拓撲（今天看到的）

| 項目 | 事實 | 來源 |
| --- | --- | --- |
| GPU 互連 | 8× A100-SXM4-80GB，NVSwitch，任兩卡 NV12 | `nvidia-smi topo -m` |
| PCIe | 4 個 Gen4 交換器，**每個交換器 = 2 張 GPU + 1 張雙埠 100 GbE 網卡**（GPU0/1 + mlx5_0/1、GPU2/3 + mlx5_2/3、GPU4/5 + mlx5_4/5、GPU6/7 + mlx5_6/7），每個交換器一條 Gen4 x16 上行到 CPU | sysfs 路徑、`results/xpcie/topology_20261028.txt` |
| CPU / 記憶體 | 2 插槽 Xeon 8358；GPU0–3 在 NUMA 0、GPU4–7 在 NUMA 1；1.5 TB | CLAUDE.md、10-29 |
| 跨節點 | 與 css-host-159 以 8 條 100 GbE RoCE 相連，條與條之間沒有路由；網卡從 GPU 讀只有約 7 GB/s／卡（寫進 GPU 是滿速 12.3） | `params.json` `flux_dispatch_e4_xnode` |
| 158 的 rail 4 | 已有 IP（10-03 時沒有） | `ip -br addr` |
| **159 現況（10-30 02:20 UTC）** | 8 張卡都被一個 SGLang 服務佔用（使用者 `rogerlee` 帳號，DP4×TP2，已跑 80 分鐘，每卡 74 GB）；**只剩 6 個網卡埠**（GPU4/5 那張網卡 mlx5 消失，rail 5/6 不通） | `ssh css-host-159 nvidia-smi / ls /sys/class/infiniband` |

對選題的意思：跨節點題目現在不能獨佔量測，而且少了一張網卡 → **跨節點題目的環境風險高**，主題目最好只需要單機。

---

## 3. 今天的環境實驗（全部在 exclusive guard 內，guard 全 CLEAN；未經 auditor）

完整數字與錨點對照：`results/topic/analysis.txt`。

### T1：背景搬運對「訓練大小」集合通訊的影響

- 做法：`scripts/topic/t1_bigcoll_pcie_v1.py`。8 卡 torchrun，eager（不用 CUDA graph）；每輪把所有（操作、大小、背景）組合以打亂順序各跑一次，30 輪；
  背景 = 每卡在另一條 stream 上連續搬 16 MiB 的釘住主機記憶體（NUMA 本地）；每個窗口 2 次暖身＋10 次計時，取 8 卡最慢者，和**同一輪**沒搬運的窗口相比（取中位數）。
- 結果（倍數 = 有搬運的時間 / 沒搬運的時間；GPU→主機 = d2h，主機→GPU = h2d）：

| 大小 | all-reduce d2h | reduce-scatter d2h | all-gather d2h | 三者 h2d |
| --- | --- | --- | --- | --- |
| 1–4 MiB | 1.41–1.68 | 1.45–1.62 | 1.59–1.64 | 1.03–1.20 |
| 16 MiB | 1.18 | **3.02** | **2.89** | 1.05–1.52 |
| 32 MiB | **2.03** | **1.93** | **2.06** | 1.08–1.13 |
| 64 MiB | 1.38 | 1.38 | 1.43 | 1.03–1.07 |
| 128–256 MiB | 1.02–1.05 | 1.03–1.08 | 1.03–1.08 | 1.01–1.02 |

- 被拖慢的是通訊，不是搬運：通訊進行中搬運仍有 12.4–12.9 GB/s／卡（單獨搬 12.6–13.0）。
- 每次多出來的時間隨大小**變少**（RS 16 MiB 多 0.26 ms、256 MiB 只多 0.04 ms）→ 不是每次固定的額外成本。
  （推論：Simple 協定每一步都有系統層級屏障，X1c 已量到屏障被 d2h 拖慢；大訊息用的通道多、屏障被平行藏住。驗證：固定通道數掃大小，看敏感度是否跟著通道數走。）

### T1 的協定比較（強制 `NCCL_PROTO`）

- 16 MiB 以上 NCCL 預設就是 Simple（時間相同）。
- LL128 對搬運鈍感得多（16 MiB ×1.18–1.34、32 MiB ×1.10–1.19），但沒搬運時大訊息比 Simple 慢。
- 有 d2h 時的實際時間：RS 16 MiB Simple 0.38 ms vs LL128 0.17 ms（**LL128 快 2.2×**）；AR 32 MiB 0.75 vs 0.43（1.75×）；AR 64 MiB 約相同；AR 128 MiB Simple 較好。
  → **最好的協定隨「有沒有背景搬運」而翻轉**，NCCL 的選擇只在沒搬運時對。

### T1b：只有部分卡在搬

| 誰在搬（d2h） | RS 16 MiB | AG 16 MiB | AR 32 MiB | AR 64 MiB |
| --- | --- | --- | --- | --- |
| 只有 GPU0 | 1.84 | 1.86 | 1.26 | 1.06 |
| GPU0＋GPU1（同一個交換器） | 2.88 | 2.81 | 1.97 | 1.35 |
| 全部 8 張 | 3.02 | 2.89 | 2.03 | 1.38 |

→ **一個交換器的上行線被搬滿，就幾乎等於全部卡都在搬**；8 卡的集合通訊被最慢的那張卡拖住（推論）。
h2d 只有 GPU0 搬時 ≤ 1.12。

### T2：跨節點傳送能不能繞過「網卡從 GPU 讀很慢」（單機 RoCE 迴環）

- 做法：`scripts/topic/t2_rdma_paths_v1.sh`，在 GPU0/GPU1 的交換器上，`ib_write_bw` 同一埠迴環（網卡讀來源、寫目的），5 輪打亂。
- 結果：主機→主機 100.8 Gb/s（滿速）；網卡讀 GPU 47.9 Gb/s，**開 8 個 QP 也只有 48.4**（不是 QP 數的問題）；
  網卡讀主機→寫 GPU 97.4 Gb/s（上行線空閒時滿速）；**但同交換器的兩張 GPU 同時往主機搬時，網卡讀主機只剩 30.5 Gb/s**（兩埠合計 45.6）。
- 意思：「先搬到主機再由網卡送」不是免費的——它自己的 d2h 就會讓網卡讀不動。這個做法在 MVAPICH2-GDR、UCX、NCCL（`NCCL_NET_GDR_READ=0`）都已存在。
  → 跨節點的「繞過讀取上限」題目**不推薦**；但 T2 是推薦題目的另一個證據：d2h 也會拖垮同一交換器上網卡的送出（對兩台訓練有用，之後可加）。
- 迴環的數字不能當跨節點數字引用（網卡同時也在寫同一交換器上的 GPU；與 E4 跨節點差 −14% / −21%，見 analysis.txt）。

### 錨點與未解的差異

- 搬運速度：單卡 h2d 21.7（M0 21.3）、同交換器兩卡 25.3–26.1（M0 25.2）相符。
- **未解**：8 卡同時搬 101–104 GB/s，比 M0 的 85.4 高 18–22%。兩者的主機緩衝區來源與搬法不同（推論；驗證：互換搬法重量）。不影響倍數結論，但正式論文前要解掉。
- NCCL 沒搬運時的頻寬比 nccl-tests（同一個 NCCL 2.21.5，單程序 8 卡）高 9–12%，在 20% 內，原因未查（啟動方式不同）。

---

## 4. 候選題目比較

文獻檢查由三個獨立的搜尋做（刻意找「已經有人做過」的證據）；重點與網址在附錄。

| # | 題目 | 推論/訓練 | 需要 159？ | 別人做到哪 | 留給我們的空間 | 10 週可行 | 被打掉的風險 | 建議 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **T** | **主機↔GPU 搬運拖慢 NVLink 集合通訊：量測＋避開的排程器** | 訓練（推論可當第二案例） | 否 | 機制見於 GitHub issue（#2218、PyTorch 論壇），無論文；ChunkFlow 只在純 PCIe 機器；PipeOffload 附錄提到同交換器兩卡互搶 | 訓練大小的系統量測（今天已有 ×2–3）、協定翻轉、一個交換器就夠；排程器＋接進 FSDP / DeepSpeed 的端到端 | 高 | 中（NVIDIA 可能修；端到端增益要靠第 1 週驗證） | **推薦** |
| F | fusion-dispatch 寫成論文（per layer、per token 數選融合或不融合） | 推論 | 已量完 | TokenWeave 用固定門檻；FlashOverlap 有預測器但只在單一重疊設計內；vLLM 在 Ampere 要手設門檻；ICML'26 workshop 呼籲「依工作量選切法」但沒做 | 模型預測＋regret＋換模型／換連線都對（跨節點簡單規則慢到 2×） | 很高 | 中（審稿會問「為何不直接離線量完」） | **備案** |
| M | MoE 冷門專家：CPU 算＋8 卡多線搬（現行） | 推論 | 否 | KTransformers 已有 prefill 逐層多線搬；單卡＋CPU 很擠 | decode 時同時分工、互搶感知 | 中 | 中高 | 第三 |
| E | A100＋RoCE 兩台 16 卡 MoE expert-parallel | 推論 | **是** | DeepEP 單機 A100 已支援；UCCL-EP 沒在 A100/RoCE 測 | 跨節點、讀取上限、多 rail | 中低（159 被佔、少一張網卡） | 中 | 不建議當主線 |
| K | Kimi-K2 級模型跨兩台服務（PP/EP/TP 選擇） | 推論 | **是** | 2610.05305 有 TP/PP 解析模型；MegaScale-Infer 等 | 容易變成部署報告 | 中低 | 中 | 不建議 |
| G | 網卡讀 GPU 慢的診斷與繞路（主機中轉） | 兩者 | **是** | MVAPICH2-GDR（2013）、UCX、NCCL `GDR_READ=0` 都有；T2 顯示中轉自己會被 d2h 拖垮 | 只剩量測診斷 | 中 | 高 | 不建議 |
| — | 兩台資料平行／HSDP、壓縮、DiLoCo | 訓練 | 是 | MiCS、ZeRO++、SDP4Bit、DiLoCo 系列 | 幾乎沒有 | — | 高 | 不建議 |
| — | rail 感知集合通訊 | 兩者 | 是 | PXN、rail-only、TACCL、TE-CCL、ForestColl | 兩台時近乎解析解 | — | 高 | 不建議 |
| — | RL 權重同步 | 訓練 | 是 | checkpoint-engine、Perplexity、SGLang P2P、Laminar | 整合工作 | — | 高 | 不建議 |
| — | TP decode all-reduce 延遲 | 推論 | 否 | 非常擠（vLLM、TRT-LLM、FlashInfer、TokenWeave、Ladder Residual） | — | — | 高 | 不建議 |
| — | copy engine 做集合通訊（省 SM） | 兩者 | 否 | all-gather 已做完（NCCL 2.28、Flux、MSCCL++）；reduce 類仍有空間 | A100 上的 RS/AR | 中 | 中 | 可併入 T 的討論 |
| — | 多卡 offload 的拓撲感知（交換器、NUMA、CPU vs DMA 互搶） | 訓練 | 否 | Deep Optimizer States、PipeOffload、BytePS 各有一塊 | 合起來建模 | 中 | 中 | **併入 T 當一個元件** |

---

## 5. 推薦題目 T 的細節

### 5.1 要講的故事

大家以為：NVLink 機器上，集合通訊走 NVLink，PCIe 上搬什麼都不影響它，所以 offload（把梯度、activation、optimizer 狀態搬到主機）可以放心和通訊重疊。
實際上：GPU→主機的搬運會讓 NVLink 上 16–64 MiB 的集合通訊慢 1.4–3 倍，而且**一個 PCIe 交換器**（兩張卡）在搬就夠。
NCCL、PyTorch、DeepSpeed 的排程都不知道這件事。

### 5.2 研究問題

- Q1 多大、在哪些條件出現：大小 × 協定 × 通道數 × 方向 × 哪幾張卡在搬 × 搬多快 × eager / graph。
- Q2 為什麼：系統層級屏障（X1c 已量到被 d2h 拖慢）？上行線排隊？用沒有 ncu、沒有 sudo 也做得到的微基準驗證。
- Q3 怎麼避開：與簡單做法（不重疊、限速、低優先權 stream、一律 LL128、ChunkFlow 式讓路）比，誰的「通訊變慢 vs 搬運完成時間」取捨最好。
- Q4 真實訓練省多少。

### 5.3 系統架構（論文的「架構」部分）

1. **干擾模型**：輸入（集合通訊大小、協定、通道數、每個交換器上正在搬的方向與速率），輸出預期的通訊時間。從 Q1 的量測校準；沿用 fusion-dispatch 的成本模型寫法。
2. **協定／通道選擇**：offload 執行期告訴 NCCL「現在有搬運」，按干擾模型選協定（NCCL tuner plugin，或預先建立兩個不同協定的 communicator 按次選）。
3. **搬運節拍器**：把 offload 切成小塊，由 GPU 端訊號（集合通訊開始／結束時寫的計數器，沿用 hxb 與 K4 的放行）只在通訊空檔放行；
   有截止時間（optimizer 更新或下一次 prefetch 前必須搬完）時，依模型決定放行多少；同一交換器的兩張卡錯開（併入 C1 的拓撲感知）。
4. **接進訓練框架**：PyTorch FSDP2 CPU offload 或 DeepSpeed ZeRO-Offload / ZeRO-3 offload，加上 Megatron 式 TP＋序列平行的 activation offload。
5. **（第二案例，可選）推論**：prefill 時 KV／專家搬運（M0 的情境），以及將來的 PCIe 晶片。

### 5.4 和既有工作的差別

| 工作 | 他們做的 | 我們多的 |
| --- | --- | --- |
| NVIDIA/nccl #2218（2026-06，已關閉，沒修） | V100 / H200 上 d2h 讓 all_gather 掉 8–11%；指出系統屏障；建議改 LL/LL128 | A100 上訓練大小 ×2–3；協定好壞隨搬運翻轉（不能一律 LL128）；一個交換器就夠；排程器＋端到端 |
| PyTorch dev-discuss（2024，H100） | 非同步 D2H 讓 all-gather 2.7 → 4.4–13 ms，沒解決 | 同上 |
| ChunkFlow（arXiv 2605.11335） | 純 PCIe 的 2×H100、diffusion 推論；預取讓路給集合通訊 | NVLink 機器（通訊根本不走 PCIe 仍受害）、訓練、方向與交換器拓撲 |
| PipeOffload（2503.01328，附錄 D） | 同交換器兩卡的 H2D/D2H 互搶，偏好錯開 | 對集合通訊的影響、協定選擇 |
| Deep Optimizer States（Middleware'24）、BytePS（OSDI'20） | CPU 更新與 PCIe 搬運搶記憶體；繞開網卡所在交換器 | 合併成一個模型與排程器 |
| X 系列（我們自己，10-28） | decode、graph 內的小 all-reduce；主因是 graph 的 event 節點（可關設定） | eager、訓練大小；**沒有設定可以關掉**——Simple 協定本身受害、LL128 大訊息又慢，只能排程 |

### 5.5 十週計劃（每步先登記預測、guard＋tmux）

| 週 | 內容 | 關卡 |
| --- | --- | --- |
| 1 | **否證實驗**：在真實訓練步驟裡量（Llama-3-8B / Qwen2.5-7B 級，FSDP2 CPU offload 與 TP＋SP activation offload 各一）——通訊因背景搬運多花的時間占一步的幾 %。同時解 101 vs 85 GB/s 的差異。 | **至少一種真實設定 ≥ 10% 的步驟時間來自干擾 → 繼續；否則換備案 F** |
| 2 | Q1 系統量測（通道數、塊大小、限速、方向、哪幾張卡、graph vs eager、較新的 NCCL 2.27/2.28 是否已修） | 新版 NCCL 若已修，論文改寫成「A100＋現行版本」並報告 |
| 3 | Q2 機制微基準（系統屏障延遲 vs 上行線負載；沿用 X1c 工具） | |
| 4–5 | 干擾模型＋協定選擇＋搬運節拍器（單元測試、正確性） | 模型預測誤差 ≤ 15% |
| 6 | 接進一個訓練框架 | 損失曲線與原版逐步一致 |
| 7–8 | 端到端評估＋對照組＋消融（只換協定 / 只節拍 / 兩者） | 事前登記的成功標準（第 1 週定） |
| 9 | 推論第二案例（可選）＋ auditor 審查 | |
| 10 | 寫作 | |

### 5.6 風險與退路

- **端到端增益小**（offload 多半和計算重疊、不是和通訊重疊）→ 第 1 週關卡；不過就換 F。
- **NVIDIA 修掉**（新版 NCCL）→ 第 2 週先測；A100＋PyTorch 2.6 搭的 NCCL 2.21 仍是大量部署的現況。
- **審稿說「只是換成 LL128」** → T1 已顯示 128 MiB 以上 LL128 較慢、只換協定不夠；要用消融證明節拍器＋選擇一起才最好。
- **機制講不清楚**（沒有 ncu、沒有 sudo）→ 論文以「可預測、可避開」為主，機制寫成有證據的推論。

### 5.7 與 Flux 的切割結合（T3，2026-10-30 使用者提問後加做）

量測：`scripts/topic/t3_flux_vs_hostcopy_v1.py`、`results/topic/t3_v1/`（Qwen2.5-32B 的層形狀，TP8＋序列平行，4096 token；30 輪打亂、guard CLEAN；融合結果與 NCCL 路徑一致）。

| 運算（每次） | 沒搬運 | 8 卡往主機搬 | 融合比不融合快（沒搬 → 有搬） |
| --- | --- | --- | --- |
| AG gate_up（大 GEMM）Flux 融合 / NCCL＋mm | 1.20 / 1.26 ms | ×1.11 / ×1.23 | 1.06× → 1.17× |
| AG qkv（小 GEMM，通訊為主） | 0.34 / 0.46 ms | **×2.12** / ×1.65 | 1.37× → 1.06× |
| RS down（大 GEMM） | 0.63 / 0.77 ms | ×1.07 / ×1.36 | 1.22× → 1.55× |
| RS o_proj | 0.26 / 0.37 ms | ×1.21 / ×1.72 | 1.42× → 2.02× |
| 只有 GEMM | — | ×1.00–1.09 | — |

意思：
1. **GEMM 夠大時，Flux 的切割本身就是緩衝**：通訊雖然變慢，仍被 GEMM 蓋住（×1.07–1.11，NCCL ×1.23–1.36）。
2. **GEMM 太小時，Flux 受害比 NCCL 更重**（qkv ×2.12）：被拖慢的 copy engine all-gather 直接露出來。Flux 每個 tile 都用系統層級的 acquire / fence（`include/flux/cuda/system_barrier.hpp:41-52`）——是否為原因未隔離（推論）。
3. **「該不該融合」會隨有沒有搬運改變**：o_proj 的融合優勢從 1.42× 變 2.02×，qkv 從 1.37× 掉到 1.06×。

→ 架構可以合成一件事：每一層決定 (a) 融合或不融合（fusion-dispatch 的成本模型加一項干擾）、(b) PCIe 搬運放在哪個窗口——放在大的融合 GEMM 期間（會被吸收），避開 qkv / 注意力這類通訊為主的窗口、
(c) 用 Flux 的方式切割搬運本身：梯度依 tile／小塊切開，由 GPU 端訊號在 GEMM 算完那一塊時放行（如同 Flux 的 GEMM+RS 在 epilogue 把每個 tile 送給別張卡，只是這裡送去主機、而且挑時間）、(d) 不融合的 NCCL 依干擾選協定。
訓練要用的反向（dgrad 與 AG/RS 對偶、wgrad 需要已收集的輸入）要自己接，屬工程；AGKernel 不能錄 CUDA graph（eager 訓練不受影響）；一組 op 建好不要銷毀（陷阱表）。

---

## 6. 備案 F：fusion-dispatch 寫成論文

- 已有：G0–G4（op regret 0.17%、block 0.02%、比 vLLM 預設省 8.0%）、E4a2（跨節點 20/21 選對、簡單規則最差慢 103%）。
- 文獻空間（搜尋結果）：TokenWeave 用離線量出的固定門檻（1K / 4K token）；FlashOverlap 有延遲預測器但只在自己的重疊設計內選切法、不會退回不融合；
  vLLM 在 Ampere 上融合門檻要手設；ICML'26 workshop 論文說最好的切法隨工作量翻轉、需要「依工作量的切法策略」但沒有做。
  **沒找到**「用 α-β＋tile 模擬預測每層、每個 M 的融合／不融合／切法，報告對 oracle 的 regret，且在沒看過的模型與慢連線上仍成立」的論文。
- 缺的：vLLM 端到端（F4）、auditor、跨節點要 159。
- 弱點：單機上「decode 用 vLLM、prefill 用 Flux」幾乎一樣好（省 7.5% vs 8.0%）。

---

## 7. 待使用者決定

1. 主題目選 T（推薦）、F（最穩）、還是繼續 M？
2. 若選 T：第 1 週的真實訓練要用哪個框架（建議 PyTorch FSDP2，裝在獨立 venv，不動 pixi）、哪個模型大小？
3. 159 上目前跑著你帳號的 SGLang 服務（DP4×TP2）——是你自己開的嗎？跨節點量測要先停它；另外 159 少了一張網卡（rail 5/6），需要管理員處理。

---

## 附錄：文獻檢查重點（2026-10-30，三個獨立搜尋；標「未查證」者沒讀原文）

**推論側**
- TokenWeave（arXiv 2505.11329，MLSys'26，H100/B200）：每步依 token 數與離線門檻切換。https://arxiv.org/abs/2505.11329
- vLLM fusion 文件（v0.19.1）：AsyncTP 門檻只在 SM90 自動設定。https://docs.vllm.ai/en/v0.19.1/design/fusions/
- FlashOverlap（2504.19519，EuroSys'26，A800 / 4090 / 910B）：延遲預測器取代 profiling。https://arxiv.org/abs/2504.19519
- Layout and Fusion Trade-offs for MoE Inference under Single-Node TP（ICML'26 workshop）。https://icml.cc/virtual/2026/75111
- DeepEP issue #162：A100 單機測過、跨節點沒測。https://github.com/deepseek-ai/DeepEP/issues/162
- UCCL-EP（2512.19849，OSDI'26）：沒有 A100、沒測 RoCE。
- Characterizing Parallelism Strategies in LLM Inference（2610.05305）；Meta 多節點 MoE（2604.23150）；NVRAR（2511.09557）。
- 未查證：Domino、Centauri、Lancet、T3 的內容；Concerto 摘要取不到。

**跨節點**
- MVAPICH2-GDR / Potluri ICPP'13：GDR 讀慢、經主機管線化（只看到 SC15 投影片）。
- UCX `rndv/put/mtype`：copy engine 中轉＋RDMA put。
- NCCL 2.21.5 原始碼：`net.cc:709` GDR 讀關閉時送端 Simple 緩衝在主機記憶體、GPU kernel 以 SM 寫入；`paths.cc:380–396` 計算能力 ≥ 8.0 預設開 GDR 讀。
- NCCL issue #1295：`GDR_READ=0` 結果不一，無解釋。https://github.com/NVIDIA/nccl/issues/1295
- HPC Advisory Council：網卡讀頻寬受 `MAX_ACC_OUT_READ`、MaxReadReq、ACS 限制（要 root）。
- FuseLink（OSDI'25）、TCCL（ASPLOS'24）、MiCS、ZeRO++、SDP4Bit、DiLoCo 系列；checkpoint-engine、Perplexity 2510.27656、SGLang P2P（2026-04-29 部落格）、Laminar 2510.12633。

**單機訓練**
- NVIDIA/nccl #2218（2026-06 開、08 關，沒修）。https://github.com/NVIDIA/nccl/issues/2218
- PyTorch dev-discuss（2024）。https://dev-discuss.pytorch.org/t/overlapping-device-to-host-copy-with-gpu-collectives/2120
- ChunkFlow（2605.11335）；NVIDIA/nccl #2532（2026-10-29，graph 內小 all-to-all 拖慢 AG/RS，GB300）。
- PipeOffload（2503.01328，附錄 D，會議未查證）；Deep Optimizer States（2410.21316）；MLP-Offload（2509.02480）；RoundPipe（2604.27085）；MemFerry（2610.09657）；BytePS（OSDI'20）；ZeRO-Infinity（2104.07857）。
- NCCL 2.28 CE 集合通訊（只有 AllGather / AlltoAll / Gather / Scatter；A100 能否跑是從原始碼推論、未測）；ConCCL、DMA Collectives、MSCCL++、TileLink。
