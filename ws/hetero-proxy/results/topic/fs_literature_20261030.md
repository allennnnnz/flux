# FS 文獻檢查（2026-10-30，背景搜尋＋本 session 親讀 Syncopate）

問題：A100 NVSwitch、TP8、bf16 上，Flux AG+GEMM 在 M = 512–4096 時離 max(AG, GEMM) 很遠（我們的資料中位數 1.30×），改 Flux 的 tile 排程能否補起來？別人做了沒？

| 工作 | 硬體 | AG+GEMM 的做法 | 與 Flux 比 | 小 M |
| --- | --- | --- | --- | --- |
| Flux（2406.06858v2） | A100 PCIe / NVLink、H800 | copy engine 依序拉、stream-K tile 依 rank 輪轉 | — | 論文自承小 m 的重疊效率在 A100 NVLink 為 14–88%（Fig. 14，度量不同）；sm80 AG 排程自 2024-07 未改；issue #120：AG 的 split-K「理論上可行、未實作」 |
| **Syncopate（OSDI'26，2601.20595）** 本 session 親讀 | **8×H100**（無 A100） | 「重排 wave 順序，讓每個 chunk 一到就被用」；窮舉 chunk 大小、tile config、5 種通訊後端（含 copy engine、TMA）、通訊 SM 數 | AG-GEMM：8 卡平均為最佳手寫基準（含 Flux）的 104%、4 卡 99.8% | **沒有列出 M 值，沒有小 M 對 Flux 的結果**；有提到 wave quantization；沒有量 copy engine 與 GEMM 的干擾 |
| TileLink（MLSys'25） | 8×H800 | 靜態 tile→rank 對應；AG 用 DMA | AG+GEMM 為 Flux 的 94.5% | 全部 S=8192 |
| Triton-distributed（2504.19442） | H800、MI308X、L20 | persistent kernel、靜態跨步＋rank 旋轉、每 tile 等待 | AG+GEMM 平均 1.09× Flux | 小 M 只在圖中（未查證） |
| ParallelKittens（MLSys'26） | H100 / B200 | SM / TMA 搬 | 0.97–2.33× Flux | 指出 Flux 等「用 copy engine 做 AG 的融合在較小矩陣上比不重疊還慢」→ **競爭解釋**（FS1b 檢驗） |
| Event Tensor（2604.13327） | 8×B200 | 動態 megakernel，但 AG+GEMM **刻意用靜態排程** | 無 | 只有 M=8192 |
| FlashOverlap（EuroSys'26） | A800 / 4090 | 只做 GEMM 之後的通訊（AR/RS/A2A），wave-group 訊號＋預測器 | 多數情況勝 Flux | 無 AG+GEMM |
| PyTorch async-TP | H100（sm90 only 的單 kernel 版） | 靜態 persistent、每 chunk 等訊號 | 無 | 官方文說分解法在小問題上有排程開銷 |
| Weave（2609.21483）、MPK（OSDI'26） | H100 等 | 動態搶 tile／事件驅動 | — | 分別只做 MoE、decode all-reduce |

背景搜尋的結論（中等風險）：
- 沒找到「依資料是否到達動態分派 tile」用在 dense AG+GEMM 的工作；沒找到在 A100、M = 512–4096、以 max(AG, GEMM) 為基準量 Flux 的工作。
- 最強威脅是 Syncopate：「依到達順序排 wave＋聯合調 chunk/tile」已在 H100 上與 Flux 打平或略勝 → A100 移植會被說成增量，除非動態分派明顯勝過「調好的靜態順序」。
- 另外兩個風險：只換 config 就補起來（FS1 檢驗）；差距其實是 copy engine 與 GEMM 互搶（FS1b 檢驗）。
