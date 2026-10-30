Supersedes: 無

# 選項 FS（改 Flux 的排程）驗證第 1、2 步：計劃與事前登記（2026-10-30，跑之前提交）

背景：fusion-dispatch 既有資料（`ws/fusion-dispatch/results/final_ag_points.csv`，M ≥ 512，40 點）顯示 Flux AG+GEMM 融合時間是
理想值 max(Flux AG 單獨, GEMM 單獨) 的中位數 1.30×、最多 1.72×；M ≤ 2048 最差（STATUS §0 FS）。

要回答：**這個差距是「選錯 config」（換設定就好，不算貢獻）還是「排程本身」（要改 Flux，可以當論文）？** 以及別人是否已經解決。

## 第 1 步：只換 config 能拿回多少（不重編）

- `AGKernel.profiling()` 會跑完所有已編譯的 config（`src/ag_gemm/ths_op/all_gather_gemm_op.cc:362-475`），用 `ProfilingContext.get_all_prof_results()` 取出每個 config 的時間。
- 同一個程序、同一種計時方式（CUDA event、5 次暖身＋10 次、取 8 卡最慢者）量：預設 config 的融合、最佳 config 的融合、Flux AG 單獨（`AllGatherOp.run`）、`gemm_only`（預設 config）、cuBLAS。
  理想值 = max(AG 單獨, min(gemm_only, cuBLAS))。
- 形狀：fusion-dispatch 的 G-FC1（49152×12288）、G-QKV（36864×12288）、L-GU（57344×8192）、L-QKV（10240×8192），加上 Qwen2.5-32B 的 gate_up（55296×5120）、qkv（7168×5120）；TP8，bf16；
  M = 512、1024、2048、4096、8192。每個 AGKernel 只建一次、不銷毀（陷阱表）；guard＋tmux。
- 先跑錨點：預設 config 的融合時間與 `final_ag_points.csv` 同點比較（方法不同，差 ≤ 20% 才繼續）。

**預測**：
- P1：M ≤ 2048 時，最佳 config 只拿回不到一半的差距（最佳融合 / 理想 的中位數 ≥ 1.15）。
- P2：M ≥ 4096 時，最佳 config 與預設差 ≤ 5%（陷阱表裡的 PCIe config 形狀除外）。

## 第 2 步：文獻

TileLink、Triton-distributed、FlashOverlap、PyTorch async-TP、Megatron／TransformerEngine 的 comm overlap，以及其他宣稱勝過 Flux 的工作：
有沒有在 A100（或一般情況）解決「M 小時 AG+GEMM 重疊差」？用什麼方法？比 Flux 快多少？

## 判定（事前登記）

- **繼續到第 3 步（kernel 內時間戳＋改排程原型）**：M = 512–2048 時，最佳 config 之後仍剩中位數 ≥ 1.15× 的差距，**且**文獻中沒有已發表的方法在 A100 上把這個差距補起來。
- **不做**：最佳 config 就把中位數差距壓到 < 1.10×（只是選 config 的問題），或已有論文做了同一件事。
- 介於之間：寫明差多少，交使用者決定。

## 補充（2026-10-30，文獻檢查回來後、FS1b 跑之前）：競爭解釋的區分實驗 FS1b

文獻（ParallelKittens，arXiv 2511.13940）認為 Flux 這類用 copy engine 做 AG 的融合在小矩陣上變慢，是因為 copy engine 本身；若差距來自「GEMM 與 copy engine 同時跑時互相拖慢」，排程改不了。
- **FS1b**：同一形狀，Flux AG（`AllGatherOp.run`，side stream）與 GEMM（`gemm_only`，main stream，讀另一份已收集好的輸入）**同時跑、彼此不等**，量兩者都結束的總時間；與 ideal = max(AG, GEMM) 及 fused 比較。
- 判讀：同時跑 ≈ ideal → 硬體資源不是限制，差距來自融合 kernel 的等待／排程（FS 可做）；同時跑 ≈ fused → 差距是資源互搶（FS 不成立）。
- 預測：M ≤ 2048 時，同時跑 / ideal 的中位數 ≤ 1.10（大部分差距來自排程）。
- 判定規則補一條：**繼續**需要 FS1b 在 M ≤ 2048 的中位數 ≤ 1.15。

## 補充 2（FS1b 跑完後、FS1c 跑之前）：FS1b 有混淆，改做 FS1c

- FS1b 結果：M ≤ 2048 時「同時跑 / 理想」中位數 1.30×（未過 ≤ 1.15）。但多個點「同時跑」≈ AG＋GEMM 相加（例 L-QKV M=8192 1.26 vs 1.31），像是被串行而不是互搶。
- 原因（讀原始碼）：`AllGatherOp.run` 一開始的跨卡屏障是 kernel（`flux_barrier_all_on_stream` → `nvshmemx_barrier_all_on_stream`，`src/ths_op/flux_shm.cc:501`）；
  FS1b 在另一條 stream 緊接著發 GEMM，GEMM 若先佔滿 SM，屏障 kernel 要等 GEMM 結束 → 整個 AG 被排到 GEMM 後面。FS1b 標為「有混淆、不用於判定」。
- **FS1c**：同 FS1b，但 GEMM 前先在 main stream 放約 30 µs 的 GPU sleep（只佔一個 SM），讓屏障先跑；總時間扣掉量到的 sleep 時間；並在同時跑時分別量 AG 與 GEMM 各自的時間，與單獨跑比。
- 預測：FS1c 的「同時跑 / 理想」在 M ≤ 2048 中位數 ≤ 1.15；同時跑時 GEMM 變慢 ≤ 10%。判定規則不變（≤ 1.15 才繼續）。
