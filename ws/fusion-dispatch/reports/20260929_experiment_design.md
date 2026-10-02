Supersedes: （無）

# Flux 開關決策器：實驗設計

日期：2026-09-29 · 作者：boss session · 狀態：**設計，未執行**

本文件回答：「要做一個決策器，在推論時依輸入大小決定開不開 Flux」這件事，
在動手實作之前要先量什麼、怎麼量、量完怎麼判斷該用哪種決策器。
所有未經量測的東西標 **[推論]** 並附驗證方法；讀程式碼得到的事實標 **[程式碼]** 並附行號。

---

## 0. 一句話

量出**每一層、每一個 M**上「Flux fused」與幾種「不 fused」路徑誰快的完整地圖，
拆解切換點附近輸贏的原因，再拿這張地圖評估三種決策器（門檻 / 查表 / 成本模型），
看哪一種在推論的 M 分佈下離 oracle 最近、開銷最小。

## 1. 問題的精確定義

### 1.1 論文講的是 m（token 數），不是 n

論文（arXiv:2406.06858）的符號：GEMM 是 (m × k) · (k × n)，**m 是 token 維度**；
(n, k) 在 AllGather 側固定為 (49152, 12288)（Section 5.1）。論文裡說 Flux 輸給不 overlap
基準的地方都講 m：

- Section 6 / Fig. 14：*"Flux could perform worse than the non-overlapping baseline in a few extremely small m cases"*；
  原因：*"when m is extremely small, the GEMM kernels typically have fewer warps, making latency hiding less efficient"*。
- Section 5.2 / Fig. 17：decoding batch 64 時 *"still has 5 cases slower than the non-overlapping vLLM baseline"*。
- H800 上 ReduceScatter m=64 有 0.95× 的例子（TMA store 沿 m 只剩 8）。

（引文取自 arXiv HTML 版，2026-09-29 讀取。寫進正式文件前請對 PDF 再核一次。）

Phase 0 看到的「N 小時 overlap 差」（M=4096 固定，N=4096 效率 ~2%）是**另一個維度**。
兩者機制相同：GEMM 不夠長，藏不住通訊。但**推論時會變的只有 M**：

- 每一層的 N、K 由模型決定，部署後固定。
- M = 這一次 forward 的總 token 數（decode 時 ≈ batch size，prefill 時 = chunk 長度），每步都在變。

所以決策器的形式是：**每一層一個 M 的函數** `f_layer(M) → {fused, 某條非 fused 路徑}`。
M 在 host 端由 tensor shape 直接得知，決策是一個 host 端分支，不佔 GPU 時間。

### 1.2 「不開 Flux」不是一條路徑

「off」至少有三種做法，而現有資料無法判斷哪一種最好（第 2 節）。實驗必須把它們都當成候選：

| 代號 | 內容 | 為什麼要量 |
| --- | --- | --- |
| **A** `fused` | `AGKernel.forward`，All2All + `use_read`（D-005），預設 hparams | 現況 |
| **B** `nccl_cublas` | `dist.all_gather_into_tensor` + `torch.matmul` | 論文與 vLLM/Megatron 的不 overlap 基準 |
| **C** `fluxag_cublas` | `flux.AllGatherOp.run`（All2All pull）+ `torch.matmul` | Flux 通訊在 M≥1024 比 NCCL 快 13–22%（E_CORRECTION E.1）；可能是最好的「off」 |
| **D** `fluxag_fluxgemm` | `AllGatherOp.run` 之後 `AGKernel.gemm_only`，串行 | 診斷用：跟 A 用同樣的 kernel，只差 overlap |
| **A\*** `fused_tuned` | 同 A，但 hparams 用該 M 的 profiling 最佳值 | 分開「fusion 本身不好」與「沒調過的 config 不好」（見 3.1）；**條件性執行**，見 5.3 |

同一輪內另外單獨計時四個元件：`t_nccl_ag`、`t_fluxag`、`t_cublas`、`t_fluxgemm`（= `gemm_only`）。

## 2. 現有資料能告訴我們什麼、不能告訴我們什麼

| 已知 | 數字 | 來源 |
| --- | --- | --- |
| M=4096, K=12288：fused vs Flux 元件相加（comm + gemm_only，分開量後相加，不是實測的串行 D） | N=4096：0.736 vs 0.746（fused 快 1.3%）；N=8192：0.677 vs 1.032（快 34%）；N=49152：2.747 vs 3.030（快 9.3%） | FLUX_BASELINE F.3/F.4 |
| Flux AG vs NCCL AG | M=1024–16384 全程快 13–22% | E_CORRECTION E.1 |
| cuBLAS GEMM 單獨時間 | **沒量過**（E.3 那個「torch 約 3.30 ms」來自 test 腳本的 20 次平均，非交錯，不能引用） | — |
| M < 1024 的任何數字 | **沒有**（通訊、GEMM、fused 都沒有） | — |

結論：

1. 現有資料**沒有任何一點**顯示 off 比 on 快。
2. 如果切換點存在，它落在**完全沒量過**的 M < 1024 區域，也就是 decode 區。
3. 「off」的主力候選 B、C 都需要 cuBLAS 的獨立時間，而那個數字不存在。

所以決策器不能用現有數字設計，必須先量。

一個粗估，只用來選形狀，不能當結論用 **[推論]**：M=64 時，GPT-3 FC1（每卡 weight 6144×12288 bf16 = 151 MB）
的 GEMM 受 weight 讀取限制，以 A100 HBM 規格 2.0 TB/s 算至少 ~75 µs；通訊只有 1.4 MB 入向，
以 188 GB/s 算頻寬項 ~7 µs。GEMM 佔絕大部分，off 最多省下通訊那一小段，輸贏看 fusion 的固定開銷。
Llama-3-70B QKV（每卡 weight 1280×8192 = 21 MB）在 M=64 時 GEMM ~10 µs、通訊頻寬項 ~5 µs，
兩者都小，由延遲主導，**最可能出現 off 勝的層**。
驗證方法：E1 地圖直接量。

## 3. 讀程式碼得到的三件事（會改變實驗設計）

### 3.1 調好的 config 只在 M 完全相符時才用得到 [程式碼]

- forward 路徑用 `(m, n, k, world, nnodes, All2All)` 當 key 查 `TuningConfigRegistry`
  （`src/ag_gemm/ths_op/gemm_with_barrier.cc:217-228`）。
- 查表是 `std::map::find`，要完全相符（`include/flux/op_registry.h:108-116`）。
- 查不到就退回「第一個註冊的 hparams」，跟形狀無關（`include/flux/op_registry.h:203` 起）。
- A100 tp8、bf16、無 bias、RCR（`transpose_weight=False`）有調過的點
  （`src/ag_gemm/tuning_config/config_ag_gemm_kernel_sm80_A100_tp8_nnodes1.cu`）：

| 每卡 n | k | 有調過的 M |
| ---: | ---: | --- |
| 6144 | 12288 | 64, 256, 512, 1024, 2048, 4096, 8192, 16384 |
| 4608 | 12288 | 64, 512, 4096, 16384 |
| 1024 | 12288 | 4096, 16384 |
| 1280 | 8192 | 4096, 16384 |
| 7168 | 8192 | 64, 512, 4096, 16384 |
| 512 | 12288 | **無** |

推論時 M 幾乎每步都不同，**絕大多數 M 會落到 fallback config**。
「Flux 在小 M 輸」可能不是 fusion 的問題，而是 fallback config 的問題。實驗必須把這兩者分開，做法見 5.3。

- 這個 build 沒開 protobuf（`build.sh:15` `WITH_PROTOBUF="OFF"`；library 裡有
  "add tune config at runtime is not supported" 字串），所以 `FLUX_TUNE_CONFIG_FILE` 無效。
- `AGKernel.profiling()` 找到最佳值後只寫進 codegen 字串（`src/ths_op/ths_op.cc:308-317`），
  不回寫 registry，之後的 `forward()` 仍用 fallback。
- Python 的 `forward` 也沒有暴露 `hparams` 參數（`src/pybind/ag_gemm.cc:63-76`；
  C++ `forward_impl` 其實有這個參數）。

所以 A\* 需要重新 build。代價見 5.3。

**附帶發現（與 diag-overlap 有關）**：Phase 0 的 N=4096（每卡 n=512）在 registry 裡**沒有**條目，
跑的是 fallback；N=8192、N=49152 在 M=4096 都有。三個點裡效率最差（~2%）的那個，恰好是唯一沒調過的。
ECT 用同一個 kernel 當 GEMM-only 參考，所以 ECT 本身自洽；但「2%」有多少來自 config，目前不知道。
**[推論]**，驗證：對 (4096, 512, 12288) 做 profiling，比較最佳 hparams 與 fallback 的 GEMM-only 時間。
這要由 boss 轉給 diag-overlap。

### 3.2 M ≤ tile 高度時，fused kernel 不可能 overlap [程式碼 + 推論]

- 每個 tile 開算前，會等它 M 範圍內**所有** data chunk 的 signal
  （`src/ag_gemm/sm80_all_gather_gemm.hpp:927-939`）。
- chunk 是每個 rank 的 shard，M/world 列。
- 小 M 的已調 tile 高度是 64、128、256 列（例如 m=64 用 (64,128,64)）。

推論：當 M ≤ TILE_M，整個 GEMM 只有一列 tile，每個 tile 都要等 8 個 shard 全部到齊。
這時 fused 必然 ≈ 通訊 + GEMM + fusion 固定開銷，**結構上**就不會比 D 快。
一般地說，可 overlap 的 tile 列組數 g = min(world, ⌈M / TILE_M⌉)，g=1 時 overlap 為零。
這給決策器一個有物理意義的特徵：切換點 M\* 應該跟 `world × TILE_M` 同量級。
驗證：E2 的分解（A − D 在 g=1 時應 ≥ 0）加 nsys 時間軸（g=1 時 GEMM kernel 不會在最後一筆 copy 結束前開始算）。

### 3.3 動態 M 已支援，但 M 必須是 world 的倍數 [程式碼]

- `AGKernel(full_m=最大 M)` 建一次；`forward` 用 `input.size(0) * world_size` 當本次 M，
  buffer 取前段（`src/ag_gemm/ths_op/all_gather_gemm_op.cc:260-261`）。
- `GemmRS` 同樣有 `max_m`（`src/pybind/gemm_rs.cc:58`）。

推論服務時一個 op 可以服務所有 M ≤ max，實驗也照這樣建（每層建一次）。
SP 佈局下每卡 M/8 列，所以 M 必須是 8 的倍數。decode batch 要 pad；**pad 的成本算在決策裡**。

## 4. 形狀網格

### 4.1 層（TP=8，bf16，無 bias，weight 為 `(N/8, K)` column-shard）

| id | 來源 | N | 每卡 n | K | 用途 |
| --- | --- | ---: | ---: | ---: | --- |
| G-FC1 | GPT-3 175B FC1（論文形狀） | 49152 | 6144 | 12288 | 主；registry 覆蓋最密 |
| G-QKV | GPT-3 175B QKV | 36864 | 4608 | 12288 | 主 |
| L-QKV | Llama-3-70B QKV（GQA 8 kv heads） | 10240 | 1280 | 8192 | 主；預期最可能 off 勝 |
| L-GU | Llama-3-70B gate+up | 57344 | 7168 | 8192 | 主 |
| P0-4096 | Phase 0 對照 | 4096 | 512 | 12288 | 錨點與 3.1 附帶發現 |
| P0-8192 | Phase 0 對照 | 8192 | 1024 | 12288 | 錨點 |

L-QKV / L-GU 的 (n, k) 與 registry 裡的 (1280, 8192)、(7168, 8192) 完全一致，
GEMM+RS 側的 (n=8192, k=1024/3584) 也一致。**[推論]** 上游是照 Llama-3-70B 調的；不影響實驗。

### 4.2 M

- **主網格**（2 的冪次）：8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384。共 12 點。
- **保留點**（不拿來擬合決策器，只拿來驗證）：24, 72, 136, 264, 520, 1032, 3072, 6144。共 8 點。
- **斷崖對**：(64, 72)、(256, 264)、(512, 520)、(1024, 1032)。前者在 registry 裡有、後者沒有，
  M 只差 8 列。直接量 3.1 的 fallback 代價，不需要重新 build。

P0-* 兩層只跑 M ∈ {1024, 2048, 4096, 8192} 加斷崖對。總計約 4×20 + 2×8 ≈ 100 個 (層, M) 點。

## 5. 量測協定

逐條對應 `CLAUDE.md` 5.1。

### 5.1 基本規則

- **交錯**（第 5 條）：每一輪內把 A、B、C、D 與四個元件各跑一次，順序每輪隨機（固定 seed，
  記錄順序），各自 CUDA event。決策量是差值 Δ = t_A − t_best_off，取逐輪差值的中位數，≥ 200 輪。
- **GPU 端對齊**：每個待測項前，在同一條 stream 上跑一個 1 元素的 NCCL all_reduce，再記 start event。
  理由：M ≤ 512 時整個 op 只有數十 µs，host `dist.barrier` 的 rank 間 skew 在同一量級，會被算進
  collective 的時間。在兩個 M 上與 host barrier 對齊做對照並報告差異（第 3 條）。
- **L2 flush**：每個待測項前寫一個 ≥ 80 MB 的 buffer（A100 L2 40 MB），不計時。
  理由：L-QKV 每卡 weight 21 MB 放得進 L2，不 flush 的話同一輪裡排在後面的 arm 會讀到熱的 weight；
  實際推論時每一層的 weight 都是冷的。
- **rank-max**（第 8 條）：每一輪 all_gather 各 rank 的 event 時間，每個 arm 取 rank 最大值，
  再算逐輪 Δ，最後取中位數。
- **SM clock**（第 6 條）：`common/measure/clock_logger.py`；丟棄 < 95% 眾數的輪次，報告丟棄數。
- **正確性**：每個 (層, M) 開始前，所有 arm 對 torch 參考 allclose（全 rank）；通訊結果對 NCCL bitwise 比對。
- **每個 M 單獨 warmup**：Flux 在新 M 第一次呼叫會 lazy init（`get_op`、gemm buffer），不能算進去。

### 5.2 兩種時間模式（都要，互相對帳）

1. **單次延遲**（主）：上述協定。
2. **穩態連發**：同一 arm 連續跑 L=16 次，中間不做 host 同步，模擬一疊層；event 包住全程，
   取 per-call = 總時間 / L。同時用 host `perf_counter` 記每次呼叫的 CPU 耗時。
   小 M 時 CPU launch 開銷可能超過 GPU 時間：**[推論]** Flux All2All pull 每次要發多筆 peer copy 加 event/stream 同步，
   NCCL 只有一個 kernel（E0 用 nsys 數實際的 launch 數）。這一點只有穩態模式看得到。

兩種模式判出的勝者不同時，這是**發現**，要寫進報告，不能只報其中一個（第 3 條）。

### 5.3 A\*（調過的 fused）怎麼量

3.1 說明了 A\* 要 rebuild。分兩步，避免白做：

1. **先不 rebuild**：用斷崖對與 A vs D 的差值判斷 fallback 的影響有多大。
   若斷崖對的 t_A 差異 < 5%，且 A 與 D 的關係在斷崖兩側一致，A\* 就降為可選項。
2. **需要時再 rebuild**，優先做法是在 pybind `forward` 暴露 `hparams` 參數（C++ `forward_impl`
   已經有），這樣 A 與 A\* 能在同一個 process 內交錯量測。次選是 `./build.sh --protobuf` 後用
   `FLUX_TUNE_CONFIG_FILE`；但 registry 是全域 singleton，A 與 A\* 只能分兩個 process 量，
   靠 B、C 當兩次 run 的橋接（兩次的 B 差 > 2% 就重跑）。

`src/` 的改動要跟 diag-overlap 協調（他們也要 instrument `src/ag_gemm/`），**需 boss 裁決**先後。

### 5.4 錨點與停止條件（第 2、4 條）

在任何新數字之前，先用新協定重現：

| 錨點 | 值 | 來源 |
| --- | ---: | --- |
| fused，M=4096，N=4096 / 8192 / 49152，K=12288 | 0.736 / 0.677 / 2.747 ms | FLUX_BASELINE F.3 |
| gemm_only，同上 | 0.278 / 0.564 / 2.562 ms | 同上 |
| Flux AG All2All pull，M=1024 / 4096 / 16384 | 0.199 / 0.468 / 1.668 ms | E_CORRECTION E.1 |
| NCCL AG，M=4096 | 0.569 ms | 同上 |

- 偏差 > 20% → 停，先查量測。
- 偏差在 5–20% → 用新協定的差異（GPU 對齊、L2 flush）解釋，解釋不了就停。

停止條件：

- 任何 arm 的 t < max(自己的通訊元件, 自己的 GEMM 元件)（快過完美 overlap）→ 基準錯，停。
- |t_D − (t_fluxag + t_fluxgemm)| > 10% → 對帳，找到原因再繼續。

### 5.5 nsys（第 9 條）

每層在切換點下方、附近、上方各取一個 M 錄時間軸，確認：

- GEMM kernel 是否在最後一筆 peer copy 結束前開始算（overlap 是否真的發生）；
- g=1 時是否確實完全串行（3.2）；
- A 比 D 多出來的固定開銷落在哪一段（event wait、local copy、barrier）。

「overlap」「串行」的主張只能用時間軸撐，不能從總時間推。

### 5.6 平手

某 (層, M) 只有在逐輪 Δ 的 p10–p90 不含 0 時才算「有勝負」；否則標為**平手**。
平手區對決策器很重要：在平手區選哪邊都行，決策器不應為此付出複雜度。報告必須明列平手點。

## 6. 實驗階段

| 階段 | 內容 | 產出 | 依賴 |
| --- | --- | --- | --- |
| **E0** | 腳本、正確性、錨點重現、對齊方式對照；在一個點上試跑 `profiling()` 量耗時 | 錨點表 | — |
| **E1** | 地圖：4.1 × 4.2 × {A, B, C, D, 四元件}，兩種時間模式 | `results/` 原始逐輪 CSV；每層 speedup(M) 曲線、M\*、平手區 | E0 |
| **E2** | 歸因：切換點兩側各 ≥ 2 點做分解（見下）；斷崖對；nsys；視需要做 A\* | 分解表、g 與 overlap 收益的關係 | E1 |
| **E3** | 決策器評估（第 7 節） | 各策略的 regret 表、建議方案 | E1 |
| **E4** | 服務整合可行性：CUDA graph capture（A、B、C 各自能否 capture）；兩條路徑同時常駐的記憶體 | 可行性表 | E3 |
| **E5** | GEMM+RS 側：`GemmRS.forward` vs `torch.matmul` + `dist.reduce_scatter_tensor`；層 = G-FC2、G-O、L-down、L-O | 同 E1 格式 | E1–E3 經 auditor 審查 |

E2 的分解。四項都是**同一輪**內量到的，逐輪相減取中位數：

```
t_A − t_B = (t_A − t_D)      overlap 收益減 fusion 固定開銷
          + (t_D − t_C)      Flux GEMM kernel vs cuBLAS
          + (t_C − t_B)      Flux AG vs NCCL AG
```

分解告訴我們決策器該動哪個開關：

- 例如 t_C < t_A < t_B：最好的 off 是 C，輸在 GEMM kernel 品質，不在 fusion。
- 若 (t_A − t_D) 在 g=1 時 > 0，就是 3.2 預測的結構性損失。

**E5 的限制**：GEMM+RS 的通訊在 sm80 無法單獨量（E_CORRECTION E.4），分解做不了，只能比端到端。
但決策器本來就只需要端到端。

## 7. 決策器評估（E3）

### 7.1 候選策略

| 策略 | 規則 | 擬合資料 |
| --- | --- | --- |
| π_on | 永遠 fused（現況） | — |
| π_off | 永遠用 E1 中整體最好的非 fused 路徑 | — |
| π_thr | 每層一個門檻 M\*，M < M\* 走 off | 主網格 |
| π_lut | 每層查表；M 取主網格最近點（另測 floor 版本） | 主網格 |
| π_model | 解析式：fused 若預測 t_A < 預測 t_off，特徵用 g、通訊 bytes、weight bytes 與 `params.json` 的頻寬 | 主網格擬合少數常數 |
| π_oracle | 每點取實測最快 | 全部 |

- π_lut 與服務框架的 CUDA graph 分桶天然對齊：vLLM 類框架會把 decode batch pad 到固定幾個 capture size，
  決策可以在 capture 時就烘進每個桶。**[推論]**，驗證：E4 確認 Flux 能否 capture，並讀目標框架的桶設定。
- π_model 是唯一能推廣到新形狀與新硬體的策略（異質晶片到手後沒有現成地圖），
  也是 `ws/cost-model` 的第一個實際使用者。

### 7.2 評估集

1. **保留點**：8 個非 2 冪次 M，直接量，不內插。
2. **合成 trace**（M 只從已量測點抽，確保 regret 用實測值算）：
   - decode 為主：M 均勻取自 [8, 512] 內的已量測點；
   - prefill 為主：M ∈ {1024, …, 16384}；
   - 混合（chunked prefill）：70% decode 步 + 30% prefill chunk。

   之後可換成真實 trace（例如 ShareGPT 長度分佈）。

### 7.3 指標

- **aggregate regret** = Σ(t_π − t_oracle) / Σ t_oracle，每個 trace 各報一次；
- **誤判率**：排除平手點；
- **最差單點損失**（ms 與 %）；
- **決策開銷**：host 端 dispatch（dict / bisect）1e6 次呼叫的 per-call 時間，對比所有有勝負的點中最小的 |Δ|。

## 8. 假設 [推論]

| # | 假設 | 依據 | 驗證 |
| --- | --- | --- | --- |
| H1 | AG+GEMM 在本機存在切換點，且在 M ≤ ~512 | 論文 Fig. 14/17；3.2 | E1 |
| H2 | g=1（M ≤ TILE_M）時 t_A ≥ t_D，差額 = fusion 固定開銷 | 3.2 | E2 分解 + nsys |
| H3 | 非 registry 的 M 有可量的 fallback 代價 | 3.1 | 斷崖對；必要時 A\* |
| H4 | 小 M 時最好的 off 可能是 B 不是 C：Flux AG 的優勢在大 M 量到，小 M 由延遲主導 | E.1 只到 M=1024 | E1 元件時間 |
| H5 | f_layer(M) 單調，一個門檻就夠 | 無，tile 量化與 registry 斷崖可能打破單調 | E3 比較 π_thr 與 π_lut |
| H6 | 單次延遲與穩態連發判出的勝者在小 M 不同 | CPU launch 開銷在 Flux 路徑較多 | 5.2 |

## 9. 已知坑

- M 必須是 8 的倍數（3.3）；SP 佈局下 M < 8 不存在。M=8（每卡 1 列）Flux 是否支援未知，
  E0 先試；不支援就記錄最小可用 M，這本身就是決策器的一條規則。
- `use_cuda_core_ag=True` 對 fp16/bf16 不能跑（`CLAUDE.md` 陷阱表），不列入 arm。
- weight 形狀 `(N/8, K)`；傳完整 N 會讓 GEMM 膨脹 8 倍（陷阱表）。
- AGKernel 與獨立的 AllGatherOp 各自配置 symmetric buffer。E0 要確認兩者可共存，
  且 6 層的 op 不同時常駐（一次建一層）。
- NCCL 小訊息會自選 LL protocol。這就是 production 會拿到的，不要設 `NCCL_PROTO`/`NCCL_ALGO`；
  記錄 NCCL 版本與環境變數。
- 跨裝置 `copy_` 的 stream 陷阱（陷阱表）：C 路徑用的是 AllGatherOp，不是 `copy_`，但 E0 要用 nsys 確認沒有意外的 default-stream 串行。

## 10. 產出與成本

- `ws/fusion-dispatch/results/`：
  - `dispatch_map_v1_<layer>.csv`：逐輪原始列，欄位為 round、arm、順序、各 rank ms、rank-max ms、SM clock、mode；
  - 摘要 CSV；
  - nsys report。
- `ws/fusion-dispatch/reports/`：E1 地圖、E2 歸因、E3 策略評估各一份。
- `common/cost_model/params.json` 新增 `flux_dispatch` 區：每層 M\*、M < 1024 的 Flux AG 與 NCCL AG 延遲
  （新錨點；Phase 0 沒有這段）。
- **GPU 時間不是瓶頸**：每層 20 個 M × 200 輪，小 M 每輪數 ms，最大點（L-GU, M=16384）每輪約 60 ms，
  每層 < 2 分鐘。不確定的是 `profiling()` 的耗時，E0 先量一點。
  主要成本是 harness 的正確性與 5.1 的對齊 / flush 驗證。

## 11. 與其他 workstream 的關係

- **diag-overlap**：
  - 3.1 附帶發現（N=4096 跑 fallback config）要轉給他們。
  - E1 在 P0-4096 的結果直接回答：在這個形狀「修好 overlap」與「乾脆關掉」哪個划算。
  - `src/ag_gemm/` 的改動需協調。
- **cost-model**：E1 地圖是它的驗證集；π_model 是它的第一個消費者。
  它的成功標準目前只涵蓋 N ∈ {2048, 4096, 8192}，之後可以擴充到 M 維度。
- **hetero-proxy**：決策器最終會長成「這次 forward 走哪條路徑 / 哪個裝置」的 dispatcher。
  異質後端出現時，arm 表多一列 PCIe 代理路徑，同一個 harness 沿用。
