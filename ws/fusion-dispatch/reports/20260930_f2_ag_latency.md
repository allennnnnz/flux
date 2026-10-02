Supersedes: （無）

# F2a / F2b：Flux AllGather 固定成本能不能降下來

日期：2026-09-30 · 作者：fusion-dispatch（boss session 兼 worker）· **尚未經 auditor 審查**

| 項目 | 路徑 |
| --- | --- |
| 腳本 | `scripts/ag_latency_v1.py`、`scripts/run_f2.sh` |
| 原始輸出 | `results/f2_ag_latency_v1/raw_K{8192,12288}.csv`、`meta_K*.json`、`log_K*.txt` |
| smoke | `results/f2_smoke/` |

方法：沿用 `dispatch_map_v2.run_mode`：

- 同輪交錯、隨機順序；
- 每項前清 L2、GPU 端對齊；
- 逐輪 rank-max，200 輪，SM clock 過濾。

bf16、TP=8；K=8192（Llama-3-70B）與 K=12288（GPT-3）；M ∈ {8, 64, 256, 512, 1024, 2048, 4096}。

正確性：每個 AllGather 項目的結果與 NCCL bitwise 相同；A_* 與 fp32 參考 allclose。

---

## 0. 結論

1. **F2a（現有開關）不可用。**
   - `use_cuda_core_local`（含 `fuse_sync`）對 bf16 直接報錯：`local_copy_and_reset.cu:232 unsupported for input_dtype=BF16`。
     它和 CUDA-core All2All（`all_gather_impls.cu:127`）一樣，只實例化了 INT8。
   - `input_buffer_copied=True`（跳過 local copy）只省 1–14 µs（K=8192，M=8–4096）。
2. **Flux AG 的 0.1 ms 裡，搬資料只佔約 40 µs，其餘約 60 µs 是同步機制。**
   同步機制包括 2 次 barrier kernel、memset、8 次 stream memop、event。
   證據：同樣 7 筆 copy 在單一 stream、不做同步（`proto_ce_1s`）只要 38 µs（M=8）。
3. **F2b 原型：單一 Triton kernel 的 AllGather 在每個 M 都比 NCCL 快。**
   原型做法：一個跨卡 flag barrier（system-scope atomic，W 個 program 平行）加上一次 P2P load 讀 7 張卡。
   - gpu 模式快 NCCL 8–22%（兩種 K、全部 M），快 Flux 1.1–4.2×；
   - M ≤ 64 時 23.6–45.1 µs，**達到關卡 G2（≤ 50 µs）**。
4. **限制**：eager 連發（steady）模式下，Triton 的 Python launch 開銷（兩次 launch）讓它在小 M 比 NCCL 慢。
   要在 eager 下受益，需寫成 C++ launch（放進 Flux，即 F2c），或只在 CUDA graph 中使用。

## 1. 結果（gpu 模式，µs，逐輪 rank-max 中位數）

### K = 8192

| M | NCCL | Flux 預設 | Flux 跳過 local copy | 7 copy / 1 stream（無同步） | 7 copy / 7 stream（無同步） | Triton gather（無同步） | **Triton 完整（平行 barrier + gather）** | Triton 完整（序列 barrier） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 29.7 | 98.7 | 96.1 | 38.3 | 50.2 | 14.3 | **23.6** | 47.1 |
| 64 | 46.1 | 105.2 | 102.5 | 46.1 | 52.2 | 31.2 | **42.0** | 63.5 |
| 256 | 73.7 | 122.1 | 121.1 | 64.0 | 75.8 | 49.2 | **59.4** | 80.9 |
| 512 | 96.3 | 147.4 | 145.1 | 89.1 | 97.8 | 70.7 | **79.9** | 102.4 |
| 1024 | 147.5 | 186.1 | 182.5 | 119.6 | 129.0 | 104.4 | **114.7** | 137.2 |
| 2048 | 218.1 | 245.5 | 236.3 | 180.1 | 191.5 | 167.9 | **178.2** | 200.7 |
| 4096 | 347.1 | 357.3 | 343.7 | 289.6 | 298.0 | 295.9 | **305.2** | 328.7 |

### K = 12288

| M | NCCL | Flux 預設 | **Triton 完整（平行 barrier）** |
| ---: | ---: | ---: | ---: |
| 8 | 29.7 | 98.8 | **24.6** |
| 64 | 50.2 | 107.8 | **45.1** |
| 256 | 82.9 | 132.1 | **71.7** |
| 512 | 114.7 | 169.4 | **98.3** |
| 1024 | 177.7 | 216.4 | **147.5** |
| 2048 | 267.3 | 302.4 | **246.3** |
| 4096 | 478.2 | 467.3 | **427.0** |

steady 模式的完整數字在 `log_K*.txt`。K=12288 M=2048 / 4096 的 steady 點 clock 過濾後只剩 61 / 2 輪，**不可引用**。

## 2. 為什麼 Triton 版比 NCCL 快，以及它還缺什麼

- 它只有一次跨卡同步，加上一次讀取 kernel。每個 program 從一張 peer 卡用 NVLink P2P load 讀一段，一次 launch 讀完 7 張卡。
- **原型省略了兩件真實版本必須有的事**：
  1. **自己的 shard 要先寫進 IPC buffer。** 最好讓上一個運算（RMSNorm）直接寫進去，否則多一次 local copy，約 2–9 µs。
  2. **barrier 的 epoch 目前由 host 傳入。** CUDA graph 重放時 host 值固定，會失效，需要改成 device 端計數器。
- **正確性限制**：barrier 用 system-scope release / acquire atomic；gather 讀取發生在 barrier 之後的另一個 kernel，stream 順序保證可見性。
  沒有測試過 8 卡之間嚴重不同步的情況 **[推論：應正確，需壓力測試]**。

## 3. 對後續的意義

- **關卡 G2 通過**（計劃第 2 節）：F2c 值得做。兩個方向：
  - **F2c-A（不改 Flux）**：決策器的 AG 側加一條「Triton AllGather + cuBLAS」路徑。
    需要 graph-safe 的 epoch，以及 producer 直接寫進 buffer。這條路在 graph 模式下每個 M 都比 NCCL 路徑快 **[推論，需 block 驗證]**。
  - **F2c-B（改 Flux）**：替 Flux 的 CUDA-core All2All AG 與 local copy 補上 BF16 / FP16 實例
    （`src/coll/all_gather_impls.cu:127`、`src/coll/local_copy_and_reset.cu:232`）。
    這樣 **fused 路徑本身**也能用上快的 AllGather，並且是 C++ launch，eager 也受益。
- **預期效果 [推論]**：
  - decode graph 的 M ≤ 256，SP 比 vLLM 預設慢，主因是每層多兩次延遲主導的通訊。
    AG 每次省 6–14 µs，每 block 兩次，約可補回 M=64 差距（每 block 39 µs）的三到七成。M=256 的 1% 差距會翻轉。
  - prefill 的收益較小：每 block 約 0.06–0.1 ms，佔 1.4–11 ms 的 1–7%。
