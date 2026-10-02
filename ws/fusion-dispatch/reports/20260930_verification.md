Supersedes: （無）

# 驗證：把能確認的事全部確認

日期：2026-09-30（完成於 2026-10-02） · 作者：fusion-dispatch（boss session 兼 worker）· **尚未經 auditor 審查**

背景：使用者要求在往下一步（F2c）之前，先把先前報告中「未驗證 / 推論」的部分能確認的都確認。
本報告逐項列出做法、原始輸出與結論。

## 0. 總表

| # | 問題 | 結論 | 狀態 |
| --- | --- | --- | --- |
| V1 | vLLM 0.8.5 實際怎麼通訊、何時用 graph | 讀原始碼確認，見第 1 節 | ✅ |
| V3 | 換成 vLLM 真正的 all-reduce，決策器還贏嗎 | decode M ≤ 384 選 vLLM 預設（同速），M=512 快 10.1%，prefill 快 9.8–17.7%，regret 0.00% | ✅ 結論不變 |
| V3b | 勝利是不是只因 vLLM 的 8 MiB 門檻 | 門檻調到 64 MiB 後仍勝：decode M=512 快 2.3%、prefill 快 5.1–10.2%（皆確定勝）；M=512 的 10% 大多來自門檻 | ✅ 勝幅縮小 |
| V4 / V4b | 4 個 block 能否推到 80 個 | 可：決策不變、regret 0.00–0.02%；decode M=512 快 13.6%，prefill 快 9.3–12.0%。**長 graph 中 c10d NCCL 每次變貴**（SP 全 NCCL 達 1.16–1.57×） | ✅ |
| V5 | graph 部署是否需要用 graph 重放時間建表 | 不需要：28 個可比點全部一致或為平手 | ✅ |
| V6 | Phase 0「Flux AG 比 NCCL 快 13–22%」與本 ws 的差異 | 兩者都對。Phase 0 量 NCCL 時 GPU 在 1155 MHz，本 ws 在 1410 MHz；NCCL 用 SM 搬資料，速度隨時脈變，Flux 用 copy engine 不受影響 **[推論，時間對應]** | ✅ 找到原因 |
| V7 | 保留輪次 < 200 的點 | AG 側 160 點全部 ≥ 200；RS 側剩 3 點（功耗上限），皆大幅勝負。判決僅 1 點從 fused 變平手（G-FC2 16384 steady） | ✅ |
| V8 | 其餘三層的 nsys | 四個 AG 層在 M ≤ 512 完全無重疊，≥ 3072 才有效（44–87%） | ✅ |
| V9 / V9b | Triton AllGather 原型在 rank 不同步時是否正確 | **單一 buffer 會讀錯**（最多 33,612 / 40,000 個 (rank, iter) 錯）；**雙 buffer 全部 0 錯** | ✅ 設計必須用雙 buffer |
| V10 | 不改碼能估出的調校收益 | AG 側只有 M ≥ 2048 有收益；RS 側 profiling 協定不可靠 | ✅ AG；❌ RS 不改碼無法確認 |
| V12 | 量測期間是否只有我們在用機器 | 9/29 起無其他使用者登入；守衛下重測 5 組關鍵點全部 CLEAN，與 9/30 差 ≤ 2%，判決相同 | ✅ 間接確認；之後一律用守衛 |
| V11 | V4 中的 NCCL-in-graph 卡死 | 最小重現（6 個 graph × 320 次 collective，c10d 與 pynccl）**未重現**；每個 M 用獨立 process 時也未再發生 | ⚠️ 原因未定 |

## 1. V1：vLLM 0.8.5.post1 的實際行為（讀原始碼）

安裝位置：獨立 venv `/home/rogerlee/venvs/vllm085-flux`，使用 `--system-site-packages`，以取得 Flux。
torch 2.6.0+cu124 與 NCCL 2.21.5 皆與 pixi 環境同版。啟動方式見 `scripts/launch_vllm_env.sh`。

| 項目 | 事實 | 出處（venv 內 `vllm/`） |
| --- | --- | --- |
| row-parallel 後的通訊 | `tensor_model_parallel_all_reduce` | `model_executor/layers/linear.py:1286-1287` |
| all-reduce 選擇 | 先試 custom all-reduce，否則 PyNccl | `distributed/device_communicators/cuda_communicator.py:52-71` |
| custom all-reduce 條件 | 位元組數 < `max_size`（預設 8 MiB）、16 的倍數、world 為 2/4/6/8 且全 NVLink | `custom_all_reduce.py:50-56, 213-225` |
| 換算到 Llama-3-70B | 每 token 16 KB（8192 × bf16）→ **M < 512 用 custom AR，M ≥ 512 用 NCCL** | — |
| 預設引擎 | V1（`VLLM_USE_V1=1`） | `envs.py:530-531` |
| CUDA graph 桶 | V1 預設 `[1, 2, 4] + range(8, 513, 8)`，即 **token 數 ≤ 512**，且不超過 `max_num_batched_tokens` | `config.py:4032-4045` |
| graph 形式 | piecewise：在 `vllm.unified_attention(_with_output)` 處切段，attention 不在 graph 內 | `config.py` `set_splitting_ops_for_v1` |
| 序列平行 | `PassConfig.enable_sequence_parallelism`，**預設 False**；開啟時把 AllReduce + RMSNorm 改寫成 ReduceScatter + RMSNorm + AllGather | `config.py:3444-3450`；`compilation/sequence_parallelism.py` |

→ D-007 中「vLLM 預設 TP 為 AllReduce」的推論**成立**。

F1 的 `tp_ar` 用 torch NCCL all-reduce，所以在 decode 小 M **低估了 vLLM**（見 V3）。
我們的 graph 測試把 attention 也錄進 graph，與 vLLM 的 piecewise 不同。因為 attention 在所有做法都相同，所以不影響差值 **[推論]**。

## 2. V3：真正的 vLLM all-reduce（4 個 block）

- 腳本：`scripts/validate_block_v3.py`（v2 + 政策 `tp_ar_vllm`）、`scripts/run_v3_block.sh`、`scripts/analyze_block_v2.py`。
- 原始輸出：`results/v3_block_vllm_ar/{cal,eval}/`、`block_summary.csv`、`analysis_log.txt`。
- `tp_ar_vllm` 用 vLLM 自己的 `init_distributed_environment` 與 `initialize_model_parallel` 建立 TP communicator，
  graph 在 `vllm.distributed.parallel_state.graph_capture()` 內錄製，會登記 custom AR 的 graph buffer，和 vLLM 相同。
- 啟動時確認：custom AR 啟用（`fully_connected=True`，`max_size=8388608`），PyNccl 啟用。

驗證 run（seed 20261001），ms：

| 階段 / 模式 | M | torch-NCCL TP | **vLLM 預設** | SP 決策器 | 兩層決策器選 | 比 vLLM 預設 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| decode / graph | 8 | 1.046 | **0.941** | 1.130 | vLLM 預設 | 0.0% |
| decode / graph | 64 | 1.264 | **1.128** | 1.426 | vLLM 預設 | 0.0% |
| decode / graph | 256 | 2.615 | **2.324** | 2.639 | vLLM 預設 | 0.0% |
| decode / graph | 384 | 3.173 | **2.857** | 3.640 | vLLM 預設 | 0.0% |
| decode / graph | 512 | 4.558 | 4.564 | **4.104** | SP 決策器 | **+10.1%** |
| prefill / eager | 1024 | 6.454 | 6.413 | **5.785** | SP 決策器 | **+9.8%** |
| prefill / eager | 2048 | 12.105 | 12.096 | **10.448** | SP 決策器 | **+13.6%** |
| prefill / eager | 4096 | 22.811 | 22.709 | **19.054** | SP 決策器 | **+16.1%** |
| prefill / eager | 8192 | 44.845 | 44.878 | **36.938** | SP 決策器 | **+17.7%** |

完整表（含 16 / 32 / 128 與 eager decode）見 `analysis_log.txt`。

1. **vLLM 的 custom AR 在 decode 比 torch NCCL 快 10–12%**（M ≤ 384）。M=512 起 vLLM 本身就用 NCCL，兩者相同。
2. **兩層決策器 regret 0.00%**（每一格）。合計比 vLLM 預設：decode graph 快 3.0%（全部來自 M=512），prefill 快 16.1%。
3. **M=384 暴露一個決策器弱點**：該 M 不在表裡，決策器借用 M=512 的選擇，使 gate_up 走 `fluxag_fluxgemm`。
   V5 在 graph 中量到這條在 M=384 是 0.415 ms，NCCL 是 0.299 ms，所以 SP 決策器 3.640 反而比 SP RS 用 Flux 的 3.202 慢。
   兩層決策器仍選 vLLM 預設，未受影響。
   → **表必須涵蓋部署的每一個 graph 桶**；「取下一個較大的桶」不是安全的退路。

## 3. V3b：vLLM custom all-reduce 門檻調到 64 MiB

- 腳本：`validate_block_v3.py --car_big_mib 64`（政策 `tp_ar_vllm_big`）。
- 原始輸出：`results/v3b_big_car/`。
- 單次 run，200 輪，seed 20261004。這是**假設性**的 vLLM 設定（預設 8 MiB），用來檢查勝利是否只來自門檻。

| 階段 | M | SP 決策器 | vLLM 預設 | vLLM 64 MiB | 比預設 | 比 64 MiB | 逐輪差 p10 / p90（vs 64 MiB） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode graph | 256 | 2.641 | 2.321 | 2.317 | −13.8% | −14.0% | （vLLM 勝） |
| decode graph | 384 | 3.638 | 2.849 | 2.847 | −27.7% | −27.8% | （vLLM 勝） |
| decode graph | 512 | **4.088** | 4.561 | 4.185 | +10.4% | **+2.3%** | −0.147 / −0.033 |
| prefill eager | 1024 | **5.762** | 6.393 | 6.075 | +9.9% | **+5.1%** | −0.399 / −0.226 |
| prefill eager | 2048 | **10.401** | 12.024 | 11.589 | +13.5% | **+10.2%** | −1.323 / −1.102 |

→ 勝利不只來自門檻，但 decode M=512 的大部分（10.4 → 2.3%）來自門檻。prefill 的收益大多保留。

## 4. V4 / V4b：80 個 block

- V4 prefill 與 decode M=64 在同一個 process；V4b 為 decode 每個 M 各一個 process（原因見 V11）。
- 腳本：`validate_block_v3.py --L 80`。
- 原始輸出：`results/v4_block_L80/eval/`（decode 為四個 `eval_M*` 合併）、`analysis_log.txt`。
- 切法的選擇沿用 **4 個 block 的校準 run**（`cal` 複製自 V3），在 80 個 block 上評估。

| 階段 | M | vLLM 預設 | SP 全 NCCL | SP 決策器 | 兩層決策器選 | 比 vLLM 預設 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| decode graph | 64 | **22.41** | 44.71 | 44.99 | vLLM 預設 | 0.0% |
| decode graph | 256 | **45.82** | 72.74 | 56.38 | vLLM 預設 | 0.0% |
| decode graph | 384 | **57.73** | 84.03 | 72.69 | vLLM 預設 | 0.0% |
| decode graph | 512 | 95.43 | 109.08 | **82.44** | SP 決策器 | **+13.6%** |
| prefill eager | 1024 | 129.79 | 133.88 | **117.74** | SP 決策器 | **+9.3%** |
| prefill eager | 4096 | 449.67 | 432.95 | **395.60** | SP 決策器 | **+12.0%** |

regret：decode 0.00%、prefill 0.02%。合計比 vLLM 預設：decode 快 5.9%，prefill 快 11.4%（比永遠 Flux 快 7.2%）。

線性度，即 80 個 block ÷（20 × 4 個 block）：

- vLLM 預設（custom AR）0.99–1.01（M=512 為 1.045，該 M 用 NCCL）；
- **SP 全 NCCL 1.16–1.57、torch-NCCL TP 1.04–1.15**，NCCL collective 越多越超線性；
- SP 決策器在 NCCL 少的 M=384 / 512 為 1.00；
- prefill 的 Flux 重度做法 1.04–1.06。

→ 4 block 的**決策**可推到 80 block。**[推論]** 長 graph 中 torch c10d NCCL 的每次成本上升，而 vLLM 的 custom AR 不受影響。
這讓小 M 的 vLLM 預設更占優，也表示 SP 在 decode 若要競爭，需要非 NCCL 的 AllGather（例如 Triton，見 V9）。
prefill 的 Flux 重度做法略為超線性，推測與長時間滿載的功耗 / 時脈有關 **[推論]**。

## 5. V5：graph 重放建表 vs gpu 模式建表

- 腳本：`scripts/op_graph_map_v1.py`。
- 原始輸出：`results/v5_op_graph/raw_graph.csv`、`compare_table.csv`、`results/v5_op_graph_log.txt`。
- 方法：Llama 四層 × M ∈ {8, 16, 32, 64, 128, 256, 384, 512}。每條可 capture 的路徑各錄一份 graph，200 輪交錯重放，清 L2、GPU 端對齊，逐輪 rank-max。

結果：32 個比較點中，M=384 的 4 點不在表內。其餘 **28 點中，24 點 graph 最佳路徑與表相同，4 點在 graph 下為平手**：
L-O M=32 / 64 / 128 與 L-down M=128，差 0–5 µs。**沒有任何一點是 graph 明確勝出、但表選錯**。

→ 用 gpu 模式建的表部署到 graph 是安全的。F0.4 看到的「graph 對 NCCL 多 7–15 µs」只影響平手區。

## 6. V6：Phase 0 的 NCCL 數字

- 原腳本重跑：`experiments/2026-09-23-phase0-a100-nvlink-pcie/scripts/flux_comm_baseline.py` 原封不動，每個 M 3 次。
  輸出寫到 `results/v6_phase0_repro/`，沒有動 Phase 0 檔案庫。

| M | NCCL（3 次） | Flux All2All pull（3 次） | NCCL / Flux | Phase 0 E.1 |
| ---: | --- | --- | ---: | --- |
| 1024 | 0.227 / 0.256 / 0.222 | 0.199 / 0.197 / 0.203 | 1.14 | 0.224 / 0.199 |
| 4096 | 0.564 / 0.569 / 0.565 | 0.472 / 0.476 / 0.471 | 1.20 | 0.569 / 0.468 |
| 16384 | 1.921 / 1.922 / 1.927 | 1.572 / 1.594 / 1.563 | 1.22 | 1.928 / 1.668 |

**Phase 0 的數字完全重現**：機器與軟體沒有變。本 ws 的 gpu 模式量到 NCCL 0.478、host 模式 0.509，差異來自量測協定。
哪一個協定要素造成差異，見下方 A/B 與 V6c。

## 6.（續）V6 A/B 與 V6c

**A/B**（`scripts/v6_protocol_ab.py`，`results/v6_phase0_repro/protocol_ab.json`）：同一輪內交錯 4 種協定 × 2 種 op，fp16，M=4096：

| 協定 | NCCL（Phase 0 統計 / 本 ws 統計） | Flux（同） |
| --- | --- | --- |
| P0：barrier → 計時 | 0.498 / 0.502 | 0.451 / 0.453 |
| P0 + L2 flush | 0.499 / 0.502 | 0.453 / 0.455 |
| host | 0.499 / 0.503 | 0.453 / 0.455 |
| gpu 對齊 | 0.472 / 0.475 | 0.441 / 0.444 |

- 兩種統計方式差 < 1%；L2 flush 沒有影響；host barrier 比 GPU 對齊多約 5%。
- **但交錯時連 P0 協定也只有 0.502，不是原腳本的 0.565**。所以差異主要不在協定，而在原腳本讓 NCCL **單獨、輕負載**地跑。

**V6c**（`results/v6_phase0_repro/v6c_clock.csv`、`v6c_marks.txt`、`v6c_M4096_rep*.csv`）：原腳本執行期間以 50 ms 記錄 GPU0 的 SM 時脈。

- 原腳本重跑兩次：NCCL 0.566 / 0.564、Flux 0.467 / 0.469，仍完全重現。
- 腳本先量 NCCL：GPU 從閒置的 210 MHz 起步，量 NCCL 的幾秒內（rep1 14:21:24–27、rep2 14:21:40–44）**SM 時脈停在 1155 MHz**，之後量 Flux 時才在 1155 / 1410 之間切換。
- 換算：0.565 × 1155 / 1410 = 0.463 ms，與本 ws 在 1410 MHz 下的 NCCL 0.475 ms 一致；Flux 的 copy engine 傳輸與 SM 時脈無關（兩邊都 ≈ 0.47）。

結論 **[推論，時間對應強；無 sudo 無法鎖時脈直接驗證]**：

- **兩邊的量測都對**，對應不同的 GPU 時脈狀態。Phase 0 的「Flux AG 比 NCCL 快 13–22%」在 SM ≈ 1155 MHz 時成立，在 1410 MHz 時不成立。
- 長時間滿載（功耗上限 1140–1245 MHz）更接近前者。
- → **決策表應在部署的時脈狀態下校準**。

## 7. V7：保留輪次 < 200 的點

- 腳本：`run_verify_resume.sh`（V7 段）。
- 原始輸出：`results/v7_rerun/{ag,rs}/`。
- 合併：`scripts/merge_points_v2.py`，後者覆蓋前者。AG 依序為 e1_map_v2 → e1_rerun_v2 → v7；RS 依序為 e5_rs_map_v1 → v7。
  輸出 `results/final_ag_points.csv`、`results/final_rs_points.csv`。
- 輪數依原保留率推算，最多 2000 輪；M=8 的 warmup 加到 100。

結果：

- **AG**：160 點全部 ≥ 200 輪保留。判決相對最初來源變化 2 點，都是先前近邊界重跑就已知的平手 ↔ 小勝，V7 未帶來新變化。
- **RS**：還剩 3 點 < 200：G-O 8192 steady 119、L-O 16384 steady 147、L-down 3072 steady 184。功耗上限下時脈起伏，2000 輪仍不足；三點皆為 Flux 大幅勝出。
  判決變化 1 點：G-FC2 16384 steady 從 fused 變平手（中位數仍 Flux 快 0.26 ms，p90 越過 0）。皆不在 Llama 決策範圍內。

## 8. V8：其餘三層的 nsys 時間軸

- 腳本：`scripts/nsys_probe_v2.py`（v2：各項目逐輪**交錯**、隨機順序；v1 分塊執行，含 clock 漂移）、
  `scripts/run_v8_nsys.sh`、`scripts/nsys_parse_v1.py`。
- 原始輸出：`results/v8_nsys/*.nsys-rep`、`parse_all.txt`（sqlite 已刪，可由 nsys-rep 重建）。

交錯的效果：同一 GEMM kernel 在串行 D 與單獨 c_fluxgemm 中的長度一致，例如 G-QKV M=2048 為 1014.7 vs 1014.7 µs、
L-GU M=4096 為 2492.6 vs 2499.9 µs。v1 的跨項漂移問題已排除。

被藏住的通訊 = 單獨 GEMM 時間 − fused GEMM 在最後一筆 shard copy 之後還跑的時間（device 0，20 次中位數，µs）：

| 層 | M | shard copy 起訖 | 單獨 GEMM | fused 在最後 copy 後 | **被藏住** | E1 判決 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| G-QKV | 64 | 116.9 | 96.3 | 96.8 | **≈ 0** | off |
| G-QKV | 2048 | 311.3 | 1014.7 | 1054.0 | **−39**（fused 反而慢） | off |
| G-QKV | 4096 | 520.3 | 1929.4 | 1550.0 | **379（73%）** | fused |
| L-QKV | 64 | 111.4 | 74.5 | 73.9 | **≈ 0** | off |
| L-QKV | 2048 | 270.0 | 218.2 | 211.1 | **7** | 平手 |
| L-QKV | 4096 | 383.3 | 555.2 | 235.3 | **320（84%）** | fused |
| L-GU | 512 | 159.8 | 295.7 | 296.4 | **≈ 0** | 平手 |
| L-GU | 3072 | 337.7 | 1648.3 | 1431.4 | **217（64%）** | fused |
| L-GU | 4096 | 399.2 | 2499.9 | 2152.7 | **347（87%）** | off（GEMM config 差） |
| G-FC1 | 4096 | 521.3 | 2218.9 | 1987.4 | **231（44%）** | fused |

結論：

- **四個 AG 層在 M ≤ 512 完全沒有重疊**，M=2048 也幾乎沒有；重疊要到 M ≥ 3072 才有效（44–87%）。
  與 design 3.2 的結構預測一致。
- **L-GU M=4096 的斷崖不是重疊失效**：通訊藏住了 87%，但 Flux GEMM 本身比 cuBLAS 慢 0.67 ms（E1 分解 D − C），
  屬於 config 問題，與 V10 的 0.71 比值一致。

## 9. V9 / V9b：Triton AllGather 原型在 rank 不同步時的正確性

- 腳本：`scripts/triton_ag_stress_v1.py`、`triton_ag_stress_v2.py`。
- 原始輸出：`results/v9_triton_stress/*.json`、`log_*.txt`。
- 每輪各 rank 隨機延遲，寫入新值，跨卡 barrier，P2P gather，然後在 GPU 上比對期望值。每項 5000 輪 × 8 卡 = 40,000 個 (rank, iter)。

| 延遲位置 | M | 最大延遲 | 單一 buffer 錯誤的 (rank, iter) | 雙 buffer |
| --- | ---: | ---: | ---: | ---: |
| 寫入前（v1） | 64 | 200 µs | 0 | 0 |
| 寫入前（v1） | 64 | 2 ms | 0 | 0 |
| 寫入前（v1） | 1024 | 200 µs | 0 | 0 |
| **barrier 後、gather 前（v2）** | 64 | 200 µs | **22,142** | **0** |
| **barrier 後、gather 前（v2）** | 64 | 2 ms | **33,612** | **0** |
| **兩處（v2）** | 1024 | 500 µs | **439** | **0** |

- v1 把延遲放在寫入前，沒有打開真正危險的時間窗：慢的卡已通過 barrier、還沒讀完，快的卡已寫入下一輪。所以 v1 的「單一 buffer 0 錯」**不能當作安全證據**。
- v2 打開這個時間窗後，**單一 buffer 大量讀錯，雙 buffer 全部正確**。
- → 正式版的 Triton AllGather **必須用雙 buffer**（依 epoch 奇偶交替），或在 gather 後加第二次 barrier（多一次同步延遲）。

## 10. V10：不改碼的調校收益估計（`AGKernel.profiling()` / `GemmRS.profiling()`）

- 腳本：`scripts/tune_upside_v2.py`（v1 + GemmRS）。
- 原始輸出：`results/v10_tune/tune_*.json`、`log_*.txt`。
- 最佳 / 預設 < 1 表示有收益。

| 層 | M=8 | 64 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L-QKV（AG） | 1.03 | 1.05 | 1.08 | 1.11 | 1.01 | 1.05 | **0.89** | **0.92** |
| L-GU（AG） | 1.07 | 0.88 * | 1.05 | 1.06 | 0.96 | **0.86** | **0.71** | 0.98 |
| L-O（RS） | 0.99 | 0.73 † | 0.79 † | 0.79 † | 1.04 | 1.18 | 1.05 | 1.07 |
| L-down（RS） | 1.03 | 0.94 † | 1.07 | 1.09 | 1.07 | 1.01 | 0.87 † | 0.87 † |

- 比值 > 1：profiling 每個 config 之間 `sleep(1)`，GPU 降頻，最佳值被高估約 5%。所以 1.0–1.1 可視為「無收益」。
- \* **L-GU M=64 的預設時間可疑**：連發量到 0.354 ms，但 E1 交錯量測只有 0.204 ms。
- † **RS 側結果不可用**：
  - GemmRS 的「預設」連發時間與 E5 的交錯量測嚴重不符，例如 L-O M=64 為 0.246 vs 0.073 ms，差 3.4×；
  - 而且不單調：M=64 比 M=512 還慢，物理上不可能。
  - **[推論]** GemmRS 內部有跨 rank 同步，無 GPU 端對齊的連發會量進等待時間。
  - GemmRS 的 profiling 只花 6 秒（AG 側約 65 秒），可選 config 很少。

結論：

- **AG 側**：小 M（≤ 512）沒有調校空間，與 E2a 一致；收益集中在 M ≥ 2048（8–29%）。
- **RS 側**：必須能從 Python 指定 hparams，才能用交錯協定逐一量測，需要改 pybind（F3b）。
  → **不改碼無法確認**，這是本報告中唯一「做不到」的項目。

## 11. V11：NCCL-in-graph 卡死

- 現象（V4，2026-09-30）：80 個 block、6 種做法各 capture 一份 graph（每份含數百個 c10d NCCL collective）、重放、刪除後，
  下一個 M 的 eager `all_gather_into_tensor` 在**所有 rank** 上卡住，GPU idle（py-spy 確認 8 個 rank 都在 `distributed_c10d.all_gather_into_tensor`）。
- 最小重現（`scripts/nccl_graph_hang_v1.py`，`results/v11_nccl_graph/log_{c10d,pynccl}.txt`）：
  - 每個 M 錄 6 個 graph × 320 次 all_gather，重放 5 次、刪除，再做一次 eager all_gather；
  - M ∈ {64, 256, 384, 512}；
  - **c10d 與 vLLM pynccl 都正常（ALL OK）**。
- V4b 改為每個 M 一個 process 後，未再發生。
- 結論：**觸發條件不是「NCCL 錄進 graph 再刪除」這麼單純**。可能需要 Flux op、vLLM 初始化（另建多個 NCCL / gloo group 與 custom AR）、多個 graph 交錯同時存在。
  **原因未定**。部署前需要在整合後的環境做長時間壓力測試（F4）。

## 12. V12：量測時的機器獨佔性（使用者 2026-10-02 提醒）

**事後能查到的**：

- `last`：2026-09-29 以來只有 rogerlee 一個帳號登入過，沒有其他使用者。
- 現在除系統服務（root、systemd 等）外，沒有其他使用者的程序；GPU 上無任何程序。
- 9/29 22:00 – 9/30 10:56 的所有量測都在同一次開機內（9/23 開機）。
  10/01 16:53 機器重開機，核心 6.8.0-142 → 6.8.0-146，GPU 驅動不變（615.71.09）。
  V6c、V9b 在重開機後執行；V6c 與 9/30 的原腳本重跑一致（NCCL 0.564–0.566）。
- 量測期間有幾次快照確認 GPU 上只有我們的程序，例如每次停 / 啟程序前後的 `nvidia-smi --query-compute-apps`。
  **但沒有持續監控**，所以不能事後證明每一輪都沒有外來 GPU 負載。

**間接證據**：同一點在不同時間、不同 run 的重現性。

- F1 / V3 的校準 run 與驗證 run（不同 seed、不同時間）各政策中位數相差約 1% 內。
- E1 的近邊界點在三次 run 中判決一致，除了已列出的平手互換。
- Phase 0 原腳本在 9/30 與 10/02 兩天各次重跑，NCCL 0.564–0.569，非常穩定。

**之後的做法**：`common/measure/exclusive_guard.py` 包住每個量測指令。

- 啟動前：GPU 上有任何程序，或其他使用者佔用 CPU，就中止（exit 3）；
- 執行中：每秒記錄不屬於這個量測的 GPU 程序與其他使用者的 CPU 使用；
- 結束時：標記 CLEAN / CONTAMINATED 與時間窗。
- 已自我測試：外部佔 GPU 時，preflight 會中止；量測中出現時會標記 CONTAMINATED。

**V12 重測**（`scripts/run_v12_exclusive_recheck.sh`，`results/v12_exclusive_recheck/`）：在守衛下重量 9/30 的關鍵點
（G-FC1 M=64 / 4096 單層；Llama 4 block decode graph M=64 / 512、prefill M=1024），與原數字比較。結果見下方。

| 量測 | 9/30 | 10/02（守衛下，CLEAN） | 差 |
| --- | ---: | ---: | ---: |
| G-FC1 M=64 gpu：fused / NCCL+cuBLAS | 0.2304 / 0.1669（E0 v2） | 0.2324 / 0.1679 | +0.9% / +0.6% |
| G-FC1 M=4096 gpu：fused / NCCL+cuBLAS | 2.5477 / 2.9875（E0 v2） | 2.5288 / 2.9706 | −0.7% / −0.6% |
| block decode graph M=64：vLLM 預設 / SP 決策器 | 1.1284 / 1.4264（V3 eval） | 1.1331 / 1.4321 | +0.4% / +0.4% |
| block decode graph M=512：vLLM 預設 / SP 決策器 | 4.564 / 4.104 | 4.565 / 4.092 | 0.0% / −0.3% |
| block prefill M=1024：vLLM 預設 / SP 決策器 | 6.413 / 5.785 | 6.399 / 5.756 | −0.2% / −0.5% |

守衛紀錄：`results/v12_exclusive_recheck/guard_*.log`。三次量測全程 0 個外來 GPU / CPU 樣本，結束時無其他使用者登入。
→ 9/30 的數據在獨佔條件下可重現，判決全部相同。**建議 boss 把「量測一律經 `exclusive_guard.py`」寫進 `CLAUDE.md` 5.1。**
