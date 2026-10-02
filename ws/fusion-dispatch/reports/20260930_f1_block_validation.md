Supersedes: （無）

# F1：決策器 v1 在真實 block 裡的效果

日期：2026-09-30 · 作者：fusion-dispatch（boss session 兼 worker）· **尚未經 auditor 審查**

| 項目 | 路徑 |
| --- | --- |
| 決策器 | `scripts/dispatcher_v1.py` |
| 建表 | `scripts/build_table_v1.py` |
| 驗證 | `scripts/validate_block_v2.py`（v1 已撤回，見第 4 節） |
| 分析 | `scripts/analyze_block_v1.py` |
| 驅動腳本 | `scripts/run_f1_block_v2.sh` |
| 原始輸出 | `results/f1_block_v2/{cal,eval}/raw_*.csv`、`meta_*.json`、`table_{steady,gpu}.json` |
| 摘要 | `block_summary.csv`、`table_check.csv`、`analysis_log.txt` |

決策表的來源：

- AG 側：E1 合併點（`results/e1_merged_points.csv`，L-QKV、L-GU）；
- RS 側：E5（`results/e5_rs_map_v1/`，L-O、L-down）。

eager 用 steady 模式的表，graph 用 gpu 模式的表。

---

## 0. 結論

1. **決策器需要兩層決定。**
   - 第一層是**切法**：vLLM 預設的「每卡完整 token + AllReduce」，或序列平行（SP）。
   - 第二層是 SP 下**每一層選哪條路徑**。
   - decode 的小 M 時，vLLM 預設每層只做 2 次通訊，SP 要 4 次，所以 vLLM 預設最快，Flux 開不開都一樣。
2. **兩層決策器在驗證 run（新 seed、新輪次）上從未選錯**：
   - regret：decode graph 0.00%、prefill 0.02%；
   - 切法的選擇完全由校準 run 決定。
3. **收益**（v2，公平的 fused RMSNorm / SiLU×up）：
   - **prefill（M = 1024–8192）比 vLLM 預設快 10.4–18.3%**，四個桶合計 16.6%；比永遠開 Flux 快 2.1%，其中 M=4096 快 7.6%。
   - **decode（CUDA graph，M = 8–512）**：M ≤ 256 選 vLLM 預設，同速；M=512 快 10.0%。七個桶合計 3.4%。
4. **決策器在 SP 下做出的逐層組合，是單一開關做不到的**：
   - decode M=256–512、prefill M=1024：QKV 用 NCCL、gate_up 用 Flux AG + Flux GEMM 串行、O / down 用 Flux GemmRS；
   - prefill M=4096：QKV 用 Flux fused、gate_up 用 Flux AG + cuBLAS。
5. **單層量出的表能預測 block 裡的差距**，只在 eager 的小 M decode 失準：

   | 情境 | 中位相對誤差 | 誤差 < 10% | 正負號錯 |
   | --- | ---: | ---: | ---: |
   | prefill（eager） | 5.5% | 10/12 | 0 |
   | decode（graph） | 9.0% | 5/8 | 0 |
   | decode（eager） | 118% | 4/16 | 9 |

   eager 小 M 失準的原因是整個 block 被 CPU launch 卡住，而 Flux 的 C++ op 比「NCCL + torch.mm」的 host 開銷低。
   單層量測時看不到這個差異。vLLM 的 decode 走 graph，不受影響。

## 1. 方法

- **模型**：4 個 Llama-3-70B block（TP=8、bf16），每個 block 的 weight 不同：
  RMSNorm → QKV → attention → O → 殘差 → RMSNorm → gate_up → SiLU×up → down → 殘差。
  - attention 用 torch SDPA（GQA，每卡 8 個 q head、1 個 kv head）；
  - decode：M 條序列各一個新 token，每個 block 各有長 1024 的 KV cache；
  - prefill：一條長 M 的序列，causal。
- **五種做法**，同一輪內隨機順序交錯，每項前清 L2、GPU 端對齊，CUDA event 包住整個 4 block，逐輪 rank-max，200 輪：

  | 做法 | 內容 |
  | --- | --- |
  | `tp_ar` | vLLM 預設 TP：每卡全部 M 個 token，row-parallel 後 NCCL all_reduce |
  | `sp_nccl` | SP，全部 NCCL + cuBLAS |
  | `sp_flux` | SP，全部 Flux 融合 op |
  | `sp_rsflux` | SP，AG 用 NCCL、RS 用 Flux |
  | `sp_dispatch` | SP，每層查表（決策器） |

- **graph 模式**：每種做法每個 M 各 capture 一份再重放。`sp_flux` 不能 capture，因為 AGKernel 不行（F0.4）；
  `sp_dispatch` 以 `graph_mode=True` 跳過不能 capture 的路徑。
- **正確性**：每種 SP 做法的輸出與 `tp_ar` 對應列的相對誤差都在 1.5–3.0%（bf16、4 個 block、不同 GEMM 與 reduce 順序），門檻 5%。
- **兩層決策器 "auto" 的評估方式**：
  - 校準 run（seed 20260930）決定每個 M 用 `tp_ar` 或 `sp_dispatch`；
  - 在驗證 run（seed 20261001，重新 capture、重新計時）上評估。
  - oracle = 驗證 run 中五種做法最快者。

## 2. 結果（驗證 run，ms，4 個 block）

| 階段 / 模式 | M | vLLM 預設 | SP 全 NCCL | SP 全 Flux | SP RS 用 Flux | SP 決策器 | 兩層決策器選 | vs vLLM 預設 | vs 永遠 Flux |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| decode / graph | 8 | 1.040 | 1.131 | — | 1.263 | 1.128 | vLLM 預設 | +0.0% | — |
| decode / graph | 16 | 1.063 | 1.160 | — | 1.282 | 1.163 | vLLM 預設 | +0.0% | — |
| decode / graph | 32 | 1.209 | 1.326 | — | 1.397 | 1.325 | vLLM 預設 | +0.0% | — |
| decode / graph | 64 | 1.282 | 1.438 | — | 1.528 | 1.438 | vLLM 預設 | +0.0% | — |
| decode / graph | 128 | 1.683 | 1.826 | — | 1.846 | 1.825 | vLLM 預設 | +0.0% | — |
| decode / graph | 256 | 2.618 | 2.837 | — | 2.642 | 2.647 | vLLM 預設 | +0.0% | — |
| decode / graph | 512 | 4.560 | 4.725 | — | 4.179 | **4.102** | SP 決策器 | **+10.0%** | — |
| prefill / eager | 1024 | 6.448 | 6.651 | 5.791 | 5.901 | **5.780** | SP 決策器 | **+10.4%** | +0.2% |
| prefill / eager | 2048 | 12.163 | 11.966 | 10.467 | 10.788 | **10.464** | SP 決策器 | **+14.0%** | +0.0% |
| prefill / eager | 4096 | 22.822 | 22.073 | 20.620 | 20.106 | **19.059** | SP 決策器 | **+16.5%** | **+7.6%** |
| prefill / eager | 8192 | 45.388 | 43.470 | 37.065 | 40.142 | **37.080** | SP 決策器 | **+18.3%** | −0.0% |

「vs」欄為兩層決策器比該做法少用的時間比例。eager 模式的 decode 見 `analysis_log.txt`：兩層決策器在 M ≤ 256 選 vLLM 預設，M=512 快 9.8%。

幾點觀察：

- **SP 全 NCCL 在 prefill 與 vLLM 預設相近**（±5%）。所以 prefill 的收益主要來自 Flux 的融合 op 加上逐層選擇，不是切法本身。
- **decode graph M ≤ 256 時 SP 比 vLLM 預設慢 1–12%**：每層多兩次延遲主導的通訊。
- **M=512 起 Flux GemmRS 的收益大於多出來的通訊次數**。

## 3. 對照計劃的關卡 G1

| 標準 | 結果 |
| --- | --- |
| decode 為主：比永遠開 Flux 快；離 oracle ≤ 2% | graph：永遠開 Flux 無法 capture，不可行；eager：比永遠開 Flux 快 10.0%；regret 0.00% ✓ |
| prefill 為主：不比永遠開 Flux 慢 > 0.5% | 快 2.1% ✓ |
| 表的單格預測與 block 實測差 < 10% | graph 5/8、prefill 10/12 在 10% 內，正負號全對；**eager decode 不成立**（見 0.5） |

## 4. 撤回與限制

- **`validate_block_v1.py` 撤回**。RMSNorm 由約 6 個 PyTorch 元素運算拼成，SiLU×up 也未融合；
  `tp_ar` 每卡要對全部 M 列做 norm，被多算了時間。v1 顯示 prefill 比 vLLM 預設快 25–32%，v2 為 10–18%。
  v1 的 SP 之間比較不受影響。原始輸出保留在 `results/f1_block_v1/`。
- **只有 4 個 block**，真實模型有 80 個。層與層的時間可相加，但 L2 / 功耗狀態在長時間跑時可能不同 **[推論]**。
- **attention 是簡化版**：SDPA，不是 vLLM 的 paged attention；decode 沒有把新 token 寫進 KV cache。
  所有做法的 attention 相同，會影響總時間的分母，不影響差值。
- **`tp_ar` 用 NCCL all_reduce**。vLLM 在小訊息用自己的 custom all-reduce，通常比 NCCL 快，
  所以 **vLLM 預設在 decode 的實際優勢可能更大** **[推論]**。F4 在 vLLM 內實測。
- **殘差加法未與 RMSNorm 融合**（vLLM 會融合）。影響 `tp_ar` 較多，因為它對全部 M 列做，但比 norm 本身小。
- **E5 只完成 Llama 兩層**（L-O、L-down），GPT-3 兩層（G-O、G-FC2）暫停，待補。
