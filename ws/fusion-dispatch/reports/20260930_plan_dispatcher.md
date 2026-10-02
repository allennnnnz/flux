Supersedes: （無）

# 決策器與 Flux 弱點優化：執行計劃

日期：2026-09-30 · 作者：boss session · 狀態：**已核准（D-007，2026-09-30）**；目標框架 vLLM 0.8.5.post1
依據：`reports/20260930_e1_e3_dispatch_map.md`（E1–E3）、`reports/20260929_e0_anchor.md`（E0）

---

## 0. 目標與衡量方式

做出一個能在推論時**實際降低「通訊 + 計算」延遲**的決策器，並修掉 Flux 在小 M 的弱點，讓決策器有更好的選項可選。

兩件事的分工：

- **決策器本身**不會讓任何一條路徑變快。它的收益是「每次都挑最快的路徑」。
  E3 已經顯示，光這一點在 decode 就能讓 AG+GEMM 這一類操作快 24–48%（相對永遠開 Flux）。
- **修 Flux 的弱點**才會讓路徑本身變快，擴大 Flux 勝出的範圍。決策器重新校準後會自動吃到收益。

| 指標 | 定義 | 用途 |
| --- | --- | --- |
| **主指標** | 一個完整 Transformer block（AG+GEMM 與 GEMM+RS 兩側都含）在 M trace 上的總延遲 | 判斷「真的有提升」 |
| 次指標 1 | 每個通訊操作的延遲：Flux AG、NCCL AG、RS | 判斷弱點修復是否有效 |
| 次指標 2 | 決策器 regret（相對 per-bucket oracle） | 判斷決策器是否選對 |

**整體成功標準**：

1. **decode 為主的 trace**：
   - 決策器的 block 延遲比「永遠開 Flux」低，並報告降低多少。預期是 AG+GEMM 側 24–48% 的一部分，
     因為 attention 等非通訊運算不受影響 **[推論]**。
   - 決策器與 oracle 差距 ≤ 2%。
2. **prefill 為主的 trace**：決策器不比「永遠開 Flux」慢超過 0.5%。
3. **弱點修復**：
   - Flux AG 在 M ≤ 64 的延遲從 0.10 ms 降到 ≤ 0.05 ms（NCCL 水準）；**或**給出證據說明硬體下限不允許。
   - 目標模型的每個斷崖 M，fused 都不再輸給調好 config 後可達的時間。

## 1. Flux 的弱點與對策

全部來自 E1–E3 的量測。每一項都有兩種處理方式：由決策器避開，或修 Flux 本身。

| # | 弱點 | 證據 | 影響 | 決策器避開 | 修 Flux |
| --- | --- | --- | --- | --- | --- |
| W1 | **AG 固定成本 ≈ 0.1 ms**：8 筆 copy 在同一 stream 串行 + 2 次 barrier + memset | nsys M=64 每筆約 14 µs；`all_gather_op.cc:553-576` | 小 M 損失的 75–90% | ✓ 選 NCCL | **F2** |
| W2 | **GEMM config 品質**：小 M 比 cuBLAS 慢；registry / fallback 在部分 M 很差（斷崖） | D − C 為小 M 損失的 15–45%；斷崖點調校上限 25–28% | 小 M 的一部分 + 斷崖 | ✓ 選 cuBLAS 路徑 | **F3** |
| W3 | **M ≤ tile 高度時結構上不可能 overlap** | nsys M=64 計算全在最後一筆 copy 之後；A − D 只有 −0.014 ms | 小 M 時 fusion 沒有收益 | ✓ 唯一解 | 不處理（要換 fusion 設計，範圍外） |
| W4 | **fused kernel 在部分未調 M 病態變慢**（G-FC1 M=3072：資料到齊後仍多 585 µs） | nsys + E1 A − D = +0.53 | 斷崖 | ✓ | F3 避開壞 config；根因交給 diag-overlap **[推論：StreamK turnstile]** |
| W5 | **CUDA graph 相容性未知** | 未測 | 決定 graph 服務時 Flux 能不能用 | 決策器需知道 | F0 先測 |

## 2. 階段與關卡

原則（D-006）：**每個「改 Flux 程式碼」的步驟之前，都先有一個不改程式碼的量測或原型證明它值得**；通不過關卡就不改。

### F0 前置（不改程式碼）

| 步驟 | 內容 | 產出 |
| --- | --- | --- |
| F0.1 | auditor 審 E0、E1–E3 兩份報告（D-004） | `reports/YYYYMMDD_audit_*.md` |
| F0.2 | ✅ 2026-09-30 決定 vLLM 0.8.5.post1（D-007）。原內容：**決定目標框架與平行切法**：序列平行（AG / RS）或 AllReduce。若是 AllReduce，off 路徑加一條「cuBLAS + NCCL AllReduce」，比較單位改為整個 block | boss / 教授決定 |
| F0.3 | **E5：GEMM+RS 地圖**。4 層（G-FC2、G-O、L-down、L-O）× 20 M × 2 模式；`GemmRS.forward` vs `torch.mm` + `reduce_scatter_tensor` | `dispatch_map_rs_v1.py`、`results/e5_*` |
| F0.4 | **E4：CUDA graph**。A、B、C 與 GemmRS 各自能否 capture 與重放；重放時間與 gpu 模式是否一致 | 可行性表 |

**關卡 G0**：auditor 通過；框架與切法確定；graph 結果已知。

### F1 決策器 v1（不改 Flux）

先交付一個**現在就能用**的決策器，選項只有現有路徑。

| 步驟 | 內容 |
| --- | --- |
| F1.1 | `dispatcher_v1.py`：後端為 Flux fused、NCCL+cuBLAS、Flux AG+cuBLAS，以及 RS 側兩條。**8 張卡必須做出相同決策**：表由 rank 0 讀取後廣播，載入時比對 checksum。M 補齊到表中的桶。記錄 graph / eager 模式 |
| F1.2 | `calibrate_v1.py`：輸入模型的層形狀與服務的 M 桶，輸出 `table.json`，附機器、驅動、Flux commit、TP、dtype、量測模式。沿用 `dispatch_map_v2.py` 協定；平手時選 NCCL 路徑（可 capture、依賴少） |
| F1.3 | `validate_block_v1.py`：**真實 block**，Llama-3-70B 形狀（RMSNorm → QKV → attention → O → RMSNorm → gate_up → SiLU → down）。attention 用 torch SDPA，decode 用固定長度 KV cache。跑三種 trace，比較永遠開 / 永遠不開 / 決策器 / per-bucket oracle，四者在同一輪交錯 |

**關卡 G1**：

- 決策器達成整體成功標準 1、2（block 層級）；
- 表的每一格預測與 block 內實測的單項時間差 < 10%。差距大就是「單層量測不符合真實」的**發現**，要回頭查。

### F2 攻 W1：Flux AG 固定成本

| 步驟 | 內容 | 改碼？ |
| --- | --- | --- |
| F2a | 掃 `AllGatherOption` 現有開關：`use_cuda_core_local=True` + `fuse_sync=True`，把 barrier、local copy、memset 合成一個 kernel（`all_gather_op.cc:465-510`）；另比 Ring1D pull。M ∈ {8 … 2048}，量 Flux AG 與 AGKernel.forward | 否 |
| F2b | **在 Flux 外做原型**，量兩種替代做法能到的延遲下限：(i) 7 筆 peer copy 分到 7 條 stream；(ii) 一個 kernel 以 P2P load 一次讀 7 張卡的 shard（Triton 或 cpp_extension，用 IPC 指標） | 否（原型在 `ws/` 內） |
| **G2** | F2a 或 F2b 有一種在 M ≤ 64 達到 ≤ 0.05 ms → 進 F2c；都沒有 → W1 視為硬體下限，交給決策器，F2 結案 | — |
| F2c | 改 Flux：優先把 CUDA-core All2All AG 補上 BF16 / FP16 實例（`src/coll/all_gather_impls.cu:127` 目前只有 S8+FP32）；次選在 `copy_all_to_all` 多 stream 發 copy。rebuild 後 bitwise 驗證，重跑受影響的 E1 區段，重新校準表 | **是** |

預期效果 **[推論]**：

- Flux AG 降到 NCCL 水準，C − B（0.058–0.064 ms）消失；
- 但小 M 時 fused 仍會輸 D − C 的 GEMM 差距（0.011–0.038 ms），且 W3 使 overlap 為零；
- 所以 decode 區多半仍選 NCCL 路徑。**主要收益在中段 M**：切換點會往下移，因為 AG 變快對所有 M 的 fused 都有利。

### F3 攻 W2 / W4：GEMM config

| 步驟 | 內容 | 改碼？ |
| --- | --- | --- |
| F3a | 對目標模型所有層形狀 × 所有 M 桶跑 `AGKernel.profiling()`（及 RS 側對應工具），列出「最佳 / 預設 < 0.9」的斷崖桶與估計收益 | 否 |
| **G3** | 斷崖桶的估計收益在 trace 上 ≥ 1% block 延遲 → 進 F3b；否則結案 | — |
| F3b | 優先：pybind `forward` 加 `hparams` 參數（C++ `forward_impl` 已支援），讓校準器把 **(路徑, config)** 一起存進表，決策器同時選 config。次選：把 `profiling()` 產生的 config code 寫進 `src/ag_gemm/tuning_config/` 再 rebuild | **是** |

F3 同時處理 W4：挑到好的 config 就避開了病態 kernel。W4 的根因（為什麼 fused 比串行還慢）是 diag-overlap 的範圍。

### F4 整合與驗收

- 整合進 F0.2 決定的框架。graph 模式下每個 batch 桶錄製時就決定路徑，執行時零開銷。
- 量記憶體開銷：兩條路徑常駐。
- 端到端服務指標：throughput、TTFT、TPOT，在真實長度分佈的 trace 上。
- auditor 審查後寫入 DECISIONS.md。

## 3. 順序與依賴

```
F0 ──► F1 ──┬──► F2a ─► F2b ─► G2 ─► F2c ──┐
            │                               ├──► 重新校準 ──► F4
            └──► F3a ─────────► G3 ─► F3b ──┘
```

- **F1 放在 F2 / F3 前面**：它不改任何程式碼就能交付收益，也提供 block 層級的驗證框架。後面的修改都用同一框架判斷「有沒有真的變快」。
- F2 與 F3 互相獨立，可以並行。
- 粗估工作量（以 session 計，不含 rebuild 等待）：F0 2–3、F1 2–3、F2a 0.5、F2b 1–2、F2c 2、F3a 0.5、F3b 1–2、F4 視框架而定。

## 4. 量測規範

沿用 `CLAUDE.md` 5.1 與 E0 建立的協定：

- 交錯 ≥ 200 輪；production cuBLAS（undo `init_seed`）；L2 flush；逐輪 rank-max；SM clock 過濾；
- 「重疊 / 串行」主張附 nsys。

另加兩條：

- **每個 session 開始先重跑 3 個 E1 錨點**：G-FC1 M=64、M=4096，L-QKV M=64。偏差 > 10% 先查環境，因為 rebuild 後最容易出錯。
- **改 src/ 前先打 git tag**，改後的 bitwise 驗證對照 NCCL。

## 5. 協調與風險

| 風險 | 應對 |
| --- | --- |
| Flux 不能 capture 進 CUDA graph | graph 桶一律選 NCCL 路徑，Flux 只用於 eager 的 prefill；F4 報告受影響範圍 |
| 目標框架是 AllReduce 不是序列平行 | F0.2 先確認；比較單位改為整個 block，Flux 勝出範圍可能再縮小 **[推論]** |
| F2 達不到 NCCL 延遲 | G2 關卡擋下，不改碼；W1 交給決策器 |
| rebuild 耗時 / 環境壞掉 | F2c 前先量一次完整 rebuild 時間；保留未改版的 build |
| `src/ag_gemm/` 與 diag-overlap 的 kernel instrument 衝突 | 需 boss 排先後。建議 F3b 的 pybind 改動先做，範圍只在 `src/pybind/` 與 `ths_op` |
| 表過期（換驅動 / Flux / 機器） | 表內含 metadata，對不上就拒絕載入並回退「M ≤ 512 → NCCL」規則 |

## 6. 需要 boss 現在決定的事

1. 核准本計劃與順序：F0 → F1 → F2 / F3 → F4。
2. **F0.2：目標框架與平行切法。**
3. D-006 的延伸：關卡 G2 / G3 通過後允許改 `src/`，建議記為 D-007。
4. 是否 commit 目前的 ws 與 results。nsys-rep 每個 20 MB。
