Supersedes: （無）

# G1：只用既有數據建預測器，檢驗能否推廣

日期：2026-10-02 · 作者：fusion-dispatch（boss 兼 worker）· 計劃：`reports/20261002_plan_general_dispatcher.md` 的 G1
**不需要 GPU；尚未經 auditor 審查。**

---

## 0. 結論（白話）

1. **關卡通過。** 計劃要求「GPT-3 ↔ Llama 互推（B）」與「gpu ↔ steady 互推（D）」的 op 層級 regret ≤ 3%。
   只用預測器、完全不量測，各軸 regret 為 **0.03–2.29%**（AG 側 1.40–2.29%，RS 側 0.03–0.15%）。
2. **混合決策器（預測 + 少量實測把關）把 AG 側 regret 再降到 0.08–0.25%**，但要量 14–28% 的點，
   比計劃 G4 的目標（≤ 15%）多。G4 要再改把關規則。
3. **新發現：融合 kernel 何時能「邊傳邊算」，可以從 tile 排程直接算出來。**
   - 把 Flux 實際用的 CUTLASS stream-K 排程逐行移植成模擬器，不擬合任何重疊比例，
     就能預測融合路徑的時間：中位誤差 0.8%（gpu）/ 1.6%（steady）。
   - 它自己算出「M ≤ 2048 不重疊、M ≥ 3072 才重疊」。原因：GEMM 工作量不到約 1.5 波時，
     每個 thread block 都會碰到最晚才到的資料，只能等。
   - 拿模型**沒看過**的 Phase 0 形狀測試：N=4096 預測藏住 3%（實測 7%），N=8192 預測 70%（實測 65%）。
     這正是 ws/diag-overlap 要解釋的現象（Phase 0「N=4096 只重疊約 2%」）**[推論，模型支持]**。
4. **剩下的大錯誤都來自 Flux 登錄表中「為 PCIe 機器調的 config」**（排在檔案前段、優先生效）。
   這些 config 在本機（NVLink）上 GEMM 慢 12–58%，模型預測不到。
   把「選中的 Flux 路徑用的是 PCIe 調校 config」當成風險訊號去實測，就能抓到。
   **這條規則是看過 G1 結果才定的**，要在 G3（沒看過的模型 / TP=4）重新驗證。
5. 對照組：
   - 機器學習基準（隨機森林）在**量過的形狀**上很準（D 軸 0.2–0.5%），
     但換到**沒量過的形狀**就退化（B 軸 3.6–4.7%），比固定門檻還差；
   - 預測器在新形狀上維持 1.4–2.3%。

**下一步**：G2，寫約 10 分鐘的校準微基準，改成「只用微基準」擬合（不再用決策表的元件欄），
重跑本報告的評估。

---

## 1. 做了什麼

### 1.1 數據（全部已在 repo）

| 數據 | 路徑 | 用途 |
| --- | --- | --- |
| AG 側 op 地圖 | `results/final_ag_points.csv` | 4 層 × 20 個 M × {gpu, steady}；四條路徑與四個元件的單獨時間 |
| RS 側 op 地圖 | `results/final_rs_points.csv` + 各點來源 run 的 `summary_items.csv` | 4 層 × 20 個 M × 2 模式；NCCL RS / AR 元件 |
| Flux AG 同步成本 | `results/f2_ag_latency_v1/raw_K*.csv` | α_sync = 0.0533 ms（gpu，M ≤ 64） |
| kernel 啟動時間 | `results/v8_nsys/parse_all.txt`、`results/e2_nsys/parse_G-FC1.txt` | 31.7 µs + 本地 shard bytes / 838 GB/s |
| 未見形狀交叉驗證 | `results/e0_anchor_v2/summary_items.csv`（P0-4096、P0-8192） | 模型從未用來擬合 |

讀取程式：`scripts/predictor_data_v1.py`。

### 1.2 預測器（`common/cost_model/predictor/`，純 Python）

每條路徑的時間 = 通訊 + 計算 − 重疊（計劃第 1 節）。

| 檔案 | 模型 | 參數怎麼來 |
| --- | --- | --- |
| `curves.py`、`comm.py` | NCCL AG / RS / AR：對 bytes 的分段 α-β 曲線。Flux AG = α_sync + W × 單筆 copy(bytes / W) | 元件欄 c_nccl_* / c_flux_ag（只用訓練列） |
| `gemm.py` | t0 + soft-max(讀資料時間, 補齊 tile 後的計算時間)；cuBLAS、Flux gemm_only、Flux GemmRS 各一組。限制 η ≤ 1、頻寬 ≤ 2.039 TB/s | 元件欄 c_cublas / c_fluxgemm |
| `flux_config.py` | 查 Flux 實際用的 config：登錄表 `(m, n, k)` 完全相符才用（第一筆生效），否則用預設 config（`_GemmHParamsT_0`：128×128×64、stream-K、RasterAlongM） | 原始碼與 build 目錄 |
| `overlap.py` | 融合 AG+GEMM：**逐 tile 的事件模擬**（見第 3 節）；GemmRS：GEMM + 外露 scatter T_scat²/(T_scat+T_gemm) + 固定成本 | AG：d_tail（無法重疊的形狀上的閉式解）、κ（重疊期間 GEMM 變慢比例，穩健損失）；RS：GemmRS GEMM 參數與 α、β 一起擬合（sm80 上 GemmRS 的通訊無法單獨量，CLAUDE.md 陷阱表） |
| `paths.py`、`profile.py` | 組合成 A / B / C / D（AG）與 A / B（RS）的預測；參數存 JSON | — |

### 1.3 推廣軸與政策（`scripts/eval_predictor_v1.py`）

**推廣軸**：測試列從不參與擬合。

| 軸 | 訓練 | 測試 |
| --- | --- | --- |
| A | 2 的冪次 M | 保留的 M（24、72、136、264、520、1032、3072、6144） |
| B / B' | GPT-3 層 / Llama 層 | Llama 層 / GPT-3 層 |
| D / D' | gpu 模式 / steady 模式 | 另一個模式（同層、同 M） |
| IN | 全部 | 全部（樣本內參考） |

**政策**：

| 政策 | 說明 |
| --- | --- |
| on | 永遠開 Flux |
| off | 永遠 NCCL + cuBLAS |
| thr512 | 固定門檻：M ≤ 512 不開 |
| rf | 隨機森林（純 Python，`scripts/rf_baseline_v1.py`） |
| pred | 只用預測器，選預測最快的 |
| hyb | 預測差距 < ε 時，實測前兩名。ε = 訓練列上路徑誤差的第 90 百分位（8.7–12.7%） |
| hyb+fb | hyb，另外「選中 Flux GEMM 路徑且用預設 config」時實測（**計劃原定規則**） |
| hyb+pcie | hyb，另外「選中 Flux GEMM 路徑且 config 是 PCIe 調校」時，與最佳的非 Flux-GEMM 路徑比（**修正規則，看過結果後才定**） |

- **regret** = Σ(選中路徑實測 − 最快路徑實測) / Σ(最快路徑實測)，定義同 `policy_eval_v1.py`；
- **探測的結果取表中的實測值**：偏樂觀，真實探測只有約 20 次、有雜訊。

---

## 2. 結果

來源：`results/g1_predictor/eval_summary.csv`、`eval_log.txt`（2026-10-02，`python3 ws/fusion-dispatch/scripts/eval_predictor_v1.py`，43 秒，重跑結果完全相同）。

### 2.1 AG 側（AG+GEMM：fused / NCCL+cuBLAS / FluxAG+cuBLAS / FluxAG+FluxGEMM）regret（%）

括號內為探測點數 / 測試點數。

| 軸 | 模式 | on | off | thr512 | rf | **pred** | hyb | hyb+fb | **hyb+pcie** |
|---|---|---|---|---|---|---|---|---|---|
| B | gpu | 7.49 | 13.80 | 2.20 | 3.58 | **1.40** | 1.35 (4/40) | 1.35 (10/40) | **0.13** (6/40) |
| B | steady | 7.17 | 12.17 | 1.99 | 3.64 | **1.58** | 0.24 (11/40) | 0.24 (14/40) | **0.24** (12/40) |
| B' | gpu | 6.11 | 13.88 | 2.85 | 4.70 | **2.29** | 1.01 (11/40) | 1.01 (15/40) | **0.11** (16/40) |
| B' | steady | 5.22 | 9.91 | 2.18 | 3.93 | **1.95** | 0.61 (10/40) | 0.52 (14/40) | **0.09** (14/40) |
| D | gpu→steady | 5.92 | 10.72 | 2.11 | 0.48 | **1.74** | 0.93 (12/80) | 0.93 (22/80) | **0.11** (18/80) |
| D' | steady→gpu | 6.61 | 13.85 | 2.61 | 0.20 | **2.04** | 0.92 (13/80) | 0.92 (23/80) | **0.08** (19/80) |
| A | gpu | 10.30 | 8.07 | 4.80 | 2.32 | **2.32** | 0.25 (5/32) | 0.25 (12/32) | **0.25** (5/32) |
| A | steady | 9.04 | 7.27 | 3.66 | 2.50 | **2.09** | 0.00 (7/32) | 0.00 (13/32) | **0.00** (7/32) |
| IN | gpu | 6.61 | 13.85 | 2.61 | 0.22 | 1.88 | 0.92 (12/80) | 0.92 (22/80) | 0.04 (18/80) |
| IN | steady | 5.92 | 10.72 | 2.11 | 0.26 | 1.85 | 0.88 (13/80) | 0.88 (23/80) | 0.06 (19/80) |

### 2.2 RS 側（GemmRS vs cuBLAS+NCCL RS）regret（%）

| 軸 | 模式 | on | off | thr512 | rf | **pred** | hyb | hyb+fb | **hyb+pcie** |
|---|---|---|---|---|---|---|---|---|---|
| B | gpu | 0.65 | 26.04 | 1.42 | 0.01 | **0.03** | 0.00 (5/40) | 0.00 (27/40) | **0.00** (5/40) |
| B | steady | 0.60 | 26.90 | 1.39 | 0.12 | **0.12** | 0.11 (6/40) | 0.11 (24/40) | **0.11** (6/40) |
| B' | gpu | 0.71 | 16.80 | 0.56 | 0.20 | **0.13** | 0.10 (6/40) | 0.00 (24/40) | **0.10** (6/40) |
| B' | steady | 0.73 | 13.98 | 0.61 | 0.19 | **0.14** | 0.00 (7/40) | 0.00 (20/40) | **0.00** (7/40) |
| D | gpu→steady | 0.69 | 17.93 | 0.85 | 0.05 | **0.15** | 0.02 (13/80) | 0.02 (51/80) | **0.02** (13/80) |
| D' | steady→gpu | 0.69 | 19.69 | 0.83 | 0.08 | **0.11** | 0.01 (13/80) | 0.01 (44/80) | **0.01** (13/80) |
| A | gpu | 0.60 | 19.30 | 0.96 | 0.03 | **0.31** | 0.01 (8/32) | 0.00 (28/32) | **0.01** (8/32) |
| A | steady | 0.72 | 20.07 | 1.00 | 0.17 | **0.34** | 0.08 (11/32) | 0.08 (26/32) | **0.08** (11/32) |

### 2.3 探測量（AG + RS 合計，佔測試點）

| 軸 | hyb | hyb+pcie | hyb+fb（計劃原定） |
|---|---|---|---|
| B gpu / steady | 11% / 21% | 14% / 22% | 46% / 48% |
| B' gpu / steady | 21% / 21% | 28% / 26% | 49% / 42% |
| D / D' | 16% / 16% | 19% / 20% | 46% / 42% |
| A gpu / steady | 20% / 28% | 20% / 28% | 62% / 61% |

- **計劃原定的「預設 config 就實測」太貴**（42–62%）：
  - 大多數點本來就跑預設 config，而且多半沒問題；
  - 真正出問題的是 PCIe 調校 config。
- ε 掃描（`results/g1_predictor/eps_sweep.csv`）：ε ≈ 5% 時大部分收益已拿到，探測約 5–12%。
  G4 可用較小的 ε 加風險規則，或改用「預期損失」決定是否探測。

### 2.4 預測準確度（測試點 MAPE）

- **元件**：NCCL 1–4%、Flux AG 0.7–1.3%（同模式）；cuBLAS 6–12%；Flux GEMM 6–20%（B 軸 steady 最差，19.8%）。
- **路徑**：AG 側 3.6–11%；RS 側 4–14%。
- **接近切換點的點**（融合與最佳非融合相差 < 15%）：AG 3.9–5.6%，RS 5.4–16.3%。
  RS 在 D 軸最差，因為 gpu 與 steady 的固定成本不同（見第 4 節）。
- **fused vs off 判斷正確率**（排除平手）：AG 87–97%，RS 93–100%。

### 2.5 預測的切換點（B 軸 gpu，模型沒看過 Llama）

A 為融合，B 為 NCCL+cuBLAS，D 為 FluxAG+FluxGEMM；完整序列見 `eval_log.txt` 的 `picks` 行。

| 層 | 實測最快 | 預測 | 說明 |
| --- | --- | --- | --- |
| L-QKV | M ≤ 1032 B；2048 起 A（2048 為平手） | M ≤ 2048 B；3072 起 A | 只差平手點 |
| L-GU | M ≤ 2048 多為 B（512 / 1024 / 2048 平手）；3072 A；**4096 B**（斷崖）；≥ 6144 A | M ≤ 2048 B；≥ 3072 A | 唯一錯誤是 4096，PCIe 調校 config 斷崖 |
| L-O、L-down | 72–256 起 A（之前多為平手） | 128 起 A | 對 |

---

## 3. 機制：融合 kernel 的排程模擬

### 3.1 模擬了什麼

全部照原始碼。

- **tile → thread block 的分配**：CUTLASS stream-K（`3rdparty/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h` 的 `get_blocks` / `get_sk_blocks` / 建構子）逐行移植。
  - SK tile 是 tile 編號 [0, sk_tiles)，它們的 block 先發；DP block 在後。
  - 出處：`gemm_universal_streamk.h:1014-1044`；Flux 同一段在 `src/ag_gemm/sm80_all_gather_gemm.hpp:860-892`。
- **Flux 把 tile 的列從本卡的列開始輪轉**：`sm80_all_gather_gemm_threadblock_swizzle.hpp` 的 `tile_m_offset`。
- **每個 tile 開始前，等所有涵蓋它的 shard 到齊**：`sm80_all_gather_gemm.hpp:923-939`。
- **shard 依 rank+1、rank+2… 順序一筆筆拉**：`src/coll/ths_op/all_gather_op.cc:553-576`。
- **StreamkDP** → avail_sms = 1（純 data-parallel）；**RasterAlongN** → 逐列排（`gemm_v2_ag_kernel.hpp:244-274`）。
- **校準**：每次迭代的時間由「資料全部到齊時的模擬時間 = 單獨 gemm_only 時間」決定。這樣波數量化已包含在內。

### 3.2 用實測元件餵模型（只檢驗重疊模型本身）

| 模式 | 融合時間 MAPE | 中位誤差 | p90 誤差 |
| --- | --- | --- | --- |
| gpu | 3.0% | 0.8% | 11.7% |
| steady | 3.4% | 1.6% | 11.3% |

最差的點：
- G-FC1 M=3072（22%）：已知的融合 kernel 斷崖；
- G-FC1 M=512；
- M=136 / 264：融合比串行慢 25–36 µs，原因不明。

### 3.3 藏住的通訊

(D − A) / Flux AG；E1 實測 vs 模型（實測元件）。

| 點 | config | 實測 | 模型 |
| --- | --- | --- | --- |
| G-QKV M=64 | 登錄 | 14% | 13% |
| L-QKV M=64 | 預設 | 15% | 14% |
| L-GU M=512 | 登錄 | 4% | 10% |
| L-QKV M=2048 | 預設 | 7% | 6% |
| L-GU M=3072 | 預設 | 51% | 74% |
| G-QKV M=4096 | 登錄 | 64% | 76% |
| L-QKV M=4096 | 登錄（PCIe） | 82% | 75% |
| L-GU M=4096 | 登錄（PCIe） | 85% | 87% |
| G-FC1 M=1024 | 登錄（PCIe） | 16% | 72% |
| G-FC1 M=4096 | 登錄（PCIe） | 51% | 88% |
| G-QKV M=2048 | 預設 | −36%（融合較慢） | 20% |
| G-FC1 M=3072 | 預設 | −133%（斷崖） | 21% |

- 小 M 的 13–15% 是 d_tail（融合 op 省下的固定成本約 14 µs），不是真正的重疊；
- nsys 的結論「M ≤ 512 無重疊，≥ 3072 藏住 44–87%」與模型一致。

### 3.4 為什麼 M=2048 不重疊、3072 會

以 L-QKV 為例：預設 config 128×128×64、stream-K、每 SM 1 個 block、108 SM。

| | M=2048 | M=3072 |
| --- | --- | --- |
| tile 數 | 160 | 240 |
| 分配 | **全部是 SK**：108 個 block 各分到約 1.5 個 tile 的工作，同時開跑 | 132 個 SK（偏移後是先到的列）+ 108 個 DP（後到的列）|
| 結果 | 每個 block 都碰到最晚的 shard，只能等到資料全到 → 不重疊 | DP 在第二波才跑 → 重疊 |

**[推論]** 一般規則：stream-K 工作量 ≲ 1.5 波時不重疊；DP config 或 ≥ 2 波時才重疊。

### 3.5 沒看過的形狀（Phase 0 錨點）

來源：`results/e0_anchor_v2`，gpu 模式。

| 形狀 | config | tile（波數） | 融合實測 | 模型（實測元件） | 模型（全預測） | 藏住：實測 / 模型 | 決策 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0-4096 M=1024 | 預設 SK | 32（0.30） | 0.2918 | 0.2861 | 0.3124 | 5% / 7% | B / B ✓ |
| P0-4096 M=4096 | 預設 SK | 128（1.19） | 0.6810 | 0.7034 | 0.7257 | **7% / 3%** | A / A ✓ |
| P0-4096 M=16384 | 預設 SK | 512（4.74） | 1.6497 | 1.6556 | 1.7069 | 49% / 49% | A / A ✓ |
| P0-8192 M=4096 | 登錄 DP | 256（2.37） | 0.6446 | 0.6242 | 0.6263 | **65% / 70%** | A / A ✓ |

**給 ws/diag-overlap 的假說 [推論]**：
- **現象**：Phase 0 的 N=4096（每卡 n=512，M=4096）只有 128 個 tile（1.19 波），預設 stream-K config 下無法重疊。
- **預測**：改用 data-parallel + RasterAlongN 的 config，融合時間預測如下（假設單獨 GEMM 時間不變）：
  - 128×128×64 → 0.584 ms；
  - 64×128×64 → 0.543 ms；
  - 對照：實測 0.681 ms；diag-overlap 的目標 < 0.55 ms。
- **驗證**：
  1. 以 `FLUX_TUNE_CONFIG_FILE` 或 `profiling()` 指定 config，量 A 與 gemm_only；
  2. nsys 看各 CTA 的開始時間。
- 已轉 boss（PROJECT 第 3 節），本 ws 不改 diag-overlap 的檔案。

---

## 4. 限制與已知問題（誠實列出）

1. **登錄表 config 斷崖預測不到**：
   - G-FC1 M=1024：Flux GEMM 是 cuBLAS 的 1.58×；L-GU M=4096：1.34×；L-QKV M=4096：1.23×。
   - 三者都是登錄表 `// PCIE` 區段的條目（`src/ag_gemm/tuning_config/config_ag_gemm_kernel_sm80_A100_tp8_nnodes1.cu`，emplace 第一筆生效）。
   - 「這些條目是為 PCIe 拓撲調的」是從區段標題推得 **[推論]**。
   - hyb+pcie 規則是看過結果後才定的，**必須在 G3 新形狀上重新驗證**（G3 的 oracle 量完前不得回頭調）。
2. **融合比串行還慢的點**（M=136、264；G-QKV 2048；G-FC1 3072）原因不明，模型不含此效應。
3. **κ（重疊期間 GEMM 變慢）不穩定**：依訓練集為 0–0.125。代表模型缺一項與形狀有關的競爭效應。
   nsys 中融合 GEMM 比單獨 GEMM 慢 3–47%，各點不同。
4. **多 block 共用 SM 時低估重疊**：模型每個 block 的速度固定，旁邊的 block 在等時不會變快。
   例：what-if 的 64×128×32（每 SM 3 個 block）預測不重疊，可能偏悲觀。
5. **D 軸不是純時脈互推**：
   - gpu 與 steady 的差異主要是量測方式（steady 連發 16 次攤掉每次 6–12 µs 的固定成本），不是時脈；
   - 大 M 時兩者都在功耗上限附近，時間相近；
   - E1 數據無法分開時脈與量測方式，所以 NCCL 的時脈修正 γ 維持 0；
   - 決策仍可互推（regret 1.7–2.0%），但元件 MAPE 達 20–30%（NCCL）；
   - 真正的時脈互推要等 G2 在兩種時脈下各做一次校準。
6. **RS 側 GemmRS 的 GEMM 參數與通訊參數一起擬合**：sm80 上無法單獨量，兩者可能互相抵換。
7. **探測量超過 15% 目標**（見 2.3）。
8. **G1 的擬合仍用決策表的元件欄**，不是獨立微基準；這正是 G2 要改的。

---

## 5. CLAUDE.md 5.3 自查

- **分母**：regret 的分母是同一張表中每點最快路徑的實測；MAPE 對同一點的實測。
- **效率 > 100% / 負差值**：
  - GEMM 擬合限制 η ≤ 1、頻寬 ≤ 2.039 TB/s（未加限制前 Flux GEMM 曾擬合到 2.19 TB/s，已修正）；
  - 數據中的負「藏住通訊」（融合較慢）列為發現（第 3.3、4 節），未丟棄。
- **輸出欄位**：元件、路徑、決策、探測、ε 掃描、逐點表都已寫出並使用。
- **錨點**（`eval_log.txt` 開頭），全部 < 20%：
  - Flux AG M=4096：模型 0.493 vs Phase 0 0.468（+5%）；
  - 每卡 179 vs 188 GB/s（−5%）；
  - NCCL AG M=4096：0.495 vs 0.478（+4%）；
  - Flux AG M=64：0.108 vs 0.10（+8%）；
  - 403 MB 時每卡：NCCL 197、Flux 228 vs all-to-all 217.66 GB/s（−10% / +5%）。
- **重疊主張**：以 E1 的 D − A 與 nsys（V8、E2）佐證；「stream-K 造成不重疊」是程式碼 + 模型推論，驗證方法見 3.5。
- **差值量測**：本階段無新量測；來源數據為交錯 ≥ 200 輪、clock 過濾。
- **原始輸出**：`results/g1_predictor/` 全部提交；重跑完全相同。
- **推論**：已標記。

---

## 6. 檔案

| 類別 | 路徑 |
| --- | --- |
| 預測器 | `common/cost_model/predictor/{__init__,curves,fitting,comm,gemm,flux_config,overlap,paths,profile}.py` |
| 腳本 | `scripts/predictor_data_v1.py`、`scripts/eval_predictor_v1.py`、`scripts/rf_baseline_v1.py` |
| 結果 | `results/g1_predictor/{eval_log.txt, eval_summary.csv, eval_points.csv, eps_sweep.csv, profile_g1_all_{gpu,steady}.json}` |

profile JSON 是用全部列擬合的 G1 參數，只供 G2 對照，**不是**校準結果。

**計劃偏差**：計劃列的 `fit_predictor_v1.py` 沒有另外寫，擬合在 `eval_predictor_v1.py` 內，profile 由它輸出。
