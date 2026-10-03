Supersedes: （無）

> 狀態：**已核准（使用者，2026-10-02）**；執行中：G0–G3 完成（`reports/20261002_g1_predictor.md` 關卡通過；`20261002_g2_calibration.md` 校準 4 分鐘；`20261002_g3_unseen.md` 預先登記的新情境），下一步 G4。
> G3 修正：G4 的決策器改為「模型（通訊 + 重疊）+ 實測單卡 GEMM + 少量探測」（G3 的錯誤來自 cuBLAS 斷崖），**使用者 2026-10-03 核准**（D-009）；G3 的 block 驗證移到 G4 一起做。
> G1 修正：計劃第 3 節第 3 步「非 registry config 一律探測」實測太貴（42–62%），且真正的斷崖來自登錄表 PCIe 區段的 config；改為「PCIe 調校 config 才探測」，G3 重驗。
> 原始計劃檔：`~/.claude/plans/jazzy-gliding-blossom.md`（本檔為 repo 內權威副本）。
> 新 session：先讀本檔與 `STATUS.md` §0，再讀 `reports/20261002_related_work.md`。

# 計劃：泛用決策器（Flux 開關 + 資料切法）

## Context（為什麼要做）

**現況**：決策器靠**查表**運作，表要事先實測，每種層形狀 × 每個 M 都量一次（`ws/fusion-dispatch/scripts/dispatcher_v1.py` + `build_table_v1.py`）。
- 已驗證：以真正的 vLLM 0.8.5 為對照，prefill 快 9.3–17.7%，regret ≈ 0（`reports/20260930_verification.md`）。
- 缺點一：換模型、換 TP 卡數、換硬體、換時脈狀態都要重量，一次要好幾小時。
- 缺點二：最終要用的 PCIe 異質晶片還沒到手，根本無法事先量。

**使用者決定（2026-10-02）**：先做泛用決策器，再做 vLLM 端到端（F4）；並參考既有論文。

**文獻調查**：查證約 100 篇論文與系統原始碼。

- **既有系統的決策方式**，都不泛用：
  - 查表：TensorRT-LLM、Comet、Flux autotune；
  - 人工門檻：vLLM、TokenWeave、Shift Parallelism；
  - 窮舉搜尋：CoCoNet、Triton-distributed。
- **沒有人同時決定「融合開關」與「TP 切法」**，也沒有人預測 decode 小批時的切換點，或證明能推廣到新卡數 / 新硬體。
- **最有證據的方法**：「物理模型 + 少量實測校準」。
  - KernelSight-LM：未見過的 GPU 上，只靠規格 12% 誤差，加一次微基準掃描後 3.8%；
  - FlashOverlap（EuroSys'26）：PCIe / NVLink 預測誤差約 3.4%，選擇達最佳的 99%；
  - tritonBLAS：解析模型的選擇效率 94.7%；
  - 對照：純 ML 模型換硬體後明顯退化（TenSet、Habitat 被他人重評）。
- **「NCCL 受 SM 時脈影響、copy engine 不受」未見前人文獻**，可能是新發現。

**目標成果**：一個「預測為主、少量實測把關」的混合決策器。
- 輸入：層形狀、M、TP 卡數、硬體參數（幾分鐘的微基準校準）。
- 輸出：每個 M 的切法 + 每層路徑。
- 驗證：在**沒量過**的模型、TP 卡數、時脈狀態、PCIe 代理上，regret 對照實測 oracle。

---

## 設計

### 0. 為何這樣設計（2026-10-02 向使用者說明的版本）

**拆成「通訊 + 計算 − 重疊」**：每條路徑都是這三件事的組合，而每個實測現象都落在其中一項。

| 實測現象 | 對應項目 |
| --- | --- |
| Flux AG 約 0.1 ms 固定成本 | 通訊 |
| 小 M 時 Flux GEMM 比 cuBLAS 慢 | 計算 |
| M ≤ 512 無重疊、≥ 3072 藏住 44–87% | 重疊 |
| 小 M 時 TP+AR 的通訊次數較少 | 切法 |

拆開後每一項都能用小測試單獨校準，換硬體只換參數。

**各項的設計理由**：
- **通訊**：α-β 是經典模型，NCCL 內部也這樣算。
  - 分段：NCCL 依大小切換協定；
  - Flux 固定成本按零件拆（copy 筆數 × 延遲 + 同步次數）：這是能推廣到其他 TP 卡數的關鍵；
  - SM 時脈修正：V6c，NCCL 受時脈影響、copy engine 不受。
- **計算**：roofline（小 M 時受讀權重限制）加 wave 修正。NeuSight / KernelSight-LM 在未見 GPU 上約 4–12% 誤差。
- **重疊**：以「shard 到達 → 涵蓋該列 tile 的 shard 到齊才能算」模擬時間軸，不用固定比例。這樣模型能自己算出切換點，不需人工門檻。
- **切法**：把通訊次數 / bytes 與 norm 列數的差加總比較。

**為何是混合式**：
- 純查表：不泛用，晶片未到無法量；
- 純 ML：換硬體退化，且無法解釋；
- 純物理模型：預測不到 config 斷崖與近平手的誤差；
- → 物理模型負責大部分決定，只在「差距 < 模型誤差」或「斷崖風險」時實測確認。這是文獻中效果最好的形式（FlashOverlap、TileSight、KernelSight-LM）。

**對異質晶片**：到手前可代入其 PCIe α/β（Phase 0：單向約 22 GB/s、雙向競爭 6.46 GB/s）與計算規格先行預測；到手後跑約 10 分鐘校準即可更新。

### 1. 預測器（共用庫，放 `common/cost_model/predictor/`，同時完成 ws/cost-model 的目標）

每條路徑的時間 = 通訊 + 計算 − 重疊，每一項都對應已觀察到的機制。

**`comm.py`：通訊模型**，每種通訊原語各一組參數：

| 原語 | 模型 | 依據 |
| --- | --- | --- |
| NCCL AG / RS / AR、vLLM custom AR | 對 log(bytes) 的分段線性 α-β（NCCL 會依大小切換 LL / LL128 / Simple 協定）。SM 時脈修正：t = t_ref·(f_ref/f)^γ | V6c：NCCL 0.565 ms @1155 MHz vs 0.475 @1410 MHz |
| Flux AllGather（copy engine） | 結構化：α = (W−1+1)·α_copy + 2·α_barrier + α_memset；資料量為逐筆串行 bytes/β_ce。與時脈無關 | E1 nsys：8 筆串行，每筆約 14 µs；F2：同步約佔 60 µs |
| 長 graph 中的 c10d NCCL | 選用的放大項（每次 collective 的額外成本隨 graph 內 collective 數上升） | V4：SP 全 NCCL 為線性外推的 1.16–1.57×；或改用 pynccl 避開 |

**`gemm.py`：GEMM 模型**（NeuSight / KernelSight-LM 式）：
- t = max(FLOPs / (P_peak·η), bytes_weight+act / B_mem) + t_launch；
- tile 數 = ⌈M/tm⌉·⌈n/tn⌉，waves = ⌈tiles / (SM·occupancy)⌉；
- η = a − b/waves，每個 kernel 家族各一組（cuBLAS、Flux CUTLASS）。Flux 的 tile 由 registry 或 fallback 取得。
- 依據：E1 小 M 時 GEMM 受讀權重頻寬限制；Flux GEMM 在小 M 比 cuBLAS 慢 7–33%。

**`overlap.py`：融合 kernel 的事件模型**：
- shard i 的到達時間由 comm 模型算出（串行到達）；
- tile 列依 shard 覆蓋分成 g = min(W, ⌈M/TILE_M⌉) 組，每組資料到齊才能算；加一個競爭係數（文獻：overlap 造成計算變慢 18.9%）；
- GEMM+RS：epilogue scatter 與後續 tile 重疊，只有最後一段 + local reduction 外露。
- 依據：nsys V8，M ≤ 512 時藏住 0、M ≥ 3072 時藏住 44–87%。

**`block.py`：切法層級的模型**：
- TP+AR vs SP 的差 = 通訊次數與 bytes 的差 + norm / residual 在 M 或 M/W 列上的頻寬成本差 + 補齊到 W 倍數的成本；
- 依據：F1 / V3，小 M 時 2 次 AR 勝過 2 AG + 2 RS。

**硬體檔**：`common/cost_model/hw_profiles/<host>_<tp>_<clock>.json`，存校準出來的參數，附來源與日期（`params.json` 的規範）。

### 2. 校準（幾分鐘的微基準，不量決策表）

`ws/fusion-dispatch/scripts/calibrate_hw_v1.py`：
- 通訊：每種原語掃約 15 個大小（1 KB – 256 MB）；
- GEMM：每個 kernel 家族在 2–3 個代表形狀上量少數幾個 M；
- 時脈：在輕負載與持續滿載兩種狀態下各量一次。
- 目標 ≤ 10 分鐘，一律包在 `exclusive_guard.py` 內。

**關鍵主張**：預測器只用這些微基準擬合，**不用**任何決策表的點。這樣才算泛用。

### 3. 決策器 `dispatcher_v2.py`（沿用 v1 的後端）

1. 用預測器算出每個 M 的「切法 + 每層路徑」預測時間，取最小值。
2. 套用限制：graph 模式不可 capture 的路徑（AGKernel）、M 補齊、記憶體。
3. **不確定性把關**：以下情況才做短探測（兩個候選交錯各約 20 次，部署時做一次並快取）：
   - 預測差距 |Δ̂| < ε·t̂（ε 取驗證得到的誤差）；
   - 選中的 Flux 路徑使用非 registry 的 config（斷崖風險）。
4. rank 0 決定後廣播，各卡核對 hash（沿用 v1 機制）。

### 4. 對照組（論文需要）

| 對照組 | 內容 |
| --- | --- |
| oracle | 現有實測表（E1 / E5 / V3） |
| vLLM 預設 | — |
| 固定門檻 | M ≤ 512 不開；仿 vLLM `sp_min_token_num` |
| 只用預測器 | 不探測 |
| 機器學習基準 | 隨機森林，Vidur 式，用其他形狀訓練 |
| 本方法 | 混合（預測 + 把關） |

---

## 分階段執行（每階段有關卡；全程遵守 CLAUDE.md 5.1、exclusive_guard、tmux、文件同步）

| 階段 | 內容 | GPU | 關卡 / 產出 |
| --- | --- | --- | --- |
| **G0** | 文獻整理寫成 `reports/<date>_related_work.md`（兩次搜尋的結果，標明已驗證 / 未驗證） | 否 | 報告 |
| **G1** | 預測器 v1：先用**既有數據**擬合與檢驗。推廣測試：(A) 沒量過的 M；(B) GPT-3 ↔ Llama 互推；(D) gpu ↔ steady 時脈互推 | 否 | op 層級 regret ≤ 3%（B、D）才往下；否則分析缺哪一項 |
| **G2** | 校準微基準 + **只用微基準**重新擬合，對全部既有實測評估 | 是，約 10 分鐘 | 預測誤差（MAPE）、regret 表 |
| **G3** | 補「沒量過情境」的實測：新模型（Qwen2.5-72B、Llama-3-8B）與 TP=4 / TP=2 的 op 地圖（M 子集）+ 一組 block 驗證。需新增可指定卡數的啟動腳本 | 是，約 2–3 小時 | 新的 oracle |
| **G4** | `dispatcher_v2` + 探測策略 + block 驗證（`validate_block_v4.py`，校準 / 驗證分開），與所有對照組比較 | 是 | **成功標準**（見下） |
| **G5** | PCIe 代理：NCCL 關掉 P2P（`NCCL_P2P_DISABLE=1`，經 host），Flux 路徑不可用，決策器需處理「路徑不可用」與很不同的 α/β。與 hetero-proxy 協調 | 是 | 代理情境的 regret |
| **G6** | 總報告 + auditor 審查 → DECISIONS；之後 F4（vLLM 端到端）直接使用 dispatcher_v2 | — | — |

可選：若實驗室有其他型號 GPU（需問教授），在 G4 後加一次跨機器驗證。

**成功標準**（沒量過的形狀與 TP 卡數上）：
- 混合決策器 block 層級 regret ≤ 2%（加權 trace）；
- 校準 ≤ 10 分鐘，探測數 ≤ 決策表點數的 15%；
- 接近切換點的預測誤差 ≤ 10%；
- 「只用預測器」與機器學習基準的結果也一併報告。

---

## 要新增 / 修改的檔案

**新增：**
- `common/cost_model/predictor/{__init__,comm,gemm,overlap,block,profile}.py`、`common/cost_model/hw_profiles/*.json`
- `ws/fusion-dispatch/scripts/`：
  - `calibrate_hw_v1.py`、`fit_predictor_v1.py`、`eval_predictor_v1.py`（regret / MAPE / 推廣軸）；
  - `dispatcher_v2.py`、`validate_block_v4.py`；
  - `launch_tp.sh`（指定卡數：`launch.sh` 第 24 行固定用全部 GPU）。

**沿用（不重寫）：**
- 量測協定：`dispatch_map_v2.py` 的 `run_mode` / `Case`、`dispatch_map_rs_v1.py` 的 `CaseRS`；
- 通訊量測：`ag_latency_v1.py`；
- 後端與一致性機制：`dispatcher_v1.py`；
- block 驗證骨架：`validate_block_v3.py`；
- 分析：`analyze_v1.py`、`policy_eval_v1.py`、`merge_points_v2.py`；
- 環境：`common/measure/{exclusive_guard,clock_logger}.py`、`launch_vllm_env.sh`；
- 錨點：`common/cost_model/params.json`。

**文件**（使用者要求所有進度寫入文件）：
- `ws/fusion-dispatch/STATUS.md`：§0 快速跟上、§1 加「第三階段」、§4；
- `JOURNAL.md`；
- `PROJECT.md`：fusion-dispatch 列；註明 ws/cost-model 的模型目標由此完成；
- `docs/DECISIONS.md`：D-008，方向改為先做泛用決策器、F4 延後；
- 每階段一份 `reports/`；
- 每階段結束 commit 並用 SSH push 到 `fusion-dispatch` 分支。

---

## 風險與對策

| 風險 | 對策 |
| --- | --- |
| Flux 參數斷崖無法預測（G-FC1 M=3072、L-GU M=4096） | 非 registry config 一律探測（決策器第 3 步） |
| fallback config 的 tile 不明 | 從 `OpRegistry` / profiling 記錄讀出，或從 nsys 推得 |
| NCCL 協定切換造成非線性 | 用分段線性 α-β |
| 長 graph 的 NCCL 放大原因不明（V11 未重現） | 列為校準項，或在 graph 中改用 pynccl |
| TP=4 時 Flux registry 條目不同 | 正好用來檢驗斷崖探測 |
| PCIe 代理下 Flux 無法運作 | 決策器必須支援「路徑不可用」，這是設計需求 |

---

## Verification（怎麼確認做對了）

1. **預測器單元檢查**：對已知錨點預測要在 20% 內，否則先懷疑模型（CLAUDE.md 錨點規則）：
   - Flux AG M=64 約 0.10 ms、NCCL AG 約 0.03–0.05 ms；
   - G-FC1 M=64 GEMM 約 0.12 ms；
   - nsys 的重疊比例。
2. `eval_predictor_v1.py` 對每個推廣軸輸出：MAPE、決策正確率（排除平手）、op 與 block 層級 regret（oracle = 實測表），以及與各對照組的比較表。
3. **block 驗證**：`validate_block_v4.py` 在 vLLM 環境中跑。校準與驗證用不同 seed，全程 `exclusive_guard` 為 CLEAN，原始逐輪 CSV 提交。
4. **G3 的新 oracle**：用 E1 的協定量；與 G2 的預測比較時不得回頭調參（防止看答案）。
5. 每階段更新 STATUS §0、JOURNAL、PROJECT，commit + push；最後安排 auditor。

---

## 附錄 A：執行所需資訊（給接手的 session）

### A.1 現有的 ground truth（G1 用，全部已在 repo）

| 數據 | 路徑 | 內容 |
| --- | --- | --- |
| AG 側 op 地圖（最終版） | `results/final_ag_points.csv` | 4 層（G-FC1、G-QKV、L-QKV、L-GU，以及 P0-* 錨點）× 20 個 M × {gpu, steady}。各 arm 中位數、最佳 off、Δ 的 p10 / p90、判決、分解（A−D、D−C、C−B）、元件時間。三次 run 合併：e1_map_v2 → e1_rerun_v2 → v7 |
| RS 側 op 地圖（最終版） | `results/final_rs_points.csv` | 4 層（L-O、L-down、G-O、G-FC2）× 20 個 M × 2 模式。A = GemmRS，B = mm + NCCL RS；元件 c_cublas、c_nccl_rs、c_nccl_ar、c_fluxgemm |
| 逐輪原始數據 | `results/e1_map_v2/raw_*.csv`、`e1_rerun_v2/`、`v7_rerun/{ag,rs}/`、`e5_rs_map_v1/raw_*.csv` | 用 `scripts/analyze_v1.py` 重算 |
| 層形狀定義 | `scripts/dispatch_map_v2.py` 的 `LAYERS`（AG：(N, K) 全尺寸，每卡 N/8）；`scripts/dispatch_map_rs_v1.py` 的 `LAYERS`（RS：(N, K)，每卡 K/8） | GPT-3 175B、Llama-3-70B |
| 單獨的通訊時間 | `final_*_points.csv` 的 `c_nccl_ag`、`c_flux_ag` 欄；RS 側 items CSV 的 `c_nccl_rs`、`c_nccl_ar` | 每個 M 一個值 |
| AG 延遲掃描 | `results/f2_ag_latency_v1/raw_K{8192,12288}.csv`、`log_*.txt` | NCCL、Flux 預設、7 copy 單 / 多 stream、Triton；M = 8…4096 |
| 重疊時間軸 | `results/e2_nsys/parse_G-FC1.txt`、`results/v8_nsys/parse_all.txt` | 每項的 shard copy 起訖、GEMM kernel 長度、最後一筆 copy 後 GEMM 還跑多久；可算出被藏住的通訊 |
| block 層級 | `results/v3_block_vllm_ar/{cal,eval}/raw_*.csv`、`block_summary.csv`（4 block）；`results/v4_block_L80/`（80 block） | 6 種政策，包含真正的 vLLM all-reduce |
| 現有決策表 | `results/v3_block_vllm_ar/table_{gpu,steady}.json` | oracle 對照 |
| 調校上限 | `results/v10_tune/tune_*.json`、`results/e2_tune/` | AG 側可信，RS 側不可信（V10） |
| 時脈證據 | `results/v6_phase0_repro/v6c_clock.csv`、`v6c_marks.txt`、`protocol_ab.json` | NCCL 隨 SM 時脈變 |
| 共用錨點 | `common/cost_model/params.json` 的 `flux_dispatch`、`flux_dispatch_phase2`、`flux_dispatch_verification` 三區 | — |

### A.2 環境與指令

**Flux 量測**（pixi）：

```bash
python3 common/measure/exclusive_guard.py --log <dir>/guard.log -- \
  pixi run --manifest-path pixi.toml ./launch.sh <script> [args]
```

**含 vLLM 的量測**（vLLM venv）：

```bash
python3 common/measure/exclusive_guard.py --log <dir>/guard.log -- \
  bash ws/fusion-dispatch/scripts/launch_vllm_env.sh <script> [args]
```

- venv 在 `/home/rogerlee/venvs/vllm085-flux`（repo 外）。不存在就跑 `bash ws/fusion-dispatch/scripts/setup_vllm_venv.sh`，會釘選 transformers 4.51.3 並下載 Qwen2.5 tokenizer。
- 長時間工作放 tmux：`tmux new -d -s <name> 'bash <script>'`。用 `verify_done/` 這類完成標記，讓腳本可以接續。
- **TP < 8**：`launch.sh` 第 24 行把 `nproc_per_node` 固定為全部 GPU（`nvidia-smi --list-gpus` 不理會 `CUDA_VISIBLE_DEVICES`）。G3 要新增 `scripts/launch_tp.sh`：設 `CUDA_VISIBLE_DEVICES` 並傳入 `--nproc_per_node=<TP>`，不要改 `launch.sh`。
- git：分支 `fusion-dispatch`；push 用 `git push git@github.com:allennnnnz/flux.git fusion-dispatch`，因為 origin 的 HTTPS 沒有憑證。

### A.3 已知的坑（每條都實際踩過）

| 坑 | 處置 |
| --- | --- |
| `flux.testing.initialize_distributed()` 呼叫 `init_seed()`，把 cuBLAS 設成非 production（`CUBLAS_WORKSPACE_CONFIG=:16:8`、deterministic、關 reduced precision）→ torch.mm 的 launch 變 5 倍慢，部分形狀 +26% | 初始化後立刻還原。照 `dispatch_map_v2.py` 結尾的寫法 |
| Flux `AGKernel.forward` 不能 capture 進 CUDA graph（cp_stream 未 join） | graph 模式下跳過（`dispatcher_v1.NOT_CAPTURABLE`） |
| capture 過 NCCL 的 process 在 `destroy_process_group()` 會卡住 | 寫完結果後 `os._exit(0)` |
| 80 block、多個 M 在同一個 process 時曾卡死（V11 未重現） | 每個 M 開一個 process |
| torchrun 會把 `--n` 當作自己參數的縮寫 | 參數名避免 `--n` |
| `pkill -f <pattern>` 會比對到自己的 shell 指令 | 用 PID，或 `grep "[x]yz"` 的寫法 |
| bf16 / fp16 不支援 Flux 的 `use_cuda_core_ag`、`use_cuda_core_local` | 不要用 |
| 每個 run 的第一個 M 容易在 GPU 升頻時被 clock 過濾掉 | warmup ≥ 30（M=8 用 100） |
| 持續負載下有功耗上限（約 1140–1245 MHz），短 run 則在 1410 MHz | 記錄時脈並分開處理；**決策表 / 校準必須在部署的時脈狀態下做** |
| 長 graph 中 c10d NCCL 會超線性變慢 | graph 中的 SP 路徑考慮改用 vLLM 的 pynccl |
| vLLM 0.8.5 V1 一定會載入 tokenizer；transformers 5.x 不相容 | 使用 `setup_vllm_venv.sh` |
| **以下為 G1–G3 新增（2026-10-02 / 03）** | |
| 同一 process 先建立再銷毀一組 Flux op、再建第二組，會隨機卡死 | 一組形狀一個 process（`run_calibration_v1.sh`、`run_g3_map_v1.sh`），或建好就不銷毀 |
| `launch.sh` 不理會 `CUDA_VISIBLE_DEVICES` | 少於 8 卡用 `scripts/launch_tp.sh <TP>` |
| Flux 登錄表 `// PCIE` 區段的 config 優先生效，在 NVLink 上反而慢 | 預測器把它當風險訊號（`flux_config.py` 的 `tuned_for`） |
| cuBLAS 在特定形狀 × M 慢 22–53%（斷崖），任何 GEMM 模型都預測不到 | G4 起 GEMM 改用實測（單卡、每形狀幾秒） |
| 自由參數會吸收模型沒有的效應：kernel 啟動時間被擬合到 0.11 ms 來吸收競爭（nsys 只有 0.03–0.05）；GemmRS 6 參數擬合讓 η 衝到上限 1.0 | 能直接量的常數就固定（nsys）；參數加物理上限（η ≤ 1、頻寬 ≤ 2.039 TB/s）；優先少參數（GemmRS 共用 gemm_only 參數後，推廣誤差 16% → 7%） |
| 斷崖點會把擬合拉偏（例：重疊期間變慢比例被 G-FC1 M=3072 拉到 0.3） | 用穩健損失（中位數 \|log 誤差\|） |
| steady 模式同 stream 連發的 peer copy 會跨呼叫重疊，逐筆 copy 時間不是單次時間（算出的同步成本會變負） | 需要單次時間的量一律用 gpu 模式 |
| Flux AG 同步成本不是常數，隨卡數變化（8 / 4 / 2 卡：0.053 / 0.033 / 0.014 ms） | 每種卡數各校準一次；要外推卡數需加這一項 |
| 極小形狀時「NCCL 再 cuBLAS 依序執行」≠ 兩者單獨相加（±5–16 µs） | 模型用相加；這些點總時間只有 0.04–0.06 ms，regret 比例會放大 |
| 只在「預測差距小」時探測，抓不到「有信心但錯」的斷崖 | 探測規則要加風險訊號（PCIe config、GEMM 實測與模型差很多） |
| 校準形狀若與評估層相同，會變成用答案擬合 | 校準形狀刻意避開評估層與登錄表條目（`calibrate_hw_v1.py`） |
| 評估新情境前沒先寫下預測，事後就分不清是預測還是調參 | 量測前把預測寫成檔案並 push（G3：`results/g3_predictions/`）；看過結果才做的分析要標「事後」 |
| 除錯時刪掉卡住那次的部分 log，事後少了證據 | 失敗的輸出也保留（改名，不刪） |
| `analyze_v1.py` 在沒有 B / C / D 項目時會出錯（例如校準數據） | 校準數據用 `fit_calibration_v1.py` 自己的彙整 |

### A.4 每個階段的第一步

- **G0**：文獻已整理成 `reports/20261002_related_work.md`（本 session 完成）。只需在論文定位時補讀 PDF 原文，例如 FlashOverlap、KernelSight-LM、TileSight、2609.07536。
- **G1**：
  1. 建 `common/cost_model/predictor/`，先寫 `comm.py`（分段 α-β）與 `gemm.py`（roofline + waves）；
  2. 用 `final_*_points.csv` 的元件欄（c_*）擬合，再組合成 arm 的預測，與 A / B / C / D 的實測比較；
  3. 以 `policy_eval_v1.py` 的 regret 定義評估；
  4. 推廣軸 (B)：只用 GPT-3 的層擬合，預測 Llama 的層，反之亦然。
  不需要 GPU。
- **G2**：寫 `calibrate_hw_v1.py`，重用 `ag_latency_v1.py` 的通訊項目與 `dispatch_map_v2.run_mode`，產生 `common/cost_model/hw_profiles/css-host-158_tp8_{light,sustained}.json`。
- **G3**：新模型形狀加進兩個 harness 的 `LAYERS`：
  - Qwen2.5-72B：hidden 8192、ffn 29568、heads 64、kv 8；
  - Llama-3-8B：hidden 4096、ffn 14336、heads 32、kv 8。
  TP=4 / 2 用 `launch_tp.sh`。
- **G4**：`dispatcher_v2.py` 重用 `dispatcher_v1.FluxDispatcher` 的後端，把 `DispatchTable` 換成預測器加探測快取；`validate_block_v4.py` 由 v3 複製後改用 dispatcher_v2。
- **G5**：先與 hetero-proxy 的 STATUS 對齊（不 enable peer access、經 host），並註明 Flux 路徑不可用。

### A.5 待使用者 / boss 決定的事（延續中）

1. 實驗室是否有其他型號的 GPU，用於跨機器驗證（要問教授）；
2. auditor session；
3. `CLAUDE.md` 是否加入守衛規則與陷阱；
4. `PHASE0_FINDINGS.md` 2.6 的改寫。
