# ws/fusion-dispatch — STATUS

**本 workstream 唯一權威。** 第 1 節由 boss 寫入，worker 不改；第 2 節起由 worker 維護。

建立：2026-09-29（boss）
最後更新：2026-10-02（boss 兼 worker，G3 完成）
狀態：**第三階段（泛用決策器）G3 完成，下一步 G4**（D-006 / D-008：最高優先；有證據前不改 `src/`）

---

## 0. 快速跟上（新 session 先讀這節；每次收工更新）

最後更新：2026-10-02

**一句話**：查表版決策器已完成並驗證（對真 vLLM 0.8.5：prefill 快 9.3–17.7%，decode M ≤ 384 同速、M=512 快 10–14%，regret ≈ 0）。
第三階段（泛用決策器）**G0–G3 完成**：
- G2：只用 4 分鐘校準微基準建參數檔，對全部 320 個既有實測點 regret AG 1.15–1.74%、RS 0.10–0.36%；
- G3：在沒量過的模型與卡數上**先預測、後量測**（預測在量測前 push）。只用預測器：
  - AG 1.76%（8 卡新模型 0.89%、4 卡 1.77%、2 卡 2.81%）；
  - RS 0.37%。
- G3 錯誤幾乎全來自 **cuBLAS 自己的斷崖**（某些形狀 × M 慢 22–53%），通訊與重疊模型都準。
  事後分析：GEMM 改成實測（單卡、便宜）、通訊 / 重疊用模型，AG regret 1.76% → 0.27%。

**G4 op 層級完成（2026-10-03，預先登記，全新測試集：Qwen2.5-32B 8 卡 / 4 卡、Llama-3-8B 4 卡，M 皆為沒用過的值）**：

| 方法 | regret |
| --- | --- |
| **模型 + 實測單卡 GEMM（g4）** | **0.17%**（AG 0.04%、RS 0.41%） |
| 加多卡把關（g4+probe） | 0.11% |
| G3 方法（只用模型） | 1.20% |
| 固定門檻 | 2.30% |

- 成本：單卡 GEMM 82 秒 = 量整張表的 7.7%；多卡把關另需 367 秒（3% 變體只探 16% 的點，效果幾乎一樣）；
- 結果：`results/g4_map/eval_g4_log.txt`；
- 新坑：steady 模式量極小 kernel 時量到的是 CPU 發 kernel 速度，數值取決於同一輪的其他項目（同一 GEMM 在探測與地圖中可差 3 倍）。

**下一步：G4 block 層級**（使用者 2026-10-03 核准做法，D-009）：先跑 `scripts/run_g4_block_prep_v1.sh`（vLLM all-reduce 校準 + block 用 M 的單卡 GEMM），再依 `build_g4_block_v1.py` 的步驟進行。
原 G4 設計（`reports/20261002_g3_unseen.md` 第 0 節）：
- 決策 = 模型（通訊 + 重疊）+ 實測單卡 GEMM + 少量多卡探測；
- GemmRS 改用 gemm_only 家族參數；
- 用 `scripts/build_table_v2.py` 產生 `dispatcher_v1` 格式的表；
- 在 vLLM 環境做 block 驗證（含 TP=4），比較所有對照組。

**G1–G3 的重點**（`reports/20261002_g1_predictor.md`、`reports/20261002_g2_calibration.md`、`reports/20261002_g3_unseen.md`）：
1. 融合 kernel 能否「邊傳邊算」，可從 CUTLASS stream-K 的 tile 排程直接模擬：
   - 自己算出 M ≤ 2048 不重疊、≥ 3072 才重疊；
   - 對沒看過的 Phase 0 形狀也對 → 可能就是 diag-overlap 要找的原因 **[推論]**。
2. 剩下的大錯誤都是 Flux 登錄表「PCIe 區段」的 config（本機慢 12–58%）。
   用它當風險訊號實測可抓到；這條規則是看過結果才定的，G3 沒測到（G3 的點都沒命中登錄表），仍待驗證。
3. 隨機森林在量過的形狀上準，換新形狀就退化。
4. 坑：同一 process 換一組 Flux op 會隨機卡死 → 校準一組形狀一個 process。
5. G3 證實模型的大膽預測：4 卡 / 2 卡時 Flux 在 M=136–512 就勝出（8 卡要到 1024–3072）。

計劃：`reports/20261002_plan_general_dispatcher.md`（含附錄 A）。文獻：`reports/20261002_related_work.md`。

**決策器 v1（目前可用的版本）怎麼決定**：查表，全部來自實測。
- 離線校準：每種層形狀 × 每個 M，量每條路徑，排序存成 JSON；
- 執行時：
  1. 依 M 選切法：TP+AllReduce 或序列平行；
  2. 序列平行時逐層選 Flux / NCCL；
  3. CUDA graph 中跳過不能 capture 的 AGKernel；
  4. 8 卡同一張表並核對 hash。
- 程式：`scripts/dispatcher_v1.py`、`build_table_v1.py`；表：`results/v3_block_vllm_ar/table_{gpu,steady}.json`。

**報告讀這幾份就夠**（依時間序）：

| 報告 | 內容 |
| --- | --- |
| `reports/20260930_e1_e3_dispatch_map.md` | 單層地圖：小 M 不開 Flux 較快，以及原因 |
| `reports/20260930_f1_block_validation.md` | 兩層決策器在 block 中有效 |
| `reports/20260930_f2_ag_latency.md` | Triton AllGather 原型比 NCCL 快 |
| `reports/20260930_verification.md` | V1–V12 驗證：真 vLLM、80 block、獨佔重測、各種疑點 |
| `reports/20261002_f4_feasibility.md` | 本機可做 vLLM 端到端；整合計劃 |
| `reports/20261002_related_work.md` | 約 100 篇相關工作；缺口與定位 |
| `reports/20261002_plan_general_dispatcher.md` | **第三階段計劃（泛用決策器）** |
| `reports/20261002_g1_predictor.md` | **G1：預測器與推廣測試（關卡通過）；融合 kernel 排程模擬** |
| `reports/20261002_g2_calibration.md` | **G2：4 分鐘校準 → 只用微基準的參數檔，對全部實測評估** |
| `reports/20261002_g3_unseen.md` | **G3：新模型 / 4 卡 / 2 卡，預先登記的預測 vs 實測；cuBLAS 斷崖** |

**環境**：
- Flux 量測：`pixi run --manifest-path pixi.toml ./launch.sh <script>`。
- 含 vLLM 的量測：`bash ws/fusion-dispatch/scripts/launch_vllm_env.sh <script>`，使用 venv `/home/rogerlee/venvs/vllm085-flux`。
  venv 在 repo 外；不見了就跑 `scripts/setup_vllm_venv.sh`。
- **所有量測都要經 `common/measure/exclusive_guard.py`**，確保無其他使用者 / 程序。
- 長時間工作放 tmux（例如 `tmux new -d -s X 'bash ...'`），斷線也不會停。
- git：工作分支 `fusion-dispatch`；push 用 SSH 網址 `git@github.com:allennnnnz/flux.git`（origin 的 HTTPS 沒有憑證）。

**預測器怎麼用**：`common/cost_model/predictor/`（純 Python，系統 python3 即可）。
- 校準（GPU，約 4 分鐘）：`tmux new -d -s g2cal 'bash ws/fusion-dispatch/scripts/run_calibration_v1.sh <out_dir>'`；
- 擬合：`python3 ws/fusion-dispatch/scripts/fit_calibration_v1.py <out_dir>`，寫到 `common/cost_model/hw_profiles/css-host-158_tp8_{gpu,steady}.json`；
- 評估：`python3 ws/fusion-dispatch/scripts/eval_profile_v1.py`（G2）；`eval_predictor_v1.py`（G1，用決策表元件擬合的對照）。

**待使用者 / boss 決定**（細節見 PROJECT.md 第 3 節）：
1. 是否有其他型號 GPU 可做跨機器驗證（問教授）；
2. 是否開獨立 auditor session；
3. `PHASE0_FINDINGS.md` 2.6 是否改寫為「依 SM 時脈而定」；
4. G1 對 diag-overlap 的假說（Phase 0 N=4096 不重疊是 stream-K 排程造成），待 diag-overlap 進場時處理。

**已決定（2026-10-03，D-009）**：
- G4 改用「模型 + 實測單卡 GEMM + 少量多卡把關」，要用新數據驗證；
- 陷阱已寫入 `CLAUDE.md` 陷阱表與 5.1 第 11 條，本 ws 的建模坑在計劃附錄 A.3。

---

## 1. 目標與成功標準（boss）

### 問題

推論時每次 forward 的 token 數 M 都在變。論文（Section 6 / Fig. 14；Section 5.2 / Fig. 17）指出
極小 m 時 Flux 可能比不 overlap 的基準慢。要做一個決策器：每一層依本次 M 決定走 Flux fused
或某條非 fused 路徑。本 workstream 先量出做這個決策需要的地圖，再評估哪種決策器夠用。

完整設計、依據與程式碼行號見 `reports/20260929_experiment_design.md`。以下是摘要。

### 關鍵已知

- 現有資料沒有任何一點顯示 off 比 on 快；M < 1024 完全沒量過；cuBLAS GEMM 的單獨時間沒量過。
- **[程式碼]** 調好的 config 只在 (m, n, k) 完全相符時生效，否則 fallback 到與形狀無關的預設
  （`include/flux/op_registry.h:190-203`）。推論時多數 M 會 fallback。
- **[程式碼]** 每個 tile 要等它 M 範圍內所有 shard 到齊（`src/ag_gemm/sm80_all_gather_gemm.hpp:927-939`）。
  **[推論]** M ≤ TILE_M 時 fused 結構上不可能 overlap。

### 實驗臂（AG+GEMM）

A `fused`（AGKernel.forward，All2All + use_read）· B `nccl_cublas` · C `fluxag_cublas` ·
D `fluxag_fluxgemm`（串行，診斷）· A\* `fused_tuned`（條件性，需 rebuild）。
另外同輪單獨計時四個元件：NCCL AG、Flux AG、cuBLAS、Flux gemm_only。

### 形狀

- 層（TP=8，bf16）：G-FC1 (6144, 12288)、G-QKV (4608, 12288)、L-QKV (1280, 8192)、L-GU (7168, 8192)，
  加 Phase 0 對照 P0-4096、P0-8192。括號內為每卡 (n, K)。
- M：主網格 8–16384（2 的冪次，12 點）+ 保留點 8 個 + 斷崖對 4 組。

### 步驟

- **E0**：harness、正確性、錨點重現（設計 5.4）、GPU 端對齊 vs host barrier 對照。
- **E1**：地圖。每個 (層, M) × 全部 arm，單次延遲與穩態連發兩種模式。
- **E2**：切換點兩側的三項分解、斷崖對、nsys；視結果決定是否做 A\*。
- **E3**：決策器評估。π_on / π_off / π_thr / π_lut / π_model 對 oracle 的 regret，
  在保留點與三個合成 trace 上計算；另量決策開銷。
- **E4**：CUDA graph capture 可行性、兩路徑常駐記憶體。
- **E5**：GEMM+RS 側（E1–E3 經 auditor 審查後才開）。

### 成功標準

1. E0 錨點全部在 20% 內；5–20% 的偏差有解釋。
2. E1 地圖完整（4 主層 × 20 M × A–D 與四元件，兩種模式）：≥ 200 輪交錯、rank-max median、
   clock 過濾，原始逐輪 CSV 已提交。每一點標為「有勝負」或「平手」。
3. 每層給出 M\*（或「[8, 16384] 內無切換點」），切換點下方、附近、上方各附一份 nsys。
4. E2 分解在每層 ≥ 2 點上三項相加與 t_A − t_B 相差 < 10%。
5. E3 給出建議策略：在三個 trace 上 regret ≤ 1%，且決策開銷 < 最小有勝負 |Δ| 的 1%。
   做不到時說明原因——這也是結論。

### 量測規範

照 `CLAUDE.md` 5.1，另加兩條本 ws 特有的規則（設計 5.1）：

- **GPU 端對齊**：待測項前在同一 stream 跑一個 1 元素的 NCCL all_reduce；
- **L2 flush**：待測項前寫 ≥ 80 MB。

小 M 時 op 只有數十 µs，這兩條不做，量到的是 skew 與 L2 熱度。

### 待 boss 裁決

- `src/` 的改動（A\* 需在 pybind `forward` 暴露 `hparams` 並 rebuild）與 diag-overlap 的 kernel instrument 衝突，需排先後。
- 設計 3.1 的附帶發現要轉給 diag-overlap：Phase 0 的 N=4096 跑的是 fallback config。

### 第二階段（2026-09-30，D-007）

目標：做出能實際降低推論通訊延遲的決策器，並修 Flux 的弱點。
計劃與成功標準見 `reports/20260930_plan_dispatcher.md`；順序 F0 → F1 → F2 / F3 → F4。
目標框架：vLLM 0.8.5.post1，off 選項加入「TP + AllReduce」。
改 `src/` 必須先通過關卡 G2 / G3。

### 第三階段（2026-10-02，D-008）：泛用決策器

查表（v1）已證明可行，並提供 oracle，但不泛用：換模型 / TP / 硬體 / 時脈都要重量，且晶片未到無法量。
目標是「物理模型預測 + 少量實測把關」的決策器，在未見過的形狀、TP 卡數、時脈狀態、PCIe 代理上，以對 oracle 的 regret 評估。

- 計劃與成功標準：`reports/20261002_plan_general_dispatcher.md`；
- 預測器放 `common/cost_model/predictor/`，同時完成 ws/cost-model 的模型目標；
- 成功標準：未見形狀 / TP 上，block regret ≤ 2%；校準 ≤ 10 分鐘；探測 ≤ 表點數的 15%。

---

## 2. 目前成立（worker 維護）

最後更新：2026-10-02（加第三階段 G1）。報告：`reports/20260929_e0_anchor.md`（E0）、`reports/20260930_e1_e3_dispatch_map.md`（E1–E3）。
**尚未經 auditor 審查。**

1. **decode 大小的 M 不開 Flux 較快，四層一致。**
   - M ≤ 512 時，NCCL all_gather + cuBLAS 在每層、每點、gpu 與 steady 兩種模式下都勝過 Flux fused
     （86/88 點，另 2 點平手，為 L-GU M=512）。
   - 所需時間是 fused 的 0.32–0.90 倍。
   - 來源：`results/e1_merged_points.csv`。方法：交錯 ≥ 200 輪、逐輪 rank-max 中位數、L2 flush、GPU 端對齊。
2. **切換點在 M ≈ 1024–3072，依層而定**；G-FC1、L-GU 不單調（有斷崖）。細節見 E1–E3 報告第 0、1 節。
3. **M ≤ 512 的損失歸因**（三項分解，156/160 點相加誤差 < 10%）：
   - Flux AG − NCCL AG 佔 75–90%：Flux AG 在 M ≤ 64 約 0.10–0.11 ms，NCCL 0.03–0.05 ms；
   - Flux GEMM − cuBLAS 佔 15–45%；
   - overlap 只收回約 0.014 ms。
4. **[nsys] M=64 沒有 overlap**：fused GEMM 在最後一筆 shard copy 後仍需 ≥ 單獨 GEMM 的時間。
   **[程式碼 + nsys]** Flux All2All 在同一 stream 串行發 8 筆 copy。
   來源：`results/e2_nsys/parse_G-FC1.txt`；`src/coll/ths_op/all_gather_op.cc:553-576`。
5. **斷崖**：
   - 上游 registry 的 config 在本機可能很差：G-FC1 M=1024，Flux GEMM 比 cuBLAS 慢 1.58×；L-GU M=4096，慢 0.67 ms。
   - fallback config 也可能很差：G-FC1 M=3072，fused kernel 比串行慢 0.53 ms；nsys 顯示資料到齊後仍多花 585 µs。
6. **調校上限**（`profiling()`，不 rebuild；估計值 [推論]）：
   - decode 區 ≤ 6%，off 仍快 18–139%；
   - 斷崖點 25–28%，其中 G-FC1 3072、L-GU 4096 會翻為 fused（比 off 快 9–17%）。
   - 來源：`results/e2_tune/`。
7. **決策器**：每層查表（π_lut）在 decode / prefill / 混合 trace 的 regret：
   G-QKV、L-QKV 0%，L-GU ≤ 0.94%，G-FC1 1.6–2.0%；永遠開 Flux 在混合 trace 為 6–24%。
   查表開銷 0.24 µs/次。殘餘 regret 全來自不在查表網格上的斷崖 M → 查表必須建在部署實際出現的 M 上。
   來源：`results/e3_policy_eval_log.txt`。
8. **量測方法**：`flux.testing.initialize_distributed()` 把 cuBLAS 放在非 production 狀態
   （torch.mm launch 13 → 71 µs；部分形狀 GPU 時間 ±26%）。任何用它做的 Flux vs torch 比較都偏向 Flux。
   來源：`results/e0_anchor/cublas_settings_v1.txt`。
9. **錨點**：新協定下 Phase 0 錨點全部在 20% 內；gemm 的偏差由 SM clock 解釋。
   Phase 0 是在持續負載的功耗上限（約 1170 MHz）下量的。

### 第二階段（2026-09-30）

報告：
- `reports/20260930_f04_cuda_graph.md`（F0.4）
- `reports/20260930_f1_block_validation.md`（F1）
- `reports/20260930_f2_ag_latency.md`（F2a / F2b）

E5 的 Llama 兩層結果在 `results/e5_rs_map_v1/`。**全部尚未經 auditor 審查。**

10. **CUDA graph**（F0.4）：
    - **Flux AGKernel 不能 capture**（`capturing stream has unjoined work`）：`cp_stream` 上的 peer copy 沒有 join 回主 stream，
      見 `all_gather_op.cc` / `all_gather_gemm_op.cc:271-276`。
    - GemmRS、AllGatherOp+cuBLAS、NCCL 各路徑都能 capture，重放 bitwise 正確。
    - graph 重放比 gpu 模式 eager 慢：NCCL 路徑 7–15 µs，Flux GemmRS 3–4 µs。
11. **GEMM+RS 側**（E5，四層）：切換點遠低於 AG 側。GPT-3 G-O / G-FC2 同模式（M ≥ 264 Flux 勝）。
    - L-O、L-down 在 M ≤ 16–24 off 勝，32–264 多為平手，**M ≥ 256 起 Flux GemmRS 一路勝出**（最多快 30%）。
12. **決策器需要兩層**（F1，v2 公平版）：
    - 先選切法：vLLM 預設 TP+AllReduce vs 序列平行；
    - 再在 SP 下逐層選路徑。
    - 兩層決策器在驗證 run 上 regret 0.00%（decode graph）/ 0.02%（prefill）。
    - 比 vLLM 預設：**prefill 快 10.4–18.3%**（合計 16.6%）；decode graph M ≤ 256 同速（選 vLLM 預設）、M=512 快 10.0%。
    - 比永遠開 Flux：prefill 快 2.1%，其中 M=4096 快 7.6%。
13. **單層表可預測 block 差距**：prefill 中位誤差 5.5%、decode graph 9.0%，正負號全對；
    **eager 小 M decode 不可**（中位 118%，9 次正負號錯），原因是 CPU launch 瓶頸。
14. **F2a**：`use_cuda_core_local` / `fuse_sync` 對 bf16 不支援（`local_copy_and_reset.cu:232`）；跳過 local copy 只省 1–14 µs。
15. **F2b**：單一 Triton kernel 的 AllGather（平行 flag barrier + P2P gather）在 gpu 模式**每個 M 都比 NCCL 快 8–22%**，
    比 Flux 快 1.1–4.2×。M ≤ 64 為 23.6–45.1 µs，**通過關卡 G2**。eager 下被 Triton 的 Python launch 開銷抵銷。

### 驗證輪（2026-09-30 – 10-02，`reports/20260930_verification.md`）

16. **vLLM 0.8.5 實況**（讀原始碼）：
    - 預設 TP + AllReduce；
    - custom AR 用於 < 8 MiB（Llama-70B 為 M < 512）；
    - CUDA graph 桶為 token ≤ 512，piecewise，attention 在 graph 外；
    - 序列平行預設關。
17. **真正的 vLLM all-reduce**（V3）：custom AR 在 decode 比 torch NCCL 快 10–12%。
    兩層決策器在 decode M ≤ 384 選 vLLM 預設（同速），M=512 快 10.1%，prefill 快 9.8–17.7%，regret 0.00%。
18. **vLLM 若把 custom AR 門檻調到 64 MiB**（V3b）：仍勝，decode M=512 快 2.3%、prefill 快 5.1–10.2%。M=512 的勝幅大多來自門檻。
19. **80 block**（V4 / V4b）：決策與 4 block 相同，regret 0.00–0.02%；decode M=512 快 13.6%，prefill 快 9.3–12.0%。
    **長 graph 中 torch c10d NCCL 每次成本上升**（SP 全 NCCL 為 1.16–1.57× 線性外推），custom AR 不受影響。
20. **graph 部署用 gpu 模式表即可**（V5）：28 個可比點全部一致或平手。
21. **Phase 0「Flux AG 快 13–22%」與本 ws 的差異是 SM 時脈狀態**（V6c）：
    原腳本量 NCCL 時 GPU 在 1155 MHz，NCCL 用 SM 搬資料受時脈影響，Flux 用 copy engine 不受影響 **[推論，時間對應]**。
    兩者都對；**決策表須在部署的時脈狀態下校準**。
22. **Triton AllGather 原型必須雙 buffer**（V9b）：單一 buffer 在 barrier 後的延遲窗中大量讀錯（最多 33,612 / 40,000），雙 buffer 0 錯。
23. **四個 AG 層 M ≤ 512 無重疊，≥ 3072 才有效**（V8 nsys）。
24. **表必須涵蓋每個部署的 graph 桶**：M=384 不在表內時，借用 512 的選擇會選錯（V3 + V5）。
25. **RS 側（GemmRS）調校上限不改碼無法確認**（V10，profiling 連發協定與交錯量測不符）。
26. **V4 的 NCCL-in-graph 卡死**：最小重現未重現，原因未定（V11）。
27. **獨佔性**（V12）：守衛下重測 5 組關鍵點全部 CLEAN，與 9/30 差 ≤ 2%。之後量測一律經 `common/measure/exclusive_guard.py`。

28. **F4 可行性**（`reports/20261002_f4_feasibility.md`）：
    - 原版 vLLM 0.8.5 V1 跑通：Llama-3-70B dummy、TP=8、batch 64 × 1024 + 128，平均 3.591 s，5 次 ±0.2%；
    - 需釘選 transformers 4.51.3，並使用 Qwen2.5 tokenizer。

### 第三階段 G1（2026-10-02，`reports/20261002_g1_predictor.md`，尚未經 auditor 審查）

來源：`results/g1_predictor/`（`scripts/eval_predictor_v1.py`，不需 GPU，重跑結果完全相同）。

29. **預測器推廣**：只用預測器（不量測）的 op 層級 regret。

    | 軸 | AG 側 | RS 側 |
    | --- | --- | --- |
    | GPT-3 → Llama | 1.40% / 1.58%（gpu / steady） | 0.03% / 0.12% |
    | Llama → GPT-3 | 2.29% / 1.95% | 0.13% / 0.14% |
    | gpu ↔ steady | 1.74% / 2.04% | 0.15% / 0.11% |
    | 沒量過的 M | 2.32% / 2.09% | 0.31% / 0.34% |

    - 對照：固定門檻（M ≤ 512 不開）AG 1.99–4.80%；永遠開 5.2–10.3%。
    - **G1 關卡（B、D ≤ 3%）通過。**
30. **混合（預測 + 實測把關）**：AG 側 0.08–0.25%、RS 側 0–0.11%，探測 14–28% 的點。
    - 計劃原定「Flux 用預設 config 就實測」要量 42–62%，太貴；
    - 改成「用 PCIe 調校 config 才實測」（看過結果才定，G3 要重驗）。
31. **融合 AG+GEMM 的時間可由 tile 排程模擬算出**（CUTLASS stream-K + Flux 列輪轉 + 每 tile 等 shard，逐行照原始碼）：
    - 用實測元件時，中位誤差 0.8%（gpu）/ 1.6%（steady）；
    - 自己算出 M ≤ 2048 不重疊、≥ 3072 才重疊。原因 **[程式碼 + 模型]**：
      stream-K 工作量 ≲ 1.5 波時，每個 block 都碰到最晚的 shard；
    - 沒看過的 Phase 0 形狀：P0-4096 M=4096 藏住 3%（實測 7%），P0-8192 70%（實測 65%），決策全對；
    - **[推論]** diag-overlap 的「N=4096 只重疊約 2%」就是這個原因。換 data-parallel config 預測可從 0.681 降到 0.54–0.58 ms。
32. **斷崖來源**：模型最大的錯誤（G-FC1 M=1024 45%、L-GU M=4096 16%）都是登錄表 `// PCIE` 區段的條目
    （本機 Flux GEMM 比 cuBLAS 慢 1.23–1.58×）。「這些是為 PCIe 拓撲調的」是從區段標題推得 **[推論]**。
33. **限制**：
    - 融合比串行慢的點（M=136 / 264、G-QKV 2048、G-FC1 3072）模型不含；
    - 重疊期間 GEMM 變慢比例 κ 依訓練集為 0–0.125，不穩定；
    - 每 SM 多個 block 時低估重疊；
    - gpu ↔ steady 的差異主要是量測方式而非時脈，NCCL 時脈修正無法從 E1 數據擬合（γ = 0）。

### 第三階段 G2（2026-10-02，`reports/20261002_g2_calibration.md`，尚未經 auditor 審查）

來源：`results/g2_calibration/`；參數檔 `common/cost_model/hw_profiles/css-host-158_tp8_{gpu,steady}.json`。

34. **校準 4 分 2 秒**（6 個 process 含啟動），守衛 6 段 CLEAN，正確性檢查全過。
    - 形狀避開 8 個評估層，也不命中 Flux 登錄表；
    - 內容：通訊 hidden 6144 × 13 個大小；GEMM / 融合 3 個 AG 形狀 + 2 個 RS 形狀 × 5 個 M。
35. **只用校準數據擬合，評估全部 320 個既有實測點**：

    | 參數檔 → 數據 | AG 只用預測 | AG 混合（hyb+pcie） | RS 只用預測 | RS 混合 |
    | --- | --- | --- | --- | --- |
    | gpu → gpu | 1.15% | 0.12% | 0.36% | 0.09% |
    | steady → steady | 1.74% | 0.06% | 0.10% | 0.09% |
    | 交叉 | 1.30–1.88% | — | 0.08–0.40% | — |

    - 與 G1 用決策表元件擬合的樣本內結果相當；
    - 錨點全部 < 10%。
36. **預測誤差**：
    - AG 路徑 MAPE 4–7%，接近切換點 4–5%；
    - RS 路徑 A 為 16.4%（gpu）/ 8.2%（steady）：GemmRS 參數抵換，gpu 版 η = 1.000 碰上限，不可當物理效率；
    - RS 的決策仍準，但預測的切換點偏晚（G-FC2、L-down）。
37. **探測量**：
    - 事先規則（ε = 校準殘差 p90 ≈ 10.7%）要量 22–23%；
    - ε 掃描顯示 ε ≈ 5% 只需 10–16%，regret ≤ 0.32%；
    - G4 要用不看測試集的方式定 ε。
38. **nsys 的 kernel 啟動常數**（唯一不是來自校準集的輸入）：±30% 或改常數，regret 變化 ≤ 0.11 個百分點，沒有實質洩漏。
39. **坑**：同一 process 先建立再銷毀一組 Flux op、再建第二組，會隨機卡死（8 rank 都在 synchronize）。
    - 一組一個 process 即正常；
    - 證據：`results/g2_smoke/debug_hang2/log.txt`；
    - 建議加進 `CLAUDE.md` 陷阱表。

### 第三階段 G3（2026-10-02，`reports/20261002_g3_unseen.md`，尚未經 auditor 審查）

來源：`results/g3_predictions/`（量測前 push，`642bf6a`）、`results/g3_map/`、`results/g3_calibration_tp{4,2}/`。

40. **4 卡 / 2 卡校準**：3 分 4 秒 / 2 分 33 秒，守衛 CLEAN。
    Flux AG 同步成本隨卡數變化（8 / 4 / 2 卡：0.053 / 0.033 / 0.014 ms）。
41. **預先登記的結果**（16 組 × 10 個 M × 2 模式 = 320 點，量測 54 分鐘，守衛全 CLEAN）。只用預測器的 regret：

    | 組 | AG | 固定門檻（AG） | RS |
    | --- | --- | --- | --- |
    | 全部 | 1.76% | 1.97% | 0.37% |
    | 8 卡（新模型） | 0.89% | 0.92% | — |
    | 4 卡 | 1.77% | 1.86% | 0.00% |
    | 2 卡 | 2.81% | 3.65% | — |

    - 預先登記的把關規則量一半的點，只降到 AG 1.21%。
42. **錯誤來源**：cuBLAS 斷崖（2 卡 L8-GU M=136 慢 49%、8 卡 L8-GU M=1024 慢 53%、4 卡 L-GU M=136 慢 22%）。
    - 通訊（NCCL 0–10%、Flux AG 1–8%）與 Flux gemm_only 都準；
    - 斷崖的預測差距大，所以「差距小才探測」抓不到。
43. **模型的大膽預測正確**：4 卡 / 2 卡時 Flux 系路徑從 M=136–512 起就最快。
44. **事後分析**（G4 設計用）：單卡 GEMM 用實測、通訊 / 重疊用模型，AG regret 1.76% → 0.27%（2 卡 2.81 → 0.13%）；RS 沒改善（GemmRS 模型待修）。
45. PCIe 調校 config 規則在 G3 沒有被測到（沒有點命中登錄表）。

## 3. 撤回表（worker 維護）

| 撤回主張 | 出處 | 原因 | 替代 |
| --- | --- | --- | --- |
| E0 v1 的 B、C、c_cublas 數字，以及由它們推出的判決（例：P0-4096 M=4096「fused 快 12%」） | `scripts/dispatch_map_v1.py`、`results/e0_anchor/` | cuBLAS 在 `init_seed()` 設定下量測，非 production | `dispatch_map_v2.py`、`results/e0_anchor_v2/`（該點改為平手） |
| 設計 H2 的「g=1 時差額 = fusion 固定開銷」 | 設計文件第 8 節 | 實測 A − D ≈ −0.014 ms，fused 不比串行差；損失來自 Flux 元件 | E1–E3 報告 2.1 |
| 設計 H6 的「小 M 時單次與穩態判出的勝者不同」 | 設計文件第 8 節 | 兩模式在 M ≤ 1032 判決一致 | E1–E3 報告第 4 節 |
| F2 報告「Triton 原型 barrier + gather 應正確 [推論]」（單一 buffer） | `reports/20260930_f2_ag_latency.md` 第 2 節 | V9b：單一 buffer 在 barrier 後的延遲窗中讀錯 | 雙 buffer（依 epoch 奇偶交替），V9b 0 錯 |
| F1 v2 的「比 vLLM 預設」數字（decode 合計 3.4%、prefill 16.6%） | `reports/20260930_f1_block_validation.md` | 對照組 `tp_ar` 用 torch NCCL，不是 vLLM 的 custom AR | V3：decode 3.0%、prefill 16.1%（`results/v3_block_vllm_ar/`） |
| F1 v1 的「prefill 比 vLLM 預設快 25–32%」 | `scripts/validate_block_v1.py`、`results/f1_block_v1/` | RMSNorm / SiLU×up 未融合，對 tp_ar 多算時間 | `validate_block_v2.py`、`results/f1_block_v2/`：10.4–18.3% |

## 4. 下一步（worker 維護）

**00.（2026-10-02 起）第三階段：泛用決策器。** 照 `reports/20261002_plan_general_dispatcher.md` 的 G1 → G6 進行。
每階段結束：更新 §0、JOURNAL、PROJECT，commit 並用 SSH push。

- ✅ G0 文獻（`reports/20261002_related_work.md`）。
- ✅ G1 預測器（`reports/20261002_g1_predictor.md`），關卡通過。
- ✅ G2 校準（`reports/20261002_g2_calibration.md`）：4 分鐘校準，只用微基準評估全部實測，AG 1.15–1.74%、RS 0.10–0.36%。
- ✅ G3 新模型 / 4 卡 / 2 卡（`reports/20261002_g3_unseen.md`）：預先登記；只用預測器 AG 1.76%、RS 0.37%；錯誤來自 cuBLAS 斷崖。
- **G4（下一個）**：
  1. 決策器 v2 = 模型（通訊 + 重疊排程模擬）+ **實測單卡 GEMM**（cuBLAS、Flux gemm_only，每個部署形狀 × M 桶，幾秒）+ 少量多卡探測。
     - 用 `scripts/build_table_v2.py`（已寫好、只用預測器的版本）產生 `dispatcher_v1` 格式的表；
     - 加 GEMM 實測輸入與探測執行器。
  2. GemmRS 改用 gemm_only 家族參數（只擬合 α、β）。G2 舊數據：RS 路徑 A MAPE 16.4% → 7.2%。
  3. 探測規則：
     - 差距 < ε；
     - 風險訊號：PCIe 調校 config、GEMM 實測與模型差很多（斷崖指標）；
     - ε 用校準集內交叉驗證決定。
  4. 切法層級（TP+AR vs SP）：寫 `predictor/block.py`（AR 模型 + norm / residual 頻寬項），用 `results/v3_block_vllm_ar/` 驗證。
  5. block 驗證：`validate_block_v4.py`（由 v3 改：模型維度與卡數可設定、讀 v2 表）。
     - 至少 Llama-3-70B TP=4 與 Qwen2.5-72B TP=8；
     - vLLM 環境；校準 / 驗證分開；守衛 + tmux。
  6. 成功標準（計劃）：沒量過的形狀 / 卡數上 block regret ≤ 2%、校準 ≤ 10 分鐘、探測 ≤ 15%、接近切換點誤差 ≤ 10%。
- 其他遺留：κ 不穩定；多 block 共用 SM 時低估重疊；steady 參數檔的 NCCL 誤差；極小形狀的串行組合誤差（G3 第 3 節）。

**0. ✅ 驗證輪完成（2026-10-02）**，見 `reports/20260930_verification.md`。下一步前需使用者確認。
影響後續設計的結論：
- Triton AG 必須雙 buffer，epoch 在 device 端；
- 表要涵蓋每個 graph 桶，並在部署時脈狀態下校準；
- SP 若要在長 graph 的 decode 競爭，需要非 NCCL 的 AG / RS；
- 量測一律經 `exclusive_guard.py`。


更新 2026-09-30，依 D-007 計劃：

1. **auditor 審查**：E0、E1–E3、F0.4、F1、F2 五份報告。
2. ✅ **E5 完成**（2026-09-30）：GPT-3 兩層與 Llama 一致。G-FC2 M ≤ 72、G-O M ≤ 16 off 勝，~128–256 平手，M ≥ 264 Flux GemmRS 勝。
   保留輪次 < 200 的點：M=8 gpu（升頻）與大 M steady（功耗上限，G-O M=6144 / 8192 只剩 28 / 32 輪），皆大幅度勝負。
3. **F2c-A（不改 Flux）**：把 Triton AllGather 做成決策器的一條 AG 路徑。
   - 需要 device 端 epoch，才能 graph-safe；
   - RMSNorm 直接寫進 IPC buffer；
   - 補一版 op 層地圖後重跑 F1 block 驗證。
   預期主要改善 decode graph 的 SP 切法 **[推論]**。
4. **F2c-B（改 Flux，G2 已通過）**：替 CUDA-core All2All AG 與 local copy 補 BF16 / FP16 實例，讓 fused 路徑也用上快的 AG。
   改前打 git tag，先量完整 rebuild 時間。
5. **F3a**：對 Llama 四層所有 M 桶跑 `profiling()`，決定關卡 G3。
6. **可選：修 AGKernel 的 graph capture**（`cp_stream` join 回主 stream，約 3 行）。
   目前 decode graph 桶的 AG 側不會選 fused，**暫無必要**。
7. **F4**：在獨立 venv 裝 vLLM 0.8.5.post1，確認它的 TP 預設、custom all-reduce、CUDA graph capture size，再整合。
8. 仍待補：E1 保留輪次 < 200 的 17 點；nsys 其餘三層。

## 5. 產出到 common 的數字

`common/cost_model/params.json` 的 `flux_dispatch` 區（2026-09-30）：

- 小 M 的 AG 延遲（Flux vs NCCL）；
- 各層 M\*；
- 持續負載 SM clock；
- 新協定下的 NCCL AG 錨點。
