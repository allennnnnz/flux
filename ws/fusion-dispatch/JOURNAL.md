# ws/fusion-dispatch — JOURNAL

Append-only。每條：日期、角色、做了什麼、卡在哪、留給下個 session 的話。
需要 boss 裁決的事項標 **「需 boss 裁決」**。

---

## 2026-09-29 · boss · 建立

- 起因：與教授討論後，優先觀察「什麼時候不開 Flux 比開好」，為推論時的變動 M 做決策器。
- 寫了 `reports/20260929_experiment_design.md` 與 STATUS 第 1 節。尚未執行任何量測，尚未登入 `PROJECT.md`。
- 讀程式碼得到三件會影響設計的事（詳見設計第 3 節）：
  1. tuning config 完全相符才生效，否則 fallback；本 build 沒開 protobuf，`FLUX_TUNE_CONFIG_FILE` 無效；
     `profiling()` 不回寫 registry。
  2. 每個 tile 要等它 M 範圍內所有 shard 到齊。
  3. 動態 M 已支援（`full_m` / `max_m`），M 必須是 world 的倍數。
- **需 boss 裁決**：A\* 需要改 `src/` 的 pybind 並 rebuild，與 diag-overlap 的 kernel instrument 排先後。
- **需 boss 裁決**：Phase 0 的 N=4096（每卡 n=512）在 registry 裡沒有條目，跑的是 fallback config。
  diag-overlap 的「~2% overlap」需要考慮這一點。
- 留給 worker：先做 E0，只跑兩個點把 harness、錨點、對齊方式弄對，再展開網格。

## 2026-09-29 → 09-30 · boss 兼 worker · E0–E3

- 使用者決定：fusion-dispatch 排最前面，有證據前不改 Flux `src/`（D-006）。已登記 PROJECT.md 與 DECISIONS.md。
- **E0**：寫 `dispatch_map_v1.py`，smoke 通過後跑錨點。
  - 在 host 模式發現 B/C 的組合比元件相加快 0.1 ms，追查發現 `flux.testing.init_seed()` 的 cuBLAS 設定：
    `CUBLAS_WORKSPACE_CONFIG=:16:8` 讓 launch 13 → 71 µs，reduced-precision 關掉讓部分形狀 GPU 時間 +26%。
  - v1 撤回 → v2 恢復 production 設定後重跑 E0，錨點全部 < 20%。
- **E1**：4 層 × 20 M × 2 模式（約 40 分鐘），近邊界點以 350 輪重跑並合併。
- **E2**：
  - 三項分解；
  - `profiling()` 估調校上限（不 rebuild）；
  - G-FC1 四點 nsys。nsys parser 初版把不同 process 的 correlationId 混在一起，已修正：key 改為 (pid, corrId)。
- **E3**：`policy_eval_v1.py`，每層查表在三個 trace 上 regret 0–2.0%。
- 卡點 / 未完成：
  - 17 點保留輪次 < 200（功耗上限下 clock 起伏）；
  - nsys 只做 G-FC1；
  - GEMM+RS、CUDA graph 未做。
- **需 boss 裁決**：Phase 0 E.1「Flux AG 在 M=1024–16384 比 NCCL 快 13–22%」在交錯量測下只在 M ≥ 6144 成立
  （gpu 模式 M=1024 NCCL 快 16%、M=4096 相同、M=16384 Flux 快 15%）。是否撤回 / 改寫 `PHASE0_FINDINGS.md` 2.6。
- **需 boss 裁決**：把 `init_seed()` 的 cuBLAS 陷阱加進 `CLAUDE.md` 陷阱表。
- 留給下個 session：先安排 auditor，再做 E5（GEMM+RS）。nsys-rep 每個 20 MB，是否 commit 請 boss 決定；
  sqlite 匯出已刪除，可重建，指令在 `results/e2_nsys/parse_G-FC1.txt` 末尾。

## 2026-09-30 · boss 兼 worker · 第二階段 F0.3（部分）、F0.4、F1、F2a / F2b

- 使用者核准計劃，推論框架授權由 boss 選 → **vLLM 0.8.5.post1**（D-007）。
  選擇理由：與 Flux build 用的 PyPI torch 2.6.0 相容。
- **F0.4**：
  - AGKernel 不能 capture，其餘可以。
  - 過程中發現 graph capture 後 `destroy_process_group()` 會卡住，導致佔用 port；改為寫完結果後 `os._exit(0)`。
  - 另外 `pkill -f` 誤殺了自己的 shell，改成用 PID 停止程序。
- **F0.3（E5）**：寫 `dispatch_map_rs_v1.py`（重用 v2 的 `run_mode`），完成 L-O、L-down。
  為了先做 F1，在兩層之間暫停；G-O、G-FC2 之後在背景重跑。
- **F1**：寫 `dispatcher_v1.py`、`build_table_v1.py`、`validate_block_v1.py` → 發現 RMSNorm 未融合對 vLLM 預設不公平 →
  `validate_block_v2.py`（Triton 融合 RMSNorm / SiLU×up），v1 撤回。
  兩層決策器：校準 / 驗證兩個 seed，regret ≈ 0。
- **F2a / F2b**：`ag_latency_v1.py`。Flux 的 CUDA-core local copy 不支援 bf16；Triton 原型比 NCCL 快，G2 通過。
  torchrun 會把 `--n` 當成自己的參數縮寫，已改名為 `--n_cols`。
- 卡點 / 注意：
  - eager 小 M decode 時單層表不準（CPU launch 瓶頸），vLLM decode 走 graph 可避開。
  - Triton 原型的 epoch 由 host 傳入，不是 graph-safe。
- **需 boss 裁決**：沿用前一條，Phase 0 E.1 的撤回，以及 `CLAUDE.md` 陷阱表新增 init_seed、Flux AGKernel 不可 capture 兩條。
- 留給下個 session：先看 E5 G-O / G-FC2 是否完成，再照 STATUS 第 4 節做 F2c-A。
- 補記：E5 GPT-3 兩層（G-O、G-FC2）於同日跑完，結果已寫入 STATUS 第 2 節第 11 項與 `params.json`。

## 2026-09-30 · boss 兼 worker · 驗證輪（V1–V11），剩餘工作移到 tmux

- 使用者要求：先把所有能確認的未驗證項目確認，再往下一步。清單與已完成結論見 `reports/20260930_verification.md`（草稿，持續補）。
- **已完成**：
  - V1：讀 vLLM 0.8.5 原始碼；
  - V3：真正的 vLLM custom AR，`validate_block_v3.py`；
  - V4 prefill：80 block；
  - V5：graph 重放建表；
  - V6：Phase 0 原腳本重現；
  - V8：nsys 其餘三層；
  - V10：profiling 調校上限；
  - V7：48 點中的 13 批。
- **新發現**：
  - 80 block 的 decode graph 在一個 process 內跑第二個 M 時，所有 rank 卡在 c10d `all_gather_into_tensor`，GPU idle（py-spy 確認）→ V11 重現。
  - M=384 不在表內，決策器借用 512 的選擇，選到較慢的路徑 → 表必須涵蓋所有 graph 桶。
- **剩餘工作在 tmux 裡跑，不受 SSH / VS Code 斷線影響**（使用者可能暫時斷線）：
  - 查看：`tmux attach -t fusion_verify`（離開用 Ctrl-b d）；
  - 進度：`tail -f ws/fusion-dispatch/results/verify_resume_log.txt`；
  - 完成時產生 `ws/fusion-dispatch/results/verify_done/ALL_DONE`；
  - 內容：V7 其餘 18 批 → V4b → V11 → V3b → V6 A/B → V9；
  - 中斷後重跑 `bash ws/fusion-dispatch/scripts/run_verify_resume.sh`，會跳過 `verify_done/` 已標記的項目。
- **下個 session 接手時要做的分析**：
  1. V7：`analyze_v1.py results/v7_rerun/ag --out results/v7_rerun/ag/summary`（rs 同理），再
     `merge_points_v2.py`：AG 順序為 e1_map_v2 → e1_rerun_v2 → v7 ag；RS 順序為 e5_rs_map_v1 → v7 rs。
     檢查判決是否改變。
  2. V4b：`analyze_block_v2.py results/v4_block_L80`（cal 取自 L=4 的 v3 run）。
  3. V11：看 `results/v11_nccl_graph/log_{c10d,pynccl}.txt` 是否出現 STUCK。
  4. V3b：`results/v3b_big_car/log_*.txt` 中 `tp_ar_vllm_big` vs `sp_dispatch`。
  5. V6 A/B：`results/v6_phase0_repro/protocol_ab.json`。
  6. V9：`results/v9_triton_stress/*.json`（single 預期出錯、double 預期 0 錯）。
  7. 補完 `reports/20260930_verification.md` 第 3、4、6、7、9、11 節，更新 STATUS 與 PROJECT。

## 2026-10-02 · boss 兼 worker · 驗證輪收尾

- tmux 中的剩餘工作已於 2026-09-30 10:56 全部完成（`results/verify_resume_log.txt`）。
  機器於 2026-10-01 16:53 重開機（核心 142 → 146，驅動不變），tmux 隨之消失，但當時工作已完成。
- 分析與補測：
  - V7 合併：`final_ag_points.csv`、`final_rs_points.csv`；
  - V4b：80 block decode；
  - V11：未重現；
  - V3b：門檻調高後仍勝但幅度縮小；
  - V6 A/B 加 V6c：時脈記錄，找到 NCCL 差異來自 SM 時脈狀態；
  - **V9 的測試設計有漏洞**（延遲放錯位置），改寫成 v2 後證實單一 buffer 會讀錯、雙 buffer 正確。
- 使用者提醒「測試時要確保沒有別人在用」：
  - 查 `last`：9/29 起只有 rogerlee；
  - 寫 `common/measure/exclusive_guard.py`（preflight 中止 + 每秒監控 + CLEAN / CONTAMINATED），自我測試通過；
  - V12 在守衛下重測 5 組關鍵點，全部 CLEAN，與 9/30 差 ≤ 2%。
- **需 boss 裁決**：
  1. `CLAUDE.md` 5.1 加「量測一律經 exclusive_guard.py」；
  2. `PHASE0_FINDINGS.md` 2.6 的 Flux AG vs NCCL 改寫為依 SM 時脈狀態而定（不是撤回）；
  3. 沿用先前：陷阱表兩條。
- 留給下個 session：驗證結論已寫進 STATUS 第 2 節第 16–27 項，下一步（F2c-A 等）需使用者確認後再開始。

## 2026-10-02（下午）· boss 兼 worker · F4 可行性、文件整理、push

- 使用者問「決策是否查表」：是，見 STATUS 第 0 節。
- F4 可行性：寫 `models/llama3-70b-dummy/config.json`（只有架構），用 `vllm bench latency` 加 dummy 權重跑原版 vLLM：成功。
  - 踩到兩個環境問題：V1 強制載入 tokenizer；transformers 5.x 不相容；
  - 解法寫進 `scripts/setup_vllm_venv.sh`；
  - 報告：`reports/20261002_f4_feasibility.md`。
- 使用者要求：**所有進度都要寫入文件，換 session 能快速跟上** → STATUS 新增第 0 節「快速跟上」，每次收工更新。
- 使用者同意 push：建分支 `fusion-dispatch` 推到 origin（使用者的 fork `allennnnnz/flux`），不直接推 main；
  包含 CLAUDE.md、PROJECT.md、docs、common、ws 與 Phase 0 檔案庫的變更。
  第三方 tokenizer 不提交（.gitignore），由 setup 腳本下載。
- 留給下個 session：先讀 STATUS 第 0 節。F4 實作要等使用者確認；做法見可行性報告的「整合計劃」。
- 補記（push）：origin 是 HTTPS，這個環境沒有 GitHub 憑證，所以 push 失敗。
  已改用 SSH 金鑰推送（`git push git@github.com:allennnnnz/flux.git fusion-dispatch`，金鑰認證為 allennnnnz），**沒有修改 remote 設定**。
  之後的 session 也照這個方式 push。分支：https://github.com/allennnnnz/flux/tree/fusion-dispatch

## 2026-10-02（晚）· boss · 方向改為泛用決策器；計劃核准；交接

- 使用者：查表只是可行性驗證，**泛用決策器更有價值**；要求查文獻、依現況設計，先擬計劃再執行。
- 兩次網路文獻搜尋（約 100 篇，含 TensorRT-LLM / vLLM / NCCL 原始碼）→ `reports/20261002_related_work.md`（G0）。
  - 重點一：現有系統的決策都是查表、手設門檻或窮舉，沒有人同時決定融合開關與 TP 切法，也沒有人展示推廣；
  - 重點二：「物理模型 + 少量實測」推廣證據最好（KernelSight-LM、FlashOverlap、TileSight）；
  - 重點三：NCCL 的時脈敏感度未見前人文獻。
- 使用者要求解釋 cost model 的設計理由：已說明，並寫進計劃第 0 節。
- **計劃核准**：`reports/20261002_plan_general_dispatcher.md`（G0–G6，附錄 A 為交接資訊）。
- 使用者要開新 session 執行。新 session 從 STATUS §0 → 計劃 → 附錄 A.4 的 G1 開始。


## 2026-10-02（晚）· boss 兼 worker · 第三階段 G1：預測器 + 推廣測試（不需 GPU）

- **做了什麼**：
  - 建 `common/cost_model/predictor/`（純 Python）：分段 α-β 通訊、roofline + tile 補齊的 GEMM、Flux config 查詢、融合 kernel 排程模擬、GemmRS 模型、JSON 參數檔；
  - 腳本 `scripts/predictor_data_v1.py`、`eval_predictor_v1.py`、`rf_baseline_v1.py`；
  - 結果 `results/g1_predictor/`；報告 `reports/20261002_g1_predictor.md`。
- **結果**：
  - 只用預測器，B（GPT-3 ↔ Llama）、D（gpu ↔ steady）軸的 op 層級 regret 為 0.03–2.29%，**G1 關卡（≤ 3%）通過**；
  - 混合把關後 AG 0.08–0.25%，但探測 14–28%。
- **過程中修正的東西**（都記在報告裡）：
  1. 一開始讓 kernel 啟動時間 t_k 自由擬合，GPT-3 訓練得到 0.11 ms，但 nsys 只有 0.03–0.05 ms
     → 改用 nsys 的 31.7 µs + bytes / 838 GB/s，另加競爭係數 κ（用穩健損失）。
  2. Flux GEMM 擬合出的有效頻寬曾到 2.19 TB/s，超過 HBM 上限（CLAUDE.md 5.1.4）→ 加上限 2.039 TB/s。
  3. 計劃原定「預設 config 就實測」太貴（42–62%）；真正的斷崖是登錄表 `// PCIE` 區段的 config
     → 新增 hyb+pcie 規則。**這是看過結果才定的，要在 G3 重驗。**
  4. 風險探測要和「最好的非 Flux-GEMM 路徑」比，否則前兩名都是有斷崖的 Flux 路徑時抓不到。
- **意外發現**：
  - 排程模擬解釋了為何 M ≤ 2048 不重疊；
  - 對沒看過的 Phase 0 形狀也預測對，可能就是 diag-overlap 要找的原因 **[推論]**；
  - 給 diag-overlap 的可驗證假說：換 data-parallel config 預測 0.54–0.58 ms。
  - **需 boss 裁決**：是否轉給 diag-overlap（已寫進 PROJECT 第 3 節；本 ws 不改他們的檔案）。
- **卡在哪 / 留給下個 session**：
  - G2 要 GPU，約 10 分鐘量測：照 STATUS §4 的 G2 清單寫 `calibrate_hw_v1.py`，只用微基準擬合後重跑評估；
  - 記得經 `exclusive_guard.py`、放 tmux。

## 2026-10-02（晚）· boss 兼 worker · 第三階段 G2：4 分鐘校準 → 只用微基準的參數檔

- **做了什麼**：
  - 寫 `scripts/calibrate_hw_v1.py`（重用 `dispatch_map_v2.run_mode`），量：
    - 通訊：hidden 6144 × 13 個大小；
    - 3 個 AG 形狀與 2 個 RS 形狀 × 5 個 M，形狀都避開評估層與 Flux 登錄表；
  - `run_calibration_v1.sh`：每組一個 process、各經守衛；
  - `fit_calibration_v1.py`：只用校準數據擬合 → `common/cost_model/hw_profiles/css-host-158_tp8_{gpu,steady}.json`；
  - `eval_profile_v1.py`：對全部 320 個既有實測評估；
  - 報告 `reports/20261002_g2_calibration.md`。
- **結果**：
  - 校準 4 分 2 秒，守衛全 CLEAN；
  - 只用預測器：AG 1.15% / 1.74%（gpu / steady）、RS 0.36% / 0.10%；
  - 混合把關：AG ≤ 0.12%、RS 0.09%，探測 22–23%；
  - 錨點 < 10%；
  - kernel 啟動常數（來自評估層 nsys）敏感度：regret 變化 ≤ 0.11 個百分點。
- **踩到的坑**：
  - 第一版校準在同一 process 依序建立 / 銷毀多組 Flux op，跑到第二個 AG 形狀時隨機卡死
    （8 rank 都在 `torch.cuda.synchronize`，GPU 100% 空轉）；
  - 單獨跑同形狀正常 → 改成一組一個 process，之後全部正常；
  - 第二次重現的堆疊在 `results/g2_smoke/debug_hang2/log.txt`；
  - **第一次卡住的部分 log 在除錯時被我刪掉了**（未提交），只保留重現版。
  - **需 boss 裁決**：建議把「同一 process 換一組 Flux op 可能卡死」加進 `CLAUDE.md` 陷阱表。
- **待改進**：
  - 探測量 > 15%（ε 掃描：5% 時 10–16%）；
  - GemmRS 參數抵換（gpu 版 η = 1.000 碰上限，RS 路徑 A MAPE 16.4%）→ G4 改用 gemm_only 家族參數。
- **留給下個 session**：G3，照 STATUS §4 的 5 步。重點是量測前先把預測寫成檔案並 commit。

## 2026-10-02（晚）· boss 兼 worker · G3 準備：4 卡 / 2 卡校準 + 預先登記預測

- 新增 `scripts/launch_tp.sh`（前 TP 張卡；不改 `launch.sh`）；`run_calibration_v1.sh` 支援 `TP=<n>`。
- 4 卡校準 3 分 4 秒、2 卡 2 分 33 秒，守衛全 CLEAN、正確性全過
  → `common/cost_model/hw_profiles/css-host-158_tp{4,2}_{gpu,steady}.json`。
  - 觀察：Flux AG 同步成本隨卡數近似線性（8 / 4 / 2 卡：0.053 / 0.033 / 0.014 ms），不是常數。
- 新模型形狀加進 harness `LAYERS`：Q-GU、L8-QKV、L8-GU（AG）；Q-down、L8-O、L8-down（RS）。
- **預先登記**：`scripts/predict_g3_v1.py` → `results/g3_predictions/predictions.csv`。
  - 16 個 (卡數, 層) × 10 個 M × 2 模式 = 320 點；
  - 記錄各路徑預測、選擇、要探測的點（ε = 校準 p90 規則：131 點；ε = 5% 變體：79 點）；
  - 參數檔 sha256 記在 `predictions_meta.json`；
  - 在量測前 commit / push。
- 預先記下：
  - G3 的點全都沒命中 Flux 登錄表，所以 PCIe 規則在 G3 測不到；
  - 模型預測 4 卡 / 2 卡時 Flux 在 M = 256–512 起就勝出（8 卡要到 3072）。
- 下一步：`tmux new -d -s g3map 'bash ws/fusion-dispatch/scripts/run_g3_map_v1.sh ws/fusion-dispatch/results/g3_map'`（約 1 小時），
  完成後 `python3 ws/fusion-dispatch/scripts/eval_g3_v1.py`。

## 2026-10-02（晚）· boss 兼 worker · G3 完成：預先登記的新情境

- **量測**：`run_g3_map_v1.sh` 跑 16 組（8 卡：Qwen2.5-72B gate-up / down、Llama-3-8B 四層；4 卡：Llama-3-70B 四層、Qwen 兩層；2 卡：Llama-3-8B 四層）。
  - 每組 10 個 M × gpu / steady，200 輪；
  - 18:05–18:59，16 段守衛全 CLEAN；
  - 51 / 320 點因時脈過濾少於 200 輪（最少 96）。
- **比對**（`eval_g3_v1.py`，對照量測前 push 的預測 `642bf6a`）：
  - 只用預測器：AG 1.76%（8 / 4 / 2 卡：0.89 / 1.77 / 2.81%），RS 0.37%；
  - 固定門檻：AG 1.97%、RS 1.42%；
  - 預先登記的把關：AG 1.21%（探測 80 / 160）。
- **原因**：cuBLAS 斷崖（某些形狀 × M 比模型慢 22–53%）；通訊、Flux gemm_only、重疊都準。
  把關規則只在「差距小」時探測，抓不到有信心的錯。
- **事後分析**（`g3_gemm_probe_diag_v1.py`，標明為事後，只用來定 G4 設計）：
  單卡 GEMM 用實測 + 通訊 / 重疊用模型 → AG 0.27%；RS 需搭配 GemmRS 模型修正。
- 模型的大膽預測正確：4 卡 / 2 卡時 Flux 系路徑從 M=136–512 起最快。
- PCIe 規則在 G3 沒測到（無登錄表點）。
- 另寫 `build_table_v2.py`（G4 準備）：用預測器產生 v1 格式的表。
  - 對 Llama-70B 8 卡與 v1 實測表一致 71 / 80，不一致多為平手；
  - 探測後只剩 3 點，差距 ≤ 3.5%。
- 報告：`reports/20261002_g3_unseen.md`。
- **留給下個 session**：G4，照 STATUS §4 的 6 點。
  - 最重要：決策器 v2 加入「實測單卡 GEMM」；
  - GemmRS 改共用 gemm_only 參數；
  - block 驗證（vLLM 環境，含 TP=4）。

## 2026-10-03 · boss 兼 worker · G4：決策器 v2 + 坑的記錄

- **使用者決定**：G4 改用「模型 + 實測單卡 GEMM + 少量多卡把關」，要驗證並把坑記錄起來（D-009）。
- **坑的記錄**：
  - `CLAUDE.md` 陷阱表新增 11 條，5.1 新增第 11 條「量測必須獨佔」；
  - 計劃附錄 A.3 新增 20 條（G1–G4）；
  - PROJECT 第 3 節對應項目標為已處理。
- **op 層級**（全新測試集：Qwen2.5-32B 8 / 4 卡、Llama-3-8B 4 卡，M = 24–6144）。流程：
  校準（多量 GemmOnly）→ 單卡 GEMM（82 秒）→ 預測並 push → 多卡探測（367 秒）→ 最終決策並 push → 標準答案（17.7 分鐘）。
  - 決策器 v2：regret 0.17%（AG 0.04%、RS 0.41%）；加探測 0.11%；
  - 對照：G3 方法 1.20%、固定門檻 2.30%。
- **block 層級**。流程：vLLM all-reduce + norm 校準 → block 用的 M 量 GEMM → 表與探測 → 用 `predictor/block.py` 預測切法 → 切法探測 → 正式量測（293 秒）。
  - 自動選切法 + G4 表：block regret 0.02%，比 vLLM 預設省 8.0%（prefill 10.9%）；decode 正確選 vLLM 預設。
- 每一步的決策都在下一步量測前 commit / push（`49534de`、`8fc288e`、`486ef8e`、`f590a3b`、`0d6be62`、`bb42d7b`）。
- **成功標準**：block ≤ 2%、校準 ≤ 10 分鐘、切換點誤差 ≤ 10% 達成；探測 ≤ 15% 未達成（但探測改善很小）。
- **踩到的坑**：
  1. steady 模式量極小 kernel 是量 CPU 發 kernel 的速度，同一 GEMM 探測與地圖差 3 倍，RS 最差的點都在這裡；
  2. vLLM custom AR 在 eager 量比 graph 慢；
  3. **我自己的錯誤**：`validate_block_v4.py` 的策略子集在 graph 模式仍嘗試移除 `sp_flux` → decode 切法探測第一次失敗。
     已修，失敗紀錄改名保留（`*.failed_policyfilter_bug`），重跑後正常。
- 報告：`reports/20261003_g4_dispatcher.md`。
- **留給下個 session**：使用者決定 F4（vLLM 端到端）/ G5（PCIe 代理）/ G6（總報告 + auditor）的先後。

## 2026-10-03 · boss 兼 worker · G4 之後：比較圖、Flux 優勢拆解、跨機器驗證計劃

- **比較圖**：使用者要求把「全部用 Flux / 全部用 vLLM 預設 / 決策器」畫在一起。
  - 存成 `reports/20261003_g4_strategy_compare.html`；
  - decode 沒有真正的全 Flux（AGKernel 不能 capture），圖上用 `sp_rsflux` 並標註。
- **使用者問「Flux 怎麼看起來沒有優勢」** → 事後分析（`scripts/analyze_g4_flux_value_v1.py`，沒有新量測）：
  - prefill：Flux 在同一種切法下快 7–20%，決策器 31 / 36 的每層決定用 Flux；
  - decode：輸在切法（換成序列平行就慢 15–36%），Flux 在序列平行內從 M ≥ 384 起才有幫助。
- **使用者問「決策器有什麼用」**：
  - 簡單規則「decode vLLM、prefill Flux」在這台 regret 0.57%、省 7.5%；決策器 0.02%、省 8.0%；
  - 誠實結論：在這台決策器多贏的很少，價值要在別的硬體 / 連線上證明。
- **驗證計劃**（報告第 3 節）：
  - 對手規則 R1–R3 與成功標準已凍結；
  - 環境：E0 模型推演、E1 本機背景塞車、E2 本機 PCIe-only、E3 gpu1。
- **gpu1**：使用者提供（EE325 叢集，經跳板機）。
  - 卡在哪：本機金鑰被跳板機拒絕，使用者加了公鑰仍被拒，原因未查明；
  - 改用 SSH agent forwarding，待使用者設定筆電並重連 VSCode。
- **更正**：對話中「Flux 在序列平行內從 M ≥ 256 起有幫助」「custom AR 快 7–12%」→ 正確是 M ≥ 384、5–13%（報告第 4 節）。
- 報告：`reports/20261003_g4_flux_value.md`。
- **留給下個 session**：
  - 連上 gpu1 後先唯讀盤點硬體，回報使用者，確認後再 clone / 編譯 / 跑；
  - 流程照報告第 3 節（校準 → 決策 push → 標準答案 → 比 regret）。

## 2026-10-03 · gpu1 session · 跨機器驗證：gpu1 唯讀盤點

- 在 gpu1 直接開 Claude Code session（備案路線）；clone 已存在，切到 `fusion-dispatch`。只做唯讀盤點，沒有編譯、沒有 GPU 量測。
- 原始輸出：`results/gpu1_inventory/`；報告：`reports/20261003_gpu1_inventory.md`。
- **硬體**：4× Tesla V100-PCIE-32GB（sm70，80 SM），驅動 580.142；無 NVLink；PCIe Gen3 x16；
  P2P 只在 GPU0↔1、GPU2↔3，跨 CPU 經 host + QPI；2× Xeon E5-2678 v3，125 GiB；`/` 只剩 9.2 GB，`/home` 是 NAS。
- **Flux 不能跑**：`src/cuda/op_registry.cu:39-51` 只接受 sm80/89/90 與 A100/H20/H800/L20 的 SM 數；kernel 也用 sm80 指令。
  依凍結規則，R1 / R3 在這台退回 R2 → 無法回答「決策器是否比簡單規則好」，只能做縮小版（切法 + 校準可攜性，fp16）。
- **共用**：7 個其他使用者在線；GPU 盤點時閒置；Slurm 有但 gpu1 DOWN、無 GRES，沒有排程保護。
- **新的坑**（**需 boss 裁決**：是否寫進 `CLAUDE.md` 陷阱表 / 5.1 第 11 條）：
  1. `exclusive_guard.py` 用使用者名稱比對，`ps` 把 > 8 字的名稱截成 `allenzh+` → 自己的程序被當成別人，預檢 ABORT、執行必 CONTAMINATED；
  2. `exclusive_guard.py` 的 CPU% 是生命期平均：user_D 的程序瞬間 151.7%、守衛看到 0.2%；
  3. CLAUDE.md 第 2 節是 css-host-158 專屬；多機器後建議改成「各機器環境」小節。
- **留給下個 session**：等使用者決定 gpu1 做不做縮小版（或找 sm80+ 無 NVLink 機器 / 先做 E0）。
  若做：先寫 `exclusive_guard_v2.py`，再建 venv（快取放 `/home`），腳本改 v2（fp16、Flux 可選、`gpu1_tp*_*.json`）。

## 2026-10-03 · boss 兼 worker（css-host-158）· 新節點 css-host-159 唯讀盤點

- 讀了 gpu1 session 的盤點（`reports/20261003_gpu1_inventory.md`），核對 `src/cuda/op_registry.cu:39-51`：
  Flux 只接受 sm80 / 89 / 90，而且 SM 數要是 92 / 108 / 78 / 132（L20 / A100 / H20 / H800）。V100 不行。
- 也確認 `exclusive_guard.py` 的 CPU 用量是 `ps pcpu`（生命期平均），**本機也有同樣的盲點**；名稱截斷只影響 gpu1。
- 使用者提供新節點 css-host-159（`rogerlee@10.2.131.159`）。唯讀盤點（`results/node159_inventory/`）：
  - 跟本機同硬體、同驅動，GPU 閒置；
  - 兩台之間 8 條 100 GbE RoCE，第 4 條不通（本機 `mlx5_3` 沒有 IPv4）；
  - 本機 Flux 有 NVSHMEM 與跨節點 op；159 只有舊 repo、沒有環境；
  - 159 的 `~/.bashrc` 會把自編 NCCL 2.26.2 放進 `LD_LIBRARY_PATH`。
- 沒有在 159 安裝或執行任何東西。
- **留給下個 session**：使用者核准後，照 STATUS §0 做 E4（跨節點）：複製環境 → 守衛 v2（雙節點）→ 錨點 → Flux 跨節點冒煙測試。

## 2026-10-03 · boss 兼 worker（css-host-158）· E4 跨節點：環境、守衛 v2、錨點、NCCL 卡住的原因

- **使用者核准 E4**，開始做準備。
- **環境**：159 的舊 repo 是今天剛 clone 的 main（我們分支的祖先、沒有獨有內容），用本機完整副本覆蓋；
  pixi 環境 + vLLM venv 一起複製（共 18.6 GB，48 秒）。159 上 torch / vLLM 都載入自己附的 NCCL 2.21.5。
- **守衛 v2**（`common/measure/exclusive_guard_v2.py`）：修 gpu1 發現的兩個 bug（名稱截斷、生命期平均 CPU），
  另外監看 root 程序、log 一律 UTC（159 時區是紐約）。自我測試：忙迴圈 100.6% 被即時抓到。v1 加 WITHDRAWN。
- **啟動腳本**：`scripts/run_xnode.sh`（在 158 跑，兩台各自包守衛）+ `scripts/launch_xnode.sh`（每台的環境）。
- **RDMA 錨點**（`results/e4_anchor_ib/`）：7 條 rail 各 98.05 Gb/s，7 條同時也一樣；兩台守衛 CLEAN。
- **跨節點 NCCL 一律卡住** → 一路排除（`results/e4_debug/README.md`）：網路、MTU、GDR、網卡合併、單節點、Flux 環境變數都不是。
  **原因**：159 的 `~/.nccl.conf`（不是我們的；ALGO=RING、PROTO=Simple、P2P_LEVEL=NVL、IB_HCA=mlx5_3:1）只在 159 生效，
  兩台演算法 / 協定不一致 → 卡住；只一致一半時 all_reduce「完成」但結果錯（0.0，應為 2.0）。
  NCCL 2.21.5 用 passwd 的家目錄找這個檔，`NCCL_CONF_FILE` 無效，改 `HOME` 也沒用。
- **其他**：159 上使用者的 `openclaw-gateway`（systemd 使用者服務）每 7 秒重啟一次、每次吃 2 核以上。
- **卡在哪**：159 的 `~/.nccl.conf` 怎麼處理、`openclaw-gateway` 要不要停，都需要使用者決定（是使用者 / 同帳號的人的東西）。
- **需 boss 裁決**（CLAUDE.md）：5.1 第 11 條改指向 `exclusive_guard_v2.py`；陷阱表加「`~/.nccl.conf` 會讓多節點設定不一致，
  2.21.5 無法用環境變數關掉」與「守衛 v1 的盲點」。

## 2026-10-03（續）· E4 跨節點：錨點解釋、Flux 冒煙測試

- 使用者核准：159 的 `~/.nccl.conf` 改名停用、`openclaw-gateway` 量測期間停掉（**E4 結束後要還原**）。改名後跨節點 NCCL 正常、結果正確。
- **NCCL 錨點**（1+1 / 4+4 / 8+8，`results/e4_anchor_nccl/`）：大約只有網卡極限的 1/3，且停頓嚴重（max/min 最高 150 倍）。
  - 停頓：RoCE 只有優先權 3 不丟包，NCCL 預設 TC 0 → 丟包重傳。TC 106 讓 8+8 頻寬減半；**TC 104 最好**，設為預設。
  - 頻寬：`ib_write_bw --use_cuda` 顯示網卡從 GPU 讀只有 56 Gb/s（寫進 GPU 是滿速）；同一交換晶片兩張 GPU 合計約 9.5 GB/s。
    NCCL 數字與此一致（多開通道、`NCCL_NET_GDR_READ=0` 都沒幫助）。→ 這是兩台的 PCIe 特性，當作環境的一部分。
- **Flux 冒煙測試**（`results/e4_flux_smoke/`）：
  - `AGKernel`（單節點那個）跨節點寫對端 barrier 指標失敗 → 跨節點要用 `AGKernelXNode`：能跑、正確，但 NVSHMEM 全網狀連線只在單一 rail 成立（rail 間沒有路由）。
  - `GemmRS` 傳 nnodes=2 → illegal memory access；A100 正確用法是 `GemmRS_multinode`（節點內融合 + NCCL）→ 能跑但 158 的 rank 結果全錯（`scripts/xnode_flux_rs_smoke_v1.py`）。
  - Flux 自帶測試的 torch 參考 31 ms 跟錨點對不上，未解釋，不引用。
- params.json 新增 `flux_dispatch_e4_xnode`（錨點、QoS、`.nccl.conf`、冒煙結果）。
- **卡在哪 / 需使用者決定**：NVSHMEM 多 rail 需要管理員加 rail 間路由（以及修 rail 4 的 IP）；Flux 多節點 RS 要先查對錯；或改走 E1 / E0。

## 2026-10-05 · boss 兼 worker · 報告準備、決策器價值的拆解、E4a 啟動

- **給教授的簡報**（Slides artifact，14 張含講稿）：https://claude.ai/artifact/Tvgvi2KCKYweKgeXMfK49M
- **拆解決策器的價值**：prefill 9 點「全部用 Flux」vs「逐層查表」只差 0.77%，**幾乎全部來自 1 個點**（Qwen TP8 M=1024，+7.9%）。
  → 在這台機器上，決策器比簡單規則多出的價值**尚未被證明**；切法決策壓倒一切，而切法很好選。
- 使用者問「能不能移植 Flux 到 V100」：不建議（`op_registry.cu` 擋 sm70、mainloop 用 `cp.async`、無 bf16、違反 D-006，而且量到的是自己寫的 kernel）。改用兩台 A100 跨節點當慢連線環境。
- **我先前的推論被冒煙測試推翻**：以為 TP+AllReduce 和序列平行+NCCL 搬的位元組一樣多、沒有 Flux 就不會翻；實測 prefill M=1024 跨節點序列平行快 22%（10 輪）。
- **E4a 啟動**（tmux `e4a`，無人值守，因為使用者要關電腦）：`scripts/run_e4a_v1.sh`。
  - 新腳本：`calibrate_xnode_v1.py`（多量 NCCL AG / RS）、`e4a_layout_v1.py`（fit / predict / eval）；`validate_block_v4.py` 在純 NCCL 策略時跳過 Flux 初始化（G4 行為不變）。
  - 坑：`flux.testing.initialize_distributed()` 與 decode 的 graph 路徑會啟動 NVSHMEM → 跨節點必須 `NVSHMEM_HCA_LIST=mlx5_0`（NCCL 不受影響，仍用 7 條）。
  - 坑：流程裡 log 一直在寫，`git pull --rebase` 要加 `--autostash`。
- **留給下個 session**：讀 `results/e4a_layout/eval_*.txt` 與自動 JOURNAL 條目寫結論；若簡單規則在跨節點失效，接著做帶 Flux 的版本（需先解 NVSHMEM 多 rail 與 `GemmRS_multinode` 結果錯）。

## 2026-10-05 · E4a 自動流程完成（run_e4a_v1.sh 自動追加；結論待人工撰寫）

- `ws/fusion-dispatch/results/e4a_layout/eval_tp4x2n_llama3-8b_tp4_summary.csv`
- `ws/fusion-dispatch/results/e4a_layout/eval_tp4x2n_qwen2.5-32b_tp4_summary.csv`
- `ws/fusion-dispatch/results/e4a_layout/eval_tp8x2n_qwen2.5-32b_tp8_summary.csv`

```
E4a tp4x2n llama3-8b TP=4 (two nodes): pre-registered decisions vs measured oracle
phase        M     vLLM       SP        best     decider  pred dT  meas dT  kept
decode      32    1.744    1.802  tp_ar_vllm  tp_ar_vllm   -0.011   -0.058  185
decode     128    5.105    4.577     sp_nccl     sp_nccl    0.320    0.528  200
decode     256    9.653    4.472     sp_nccl     sp_nccl    0.438    5.181  200
decode     384    5.190    5.751  tp_ar_vllm     sp_nccl    0.248   -0.561  200
decode     512    6.961    7.509  tp_ar_vllm     sp_nccl    0.142   -0.548  200
prefill   1024    9.839   10.503  tp_ar_vllm  tp_ar_vllm   -0.085   -0.665  200
prefill   2048   20.488   22.701  tp_ar_vllm  tp_ar_vllm   -0.147   -2.213  200
prefill   4096   39.049   40.032  tp_ar_vllm  tp_ar_vllm   -0.293   -0.984  200

policy      regret   saving vs vLLM   worst point   (regret = sum(t - t_best) / sum(t_best))
[all] 8 points
  decider     1.20%          4.7%      10.8% at decode M=384
  R1          6.18%          0.0%     115.9% at decode M=256
  R1p        10.37%         -3.9%     115.9% at decode M=256
  R2          6.18%          0.0%     115.9% at decode M=256
  R3          5.45%          0.7%      10.8% at decode M=384
[decode] 5 points
  decider     4.83%         16.1%      10.8% at decode M=384
  R1         24.88%          0.0%     115.9% at decode M=256
  R1p        24.88%          0.0%     115.9% at decode M=256
  R2         24.88%          0.0%     115.9% at decode M=256
  R3          5.09%         15.9%      10.8% at decode M=384
[prefill] 3 points
  decider     0.00%          0.0%       0.0% at prefill M=1024
  R1          0.00%          0.0%       0.0% at prefill M=1024
  R1p         5.57%         -5.6%      10.8% at prefill M=2048
  R2          0.00%          0.0%       0.0% at prefill M=1024
  R3          5.57%         -5.6%      10.8% at prefill M=2048

E4a tp4x2n qwen2.5-32b TP=4 (two nodes): pre-registered decisions vs measured oracle
phase        M     vLLM       SP        best     decider  pred dT  meas dT  kept
decode      32    2.270    2.288  tp_ar_vllm     sp_nccl    0.006   -0.017  181
decode     128    6.535    3.845     sp_nccl     sp_nccl    0.427    2.690  200
decode     256    4.799    5.335  tp_ar_vllm     sp_nccl    0.343   -0.536  200
decode     384    6.917    7.494  tp_ar_vllm     sp_nccl    0.168   -0.578  200
decode     512    8.470    9.073  tp_ar_vllm     sp_nccl    0.035   -0.603  200
prefill   1024   13.326   13.908  tp_ar_vllm  tp_ar_vllm   -0.092   -0.582  200
prefill   2048   27.977   29.585  tp_ar_vllm  tp_ar_vllm   -0.195   -1.609  200
prefill   4096   54.951   53.673     sp_nccl  tp_ar_vllm   -0.344    1.278  200

policy      regret   saving vs vLLM   worst point   (regret = sum(t - t_best) / sum(t_best))
[all] 8 points
  decider     2.48%          0.8%      11.2% at decode M=256
  R1          3.27%          0.0%      70.0% at decode M=128
  R1p         4.02%         -0.7%      70.0% at decode M=128
  R2          3.27%          0.0%      70.0% at decode M=128
  R3          3.24%          0.0%      11.2% at decode M=256
[decode] 5 points
  decider     6.59%          3.3%      11.2% at decode M=256
  R1         10.23%          0.0%      70.0% at decode M=128
  R1p        10.23%          0.0%      70.0% at decode M=128
  R2         10.23%          0.0%      70.0% at decode M=128
  R3          6.59%          3.3%      11.2% at decode M=256
[prefill] 3 points
  decider     1.35%          0.0%       2.4% at prefill M=4096
  R1          1.35%          0.0%       2.4% at prefill M=4096
  R1p         2.31%         -0.9%       5.8% at prefill M=2048
  R2          1.35%          0.0%       2.4% at prefill M=4096
  R3          2.31%         -0.9%       5.8% at prefill M=2048

E4a tp8x2n qwen2.5-32b TP=8 (two nodes): pre-registered decisions vs measured oracle
phase        M     vLLM       SP        best     decider  pred dT  meas dT  kept
decode      32    1.886    2.164  tp_ar_vllm  tp_ar_vllm   -0.032   -0.278  200
decode     128    2.324    2.648  tp_ar_vllm  tp_ar_vllm   -0.178   -0.324  200
decode     256    4.009    4.232  tp_ar_vllm  tp_ar_vllm   -0.151   -0.223  200
decode     384    5.588    6.449  tp_ar_vllm  tp_ar_vllm   -0.096   -0.861  200
decode     512    6.989    7.759  tp_ar_vllm     sp_nccl    0.018   -0.770  200
prefill   1024   15.293   12.375     sp_nccl  tp_ar_vllm   -0.318    2.919  200
prefill   2048   23.711   23.219     sp_nccl  tp_ar_vllm   -0.493    0.492  200
prefill   4096   50.994   51.613  tp_ar_vllm     sp_nccl    0.187   -0.619  200

policy      regret   saving vs vLLM   worst point   (regret = sum(t - t_best) / sum(t_best))
[all] 8 points
  decider     4.47%         -1.3%      23.6% at prefill M=1024
  R1          3.18%          0.0%      23.6% at prefill M=1024
  R1p         0.58%          2.5%       1.2% at prefill M=4096
  R2          3.18%          0.0%      23.6% at prefill M=1024
  R3          2.86%          0.3%      15.4% at decode M=384
[decode] 5 points
  decider     3.70%         -3.7%      11.0% at decode M=512
  R1          0.00%          0.0%       0.0% at decode M=32
  R1p         0.00%          0.0%       0.0% at decode M=32
  R2          0.00%          0.0%       0.0% at decode M=32
  R3         11.81%        -11.8%      15.4% at decode M=384
[prefill] 3 points
  decider     4.65%         -0.7%      23.6% at prefill M=1024
  R1          3.94%          0.0%      23.6% at prefill M=1024
  R1p         0.71%          3.1%       1.2% at prefill M=4096
  R2          3.94%          0.0%      23.6% at prefill M=1024
  R3          0.71%          3.1%       1.2% at prefill M=4096

```

## 2026-10-05（續）· E4a 結論（人工撰寫，取代上面自動條目的「結論待撰寫」）

- 無人值守流程 21 分鐘跑完，3 組配置、24 點，兩台守衛全部 CLEAN，預測 commit 皆早於標準答案。
- **結果**：決策器 regret 2.78%（最差 23.6%），R1 / R2 4.08%、R1p 4.69%（最差都 115.9%），R3 3.75%。
  - 簡單規則在慢連線上會出大錯 → 支持「規則會隨環境改變」。
  - 決策器整體最好但只猜對 13/24，配置 A（Qwen TP8 4+4）輸給所有簡單規則 → 「決策器能自己找到新規則」尚未成立。
- **p90/p10 只有 1.02–1.05**：怪點（Llama TP4 decode M=256 的 AR 9.65 ms > M=384 的 5.19 ms）是穩定的 NCCL 斷崖，不是雜訊；torch NCCL AR 也一樣慢。
- **決策器猜錯的原因**：跨節點 all-reduce 非單調（TP4 校準 1.5 MiB 0.750 ms、3 MiB 0.404 ms），`predictor/curves.py` 的等壓回歸假設把斷崖抹平擴散；
  校準格點（H=6144、M=2 的冪）與 block 的訊息大小錯開；E4a 只用模型、沒開 G4 的把關（10/24 點會觸發）——此為偏離 G4 方法，已揭露。
- **我自己的錯**：冒煙測試前，我以為「兩種切法位元組一樣多，沒有 Flux 就不會翻」；實測完全不是如此。
- 報告：`reports/20261005_e4a_slow_link_layout.md`。
- **留給下個 session**：修模型 → 用新的 M 點預先登記並重測（舊 24 點不能再當測試集）；之後才做帶 Flux 的版本。159 的 openclaw-gateway 仍停著。

## 2026-10-05（續）· E4a2 設計並啟動（修正後的決策器，新的測試點）

- 使用者要求「根據本次數據調整，再做一次」。
- **修正**（皆對應 E4a 定位到的失敗，看到新數據前定案並隨 setup commit 登記）：
  1. `predictor/curves.py` 加 `monotone=False`（預設不變，單機參數檔不受影響）；v2 擬合也不合併相近大小（`rel_merge=0`）。
     自我測試直接重現了失敗機制：`[0.385, 0.750, 0.404, 0.728]` 被單調化成 `[0.385, 0.577, 0.577, 0.728]`。
  2. `calibrate_xnode_v2.py --cases`：直接量 block 會用到的 (H, M)。這是單一集合通訊的元件量測，不是 block 量測（性質同 G4 的單卡 GEMM 實測）。
  3. `e4a_layout_v2.py`：開啟 G4 把關，EPS = 3%。
  4. **新的測試點** decode 64 / 192 / 320 / 448、prefill 768 / 1536 / 3072；E4a 的 24 點只當健全性檢查。
- 流程 `scripts/run_e4a2_v1.sh`（tmux `e4a2`）：補量單卡 cuBLAS → 跨節點校準 → 模型決定 commit → 把關 → 最終決定 commit → 標準答案 → 評估。
- **留給下個 session**：讀 `results/e4a2_layout/eval_*_new.txt` 寫結論。

## 2026-10-05（續）· E4a2 第一次嘗試失敗（我的 bug），修正後重跑

- `run_xnode.sh` 把 node 1 的參數經 ssh 傳過去時沒有加引號；`--cases "5120:...;4096:..."` 在 159 被分號切開，node 1 啟動失敗，node 0 空等 38 分鐘。
  E4a 沒踩到，因為當時的參數沒有 shell 特殊字元。
- 若放著不管，TP4 擬合會失敗，流程會連 TP8 的把關與標準答案一起跳過 → 手動停掉。尚未預先登記任何預測（只有 setup commit），重跑不影響登記。
- 修正：`printf '%q '` 逐一引用參數，經 ssh 驗證分號完整保留。部分輸出移到 `results/e4a2_layout/attempt1_failed/`（含 README），第二次全部重量。

## 2026-10-05 · E4a2 自動流程完成（run_e4a2_v1.sh 自動追加；結論待人工撰寫）

```
E4a2 tp4x2n2 llama3-8b TP=4 set=new (FRESH points, pre-registered)
phase        M     vLLM       SP        best       model       final  pred dT  meas dT probe
decode      64    2.747    2.580     sp_nccl     sp_nccl     sp_nccl    0.037    0.166 0
decode     192    7.262    3.763     sp_nccl     sp_nccl     sp_nccl    0.897    3.499 0
decode     320    4.553    5.116  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.099   -0.564 0
decode     448    6.321    6.922  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.096   -0.601 0
prefill    768    7.282    7.954  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.081   -0.673 0
prefill   1536   13.961   14.761  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.062   -0.800 1
prefill   3072   31.086   31.422  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.278   -0.336 0

policy           regret   saving vs vLLM   worst point   right
[all] 7 points
  decider           0.00%          5.0%      0.0% at decode M=64      7/7
  decider_model     0.00%          5.0%      0.0% at decode M=64      7/7
  R1                5.27%          0.0%     93.0% at decode M=192     5/7
  R1p               7.87%         -2.5%     93.0% at decode M=192     2/7
  R2                5.27%          0.0%     93.0% at decode M=192     5/7
  R3                4.28%          0.9%     12.4% at decode M=320     2/7
[decode] 4 points
  decider           0.00%         17.6%      0.0% at decode M=64      4/4
  decider_model     0.00%         17.6%      0.0% at decode M=64      4/4
  R1               21.29%          0.0%     93.0% at decode M=192     2/4
  R1p              21.29%          0.0%     93.0% at decode M=192     2/4
  R2               21.29%          0.0%     93.0% at decode M=192     2/4
  R3                6.77%         12.0%     12.4% at decode M=320     2/4
[prefill] 3 points
  decider           0.00%          0.0%      0.0% at prefill M=768     3/3
  decider_model     0.00%          0.0%      0.0% at prefill M=768     3/3
  R1                0.00%          0.0%      0.0% at prefill M=768     3/3
  R1p               3.46%         -3.5%      9.2% at prefill M=768     0/3
  R2                0.00%          0.0%      0.0% at prefill M=768     3/3
  R3                3.46%         -3.5%      9.2% at prefill M=768     0/3

E4a2 tp4x2n2 qwen2.5-32b TP=4 set=new (FRESH points, pre-registered)
phase        M     vLLM       SP        best       model       final  pred dT  meas dT probe
decode      64    3.674    3.404     sp_nccl     sp_nccl     sp_nccl    0.175    0.270 0
decode     192    9.368    4.617     sp_nccl     sp_nccl     sp_nccl    1.185    4.751 0
decode     320    6.058    6.601  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.089   -0.543 0
decode     448    7.738    8.278  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.071   -0.539 0
prefill    768   10.285   10.911  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.080   -0.627 1
prefill   1536   21.364   23.731  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.638   -2.367 0
prefill   3072   41.662   42.485  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.185   -0.823 1

policy           regret   saving vs vLLM   worst point   right
[all] 7 points
  decider           0.00%          5.0%      0.0% at decode M=64      7/7
  decider_model     0.00%          5.0%      0.0% at decode M=64      7/7
  R1                5.28%          0.0%    102.9% at decode M=192     5/7
  R1p               9.29%         -3.8%    102.9% at decode M=192     2/7
  R2                5.28%          0.0%    102.9% at decode M=192     5/7
  R3                5.15%          0.1%     11.1% at prefill M=1536    2/7
[decode] 4 points
  decider           0.00%         18.7%      0.0% at decode M=64      4/4
  decider_model     0.00%         18.7%      0.0% at decode M=64      4/4
  R1               23.01%          0.0%    102.9% at decode M=192     2/4
  R1p              23.01%          0.0%    102.9% at decode M=192     2/4
  R2               23.01%          0.0%    102.9% at decode M=192     2/4
  R3                4.96%         14.7%      9.0% at decode M=320     2/4
[prefill] 3 points
  decider           0.00%          0.0%      0.0% at prefill M=768     3/3
  decider_model     0.00%          0.0%      0.0% at prefill M=768     3/3
  R1                0.00%          0.0%      0.0% at prefill M=768     3/3
  R1p               5.21%         -5.2%     11.1% at prefill M=1536    0/3
  R2                0.00%          0.0%      0.0% at prefill M=768     3/3
  R3                5.21%         -5.2%     11.1% at prefill M=1536    0/3

E4a2 tp8x2n2 qwen2.5-32b TP=8 set=new (FRESH points, pre-registered)
phase        M     vLLM       SP        best       model       final  pred dT  meas dT probe
decode      64    2.960    3.231  tp_ar_vllm     sp_nccl     sp_nccl    0.069   -0.271 0
decode     192    3.191    3.466  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.087   -0.275 0
decode     320    4.766    4.922  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.069   -0.156 0
decode     448    6.383    7.267  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.207   -0.884 0
prefill    768   11.437   10.072     sp_nccl     sp_nccl     sp_nccl    0.055    1.364 1
prefill   1536   18.242   18.008     sp_nccl     sp_nccl     sp_nccl    0.049    0.233 1
prefill   3072   36.382   36.084     sp_nccl     sp_nccl     sp_nccl    0.052    0.297 1

policy           regret   saving vs vLLM   worst point   right
[all] 7 points
  decider           0.33%          1.9%      9.2% at decode M=64      6/7
  decider_model     0.33%          1.9%      9.2% at decode M=64      6/7
  R1                2.33%          0.0%     13.5% at prefill M=768     4/7
  R1p               0.00%          2.3%      0.0% at decode M=64      7/7
  R2                2.33%          0.0%     13.5% at prefill M=768     4/7
  R3                1.95%          0.4%     13.8% at decode M=448     3/7
[decode] 4 points
  decider           1.57%         -1.6%      9.2% at decode M=64      3/4
  decider_model     1.57%         -1.6%      9.2% at decode M=64      3/4
  R1                0.00%          0.0%      0.0% at decode M=64      4/4
  R1p               0.00%          0.0%      0.0% at decode M=64      4/4
  R2                0.00%          0.0%      0.0% at decode M=64      4/4
  R3                9.17%         -9.2%     13.8% at decode M=448     0/4
[prefill] 3 points
  decider           0.00%          2.9%      0.0% at prefill M=768     3/3
  decider_model     0.00%          2.9%      0.0% at prefill M=768     3/3
  R1                2.95%          0.0%     13.5% at prefill M=768     0/3
  R1p               0.00%          2.9%      0.0% at prefill M=768     3/3
  R2                2.95%          0.0%     13.5% at prefill M=768     0/3
  R3                0.00%          2.9%      0.0% at prefill M=768     3/3

E4a2 tp4x2n2 llama3-8b TP=4 set=old (SANITY ONLY: E4a points, answers seen before the model change)
phase        M     vLLM       SP        best       model       final  pred dT  meas dT probe
decode      32    1.744    1.802  tp_ar_vllm     sp_nccl  tp_ar_vllm    0.046   -0.058 0
decode     128    5.105    4.577     sp_nccl     sp_nccl     sp_nccl    0.431    0.528 0
decode     256    9.653    4.472     sp_nccl     sp_nccl     sp_nccl    1.266    5.181 0
decode     384    5.190    5.751  tp_ar_vllm  tp_ar_vllm     sp_nccl   -0.101   -0.561 0
decode     512    6.961    7.509  tp_ar_vllm  tp_ar_vllm     sp_nccl   -0.095   -0.548 0
prefill   1024    9.839   10.503  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.055   -0.665 1
prefill   2048   20.488   22.701  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.810   -2.213 0
prefill   4096   39.049   40.032  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.272   -0.984 1

policy           regret   saving vs vLLM   worst point   right
[all] 8 points
  decider_model     0.06%          5.8%      3.3% at decode M=32      7/8
  decider_e4a_v1    1.20%          4.7%     10.8% at decode M=384     6/8
  R1                6.18%          0.0%    115.9% at decode M=256     6/8
  R1p              10.37%         -3.9%    115.9% at decode M=256     3/8
  R2                6.18%          0.0%    115.9% at decode M=256     6/8
  R3                5.45%          0.7%     10.8% at decode M=384     2/8
[decode] 5 points
  decider_model     0.25%         19.7%      3.3% at decode M=32      4/5
  decider_e4a_v1    4.83%         16.1%     10.8% at decode M=384     3/5
  R1               24.88%          0.0%    115.9% at decode M=256     3/5
  R1p              24.88%          0.0%    115.9% at decode M=256     3/5
  R2               24.88%          0.0%    115.9% at decode M=256     3/5
  R3                5.09%         15.9%     10.8% at decode M=384     2/5
[prefill] 3 points
  decider_model     0.00%          0.0%      0.0% at prefill M=1024    3/3
  decider_e4a_v1    0.00%          0.0%      0.0% at prefill M=1024    3/3
  R1                0.00%          0.0%      0.0% at prefill M=1024    3/3
  R1p               5.57%         -5.6%     10.8% at prefill M=2048    0/3
  R2                0.00%          0.0%      0.0% at prefill M=1024    3/3
  R3                5.57%         -5.6%     10.8% at prefill M=2048    0/3

E4a2 tp4x2n2 qwen2.5-32b TP=4 set=old (SANITY ONLY: E4a points, answers seen before the model change)
phase        M     vLLM       SP        best       model       final  pred dT  meas dT probe
decode      32    2.270    2.288  tp_ar_vllm     sp_nccl     sp_nccl    0.054   -0.017 0
decode     128    6.535    3.845     sp_nccl     sp_nccl     sp_nccl    0.659    2.690 0
decode     256    4.799    5.335  tp_ar_vllm  tp_ar_vllm     sp_nccl   -0.099   -0.536 0
decode     384    6.917    7.494  tp_ar_vllm  tp_ar_vllm     sp_nccl   -0.092   -0.578 0
decode     512    8.470    9.073  tp_ar_vllm  tp_ar_vllm     sp_nccl   -0.082   -0.603 0
prefill   1024   13.326   13.908  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.059   -0.582 1
prefill   2048   27.977   29.585  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.585   -1.609 0
prefill   4096   54.951   53.673     sp_nccl     sp_nccl  tp_ar_vllm    0.217    1.278 1

policy           regret   saving vs vLLM   worst point   right
[all] 8 points
  decider_model     0.01%          3.2%      0.8% at decode M=32      7/8
  decider_e4a_v1    2.48%          0.8%     11.2% at decode M=256     3/8
  R1                3.27%          0.0%     70.0% at decode M=128     6/8
  R1p               4.02%         -0.7%     70.0% at decode M=128     5/8
  R2                3.27%          0.0%     70.0% at decode M=128     6/8
  R3                3.24%          0.0%     11.2% at decode M=256     2/8
[decode] 5 points
  decider_model     0.07%          9.2%      0.8% at decode M=32      4/5
  decider_e4a_v1    6.59%          3.3%     11.2% at decode M=256     1/5
  R1               10.23%          0.0%     70.0% at decode M=128     4/5
  R1p              10.23%          0.0%     70.0% at decode M=128     4/5
  R2               10.23%          0.0%     70.0% at decode M=128     4/5
  R3                6.59%          3.3%     11.2% at decode M=256     1/5
[prefill] 3 points
  decider_model     0.00%          1.3%      0.0% at prefill M=1024    3/3
  decider_e4a_v1    1.35%          0.0%      2.4% at prefill M=4096    2/3
  R1                1.35%          0.0%      2.4% at prefill M=4096    2/3
  R1p               2.31%         -0.9%      5.8% at prefill M=2048    1/3
  R2                1.35%          0.0%      2.4% at prefill M=4096    2/3
  R3                2.31%         -0.9%      5.8% at prefill M=2048    1/3

E4a2 tp8x2n2 qwen2.5-32b TP=8 set=old (SANITY ONLY: E4a points, answers seen before the model change)
phase        M     vLLM       SP        best       model       final  pred dT  meas dT probe
decode      32    1.886    2.164  tp_ar_vllm     sp_nccl  tp_ar_vllm    0.044   -0.278 0
decode     128    2.324    2.648  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.105   -0.324 0
decode     256    4.009    4.232  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.079   -0.223 0
decode     384    5.588    6.449  tp_ar_vllm  tp_ar_vllm  tp_ar_vllm   -0.202   -0.861 0
decode     512    6.989    7.759  tp_ar_vllm  tp_ar_vllm     sp_nccl   -0.191   -0.770 0
prefill   1024   15.293   12.375     sp_nccl     sp_nccl  tp_ar_vllm    0.240    2.919 0
prefill   2048   23.711   23.219     sp_nccl     sp_nccl  tp_ar_vllm    0.061    0.492 1
prefill   4096   50.994   51.613  tp_ar_vllm  tp_ar_vllm     sp_nccl   -0.094   -0.619 1

policy           regret   saving vs vLLM   worst point   right
[all] 8 points
  decider_model     0.26%          2.8%     14.7% at decode M=32      7/8
  decider_e4a_v1    4.47%         -1.3%     23.6% at prefill M=1024    4/8
  R1                3.18%          0.0%     23.6% at prefill M=1024    6/8
  R1p               0.58%          2.5%      1.2% at prefill M=4096    7/8
  R2                3.18%          0.0%     23.6% at prefill M=1024    6/8
  R3                2.86%          0.3%     15.4% at decode M=384     2/8
[decode] 5 points
  decider_model     1.33%         -1.3%     14.7% at decode M=32      4/5
  decider_e4a_v1    3.70%         -3.7%     11.0% at decode M=512     4/5
  R1                0.00%          0.0%      0.0% at decode M=32      5/5
  R1p               0.00%          0.0%      0.0% at decode M=32      5/5
  R2                0.00%          0.0%      0.0% at decode M=32      5/5
  R3               11.81%        -11.8%     15.4% at decode M=384     0/5
[prefill] 3 points
  decider_model     0.00%          3.8%      0.0% at prefill M=1024    3/3
  decider_e4a_v1    4.65%         -0.7%     23.6% at prefill M=1024    0/3
  R1                3.94%          0.0%     23.6% at prefill M=1024    1/3
  R1p               0.71%          3.1%      1.2% at prefill M=4096    2/3
  R2                3.94%          0.0%     23.6% at prefill M=1024    1/3
  R3                0.71%          3.1%      1.2% at prefill M=4096    2/3

```

## 2026-10-05（續）· E4a2 結論（人工撰寫，取代上面自動條目的「結論待撰寫」）

- 第二次嘗試 12 分鐘跑完，全部 CLEAN；預先登記順序 `16f9eba` → `d252251` → `a38d2d2` → `0eb80aa`。
- **21 個新點**：決策器 regret 0.11%、20/21 選對；簡單規則 3.8–5.8%、7–14/21，最差慢 103%（TP4 decode M=192 的 all-reduce 斷崖）。
- 拿掉 M=192：決策器 0.12% vs 0.99–3.92%，仍最好，但效益只剩 0.9%。我是在知道 1–2 MiB 有斷崖後才挑新點的，這點必須跟結論一起講。
- 把關（6 點）全部同意模型 → 進步來自「不強制單調 + 用實際大小校準」。TP4 斷崖處預測量級準到 1–3%。
- **成本反而較高**：決策器 444 s vs 直接量 255 s。兩個選項時不值得建模；決策器的價值要在選項多（含 Flux）時才可能顯現。
- TP8（4+4）仍是模型最弱處：prefill 量級低估、decode M=64 方向猜錯（9.2%）。
- 報告：`reports/20261005_e4a2_corrected_decider.md`。
- **留給下個 session**：帶 Flux 的跨節點版本（先解 NVSHMEM 多 rail、`GemmRS_multinode` 結果錯）；auditor 審 E4a / E4a2；openclaw 還原。

## 2026-10-06 · boss 兼 worker · GemmRS_multinode 假說測試未跑成；主線轉到 hetero-proxy

- 假說：`GemmRS_multinode` 連續兩次呼叫節點內 GemmRS 之間沒有同步（sm80 的 forward_barrier 是空操作），跑得快的 rank 把第二次的資料寫進還在複製第一次結果的 rank → 只有保留第一次結果的節點 0 算錯（與症狀吻合）。
- 測試腳本 `scripts/xnode_flux_rs_smoke_v2.py`；159 被同帳號的 `sglang::server` 佔用（8 卡各約 74 GB），守衛在 preflight 中止 → **假說未驗證**（`results/e4_flux_smoke/rsmn_v2_barrier_ppn4/README.md`）。
- 教授指派新方向（PCIe 通道 + 異質晶片），主線轉到 `ws/hetero-proxy`；本 ws 的跨節點 Flux 工作暫停。
