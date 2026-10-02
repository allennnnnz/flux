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
