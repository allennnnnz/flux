# PROJECT.md — 單一入口

**boss session 專屬。** worker 只讀。

最後更新：2026-10-30（選題重找）

閱讀順序：`CLAUDE.md` → 本檔 → `ws/<你的 workstream>/STATUS.md` → 該 JOURNAL 最近條目。
Phase 0 背景：`docs/PHASE0_FINDINGS.md`（設計規則）；細節才看 `docs/PHASE0_STATUS.md`。

---

## 1. 目前狀態一句話

Phase 0 結案。A100 雙通道方案取消（DECISIONS D-001）。2026-09-29 與教授討論後新增
`ws/fusion-dispatch`（何時不開 Flux 比開好），列為最高優先（D-006）。查表版決策器完成並驗證；
第三階段（泛用決策器，D-008 / D-009）**G0–G4 完成**：
- 決策器 v2 =「模型（通訊 + 重疊）+ 實測單卡 GEMM + 少量多卡把關」；
- 在全新情境（Qwen2.5-32B 8 / 4 卡、Llama-3-8B 4 卡）上，決策都在量測前 push：
  - op regret 0.17%（G3 方法 1.20%、固定門檻 2.30%）；
  - **block regret 0.02%，比 vLLM 預設省 8.0%（prefill 10.9%）**；
- 計劃成功標準除探測量外全部達成；
- 陷阱已寫入 `CLAUDE.md`。

G4 之後的事後分析（`ws/fusion-dispatch/reports/20261003_g4_flux_value.md`）：
- prefill 的好處大多來自 Flux：同一種切法下快 7–20%；
- decode 輸在切法；
- 在這台，簡單規則「decode vLLM、prefill Flux」幾乎跟決策器一樣好（省 7.5% vs 8.0%），所以決策器的價值要在別的硬體上證明。

下一步：**跨機器驗證**。使用者提供 gpu1（EE325）；**2026-10-03 唯讀盤點完成：4× V100-PCIE（sm70）、無 NVLink，Flux 不能跑 → 只能做縮小版（切法 + 校準可攜性），核心問題需要 sm80+ 無 NVLink 的機器，待使用者決定**（`ws/fusion-dispatch/reports/20261003_gpu1_inventory.md`）；使用者另外提供 **css-host-159**（同型 8× A100，與本機以 8 條 100 GbE RoCE 相連，唯讀盤點 `ws/fusion-dispatch/results/node159_inventory/`），可做跨節點驗證（Flux 能跑、連線慢），**已核准、進行中**：環境已複製、守衛 v2、RDMA 錨點 12.3 GB/s/rail；159 的 `~/.nccl.conf` 已停用、NCCL 跨節點正常（頻寬受 GPU→網卡讀取限制，約 7 GB/s/GPU）；Flux 跨節點：AG 能跑但 NVSHMEM 只能用單一 rail（缺 rail 間路由），多節點 RS 結果錯 → 方向待使用者決定；**10-05 E4a 完成（`ws/fusion-dispatch/reports/20261005_e4a_slow_link_layout.md`）：慢連線上簡單規則最差慢 116%（單機 7.9%）；決策器整體 regret 最低（2.78% vs 4.1–4.7%）但只猜對 13/24，原因已定位（跨節點 NCCL 非單調、模型假設單調）；下一步修模型後用新點重測**；**10-05 E4a2 完成（`ws/fusion-dispatch/reports/20261005_e4a2_corrected_decider.md`）：修正後的決策器在 21 個新點上 20/21 選對、regret 0.11%，簡單規則 3.8–5.8%（最差慢 103%）；限制：新點在已知斷崖區間下挑選、只有兩種切法、此規模下決策器比直接量貴**；對手規則與成功標準已凍結。F4 / G6 待決定。**待 auditor**。其餘三個 workstream 尚未有 worker session 進場。

## 2. Workstream 總表

| workstream | 優先 | 狀態 | 依賴 | 負責 session | 最近更新 |
| --- | --- | --- | --- | --- | --- |
| `ws/fusion-dispatch` | **最高**（D-006、D-007、D-008） | 查表版決策器完成並驗證；**第三階段：泛用決策器，G0–G4 完成（`reports/20261003_g4_dispatcher.md`：op 0.17%、block 0.02%），下一步：跨機器驗證（gpu1 盤點完成：V100 PCIe，Flux 不能跑，待決定範圍）**；F4 延後；待 auditor。新 session 先讀其 STATUS §0 | — | boss 兼 worker | 2026-10-03 |
| `ws/diag-overlap` | 第二 | 未開始 | — | 未指派 | — |
| `ws/hetero-proxy` | **最高（2026-10-06 教授指派）** | PCIe 異質協同：介面原型 hxb 完成；真實模型上 PCIe 晶片能與 NVLink 協同（每字代價 0.3–0.8 ms），但尚未加速（晶片被串在路線上）；C（PCIe 額外通道）因干擾 NVLink 無效益；2026-10-15 初步測試主流角色：MoE 專家外放有用（比搬權重快 5–7×）、attention 外放對低頻寬裝置無用；**2026-10-16 第一個加速實證：CPU 當 KV 外存，被擠掉的對話回來時經 PCIe 逐層搬回、邊搬邊算——單一請求比重算快 3.6×（1k）–19×（32k），逐位元正確；多人服務（K2）輕負載首字快 2.15×、重負載首字中位數快 35×、吞吐量 +7.5%，外存的尾端受 GPU 上正在生成的對話數限制**；**2026-10-28 教授建議不做 KV 題目 → 新題目 X 系列：PCIe 搬運對 TP 推論的干擾（K4.1：背景搬運讓 decode 慢 45–145%）與 Flux 式訊號驅動排程（壓到約 3%）**；**X1d（2026-10-28）：NCCL 小 all-reduce 被拖慢的主因是 NCCL 預設在 CUDA graph 內每個操作前後加的 event 節點（`NCCL_GRAPH_MIXING_SUPPORT=1`），關掉後 decode 一步的干擾 +197–202% → +2.3–2.5%、沒搬運時也快 24%；Simple 協定另受系統層級屏障影響（機制已見於 NVIDIA/nccl #2218）**；**2026-10-30 選題重找（使用者：通訊優化、推論或訓練、10 週一篇架構完整的論文）：T1 實測 8 卡背景 GPU→主機搬運讓 16–32 MiB 集合通訊慢 2–3 倍、只要一個交換器（2 卡）在搬就幾乎一樣、最好的 NCCL 協定隨搬運翻轉 → 推薦題目 T「訓練時 offload 干擾的量測＋排程器」（單機即可），備案 F「fusion-dispatch 成文」，MoE 第三；待使用者決定（`ws/hetero-proxy/reports/20261030_topic_search.md`）**；之前 2026-10-29 選題：使用者認為關設定不算貢獻 → MoE 冷門專家放在 PCIe 裝置、8 張 NVLink GPU 一起用；以 KTransformers＋SGLang 的 8 卡做法為改良對象（原始碼確認只有 GPU0 和 CPU 交換、用 CPU 回呼等待），M0 驗證通過（CPU 算＋8 卡經 4 條上行線搬來算：4 token 1.24×、單線只有 1.06×；互搶吃掉約一半理想增益），真實模型用本機已有的 Kimi-K2.6（`reports/20261029_lit_moe_hybrid.md`、`results/moehy/m0_v1/`）**；新 session 先讀其 STATUS §0 | — | boss 兼 worker | 2026-10-29 |
| `ws/cost-model` | 第三 | 模型目標由 fusion-dispatch 第三階段執行；預測器 v1 已建於 `common/cost_model/predictor/`（G1），校準參數檔在 `common/cost_model/hw_profiles/`（G2）（D-008） | — | 未指派 | 2026-10-02 |

各 workstream 的目標、步驤、成功標準寫在各自的 `STATUS.md` 第 1 節，由 boss 在建立時
寫入，worker 不改目標、只更新進度。以下是簡述。

### ws/fusion-dispatch — 何時不開 Flux 比開好（推論的變動 M）

推論時每層 N、K 固定，M（token 數）每步變。論文指出極小 m 時 Flux 可能輸給不 overlap 的基準。
先量出「每層 × 每個 M」fused 與幾種非 fused 路徑的地圖，再評估門檻 / 查表 / 成本模型三種決策器。
設計：`ws/fusion-dispatch/reports/20260929_experiment_design.md`。
**規則：量到證據顯示值得之前，不改 `src/`**（D-006）。

### ws/diag-overlap — Flux 在 comm-bound 形狀下不 overlap

Phase 0 發現 N=4096（M=4096, K=12288, bf16, All2All pull）的 AG+GEMM 總時間 0.736 ms，
距完全串行 0.746 ms 只差 1.3%；overlap efficiency ≈ 2%，對比 N=8192 的 76%、
N=49152 的 60%。理想下界 max(comm 0.468, GEMM 0.278) = 0.468 ms。

**成功標準**：N=4096 總時間降到 0.55 ms 以下；或給出 nsys / kernel 計時證據證明下界
不可達並解釋原因。

### ws/hetero-proxy — 後端抽象層與 A100 代理

Phase 3 的起點。先補 Phase 0 兩項量測（雙向競爭位置、staging 固定開銷），再定義後端
介面與能力分級，用 A100 模擬 PCIe-only 裝置。先把等級 0 完整流程跑通。

### ws/cost-model — 預測模型

輸入形狀、頻寬、延遲、同步粒度、延遲隱藏方式；輸出預期總時間與 overlap efficiency。
用 diag-overlap 的實測驗證預測誤差。**在前兩者有第一批數字前不開工。**

## 3. 跨 workstream 的待裁決事項

worker 在 JOURNAL 標「需 boss 裁決」的事項會被 boss 搬到這裡。

| 日期 | 事項 | 來源 | 狀態 |
| --- | --- | --- | --- |
| 2026-09-29 | Phase 0 的 N=4096（每卡 n=512, K=12288, bf16, RCR, 無 bias）在 A100 tp8 tuning registry 中**無條目**，跑的是 fallback hparams；N=8192、N=49152 在 M=4096 有條目。diag-overlap 的「~2% overlap」需考慮 config 因素。驗證：對該形狀做 `profiling()`，比較最佳 hparams 與 fallback 的 GEMM-only。 | fusion-dispatch 設計 3.1 | 待 diag-overlap 進場時處理 |
| 2026-09-29 | fusion-dispatch 的 A\* arm 需在 pybind `forward` 暴露 `hparams` 並 rebuild；diag-overlap 也要 instrument `src/ag_gemm/`。 | fusion-dispatch 設計 5.3 | 2026-09-30：E2 估計 decode 區調校收益 ≤ 6%，不值得；只有少數斷崖 M 有 9–17%。建議暫不改，等部署 M 桶確定 |
| 2026-09-30 → 10-02 更新 | Phase 0 E.1「Flux AG 比 NCCL 快 13–22%」：原腳本 10/02 完全重現；與本 ws 的差異來自 SM 時脈狀態（原腳本量 NCCL 時 GPU 在 1155 MHz；NCCL 受 SM 時脈影響、Flux copy engine 不受）。**建議改寫為「依時脈狀態而定」，不撤回**。 | fusion-dispatch 驗證報告 V6 | 待裁決 |
| 2026-10-02 | 量測一律經 `common/measure/exclusive_guard.py`（使用者要求確保獨佔）。建議寫入 `CLAUDE.md` 5.1。 | fusion-dispatch 驗證報告 V12 | **已處理 2026-10-03**：寫入 `CLAUDE.md` 5.1 第 11 條（D-009） |
| 2026-09-30 | Flux `AGKernel.forward` 不能被 CUDA graph capture（cp_stream 未 join）；`use_cuda_core_local` / CUDA-core AG 皆不支援 bf16。建議加進 `CLAUDE.md` 陷阱表。 | fusion-dispatch F0.4、F2 報告 | **已處理 2026-10-03**：寫入陷阱表（D-009） |
| 2026-10-30 | **論文主題目三選一**：T（訓練時主機↔GPU 搬運拖慢 NVLink 集合通訊：量測＋干擾模型＋協定選擇＋搬運節拍器，接進 FSDP2 / DeepSpeed）、F（fusion-dispatch 成文）、M（MoE 冷門專家）。另：css-host-159 的 8 卡被使用者帳號的 SGLang 服務佔用、少一張網卡（rail 5/6） | `ws/hetero-proxy/reports/20261030_topic_search.md` 第 7 節 | 待使用者決定題目；**使用者 10-30：只用 css-host-158**（不再需要 159）；**10-30 驗證完成（`ws/hetero-proxy/reports/20261030_t_validation_result.md`）：T 有條件值得——現象真實、新、改設定修不掉，但訓練整步能拿回 3–8%（峰值損失 24%）；選 T / F / F＋T** |
| 2026-10-02 | **給 ws/diag-overlap 的假說**（fusion-dispatch G1）[推論]：<br>• 現象：Phase 0 N=4096（每卡 n=512、K=12288、M=4096）幾乎不重疊。<br>• 原因：融合 GEMM 只有 128 個 tile（1.19 波），預設 config 是 stream-K，每個 block 都要等最晚到的 shard。<br>• 模型對沒看過的 P0-4096 / P0-8192 預測藏住 3% / 70%，實測 7% / 65%。<br>• 預測：換 data-parallel + RasterAlongN config 可降到 0.54–0.58 ms（目標 < 0.55）。<br>• 驗證：指定 config 量 A 與 gemm_only，並用 nsys 看 CTA 開始時間。 | fusion-dispatch G1 報告 3.5 | 待 diag-overlap 進場時處理 |
| 2026-10-02 | Flux 登錄表 `// PCIE` 區段的 config 優先生效（emplace 第一筆），在本機 NVLink 上 GEMM 比 cuBLAS 慢 1.23–1.58×（G-FC1 M=1024、L-GU / L-QKV M=4096）。建議加進 `CLAUDE.md` 陷阱表。 | fusion-dispatch G1 報告第 4 節 | **已處理 2026-10-03**：寫入陷阱表（D-009） |
| 2026-10-02 | 同一 process 內先建立再銷毀一組 Flux op（AGKernel / AllGatherOp）、再建第二組，會隨機卡死（所有 rank 卡在 synchronize，GPU 空轉）；一組一個 process 即正常。建議加進 `CLAUDE.md` 陷阱表。 | fusion-dispatch G2 報告第 3 節 | **已處理 2026-10-03**：寫入陷阱表（D-009） |
| 2026-09-30 | `flux.testing.initialize_distributed()` → `init_seed()` 把 cuBLAS 設成非 production（launch 13 → 71 µs、部分形狀 +26%），任何 Flux vs torch 比較都偏向 Flux。建議加進 `CLAUDE.md` 陷阱表。 | fusion-dispatch E0 報告第 1 節 | **已處理 2026-10-03**：寫入陷阱表（D-009） |

## 4. 下一步（boss）

0. `ws/fusion-dispatch`：
   - 跨機器驗證：gpu1 唯讀盤點完成（2026-10-03，V100 PCIe sm70，Flux 不能跑）→ **決定**：gpu1 做縮小版 / 找 sm80+ 無 NVLink 機器 / 先做 E0；
     `exclusive_guard.py` 在 gpu1 有兩個誤判 bug（使用者名稱截斷、CPU% 生命期平均），量測前要先修（v2）；
   - 決定 F4（vLLM 端到端，用決策器 v2）/ G6（總報告）的先後；
   - 安排 auditor 審 E0、E1–E3、G1–G4 報告。
1. 指派第一個 worker session 到 `ws/diag-overlap`。
2. `ws/hetero-proxy`（**2026-10-06 起為主線**，教授指派）：**2026-10-30 選題重找，等使用者在 T / F / M 中選一個；選 T 則第 1 週先做否證實驗（真實訓練步驟中干擾占比 ≥ 10%）。** 以下為較早的狀態——H0、I1–I4 完成（2026-10-09，未審）。**2026-10-14 定案 E2E 計劃**（STATUS §0）：Qwen2.5-32B 真實權重，A = 8 卡 NCCL、B = 8 卡 + CPU（PCIe）、C = 8 卡 NVLink + PCIe 同時傳；已告知使用者 B 預期稍慢、C 本機預期最多約 +0.5%（推論）。之後：I3 / I4 報告、auditor。待教授確認晶片型號。
3. 兩者各有第一份 report 後，安排 auditor。
4. 第一批 auditor 通過的結論寫進 `DECISIONS.md`，並更新 `params.json`。

## 5. 已關閉

- **Phase 0**（2026-09-23）：檔案庫 `experiments/2026-09-23-phase0-a100-nvlink-pcie/`，
  記錄 `docs/PHASE0_STATUS.md`，結論 `docs/PHASE0_FINDINGS.md`。
- **A100 NVLink+PCIe 雙通道 Phase 1 / Phase 2**：取消，見 D-001。
