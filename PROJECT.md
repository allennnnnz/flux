# PROJECT.md — 單一入口

**boss session 專屬。** worker 只讀。

最後更新：2026-10-02（boss）

閱讀順序：`CLAUDE.md` → 本檔 → `ws/<你的 workstream>/STATUS.md` → 該 JOURNAL 最近條目。
Phase 0 背景：`docs/PHASE0_FINDINGS.md`（設計規則）；細節才看 `docs/PHASE0_STATUS.md`。

---

## 1. 目前狀態一句話

Phase 0 結案。A100 雙通道方案取消（DECISIONS D-001）。2026-09-29 與教授討論後新增
`ws/fusion-dispatch`（何時不開 Flux 比開好），列為最高優先（D-006）。E0–E3 完成：
decode 大小（M ≤ 512）四層都是不開 Flux 較快，每層查表的決策器 regret 0–2%；**待 auditor**。
其餘三個 workstream 尚未有 worker session 進場。

## 2. Workstream 總表

| workstream | 優先 | 狀態 | 依賴 | 負責 session | 最近更新 |
| --- | --- | --- | --- | --- | --- |
| `ws/fusion-dispatch` | **最高**（D-006、D-007） | 決策器完成並驗證（真 vLLM 對照：prefill 快 9.3–17.7%、80 block 成立）；F4 端到端可行性已確認，待使用者確認後開始；待 auditor。新 session 先讀其 STATUS 第 0 節 | — | boss 兼 worker | 2026-10-02 |
| `ws/diag-overlap` | 第二 | 未開始 | — | 未指派 | — |
| `ws/hetero-proxy` | 第二，可與上並行 | 未開始 | — | 未指派 | — |
| `ws/cost-model` | 第三 | 未開始 | 前兩者的數字 | 未指派 | — |

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
| 2026-10-02 | 量測一律經 `common/measure/exclusive_guard.py`（使用者要求確保獨佔）。建議寫入 `CLAUDE.md` 5.1。 | fusion-dispatch 驗證報告 V12 | 待裁決 |
| 2026-09-30 | Flux `AGKernel.forward` 不能被 CUDA graph capture（cp_stream 未 join）；`use_cuda_core_local` / CUDA-core AG 皆不支援 bf16。建議加進 `CLAUDE.md` 陷阱表。 | fusion-dispatch F0.4、F2 報告 | 待裁決 |
| 2026-09-30 | `flux.testing.initialize_distributed()` → `init_seed()` 把 cuBLAS 設成非 production（launch 13 → 71 µs、部分形狀 +26%），任何 Flux vs torch 比較都偏向 Flux。建議加進 `CLAUDE.md` 陷阱表。 | fusion-dispatch E0 報告第 1 節 | 待裁決 |

## 4. 下一步（boss）

0. `ws/fusion-dispatch`：安排 auditor 審 `reports/20260929_e0_anchor.md` 與 `reports/20260930_e1_e3_dispatch_map.md`；之後做 E5（GEMM+RS）與 E4（CUDA graph）。
1. 指派第一個 worker session 到 `ws/diag-overlap`。
2. `ws/hetero-proxy` 可同時開一個 worker，先做 1a/1b 兩項補充量測。
3. 兩者各有第一份 report 後，安排 auditor。
4. 第一批 auditor 通過的結論寫進 `DECISIONS.md`，並更新 `params.json`。

## 5. 已關閉

- **Phase 0**（2026-09-23）：檔案庫 `experiments/2026-09-23-phase0-a100-nvlink-pcie/`，
  記錄 `docs/PHASE0_STATUS.md`，結論 `docs/PHASE0_FINDINGS.md`。
- **A100 NVLink+PCIe 雙通道 Phase 1 / Phase 2**：取消，見 D-001。
