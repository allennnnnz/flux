# ws/diag-overlap — STATUS

**本 workstream 唯一權威。** 第 1 節由 boss 寫入，worker 不改；第 2 節起由 worker 維護。

建立：2026-09-23（boss）
最後更新：2026-09-23（boss，建立）
狀態：**未開始**

---

## 1. 目標與成功標準（boss）

### 問題

Flux AG+GEMM 在 comm-bound 形狀下幾乎不 overlap。Phase 0 量到（`docs/PHASE0_FINDINGS.md` 2.6）：

| N | overlap 總時間 | GEMM-only | comm（獨立） | 完全串行 | 理想下界 max(comm, GEMM) | 效率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4096** | **0.736 ms** | 0.278 | 0.468 | 0.746 | **0.468** | **~2%** |
| 8192 | 0.677 | 0.564 | 0.468 | 1.032 | 0.564 | 76% |
| 49152 | 2.747 | 2.562 | 0.468 | 3.030 | 2.562 | 60% |

N=4096 的實測距完全串行只差 1.3%。這是最大已知改善空間（36%），也是異質計算的前置
問題：若 Flux 在 comm-bound 形狀下連 NVLink 都藏不住，換成慢 40 倍的 PCIe 只會更糟。

### 形狀

M=4096, K=12288, world 8, **bf16**, **All2All + `use_read=True`**（D-005）。
主：N=4096。對照：N=2048、N=8192。weight column-shard 為 `(N/8, K)`。

### 步驟

1. **kernel 端計時**：在 AG-fused GEMM kernel 內用 `%globaltimer` 記錄每個 CTA
   等 signal 的開始與結束時間；host / copy 端記錄每個 signal 被設定的時間。畫出
   「signal 到達時間」與「CTA 開始計算時間」的分佈。
2. **確認 All2All 模式下每個 shard 的 signal 數量與 comm tile 大小。**
   入口：`src/coll/ths_op/all_gather_op.cc`（`ag_signal_ptr()`、`copy_all_to_all`）、
   `src/ag_gemm/ths_op/all_gather_gemm_op.cc`（`forward_impl`）、
   `src/ag_gemm/sm80_all_gather_gemm_threadblock_swizzle.hpp`（tile → signal 對應）。
3. **若 signal 集中在尾端才到位**：把 comm tile 依序減半（直到 GEMM tile 大小），
   量 overlap efficiency 變化。
4. **回報**與理想下界 0.468 ms 的差距，以及失效原因的判斷。

### 成功標準

- N=4096 總時間從 0.736 ms 降到 **0.55 ms 以下**；**或**
- 給出 nsys 時間軸 + kernel 內計時證據，證明下界不可達並解釋原因。

### 量測規範

- ECT 用交錯法（`CLAUDE.md` 5.1 第 5 條），≥200 輪，`common/measure/clock_logger.py`。
  可直接沿用 `experiments/2026-09-23-phase0-a100-nvlink-pcie/scripts/flux_ag_gemm_baseline_v2.py`
  的骨架。
- GEMM-only 參考用 `AGKernel.gemm_only()`（同一 kernel，signal 全 true）。
- 任何「並行 / 串行」主張附 nsys profile。

### 假設（推論，待驗證）

- **[推論]** N=4096 時 GEMM（0.278 ms）短於通訊（0.468 ms），GEMM 把自己能算的 tile
  很快算完，剩下時間都在等最後到達的 shard；All2All 的 signal 粒度是整個 shard
  （12.6 MB），沒有 sub-shard 的 tile 級 signal 可讓 GEMM 提早開始。
  驗證：步驟 1、2。

---

## 2. 目前成立（worker 維護）

（尚無。）

## 3. 撤回表（worker 維護）

| 撤回主張 | 出處 | 原因 | 替代 |
| --- | --- | --- | --- |
| （無） | | | |

## 4. 下一步（worker 維護）

1. 讀 `CLAUDE.md` 第 2 節陷阱表與第 5 節準則。
2. 從步驟 2 開始（讀程式碼確認 signal 粒度）——比步驤 1 便宜，且決定步驤 1 要 instrument 什麼。
3. 重跑 N=4096 基線一次，確認環境與 Phase 0 數字一致（錨點：0.736 / 0.278 / 0.468）。

## 5. 產出到 common 的數字

（尚無。有的話寫進 `common/cost_model/params.json` 的 `flux.overlap` 區。）
