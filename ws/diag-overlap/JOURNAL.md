# ws/diag-overlap — JOURNAL

Append-only。每條：日期、角色、做了什麼、卡在哪、留給下個 session 的話。
需要 boss 裁決的事項標 **「需 boss 裁決」**。

---

## 2026-09-23 · boss · 建立

- 建立 workstream，寫入 STATUS 第 1 節目標與成功標準。
- Phase 0 相關數字與入口已整理在 STATUS 1 與 `docs/PHASE0_FINDINGS.md` 2.6。
- 可直接沿用的腳本：`experiments/2026-09-23-phase0-a100-nvlink-pcie/scripts/flux_ag_gemm_baseline_v2.py`
  （交錯 ECT harness）、`flux_comm_baseline.py`（獨立 AG 計時）。
- 留給 worker：先讀程式碼確認 signal 粒度（STATUS 步驟 2），再決定 instrument 範圍。
