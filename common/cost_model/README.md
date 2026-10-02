# common/cost_model

`params.json` — 所有 workstream 共用的量測參數。

## 規則

- 每個數字附 `source`（檔路徑）、`date`、`method` 一句話、`status`
  （`measured` / `inferred` / `derived` / `not measured`）。
- **只有量測方能改自己量的數字。** 讀取方不改。
- 新量測寫入前先對照既有錨點；偏差 > 20% 且無解釋 → 先懷疑量測（`CLAUDE.md` 5.1 第 2 條）。
- `inferred` 條目要附「如何驗證」與負責的 workstream；驗證後改 `measured` 並補 source。
- 舊值被取代時不刪，加 `superseded_by` 與日期，值搬進 `history` 陣列。

## 目前狀態

2026-09-23 由 boss 以 Phase 0 數字初始化。`inferred` / `not measured` 條目：

| 條目 | 負責 |
| --- | --- |
| `pcie_bidirectional.contention_locus` | ws/hetero-proxy 1a |
| `staging.fixed_overhead_ms` | ws/hetero-proxy 1b |
| `flux.overlap.*` | ws/diag-overlap |

查詢介面（`query.py` 等）由 `ws/cost-model` 建立。
