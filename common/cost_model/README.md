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

## predictor/（2026-10-02，ws/fusion-dispatch 第三階段 G1，D-008）

分析式成本模型：每條路徑的時間 = 通訊 + 計算 − 重疊，用來決定 Flux 開或關，以及走哪條通訊路徑。
純 Python（無 numpy），pixi、vLLM venv、系統 python3 都能跑。

| 檔案 | 內容 |
| --- | --- |
| `curves.py` | 分段 α-β 曲線 |
| `comm.py` | NCCL 集合通訊；Flux AllGather = α_sync + W × copy(bytes / W) |
| `gemm.py` | roofline + tile 補齊 |
| `flux_config.py` | 查 Flux 實際用的 config |
| `overlap.py` | 融合 AG+GEMM 用 CUTLASS stream-K tile 排程的事件模擬；GemmRS 模型 |
| `paths.py` | 各路徑的預測時間 |
| `profile.py` | 參數檔（JSON，附來源）與擬合 |

- 設計與驗證：`ws/fusion-dispatch/reports/20261002_plan_general_dispatcher.md`、`20261002_g1_predictor.md`。
- 校準後的參數檔放 `hw_profiles/<host>_tp<W>_<state>.json`（G2 建立）。
- G1 的參數（用決策表元件擬合，僅供對照）在 `ws/fusion-dispatch/results/g1_predictor/`。
- 使用範例：

```python
import sys; sys.path.insert(0, "common/cost_model")
from predictor import HardwareProfile, FluxConfigs, predict_ag
prof = HardwareProfile.load("ws/fusion-dispatch/results/g1_predictor/profile_g1_all_gpu.json")
arms, comps, cfg = predict_ag(prof, FluxConfigs(8), M=4096, n=1280, K=8192)  # ms per path
```
