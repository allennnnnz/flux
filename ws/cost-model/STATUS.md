# ws/cost-model — STATUS

**本 workstream 唯一權威。** 第 1 節由 boss 寫入，worker 不改；第 2 節起由 worker 維護。

建立：2026-09-23（boss）
最後更新：2026-09-23（boss，建立）
狀態：**未開始 — 阻塞中**，等 `ws/diag-overlap` 與 `ws/hetero-proxy` 的第一批數字。

---

## 1. 目標與成功標準（boss）

### 問題

把 Phase 0 與後續量測整合成預測模型：

- **輸入**：形狀（M, N, K, world）、各路徑頻寬與延遲、同步粒度（signal 對應多少 bytes）、
  延遲隱藏方式（動態 / 靜態）、拓撲配置。
- **輸出**：預期總時間、overlap efficiency、瓶頸路徑。

### 為什麼需要

- 異質計算的晶片還沒到；要在到手前決定介面與能力等級，只能靠模型。
- Phase 0 已有一個驗證過的雛形：`t(α) = max((1-α)S/B_nv, αS(1/B_d2h + 1/B_h2d))`
  在 α=0.05 預測 109 GB/s、實測 106.6（D_CORRECTION D.3）。模型要從這裡長出來。

### 步驟

1. 讀 `common/cost_model/params.json` 的現有錨點；設計 schema 讓後續 workstream 直接追加。
2. 先做**通訊側**模型：給定拓撲配置與 chunk，預測 staging / NVLink / PCIe 時間。
   用 Phase 0 的 D 段 α 掃描（7 點 + switch-aware 5 點）驗證。
3. 再做**overlap 模型**：給定 GEMM tile 時間分佈與 signal 到達時間分佈，預測總時間。
   用 `ws/diag-overlap` 步驟 1 量到的分佈驗證——**這是阻塞點**。
4. 加入延遯隱藏方式維度（動態 vs 靜態），用 `ws/hetero-proxy` 等級 0 的實測驗證。
5. 回報每個預測的誤差；誤差 > 15% 的情境列為模型缺口。

### 成功標準

- 對 Phase 0 D 段 12 個 α 點，預測誤差 < 10%。
- 對 diag-overlap 的 N ∈ {2048, 4096, 8192} 實測，預測誤差 < 15%。
- 模型以 `common/cost_model/` 下可 import 的介面提供，其他 ws 可查詢。

---

## 2. 目前成立（worker 維護）

（尚無。）

## 3. 撤回表（worker 維護）

| 撤回主張 | 出處 | 原因 | 替代 |
| --- | --- | --- | --- |
| （無） | | | |

## 4. 下一步（worker 維護）

阻塞。可先做步驟 1、2（不依賴其他 ws）。

## 5. 依賴

| 需要 | 來自 | 狀態 |
| --- | --- | --- |
| signal 到達 / CTA 開始時間分佈 | ws/diag-overlap 步驟 1 | 未開始 |
| 競爭位置判定 | ws/hetero-proxy 1a | 未開始 |
| staging 固定開銷 | ws/hetero-proxy 1b | 未開始 |
