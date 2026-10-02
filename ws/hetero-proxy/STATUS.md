# ws/hetero-proxy — STATUS

**本 workstream 唯一權威。** 第 1 節由 boss 寫入，worker 不改；第 2 節起由 worker 維護。

建立：2026-09-23（boss）
最後更新：2026-09-23（boss，建立）
狀態：**未開始**

---

## 1. 目標與成功標準（boss）

### 問題

Phase 3 的起點：為「PCIe-only 晶片 + A100 NVLink 域」的異質計算建立後端抽象層，
並以 A100 代理尚未到手的晶片。設計輸入是 `docs/PHASE0_FINDINGS.md` 第 3 節的八條規則。

### 步驟

**1. 先補 Phase 0 兩項量測**（各自出一份 `reports/`，數字進 `params.json`）：

- **1a. 雙向競爭位置驗證。** Phase 0 只量到「同一張 GPU 同時 H2D + D2H 時 H2D 掉到
  6.46 GB/s」；競爭發生在哪一層是**推論**。設計：
  - GPU0 只 H2D + GPU2 只 D2H（不同 switch）同時進行；
  - 對照 GPU0 只 H2D + GPU1 只 D2H（同 switch）；
  - 對照 GPU0 單獨 H2D。
  判讀：不同 switch 不降、同 switch 降 → 競爭在 switch 上行；兩者皆不降 → 只在單卡
  端點；兩者皆降 → 主機側。
  工具：`experiments/2026-09-23-phase0-*/scripts/bidirectional_bandwidth_v2.py` 已有
  `--direction h2d|d2h`，但目前**對所有 device 套同一方向**；需改成 per-device 方向
  （小改：把 `direction` 變成 dict）。沿用其逐 copy event + 重疊窗口方法。
- **1b. staging 機制固定開銷。** α=0 但啟用 staging 機制（建立 stream、event、
  pinned buffer、雙緩衝 handshake）相對純 NVLink 的時間增加。工具：`dual_path_v3.py`
  加一個 `--staging-overhead-only` 模式（走完 handshake 但 staging_elems=0 或極小）。
  這個數字是等級 0/1 後端的固定成本項。

**2. 定義後端介面**：註冊 buffer、非同步收送 comm tile、設定 / 等待 signal、
執行計算單位、回報能力。

**3. 能力分級**：
- 等級 0：算子級同步、經 host。
- 等級 1：tile 級 flag、經 host。
- 等級 2：PCIe P2P + kernel 內等待。
- 另一維度——延遲隱藏方式：**動態**（類 GPU，硬體排程）或**靜態**（類 NPU，需嚴格
  順序 + 軟體 pipeline + 時間餘裕）。

**4. A100 代理**：**不 enable peer access**，跨邊界一律顯式 D2H + H2D；用 NVLink 流量
計數器確認代理流量為零（否則量到的是 NVLink）。NPU 行為模擬：少量 persistent CTA +
顯式雙緩衝。

**5. 配置**：
- 同 NUMA：TP=2（GPU0–1），閘道 GPU1，代理 GPU2。
- 跨 NUMA：TP=4（GPU0–3），閘道 GPU3，代理 GPU4。

**先把等級 0 的完整流程跑通，再往上加。**

### 成功標準

- 1a、1b 各有 report + params.json 條目，1a 明確判定競爭位置。
- 等級 0 端到端跑通：AG+GEMM 一個 TP 域 + 一個代理裝置，結果 allclose，
  NVLink 計數器確認代理流量為零，時間有交錯 ECT 量測。

### 設計約束（來自 PHASE0_FINDINGS 第 3 節）

閘道 GPU 避免同時收送；生產者消費者跨 switch 組（0/1、2/3、4/5、6/7）；pinned buffer
綁來源 GPU 的 NUMA node；chunk ≥ 16 MiB；等級 0 建模用競爭態 6.46 GB/s，不用 nominal。

---

## 2. 目前成立（worker 維護）

（尚無。）

## 3. 撤回表（worker 維護）

| 撤回主張 | 出處 | 原因 | 替代 |
| --- | --- | --- | --- |
| （無） | | | |

## 4. 下一步（worker 維護）

1. 1a：改 `bidirectional_bandwidth_v2.py` 為 per-device 方向，複製到本 ws 的 `scripts/`
   （保留原檔不動），跑三組配置各 20 repeats。
2. 1b：`dual_path_v3.py` 加 overhead-only 模式，同樣複製到本 ws。

## 5. 產出到 common 的數字

（尚無。1a → `pcie.contention_locus`；1b → `staging.fixed_overhead_ms`。）
