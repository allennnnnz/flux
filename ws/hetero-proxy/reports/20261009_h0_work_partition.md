Supersedes: （無）

# H0：一顆 PCIe 晶片放在 8×A100 NVLink 域旁邊，該扮演什麼角色

- 日期：2026-10-09。性質：**解析模型估算 [推論]**，不含新量測。尚未經 auditor 審查。
- 腳本 `scripts/h0_partition_v1.py`；輸出 `results/h0_partition/h0_partition.csv`、`h0_partition_log.txt`。
- 輸入：
  - Qwen2.5-32B（H=5120、64 層）；8×A100 TP8 域每層時間取 G4 block 實測最佳策略
    （`ws/fusion-dispatch/results/g4_block_oracle/`，4 層含 attention，decode 為 CUDA graph）；
  - PCIe：中轉最佳單向 22.2 GB/s（`docs/PHASE0_FINDINGS.md` 2.5）、同一張卡同時收送時 H2D 6.46 GB/s（2.4）；
    每次傳輸固定成本 20 µs（**假設**，待 1b 量測）；
  - 晶片算力 s（以一張 A100 為 1）；晶片跑一整層（不切分）約需 8 × t_gpu / s（把 TP 通訊也算成計算，對晶片偏悲觀）。

## 結果

| 角色 | PCIe 負擔 | 結論 |
| --- | --- | --- |
| **TP 成員**（任何份額） | 每層要收兩次輸入、送回兩次部分和 = 4·M·H·2 bytes；PCIe 時間是 TP8 域整層時間的 **0.8–1.6×（M=32）到 2.7–9.2×（M=4096）**，與晶片快慢無關 | **排除**：晶片那條路本身就比整層慢，每一層都被拖慢 |
| **Pipeline 一段**（平衡 k 層） | 每次通過只收送一次 M·H 啟動值；佔該段時間 0.6%（M=32）到 2.4%（M=4096） | **可行**；吞吐量 +s/8（A100 級 +12.5%、s=1/4 +3.1%、CPU 約 +0.4%）；平衡時單次通過延遲約 1.8–2.0× |
| Attention 卸載（decode，Lamina 式） | 每層 q、k、v 進、輸出回 = 佔該層時間 42%（M=32）到 74%（M=256） | 只有跨層 / 兩批交錯 pipeline 才可能；好處主要是 KV cache 容量，不是算力 |
| Draft 小模型（推測解碼） | 只有 token id / logits | PCIe 不是問題，也沒有大傳輸可重疊；好處來自 draft 與目標模型平行 |

## 結論與對實作的影響

1. **第一個真實角色選 pipeline 段**。Flux 式的 chunk 重疊做在段的邊界：下一段的第一個 GEMM 等每個 row chunk 到就開算；上一段的最後一個 GEMM 算完一個 chunk 就開始送。
2. **單顆晶片的效益上限是 s/8**。這套系統的價值在「能接任何加速器、讓它不拖慢 NVLink 域」，而不是單顆晶片帶來大加速。
3. Pipeline 平衡時 decode 延遲約翻倍 → 適合吞吐量導向的服務（多個請求交錯填滿 pipeline）；低延遲場景要讓晶片段少於平衡值。
4. 實作（I2 / I3）的示範因此設計成「GPU 段 → 晶片段 → GPU 段」的雙向 chunk 串流，正是 pipeline 邊界的通訊型態。

## 如何驗證

- 每次傳輸固定成本以 1b 實測取代 20 µs 假設；
- I3 / I4 實測邊界傳輸與重疊效益，對照本模型；
- 晶片到手後以實際 s 重算平衡層數與吞吐量。
