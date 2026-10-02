# common/measure

共用量測工具。實作 `CLAUDE.md` 5.1 的準則，讓各 workstream 不必各寫一套、也不會各錯一套。

## 現有

| 檔案 | 用途 | 來源 |
| --- | --- | --- |
| `clock_logger.py` | 背景記錄 SM clock，丟棄降頻輪次（準則 6） | 抽自 Phase 0 `flux_ag_gemm_baseline_v2.py` |
| `exclusive_guard.py` | 包住任何量測指令：啟動前確認 GPU 無其他程序、無其他使用者佔 CPU（否則中止），執行中每秒監控，結束標記 CLEAN / CONTAMINATED 與受干擾時間窗 | ws/fusion-dispatch，2026-10-02（使用者要求量測時確保獨佔） |

## 待抽出（第一個需要的 workstream 負責，抽出後在此登記）

| 工具 | 準則 | 目前所在 |
| --- | --- | --- |
| 交錯 ECT harness（同輪相鄰執行 A、B，各自 CUDA event，逐輪差值中位數，≥200 輪） | 5 | `experiments/2026-09-23-phase0-*/scripts/flux_ag_gemm_baseline_v2.py` 的 main loop |
| 逐 copy event 雙向計時 + 重疊窗口計算 | 7 | 同上 `bidirectional_bandwidth_v2.py` 的 `worker` / `bytes_in_window` |
| rank-max median 聚合 | 8 | 同上 `flux_comm_baseline.py` 的 `_bench` |
| nsys sqlite 解析（依 bytes 分類 copy、併發度 sweep line） | 9 | Phase 0 session 內的 ad-hoc 腳本，未存檔；需重寫 |
| 錨點對照（讀 `params.json`，偏差 > 20% 警告） | 2 | 尚無 |

抽出時保留原檔不動；原檔屬 Phase 0 檔案庫。
