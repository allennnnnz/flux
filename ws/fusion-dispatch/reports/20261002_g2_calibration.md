Supersedes: （無）

# G2：只用 4 分鐘的校準微基準建參數檔，對全部既有實測評估

日期：2026-10-02 · 作者：fusion-dispatch（boss 兼 worker）· 計劃：`reports/20261002_plan_general_dispatcher.md` 的 G2
前一階段：`reports/20261002_g1_predictor.md`。**尚未經 auditor 審查。**

---

## 0. 結論（白話）

1. **校準只要 4 分 2 秒**：從啟動到結束，含 6 個 process 的啟動時間。計劃目標是 ≤ 10 分鐘。
   - 量測經 `exclusive_guard.py`，6 段全部 CLEAN；
   - 形狀**刻意避開** 8 個評估層，也都不命中 Flux 登錄表。
2. **只用這些微基準擬合（完全不用決策表的點）**，預測全部 320 個既有實測點：

   | 參數檔 → 數據 | AG 只用預測器 | AG 混合把關 | RS 只用預測器 | RS 混合把關 |
   | --- | --- | --- | --- | --- |
   | gpu → gpu | **1.15%** | **0.12%** | **0.36%** | **0.09%** |
   | steady → steady | **1.74%** | **0.06%** | **0.10%** | **0.09%** |

   - 與 G1「用決策表元件擬合」的樣本內結果相當（AG 1.85–1.88%、RS 0.12–0.14%）；
   - 也就是說，**換成獨立的微基準後沒有變差**。
3. **錨點全部在 10% 內**，比 G1 更好。模型沒看過的 Phase 0 形狀，決策仍全對。
4. 「融合 kernel 重疊」的模擬結論不變：P0-4096 M=4096 藏住 3%（實測 7%），P0-8192 70%（實測 65%）。
5. **待改進**：
   - 探測量：事先定的門檻（ε = 校準殘差 p90 = 10.6–10.7%）要量 22–23% 的點。
     ε 掃描顯示 ε ≈ 5% 時只需 10–16%，regret 仍 ≤ 0.32%。G4 要用不看測試集的方式定 ε。
   - GemmRS 的擬合有參數抵換（gpu 版 η 碰到上限 1.0），RS 側路徑 A 的 MAPE 16.4%。
     決策仍準（0.36%），但預測的切換點偏晚（G-FC2：實測 136 起，預測 1024 起）。
6. **過程中踩到的坑**：同一 process 內先建立再銷毀一組 Flux op、再建第二組，會隨機卡死
   （8 個 rank 都卡在 `torch.cuda.synchronize`，GPU 空轉等信號）。
   - 單獨跑同一形狀不會卡；
   - 改成「一組形狀一個 process」後全部正常；
   - 與附錄 A.3「多個 M 同一 process 曾卡死」是同一類問題。

**下一步**：G3（新模型 Qwen2.5-72B、Llama-3-8B 與 TP=4 / 2 的 op 地圖，約 2–3 小時 GPU）。
量測前先把預測寫成檔案並提交，量完才比對，不得回頭調參。

---

## 1. 做了什麼

### 1.1 校準微基準（`scripts/calibrate_hw_v1.py`、`scripts/run_calibration_v1.sh`）

| 組 | 形狀 | M | 量的項目 | 輪數（gpu / steady） |
| --- | --- | --- | --- | --- |
| comm | hidden 6144（沒有評估層用這個） | 8 … 32768（98 KB – 403 MB，13 個） | NCCL AG / RS / AR、Flux AllGather、單筆 peer copy、7 筆串行 copy | 200 / 100 |
| ag | (n, K) = (2560, 5120)、(5120, 10240)、(1536, 16384) | 16、128、512、2048、8192 | cuBLAS、Flux gemm_only、Flux AllGather、融合 AGKernel | 200 / 50 |
| rs | (N, k) = (6144, 2560)、(10240, 1280) | 同上 | cuBLAS、Flux GemmRS | 200 / 50 |

- 協定：`dispatch_map_v2.run_mode`，與 E1 相同（同輪交錯、128 MB L2 flush、GPU 端對齊、rank-max、時脈過濾）。
- 正確性檢查全部通過：融合 / gemm_only / cuBLAS 對 fp32 參考；Flux AG 與 NCCL 結果逐位元相同；GemmRS 對 mm + all_reduce 參考。
- 保留輪數：gpu 幾乎全保留；steady 大 M 時因功耗上限降頻，丟掉部分輪次（最少 31 / 50）。
- 原始輸出：`results/g2_calibration/raw_*.csv`、`meta_*.json`、`guard_*.log`、`run_log.txt`。

### 1.2 擬合（`scripts/fit_calibration_v1.py`）

- 照 G1 的 `predictor.fit_profile`，只餵校準數據。
- 輸出：
  - `common/cost_model/hw_profiles/css-host-158_tp8_{gpu,steady}.json`（附來源、守衛結果、樣本數、校準殘差）；
  - 中位數表：`results/g2_calibration/summary_calibration.csv`；
  - 擬合紀錄：`fit_log.txt`。
- **命名**：計劃寫 `light / sustained`，改用 `gpu / steady`。
  G1 發現兩者的差別主要是量測方式（單次 vs 連發），不是時脈。
- **Flux AG 同步成本 α_sync = 0.0530 ms**：
  - 取 gpu 模式 M ≤ 64 時 c_flux_ag − 8/7 × 7 筆 copy 的中位數，與 G1 用 F2 數據得到的 0.0533 一致；
  - steady 模式連發的 copy 會跨呼叫重疊，算出負值，所以兩個參數檔都用 gpu 的值；
  - 這個數在 8 卡時不影響預測，只用於換卡數外推。
- **單筆 copy 的曲線**存進參數檔，供 G3 換卡數使用。

### 1.3 評估（`scripts/eval_profile_v1.py`）

- 不擬合任何東西，對全部 320 列（8 層 × 20 個 M × 2 模式）算誤差、決策、regret。
- 政策與 regret 定義同 G1。沒有隨機森林：校準集沒有 op 層級的路徑可訓練。
- ε = 參數檔中記錄的校準殘差 p90（事先定的規則）；另做 ε 掃描當診斷。

---

## 2. 結果

來源：`results/g2_calibration/eval_summary.csv`、`eval_log.txt`、`eps_sweep.csv`（2026-10-02）。

### 2.1 regret（%）

括號內為探測點數 / 80。

| 參數檔 → 數據 | 側 | on | off | thr512 | **pred** | hyb | **hyb+pcie** |
|---|---|---|---|---|---|---|---|
| gpu → gpu | AG | 6.61 | 13.85 | 2.61 | **1.15** | 0.56 (19) | **0.12** (23) |
| gpu → gpu | RS | 0.69 | 19.69 | 0.83 | **0.36** | 0.09 (13) | **0.09** (13) |
| steady → steady | AG | 5.92 | 10.72 | 2.11 | **1.74** | 0.88 (12) | **0.06** (18) |
| steady → steady | RS | 0.69 | 17.93 | 0.85 | **0.10** | 0.09 (19) | **0.09** (19) |
| gpu → steady | AG | 5.92 | 10.72 | 2.11 | 1.30 | 0.63 (19) | 0.29 (23) |
| gpu → steady | RS | 0.69 | 17.93 | 0.85 | 0.40 | 0.14 (13) | 0.14 (13) |
| steady → gpu | AG | 6.61 | 13.85 | 2.61 | 1.88 | 0.93 (12) | 0.06 (18) |
| steady → gpu | RS | 0.69 | 19.69 | 0.83 | 0.08 | 0.07 (19) | 0.07 (19) |

**只用預測器時最差的點**：
- AG：G-FC1 M=1024（45% / 34%），PCIe 調校 config 的斷崖，hyb+pcie 抓得到；
- RS：L-down M=256（19%）、G-FC2 M=72（15%）。

### 2.2 預測誤差（MAPE，全部 320 點都沒參與擬合）

| 參數檔 → 數據 | AG 各路徑 | AG 接近切換點 | RS 路徑 A / B | RS 接近切換點 |
| --- | --- | --- | --- | --- |
| gpu → gpu | 4.3–6.9% | 4.8% | 16.4% / 5.4% | 13.0% |
| steady → steady | 4.9–7.2% | 4.2% | 8.2% / 5.7% | 7.7% |

**元件**（gpu / steady）：

| 元件 | gpu | steady | 說明 |
| --- | --- | --- | --- |
| NCCL AG | 5.5% | 12.3% | steady 較高 |
| NCCL RS | 2.8% | 12.5% | steady 較高 |
| Flux AG | 2.2% | 2.3% | |
| cuBLAS | 7.1% | 7.2% | |
| Flux GEMM | 9.5% | 11.2% | |

steady 的 NCCL 誤差較高：校準只連發 16 次同一個 collective，評估數據則在同一輪中與 GEMM 交錯。**[推論]**

**fused vs off 判斷正確率**（排除平手）：AG 68–73 / 73–76，RS 52–53 / 53–57。

### 2.3 錨點

gpu 參數檔；`eval_log.txt` 開頭。

| 項目 | 模型 | 錨點 | 偏差 |
| --- | --- | --- | --- |
| Flux AG M=4096 | 0.462 ms | 0.468（Phase 0） | −1.3% |
| Flux AG 每卡 | 191 GB/s | 188 | +1.3% |
| NCCL AG M=4096 | 0.478 | 0.478 | 0.0% |
| Flux AG M=64 | 0.106 | 0.10 | +6.4% |
| NCCL AG M=64 | 0.048 | 0.051 | −6.0% |
| cuBLAS G-FC1 M=64 | 0.120 | 0.120 | +0.4% |
| 403 MB 每卡 | NCCL 203 / Flux 238 GB/s | all-to-all 217.66 | −6.5% / +9.4% |

### 2.4 唯一不是來自校準集的輸入：kernel 啟動時間

`t_k = 31.7 µs + 本地 shard bytes / 838 GB/s` 是從評估層的 nsys 擬合的（G1）。

敏感度（`eval_log.txt`「sensitivity」）：

| 改法 | AG regret（gpu） | AG regret（steady） |
| --- | --- | --- |
| 原值 | 1.15% | 1.74% |
| t_k0 × 0.7 / × 1.3 | 1.15% / 1.15% | 1.74% / 1.71% |
| 拿掉本地複製項 | 1.15% | 1.74% |
| 改成常數 0.040 ms | 1.15% | 1.63% |

決策不依賴它，所以沒有實質洩漏。

### 2.5 ε 掃描（hyb+pcie，診斷用）

| 參數檔 → 數據 | ε = 0 | ε = 5% | ε = 10%（≈ 事先規則） |
| --- | --- | --- | --- |
| gpu：AG / RS | 0.14% (7) / 0.37% (0) | 0.13% (11) / 0.32% (5) | 0.13% (22) / 0.09% (13) |
| steady：AG / RS | 0.81% (7) / 0.10% (0) | 0.06% (14) / 0.09% (11) | 0.06% (17) / 0.09% (18) |

ε = 0 時仍有 7 個探測，來自 PCIe 調校 config 的風險規則。

---

## 3. 問題與限制

1. **GemmRS 參數抵換**：
   - GemmRS 的 GEMM 參數與 scatter 頻寬一起擬合，gpu 版 η = 1.000（碰到上限）、bw = 0.73 TB/s；
   - 低 bw 實際上是把 epilogue 寫到遠端的成本算進記憶體項，與 scatter 項重複；
   - 因此 η 不可當成物理效率解讀；
   - **G4 建議**：GemmRS 的 GEMM 部分改用 Flux gemm_only 家族的參數（同一 CUTLASS kernel 家族），只擬合 α_rs 與 scatter 頻寬（2 個參數）。這是 G2 結果後才提的改法，G3 的預測先用現行版本。
2. **探測量**：事先規則下 22–23%，超過 15%。ε 要在不看測試集的前提下重定（例如用校準集內交叉驗證）。
3. **steady 參數檔的 NCCL 誤差**：12%，見 2.2。
4. **G1 的限制仍在**：
   - 融合比串行慢的點；
   - κ 依數據不同：校準得 0.100 / 0.125，G1 為 0–0.125；
   - 多 block 共用 SM 時低估重疊。
5. **卡死**：
   - 現象：同一 process 換一組 Flux op 後隨機卡死。原因未查明 **[推論：IPC / 對稱記憶體重用]**；
   - 證據：`results/g2_smoke/debug_hang2/log.txt`（堆疊顯示 8 個 rank 都在 synchronize）；
   - 第一次卡住的部分輸出在除錯時被刪除，只保留第二次的重現；
   - 對策：`run_calibration_v1.sh` 一組一個 process。建議加進 `CLAUDE.md` 陷阱表。

---

## 4. CLAUDE.md 5.3 自查

- **分母**：同 G1（每點最快路徑的實測）。
- **效率 > 100%**：無。GemmRS 的 η = 1.000 是邊界值，說明見第 3 節，不當成物理量發表。
- **錨點**：全部 < 10%。
- **重疊主張**：同 G1（E1 的 D − A、nsys V8 / E2、程式碼）。
- **輪數與時脈**：gpu 200 輪、steady 50–100 輪，時脈過濾，各組的眾數時脈記錄在 `meta_*.json`。
- **原始輸出**：全部提交；可重算：

  ```bash
  python3 ws/fusion-dispatch/scripts/fit_calibration_v1.py ws/fusion-dispatch/results/g2_calibration
  python3 ws/fusion-dispatch/scripts/eval_profile_v1.py
  ```
- **推論**：已標記。

---

## 5. 檔案

| 類別 | 路徑 |
| --- | --- |
| 腳本 | `scripts/calibrate_hw_v1.py`、`scripts/run_calibration_v1.sh`、`scripts/fit_calibration_v1.py`、`scripts/eval_profile_v1.py`；`scripts/eval_predictor_v1.py` 的 anchors / p0_crosscheck 加了標題參數，結果不變 |
| 參數檔 | `common/cost_model/hw_profiles/css-host-158_tp8_{gpu,steady}.json` |
| 結果 | `results/g2_calibration/`（原始、摘要、擬合紀錄、評估）；`results/g2_smoke/`（冒煙測試與卡死重現） |
