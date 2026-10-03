Supersedes: （無）

# G4：決策器 v2（模型 + 實測單卡 GEMM + 少量多卡把關），在全新情境上驗證

日期：2026-10-03 · 作者：fusion-dispatch（boss 兼 worker）· 做法由使用者核准（D-009）
前幾階段：`reports/20261002_g1_predictor.md`、`20261002_g2_calibration.md`、`20261002_g3_unseen.md`。**尚未經 auditor 審查。**

---

## 0. 結論（白話）

1. **做法**：
   - 通訊與「邊傳邊算」的重疊用模型算；
   - 矩陣乘法（GEMM）改成實測。只在單卡上跑，每個模型幾十秒；
   - 預測差距小的地方再做少量多卡實測。
2. **全新測試集**（G1–G3 都沒用過）：Qwen2.5-32B 在 8 卡 / 4 卡、Llama-3-8B 在 4 卡；M 也全是沒用過的值。
   每一步的決策都在量標準答案**之前** commit / push。
3. **op 層級**（192 點，選錯路徑多花的時間比例）：

   | 方法 | regret |
   | --- | --- |
   | **模型 + 實測 GEMM** | **0.17%**（AG 0.04%、RS 0.41%） |
   | 再加多卡把關 | 0.11% |
   | G3 方法（只用模型） | 1.20% |
   | 固定門檻（M ≤ 512 不開） | 2.30% |
   | 永遠開 Flux | 5.60% |

4. **block 層級**（整個 transformer block，含「要用 vLLM 預設的 TP + AllReduce，還是序列平行」的切法選擇；24 個情境）：

   | 方法 | block regret | 比 vLLM 預設省 |
   | --- | --- | --- |
   | **自動選切法 + G4 表** | **0.02%** | **8.0%**（prefill 10.9%、decode 0%） |
   | 永遠用 vLLM 預設 | 8.71% | 0% |
   | 永遠序列平行 + G4 表 | 4.62% | 3.8% |
   | 永遠序列平行 + G3 表 | 5.95% | 2.5% |

   - decode 正確選 vLLM 預設，序列平行在 decode 慢 15–22%；
   - prefill 選序列平行 + G4 表，比 vLLM 快 6.4–13%。
5. **計劃的成功標準**：

   | 標準 | 結果 |
   | --- | --- |
   | block regret ≤ 2% | 0.02%，**達成** |
   | 校準 ≤ 10 分鐘 | 每種卡數 2.5–4 分鐘 + block 校準約 45 秒，**達成** |
   | 接近切換點的預測誤差 ≤ 10% | AG 1.9%、RS 5.8%，**達成** |
   | 探測 ≤ 15% 的點 | **未達成**：事先規則下 op 38%、block 表 33%；3% 變體 16% |

   不做多卡探測時已有 0.17%（op），所以多卡探測的價值很小，之後可以大幅減少。
6. **成本**：
   - 量整張決策表：這個測試集 17.7 分鐘；
   - 新方法需要的單卡 GEMM 實測：82 秒（7.7%）；
   - 加上多卡探測：367 秒（另加 35%）。
7. **新的坑**：
   - steady 模式量極小 kernel 時，量到的是 CPU 發 kernel 的速度，數值取決於同一輪的其他項目；
   - vLLM custom all-reduce 在 eager 模式比 graph 模式慢（要先複製到註冊緩衝區）；
   - 我自己的程式錯誤一個（已修，紀錄保留）。

   都已寫進計劃附錄 A.3 / `CLAUDE.md`。

---

## 1. 做了什麼（每一步都在下一步量測前 commit）

| 步驟 | 內容 | 時間 | commit |
| --- | --- | --- | --- |
| 1. 校準 | `run_calibration_v1.sh --rs_gemm_only`，8 / 4 / 2 卡；多量 flux.GemmOnly，供 GemmRS 模型用 | 4:07 / 3:10 / 2:38 | `49534de` |
| 2. 單卡 GEMM | `probe_gemm_v1.py`：各層 × 各 M 的 cuBLAS、AGKernel.gemm_only、GemmOnly（gpu 50 輪 / steady 20 輪） | 82 秒 | `49534de` |
| 3. 預測 | `predict_g4_v1.py predict`：192 點 × 三種方法；探測清單 73 點（事先規則：差距 < max(3%, 校準殘差 p90)） | — | `49534de` |
| 4. 多卡探測 | `run_g4_probes_v1.py`：24 個短 run（30 輪），73 點中改變 15 個選擇 | 367 秒 | `8fc288e` |
| 5. 標準答案 | `run_g4_map_v1.sh`：12 組 × 8 個 M × 2 模式，200 輪 | 17.7 分鐘 | `486ef8e` |
| 6. block 校準 | `calibrate_block_v1.py`（vLLM 環境）：vLLM all-reduce 曲線 + RMSNorm / add；block 用的 M 量單卡 GEMM | 2:40 | `f590a3b` |
| 7. block 表 | `build_g4_block_v1.py tables`：sp_g4 / sp_g3 表；32 / 96 條目要探測 | — | `f590a3b` |
| 8. 表探測 → 切法預測 | 17 個短 run；用 `predictor/block.py` 預測每個 M 的切法；9 個接近要探測 | 215 秒 | `0d6be62` |
| 9. 切法探測 | 9 點全部同意模型 | 約 2 分鐘 | `bb42d7b` |
| 10. block 標準答案 | `validate_block_v4.py`：全部策略 200 輪；decode 用 CUDA graph、prefill 用 eager；4 個 block | 293 秒 | 本報告 |

- 所有 GPU 量測都經 `exclusive_guard.py`，全部 CLEAN；
- 正確性檢查全過。block 的每個序列平行策略輸出都和參考一致，相對誤差約 1.3%，門檻 5%。

**預測器的變更**（`common/cost_model/predictor/`）：
- `predict_ag` / `predict_rs` 可吃實測 GEMM；
- GemmRS 改為 2 參數，GEMM 部分用實測 GemmOnly（G2 已在舊數據上驗證比 6 參數好）；
- `block.py`：切法模型。

TP + AllReduce 與序列平行的差 = 4 個 cuBLAS（實測）+ 2 次 vLLM all-reduce + M 列的 norm / add，減去 4 條序列平行路徑 + M/W 列的 norm / add。attention 與 SiLU 在兩種切法中相同，互相抵消。

---

## 2. op 層級結果

來源：`results/g4_map/eval_g4_log.txt`、`eval_g4_summary.csv`。

| 組 | on | off | thr512 | G3 方法 | G4 只用模型 | **g4** | g4 + 探測 | g4 + 探測（3%） |
|---|---|---|---|---|---|---|---|---|
| 全部（192） | 5.60 | 11.68 | 2.31 | 1.20 | 1.17 | **0.17** | 0.11 | 0.14 |
| AG（96） | 8.30 | 6.37 | 2.86 | 1.55 | 1.55 | **0.04** | 0.03 | 0.03 |
| RS（96） | 0.96 | 20.80 | 1.35 | 0.60 | 0.51 | **0.41** | 0.25 | 0.32 |
| 8 卡 Qwen2.5-32B | 8.95 | 13.60 | 2.46 | 0.33 | 0.32 | **0.24** | 0.09 | 0.16 |
| 4 卡 Qwen2.5-32B | 4.36 | 9.30 | 3.10 | 1.84 | 1.82 | **0.15** | 0.08 | 0.13 |
| 4 卡 Llama-3-8B | 4.05 | 13.86 | 0.66 | 1.00 | 0.94 | **0.15** | 0.20 | 0.12 |
| gpu 模式 | 5.83 | 10.83 | 2.29 | 0.57 | 0.57 | **0.08** | 0.03 | 0.04 |
| steady 模式 | 5.38 | 12.52 | 2.32 | 1.82 | 1.76 | **0.27** | 0.19 | 0.23 |

單位 %。

**g4 的路徑時間誤差**：

| 側 | 路徑 | MAPE |
| --- | --- | --- |
| AG | A / B / C / D | 2.5 / 4.4 / 1.5 / 1.8% |
| AG | 接近切換點的 47 點 | 1.9% |
| RS | A / B | 8.5 / 5.2% |
| RS | 接近切換點的 38 點 | 5.8% |

**最差的點**：RS 的 L8-O（4 卡，每卡 k = 1024）在 M = 24 / 96、steady 模式，33–35%。
原因見第 4 節第 1 點，同一 GEMM 在探測與地圖中可差 3 倍。

---

## 3. block 層級結果

來源：`results/g4_block_oracle/eval_g4_block_log.txt`。

| 組 | vLLM 預設 | SP + NCCL | SP + Flux | SP + G3 表 | SP + G4 表 | **自動選切法** |
|---|---|---|---|---|---|---|
| 全部 24 點：regret | 8.71 | 15.45 | 0.80（只有 prefill） | 5.95 | 4.62 | **0.02** |
| 全部 24 點：比 vLLM 省 | 0 | −6.2 | 10.2 | 2.5 | 3.8 | **8.0** |
| decode（graph，15 點）regret | 0.00 | 21.71 | — | 17.61 | 15.90 | **0.00** |
| prefill（eager，9 點）regret | 12.25 | 12.91 | 0.80 | 1.21 | **0.03** | **0.03** |
| prefill 比 vLLM 省 | 0 | −0.6 | 10.2 | 9.8 | 10.9 | **10.9** |

單位 %。

- 各組最差的點只差 0.3%（Llama-3-8B 4 卡 prefill M=1024：實測最快是永遠開 Flux 3.296，自動選 3.308）。
- 不做切法探測（只靠 block 模型）結果相同：0.02%。
- 每組比 vLLM 省：Llama-3-8B 4 卡 9.1%、Qwen2.5-32B 4 卡 6.4%、8 卡 9.5%（全部 decode + prefill 點合計）。

---

## 4. 新發現的坑與限制

1. **steady 模式量極小 kernel**：
   - 連發 16 次時量到的是 CPU 發 kernel 的速度；
   - 同一輪有慢的通訊時，CPU 先跑到前面，量到的是純 GPU 時間；
   - 例：Qwen2.5-32B 8 卡 O 投影 M=96 的 cuBLAS，單卡探測 0.034 ms、地圖 0.012 ms；
   - gpu 模式沒有這個問題（偏差 ≤ 6%）；
   - RS 側最差的點都在這裡。**影響**：eager 小 M 的決策要特別小心（F1 已發現 eager 小 M decode 受 CPU launch 限制）。
2. **vLLM custom all-reduce**：eager 模式要先複製到註冊緩衝區，比 graph 模式慢。
   - 校準在 eager 量，所以 decode（graph）時模型高估 TP + AllReduce 的時間，偏向選序列平行；
   - 這次 decode 的差距夠大（vLLM 預設快 15–22%），決策不受影響；
   - 若要更準，需在 graph 中校準 all-reduce。
3. **我的程式錯誤**：`validate_block_v4.py` 的策略子集選項在 graph 模式仍嘗試移除 `sp_flux`。
   - decode 的切法探測第一次失敗；
   - 已修正；失敗紀錄改名保留（`*.failed_policyfilter_bug`）。
4. **探測量**：事先規則超過 15%，但探測帶來的改善很小（op 0.17 → 0.11%）。
   - 建議：G5 / F4 只在「預測差距 < 3%」或有風險訊號時探測（3% 變體：16% 的點，0.14%）。
5. **範圍限制**：
   - block 驗證只有 3 組情境、4 個 block、隨機權重，attention 用 PyTorch SDPA；
   - 還沒在真正的 vLLM 端到端（F4）上跑；
   - V4 曾發現長 graph 中 c10d NCCL 會超線性變慢，這次 decode 選 vLLM 預設，不受影響。
6. **PCIe 調校 config 規則**：G4 的點都沒命中登錄表，仍未測到。

---

## 5. CLAUDE.md 5.3 自查

- **預先登記**：每一步的決策都在下一步量測前 push，commit 見第 1 節表格。
- **分母**：
  - op：每點最快路徑的實測；
  - block：每點最快的可部署策略實測（不含 torch NCCL 的 tp_ar 橋接策略）。
- **自我一致**：單卡 GEMM 探測 vs 地圖內同一 GEMM，gpu 模式中位偏差 0–0.2%（最大 4–20%），steady 中位 +2–4%（極小 kernel 最大 188%，見第 4 節第 1 點），已列為發現。
- **守衛**：全部 CLEAN（校準 18 段、GEMM 6 段、探測 41 段、地圖 12 段、block 校準 2 段、切法探測 4 段、block 正式 6 段）。
- **原始輸出**：
  - `results/g4_calibration_tp{8,4,2}/`、`g4_gemm/`、`g4_predictions/`、`g4_probes/`、`g4_map/`；
  - `g4_block_cal_tp{8,4}/`、`g4_block_gemm/`、`g4_block_tables/`、`g4_block_probes/`、`g4_block_layout_probes/`、`g4_block_oracle/`、`g4_block_smoke/`。
- **可重算**：`python3 ws/fusion-dispatch/scripts/eval_g4_v1.py`、`eval_g4_block_v1.py`。

---

## 6. 下一步

- **F4**（vLLM 端到端）可以開始：`build_table_v2.py` 產生 `dispatcher_v1` 格式的表，切法用 `predictor/block.py`；
  之前延後 F4 的原因（等泛用版）已解除。
- **G5**：PCIe 代理（NCCL 關 P2P、Flux 路徑不可用），與 hetero-proxy 協調。
- **G6**：總報告 + auditor 審查 → DECISIONS。
