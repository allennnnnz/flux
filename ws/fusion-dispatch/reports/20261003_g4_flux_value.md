Supersedes: （無）

# G4 之後：Flux 的優勢在哪裡、決策器比簡單規則多什麼、怎麼驗證決策器有用

- 日期：2026-10-03
- 性質：**事後分析**，只重算 G4 block 的既有量測，沒有重新量測或重新擬合。尚未經 auditor 審查。
- 數據來源：`results/g4_block_oracle/raw_*.csv`（`validate_block_v4.py`，4 層 block、ctx 1024，所有策略同一輪隨機交錯，
  每輪清 L2，200 輪，rank-max 中位數，`exclusive_guard` CLEAN，2026-10-03 量測）；
  預先登記的切法 `results/g4_block_tables/layout_g4.csv`、每層路徑 `results/g4_block_tables/table_g4_*.json`。
- 重算：`python3 scripts/analyze_g4_flux_value_v1.py` → `results/g4_block_oracle/flux_value_log.txt`、`flux_value_points.csv`。
- 圖：`reports/20261003_g4_strategy_compare.html`（三種策略：全部用 vLLM 預設 / 全部用 Flux / 決策器；瀏覽器直接開）。

起因：使用者看比較圖時問「怎麼看起來 Flux 沒有任何優勢」，接著問「如果決策器選的跟『decode 用 vLLM、prefill 用 Flux』一樣，
決策器有什麼用」，以及「怎麼驗證決策器有用」。本報告回答這三個問題。

## 1. Flux 的優勢在哪裡

圖上「全部用 Flux」在 decode 比 vLLM 預設慢 7–46%，看起來 Flux 沒有優勢。拆開後是兩件不同的事：

| 每個點拆成兩段 | 定義 | decode（M = 32–512） | prefill（M = 1024–4096） |
| --- | --- | --- | --- |
| 切法代價 | SP + 全 NCCL ÷ vLLM 預設 − 1（還沒用 Flux） | **+15% ～ +36%** | −3% ～ +7% |
| Flux 在 SP 內的效果 | min(SP 全 Flux, SP + Flux RS) ÷ SP + 全 NCCL − 1（同一種切法） | −7% ～ +16%（見下） | **−7% ～ −20%** |
| vLLM custom AR vs 一般 NCCL AR | vLLM 預設 ÷ TP + NCCL AR − 1 | −5% ～ −13% | −0.5% ～ +0.3% |

- **prefill：Flux 確實有優勢。** 同一種切法下比 NCCL 快 7–20%，比 vLLM 預設快 5–16%。
  決策器在 prefill 的 36 個「層 × M」決定中，**31 個用 Flux 路徑**（29 個融合 kernel，2 個 Flux AG + cuBLAS），
  只有 5 個用 NCCL。所以決策器在 prefill 省下的 10.9%，大部分就是 Flux 的功勞；圖上標「決策器」，把這件事藏起來了。
- **decode：輸的主要是切法，不是 Flux 本身。**
  - 用 Flux 就要把 TP + AllReduce 換成序列平行（SP），每個子層從 1 次 AllReduce 變成 AG + RS 2 次通訊。
    decode 資料量小，每次通訊幾乎都是固定開銷；vLLM 的 custom AllReduce 錄進 CUDA graph 後又特別快。
    光換切法就慢 15–36%。
  - 在 SP 內比，Flux 的效果依 M 而定：M = 32 慢 12–16%、M = 128 慢 2–5%、M = 256 有好有壞（−5.9% ～ +0.8%）、
    **M ≥ 384 快 2–7%**。小 M 能藏的通訊太少，Flux 的固定開銷蓋過好處。
  - decode 的「全部用 Flux」只能用 Flux RS，因為 AGKernel 不能 capture 進 CUDA graph（`CLAUDE.md` 陷阱表）。
- op 層級也一致（`results/g4_map/eval_g4_log.txt`）：GEMM+RS 永遠用 Flux 在 prefill regret 0.00%，
  永遠不用 Flux 則 22.05%；AG 永遠用融合 kernel 在 decode regret 40.83%。

### 為什麼比論文小 [推論]

論文摘要寫 8 卡上 prefill 最多 1.66×、decode 最多 1.30×（對 vLLM）。我們 block 層級只有 5–16%。可能原因：

1. 這台是 NVSwitch（600 GB/s），通訊佔比小，能藏的少；論文的大數字可能來自通訊慢的機器。
2. 對手變強：vLLM 0.8.5 有 custom AllReduce + CUDA graph；上表顯示它在 decode 比一般 NCCL AllReduce 快 5–13%。
   即使對手換成一般 NCCL AllReduce（TP + NCCL AR），decode 的 Flux 仍是 −0.6% ～ +29%（只有 Qwen 8 卡 M = 512 打平）。

如何驗證：對照論文各機器（PCIe / NVLink）分開列的加速圖；或在通訊慢的機器上重做本報告第 1 節的拆解（第 3 節的 E2 / E3）。

## 2. 簡單規則 vs 決策器

簡單規則 `rule_phase`：decode（CUDA graph，M ≤ 512）用 vLLM 預設，prefill（eager，M ≥ 1024）用 SP 全 Flux。

| 範圍 | 策略 | regret | 比 vLLM 預設省 | 最差的點 |
| --- | --- | --- | --- | --- |
| 全部 24 點 | vLLM 預設 | 8.71% | 0% | +19.7%（Qwen 8 卡 prefill M = 4096） |
| | **簡單規則** | **0.57%** | **7.5%** | **+7.9%（Qwen 8 卡 prefill M = 1024，比 vLLM 預設還慢）** |
| | 決策器（auto_g4） | 0.02% | 8.0% | +0.3% |
| prefill 9 點 | 簡單規則 | 0.80% | 10.2% | 同上 |
| | 決策器 | 0.03% | 10.9% | +0.3% |

- **在這台機器、這三個配置上，簡單規則幾乎跟決策器一樣好**（拿到 8.0% 中的 7.5%）。只部署在這裡的話，用簡單規則即可。
- 簡單規則唯一明顯出錯的點在 M = 1024，也就是 decode / prefill 交界。這段（M = 513–1023）block 層級沒有量過；
  vLLM 的混合批次常落在這裡。
- 決策器的價值不在這台機器上多贏的 0.5%，而在：
  1. **規則是量出來的，事先不知道。** op 層級的直覺規則：全部用 Flux 5.60%（最差 +197%）、
     全部不用 Flux 11.67%、「M > 512 才用 Flux」2.30%（最差 +50%），決策器 0.17%（`results/g4_map/eval_g4_log.txt`）。
  2. **換硬體時規則可能改變。** 例如 vLLM 在 PCIe 且 TP > 2 時會關掉 custom AllReduce（`reports/20261002_related_work.md`），
     decode 的「vLLM 預設」會變慢，切換點也會移動 **[推論]**。
  3. **找規則的成本。** 在新機器上把所有組合量一遍（op 層級標準答案）花 1062 秒；決策器只需單卡 GEMM 82 秒（7.7%），
     加多卡把關共 42%（`results/g4_map/eval_g4_log.txt`）。
- 結論：決策器是**找出規則的工具**。它值不值得，要看在別的連線 / 硬體上，簡單規則還成不成立。

## 3. 怎麼驗證決策器有用（提案，待使用者核准）

**「有用」的定義**：換一個環境，(a) 決策器重新校準（≤ 10 分鐘）後 block regret ≤ 2%；而且
(b) 至少一個環境中，在本機學到並凍結的簡單規則 regret ≥ 5%。
只有 (a) 成立、(b) 不成立時，誠實的結論是「決策器只是確認規則的工具」。

**凍結的對手規則**（本報告 commit 時即凍結，早於任何新環境的量測）：

| 名稱 | 規則 |
| --- | --- |
| R1 `rule_phase` | M ≤ 512 → vLLM 預設（TP + AllReduce）；M > 512 → SP 全 Flux（graph 中 AG 不能用 Flux 時改 `sp_rsflux`） |
| R2 `vllm_default` | 永遠 vLLM 預設 |
| R3 `all_flux` | 永遠 SP + Flux（graph 中用 `sp_rsflux`） |
| op 層級 | `on`（永遠融合）、`off`（永遠 NCCL + cuBLAS）、`thr512`，定義同 `scripts/eval_g4_v1.py` |

路徑在某環境不可用時（例如 E2 沒有 Flux），該規則退回 R2；這個退回規則也在此凍結。

**流程**（同 G3 / G4）：新環境校準 → 決策器的決定 commit / push → 量標準答案 → 比 regret（決策器 vs R1–R3）。
所有量測經 `exclusive_guard`，長工作放 tmux。

**可用的環境**：

| 編號 | 環境 | Flux 能跑嗎 | 可信度 | 備註 |
| --- | --- | --- | --- | --- |
| E0 | 模型推演：把通訊曲線換成 PCIe（Phase 0：staging ring 每卡 5.87 GB/s）或半速 NVLink，看決策表與 R1 的預測 regret | 模型內可以 | 只是預測 [推論] | 不用 GPU；用來決定 E1 / E2 值不值得做 |
| E1 | 本機背景塞車：另一支程式持續在 GPU 間搬資料，佔掉部分 NVLink | 能 | 中（也會搶 HBM 頻寬） | 要讓 `exclusive_guard` 認得這支程式；用 nsys 與錨點確認頻寬真的下降 |
| E2 | 本機 PCIe-only（原 G5）：`NCCL_P2P_DISABLE=1`，經 host 走 PCIe；vLLM custom AR 關閉（同 vLLM 在 PCIe 上的行為） | 不能（Flux 直接讀對端記憶體，本機一定走 NVLink） | 高，但只能驗證切法決策 | 需處理「路徑不可用」 |
| E3 | **另一台機器 gpu1**（使用者提供，EE325 叢集，經跳板機） | 視硬體 | 最高 | 硬體型號與連線方式待盤點 |

使用者已提供 gpu1，**E3 優先**。

**gpu1 目前狀態（2026-10-03）**：
- 從 css-host-158 用本機金鑰連跳板機被拒（`Permission denied (publickey,password)`）。
  使用者把本機公鑰加進 `authorized_keys` 後仍被拒，原因未查明。
- 改用 SSH agent forwarding：使用者在筆電的 ssh 設定中，對連 css-host-158 的 Host 加 `ForwardAgent yes` 後重連 VSCode。待使用者設定。
- 備案：用 VSCode Remote-SSH 直接在 gpu1 開 Claude Code session，照 `CLAUDE.md` 程序接手。
- 連上後的第一步只做唯讀盤點：GPU 型號 / 數量、NVLink 或 PCIe（`nvidia-smi topo -m`）、驅動 / CUDA、
  有無其他使用者、磁碟；再評估 clone、編譯 Flux、建 vLLM venv 的成本，回報後才開始實驗。

## 4. 對話中說法的更正

對話中先口頭回答過，以下以本報告的數字為準：
- 「decode 時 Flux 在 SP 內從 M ≥ 256 起快 2–7%」→ 正確是 **M ≥ 384 起快 2–7%**；M = 256 有好有壞（−5.9% ～ +0.8%）。
- 「vLLM custom AR 比一般 NCCL AR 快 7–12%」→ 正確是 **5–13%**。
