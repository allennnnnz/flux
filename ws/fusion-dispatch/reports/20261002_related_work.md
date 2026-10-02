Supersedes: （無）

# 相關文獻：融合開關 / TP 切法的決策，以及效能預測模型

日期：2026-10-02 · 作者：fusion-dispatch（boss session）· 計劃 G0 的產出（`reports/20261002_plan_general_dispatcher.md`）

## 方法與查證等級

兩次網路搜尋，查證約 100 篇論文與系統原始碼。

| 標記 | 意義 |
| --- | --- |
| [V] | 打開過 arXiv 摘要 / HTML、會議頁面或原始碼 |
| [V-s] | 只透過搜尋片段或第二手頁面確認（原頁面被擋） |
| [U] | 未能確認 |

- 細部數字（門檻、誤差）多半經由「摘要式網頁讀取」取得，**不是**逐頁讀 PDF。寫進論文前要回原文核對。
- **各論文自報的誤差不能直接比較**：
  - NeuSight 重測 Habitat 得 220.9%，Habitat 自報 11.8%；
  - PipeWeave 重測 NeuSight 得 43.5%，NeuSight 自報 9.7%。
- 偏好有報告 **regret / 選擇正確率** 的工作。

---

## 1. 計算-通訊融合 / 重疊系統，以及它們怎麼決定

| 工作 | 出處 | 怎麼決定 | 處理「何時不該融合」？ |
| --- | --- | --- | --- |
| **Flux**，Chang et al. [V] | arXiv 2406.06858 | 對模板參數、傳輸方式（pull / push）、tiling 自動調參 | 承認極小 m 時可能比不融合慢，**沒有退回機制** |
| **TileLink**，Zheng et al. [V] | MLSys'25，2503.20313 | 由程式設計者選 tile / 順序 | 未討論 |
| **Triton-distributed**，Zheng et al. [V] | 2504.19442 | 分散式 autotuner：各 rank 實測後全域同步選最佳 | 只有協定隨大小選擇，沒有融合開關 |
| **FlashOverlap**，Hong et al. [V] | EuroSys'26，2504.19519 | **預測式搜尋**：profiled GEMM（扣掉通訊佔的 SM）+ 實測頻寬曲線內插；預測誤差 3.41%（RTX 4090 PCIe）/ 3.44%（A800），選擇達最佳的 99% 以上 | 部分處理：切太碎時頻寬利用率差。**最接近我們的預測式做法，且涵蓋 PCIe** |
| **CoCoNet**，Jangda et al. [V] | ASPLOS'22，2105.05720 | 窮舉：產生全部排程並實測取最快 | 隱含處理（不融合也在搜尋空間內） |
| **T3**，Pati et al. [V] | ASPLOS'24，2401.16677 | 硬體 / 軟體共同設計，靜態機制 | 否 |
| **FiCCO**，Pal et al. [V] | 2512.10236（MI300X） | 啟發式：op-to-byte 比、MK+NK+MN 流量、M vs K，在 4 種重疊排程中選 | **否**（序列執行不在選項中）。未見情境選對 81%，選錯時損失約 14% |
| **Analytical Resource Management for MoE overlap**，Liu et al. [V] | SC'26 workshop，2609.07536 | 在 Flux / COMET（A100）內用 wave 量化的解析模型選通訊 CTA 數 | 只有單一旋鈕。**對 oracle 平均 regret 3.22%，是最強的「解析式 + regret 評估」前例** |
| **Comet**，Zhang et al. [V] | MLSys'25，2502.19811 | **離線查表**：部署前 profile 每種設定 | 否 |
| **Syncopate** [V 摘要] | OSDI'26，2601.20595 | chunk 排程 | 未說明 |
| **Entwine** [V 摘要] | 2609.11562 | 啟發式 | 否 |
| **mKernel** [V 摘要] | 2609.13585 | GPU 上的控制器執行期調 SM 分配 | 否 |
| **ParallelKittens** [V 摘要] | 2511.13940 | 原語分析 | 否 |
| **Google decomposition**，Wang et al. [V-s] | ASPLOS'23 | [U] | [U] |
| **AMD fused collectives**，Punniyamurthy et al. [V] | 2305.06942 | 無選擇模型 | 否 |

### 1.1 以 batch / sequence 切分做重疊（推論服務）

| 工作 | 出處 | 怎麼決定 | 備註 |
| --- | --- | --- | --- |
| **TokenWeave**，Gond et al. [V] | MLSys'26，2505.11329 | **人工固定門檻**：H100 上 dense 1K token、Mixtral 4K | 小批時不切分，只用融合的 AllReduce–RMSNorm。**最接近我們問題的服務端類比，但門檻是手設的** |
| **NanoFlow**，Zhu et al. [V] | OSDI'25，2408.12757 | MILP 加上 profiling 的干擾搜尋 | 假設高吞吐場景 |
| **ISO**，Xiao & Su [V] | 2409.11155 | [U] | 只用於 prefill |
| **Domino**，Wang et al. [V] | 2409.15241 | grid search | 小批收益低 |
| **Liger**，Du et al. [V 摘要] | PPoPP'24 | 「競爭係數」預估干擾 | — |
| **Centauri**，Chen et al. [V 摘要] | ASPLOS'24 | [U] | — |
| **PyTorch Async-TP** [V] | PyTorch 論壇 2024-09 | torch.compile 改寫，**無大小門檻** | 承認小問題（推論）效果差 |
| **Megatron / TE userbuffers** [V] | NVIDIA 文件 | 依硬體與形狀**手寫 preset** | — |
| **Kraken**、**Ladder Residual** [V] | NeurIPS'24、ICML'25 | 修改模型架構，非執行期決策 | — |
| 重疊特性研究，Lee et al. [V] | 2507.03114 | — | **重疊時計算平均變慢 18.9%（最多 40%）→ 成本模型需要競爭項** |

## 2. 生產框架怎麼在執行期依大小選路徑

| 框架 | 決策方式（原始碼） |
| --- | --- |
| **TensorRT-LLM AllReduceStrategy AUTO** [V] | `selectStrategyLookUpTable`：**離線表**，以 SM90 / SM100、log2(TP)、fusion op、log2(hidden)、log2(tokens ≤ 16384) 為索引；超出範圍退回 NCCL。**等於工業化的查表，也就是我們目前的做法** |
| **vLLM custom all-reduce** [V] | `max_size` 8 MiB 門檻；一階段 / 兩階段：`(ws≤4 且 <512KB) 或 (ws≤8 且 <256KB)`；PCIe 且 TP>2 時停用 |
| **vLLM SequenceParallelismPass + async TP** [V] | 正是「切法 + 融合」：AllReduce+RMSNorm 改寫成 RS+RMSNorm+AG，再融合 AG+GEMM / GEMM+RS。只在 `tokens ≥ sp_min_token_num` 時啟用，常數來自每種 compute capability 的表。註解寫著「小批時切分與收集的開銷大於收益」。**手寫的解析門檻** |
| **DeepSpeed-Inference** [V] | 未見決策機制 |

## 3. 依 batch 大小選 TP / SP 切法或平行方式

| 工作 | 出處 | 決策方式 |
| --- | --- | --- |
| **Shift Parallelism / Arctic Inference** [V] | 2509.16495、2507.11830 | 小批全 TP、大批 SP×TP，以 `n > threshold` 切換（門檻怎麼來未說明）。**與我們的切法決策直接相關** |
| **Seesaw** [V] | MLSys'25，2503.06433 | prefill / decode 不同平行方式，以效能模型離線選定 |
| **Nitsum** [V 摘要] | 2605.05467 | TP 度數當執行期旋鈕，方法 [U] |
| **Learning to Shard**（強化學習）[V] | 2509.00217 | 不含融合選擇 |
| **APEX** [V] | 2411.17651 | 模擬器搜尋 |

## 4. 集體通訊演算法 / 協定選擇

| 工作 | 決策方式 |
| --- | --- |
| **NCCL 內部 tuning** [V，原始碼 v2.21.5 `tuning.cc`] | 每種（演算法 × 協定）用 α-β：`time = lat·latCount + nBytes/(1000·bw)`，取最小。延遲 = baseLat + hwLat；頻寬含協定折減（LL ×0.5 等）。**不建模計算競爭** |
| **NCCL tuner plugin** [V] | v3 可直接修改 `collCostTable[algo][proto]`：**可插入學習模型的掛勾** |
| **Demystifying NCCL**，Hu et al. [V] | 2507.04786。每跳延遲約 6 µs（Simple）/ 1 µs（LL）/ 2 µs（LL128）；作者認為封閉式建模不實際 → α 強烈依賴協定 |
| **AutoCCL** [V] | NSDI'25。線上調 6 個參數，考慮計算干擾 |
| **Lagom** [V] | 2602.20656。統一成本模型加優先序搜尋，**明確建模重疊** |
| **TACCL**、**TE-CCL**、**SCCL**、**MSCCLang / MSCCL++**、**Blink** [V] | 演算法合成：MILP / SMT / 流量最佳化，用 α-β 鏈路模型 |
| **ACCLAiM** [V-s] | CLUSTER'22。ML 選 MPI 演算法 |
| **NCCLbpf** [V] | 2603.11438。eBPF 政策，每次決策 80–130 ns |
| **Synchronization tax**，Devraj et al. [V] | 2608.22503。加入同步項的 Hockney 模型，8 GPU 時同步可佔 collective 的 50% 以上 → **小 M 固定成本要有同步項** |
| **ConCCL**，Agrawal et al. [V] | ISPASS'25，2412.14335。SM 型通訊與計算並行只達理想加速的 21%，改用 DMA / copy engine 達 72% → **支持 SM 路徑 vs copy engine 分開建模** |
| **NVRAR** [V] | 2511.09557。小訊息 all-reduce 比 NCCL 快 1.9–3.6× |

**未找到**任何把 NCCL 延遲建模為 SM 時脈函數，或對比 SM 型與 copy engine 型傳輸時脈敏感度的論文。**本 ws 的 V6c 觀察可能是新的。**

## 5. 效能預測模型

### 5.1 kernel / 算子，跨硬體

| 工作 | 出處 | 方法 | 誤差 / 推廣 |
| --- | --- | --- | --- |
| **KernelSight-LM**，Yao et al. [V] | 2606.28565 | roofline × 學習的效率 η（無因次特徵：算術強度、與 ridge 的距離、每 SM 工作量、occupancy、wave 數、L2）；通訊用 α0 + 2(P−1)/P·S/β | 未見 GB200：只用規格（Tier A）12.1%，**加一次微基準掃描（Tier B）3.8%**；端到端 TPOT 12.8% / 6.2%。**「模型 + 少量實測」最清楚的證據** |
| **TileSight**，Mo et al. [V] | 2607.22432 | 純解析，tile 為中心：tile 內管線、tile 間 L2、跨裝置 α-β（含 AG+GEMM 等融合 kernel） | GEMM 12.35% MAPE；融合分散式 kernel 16.18% wMAPE（≤ 32 GPU）；保留前 5% 預測排程即達最佳的 99.66%；校準只要幾秒的探測 |
| **NeuSight**，Lee et al. [V] | ASPLOS'25，2407.13853 | tile 化，waves = ⌈tiles/SM⌉，MLP 只預測 utilization = α − β/waves，以 roofline 為上限 | 未見 H100 / L4：推論 9.7%；4-GPU 7.7% |
| **PipeWeave**，Zhang et al. [V] | ISCA'26，2601.14910 | 模擬 SM 排程 + 每種 kernel 一個 MLP 預測效率 | 未見 GPU 11.4%；通訊只用隨機森林，較粗 |
| **Habitat**，Yu et al. [V] | ATC'21，2102.00527 | wave scaling（頻寬、每 wave 區塊數、**時脈**比例）+ MLP | 11.8%（自報）；**唯一把時脈比放進縮放律的** |
| **tritonBLAS**，Swann et al. [V] | 2512.04226 | 第一原理 GEMM 延遲模型 + 少量微基準常數 | MI300X 選擇效率 94.7%，零 autotune |
| **LLMCompass** [V] | ISCA'24 | 階層式 tile 模擬 | 算子約 10.4%、推論 4.1% |
| **AMALI** [V-s] | ISCA'25 | 解析 warp / pipe 模型 | A100 23.59% |
| **GenZ** [V] | 2406.01698 | 解析 | 5.82% |
| 其他 | — | Li / Sun / Jog（MICRO'23）、DeLTA、ISAAC、Stream-K（wave 量化造成 GEMM 大幅波動）、nvMatmulHeuristics、TVM / Ansor / TenSet / CDMPP（**跨架構遷移效果差**）、HELP、LitePred | — |

### 5.2 模擬器 / 規劃器（預測各設定的時間並選平行方式）

| 工作 | 出處 | 方法 | 誤差 |
| --- | --- | --- | --- |
| **Vidur** [V] | MLSys'24，2405.05465 | profile 算子，隨機森林內插（因多項式抓不到 tile / wave 量化）；collective 獨立 profile | < 9%；需要在目標硬體上 profile |
| **Echo** [V] | 2412.12487 | 白箱 NCCL 模型 + XGBoost 預測重疊變慢 | 通訊 7–14%；重疊變慢只有 43–61% 落在 5% 內 |
| 其他 | — | Splitwise（< 3%）、DistServe（< 2%）、APEX、AIConfigurator（無法外推到未量測 GPU）、Calculon、vTrain、Proteus（3%，保持策略排序）、Lumos、Alpa、FlexFlow（30% 內，但保持排序）、LLMServingSim2.0、SimAI、ATLAHS、ASTRA-sim 2.0 | — |

### 5.3 線上 / 自適應選擇

- **AutoCCL**：線上，藏在前幾次迭代中；
- **Kernel Tuner BO**、**OpenTuner**（bandit 風格）、**DynaTune**（UCB）、**Seer**（CGO'24，決策樹選 kernel）；
- **TileSight**、**KernelSight-LM Tier B**、**nvMatmulHeuristics**：「模型取 top-k 再實測」。

---

## 6. 缺口與我們的定位

1. **沒有系統同時決定融合開關與 TP 切法。**
   - 選項：TP+AllReduce（custom / NCCL）、SP 不融合、SP + 融合 AG+GEMM / GEMM+RS。
   - vLLM 最接近，但拆成三個彼此獨立的手設門檻。
2. **既有決策器用的四種方法都不泛用**：
   - 查表：TensorRT-LLM、Comet、Flux autotune、我們的 v1；
   - 固定門檻：TokenWeave、Shift、vLLM；
   - 窮舉 / 線上搜尋：CoCoNet、Triton-distributed、AutoCCL；
   - 解析模型但只處理較窄的旋鈕：NCCL、FlashOverlap、2609.07536、FiCCO、NanoFlow。
3. **decode 區被承認但沒被建模**：Flux、Async-TP、Domino 都說小 M 會輸，沒有人預測切換點。
4. **很少報告對實測 oracle 的 regret**：只有 2609.07536（3.22%）與 FiCCO（81% / 14%）。
5. **沒有展示推廣到新 TP 卡數或新硬體**。PCIe 只有 FlashOverlap / ISO（RTX 4090）與框架的退回機制。

**定位**：涵蓋整個空間（融合開關 + 切法，跨服務的 M，含 decode）的泛用決策器，以 regret 對實測 oracle 評估，展示對未見形狀 / TP / 時脈 / PCIe 代理的推廣。

## 7. 對設計的直接影響（已反映在計劃第 0、1 節）

| 文獻依據 | 設計 |
| --- | --- |
| 2609.07536、TileSight、NeuSight | GEMM 用 tile / wave 量化 + 小 M 的讀權重下限 |
| NCCL tuning.cc、Synchronization tax、Demystifying NCCL | 通訊每條路徑各自的 α-β，α 依協定分段，含同步項 |
| ConCCL + 本 ws V6c | SM 型通訊（NCCL、custom AR）的 α 隨 SM 時脈縮放；copy engine 型（Flux AG）不隨時脈 |
| 2507.03114、Liger、AutoCCL | 融合路徑加競爭項 |
| FlashOverlap、TileSight top-k、KernelSight-LM Tier B | 混合式：模型篩到前 2–3 名，只在預測差距 < 模型誤差時探測 |
| KernelSight-LM | 校準：每台機器幾分鐘的微基準 |
| 2609.07536、FiCCO | 評估：對 oracle 的 regret、選擇正確率、選錯時的損失 |
| vLLM 的 per-compile-range 決策、NCCL tuner cost table | 可能的部署掛勾 |

## 8. 寫論文前要回原文核對的項目

- FlashOverlap 的預測器細節與 PCIe 結果；
- KernelSight-LM Tier A / B 的設定；
- TileSight 的融合 kernel 評估；
- 2609.07536 的 regret 定義；
- TokenWeave / Shift Parallelism 的門檻來源；
- vLLM `sp_min_token_num` 常數表（看它的原始碼版本）。
