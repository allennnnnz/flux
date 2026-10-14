# ws/hetero-proxy — JOURNAL

Append-only。每條：日期、角色、做了什麼、卡在哪、留給下個 session 的話。
需要 boss 裁決的事項標 **「需 boss 裁決」**。

---

## 2026-09-23 · boss · 建立

- 建立 workstream，寫入 STATUS 第 1 節。
- 1a 的三組配置與判讀邏輯已寫在 STATUS；`PHASE0_FINDINGS.md` 2.4 標為推論的那句
  就是 1a 要驗證的。
- 留給 worker：Phase 0 的兩個腳本要複製進本 ws 再改，不要改 `experiments/` 底下的原檔。

## 2026-10-06 · boss 兼 worker · 教授新方向；文獻與開源調查

- **教授新方向**：PCIe 也要能用 Flux 的邏輯做通訊 / 計算重疊，配合 NVLink 環境；節點外的不同廠牌 PCIe 晶片要能協同。由本 ws 承接（與第 1 節目標一致）。
- **調查**：三路平行（repo 程式碼、論文、開源），子代理擷取原文核對；FlexLink、ConCCL、FlagCX 三項由我親自抽查原文。報告 `reports/20261006_survey_pcie_overlap_hetero.md`。
- **關鍵發現**：
  - Flux 的重疊機制與傳輸無關（tile 等 GPU 記憶體旗標）→ 等級 1 的切入點。
  - NVIDIA 之間的 PCIe + Flux 重疊已存在；NVLink 節點內的 PCIe 額外頻寬上限低（FlexLink +26%、D-001）。
  - 研究空白：不同廠牌 PCIe 晶片以 tile 級重疊加入 NVLink 域；同機跨廠牌 PCIe P2P 無人量過；無開源跨廠牌共用 tile 旗標。
- **卡在哪**：方向解讀（連外部晶片的通道 vs 節點內額外頻寬）與晶片型號 / 軟體堆疊需教授確認。
- **留給下個 session**：確認方向後，先做 1a / 1b，再做「能否從 Flux 外部寫 AG-GEMM 旗標」的技術探針。等級 0 / 1 只需 css-host-158 單機。

## 2026-10-09 · boss 兼 worker · FlexLink 細節；後續計劃 H0–H5

- 讀 FlexLink 全文，細節補進調查報告附錄 A（主機中轉雙緩衝、copy engine、GPU 輪詢、單調計數器；不與計算重疊；8 卡 AllReduce 只 +2%）。
- 確認 `nvidia-smi nvlink -gt d` 免 sudo 可讀累計 Tx / Rx → 代理隔離可驗證。
- **粗估改變了順序** [推論]：晶片若當 TP=4 成員，每層 PCIe 收送量（M=4096：125.8 MB，5.67–19.48 ms）不少於 GPU 算一層（5.03 ms）→ 重疊也救不了；
  當 pipeline 段只在邊界搬 84 MB。→ 工作切分（H0）移到最前面，Flux 式重疊做在選定角色的通訊上。
- 計劃 H0–H5 寫入 STATUS §0。等教授確認方向解讀與晶片型號後開始；H0 不用 GPU，可先做。

## 2026-10-09 · boss 兼 worker · 開始實作；H0 完成

- 使用者核准「開始所有實作」，並採用「先用 CPU 當第一個非 NVIDIA 加速器、做出介面式系統」的方向（Python 原型）。
- 環境：Triton 3.2（原子操作有 acquire / scope）、**無 cuda-python**（不能 `cuStreamWriteValue32`）→ GPU 旗標改以 copy engine 搬 4 bytes 寫入；
  `launch.sh` 的 `CUDA_DEVICE_MAX_CONNECTIONS=1` 會讓搬運 stream 排在等旗標的 GEMM 後面 → 原型不經 `launch.sh`。
  **設計規則：GPU 旗標絕不能用 kernel 寫**（等旗標的 tile 佔住 SM 時，寫旗標的 kernel 排不上去 → 死結）。
- **H0**（`reports/20261009_h0_work_partition.md`，推論）：TP 成員排除；pipeline 段可行、上限 +s/8；示範設計成雙向 chunk 串流。

## 2026-10-09 · boss 兼 worker · I1–I4 實作與量測

- **做了什麼**：介面規格 I1；套件 `scripts/hxb/`（api、cudamem〔ctypes 呼叫 libcuda stream memory ops〕、cpu_backend〔獨立裝置程序 + C 寫的 DMA 引擎〕、
  bridge、gpu_consumer〔Triton 每塊旗標 GEMM〕、pipeline、proxy_backend〔A100 代理〕）；驗收測試 CPU 9/9、代理 7/7；
  I2 特性量測 v1 → v2；I3（6 組 × 6 種 n × 200 輪）與 I4（4 組 × 6 種 n × 200 輪）正式量測，全部 guard CLEAN；結果見 STATUS 第 2 節。
- **踩過的坑（每條都寫進程式註解或規格）**：
  - 模組延遲載入會同步整個 context，與停在 WaitValue 的 stream 死結 → `CUDA_MODULE_LOADING=EAGER` + 暖機 + Triton `do_not_specialize`。
  - DMA worker 的 non-temporal store 要在 worker 自己的執行緒 sfence；測試用 `value()+1` 推目標值是競爭（規格第 4.3 條補上）。
  - 我自己重犯 Phase 0 的錯：試跑的雙向量測沒限定「兩方向同時在飛的窗口」，量到 20 GB/s；修正後 6.0–6.6，與錨點一致。
  - 特性量測 v1 A 段用快取熱的緩衝區，高估裝置 DMA 約 2 倍 → v2（GPU 剛寫入的資料）。
  - 控制執行緒放在計算核心的 HT 兄弟核，計算慢 3–4 倍 → 專屬控制核心；暫存區依形狀重配，每次 20 ms page fault → 只增不減的暫存區。
  - 重複 cudaHostRegister 留下黏著錯誤 712；驗證用的 reference 跨卡 `.to()` 走了 NVLink → 改經 CPU。
- **事前登記**：I3 / I4 腳本在正式量測前 commit；I4 的 3 輪試跑發生在 commit 之前，已顯示 P2 可能不過，預測未改（`results/i4_proxy/dev_runs/README.md`）。
  開發用試跑的 log 歸檔在 `results/i3_overlap/dev_runs/`、`results/i4_proxy/dev_runs/`，標明不是量測。
- **卡在哪**：模型在 n ≥ 16 與 link-bound 情況失準（每塊固定開銷、裝置 DMA 在 pipeline 內慢 33% 未隔離）。
- **留給下個 session**：I3 / I4 分析報告未寫；I5 腳本已寫未跑。

## 2026-10-14 · boss 兼 worker · 與使用者討論 → 真實模型端到端計劃

- 向使用者用白話說明設計（主機記憶體中轉 + 單調計數器、依序單線道保證順序、只在 pipeline 交接點走 PCIe、大小兩層重疊）。
  使用者指出兩點：GPU 在晶片計算時是否閒置（→ 需要多批輪流的「大重疊」，原型未做）、應以真實模型與 NCCL 基準比端到端時間。
- **定案**（STATUS §0 E2E 計劃）：Qwen2.5-32B-Instruct 真實權重；A = 8 卡 NCCL；B = 8 卡 + CPU（PCIe，CPU 做 final RMSNorm）；
  C = 8 卡 NVLink + PCIe 同時傳；先固定批次再 ShareGPT。替身只用 CPU（使用者決定）。
- **事先告知使用者的預期（推論）**：B 比 A 慢約 0.5–2%（CPU 太慢，只能證明協同與量通道代價，無法加速）；C 在本機最多約 +0.5%，可能量不出差別。使用者知悉，三組都做。
- 機器上沒有 vLLM、沒有 Qwen 權重；ShareGPT 資料集在 HF 快取。
- I4 正式量測結果（10-09 跑完）今天才 commit（70e6143）；STATUS / JOURNAL / PROJECT 在 10-09 收工時漏更新，今天補上。
- **留給下個 session**：E2E 步驟 1（下載權重、推論程式、與 HF 一致、實測 CPU 一層、量 A）。

## 2026-10-14 · boss 兼 worker · E2E 步驟 1：真實模型 + 基準 A

- 下載 Qwen2.5-32B-Instruct（62 GB，HF 快取）；側邊 venv `~/.venvs/hetero-proxy-hf`（system-site-packages 沿用 pixi 的 torch；
  加裝 transformers 4.49 / safetensors / accelerate），不動共用 pixi 環境。
- `scripts/e2e/qwen_tp_v1.py`：精簡 TP8 推論（NCCL）。與 HF 單卡參考（`hf_reference_v1.py`，手寫 argmax，避開 generation_config 的
  repetition_penalty）比對：4 個提示各 64 字**全部一致**，第一步 top-20 logit 差 ≤ 0.375（`results/e2e/check/`）。
- CPU 算一層真實模型層（`cpu_layer_time_v1.py`，NUMA1 26 核，guard CLEAN）：prefill 4096 字 2054 ms、decode B=32 109 ms、B=1 13 ms
  → 確認「整層給 CPU 比 8 卡跑完整模型還慢」（推論得到驗證，decode B=32 比估計更慢）。
- 基準 A（eager，guard CLEAN，20 輪）：prefill 1×1024 86.6 ms、8×1024 532 ms、4×4096 1056 ms；decode 每字 47.7–48.6 ms 且不隨批次變
  → **decode 受 Python 發指令速度限制**（G4 用 CUDA graph 約 11.5 ms）。prefill 也比 G4 的最佳策略慢約 45%（eager、未融合）。
- **待與使用者討論**：(1) decode 要加 CUDA graph 才接近實際服務；(2) B 若 CPU 只做「最後的正規化」，prefill 只需最後一個位置 → 每次只傳幾 KB，
  幾乎不經過切塊重疊；建議改成「中間某層的輸入正規化」（全部 token 經 PCIe、回來後經 NVLink 廣播給 8 卡、該層 QKV 逐塊等旗標）。
