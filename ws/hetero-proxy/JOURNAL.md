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

## 2026-10-14 · boss 兼 worker · E2E 步驟 2：B（CPU 經 PCIe）

- 使用者決定：decode 一律 CUDA graph；B 兩種位置都做（Bfinal / Bmid）。
- `qwen_tp_v2.py`：CUDA-graph decode（固定 KV 視窗 + 位置遮罩的 SDPA）；B 的交棒走 hxb（rank 0 的 GPU0 ↔ NUMA1 CPU 裝置），回來後 NCCL 廣播給 8 卡；
  hxb Outbound / Inbound 拆成 post_device / post_gpu，整段 run 的裝置指令先送，指令傳送不在關鍵路徑上；CPU 後端加 `rmsnorm`（HF 順序）。
- 卡過的坑：(1) graph capture 前沒進 inference_mode → 靜態張量不能原地改；(2) 例外發生時 finally 裡的 barrier 讓其他 rank 永遠等，例外訊息被吞掉
  → 例外先印、出錯不做集體收尾；(3) NCCL 進過 CUDA graph 後 `destroy_process_group` 卡住 → 結果寫完直接 `os._exit`。
- 正確性：v2 四種設定 4 題 × 64 字彼此完全一致；graph 版 A 對 HF 在英文題第 5 字分歧（HF top-2 差 0.625 = 5 個 bf16 ulp，近平手）。
- **run1**（v2，guard CLEAN）：prefill Bfinal +0.1–0.8%；Bmid1 +5 / +16 / +14%（遠超預測 2–6%）；Bmid8 在 8×1024 藏掉 82%；
  decode Bfinal +1.5–2.5%，**Bmid 反而比 A 快 4%** → 兩個疑點，run1 存檔附 README，不作結論。
- 疑點 1 查明：CPU rmsnorm 一次配置整塊暫存 → 8192 行 60 ms（1.4 GB/s，`results/e2e/diag/`）→ 改分塊 15.5 ms → 改 C / OpenMP 約 10 ms（64 行 0.54 → 0.16 ms）。
  C 版與 GPU 版有 0.043% 元素差 1 個 bf16 ulp（fp32 加總順序）；check v3：Bmid 在英文題第 48 字與 A 分歧，該處 HF top-2 logit **完全相同**（20.75 / 20.75）。
  事前登記的 S4（全部與 A 一致）因此不過，照實記錄。
- 疑點 2：v3 加 AsplitF / AsplitM 對照（同樣拆兩個 graph、不經 CPU），B 減對應 Asplit = 純交棒代價。run2 已在跑（tmux e2e_b3）。
- 待查：本推論程式 decode 偏慢（A b32 每字 43 ms，G4 最佳約 11.5 ms）→ 百分比低估交棒代價，報告要附絕對毫秒；之後用 nsys 看時間花在哪。

## 2026-10-14 · boss 兼 worker · E2E run2 與 C 頻寬量測

- run2（v3，guard CLEAN，1898 s）：AsplitM 比 A 快 2.3–6.5% → run1「Bmid 比 A 快」是拆 graph 的效果（S2 成立）。
  交棒純代價（B − 對應 Asplit）：decode 0.32–1.04 ms / 步；prefill Bmid1 8×1024 +26.9 ms（run1 87 ms，C 版 rmsnorm 之後）、Bmid8 +14.5 ms。
  S1 幾乎成立（Bmid1 b64 1.04 ms > 1 ms 不過）、S3 成立、S4 不過（平手翻轉）。guard 一筆 SELF_CPU 警告（自己的範圍外程序 52.8%，一次）。
- C 頻寬（`c_allreduce_bw_v1.py`，guard CLEAN）：所有 f > 0 都比 NCCL-only 慢；首次執行因 sizes 環形索引越界（1024 vs 512 格）失敗，修正後重跑。
  主機路徑本身未優化（torch 加總、輪詢），但理想上限也只有 all-reduce −5.6%、prefill 端到端約 −1.3%（推論）。
- **留給下個 session**：等使用者決定 C 的去留；ShareGPT（步驟 4）；用 nsys 查本推論程式 decode 的 graph 啟動開銷。

## 2026-10-14 · boss 兼 worker · 推論程式加速（v4 / v5）、B 的 v5 量測、C 的診斷

- 使用者：C 不就此結論，要數據說明原因並試並存排程（計劃經確認）；先加快推論程式再 ShareGPT。
- **推論程式**：nsys（`results/e2e/nsys_decode/`）decode 每步 3850 個 kernel、逐元素小 kernel 佔一半 → v4 Triton 融合（add+RMSNorm、RoPE+KV、
  decode attention、silu*up；除 attention 外與 torch 路徑逐位元相同）→ 709 kernel/步；nsys v4 顯示 attention 佔 23% → v5 split-K attention（graph 內 4×）。
  A decode b1 / b32 / b64：v3 22.1 / 43.0 / 61.2 → v5 14.8 / 16.4 / 18.1 ms；prefill 8×1024 535 → 398 ms。剩下最大的是 NCCL（約 25%，A 定義為預設 NCCL，不動）。
  與 HF 的分歧都在近平手處（gap 0–0.625，`check/hf_margin_v4/v5*`）；v5 的 B 對 A 只在 HF top-2 完全相同處翻轉。
- **B v5**（`results/e2e/bench_v5/`，guard CLEAN 0 警告）：交棒純代價 decode 0.23–0.77 ms / 步（Bfinal +1.9–3.8%）；prefill Bmid8 +3.6%、Bmid1 +6.1–8.6%、Bfinal 約 0。
  事前登記：T1 過、T2 / T3 邊界不過各一、**T4 不過（拆 graph 的效果變大：AsplitM 比 A 快 5.5–10%）**，原因未查（推論：單一大 graph 有額外開銷；驗證：nsys 比對 A 與 AsplitM）。
- **C 診斷**（`results/e2e/c_diag/`，guard CLEAN）：8 卡同時 D2H / H2D 每卡約 10 GB/s（單卡 24 / 22）；同時收送時 H2D 只剩 2.3–2.9；
  主機加總 8 份：torch 16 MiB 43 ms → C 32 執行緒 6.9 ms（19 GB/s 輸入）；同步來回 0.05 ms；
  **NCCL 與 PCIe 拷貝同時跑時 NCCL 變慢 2–5×**。對照：GPU 內部拷貝當背景不受影響；主機排指令只要 0.05 ms；
  nsys（`c_diag/nsys/analysis.txt`）：NCCL kernel 與 PCIe 拷貝在時間上確實重疊，重疊者中位數 1.36 ms、不重疊 0.82 ms → 是干擾，不是被排在後面。
  （第一次 nsys 用 os._exit 結束，CUPTI 沒寫出 kernel 紀錄 → 改正常收尾重錄。）
- 結論方向：C 慢的主因依序是 (1) PCIe 流量讓 NVLink all-reduce 本身變慢、(2) 8 卡同時用 PCIe 每卡只剩約 10 GB/s（同時收送 2.5）、(3) 原 torch 主機加總慢。
  → 「與 NCCL 同時傳」的排程方向受 (1) 限制；可能的方向是讓 PCIe 只在 NVLink 閒置時（GEMM 計算期間）傳。待與使用者討論。

## 2026-10-14 · boss 兼 worker · 給教授的進度投影片

- 13 頁（概念、介面原型量測、真實模型 A / B / C 結果與原因、結論、下一步），中文講稿：https://claude.ai/artifact/2Y8G4VW3TX2z5t9rdfHuht
- 數字全部取自已提交的 results（i3_overlap、i4_proxy、e2e/bench_v5、e2e/c_bw、e2e/c_diag、e2e/cpu_layer）；報告人姓名留佔位。

## 2026-10-15 · boss 兼 worker · 異質角色初步測試 P1

- 使用者：目前只跑通沒有加速；要查主流做法並「先初步測試有用再延伸」。調查（P/D 分離、attention 分離、MoE 專家外放、自動分配；
  Lamina、MegaScale-Infer、KTransformers、Fiddler、HybriMoE、HexGen-2、HMA-Serve、A100+GroqRack）共同原則：依特性分工、兩邊同時忙、只傳小資料並重疊、排程器分配。
- P1 四項測試（`scripts/hetero_roles/`，`results/hetero_roles/p1/`，guard CLEAN；C / OpenMP kernel 加在 hxb/csrc/hxbdma.c）：
  T1 attention 外放 ≤ 0.5%（無用）；T2 MoE 專家在裝置上算比搬權重快 5.3–7.3×（有用）；T3 KV 存主機續聊推估 ~15×（有用，待端到端）；T4 CPU 做 decode +1.8%（無用）。
- 自我修正：10-14 回覆說 HTX301 適合 attention 外放，T1 顯示 attention 吃頻寬，低頻寬裝置不適合。
- 量測錯誤兩則（已修，見 STATUS）：專家權重留在 L3 造成讀取速度超過 DRAM 錨點；頻寬 kernel 被加法延遲限制。
- **留給下個 session**：與使用者確認延伸方向（建議 MoE 專家外放的端到端原型；KV 存放的端到端量測可順便做）。

## 2026-10-16 · boss 兼 worker · KV 外存 K0 / K1（CPU 當 KV 存放層，單一請求）

- 使用者決定：晶片短期拿不到，先在本機 CPU 上做出加速；選 KV 外存。釐清與 LMCache / SGLang HiCache 的異同後同意「計劃好就開始做」。
  計劃 K0–K3 寫進 STATUS §0（借鑑逐層重疊、寫入時機、Cake；自己的部分是裝置主動送、干擾排程、介面不綁 NVIDIA）。
- 正確性參考：`kvtier/hf_reference_kv_v3.py`（v1 / v2 撤回：ShareGPT 檔案把對話切成約 2k token 一段，長 AI 回答本身也被切開；v3 接回連續段、合併同角色訊息）。
  3 段真實對話，歷史 1639 / 3590 / 8354 token。
- 程式演進（每版都留著，撤回者加 banner）：
  v1 check 抓到 decode graph 暖身蓋掉 KV 位置 0 → v2 修（同一 bug 影響 E2E v3–v5 check → `e2e/qwen_tp_v6.py` 重驗）；
  v2 smoke：eager 的新一輪 128 token 約 50 ms 都是 Python 發 kernel 的開銷 → v3 全部計時路徑錄 CUDA graph，graph 內用 hxb stream memory op 計數器逐層等搬運（先探針確認可錄進 graph）；
  v3 bench 錄 graph 時 OOM（NCCL 用過的 tensor 在錄製中不重用，32k 重算累積 42 GB）→ v4 all-reduce 用固定緩衝區；
  v4 bench 兩次卡死（8 rank 都在 eager all-reduce、NCCL 工作佇列記為滿、GPU 閒置；gdb + faulthandler 定位）→ v5 給 graph 獨立 communicator，同設定 3 輪除錯跑完。
- **K1 結果**（`results/kvtier/bench_v5/analysis.txt`，200 輪，guard CLEAN）：搬回 + 邊搬邊算比重算快 3.6×（1k）/ 9.9×（4k）/ 15.9×（8k）/ 17.2×（16k）/ 19.4×（32k）；
  8k 以下只比 KV 沒被擠掉多 1.5–7 ms；16k 以上由 PCIe 決定（每卡約 10 GB/s，GPU4–7 比 0–3 慢約 15%）；Cake 在這台機器全面變慢。事前預測 G3 / G4 / G6 成立，G1 / G2 / G5 / G7 不成立（原因見 STATUS）。
- 卡過的坑：`pkill -f <腳本名>` 又把自己的 shell 殺掉（第二次）——停程序一律用 PID。
- E2E v6 重驗（`results/e2e/check_v6/`）：修正後 en_explain A vs HF 由 5/64 變 50/64（原本的「平手」解釋是錯的，是 KV 位置 0 被蓋掉）；Bfinal 4/4 與 A 相同；Bmid 在兩個近似平手處（差 0.000 / 0.375）與 A 不同。已寫入撤回表。
- **留給下個 session**：K2（系統層級：ShareGPT 多人多輪 + GPU KV 容量受限，量被擠掉的比例、平均首字時間、吞吐量、搬運對他人 decode 的干擾）；K3（裝置主動送）。

## 2026-10-16（續）· boss 兼 worker · K2 多人服務迴圈

- 使用者選 K2 完整版、不裝 vLLM 參考。新程式（`scripts/kvtier/`）：`k2_trace_v1.py`（3000 段 ShareGPT 對話，總長中位數 1509、99% 2452 token——
  資料集清理後偏短，報告須寫明）、`k2_kernels.py`（slot 對照表版 decode kernel，測試全過）、`k2_engine_v1.py`（110 個 graph：decode 依批次、
  新對話 / 重算依長度、接續依「新句長度 × 歷史視窗」，歷史長度從 GPU 記憶體讀，hxb 計數器把關搬回）、`k2_serve_v1.py`（排程器，rank 0 決定、gloo 廣播）。
- 卡過：graph 必須在 inference mode 內重播（第二次遇到）；排程器嚴格先來先服務會卡死（排頭等格子、格子被排在後面的人佔著）→ 改為找第一個排得進去的人。
- engine 正確性（`results/kvtier/k2/check_v1`）：搬回 = 留在 GPU 逐位元相同；與 HF 分歧只在平手處；同批 1 人與 2 人的生成會在平手處不同（批次大小改變 GEMM 數值）。
- 試跑（pilot0）→ 登記預測 → 正式量測（loadL、loadH，各 3 種做法 × 2 次 × 600 s，12 次 guard 全 CLEAN）。結果與預測見 STATUS §0。
  輕負載：外存 ≈ 理想、比重算快 2.15×、吞吐量 +1.7%，L1–L6 全成立。重負載：外存首字中位數比重算快 35×、吞吐量 +7.5%（H2 預測 ≥ 10% 不成立），
  外存的尾端是排隊等格子（正在生成的對話佔滿 GPU），不是搬運。
- 待查：背景存 KV 時他人 decode 反而快 1.5–2.8%（兩種負載同方向）。
- **留給下個 session**：與使用者確認下一步（K3 裝置主動送 / 干擾待查實驗 / 整理報告 / auditor 審查）。

## 2026-10-17 · boss 兼 worker · K4 方向一：避開 NVLink 的搬運排程

- 使用者選方向一（理由：Flux 指出 TP 的 all-reduce 在必經路線上）。計劃 K4.1–K4.4 見 STATUS §0。
- K4.1 第一次在錄 graph 時卡死（暖身走 graph 專用 communicator，同一 communicator 混用逐個發與錄進 graph）→ `k2_engine_v2.py`：暖身改走預設 communicator。
- K4.1 v2：背景搬回 decode +48%、存出約 +145%（graph、打亂、兩次一致）。D2H 時運算總在搬運結束時才結束，先懷疑佇列共用，`CUDA_DEVICE_MAX_CONNECTIONS=32` 重跑無變化。
- 逐個發 kernel 的分解實驗（diag v1–v3）：發現小 all-reduce 受害最深，但 v2 / v3 的基準在不同次執行間漂移 50%、出現不可能的結果 → 撤回，教訓：小 NCCL 迴圈不能用逐個發的方式量。
- nsys 兩次寫不出報告（第一次是 `os._exit` 害的，第二次在 8 卡 torchrun 下卡住）→ 改以直接實驗檢驗假設。
- `k4_place_v1.py`（graph、打亂）：搬運只在最大的矩陣乘法開始時放行一小塊（0.5 MB）→ decode 只慢 2.8–3.7%；放在 all-reduce 開始時 5.3–11.9%；全速 45–145%；塊放大到 1 MB 優勢消失。
- **留給下個 session**：K4.2 / K4.3 接進服務迴圈（背景搬回、Flux 式放行；每層多個放行點提高速度）+ 長對話使用者（文件問答）；查 K2「存 KV 沒拖慢」與 K4.1 的矛盾。

## 2026-10-19 · boss 兼 worker · 報告整理

- 完整報告 `reports/20261019_kvtier_k1_k2.md`（K1 / K2 全部細節，每個數字附來源）；機制圖解頁 https://claude.ai/artifact/3RUzQ2kSwebvJhaKjrhTBN 。
- 給教授的量測報告頁（Dashboard）https://claude.ai/artifact/Xn8qrYKMPiH6sFoKpgs4N7 ：所有數字來自 `results/report_prof_20261019/*.csv`，
  由 `scripts/kvtier/export_report_v1.py` 從已提交的原始輸出重算（與各 analysis.txt 一致）。涵蓋 P1、K1、K2、K4.1、正確性、事前預測、更正與限制、下一步。
- 文獻初查（論文方向）：KV 外存與多輪對話已有多篇（CachedAttention、SwiftCache…）；PCIe 爭用有 2025 年的多租戶研究，但沒找到「KV 搬運拖慢 TP decode」及其解法 → 建議以 K4 為論文主軸，待使用者確認論文形式。

## 2026-10-28 · boss 兼 worker · 新題目 X 系列：PCIe 搬運干擾 TP 推論

- 教授建議不做 KV cache 題目（太多人做）。改題：PCIe 搬運對 TP 推論的干擾與 Flux 式排程（K4.1 的發現與 KV 無關）。計劃 X1–X5 寫入 STATUS §0、PROJECT。
- 工具：Triton kernel 內用 `%globaltimer` 打時間戳（graph 內可用、解析度約 1 µs），不再依賴在本機 8 卡下不可靠的 nsys。
- X1：推翻「kernel 啟動被延遲」的推論——graph 內極小 kernel 的間隔不變、GEMM 幾乎不變，受害的是 NCCL 小 all-reduce（+69% / +594%）。
- X1b / X1c（CUDA C，load_inline；要加 pixi 的 lib 路徑才連得到 libcudart；peer access 用擴充自己開）：NVLink 互傳小旗標只有在用系統層級屏障時才被拖慢（×2–4.6）；GPU 層級或不加屏障完全不受影響。
- NCCL 三種協定都受影響 → 下一步看 NCCL 原始碼、寫 GPU 層級同步的 all-reduce。
- 報告更新：量測報告頁（Dashboard）第 3 版——「原因」卡片改成系統層級屏障（x1_fence 圖）、加 X1 受害者表與協定表、撤回啟動空檔說法；
  說明頁 https://claude.ai/artifact/3RUzQ2kSwebvJhaKjrhTBN 第 2 版——更正「kernel 啟動空檔變大 10 倍」三處、進度表加 X1、註明改題。
- 拓撲（`results/xpcie/topology_20261028.txt`）：GPU0/1、2/3、4/5、6/7 各兩張共用一個 PCIe 交換器與一條 Gen4 x16 上行（root port 09:02.0 等）。
  解釋 X1b/X1c 的跨卡現象：只有 GPU1 存出，GPU0 讀主機 ×2.9、系統屏障互傳 ×3.2——GPU0 的請求在共用上行裡排在 GPU1 的寫入後面（推論）。
- 使用者質疑「快的 NVLink 為何要等 PCIe、難道沒人遇過」→ 查文獻與 NCCL 原始碼（不佔 GPU）：
  - **NVIDIA/nccl #2218（2026-06）已有人找到同一個屏障機制**（改 GPU 層級屏障影響大減；NVIDIA 工程師：sys membar 要確保所有流量在系統 POC 可見，對無關流量敏感）。
    #1023（2023）維護者還說走 NVLink 不該受干擾。→ 機制不是我們的貢獻，STATUS 的故事與差異已改。
  - NCCL 2.19.3 原始碼（repo 附的，非 2.21.5）：Simple 每段前有 sys 屏障；LL128 在 A100 用 GPU 層級；LL 沒有屏障——但 LL 實測同樣受害 → NCCL 部分的原因**未證實**。
  - 報告頁「原因」卡片目前寫成已確定，需要在 X1d 之後修正（**待辦**）。
- **留給下個 session**：X1d（對端寫入的極小 kernel、NVLink 背景搬運、同 / 不同交換器、塊大小）待使用者同意後跑。
- **X1d（使用者同意後執行）**：`x1d_scope_v1.py`（單一程序控制 GPU0/1/2）、`x1e_graphdump_v1.py`、`run_x1d_v1.sh`、`run_x1d_rep_v1.sh`，全部 guard CLEAN。
  - 試跑後發現 NCCL 在 graph 內每個 all-reduce 前後各加一個 event 節點（`NCCL_GRAPH_MIXING_SUPPORT=1` 預設）→ 加了 `tiny_ev` / `tiny_evrec` 受測項與 D7 / D8 預測（正式跑之前）。
  - 結果：event 記錄節點被本卡存出拖慢 ×4.3–5.1；`NCCL_GRAPH_MIXING_SUPPORT=0` 讓 LL all-reduce 的干擾 +594% → +5%、decode 一步 +202% → +2.5%（兩次獨立程序重現），沒搬運時也快 24–26%。
    Simple 仍 +203%（每段系統層級屏障）。系統屏障也被本卡 NVLink 搬運拖慢（D2 不成立）、被同交換器鄰卡拖慢（D1 成立）。LL 資料路徑與 kernel 結束都不受影響（D4 成立、D5 不成立）。
  - 自我一致性：「讀主機記憶體＋本卡存出」跨程序雙峰（1.7 / 5.0 µs），程序內穩定 → 記為發現，原因未知。
  - 撤回表加兩條（「全部 NCCL 都是系統屏障」、「系統屏障慢是 PCIe 獨有」）。
  - 報告更新：量測報告頁第 4 版（「原因」卡片改成 mixing 開／關對照圖＋單卡拆解表＋三協定表；屏障圖改成 Simple 的原因；比喻、限制、下一步更正；新資料集 x1d_scope、x1_nomix，所有新數字綁資料集）；說明頁第 3 版（干擾原因、進度表加 X1d、新意降為「中」、論文主軸）。
- **留給下個 session**：與使用者確認方向——K4.1 在 mixing=0 下重量、vLLM / SGLang 實際的 all-reduce 路徑、塊大小實驗解釋 182 µs 放大、Simple / prefill 大小的干擾。

## 2026-10-29 · boss 兼 worker · 選題：MoE 專家「多 GPU＋PCIe 裝置」混合推論（文獻與原始碼調查，未佔 GPU）

- 使用者：關掉 NCCL 設定不算貢獻，要排程相關的論文題目。提了四個候選（A 背景載入權重、B MoE 異質排程、C 異質 TP 切分、D fusion-dispatch），推薦 A。
- 使用者要求說明 A 的細節 → 細算後自我更正三點：(1) 每卡 8 GB 以 9–12 GB/s 約 0.7–0.9 秒搬完，停機載入不到 1 秒，問題太小；
  (2)「一半 PCIe、一半 NVLink 交換」錯誤——TP 下每卡要的資料不同，4 條上行線都滿時轉送不增加頻寬；(3)「prefill 關設定後仍慢約 2 倍」錯誤——+203% 是強迫小訊息走 Simple，真模型 prefill 在 K4.1 只 +6.7%。
  → 建議把 A 併入 B：PCIe 上有持續、在關鍵路徑上的流量（冷門專家放在 PCIe 裝置）。使用者第一次說明看不懂，改用廚房比喻重講。
- 使用者：「查一下別人有沒有在做，可以直接從人家做法中去改善」→ 文獻＋讀 KTransformers / SGLang 原始碼，報告 `reports/20261029_lit_moe_hybrid.md`。
  - 單 GPU＋CPU 很擠；GPU＋NDP（MoNDE、DynaNDE）是單一加速器；KTransformers＋SGLang（TP8＋CPU 專家）是最接近且開源的改良對象。
  - 原始碼確認：只有 tp_rank 0 和 CPU 交換；GPU 用 `cudaLaunchHostFunc` 回呼等 CPU；專家分配靜態；kt-kernel 已支援昇騰、摩爾線程、沐曦、AMD 當「GPU 端」，慢的一端一律是 CPU。
  - 本機：2× Xeon Platinum 8358（無 AMX）、主機記憶體 1.5 TB、磁碟剩 672 GB。
- 文獻工具：scratchpad 用 `pip install --target` 裝 pypdf 讀 PDF（本機無 pdftotext / pdftoppm）；GitHub 原始碼用 sparse clone（無 gh）。
- **留給下個 session**：等使用者同意第一步（另建 venv 裝 SGLang＋kt-kernel，TP8 小 MoE，量每層 MoE 的時間組成＋回呼 vs 計數器微基準）。
- 使用者：「打算怎麼改良」→ 主軸：CPU 算一部分冷門專家，同時 8 卡各經自己的 PCIe 線搬另一部分的 1/8 來算（報告第 6 節）；P1 單插槽數字估約 1.9×。
- 使用者：「開始驗證這個題目能不能做，目標是快速生出一份論文」→ M0（`scripts/moehy/m0_split_v1.py`、`run_m0_v1.sh`，tmux＋guard）：
  - 寫腳本時發現 P1 的 CPU 只用一個插槽，KTransformers 用兩個 → 1.9× 偏高；事前預測改登記 1.15–1.30×（跑之前）。
  - 冒煙測試（`results/moehy/m0_smoke/`）：/proc/self/numa_maps 會把相鄰的配置合成一段，NUMA 頁數混在一起 → 改用 libnuma move_pages 逐頁抽樣（正式跑 256/256 頁都在指定節點；冒煙的 NUMA 欄位不可信）。
  - 正式 30 輪（`results/moehy/m0_v1/`，guard CLEAN）：4 token 1.24×（單線 1.06×）、1 token 1.10×、16 token 2.03×（CPU kernel 受計算限制，偏高）；互搶吃掉約一半理想增益，且不是 DRAM 頻寬被吃滿。判定規則達成 → 可做。
  - 主機→GPU 頻寬上限：卡 21、交換器 25、插槽 42、8 卡 85 GB/s → 寫入 `common/cost_model/params.json`（h2d_topology_m0、cpu_dma_interference_m0）。
  - 可行性：本機已有 Kimi-K2.6（555 GB int4，專家 7168×2048），8×80 GB 放不下＋KV → 真實動機；kt-kernel 的 RAWINT4 支援本機 CPU；Apex 全文確認單卡單線。
- **留給下個 session**：使用者同意後 M1（kt-kernel RAWINT4 量 Kimi-K2.6 專家 CPU 速度）、M2（SGLang＋kt-kernel TP8 跑 Kimi-K2.6、量每層時間組成）。

## 2026-10-30 · boss 兼 worker · 選題重找（通訊優化、推論或訓練、10 週論文）＋環境實驗 T1 / T1b / T2

- 使用者：重新找適合這台環境的題目，可以先看拓撲、做一點環境實驗；通訊優化，推論或訓練都可；10 週內一篇架構完整的論文，不必頂會。
- 拓撲：每個 PCIe 交換器 = 2 GPU＋1 張雙埠 100 GbE 網卡（PXB）；158 的 rail 4 已有 IP。159：8 卡被使用者帳號 `rogerlee` 的 SGLang 服務佔用（DP4×TP2，已跑 80 分鐘），且只剩 6 個網卡埠（GPU4/5 交換器的網卡不見）→ 沒有做跨節點量測。
- 三個背景文獻搜尋（推論側、跨節點、單機訓練），刻意找「已做過」的證據；重點與網址整理在報告附錄。
- T1（`scripts/topic/t1_bigcoll_pcie_v1.py`、`run_t1_v1.sh`）：8 卡背景 d2h 讓 16 MiB RS/AG ×2.9–3.0、32 MiB ×2、64 MiB ×1.4，≥ 128 MiB ≤ ×1.08；h2d ≤ ×1.5。
  強制 Simple / LL128：16 MiB 以上預設就是 Simple；LL128 鈍感，有 d2h 時 RS 16 MiB 快 2.2×，128 MiB 又是 Simple 好。
  冒煙時發現「每卡自己整段的平均速度」被先搬完的卡灌水 → 改成只算中間一半；改了仍是 12.6–13.0 GB/s／卡（8 卡 101–104），比 M0 的 85.4 高 18–22%，未解，記在 analysis.txt。
- T1b（加 `--bg_ranks`）：只有 GPU0＋GPU1 搬 ≈ 8 卡都搬；只有 GPU0 搬 RS 16 MiB ×1.84。
- T2（`t2_rdma_paths_v1.sh`、`t2_copy_bg_v1.py`，同埠迴環）：GDR 讀 48 Gb/s（8 QP 一樣）；主機來源 97.4，但同交換器兩卡 d2h 時掉到 30.5。原本以為上下行方向不同不會互搶——錯。
- nccl-tests 錨點：要用 pixi 的 libcudart.so.12 與 libnccl（系統 /usr/local/cuda 是 13.x、nccl-tests 連系統 libnccl）；T1 無搬運頻寬比 nccl-tests 高 9–12%。
- 報告 `reports/20261030_topic_search.md`：推薦 T（訓練時 offload 干擾的量測＋排程器）、備案 F（fusion-dispatch 成文）、M 第三；跨節點題目不建議當主線。
- 撤回表加一條：「背景搬運對大訊息影響小（約 7%）」只對 ≥ 128 MiB 或 h2d 成立。
- **留給下個 session**：等使用者選題。若選 T：第 1 週否證實驗（真實 FSDP2 CPU offload、TP＋SP activation offload 訓練步驟中干擾占比 ≥ 10%），獨立 venv 裝訓練框架；同時解 101 vs 85 GB/s。
- （續 10-30）使用者決定只用 158；問 offload 怎麼做、為何選 T → 說明主情境（Qwen2.5-32B 全參數微調、TP8＋SP、Adam 狀態放主機）並記入 STATUS。
- 使用者問「搬運能否結合 Flux 的切割」→ 先讀 Flux 同步：每個 tile 用 `ld.acquire.sys` / `fence.acq_rel.sys`（`system_barrier.hpp:41-52`）→ T3（`t3_flux_vs_hostcopy_v1.py`，30 輪、guard CLEAN、融合結果 = NCCL 路徑）：
  大 GEMM 層 Flux 吸收干擾（×1.07–1.11 vs NCCL ×1.23–1.36）、qkv 小 GEMM 層 Flux 更慘（×2.12 vs ×1.65）；融合優勢隨搬運改變。報告加 5.7 節：每層「融合與否＋搬運窗口＋Flux 式切塊放行＋協定」的合併架構。

## 2026-10-30（下午）· boss 兼 worker · 題目 T 的驗證

- 使用者：先規劃並做足夠的實驗，確認整體值得投入，再開始規劃。
- 事前登記：`reports/20261030_t_validation_plan.md`（四條判定規則，跑之前提交 8e1fca8）。
- V1 / V2（`t1_bigcoll_pcie_v2.py`）：限速照比例減輕；通道數 32 更敏感、4 很慢；NCCL 2.32.3（`~/opt/nccl-2.32.3`、LD_PRELOAD）仍 ×1.9–2.8；LL128 大訊息慢 23–29%。
- V3 訓練步驟原型（`v3_step_v1/v2/v3.py`）：手寫 TP8＋SP autograd，梯度對 fp32 參考 PASS。
  v1 的窗口排程在 activation offload 卡住（最後幾層的資料還沒搬完反向就要用）→ v2 加放行點、最後一層不搬、加限速對照；v3 加混合排程（失敗）。
  結果：naive 損失 1–6%，峰值 24%（延遲更新、4096 × 1）；窗口排程收回 61–81%（4/6 情境）；最平衡的那格沒有好解（鏈路必須在通訊時也搬）。
  2048 token 變異大（推論：CPU 發 kernel 受限），不列入判定。
- 文獻再查：ChunkFlow §2.3 明寫 NVLink 機器上兩者不干擾（沒量過）；PyTorch 論壇的 rank skew 只解釋同步搬運那一次，非同步仍慢、未解。
- 結果報告 `reports/20261030_t_validation_result.md`：有條件值得；待使用者在 T / F / F＋T 中選。
