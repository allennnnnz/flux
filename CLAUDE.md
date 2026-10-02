# CLAUDE.md — 專案指引與協作制度

每個 session 開始時：先讀本檔 → 再讀 `PROJECT.md` → 然後**只讀**你被指派的
`ws/<workstream>/` 目錄。不要動其他 workstream 的檔案。

本檔記錄**穩定不變**的東西：專案是什麼、機器長什麼樣、制度、量測與報告準則。
會變的東西（目前有哪些 workstream、各自狀態、下一步）在 `PROJECT.md`。
本檔改動需 boss session 核准。

---

## 1. 專案是什麼

改良 Flux（arXiv:2406.06858，GEMM + 通訊 kernel fusion）的實作，最終目標是
**異質計算**：一顆只能透過 PCIe 連接（無 NVLink）的國產計算晶片，與 8 張 A100
的 NVLink 域協同計算。晶片尚未到手，先以 A100 代理裝置開發。

Phase 0（PCIe 與 Flux 特性量測）已結案。結論與設計規則見
`docs/PHASE0_FINDINGS.md`，完整記錄見 `docs/PHASE0_STATUS.md`（唯讀）。

Phase 0 最重要的一課不是任何數字，而是它的失敗模式——四輪修正的每一個缺陷
都是同一件事：**量測拿錯了基準，而推翻它的證據早就印在腳本自己的輸出裡。**
第 5 節的準則就是為了不再犯這個錯。

## 2. 環境

- `css-host-158`，8× A100-SXM4-80GB，NVSwitch，所有 GPU 對 `NV12`，驅動 615.71.09。
- NUMA：GPU0–3 → node 0（CPU 0-31,64-95），GPU4–7 → node 1（CPU 32-63,96-127）。
- **無 sudo。** `lspci -vv` 回 `<access denied>`；PCIe 鏈路狀態用 sysfs
  `/sys/bus/pci/devices/<BDF>/current_link_{speed,width}`。
- pixi 環境；PyTorch 2.6.0+cu124；`nvidia.nvshmem` 可 import。
- 可用工具：`nsys`（`/usr/local/bin`）、`nvcc`（`/usr/local/cuda/bin`）、
  `nvbandwidth` v0.10（需自行從 GitHub 編譯，不需 Boost）。無 `ncu`、無 `pynvml`。

### 執行 Flux

```bash
pixi run --manifest-path pixi.toml ./launch.sh <script> [args]    # 必須在 repo 根目錄
```

`./launch.sh` 單獨跑會 `torchrun: command not found`。

### 已知陷阱（每一條都實際花掉過時間）

| 陷阱 | 事實 |
| --- | --- |
| PyTorch 跨裝置 `copy_` | 在**來源**裝置的 current stream 執行（`aten/.../Copy.cu`），不是目的裝置。設錯邊會讓每卡送出串行在 default stream，Phase 0 因此慢 5.7 倍。 |
| Peer copy 大小 | 每卡每輪 < 1 GiB 是 per-copy overhead bound；**stream 數無關**（1 條與 7 條相同）。每卡 egress 一筆 copy 就飽和，峰值併發 = GPU 數。 |
| CUPTI copy kind | pinned-host→device 被標成 `Peer-to-Peer`，不是 `HtoD`。依 bytes 區分，不依 kind。 |
| AG+GEMM weight 形狀 | repo 慣例 column-shard：每 rank 持 `(N // world_size, K)`（`test_ag_kernel.py:99`）。傳完整 N 會讓 GEMM 膨脹 8 倍。 |
| `staging_stream_bench` CLI | `--devices 0,1` 用空格；`--devices=0,1` 直接 exit。 |
| `use_cuda_core_ag=True` | fp16 不能跑：`ag_a2a_mode` 只對 INT8+FP32 scale 實例化（`all_gather_impls.cu:127`）。 |
| GEMM_RS 通訊 | sm80 單節點**無法隔離**：scatter 在 GEMM epilogue 內經 `output_scatter_ptrs` 直寫對端；`forward_reduce_scatter` 只做 local reduce，`forward_barrier` 是 no-op。不要找 API，要量就得 instrument kernel。 |
| Flux AG 模式 | 這台機器上 **All2All + `use_read=True`** 最快；Ring2D 走 `copy_ring_push_2d_pcie`，為 PCIe 拓撲設計，在 NVSwitch 上是最慢的之一。Phase 0 原始量測全用 Ring2D。 |

### Flux 內部你會用到的入口

- `flux.AllGatherOp`（`src/pybind/flux_coll_op.cc:37`）：Flux AG 通訊路徑獨立暴露，
  `run()` 自含 local copy + barrier + transfer，與 `AGKernel.forward()` 內部同一段程式碼。
- `AGKernel.gemm_only()`（`src/ag_gemm/ths_op/all_gather_gemm_op.cc:167`）：**同一個**
  AG-fused GEMM kernel，barrier 為 `torch::ones({world_size})`（所有 signal 預設 true），
  輸入已就位。這是正確的 GEMM-only 參考，不是另一個 op。
- `ag_signal_ptr()`、per-tile signal 機制：`src/coll/ths_op/all_gather_op.cc`。

## 3. 檔案架構

```
CLAUDE.md                  本檔。制度與專案指引。改動需 boss 核准。
PROJECT.md                 單一入口：workstream 總表、目前狀態、下一步。boss 專屬。
docs/
  PHASE0_FINDINGS.md       Phase 0 結論與設計規則（唯讀）
  PHASE0_STATUS.md         Phase 0 完整記錄（唯讀）
  DECISIONS.md             決策日誌（append-only，boss 專屬）
  <其餘>                   上游 Flux 文件，不要改
experiments/
  2026-09-23-phase0-*/     Phase 0 原始檔案庫：scripts、results、修正文件。唯讀。
ws/
  <workstream>/
    STATUS.md              該 workstream 唯一權威
    JOURNAL.md             工作日誌（append-only）
    scripts/               程式；撤回的腳本加 WITHDRAWN banner，不刪除
    results/               原始輸出（CSV、nsys report、log）——一律提交
    reports/               分析報告，檔名 YYYYMMDD_主題.md
common/
  cost_model/params.json   共用量測參數；每個數字附 source / date / method
  measure/                 共用量測工具（clock 記錄、交錯 ECT harness 等）
```

規則：
- `PROJECT.md`、`DECISIONS.md`、`CLAUDE.md` 只有 boss session 能改。
- workstream 內 `STATUS.md` 是唯一權威。報告與 STATUS 矛盾時以 STATUS 為準，
  並在 STATUS 的撤回表登記，格式固定：`| 撤回主張 | 出處 | 原因 | 替代 |`。
- **沒有原始輸出的數字視同不存在。** results/ 一律提交。
- 跨 workstream 需要的數字（頻寬、延遲、開銷）由量測方寫進
  `common/cost_model/params.json`，附 source 檔路徑與日期；其他 workstream 只讀不改。
- `docs/` 底下除了本專案的四個檔案，其餘是上游 Flux 文件，不要動。

## 4. Session 角色與程序

**boss session**（使用者主導）：指派 workstream、審查各 STATUS、更新 `PROJECT.md`、
把重大結論寫進 `DECISIONS.md`。裁決跨 workstream 的矛盾。

**worker session**（單一 workstream）：讀 CLAUDE.md → PROJECT.md → 自己的
`ws/<name>/STATUS.md` 與 JOURNAL 最近條目。只在自己目錄內工作。結束前完成收工程序。

**auditor session**（審查）：不修程式，只驗證方法與數字。輸出審查報告到該
workstream 的 `reports/YYYYMMDD_audit_<主題>.md`。**每個要進 DECISIONS.md 的結論，
之前必須經過至少一次獨立審查。** auditor 的檢查清單見第 5.3 節。

### worker 收工程序（每次 session 結束前必做）

1. 更新 `ws/<name>/STATUS.md`：現在成立什麼、撤回什麼、下一步是什麼。
2. `JOURNAL.md` 追加一條：日期、做了什麼、卡在哪、留給下個 session 的話。
3. 提交所有程式與 `results/`。
4. 若產生跨 workstream 的數字，更新 `common/cost_model/params.json`。
5. 若發現既有結論有誤：在自己 STATUS 的撤回表登記，並在 JOURNAL 標記
   **「需 boss 裁決」**。不要直接改別人的檔案、`PROJECT.md` 或 `docs/`。

## 5. 量測與報告準則

### 5.1 量測

1. **基準獨立**：任何比率（效率、加速比、增益）的分母必須是對**同一條路徑**的獨立
   量測，並註明來源檔。禁止的實例：拿 NCCL 當 Flux 通訊基準；用同一個 wall clock 同時
   當 H2D 與 D2H 的分母。
2. **錨點對照**：任何新的頻寬 / 延遲數字，寫入文件前先對照 `params.json` 中的已知錨點
   （單對 peer copy 270 GB/s、Flux All2All 188 GB/s/卡、PCIe 單向 ~22 GB/s、nvbandwidth）。
   偏差超過 20% 且無解釋 → 先懷疑量測，不是硬體。Phase 0 的 D 段缺陷本可在五秒內被
   這條規則攔下。
3. **自我一致**：腳本若同時產出多個視角（wall clock、CUDA event、nsys），必須互相對帳
   後才能報告。不一致是**發現**，不是雜訊，不得丟棄。Phase 0 的雙向缺陷，證據在自己
   的 CSV 裡放了兩小時。
4. **效率 > 100% = 基準錯誤**，立即停止並回查，不得發表。
5. **差值量測交錯進行**：要量 A − B（如 ECT = overlap − GEMM-only），A 與 B 在同一輪內
   相鄰執行、各自用 CUDA event，取逐輪差值的中位數，≥ 200 輪。**禁止**兩個分別平均的
   迴圈相減——那會產生負值。
6. **背景記錄 SM clock**，丟棄降頻輪次。用 `common/measure/clock_logger.py`。
7. **雙向傳輸**各方向獨立計時，並報告**限定在兩方向同時在飛的窗口**內的頻寬。
   wall clock 混合了競爭與非競爭兩個階段，哪個都不代表。
8. **所有頻寬同時給 per-GPU 與 aggregate**；集合操作報 **rank-max median**。
9. **Nsight Systems 驗證時間軸**：任何關於「並行」「重疊」「串行」的主張，都要有
   profile 佐證，而不是從總時間推。
10. **可能時用外部工具交叉驗證**（nvbandwidth、NCCL 參考值），並報告偏差。

### 5.2 報告與文件

- 未經量測驗證的推論，寫進任何文件時標記 **「推論」**，並附「如何驗證」一句。
- 報告檔名 `reports/YYYYMMDD_主題.md`，首行寫 `Supersedes:`（若有）；被取代時
  在舊報告頂部加 `Superseded-by:`，不刪除。
- 腳本版本化 `name_v2.py`、`name_v3.py`。撤回的腳本頂部加註：
  ```
  # WITHDRAWN <date>: <一句原因>. Replaced by <path>.
  ```
  保留供稽核，不刪除。
- 數字進文件時附三件事：來源檔路徑、量測日期、方法一句話。

### 5.3 auditor 檢查清單

審查報告至少回答：
- [ ] 每個比率的分母是什麼？是同一路徑的獨立量測嗎？
- [ ] 有沒有任何效率 > 100%、負的差值、或兩個方向數字完全相同？
- [ ] 腳本的所有輸出欄位都被用上了嗎？有沒有被印出來又被丟掉的矛盾欄位？
- [ ] 頻寬數字對照過錨點了嗎？偏差有解釋嗎？
- [ ] 「並行 / 重疊」的主張有 nsys 佐證嗎？
- [ ] 差值量測是交錯的嗎？輪數夠嗎？有記錄 clock 嗎？
- [ ] results/ 有原始輸出嗎？能從原始輸出重算出報告裡的數字嗎？
- [ ] 標記為「推論」的東西有沒有被當成事實引用？
