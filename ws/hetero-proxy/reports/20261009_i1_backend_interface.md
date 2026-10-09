Supersedes: （無）

# I1：加速器後端介面規格（hxb：heterogeneous exchange bridge）

- 日期：2026-10-09（同日依 I2 / I3 實作修訂：`host_alloc`、`flush`、目標值計數、第 6 節的實作方式與四條硬規則）。性質：設計規格。尚未經 auditor 審查。
- 實作：`ws/hetero-proxy/scripts/hxb/`（Python 原型）。
- 來源：hetero-proxy STATUS 第 1 節的能力分級；調查報告（`reports/20261006_survey_pcie_overlap_hetero.md`）中
  FlexLink（主機中轉雙緩衝、單調計數器）、FlashOverlap（tile 計數觸發傳輸）、MSCCL++（主機代理執行緒）、FlagCX / NIXL / PyTorch PrivateUse1（外掛後端）的共同點。

## 1. 架構

```
NVIDIA 域（CUDA / Flux / NCCL，不改）
    ▲  GPU 旗標（copy engine 寫入）、copy engine 搬資料（D2H / H2D）
    │
 Bridge：主機代理執行緒 + pinned 中轉環形緩衝 + 單調計數器
    │
    ▼  只透過 Backend 介面
 加速器後端：CPU（第一個）｜A100 代理（關閉 peer access）｜NPU 型模擬｜實體晶片（到手後）
```

設計原則：

1. **加速器不必認識 NVIDIA**。後端只處理自己的記憶體與主機 pinned 記憶體之間的搬運；跨到 GPU 的那一段由 Bridge 用 CUDA 做。
   因此任何能 DMA 主機記憶體的裝置都能接（調查顯示沒有任何現成方案支援跨廠牌 GPU↔晶片 P2P，等級 2 留作選配）。
2. **Flux 的邏輯在兩端都成立**：消費者的計算單位宣告「等第 k 塊到了才開始」，生產者每完成一塊就推進一個計數器。
3. **介面最小化**：只放所有加速器都做得到的操作。

## 2. 物件

| 物件 | 說明 |
| --- | --- |
| `Caps` | 後端能力：`level`（0 / 1 / 2）、`scheduling`（`dynamic` 類 GPU / `static` 類 NPU）、`dma_engines`（0 = 搬運與計算不能同時進行）、`memory`（`device_local` / `host_shared`）、`dtypes`、`peak_tflops`、`link_gbps`（估計值） |
| `DevBuf` | 裝置記憶體（不透明控制代碼），有形狀與 dtype |
| `HostBuf` | Bridge 擁有的 pinned 主機記憶體（GPU copy engine 與裝置 DMA 都能存取） |
| `Queue` | 裝置上的依序執行佇列（類似 CUDA stream） |
| `Signal` | 裝置端的 64-bit 單調計數器；主機可讀 / 可等；裝置操作可以「等 ≥ v」與「完成時設為 v」 |

## 3. 後端介面（`hxb.api.Backend`）

| 方法 | 語意 |
| --- | --- |
| `caps() -> Caps` | 回報能力 |
| `alloc(shape, dtype) -> DevBuf` / `free(buf)` | 配置 / 釋放裝置記憶體 |
| `host_alloc(shape, dtype) -> HostBuf` | 配置裝置 runtime 能存取的主機記憶體（中轉區）；Bridge 再向 CUDA 註冊同一塊（`cudaHostRegister` portable + mapped），兩邊的 DMA 都直接讀寫它 |
| `queue() -> Queue` | 建立佇列 |
| `signal() -> Signal` | 建立計數器（初值 0） |
| `copy_in(q, dst, dst_rows, src_host, src_rows, wait, done)` | 非同步：主機 → 裝置 |
| `copy_out(q, dst_host, dst_rows, src, src_rows, wait, done)` | 非同步：裝置 → 主機 |
| `launch(q, op, args, wait, done)` | 非同步：執行一個計算單位（`op` 為後端支援的運算名稱，如 `gemm`、`gelu_gemm`） |
| `Signal.value()` / `Signal.wait(v, timeout)` | 主機端讀取 / 阻塞等待 ≥ v |
| `flush()` | 等到所有已送出的指令都**送達**裝置（送達，不是執行完）。送出是同步的後端不必實作 |
| `sync()` | 等所有佇列完成 |

`wait` 是 `[(Signal, v), ...]`，`done` 是 `(Signal, v)` 或 `None`。

## 4. 語意規則（後端必須遵守）

1. **依序**：同一佇列的操作依提交順序開始；第 k 個操作在第 k−1 個完成、且 `wait` 中每個計數器都 ≥ 指定值之後才開始。
2. **釋放語意**：操作完成時才把 `done` 計數器設為 v；任何觀察到 ≥ v 的一方，都看得到該操作的所有記憶體寫入。
3. **單調**：計數器只增不減，觀察一律用「≥」，不歸零（避免 FlexLink 指出的舊值問題）。跨迭代用世代編號：第 g 次迭代的第 i 塊用 `g·n + i + 1`。
   **目標值由送出方自己計數，絕不能用 `value() + 1` 推算**：前一個非同步操作還沒完成時，兩個操作會拿到同一個目標值，等待被錯的操作滿足（I2 的驗收測試自己就犯過，表現為 200 次中 1 次讀到舊值）。
4. **緩衝所有權**：`copy_in` 的來源、`copy_out` 的目的地，在 `done` 達成前不得被改動。
5. **不得死結**：後端內部不能為了等待而佔住執行搬運所需的資源（例：靜態型裝置若只有一個佇列，必須讓 Bridge 先提交搬運再提交依賴它的計算）。

## 5. 能力分級

| 等級 | 意義 | Bridge 的做法 |
| --- | --- | --- |
| 0 | 算子級：整塊傳完才算 | 一次搬一整塊（chunk 數 = 1） |
| 1 | chunk 級：經主機中轉、每塊一個計數器 | 切成 n 塊串流，計數器一塊推進一次 |
| 2 | 直接 P2P：裝置可直接 DMA GPU 記憶體 | 選配，介面保留，原型不實作 |

另一維度 `scheduling`：`dynamic` 裝置可多佇列、硬體自行排程；`static` 裝置只保證單一固定順序，Bridge 必須照「搬入第 i 塊 → 算第 i 塊 → 搬出第 i 塊」交錯提交（軟體 pipeline）。

## 6. Bridge 與 GPU 端

- **中轉環**：每個方向一組 pinned slot（預設 4 個），slot 在「下游確認用完」後才重用。
- **GPU → 裝置**：GPU 生產者每完成一塊記錄一個 CUDA event → copy stream：D2H 到 slot → 代理執行緒等 event → `copy_in`，`done` 推進裝置端計數器。
- **裝置 → GPU**：代理執行緒等裝置計數器 → `copy_out` 到 slot → copy stream：H2D 到目的位置，接著用 **copy engine 把計數器值（從唯讀 pinned 數值表）寫進 GPU 旗標**。
- **GPU 消費者**：Triton 的 chunk-signaled GEMM，每個 tile 開算前以 `acquire`（system scope）讀旗標，直到 ≥ 該塊的目標值；tile 依 chunk 順序排列。
- **實作方式（2026-10-09 實測可行）**：沒有 cuda-python，以 ctypes 直接呼叫 libcuda 的 `cuStreamWriteValue{32,64}_v2` / `cuStreamWaitValue64_v2`（Flux 寫 per-tile 旗標用的同一套 stream memory operations）。
  計數器放在共享主機記憶體並向 CUDA 註冊，GPU stream 可以直接等待 / 寫入它，主機 → GPU 等待 → GPU 寫回主機 來回約 11 µs。
  因此 GPU 端所有操作都**事先排進 stream**，關鍵路徑上沒有主機代理執行緒（FlexLink 的做法）；GPU 旗標用 `cuStreamWriteValue32` 寫（不是上面原先設想的 4-byte H2D）。
- **四條硬規則**（每條都實際卡死過或量錯過）：
  1. **GPU 旗標只能用 stream memory op 寫，不能用 kernel**——等待中的 tile 佔住 SM 時，寫旗標的 kernel 排不上去會死結。
  2. **不能設 `CUDA_DEVICE_MAX_CONNECTIONS=1`**（`launch.sh` 預設會設）——copy stream 會排在等待中的 GEMM 後面而死結。
  3. **任何 stream 停在 WaitValue 之前，所有 kernel 都必須已載入**：模組載入（cuBLAS / Triton 第一次 launch）會同步整個 context，與停住的 stream 死結。
     用 `CUDA_MODULE_LOADING=EAGER` 加上逐一暖機每個形狀；Triton 的 `target` 參數要 `do_not_specialize`，否則數值改變（1 → 2）會觸發重新編譯與載入。
     所有 pipeline 也要在任何 gated pass 之前建好（`cudaHostRegister` 同理）。
  4. **同一張 GPU 不要同時送出和收回**：同卡同時 D2H + H2D 時，H2D 掉到 6.1 GB/s（I2 特性量測 D 段，對照 Phase 0 錨點 6.46）。
     Outbound 和 Inbound 可以放在不同 GPU（TP 域內 activation 每張卡都有）；GPU0 送 + GPU2 收時兩方向各約 23.7 / 21.3 GB/s。
- **追蹤**：Bridge 與後端記錄每個操作的開始 / 結束時間（主機時鐘），GPU 端以 CUDA event 與 nsys 佐證重疊（CLAUDE.md 5.1.9）。

## 7. 新加速器要做什麼

實作一個 `Backend` 子類：記憶體、佇列、計數器、`copy_in` / `copy_out`（自家 DMA 或 runtime 的拷貝）、`launch` 支援的運算，以及正確回報 `Caps`。
不需要碰 CUDA、NVLink、NCCL 或 Flux。驗收：`scripts/test_hxb_backend_v1.py` 的語意測試全部通過（依序、釋放語意、單調、不死結、正確性）。
