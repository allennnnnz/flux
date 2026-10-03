Supersedes: （無）

# gpu1 唯讀盤點：硬體、共用狀況、能做驗證計劃的哪些項目

- 日期：2026-10-03 · 作者：fusion-dispatch（在 gpu1 上的 session）
- 性質：**唯讀盤點**。沒有編譯、沒有跑任何 GPU 量測、沒有改程式。
- 原始輸出：`results/gpu1_inventory/00_host.txt` … `17_slurm_nodes.txt`（2026-10-03 16:42–16:46 收集）。
- 對照的計劃：`reports/20261003_g4_flux_value.md` 第 3 節（R1–R3 與成功標準已凍結，本報告不改）。

## 0. 結論（白話）

1. **gpu1 是 4 張 V100-PCIE-32GB（sm70），沒有 NVLink，PCIe Gen3。** 和 css-host-158（8× A100 SXM、NVSwitch）差很多。
2. **Flux 在這台不能跑。** 不是編譯慢，是程式碼直接拒絕：`src/cuda/op_registry.cu:39` 只接受 sm80 / 89 / 90，
   `:46-51` 只認得 A100 / H20 / H800 / L20 的 SM 數（V100 是 80 個 SM）。Flux 的 GEMM kernel 也用到 sm80 才有的指令。
   要支援 V100 等於重寫 kernel，又違反 D-006（有證據前不改 `src/`）。**不建議。**
3. **V100 沒有 bf16 tensor core。** G4 全部用 bf16；在這台必須改 fp16，這是一個實驗條件的改變，要寫明。
4. 依凍結的規則「路徑不可用時退回 R2」，在 gpu1 上 **R1、R3 都退回 R2（vLLM 預設）**，三條對手規則變成同一條。
   決策器只能在「TP + AllReduce」和「序列平行 + NCCL」之間選。**這台無法回答原本的核心問題**
   （Flux 行為不同時，決策器是否比簡單規則好）。
5. 這台能做的是：**在真正的 PCIe 機器上驗證「切法決策 + 模型校準」能不能搬到新架構**（新 GPU、新連線、fp16），
   等於計劃第 3 節 E2（PCIe-only，Flux 不可用）的真機版。它也和專案最終目標（PCIe 連接的晶片）有關。
6. **共用機器**：盤點時 28 個登入、7 個其他使用者在線，但 **GPU 上沒有任何計算程序**（只有 Xorg 的顯示程序，各 4 MiB）。
   Slurm 存在但 gpu1 狀態是 DOWN、沒有 GPU 資源設定 → 大家直接用，**沒有排程保護**，別人隨時可能開 GPU 工作。
7. **`exclusive_guard.py` 在這台會誤判**（兩個 bug，見第 4 節）：
   - 使用者名稱被 `ps` 截成 8 字（`allenzh+`），自己的程序被當成「別人」→ 會一直 ABORT 或 CONTAMINATED；
   - CPU 用「整個生命期平均」，別人突然吃滿 CPU 抓不到（實例：user_D 的程序瞬間 151.7%，守衛看到 0.2%）。
   - 建環境前要先修（建議 `exclusive_guard_v2.py`）。
8. **建環境估計**：不用編譯 Flux。vLLM venv（torch 2.6.0+cu124 + vLLM 0.8.5.post1）約 15–30 分鐘下載安裝（裝在 NAS 上，速度未測）；
   腳本改成「不依賴 Flux、fp16、不寫死 `/home/rogerlee`」約 2–4 小時；校準 + 標準答案約 30–60 分鐘 GPU 時間。

## 1. 硬體與軟體

| 項目 | gpu1 | css-host-158 | 來源 |
| --- | --- | --- | --- |
| GPU | **4× Tesla V100-PCIE-32GB** | 8× A100-SXM4-80GB | `01_nvidia_smi.txt` |
| 架構 | **sm70**（Volta），80 SM | sm80，108 SM | `05_gpu_query.csv` |
| 記憶體 | 32 GB HBM2 | 80 GB HBM2e | 同上 |
| bf16 tensor core | **沒有** | 有 | 規格 [推論，未量] |
| SM 最高時脈 | 1380 MHz；功耗上限 250 W | 1410 MHz | `05_gpu_query.csv` |
| 驅動 / CUDA | 580.142 / CUDA 13.0 driver | 615.71.09 | `01_nvidia_smi.txt` |
| 系統 nvcc | `/usr/bin/nvcc` 12.0；`/usr/local/cuda` 11.6 | pixi 12.4 | `14_tools.txt` |
| persistence mode | **Disabled**（無 sudo 無法開） | — | `05_gpu_query.csv` |
| OS | Ubuntu 24.04.4，kernel 6.8.0-111 | — | `00_host.txt` |

### 卡之間怎麼連

```
       GPU0  GPU1  GPU2  GPU3   NUMA
GPU0    X    PHB   SYS   SYS     0
GPU1   PHB    X    SYS   SYS     0
GPU2   SYS   SYS    X    PHB     1
GPU3   SYS   SYS   PHB    X      1
```

- **沒有 NVLink**（`nvidia-smi topo -p2p n` 全部 NS，`04_nvlink_status.txt`）。
- 每張卡 PCIe **Gen3 x16**（8.0 GT/s，current = max），各自直接接在 CPU root port 上，沒有 PCIe switch（`08_pcie_sysfs.txt`、`10_lspci_tree.txt`）。
- **P2P 只在同一顆 CPU 下的一對卡之間可用**：GPU0↔1、GPU2↔3 是 OK；跨 CPU 是 TNS（拓撲不支援）（`07_p2p.txt`）。
  跨對的通訊要經 host 記憶體 + CPU 之間的 QPI。
- 理論單向 ~15.75 GB/s，實際一般約 12–13 GB/s [推論，未量；建環境後用 nvbandwidth 或 peer copy 量]。
  對照 css-host-158 單對 peer copy 270 GB/s，**慢約 20 倍**。

### CPU / 記憶體 / 磁碟

- 2× Xeon E5-2678 v3（Haswell，12 核 × 2 執行緒 × 2 顆 = 48 執行緒）；NUMA 0 = CPU 0-11,24-35（GPU0/1），NUMA 1 = CPU 12-23,36-47（GPU2/3）。
- 記憶體 125 GiB，盤點時可用約 104 GiB；swap 2 GiB 幾乎用完（1.7 GiB）。
- 磁碟：`/` 只剩 **9.2 GB（98%）**，`/tmp` 也在這裡；`/home` 是 NAS（`nas:/volume1/homes`，剩 8.5 TB）。
  → 建環境時 uv / pixi 快取、`TMPDIR` 都要指到 `/home` 底下，不能塞 `/tmp`。NAS 上 import / 編譯可能較慢。

### 工具

| 工具 | 狀態 |
| --- | --- |
| nsys | `/usr/bin/nsys` 2022.4.2（舊版，可用） |
| ncu | `/usr/bin/ncu` 2022.4.1（css-host-158 沒有） |
| pixi | 0.59.0（`~/.pixi/bin`） |
| uv | 0.11.14；也有 micromamba、`~/anaconda3` |
| python | 系統 3.12.3（G4 的 venv 是 3.11，uv 可自行下載 3.11） |
| tmux / git / gcc 13.3 / cmake 3.28 / numactl / docker | 有 |
| ninja、nvbandwidth、pip3 | 沒有（可由 venv / 自行編譯） |
| Slurm | 有，但 gpu1 = DOWN、`Gres=(null)`（`17_slurm_nodes.txt`），不能用來搶獨佔 |

## 2. 共用狀況（`13_users_procs.txt`、`15_guard_dryrun_and_top.txt`）

- 機器開機 62 天，28 個登入，其他使用者：user_A、user_G、user_D、user_E、user_F、user_H、user_B、user_C。多數掛在 tmux 裡。
- **GPU：4 張都閒置**，只有 Xorg 顯示程序，沒有計算程序。
- CPU：整體約 93% idle，load 3.5–4.3（48 執行緒）。持續的負載：root 的 cadvisor（~35–45%）、containerd；
  其他使用者有幾個常駐的 opencode / codex / python 程序；user_D 有一個程序短暫衝到 151.7%。
- 在這台，**跨卡通訊要經 host 記憶體與 QPI**，所以別人的 CPU / 記憶體密集工作會比在 css-host-158 更容易干擾通訊量測 [推論]。

## 3. Flux / vLLM 能不能在這台跑

| 元件 | 結論 | 根據 |
| --- | --- | --- |
| Flux 編譯 | 用 `--arch 70` 編不起來或跑不了 | CMake 預設只有 80/89/90（`CMakeLists.txt:38-48`）；kernel 用 sm80 指令（cp.async 等） |
| Flux 執行 | 即使編好也會在初始化就停止 | `op_registry.cu:39` `FLUX_CHECK(arch_num == 80 \|\| 89 \|\| 90)`；`:46-51` SM 數 80 不在清單 |
| 移植 Flux 到 sm70 | 不建議 | 要重寫 GEMM kernel；違反 D-006 |
| PyTorch 2.6.0+cu124 | 可以 | cu124 版包含 sm70；驅動 580 夠新 [推論，建環境時確認] |
| vLLM 0.8.5.post1 | 可以裝；我們只用它的 communicator / custom AR，不跑完整引擎 | V100 上 V1 引擎會退回 V0（FlashAttention 需要 sm80）[推論]；block 驗證用 PyTorch SDPA，不受影響 |
| vLLM custom AllReduce | TP=4 **會關掉**（PCIe 且 > 2 卡、沒有全連通）；TP=2 在 GPU0/1 這對有 P2P，**可能會開** | `reports/20261002_related_work.md`；vLLM 原始碼 [推論，建環境時實測確認] |
| bf16 | 不能用（無 tensor core 支援，極慢） | 全部改 fp16 |
| 現有腳本 | 要改：寫死 `/home/rogerlee`（31 處）、bf16、`import flux`、`launch.sh` 用 pixi + Flux 路徑 | `grep` 結果 |

## 4. `exclusive_guard.py` 在這台的問題

乾跑（只呼叫它的函式，不執行任何指令）結果在 `15_guard_dryrun_and_top.txt`、`16_guard_issues.txt`：

1. **使用者名稱被截斷**：`ps -eo user=` 把 `allenzhuang0117`（15 字）印成 `allenzh+`，守衛用 `u != ME` 比對，
   **自己的程序被當成別人的**。乾跑時自己的一個程序（60%）就被列為 other_user_cpu_hogs → 預檢會 ABORT；
   執行時 torchrun 的 worker 吃滿 CPU，也會被記為 FOREIGN_CPU → 每次都 CONTAMINATED。
   css-host-158 上的帳號名稱不超過 8 字，所以沒出事。
2. **CPU% 是整個生命期平均**：user_D 的 PID 2245467 已跑 2 天，top 瞬間 151.7%，`ps pcpu` 只有 0.2%，守衛抓不到。
   在共用機器上，長壽程序突然變忙是最常見的情況。
3. 守衛不看記憶體頻寬 / PCIe 流量。在 gpu1 這比 css-host-158 重要（見第 2 節）。

建議修法（建環境第一步，`common/measure/exclusive_guard_v2.py`，v1 加 WITHDRAWN 註記但保留）：
- 用 uid 比對（`ps -eo pid=,ppid=,uid=,...` 或 `user:64`），而且把被包住指令的子孫排除，不靠名字；
- CPU% 改成兩次 `/proc/<pid>/stat` 取樣的差值（每秒的瞬時值）；
- 門檻：其他使用者瞬時合計 > 1 核（可調）就記 FOREIGN_CPU；cadvisor 等 root 常駐程序照舊排除；
- 每次報告數字附「乾淨率」：在共用機器上 CLEAN 可能很難一次拿到，所以要能丟掉被污染的輪次、只重跑那幾段。

## 5. 適合做驗證計劃第 3 節的哪些項目

| 計劃項目 | 在 gpu1 能做嗎 | 說明 |
| --- | --- | --- |
| E3 原定（Flux 在新硬體上的表現 + 決策器 vs 簡單規則） | **不能** | Flux 不能跑；R1 / R3 依凍結規則退回 R2 |
| E3 縮小版 = E2 真機版：切法決策（TP + AR vs 序列平行 + NCCL）+ 模型重新校準 | **能** | 4 卡（TP=4，跨 QPI）與 2 卡（TP=2，同一對有 P2P）；fp16；Qwen2.5-32B / Llama-3-8B 的 block（32 GB 夠放 4 個隨機權重 block [推論]） |
| 成功標準 (a)：校準 ≤ 10 分鐘、block regret ≤ 2% | 能測 | 但決策空間小（只剩切法與 NCCL / cuBLAS 路徑），比 A100 上容易 |
| 成功標準 (b)：某環境簡單規則 regret ≥ 5% | 能測，但意義不同 | 這裡 R1 = R2 = vLLM 預設；只有「序列平行 + NCCL 在某些 M 快 ≥ 5%」時才成立 |
| E0 模型推演 | 能（不用 GPU） | 可先用 gpu1 的 PCIe 規格推演，決定縮小版值不值得做 |
| E1 本機背景塞車 | 不適合 | 本來就是 css-host-158 的項目 |

**我的建議**（請你決定）：
1. gpu1 只做「縮小版」：驗證決策器的校準流程能不能搬到 V100 + PCIe + fp16，以及 PCIe 上切法的切換點。
   結果可以支持「決策器可攜」，但**不能**證明「決策器比簡單規則好」。
2. 要回答核心問題，需要一台 **sm80 以上、沒有 NVLink（或 NVLink 較慢）** 的機器，例如 A100 PCIe / A10 / L40。
   如果 EE325 叢集的 gpu2–5 有不同的卡，值得問一下（Slurm 設定裡 gpu1–5 的 CPU / 記憶體相同，沒有 GPU 資訊）。
3. 不論選哪個，可以先做 E0（不用 GPU）。

## 6. 建環境估計（若核准縮小版）

| 步驟 | 估計時間 | 備註 |
| --- | --- | --- |
| `exclusive_guard_v2.py` + 自我測試 | 30 分鐘 | 先在無 GPU 的指令上測 |
| uv venv：python 3.11 + torch 2.6.0+cu124 + vLLM 0.8.5.post1 + transformers 4.51.3 | 15–30 分鐘 | 快取與 `TMPDIR` 指到 `/home`；`/` 只剩 9 GB |
| 腳本改版（v2）：路徑用 repo 相對 / 環境變數、fp16、Flux 可選、hw_profiles 用 `gpu1_tp*_*.json` | 2–4 小時 | 只動 `ws/fusion-dispatch/scripts/` 與新增檔案，不改 css-host-158 的檔案 |
| 錨點：PCIe 單向 / 雙向、同對 P2P vs 跨 QPI（peer copy 或自編 nvbandwidth） | 20 分鐘 | 先確認 ~12 GB/s 量級，再開始校準 |
| 校準 → 決策 push → 標準答案 → regret | 30–60 分鐘 GPU | 全程經守衛；有人用 GPU 就暫停 |

## 7. 新的坑（給 STATUS / JOURNAL；要不要寫進 CLAUDE.md 由 boss 裁決）

1. `exclusive_guard.py` 依使用者名稱比對，`ps` 會把超過 8 字的名稱截斷 → 自己被當成別人。
2. `exclusive_guard.py` 的 CPU% 是生命期平均，抓不到長壽程序的突發負載。
3. gpu1：Flux 只支援 sm80 / 89 / 90 與特定 SM 數（`op_registry.cu:39-51`），V100 不能用。
4. gpu1：`/` 與 `/tmp` 只剩 9 GB，大型下載 / 編譯暫存要放 `/home`（NAS）。
5. gpu1：P2P 只在 GPU0↔1、GPU2↔3；跨對經 host + QPI。
