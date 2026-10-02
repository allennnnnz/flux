# Phase 0 Findings — 結論與設計規則

**唯讀。** Phase 0 於 2026-09-23 結案。完整記錄、撤回表、方法細節在
`PHASE0_STATUS.md`；原始資料與腳本在 `experiments/2026-09-23-phase0-a100-nvlink-pcie/`。
本檔只放後續 workstream 需要**消費**的東西。

每條事實標示狀態：**[量測]** = 有原始輸出且經交叉驗證；**[推論]** = 由量測推出、
尚未直接驗證，附驗證方法。

---

## 1. 決策

**A100 上不做 NVLink + PCIe 雙通道。**（DECISIONS D-001）

三條獨立依據：
1. NVLink 217.66 GB/s/卡 vs. 主機中轉 5.3 GB/s/卡，比值 41。第二條通道的理論上限
   ≈ +2.6%（若兩路徑完美可加）。**[量測]**
2. 實測掃描 α ∈ {0.01, …, 0.12}：**每一個 α 都是淨損失且單調**（−3% 到 −80%）。
   有無 switch-aware ring 皆同。兩路徑不可加：staging 是 D2H→H2D 串行鏈，
   21 MB 要 3.79 ms，與同輪 NVLink 搬 0.98 GiB 的 4.96 ms 同量級。**[量測]**
3. 在 N=49152 通訊已被 GEMM 藏住：AG 快 17% 只換到端到端 0.9%。**[量測]**

什麼會改變這個決策：兩條路徑頻寬在同一數量級（無 NVLink 的機器、PCIe-only 晶片、
跨節點）。這正是異質計算的情境——所以 Phase 0 的結論不是「PCIe 沒用」，是
「**PCIe 在 A100 內部沒用，但它的行為特性是設計異質後端的輸入**」。

## 2. 機器事實

```
        ┌────────── NVSwitch（所有 GPU 對 NV12）──────────┐
     [GPU0][GPU1][GPU2][GPU3]        [GPU4][GPU5][GPU6][GPU7]
        │    │     │    │               │    │     │    │
      ┌─┴────┴┐ ┌──┴────┴┐          ┌───┴────┴┐ ┌──┴────┴┐
      │PEX880x│ │PEX880x │          │PEX880x  │ │PEX880x │   4 顆 PCIe Gen4 switch
      └───┬───┘ └───┬────┘          └────┬────┘ └───┬────┘
       Gen4 x16  Gen4 x16              Gen4 x16  Gen4 x16     每顆一條上行
          └─────────┘                     └─────────┘
            NUMA 0                          NUMA 1
```

### 2.1 拓撲 [量測]

- GPU **兩兩共用**一條 PCIe switch 上行：0/1、2/3、4/5、6/7。由對照實驗確認：
  同 switch 兩卡 H2D 合計 25.5 GB/s（≈ 一條上行），不同 switch 兩卡 42.9 GB/s（≈ 兩倍單卡）。
- root port 與 switch 上行皆 16.0 GT/s ×16。switch 內部鏈路規格因無 sudo 無法確認。
- `nvidia-smi topo -m` 的 `NV12` 描述 NVLink，**對 PCIe 拓撲沒有任何資訊**。

### 2.2 NVLink [量測]

| 情境 | 每卡 |
| --- | ---: |
| 8 卡全互連（每卡向 7 對端送出） | **217.66 GB/s** |
| 單對 peer copy | 270.8 GB/s |
| Flux AllGather All2All pull，每卡入向 | 188.2 GB/s |

- 每卡 egress **一筆 copy 就飽和**；nsys 峰值併發 = 8（每卡 1 筆）。多開 stream 無效。
- 每卡每輪 < 1 GiB 時受 per-copy overhead 支配：64 MiB → 34.7、256 MiB → 129.6、
  1024 MiB → 210.9 GB/s/卡。

### 2.3 PCIe 單向 [量測，nvbandwidth 交叉驗證至 0.2%]

| 情境 | H2D | D2H |
| --- | ---: | ---: |
| 單卡 | 21.795 | 23.924 |
| 8 卡同時，NUMA local，每卡 | 12.647 | 13.195 |
| 8 卡同時，NUMA remote，每卡 | 10.302 | 9.448 |

NUMA remote 綁定損失 18–28%。

### 2.4 PCIe 雙向：不對稱競爭 [量測]

同一張 GPU 同時做 H2D 與 D2H 時（nsys 逐筆分類）：

| 方向 | 無對向流量 | 有對向流量 |
| --- | ---: | ---: |
| H2D | 22.01 | **6.46**（÷3.4） |
| D2H | — | 22.52（不受影響） |

nvbandwidth 獨立量到雙向 H2D 8.88 / D2H 22.13，同一結構。

**[推論]** 競爭發生在 GPU 端點或其 switch 上行，而非主機記憶體控制器。
驗證方法：GPU0 只 H2D + GPU2 只 D2H（不同 switch）並行，對照 GPU0 只 H2D +
GPU1 只 D2H（同 switch）。若前者 H2D 不降、後者降 → 競爭在 switch 上行；若兩者都不降 →
競爭只在單卡端點；若兩者都降 → 在主機側。這是 `ws/hetero-proxy` 步驟 1a。

### 2.5 主機中轉（staging） [量測]

雙緩衝 producer/consumer，`cuStreamWriteValue32` / `cuStreamWaitValue32`，NUMA-local pinned。

| 配對 | 拓撲 | 最佳 GB/s | 對單向上限 21.8 |
| --- | --- | ---: | ---: |
| 0 → 1 | 同 switch | 13.9 | 64% |
| **0 → 2** | 不同 switch，同 NUMA | **22.2** | ~100% |
| 3 → 4、0 → 4 | 跨 NUMA | 18.6 | 85% |
| 8 卡 ring，每卡 | 每卡同時收發 | 5.87 | 對競爭態 H2D 6.46 的 **91%** |

- 單配對：來源只 D2H、目的只 H2D，**無 GPU 同時收發**，上限是單向 min(H2D, D2H)。
- ring：**每卡都同時收發**，上限退回競爭態 H2D 6.46。switch 配對只是二階效應
  （switch-aware ring 5.20 vs naive 5.35，無差）。
- chunk 大小：ring 以 16 MiB 最佳，1 MiB 崩到 18.7 GB/s 合計；單配對 1 MiB 仍有 13.6。

### 2.6 Flux 自身特性 [量測]

| 事實 | 數字 |
| --- | --- |
| Flux AllGather vs NCCL（M=4096, K=12288, world 8） | All2All pull 0.468 ms vs NCCL 0.569 ms，**快 22%**；在 M=1024–16384 全程快 13–22% |
| Ring2D（Phase 0 原始量測全用的模式） | 0.562 ms，≈ NCCL，最慢之一 |
| AG+GEMM overlap efficiency，bf16，All2All | N=4096：**~2%**；N=8192：76%；N=49152：60% |
| N=4096 的絕對數字 | overlap 0.736 / GEMM-only 0.278 / comm 0.468 / 完全串行 0.746 ms |
| AG 模式對端到端的影響 | N=4096 差 10.7%、N=8192 差 9.9%、N=49152 差 0.9% |

**N=4096 幾乎零重疊是目前最大的已知改善空間**（理想下界 0.468，實測 0.736，
差 36%），也是異質計算的前置問題——如果 Flux 在 comm-bound 形狀下連 NVLink 都藏不住，
換成慢 40 倍的 PCIe 只會更糟。這是 `ws/diag-overlap`。

## 3. 設計規則（給 hetero-proxy 與後續）

由上述事實直接推出。每條標示依據。

1. **跨 PCIe 流量盡量單向。** 一張 GPU 同時收發時 H2D 掉到 6.5 GB/s（2.4）。
   閘道 GPU 的收與送若無法避免同時，預期 H2D 側只有單向的 30%。
2. **生產者與消費者放在不同 switch。** 同 switch 配對 13.9 vs 不同 switch 22.2 GB/s（2.5）。
   在這台機器上：0/1、2/3、4/5、6/7 各為一組，配對時跨組。
3. **能跨 NUMA 就別跨。** 跨 NUMA staging 18.6 vs 同 NUMA 22.2（2.5）；8 卡 remote 綁定損 18–28%（2.3）。
   若閘道必須跨 NUMA，pinned buffer 綁在**來源** GPU 的 node。
4. **staging chunk 16 MiB 起跳**；多對並行時不要低於 4 MiB（2.5）。
5. **不要指望 PCIe 與 NVLink 相加。** 它們的頻寬比是 41 倍，且 staging 是串行鏈（第 1 節）。
   異質裝置的通訊要靠**隱藏**（與計算重疊）而不是靠**分流**。
6. **先修好 comm-bound 形狀的 overlap，再談異質。**（2.6）
7. **A100 代理 PCIe-only 裝置時，關 peer access、所有跨邊界流量顯式 D2H + H2D**，
   並以 NVLink 計數器確認代理流量為零；否則量到的是 NVLink。
8. **能力分級的第 0 級（算子級同步、經 host）應先用 2.4 的競爭態數字建模**，
   不是用 nominal PCIe 頻寬——否則會高估 3 倍。

## 4. 錨點數字（進 `common/cost_model/params.json`）

任何新量測寫入文件前先對照這些。偏差 > 20% 且無解釋 → 先懷疑量測。

| 錨點 | 值 | 來源 |
| --- | ---: | --- |
| 單對 NVLink peer copy | 270.8 GB/s | D_CORRECTION D.0 |
| 8 卡全互連 NVLink，每卡 | 217.66 GB/s | D_CORRECTION D.1 |
| Flux All2All pull AG，每卡入向 | 188.2 GB/s | E_CORRECTION E.1 |
| PCIe 單卡單向 H2D / D2H | 21.795 / 23.924 GB/s | B_REDO B.5；nvbandwidth 21.82 / 23.99 |
| PCIe 競爭態 H2D | 6.46 GB/s | BIDIR_CORRECTION X.3 |
| 一條 switch 上行（兩卡合計） | ~25.5 GB/s | B_REDO B.4 |
| staging 最佳單配對 | 22.2 GB/s | BIDIR_CORRECTION X.5 |
| staging ring，每卡 | 5.87 GB/s | CDE_REPORT C |
| NCCL all_gather，M=4096 K=12288 world 8 | 0.569 ms | E_CORRECTION E.1 |

## 5. Phase 0 的方法教訓（已寫入 CLAUDE.md 第 5 節）

四個缺陷、一個模式：**基準拿錯，證據在自己輸出裡。**

| 缺陷 | 一句話 | 攔截它的規則 |
| --- | --- | --- |
| Flux 基準其實是 NCCL | 同一段 NCCL 在三種 ring mode 下讀出 0.50/0.65/0.70 | 基準獨立；自我一致 |
| NVLink 慢 5.7 倍 | 跨裝置 copy 放錯 stream；38 GB/s 沒人對照 270 | 錨點對照 |
| H2D = D2H | 同一運算式印兩次；自己的 event 欄位差 1.75 倍被丟掉 | 自我一致；效率 > 100% 停 |
| ECT 出負值 | 兩個分別平均的迴圈相減 | 差值交錯量測 |
