Supersedes: 無（接 `20261030_fs_validation_result.md`）

# 選項 FS 第 3 步：找出 Flux 重疊差距的原因並做原型（計劃與事前登記，2026-10-30，跑之前提交）

已知（`20261030_fs_validation_result.md`）：M ≤ 2048 時 Flux 最佳設定仍是理想的 1.35×；不用 Flux 時 copy engine 搬運與 cuBLAS 同時跑互不拖慢（FS1d）；
Flux 的 all-gather 與 Flux 的 `gemm_only` 同時跑時，all-gather 變慢 2.8×（FS1c）。

## 3a：不重編，拆原因

FS1c 與 FS1d 有兩個差別：(1) 旁邊跑的 GEMM：Flux 的 `gemm_only`（每個 tile 開頭用 `ld.acquire.sys` 等旗標，`include/flux/cuda/system_barrier.hpp:41`）vs cuBLAS；
(2) 搬運路徑：Flux 的 all-gather（開頭跨卡屏障 kernel、拉取、每塊 `cuStreamWriteValue` 寫旗標）vs 單純推送。

- **E1**（torchrun、Flux）：同一個 Flux all-gather，旁邊分別跑 Flux `gemm_only` 與 cuBLAS（同形狀），同一程序內交錯、分開計時 all-gather 與 GEMM。
  - 預測：換成 cuBLAS 後，all-gather 變慢的中位數 ≤ 1.3×（M ≤ 2048）→ 原因在 Flux 的 GEMM kernel（每 tile 的系統層級等待）。
- **E2**（只在 E1 不成立時做；單一程序、不用 Flux）：依序加入 Flux 的步驟——推送 → 拉取 → 拉取＋每塊寫旗標 → ＋開頭屏障——旁邊跑 cuBLAS，看哪一步開始變慢。

## 3b：重編 Flux（依 D-006：改前打 git tag `pre-fs-mod`、備份目前的 `build/lib` 與 `python/flux/lib`、改後做 bitwise 驗證）

- 依 3a 找到的原因做最小修改。例：若是 GEMM 每 tile 的系統層級等待 → 改成 GPU 層級（旗標由本卡的 stream 寫在本卡記憶體，推論 GPU 層級即可看見；要驗證正確性）。
- kernel 內時間戳（`%globaltimer`）：每個 CTA 每個 tile 的開始、等待結束、完成時間；資料每塊到達時間。
- 記錄 `build.sh --arch 80` 的編譯時間。

## 判定（事前登記，沿用第 3 步規則）

- **確定做 FS**：找到佔差距一半以上的原因，且原型在 M = 512–2048（6 種層形狀）讓融合時間的中位數比現在的 Flux（預設設定）快 ≥ 1.15×，結果逐位元或 allclose 正確。
- **不做**：3a、3b 都找不到可修改的原因，或原型 < 1.05×。
- 介於之間交使用者。
