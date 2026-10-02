Supersedes: （無）

# F4 可行性：本機能否做 vLLM 端到端測試

日期：2026-10-02 · 作者：fusion-dispatch（boss session 兼 worker）

## 結論

**可以。** 原版 vLLM 0.8.5（V1 引擎、torch.compile、piecewise CUDA graph）在本機以 Llama-3-70B 架構、
dummy 隨機權重、TP=8 跑通 `vllm bench latency`，數字穩定。整合決策器所需的元件都已到位：

- Flux 與 vLLM 可在同一環境共存（驗證輪 V3 已在 vLLM 環境內跑 Flux op）；
- 支援外部模型註冊（`ModelRegistry.register_model`）；
- 有 `vllm bench {latency, throughput, serve}`。

## 量測（`results/f4_feasibility/`）

指令：`vllm bench latency --model ws/fusion-dispatch/models/llama3-70b-dummy --tokenizer ws/fusion-dispatch/models/qwen25-tokenizer
--load-format dummy --tensor-parallel-size 8 --dtype bfloat16 --max-model-len 4096 --input-len 1024 --output-len 128
--batch-size 64 --num-iters-warmup 2 --num-iters 5`。以 `common/measure/exclusive_guard.py` 包住，結果 CLEAN。

| 項目 | 值 |
| --- | --- |
| 單次請求批（64 條 × 輸入 1024 + 輸出 128）平均延遲 | **3.591 s**（5 次 3.583–3.597，p90 3.596） |
| 引擎初始化（profile + KV cache + warmup） | 142 s |
| torch.compile（一般形狀） | 約 56 s / rank（之後有 cache） |
| CUDA graph capture（67 個桶，1–512） | 36 s，1.05 GiB |
| KV cache 容量 | 約 125 萬 token（4096 token 請求可並行約 305 條） |

## 遇到的環境問題與處理（已寫進 `scripts/setup_vllm_venv.sh`）

1. vLLM V1 即使加 `--skip-tokenizer-init` 仍會載入 tokenizer（`vllm/v1/engine/llm_engine.py:77`）。
   → 用 Qwen2.5 的 tokenizer（Apache-2.0，詞彙 151,665 ≥ Llama-3 的 128,256）。輸入是隨機 token，不影響速度。
2. uv 預設裝到 transformers 5.17，vLLM 0.8.5 不相容（`Qwen2Tokenizer has no attribute all_special_tokens_extended`）。
   → 釘選 `transformers==4.51.3`。
3. 模型設定檔 `models/llama3-70b-dummy/config.json` 只有架構，沒有權重（dummy 模式隨機初始化，速度與真權重相同）。

## 整合計劃（下一步，尚未開始）

寫一個自訂模型 `FluxLlamaForCausalLM`，以外部註冊方式掛進 vLLM，不改 vLLM 原始碼：

- 沿用 vLLM 的 attention、KV cache、排程、取樣；
- 每個 decoder layer 的線性層與通訊交給兩層決策器：TP + AllReduce（vLLM 預設），或 SP + 逐層選 Flux / NCCL；
- SP 需要 M 補到 8 的倍數，模型入口切分、出口收集。

待解決的技術點：

1. **torch.compile 相容性**：V1 會 compile 模型，Flux op 是不透明的 C++ 呼叫，可能造成 graph break。
   備案：用 V0 引擎（整個 forward 錄 CUDA graph），或把 Flux op 包成 torch custom op。
2. **graph capture 時的路徑過濾**：capture 期間決策器自動跳過 AGKernel（F0.4）。
3. **長 graph 中 c10d NCCL 變慢**（V4）：SP 的 NCCL 路徑改用 vLLM 的 pynccl。
4. **V11 的卡死**：整合後做長時間壓力測試。

比較對象：原版 vLLM、決策器版、永遠 SP + Flux 版。
指標：`vllm bench latency`（decode / prefill 各 batch）、`vllm bench serve`（真實長度分佈的請求流：throughput、TTFT、TPOT）。
