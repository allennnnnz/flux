################################################################################
# hetero-proxy E2E (2026-10-14): HuggingFace reference for the TP engine's correctness check.
# Qwen2.5-32B-Instruct, bf16, SDPA, ONE GPU (65 GB fits an 80 GB A100). Greedy decoding written out by hand (pure
# argmax over the raw logits, KV cache): `generate()` would apply generation_config.json (repetition_penalty 1.05) and
# change the tokens. Records per prompt: chat-templated prompt ids, n_new greedy tokens, decoded text, and the top-20
# (id, logit) of the first step.
# Usage: pixi run --manifest-path pixi.toml ~/.venvs/hetero-proxy-hf/bin/python \
#            ws/hetero-proxy/scripts/e2e/hf_reference_v1.py --out <dir> [--gpu 0] [--n_new 64]
################################################################################
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_tp_v1 import model_dir  # noqa: E402

PROMPTS = [
    ("en_explain", "Explain in three sentences why the sky is blue."),
    ("zh_explain", "用簡單的中文解釋什麼是 PCIe，以及它和 NVLink 有什麼不同。"),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("math", "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Show the steps."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_new", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = model_dir()
    tok = AutoTokenizer.from_pretrained(d)
    dev = torch.device(f"cuda:{args.gpu}")
    model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": args.gpu})
    model.eval()
    items = []
    for name, text in PROMPTS:
        msgs = [{"role": "user", "content": text}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            logits, past = out.logits[:, -1].float(), out.past_key_values
            top = torch.topk(logits[0], 20)
            first_top = [[int(i), float(v)] for i, v in zip(top.indices, top.values)]
            toks = []
            nxt = logits.argmax(-1)
            for s in range(args.n_new):
                toks.append(int(nxt))
                if s == args.n_new - 1:
                    break
                out = model(nxt[:, None], past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1].float().argmax(-1)
        items.append(dict(name=name, prompt=text, prompt_ids=ids[0].tolist(), hf_tokens=toks,
                          hf_text=tok.decode(toks), hf_first_logits_top=first_top))
        print(f"[hf] {name}: {tok.decode(toks)[:120]!r}", flush=True)
    with open(os.path.join(args.out, "hf_reference.json"), "w") as f:
        json.dump(dict(model=d, n_new=args.n_new, torch=torch.__version__,
                       transformers=__import__("transformers").__version__, items=items), f, indent=1,
                  ensure_ascii=False)


if __name__ == "__main__":
    main()
