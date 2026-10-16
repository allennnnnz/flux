################################################################################
# hetero-proxy KV tier v2 (2026-10-16): HuggingFace reference for the KV-tier correctness check - real multi-turn
# conversations from ShareGPT (ShareGPT_V3_unfiltered_cleaned_split.json, already in the HF cache).
# v2: the file splits conversations into ~2k-token segments (ids <conv>_0, <conv>_1, ...; v1 therefore found no
# history longer than 2k). v2 reassembles every conversation whose segments are contiguous (0..n-1; "cleaned"
# dropped segments of 13391 of 50142 conversations, those are skipped) and prunes by characters before tokenizing.
# For each target history length (default ~2k / ~4k / ~8k tokens) the FIRST conversation in file order whose
# history (turns before some human turn k >= 2, ending with a gpt turn) has a chat-templated length in the window and
# whose new human message adds 20..400 tokens. Records per item:
#   prompt_ids     chat template of history + new message, with the generation prompt (Qwen2.5 default system prompt)
#   prefix_len P   length of the templated history alone; asserted to be an exact prefix of prompt_ids, so the KV of
#                  prompt_ids[:P] is what a serving system would have cached after the previous turn
#   hf_tokens      n_new greedy tokens (pure argmax over raw logits, KV cache; as hf_reference_v1.py)
#   hf_gaps        top-1 minus top-2 logit at every generated step (explains divergences at near-ties)
#   hf_first_logits_top  top-20 (id, logit) of the first step
# Qwen2.5-32B-Instruct bf16, SDPA, ONE GPU (65 GB).
# Usage: pixi run --manifest-path pixi.toml ~/.venvs/hetero-proxy-hf/bin/python \
#            ws/hetero-proxy/scripts/kvtier/hf_reference_kv_v2.py --out <dir> [--gpu 0] [--n_new 64]
################################################################################
import argparse
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2e"))
from qwen_tp_v1 import model_dir  # noqa: E402

SHAREGPT = "~/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/*/" \
           "ShareGPT_V3_unfiltered_cleaned_split.json"
ROLE = {"human": "user", "gpt": "assistant"}


def conversations():
    """Reassembled conversations in file order of their first segment: (id, turns)."""
    data = json.load(open(glob.glob(os.path.expanduser(SHAREGPT))[0]))
    groups, order = {}, []
    for c in data:
        base, _, suf = c["id"].rpartition("_")
        if not suf.isdigit():
            base, suf = c["id"], "0"
        if base not in groups:
            groups[base] = []
            order.append(base)
        groups[base].append((int(suf), c["conversations"]))
    for base in order:
        segs = sorted(groups[base], key=lambda s: s[0])
        if [s for s, _ in segs] != list(range(len(segs))):
            continue
        yield base, [t for _, ts in segs for t in ts]


def select(tok, windows, t_lo=20, t_hi=400):
    found = {}
    lo_all, hi_all = min(lo for lo, _ in windows.values()), max(hi for _, hi in windows.values())
    for cid, turns in conversations():
        if len(turns) < 3 or any(t.get("from") != ("human" if j % 2 == 0 else "gpt") for j, t in enumerate(turns)):
            continue
        msgs = [{"role": ROLE[t["from"]], "content": t["value"]} for t in turns]
        chars = 0
        for k in range(2, len(msgs), 2):  # msgs[k] is a human turn, msgs[:k] ends with a gpt turn
            chars += len(msgs[k - 2]["content"]) + len(msgs[k - 1]["content"])
            if chars < 1.5 * lo_all:  # >= 1.5 characters per token on this data, so P < lo_all for sure
                continue
            hist = tok.apply_chat_template(msgs[:k], tokenize=True)
            P = len(hist)
            want = [w for w in windows if w not in found and windows[w][0] <= P <= windows[w][1]]
            if not want:
                if P > max(hi for _, hi in windows.values()):
                    break
                continue
            full = tok.apply_chat_template(msgs[:k + 1], tokenize=True, add_generation_prompt=True)
            if not (t_lo <= len(full) - P <= t_hi) or full[:P] != hist:
                continue
            found[want[0]] = dict(name=want[0], conv_id=cid, turn=k, prefix_len=P, prompt_ids=full)
            break
        if len(found) == len(windows):
            break
    return [found[w] for w in windows if w in found]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_new", type=int, default=64)
    ap.add_argument("--windows", default="hist2k:1500:2500,hist4k:3500:4500,hist8k:7000:9000")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = model_dir()
    tok = AutoTokenizer.from_pretrained(d)
    windows = {w.split(":")[0]: (int(w.split(":")[1]), int(w.split(":")[2])) for w in args.windows.split(",")}
    items = select(tok, windows)
    for it in items:
        print(f"[select] {it['name']}: conv {it['conv_id']} turn {it['turn']}, history {it['prefix_len']} tokens, "
              f"new {len(it['prompt_ids']) - it['prefix_len']} tokens", flush=True)
    dev = torch.device(f"cuda:{args.gpu}")
    model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": args.gpu})
    model.eval()
    for it in items:
        ids = torch.tensor([it["prompt_ids"]], device=dev)
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            logits, past = out.logits[:, -1].float(), out.past_key_values
            top = torch.topk(logits[0], 20)
            it["hf_first_logits_top"] = [[int(i), float(v)] for i, v in zip(top.indices, top.values)]
            toks, gaps = [], []
            for s in range(args.n_new):
                t2 = torch.topk(logits[0], 2).values
                gaps.append(float(t2[0] - t2[1]))
                nxt = logits.argmax(-1)
                toks.append(int(nxt))
                if s == args.n_new - 1:
                    break
                out = model(nxt[:, None], past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1].float()
        it["hf_tokens"], it["hf_gaps"], it["hf_text"] = toks, gaps, tok.decode(toks)
        print(f"[hf] {it['name']}: min gap {min(gaps):.3f}  {it['hf_text'][:100]!r}", flush=True)
    with open(os.path.join(args.out, "hf_reference_kv_v2.json"), "w") as f:
        json.dump(dict(model=d, n_new=args.n_new, torch=torch.__version__,
                       transformers=__import__("transformers").__version__, items=items), f, indent=1,
                  ensure_ascii=False)


if __name__ == "__main__":
    main()
