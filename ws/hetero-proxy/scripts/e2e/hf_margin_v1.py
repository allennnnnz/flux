################################################################################
# hetero-proxy E2E diagnostic (2026-10-14): where a configuration's greedy tokens leave the HF reference, was it a
# near-tie? HF Qwen2.5-32B (one GPU, bf16) is teacher-forced on the prompt + the common prefix; at the first
# differing position we report HF's top-2 tokens, their logit gap, and whether the configuration chose HF's runner-up.
# Usage: pixi run --manifest-path pixi.toml ~/.venvs/hetero-proxy-hf/bin/python hf_margin_v1.py \
#            --reference <hf_reference.json> --check <check_v2.json> --out <json>
################################################################################
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_tp_v1 import model_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--check", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ref = {it["name"]: it for it in json.load(open(args.reference))["items"]}
    rows = json.load(open(args.check))
    model = AutoModelForCausalLM.from_pretrained(model_dir(), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0}).eval()
    out = []
    for row in rows:
        it = ref[row["name"]]
        for cfg, r in row["results"].items():
            d = r["identical_prefix"]
            if d >= len(it["hf_tokens"]):
                continue
            ids = torch.tensor([it["prompt_ids"] + it["hf_tokens"][:d]], device="cuda:0")
            with torch.inference_mode():
                lg = model(ids).logits[0, -1].float()
            top = torch.topk(lg, 2)
            rec = dict(name=row["name"], config=cfg, position=d, hf_top2=top.indices.tolist(),
                       hf_top2_logits=top.values.tolist(), gap=float(top.values[0] - top.values[1]),
                       config_token=r["tokens"][d], config_chose_runner_up=r["tokens"][d] == int(top.indices[1]))
            out.append(rec)
            print(f"[margin] {row['name']} {cfg}: diverges at {d}; HF top-2 logits {rec['hf_top2_logits']} "
                  f"(gap {rec['gap']:.3f}); config chose HF's runner-up: {rec['config_chose_runner_up']}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
