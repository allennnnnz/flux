"""FS step 3b (2026-10-30): bitwise reference for Flux AGKernel.forward before/after modifying Flux (D-006).
For each (layer, M): deterministic inputs per rank, one default forward, sha256 of the output bytes per rank.
Run once with the current build (--tag before) and again after rebuilding (--tag after_default, FLUX_FS_ARRIVAL unset);
the hashes must be identical. Launch with ./launch.sh.
"""
import argparse
import hashlib
import json
import os

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

DT = torch.bfloat16
LAYERS = {"G-FC1": (49152, 12288), "G-QKV": (36864, 12288), "L-GU": (57344, 8192), "L-QKV": (10240, 8192),
          "Q-GU": (55296, 5120), "Q-QKV": (7168, 5120)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--Ms", default="512,1024,2048,4096")
    args = ap.parse_args()
    tp = initialize_distributed()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)
    rank, W = tp.rank(), tp.size()
    nnodes = flux.testing.NNODES()
    Ms = [int(m) for m in args.Ms.split(",")]
    opt = flux.AllGatherOption()
    opt.mode = flux.AGRingMode.All2All
    opt.use_read = True
    opt.use_cuda_core_local = False
    opt.use_cuda_core_ag = False
    opt.fuse_sync = False
    opt.input_buffer_copied = False
    res = {}
    for li, (name, (N, K)) in enumerate(LAYERS.items()):
        n = N // W
        op = flux.AGKernel(tp, nnodes, max(Ms), n, K, DT, output_dtype=DT)
        g = torch.Generator(device="cuda").manual_seed(1000 * li + rank)
        w = (torch.randn((n, K), device="cuda", generator=g) * 0.01).to(DT)
        for M in Ms:
            x = torch.randn((M // W, K), device="cuda", generator=g).to(DT)
            out = torch.empty((M, n), dtype=DT, device="cuda")
            op.forward(x, w, output=out, transpose_weight=False, all_gather_option=opt)
            torch.cuda.synchronize()
            res[f"{name}_{M}"] = hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    allres = [None] * W
    dist.all_gather_object(allres, res)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        json.dump({"tag": args.tag, "env_FLUX_FS_ARRIVAL": os.environ.get("FLUX_FS_ARRIVAL"), "ranks": allres},
                  open(os.path.join(args.out, f"hash_{args.tag}.json"), "w"), indent=1)
        print("hashed", len(res), "cases x", W, "ranks")
    dist.barrier()
    os._exit(0)  # do not destroy Flux ops (CLAUDE.md trap)


if __name__ == "__main__":
    main()
