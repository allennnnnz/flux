################################################################################
# hetero-proxy PCIe-only check (2026-11-02): can the 8x A100 NVSwitch box make GPU-to-GPU traffic avoid NVLink without
# sudo? NCCL transport is chosen by env: default = P2P over NVLink; NCCL_P2P_DISABLE=1 -> SHM through pinned host memory
# (PCIe both ways); + NCCL_SHM_DISABLE=1 -> NET (IB loopback through the NICs on each PCIe switch). This script times
# all_reduce / all_gather / reduce_scatter (bf16) per size, rank-max median of CUDA-event times, busbw in the
# nccl-tests convention; the transport actually used is read from NCCL_DEBUG=INFO lines in stdout.
# Anchors (params.json): NVLink pair 270 GB/s; PCIe one GPU 21-23 GB/s one way; same-switch pair 25 total;
# simultaneous send+receive per GPU 5.9-6.5 GB/s (Phase 0 / I2); NIC read from GPU ~7 GB/s (E4).
# Usage (repo root, guard): NCCL_P2P_DISABLE=1 pixi run --manifest-path pixi.toml python -m torch.distributed.run
#   --nproc_per_node 8 ws/hetero-proxy/scripts/pcieonly/pcie_only_check_v1.py --out <dir> --tag shm
################################################################################
import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

SIZES_MIB = (1, 16, 128)
OPS = ("all_reduce", "all_gather", "reduce_scatter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    res = {}
    for mib in SIZES_MIB:
        n = mib * 2 ** 20 // 2
        x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        out_ag = torch.empty(n * world, dtype=torch.bfloat16, device="cuda")
        out_rs = torch.empty(n // world, dtype=torch.bfloat16, device="cuda")
        for op in OPS:
            def call():
                if op == "all_reduce":
                    dist.all_reduce(x)
                elif op == "all_gather":
                    dist.all_gather_into_tensor(out_ag, x)
                else:
                    dist.reduce_scatter_tensor(out_rs, x)
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.iters):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                dist.barrier()
                a.record()
                call()
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
            t = torch.tensor([statistics.median(ts)], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            ms = t.item()
            nbytes = mib * 2 ** 20 * (world if op == "all_gather" else 1)  # all_gather: output size
            factor = 2 * (world - 1) / world if op == "all_reduce" else (world - 1) / world
            res[f"{op}|{mib}MiB"] = dict(ms=ms, busbw_GBps=nbytes / (ms / 1e3) * factor / 1e9)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        json.dump(dict(tag=args.tag, env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                       nccl=torch.cuda.nccl.version(), results=res),
                  open(os.path.join(args.out, f"check_{args.tag}.json"), "w"), indent=1)
        for k, v in res.items():
            print(f"[{args.tag}] {k:24s} {v['ms']:8.3f} ms  busbw {v['busbw_GBps']:7.1f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
