################################################################################
# fusion-dispatch E4 anchor: NCCL all_gather / reduce_scatter / all_reduce across css-host-158 + css-host-159
# (RoCE), to check the cross-node bandwidth against the NIC line rate (ib_write_bw) before any model fit
# (CLAUDE.md 5.1.2). Launch with run_xnode.sh (2 nodes x nproc_per_node).
# Per size: 5 warm-up, --iters timed calls, each one between CUDA events after a barrier; per call the
# rank-max time; report the median. busbw: AG / RS = (n-1)/n * total / t, AR = 2(n-1)/n * size / t
# (nccl-tests convention). Also prints NCCL's transport choice (NCCL_DEBUG=INFO lines are in node*.out).
# Usage: run_xnode.sh <out> <nproc_per_node> pixi ws/fusion-dispatch/scripts/xnode_nccl_anchor_v1.py --out <out>
################################################################################
import argparse
import csv
import os
import statistics

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sizes_mib", default="1,4,16,64,256,1024")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    host = os.uname().nodename
    rows = []
    for coll in ("all_gather", "reduce_scatter", "all_reduce"):
        for mib in [int(x) for x in a.sizes_mib.split(",")]:
            total = mib * 2 ** 20  # bytes of the full (gathered / reduced) tensor
            n_el = total // 2  # fp16 elements
            shard = n_el // world
            if coll == "all_gather":
                inp = torch.randn(shard, device="cuda", dtype=torch.float16)
                out = torch.empty(shard * world, device="cuda", dtype=torch.float16)
                fn = lambda: dist.all_gather_into_tensor(out, inp)  # noqa: E731
            elif coll == "reduce_scatter":
                inp = torch.randn(shard * world, device="cuda", dtype=torch.float16)
                out = torch.empty(shard, device="cuda", dtype=torch.float16)
                fn = lambda: dist.reduce_scatter_tensor(out, inp)  # noqa: E731
            else:
                buf = torch.randn(n_el, device="cuda", dtype=torch.float16)
                fn = lambda: dist.all_reduce(buf)  # noqa: E731
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.iters):
                dist.barrier()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                t = torch.tensor([e0.elapsed_time(e1)], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                ts.append(t.item())
            med = statistics.median(ts)
            factor = 2 * (world - 1) / world if coll == "all_reduce" else (world - 1) / world
            busbw = factor * total / (med * 1e-3) / 1e9
            rows.append([coll, world, mib, f"{med:.4f}", f"{min(ts):.4f}", f"{max(ts):.4f}", f"{busbw:.2f}"])
            if rank == 0:
                print(f"[anchor] {coll:<15} world={world:<3} {mib:>5} MiB  median {med:8.3f} ms  busbw {busbw:7.2f} GB/s",
                      flush=True)
            del fn
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"nccl_anchor_w{world}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["collective", "world", "size_mib_total", "rankmax_median_ms", "min_ms", "max_ms", "busbw_GBps"])
            w.writerows(rows)
        print(f"[anchor] host {host} wrote {a.out}/nccl_anchor_w{world}.csv", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
