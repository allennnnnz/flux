################################################################################
# hetero-proxy X1e (2026-10-28): what does a CUDA graph holding one NCCL all-reduce contain? X1d smoke
# (results/xpcie/x1d_smoke): fence-free LL-style NVLink messages and kernels that end after remote stores are NOT slowed by
# background copies, yet NCCL LL all-reduces in graphs are (+182 us each under D2H on every GPU, x1_proto_LL; LL128
# +180 us: a per-operation cost, not per byte). Candidate: nodes NCCL adds to the captured graph besides its kernel
# (event / memset / host nodes, cross-stream joins that the GPU executes as semaphore operations).
# Same setup as x1_dispatch_v1.py (graph-only communicator, eager warm-up on the default one); captures
# [all_reduce(16x5120 bf16)] and [stamp, all_reduce] x 2, writes the graphs as Graphviz dot (cudaGraphDebugDotPrint via
# torch) and NCCL's INFO log (NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH: chosen algorithm / protocol).
# Usage (repo root, exclusive guard):
#   env -u CUDA_DEVICE_MAX_CONNECTIONS NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING CUDA_MODULE_LOADING=EAGER \
#     pixi run --manifest-path pixi.toml ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
#     ws/hetero-proxy/scripts/xpcie/x1e_graphdump_v1.py --out <dir>
################################################################################
import argparse
import os
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from x1_dispatch_v1 import stamp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    g_graph = dist.new_group(list(range(world)))
    one = torch.zeros(1, device=dev)
    dist.all_reduce(one, group=g_graph)
    torch.cuda.synchronize()
    small = torch.randn(16, 5120, device=dev, dtype=torch.bfloat16)
    buf = torch.zeros(64, dtype=torch.int64, device=dev)

    def fn_ar(group):
        dist.all_reduce(small, group=group)

    def fn_mix(group):
        for i in range(2):
            stamp(buf, i)
            dist.all_reduce(small, group=group)

    for name, fn in (("ar1", fn_ar), ("stamp_ar2", fn_mix)):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(None)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        g.enable_debug_mode()
        with torch.cuda.graph(g):
            fn(g_graph)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        if rank == 0:
            g.debug_dump(os.path.join(args.out, f"{name}_rank0.dot"))
            print(f"[x1e] dumped {name}", flush=True)
    dist.barrier()
    sys.stdout.flush()
    os._exit(0)  # destroy_process_group() hangs after graph-captured NCCL (qwen_tp_v5.py)


if __name__ == "__main__":
    main()
