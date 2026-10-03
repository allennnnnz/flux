################################################################################
# fusion-dispatch E4 smoke: Flux's sm80 multi-node GEMM+RS (flux.GemmRS_multinode, python/flux/gemm_rs_sm80.py:
# fused GemmRS inside each node + NCCL send/recv between nodes) across css-host-158 + css-host-159.
# Correctness only: output vs torch reference (matmul then NCCL reduce_scatter over the whole TP group).
# Usage: run_xnode.sh <out> <ppn> pixi ws/fusion-dispatch/scripts/xnode_flux_rs_smoke_v1.py M N K
################################################################################
import os
import sys

import torch
import torch.distributed as dist

import flux

M, N, K = (int(x) for x in sys.argv[1:4])
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local_world = int(os.environ["LOCAL_WORLD_SIZE"])
nnodes = world // local_world
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
tp = dist.new_group(list(range(world)), backend="nccl")
flux.init_flux_shm(tp)
torch.manual_seed(rank)
k_local = K // world
x = (torch.randn(M, k_local, device="cuda") * 0.1).to(torch.bfloat16)
w = (torch.randn(N, k_local, device="cuda") * 0.1).to(torch.bfloat16)
op = flux.GemmRS_multinode(tp, nnodes, M, N, torch.bfloat16, transpose_weight=False, fuse_reduction=False)
out = op.forward(x, w, None)
torch.cuda.synchronize()
full = (x.float() @ w.float().t()).to(torch.bfloat16)
ref = torch.empty(M // world, N, device="cuda", dtype=torch.bfloat16)
dist.reduce_scatter_tensor(ref, full, group=tp)
torch.cuda.synchronize()
err = (out.float() - ref.float()).abs().max().item()
scale = ref.float().abs().max().item()
ok = tuple(out.shape) == tuple(ref.shape) and err <= 2e-2 * max(scale, 1e-3)
t = torch.tensor([0 if ok else 1], device="cuda")
dist.all_reduce(t, group=tp)
print(f"[rs_smoke] rank {rank} host {os.uname().nodename} out {tuple(out.shape)} max_abs_err {err:.4g} ref_max {scale:.4g} "
      f"{'OK' if ok else 'MISMATCH'}", flush=True)
if rank == 0:
    print(f"[rs_smoke] {'ALL RANKS OK' if t.item() == 0 else f'{t.item()} RANKS MISMATCH'} nnodes={nnodes} world={world} "
          f"M={M} N={N} K={K}", flush=True)
dist.barrier()
dist.destroy_process_group()
