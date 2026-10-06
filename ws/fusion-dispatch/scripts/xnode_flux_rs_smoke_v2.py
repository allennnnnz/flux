################################################################################
# fusion-dispatch E4 smoke v2 (2026-10-06): why does flux.GemmRS_multinode return wrong results on node 0?
# Hypothesis: python/flux/gemm_rs_sm80.py calls the intra-node GemmRS twice in a row (one call per node
# chunk) with no synchronisation in between, and on sm80 GemmRS's forward_barrier is a no-op (CLAUDE.md
# trap table). A rank that starts call 2 early scatters (epilogue P2P writes) into a peer's buffer while
# that peer is still copying call 1's result out. Node 0 keeps call 1's result -> corrupted; node 1 keeps
# call 2's (nothing comes after it) -> correct. That matches results/e4_flux_smoke/rsmn_*_rail1.
# Test (one Flux op, built once): forward exactly as in the wrapper ("orig"), then the same with
# torch.cuda.synchronize() + barrier on the intra-node group after each chunk's copy ("fixed").
# Correctness against the torch reference (matmul, then NCCL reduce_scatter over the whole TP group).
# Usage: run_xnode.sh <out> <ppn> pixi ws/fusion-dispatch/scripts/xnode_flux_rs_smoke_v2.py M N K
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
full = (x.float() @ w.float().t()).to(torch.bfloat16)
ref = torch.empty(M // world, N, device="cuda", dtype=torch.bfloat16)
dist.reduce_scatter_tensor(ref, full, group=tp)
torch.cuda.synchronize()


def forward(fixed):
    """Same steps as GemmRS_multinode.forward; fixed=True adds a sync + intra-node barrier per chunk."""
    outputs = []
    for n in range(op.nnodes):
        o = op.cpp_op.forward(x[n * op.max_m:(n + 1) * op.max_m], w, None)
        buf = torch.empty_like(o)
        buf.copy_(o)
        if fixed:
            torch.cuda.synchronize()
            dist.barrier(op.tp_group_intra)
        outputs.append(buf)
    recv = torch.empty_like(outputs[0])
    out = outputs[op.node_id]
    for n in range(1, op.nnodes):
        node_send = (n + op.node_id) % op.nnodes
        r_send = (n * op.local_world_size + op.rank) % op.world_size
        r_recv = (op.rank - n * op.local_world_size + op.world_size) % op.world_size
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, outputs[node_send], r_send, op.tp_group),
                                       dist.P2POp(dist.irecv, recv, r_recv, op.tp_group)])
        [q.wait() for q in reqs]
        out.add_(recv)
    torch.cuda.synchronize()
    return out


for variant in ("orig", "fixed"):
    bad_runs = 0
    for _ in range(5):  # a race: repeat to see whether it is consistent
        out = forward(variant == "fixed")
        err = (out.float() - ref.float()).abs().max().item()
        ok = err <= 2e-2 * max(ref.float().abs().max().item(), 1e-3)
        t = torch.tensor([0 if ok else 1], device="cuda")
        dist.all_reduce(t, group=tp)
        bad_runs += int(t.item() > 0)
        mine_bad = 0 if ok else 1
        dist.barrier(tp)
    print(f"[rs_smoke2] {variant:<5} rank {rank} node {op.node_id} last max_abs_err {err:.4g} "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    if rank == 0:
        print(f"[rs_smoke2] {variant}: {bad_runs}/5 runs had a mismatching rank (nnodes={nnodes}, world={world}, "
              f"M={M} N={N} K={K})", flush=True)
    dist.barrier(tp)
dist.destroy_process_group()
