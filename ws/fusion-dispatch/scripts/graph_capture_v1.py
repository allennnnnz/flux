################################################################################
# fusion-dispatch F0.4: can each path be captured into a CUDA graph and replayed
# correctly, and how does replay time compare with eager?
#
# One item per process (a failed capture must not take the others down).
# Items (Llama-3-70B TP=8 shapes by default):
#   A_fused      AGKernel.forward                         x(M/8,K) w(n,K) -> (M,n)
#   B_nccl       all_gather_into_tensor + torch.mm       same shapes
#   C_fluxag     AllGatherOp.run + torch.mm              same shapes
#   R_fused      GemmRS.forward                          x(M,k) w(N,k) -> (M/8,N)
#   R_nccl       torch.mm + reduce_scatter_tensor        same shapes
#   AR_nccl      torch.mm + all_reduce (vLLM default TP) x(M,k) w(N,k) -> (M,N)
# Per M: eager reference -> capture (3 warmup on a side stream, PyTorch docs
# pattern) -> replay x3, compare with eager (allclose + bitwise flag) -> timing:
# eager vs replay interleaved, L2 flush + GPU-side align before each, rank-max median.
# Usage: ./launch.sh graph_capture_v1.py --item A_fused --Ms 64,1024 --out <json>
################################################################################
import argparse
import json
import os
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

ap = argparse.ArgumentParser()
ap.add_argument("--item", required=True)
ap.add_argument("--Ms", default="64,1024")
ap.add_argument("--ag_nk", default="10240,8192", help="AG side full N, K (L-QKV)")
ap.add_argument("--rs_nk", default="8192,8192", help="RS side N, full K (L-O)")
ap.add_argument("--rounds", type=int, default=100)
ap.add_argument("--out", required=True)
args = ap.parse_args()

TP = initialize_distributed()
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
RANK, W = TP.rank(), TP.size()
dt = torch.bfloat16
Ms = [int(x) for x in args.Ms.split(",")]
MAXM = 16384
FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
ALIGN = torch.zeros(1, device="cuda")
N_ag, K_ag = map(int, args.ag_nk.split(","))
N_rs, K_rs = map(int, args.rs_nk.split(","))
n_ag, k_rs = N_ag // W, K_rs // W


def log(*a):
    if RANK == 0:
        print(*a, file=sys.stderr, flush=True)


def ag_opt():
    o = flux.AllGatherOption()
    o.mode = flux.AGRingMode.All2All
    o.use_read = True
    o.use_cuda_core_local = False
    o.use_cuda_core_ag = False
    o.fuse_sync = False
    o.input_buffer_copied = False
    return o


def build(item, M):
    """Returns fn() -> output tensor (static buffers, so graph replay writes the same memory)."""
    g = torch.Generator(device="cuda").manual_seed(1234 + RANK)
    if item in ("A_fused", "B_nccl", "C_fluxag"):
        x = torch.randn(M // W, K_ag, device="cuda", generator=g).to(dt)
        w = (torch.randn(n_ag, K_ag, device="cuda", generator=g) * 0.01).to(dt)
        out = torch.empty(M, n_ag, device="cuda", dtype=dt)
        if item == "A_fused":
            op = STATE.setdefault("ag_kernel", flux.AGKernel(TP, 1, MAXM, n_ag, K_ag, dt, output_dtype=dt))
            opt = ag_opt()
            return lambda: op.forward(x, w, output=out, transpose_weight=False, all_gather_option=opt)
        full = torch.empty(M, K_ag, device="cuda", dtype=dt)
        if item == "B_nccl":
            def f():
                dist.all_gather_into_tensor(full, x, group=TP)
                return torch.mm(full, w.t(), out=out)
            return f
        agop = STATE.setdefault("ag_op", flux.AllGatherOp(TP, 1, MAXM, K_ag, dt))
        buf = agop.local_input_buffer()[:M]
        opt = ag_opt()

        def f():
            agop.run(x, None, opt, torch.cuda.current_stream().cuda_stream)
            return torch.mm(buf, w.t(), out=out)
        return f
    x = torch.randn(M, k_rs, device="cuda", generator=g).to(dt)
    w = (torch.randn(N_rs, k_rs, device="cuda", generator=g) * 0.01).to(dt)
    if item == "R_fused":
        op = STATE.setdefault("gemm_rs", flux.GemmRS(TP, 1, MAXM, N_rs, dt, dt, transpose_weight=False))
        return lambda: op.forward(x, w)
    part = torch.empty(M, N_rs, device="cuda", dtype=dt)
    if item == "R_nccl":
        out = torch.empty(M // W, N_rs, device="cuda", dtype=dt)

        def f():
            torch.mm(x, w.t(), out=part)
            dist.reduce_scatter_tensor(out, part, group=TP)
            return out
        return f
    if item == "AR_nccl":
        def f():
            torch.mm(x, w.t(), out=part)
            dist.all_reduce(part, group=TP)
            return part
        return f
    raise ValueError(item)


def agree(flag):
    t = torch.tensor([1 if flag else 0], device="cuda", dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=TP)
    return bool(t.item())


def timed(fn, rounds):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(rounds)]
    for r in range(rounds):
        FLUSH.zero_()
        torch.cuda._sleep(400_000)
        dist.all_reduce(ALIGN, group=TP)
        ev[r][0].record()
        fn()
        ev[r][1].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in ev]


STATE = {}
result = {"item": args.item, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}
for M in Ms:
    rec = {}
    try:
        fn = build(args.item, M)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ref = fn().clone()
        torch.cuda.synchronize()
        # capture (PyTorch docs: warm up on a side stream first)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        dist.barrier(TP)
        graph = torch.cuda.CUDAGraph()
        err = ""
        try:
            with torch.cuda.graph(graph):
                gout = fn()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            err = "".join(traceback.format_exception_only(type(exc), exc)).strip()[-400:]
        cap_ok = agree(not err)
        rec["capture_ok"] = cap_ok
        rec["capture_error_rank0"] = err
        if cap_ok:
            ok_close, ok_equal = True, True
            for _ in range(3):
                gout.zero_()
                graph.replay()
                torch.cuda.synchronize()
                ok_close &= torch.allclose(gout.float(), ref.float(), atol=0.05, rtol=0.05)
                ok_equal &= torch.equal(gout, ref)
            rec["replay_allclose_all_ranks"] = agree(ok_close)
            rec["replay_bitwise_all_ranks"] = agree(ok_equal)
            # interleaved timing: eager vs replay
            te, tg = [], []
            for blk in range(args.rounds // 10):
                te += timed(fn, 10)
                tg += timed(graph.replay, 10)
            tt = torch.tensor([te, tg], device="cuda", dtype=torch.float64)
            gat = [torch.zeros_like(tt) for _ in range(W)]
            dist.all_gather(gat, tt, group=TP)
            mx = torch.stack(gat).max(dim=0).values.cpu()
            rec["eager_ms_rank_max_median"] = statistics.median(mx[0].tolist())
            rec["graph_ms_rank_max_median"] = statistics.median(mx[1].tolist())
    except Exception as exc:  # noqa: BLE001
        rec["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()[-400:]
    result["cases"][M] = rec
    log(f"[{args.item} M={M}] {rec}")
if RANK == 0:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
dist.barrier(TP)
torch.cuda.synchronize()
# destroy_process_group() hangs after NCCL work was captured into a graph (observed
# 2026-09-30 for B_nccl / R_nccl: results written, then teardown blocked until timeout).
# Exit without teardown; results are already on disk.
sys.stderr.flush()
os._exit(0)
