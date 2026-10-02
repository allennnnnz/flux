################################################################################
# fusion-dispatch F1.3 v3 (verification V3): v2 + a REAL vLLM 0.8.5 all-reduce baseline.
#
# v3 vs v2: adds policy tp_ar_vllm = the vLLM default TP layout with vLLM's own
# all-reduce dispatch (vllm/distributed/device_communicators/cuda_communicator.py:52-71):
# CustomAllreduce when the tensor is < 8 MiB on a fully-NVLinked node, else PyNccl.
# vLLM's distributed init builds that communicator; graphs for tp_ar_vllm are captured inside
# vllm.distributed.parallel_state.graph_capture() (registers custom-AR graph buffers), as vLLM
# does. Must run in the vLLM venv:
#   bash ws/fusion-dispatch/scripts/launch_vllm_env.sh validate_block_v3.py ...
# tp_ar (torch.distributed NCCL all_reduce) is kept as a bridge to v2 numbers.
#
# v2 vs v1: RMSNorm and SiLU*up are single fused Triton kernels (vLLM uses fused CUDA
# kernels for both). v1 built RMSNorm from ~6 PyTorch elementwise kernels, which
# over-charged the tp_ar layout: it normalises all M rows on every rank, SP only M/W.
#
# A stack of L Llama-3-70B blocks (TP=8, bf16, distinct weights per block):
#   RMSNorm -> QKV (col) -> attention (SDPA, GQA 8 local q heads / 1 local kv head)
#   -> O (row) -> +res -> RMSNorm -> gate_up (col) -> SiLU*up -> down (row) -> +res
# phase=decode : M sequences x 1 new token, KV cache length --ctx per block
# phase=prefill: one sequence of M tokens, causal
#
# Policies (same weights, same input):
#   tp_ar        vLLM default TP: every rank holds all M tokens; row-parallel output
#                all_reduce'd (cuBLAS + NCCL)                 [reference layout]
#   sp_nccl      sequence-parallel, every AG/RS via NCCL + cuBLAS
#   sp_flux      sequence-parallel, every AG/RS via Flux fused ops
#   sp_rsflux    AG via NCCL, RS via Flux (capturable mix)
#   sp_dispatch  sequence-parallel, per-layer choice from the dispatch table
# Modes: eager (CPU launches every op; what vLLM does without graphs, e.g. prefill)
#        graph (each policy captured per M, replayed; sp_flux is skipped because
#               AGKernel cannot be captured, F0.4; sp_dispatch uses graph_mode=True)
# Protocol: policies interleaved per round in a shared random order; L2 flush,
# GPU sleep pad, 1-elem NCCL all_reduce alignment before each; CUDA events around
# the whole stack; per-round rank-max; >= 200 rounds; SM-clock filter.
# Correctness: every SP policy's output must match rows [r*M/W,(r+1)*M/W) of tp_ar.
# Usage: ./launch.sh validate_block_v1.py --phase decode --Ms 8,64,512 --table t.json --out_dir d
################################################################################
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "measure"))
from clock_logger import ClockLogger  # noqa: E402
from dispatcher_v1 import DispatchTable, FluxDispatcher  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

DT = torch.bfloat16
H, NQ, NKV, HD, FFN = 8192, 64, 8, 128, 28672  # Llama-3-70B


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * N + offs, (x * r * w).to(tl.bfloat16), mask=m)


@triton.jit
def _silu_mul_kernel(gu_ptr, y_ptr, F_, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = col < F_
    g = tl.load(gu_ptr + row * 2 * F_ + col, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * F_ + F_ + col, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * F_ + col, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=m)


def rmsnorm(x, w, eps=1e-5):
    x = x.contiguous()
    y = torch.empty_like(x)
    _rmsnorm_kernel[(x.shape[0],)](x, w, y, x.shape[1], eps, BLOCK=triton.next_power_of_2(x.shape[1]),
                                   num_warps=8)
    return y


def silu_mul(gu):
    gu = gu.contiguous()
    M, F2 = gu.shape
    y = torch.empty(M, F2 // 2, device=gu.device, dtype=gu.dtype)
    _silu_mul_kernel[(M, triton.cdiv(F2 // 2, 1024))](gu, y, F2 // 2, BLOCK=1024)
    return y


class Block:
    def __init__(self, W, g):
        self.nq, self.nkv = NQ // W, NKV // W
        self.qkv_n = (self.nq + 2 * self.nkv) * HD
        s = 0.02
        mk = lambda *shape: (torch.randn(*shape, device="cuda", generator=g) * s).to(DT)  # noqa: E731
        self.w_qkv = mk(self.qkv_n, H)               # col-parallel (n, K)
        self.w_o = mk(H, self.nq * HD)               # row-parallel (N, k)
        self.w_gu = mk(2 * FFN // W, H)              # col-parallel
        self.w_down = mk(H, FFN // W)                # row-parallel
        self.ln1 = torch.ones(H, device="cuda", dtype=DT)
        self.ln2 = torch.ones(H, device="cuda", dtype=DT)
        self.kv = None

    def set_kv(self, M, ctx, g):
        if ARGS.phase == "decode":
            self.kv = (torch.randn(M, self.nkv, ctx, HD, device="cuda", generator=g).to(DT),
                       torch.randn(M, self.nkv, ctx, HD, device="cuda", generator=g).to(DT))

    def attention(self, qkv):
        M = qkv.shape[0]
        q, k, v = qkv.split([self.nq * HD, self.nkv * HD, self.nkv * HD], dim=-1)
        if ARGS.phase == "decode":
            q = q.view(M, self.nq, 1, HD)
            kc, vc = self.kv  # the new token's k/v would be appended; cost of that append is ignored
            o = F.scaled_dot_product_attention(q, kc, vc, enable_gqa=True)
            return o.reshape(M, self.nq * HD)
        q = q.view(1, M, self.nq, HD).transpose(1, 2)
        k = k.view(1, M, self.nkv, HD).transpose(1, 2)
        v = v.view(1, M, self.nkv, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return o.transpose(1, 2).reshape(M, self.nq * HD)

    def forward_tp_ar(self, x):  # x: (M, H) replicated
        h = rmsnorm(x, self.ln1)
        a = self.attention(torch.mm(h, self.w_qkv.t()))
        o = torch.mm(a, self.w_o.t())
        dist.all_reduce(o, group=TP)
        x = x + o
        h = rmsnorm(x, self.ln2)
        d = torch.mm(silu_mul(torch.mm(h, self.w_gu.t())), self.w_down.t())
        dist.all_reduce(d, group=TP)
        return x + d

    def forward_tp_ar_vllm(self, x, big=False):  # vLLM default: custom AR below 8 MiB, PyNccl above
        ar = BIG_AR if big else VLLM_AR          # big: custom AR up to --car_big_mib (V3b fairness check)
        h = rmsnorm(x, self.ln1)
        a = self.attention(torch.mm(h, self.w_qkv.t()))
        x = x + ar(torch.mm(a, self.w_o.t()))
        h = rmsnorm(x, self.ln2)
        d = torch.mm(silu_mul(torch.mm(h, self.w_gu.t())), self.w_down.t())
        return x + ar(d)

    def forward_sp(self, x, D):  # x: (M/W, H) sequence shard
        h = rmsnorm(x, self.ln1)
        a = self.attention(D.ag_gemm(h, self.w_qkv))
        x = x + D.gemm_rs(a, self.w_o)
        h = rmsnorm(x, self.ln2)
        return x + D.gemm_rs(silu_mul(D.ag_gemm(h, self.w_gu)), self.w_down)


def main():
    Ms = [int(m) for m in ARGS.Ms.split(",")]
    table = DispatchTable.load_broadcast(ARGS.table, TP)
    g = torch.Generator(device="cuda").manual_seed(4321 + RANK)  # weights differ per rank (shards)
    blocks = [Block(W, g) for _ in range(ARGS.L)]
    max_m = max(Ms)
    b0 = blocks[0]
    shapes_ag = [(b0.qkv_n, H), (b0.w_gu.shape[0], H)]
    shapes_rs = [(H, b0.w_o.shape[1]), (H, b0.w_down.shape[1])]
    Ds = {
        "sp_nccl": FluxDispatcher(TP, table, max_m, force={"ag": "nccl", "rs": "nccl"}),
        "sp_flux": FluxDispatcher(TP, table, max_m, force={"ag": "flux", "rs": "flux"}),
        "sp_rsflux": FluxDispatcher(TP, table, max_m, force={"ag": "nccl", "rs": "flux"}),
        "sp_dispatch": FluxDispatcher(TP, table, max_m, graph_mode=(ARGS.mode == "graph")),
    }
    # all dispatchers share backends (one Flux op per shape): build once, then alias
    base = Ds["sp_dispatch"]
    base.prepare(shapes_ag, shapes_rs)
    for D in Ds.values():
        D._agk, D._agop, D._rs, D._gbuf = base._agk, base._agop, base._rs, base._gbuf
    policies = ["tp_ar", "tp_ar_vllm", "sp_nccl", "sp_flux", "sp_rsflux", "sp_dispatch"]
    if CA_BIG is not None:
        policies.insert(2, "tp_ar_vllm_big")
    if ARGS.mode == "graph":
        policies.remove("sp_flux")  # AGKernel not capturable (F0.4)
    FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
    ALIGN = torch.zeros(1, device="cuda")
    rng = random.Random(ARGS.seed)
    out_rows, meta = [], {"args": vars(ARGS), "table_digest": table.digest(), "table_meta": table.data.get("meta"),
                          "date": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}
    for M in Ms:
        gx = torch.Generator(device="cuda").manual_seed(99 + M)  # same input on every rank
        x_full = torch.randn(M, H, device="cuda", generator=gx).to(DT)
        m = M // W
        x_sp = x_full[RANK * m:(RANK + 1) * m].contiguous()
        for b in blocks:
            b.set_kv(M, ARGS.ctx, g)
        for D in Ds.values():
            D.counts.clear()
            D.misses.clear()

        def run(pol):
            if pol in ("tp_ar", "tp_ar_vllm", "tp_ar_vllm_big"):
                y = x_full
                for b in blocks:
                    if pol == "tp_ar":
                        y = b.forward_tp_ar(y)
                    elif pol == "tp_ar_vllm":
                        y = b.forward_tp_ar_vllm(y)
                    else:
                        y = b.forward_tp_ar_vllm(y, big=True)
                return y
            y = x_sp
            for b in blocks:
                y = b.forward_sp(y, Ds[pol])
            return y

        # correctness + warmup
        ref = run("tp_ar")[RANK * m:(RANK + 1) * m].float()
        chk = {}
        for pol in policies[1:]:
            y = run(pol).float()
            if pol in ("tp_ar_vllm", "tp_ar_vllm_big"):
                y = y[RANK * m:(RANK + 1) * m]
            err = (y - ref).abs().max().item()
            rel = err / max(ref.abs().max().item(), 1e-6)
            ok = torch.tensor([1 if rel < ARGS.rel_tol else 0], device="cuda", dtype=torch.int32)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=TP)
            chk[pol] = {"max_abs_err": err, "rel_err": rel, "ok_all_ranks": bool(ok.item())}
        counts = {pol: {f"{k[0]}:{k[1]}": v for k, v in Ds[pol].counts.items()} for pol in Ds}
        misses = {f"{k}": v for k, v in Ds["sp_dispatch"].misses.items()}
        for _ in range(ARGS.warmup):
            for pol in policies:
                run(pol)
        torch.cuda.synchronize()
        fns = {pol: (lambda p=pol: run(p)) for pol in policies}
        if ARGS.mode == "graph":
            graphs = {}
            for pol in policies:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        run(pol)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                dist.barrier(TP)
                gr = torch.cuda.CUDAGraph()
                if pol == "tp_ar_vllm_big":
                    with VLLM_GRAPH_CAPTURE(torch.device(f"cuda:{LOCAL_RANK}")) as gctx, CA_BIG.capture():
                        with torch.cuda.graph(gr, stream=gctx.stream):
                            run(pol)
                elif pol == "tp_ar_vllm":
                    # vLLM's capture context: side stream + CustomAllreduce.capture() (buffer registration)
                    with VLLM_GRAPH_CAPTURE(torch.device(f"cuda:{LOCAL_RANK}")) as gctx:
                        with torch.cuda.graph(gr, stream=gctx.stream):
                            run(pol)
                else:
                    with torch.cuda.graph(gr):
                        run(pol)
                graphs[pol] = gr
            torch.cuda.synchronize()
            fns = {pol: graphs[pol].replay for pol in policies}
        ev = [{p: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for p in policies}
              for _ in range(ARGS.rounds)]
        orders, windows = [], []
        with ClockLogger(LOCAL_RANK, 0.02) as clk:
            for r in range(ARGS.rounds):
                order = policies[:]
                rng.shuffle(order)
                orders.append(order)
                t0 = time.perf_counter()
                for pol in order:
                    FLUSH.zero_()
                    torch.cuda._sleep(ARGS.pad_cycles)
                    dist.all_reduce(ALIGN, group=TP)
                    ev[r][pol][0].record()
                    fns[pol]()
                    ev[r][pol][1].record()
                torch.cuda.synchronize()
                windows.append((t0, time.perf_counter()))
            clocks = [clk.min_clock_between(*w_) for w_ in windows]
        t = torch.tensor([[ev[r][p][0].elapsed_time(ev[r][p][1]) for p in policies] for r in range(ARGS.rounds)],
                         device="cuda", dtype=torch.float64)
        ck = torch.tensor([c if c else -1 for c in clocks], device="cuda", dtype=torch.float64)
        gt, gk = [torch.zeros_like(t) for _ in range(W)], [torch.zeros_like(ck) for _ in range(W)]
        dist.all_gather(gt, t, group=TP)
        dist.all_gather(gk, ck, group=TP)
        if ARGS.mode == "graph":
            del graphs
        if RANK == 0:
            T = torch.stack(gt).cpu().max(0).values  # rank-max (rounds, policies)
            Kc = torch.stack(gk).cpu()
            kept = [True] * ARGS.rounds
            for rk in range(W):
                vals = [int(v) for v in Kc[rk].tolist() if v > 0]
                if vals:
                    mo = statistics.mode(vals)
                    for r in range(ARGS.rounds):
                        if 0 < Kc[rk, r].item() < 0.95 * mo:
                            kept[r] = False
            ks = [r for r in range(ARGS.rounds) if kept[r]]
            med = {p: statistics.median(T[r, j].item() for r in ks) for j, p in enumerate(policies)}
            jd = policies.index("sp_dispatch")
            diffs = {}
            for j, p in enumerate(policies):
                d = sorted(T[r, jd].item() - T[r, j].item() for r in ks)
                diffs[p] = (d[len(d) // 10], statistics.median(d), d[9 * len(d) // 10])
            meta["cases"][M] = {"check": chk, "counts": counts, "misses": misses, "kept": len(ks),
                                "median_ms": med, "dispatch_minus_policy_p10_p50_p90": diffs}
            for r in range(ARGS.rounds):
                for j, p in enumerate(policies):
                    out_rows.append([ARGS.phase, ARGS.mode, ARGS.L, M, r, p, orders[r].index(p),
                                     f"{T[r, j].item():.5f}", int(kept[r])])
            best = min(med, key=med.get)
            line = "  ".join(f"{p}={med[p]:.4f}" for p in policies)
            print(f"[{ARGS.phase} {ARGS.mode} L={ARGS.L} M={M}] kept {len(ks)}  {line}  best={best}  "
                  f"dispatch counts={counts['sp_dispatch']} misses={misses}  "
                  f"check={ {p: round(c['rel_err'], 4) for p, c in chk.items()} }", file=sys.stderr, flush=True)
    if RANK == 0:
        os.makedirs(ARGS.out_dir, exist_ok=True)
        tag = f"{ARGS.phase}_{ARGS.mode}_L{ARGS.L}"
        with open(os.path.join(ARGS.out_dir, f"raw_{tag}.csv"), "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["phase", "mode", "L", "M", "round", "policy", "order_pos", "rank_max_ms", "kept"])
            w_.writerows(out_rows)
        json.dump(meta, open(os.path.join(ARGS.out_dir, f"meta_{tag}.json"), "w"), indent=1, default=str)
    dist.barrier(TP)
    torch.cuda.synchronize()
    sys.stderr.flush()
    os._exit(0)  # teardown after graph capture can hang (F0.4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["decode", "prefill"], required=True)
    ap.add_argument("--mode", choices=["eager", "graph"], default="eager")
    ap.add_argument("--Ms", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pad_cycles", type=int, default=400_000)
    ap.add_argument("--rel_tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--car_big_mib", type=int, default=0,
                    help="V3b: add policy tp_ar_vllm_big = vLLM custom all-reduce with this max size (MiB)")
    ap.add_argument("--out_dir", required=True)
    ARGS = ap.parse_args()
    TP = initialize_distributed()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)  # production cuBLAS (E0 finding)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    RANK, W = TP.rank(), TP.size()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    # vLLM's own distributed init: builds the TP GroupCoordinator whose CudaCommunicator holds
    # CustomAllreduce (< 8 MiB, fully NVLinked) and PyNccl (otherwise), exactly as in serving.
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                  tensor_model_parallel_all_reduce)
    from vllm.distributed.parallel_state import get_tp_group, graph_capture
    init_distributed_environment(world_size=W, rank=RANK, distributed_init_method="env://",
                                 local_rank=LOCAL_RANK, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=W)
    _comm = get_tp_group().device_communicator
    CA = _comm.ca_comm
    VLLM_AR = tensor_model_parallel_all_reduce
    VLLM_GRAPH_CAPTURE = graph_capture
    CA_BIG, BIG_AR = None, None
    if ARGS.car_big_mib > 0:
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        CA_BIG = CustomAllreduce(group=get_tp_group().cpu_group, device=torch.device(f"cuda:{LOCAL_RANK}"),
                                 max_size=ARGS.car_big_mib * 1024 * 1024)

        def BIG_AR(t):  # same dispatch as CudaCommunicator.all_reduce, larger custom-AR ceiling
            if not CA_BIG.disabled and CA_BIG.should_custom_ar(t):
                return CA_BIG.custom_all_reduce(t)
            return _comm.pynccl_comm.all_reduce(t)
    if RANK == 0:
        print(f"vLLM custom AR present={CA is not None} disabled={getattr(CA, 'disabled', None)} "
              f"fully_connected={getattr(CA, 'fully_connected', None)} max_size={getattr(CA, 'max_size', None)}; "
              f"pynccl disabled={getattr(_comm.pynccl_comm, 'disabled', None)}", file=sys.stderr, flush=True)
    main()
