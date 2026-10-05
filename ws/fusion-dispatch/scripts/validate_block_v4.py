################################################################################
# fusion-dispatch G4 block validation (validate_block_v4.py; derived from validate_block_v3.py).
# v4 vs v3: the model is a parameter (--model llama3-70b | qwen2.5-72b | qwen2.5-32b | llama3-8b; TP =
# world size), and two dispatch tables are compared side by side:
#   sp_g4   table built by build_table_v2.py with the G4 method (model comm + overlap, measured GEMMs,
#           path probes) -- D-009
#   sp_g3   table built from the model alone with the G2/G3 profiles (the G3 method)
# Kept from v3: tp_ar (torch NCCL all-reduce bridge), tp_ar_vllm (vLLM 0.8.5 default: CustomAllreduce
# < 8 MiB, PyNccl above), sp_nccl, sp_flux (eager only; AGKernel cannot be captured), sp_rsflux.
# The layout decision (tp_ar_vllm vs sp_g4 per M) is NOT made here: it is pre-registered offline
# (build_table_v2.py --block) and scored against these measurements.
# Must run in the vLLM venv:
#   TP=<n> bash ws/fusion-dispatch/scripts/launch_vllm_env.sh validate_block_v4.py --model qwen2.5-32b \
#       --phase decode --mode graph --Ms 32,128 --table_g4 t4.json --table_g3 t3.json --out_dir <dir>
# Protocol (unchanged from v3): policies interleaved per round in a shared random order; L2 flush,
# GPU sleep pad, 1-elem NCCL all_reduce alignment before each; CUDA events around the whole stack;
# per-round rank-max; >= 200 rounds; SM-clock filter. Correctness: every SP policy's output must match
# rows [r*M/W,(r+1)*M/W) of tp_ar.
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
MODELS = {"llama3-70b": (8192, 64, 8, 128, 28672), "qwen2.5-72b": (8192, 64, 8, 128, 29568),
          "qwen2.5-32b": (5120, 40, 8, 128, 27648), "llama3-8b": (4096, 32, 8, 128, 14336)}
H, NQ, NKV, HD, FFN = MODELS["llama3-70b"]  # replaced from --model in __main__


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
    table = DispatchTable.load_broadcast(ARGS.table_g4, TP)
    table_g3 = DispatchTable.load_broadcast(ARGS.table_g3, TP)
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
        "sp_g4": FluxDispatcher(TP, table, max_m, graph_mode=(ARGS.mode == "graph")),
        "sp_g3": FluxDispatcher(TP, table_g3, max_m, graph_mode=(ARGS.mode == "graph")),
    }
    # all dispatchers share backends (one Flux op per shape): build once, then alias
    base = Ds["sp_g4"]
    # E4 (2026-10-05): NCCL-only policy subsets (cross-node, where Flux ops cannot be built) skip the
    # Flux backends; G4 behaviour (any Flux policy present) is unchanged.
    _flux_pols = {"sp_flux", "sp_rsflux", "sp_g4", "sp_g3"}
    if not ARGS.policies or _flux_pols & set(ARGS.policies.split(",")):
        base.prepare(shapes_ag, shapes_rs)
    for D in Ds.values():
        D._agk, D._agop, D._rs, D._gbuf = base._agk, base._agop, base._rs, base._gbuf
    policies = ["tp_ar", "tp_ar_vllm", "sp_nccl", "sp_flux", "sp_rsflux", "sp_g4", "sp_g3"]
    if ARGS.policies:
        keep = ["tp_ar"] + [p for p in ARGS.policies.split(",") if p != "tp_ar"]  # tp_ar = correctness reference
        policies = [p for p in policies if p in keep]
    if CA_BIG is not None:
        policies.insert(2, "tp_ar_vllm_big")
    if ARGS.mode == "graph" and "sp_flux" in policies:
        policies.remove("sp_flux")  # AGKernel not capturable (F0.4)
    FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
    ALIGN = torch.zeros(1, device="cuda")
    rng = random.Random(ARGS.seed)
    out_rows, meta = [], {"args": vars(ARGS), "model": [H, NQ, NKV, HD, FFN], "world": W,
                          "table_g4_digest": table.digest(), "table_g3_digest": table_g3.digest(),
                          "table_meta": table.data.get("meta"),
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
        misses = {f"{k}": v for k, v in Ds["sp_g4"].misses.items()}
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
            jd = policies.index("sp_g4") if "sp_g4" in policies else 0
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
                  f"g4 counts={counts['sp_g4']} g3 counts={counts['sp_g3']} misses={misses}  "
                  f"check={ {p: round(c['rel_err'], 4) for p, c in chk.items()} }", file=sys.stderr, flush=True)
    if RANK == 0:
        os.makedirs(ARGS.out_dir, exist_ok=True)
        tag = f"{ARGS.model}_tp{W}_{ARGS.phase}_{ARGS.mode}_L{ARGS.L}" + (f"_{ARGS.tag}" if ARGS.tag else "")
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
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--table_g4", required=True)
    ap.add_argument("--table_g3", required=True)
    ap.add_argument("--policies", default="", help="subset (layout probes); tp_ar is always kept as reference")
    ap.add_argument("--tag", default="")
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
    H, NQ, NKV, HD, FFN = MODELS[ARGS.model]
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
