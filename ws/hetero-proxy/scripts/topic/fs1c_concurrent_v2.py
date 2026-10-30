"""FS1c (2026-10-30; plan addendum 2): FS1b's concurrent item was confounded - AllGatherOp.run starts with a barrier
KERNEL (nvshmemx_barrier_all_on_stream), so a GEMM launched right after on another stream can take every SM first and
push the whole all-gather behind it. Here the GEMM launch is delayed by a ~30 us GPU sleep (one SM) so the barrier runs
first; the measured sleep time is subtracted. Inside the concurrent run the all-gather (side stream) and the GEMM (main
stream) are also timed separately and compared with their standalone times.
FS1b docstring follows.

FS1b (2026-10-30; plan addendum in reports/20261030_fs_validation_plan.md): is Flux AG+GEMM's overlap gap caused by
the fused kernel's waiting (scheduling) or by copy-engine / GEMM resource contention?
New item 'concurrent': AllGatherOp.run on a side stream and gemm_only on the main stream at the same time, neither
waiting for the other (gemm_only reads an already gathered copy); time = fork to join on the main stream.
concurrent ~ max(ag, gemm_only) -> no contention, the gap is scheduling; concurrent ~ fused -> contention.
No profiling. Derived from fs1_profile_v1.py; its docstring follows.

FS step 1 (2026-10-30; plan reports/20261030_fs_validation_plan.md): how much of Flux AG+GEMM's overlap gap does
choosing the best compiled config recover, without changing Flux?

Per (layer, M), TP8 bf16, All2All + use_read:
  fused_default : AGKernel.forward (registry default config)
  ag            : AllGatherOp.run (same AG code path as inside forward)
  gemm_only     : AGKernel.gemm_only on the gathered input (default config, all signals set)
  cublas        : torch.mm on the gathered input
  fused_best    : AGKernel.profiling() visits every compiled config (5 warm-up + 10 calls each, per-call GpuTimer,
                  rank-max); best time and top-5 from ProfilingContext.get_latest_prof_result().
The first four use the same method as profiling(): per-call CUDA events, 5 warm-up + 10 calls, mean, rank-max; repeated
--rounds times in shuffled order, median. ideal = max(ag, min(gemm_only, cublas)).
Each AGKernel / AllGatherOp is built once and never destroyed (CLAUDE.md trap). Launch with ./launch.sh.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../common/measure"))
from clock_logger import ClockLogger  # noqa: E402

DT = torch.bfloat16
SLEEP_CYCLES = 42000  # ~30 us at 1.41 GHz
LAYERS = {"G-FC1": (49152, 12288), "G-QKV": (36864, 12288), "L-GU": (57344, 8192), "L-QKV": (10240, 8192),
          "Q-GU": (55296, 5120), "Q-QKV": (7168, 5120)}


def ag_option():
    opt = flux.AllGatherOption()
    opt.mode = flux.AGRingMode.All2All
    opt.use_read = True
    opt.use_cuda_core_local = False
    opt.use_cuda_core_ag = False
    opt.fuse_sync = False
    opt.input_buffer_copied = False
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=",".join(LAYERS))
    ap.add_argument("--Ms", default="512,1024,2048,4096,8192")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--no_profile", action="store_true", default=True)
    args = ap.parse_args()

    tp = initialize_distributed()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)  # undo flux init_seed() (CLAUDE.md trap)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    rank, W = tp.rank(), tp.size()
    lr = int(os.environ.get("LOCAL_RANK", 0))
    nnodes = flux.testing.NNODES()
    Ms = [int(m) for m in args.Ms.split(",")]
    maxm = max(Ms)
    opt = ag_option()
    stream = torch.cuda.current_stream()
    side = torch.cuda.Stream()

    def timeit(fn):
        for _ in range(5):
            fn()
        ts = []
        for _ in range(10):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            ts.append((s, e))
        torch.cuda.synchronize()
        t = torch.tensor([sum(s.elapsed_time(e) for s, e in ts) / len(ts)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    results = []
    ops = {}
    for name in args.layers.split(","):
        N, K = LAYERS[name]
        n = N // W
        ops[name] = (flux.AGKernel(tp, nnodes, maxm, n, K, DT, output_dtype=DT), flux.AllGatherOp(tp, nnodes, maxm, K, DT),
                     (torch.randn((n, K), device="cuda") * 0.01).to(DT))
    with ClockLogger(lr, period_s=0.25) as clk:
        for name in args.layers.split(","):
            N, K = LAYERS[name]
            op, ag, w = ops[name]
            for M in Ms:
                x = torch.randn((M // W, K), device="cuda").to(DT)
                full = torch.empty((M, K), dtype=DT, device="cuda")
                dist.all_gather_into_tensor(full, x)
                out = torch.empty((M, N // W), dtype=DT, device="cuda")
                out2 = torch.empty_like(out)
                cev = []

                def concurrent():
                    e = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
                    e[0].record()
                    side.wait_event(e[0])
                    e[1].record(side)
                    ag.run(x, None, opt, side.cuda_stream)
                    e[2].record(side)
                    torch.cuda._sleep(SLEEP_CYCLES)
                    e[3].record()
                    op.gemm_only(full, w, output=out2, transpose_weight=False)
                    e[4].record()
                    stream.wait_stream(side)
                    e[5].record()
                    cev.append(e)

                items = {
                    "concurrent": concurrent,
                    "fused_default": lambda: op.forward(x, w, output=out, transpose_weight=False, all_gather_option=opt),
                    "ag": lambda: ag.run(x, None, opt, stream.cuda_stream),
                    "gemm_only": lambda: op.gemm_only(full, w, output=out2, transpose_weight=False),
                    "cublas": lambda: torch.mm(full, w.t(), out=out2),
                }
                # correctness of the default fused path once
                op.forward(x, w, output=out, transpose_weight=False, all_gather_option=opt)
                torch.mm(full, w.t(), out=out2)
                torch.cuda.synchronize()
                ok = bool(torch.allclose(out.float(), out2.float(), atol=0.05, rtol=0.05))
                meas = {k: [] for k in items}
                t0 = time.perf_counter()
                for r in range(args.rounds):
                    order = list(items)
                    random.Random(list(LAYERS).index(name) * 100000 + M * 10 + r).shuffle(order)  # same order on every rank
                    for k in order:
                        dist.barrier()
                        meas[k].append(timeit(items[k]))
                torch.cuda.synchronize()
                # per-call breakdown of the concurrent item (all timed calls; warm-ups included are few and similar)
                tot = [e[0].elapsed_time(e[5]) for e in cev]
                slp = [e[0].elapsed_time(e[3]) for e in cev]
                agd = [e[1].elapsed_time(e[2]) for e in cev]
                gmd = [e[3].elapsed_time(e[4]) for e in cev]
                br = torch.tensor([statistics.median(tot) - statistics.median(slp), statistics.median(agd),
                                   statistics.median(gmd), statistics.median(slp)], device="cuda")
                dist.all_reduce(br, op=dist.ReduceOp.MAX)
                t1 = time.perf_counter()
                rec = {"layer": name, "N": N, "K": K, "M": M, "allclose_default": ok,
                       **{k: statistics.median(v) for k, v in meas.items()}, "raw": meas,
                       "conc_minus_sleep": br[0].item(), "conc_ag": br[1].item(), "conc_gemm": br[2].item(),
                       "conc_sleep": br[3].item(),
                       "clock_meas_min": clk.min_clock_between(t0, t1),
                       "clock_meas_mode": (statistics.mode([c for t, c in clk.samples if t0 <= t <= t1])
                                           if any(t0 <= t <= t1 for t, c in clk.samples) else None)}
                if not args.no_profile:
                    ctx = flux.ProfilingContext(f"{name}_{M}")
                    dist.barrier()
                    op.profiling(x, w, output=out, transpose_weight=False, all_gather_option=opt, prof_ctx=ctx)
                    torch.cuda.synchronize()
                    rec["prof_top5"] = ctx.get_latest_prof_result()
                    rec["prof_all"] = list(ctx.get_all_prof_results())
                rec["clock_prof_min"] = clk.min_clock_between(t1, time.perf_counter())
                results.append(rec)
                if rank == 0:
                    print(json.dumps({k: v for k, v in rec.items() if k not in ("raw", "prof_all")}), flush=True)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "fs1c_raw.json"), "w") as f:
            json.dump(results, f, indent=1)
    dist.barrier()
    os._exit(0)  # do not tear down Flux ops (CLAUDE.md trap: destroy/recreate can hang)


if __name__ == "__main__":
    main()
