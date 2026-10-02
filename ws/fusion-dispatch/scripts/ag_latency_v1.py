################################################################################
# fusion-dispatch F2a + F2b: can Flux's AllGather fixed cost (~0.10 ms at M<=64,
# W1 in reports/20260930_plan_dispatcher.md) be cut WITHOUT changing Flux src?
#
# F2a (existing AllGatherOption knobs, flux.AllGatherOp.run):
#   flux_default        use_cuda_core_local=False, fuse_sync=False (what AGKernel uses today)
#   flux_cclocal        use_cuda_core_local=True                  (local copy + reset by a kernel)
#   flux_cclocal_fused  use_cuda_core_local=True, fuse_sync=True  (barrier fused into that kernel,
#                       src/coll/ths_op/all_gather_op.cc:465-510)
#   flux_copied         input_buffer_copied=True: producer already wrote its shard into
#                       the buffer (skips the local copy; lower bound for a fused producer)
#   A_default / A_cclocal_fused   AGKernel.forward with those options (does the fused op gain?)
# F2b (prototypes outside Flux, IPC buffers from flux.create_tensor_list; shards pre-staged;
#      NO cross-rank synchronisation inside -> these are LOWER BOUNDS, a real version needs
#      at least one barrier):
#   proto_ce_1s    7 peer pulls with Tensor.copy_ on one stream (copy engine, serial)
#   proto_ce_7s    same 7 pulls on 7 streams (fork/join with events)
#   proto_triton   one Triton kernel loading all 7 peers' shards over NVLink (P2P loads)
#   proto_triton_parbar  same with the barrier's W signal/wait pairs in W parallel programs
#   proto_triton_bar  a Triton cross-rank flag barrier (system-scope atomics over NVLink)
#                  followed by proto_triton: the honest estimate of a complete AllGather
#                  (only the producer's in-place write of its own shard is not included)
# Reference: nccl = dist.all_gather_into_tensor.
# Protocol: dispatch_map_v2.run_mode (interleaved, L2 flush, gpu/steady, rank-max, clock filter).
# Usage: ./launch.sh ag_latency_v1.py --K 8192 --Ms 8,64,512 --modes gpu,steady --out_dir <dir>
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
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(__file__))
import dispatch_map_v2 as dm  # noqa: E402
import flux  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

DT = torch.bfloat16


@triton.jit
def _p2p_gather(ptrs, dst, chunk, rank, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    src_rank = (rank + 1 + tl.program_id(1)) % WORLD
    base = tl.load(ptrs + src_rank).to(tl.pointer_type(tl.bfloat16))
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < chunk
    v = tl.load(base + src_rank * chunk + offs, mask=m)
    tl.store(dst + src_rank * chunk + offs, v, mask=m)


@triton.jit
def _p2p_barrier_par(flag_ptrs, rank, epoch, WORLD: tl.constexpr):
    # parallel variant: program p signals rank p and waits for rank p's signal
    p = tl.program_id(0)
    dst = tl.load(flag_ptrs + p).to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(dst + rank, epoch, sem="release", scope="sys")
    me = tl.load(flag_ptrs + rank).to(tl.pointer_type(tl.int32))
    v = tl.atomic_add(me + p, 0, sem="acquire", scope="sys")
    while v < epoch:
        v = tl.atomic_add(me + p, 0, sem="acquire", scope="sys")


@triton.jit
def _p2p_barrier(flag_ptrs, rank, epoch, WORLD: tl.constexpr):
    # signal: store `epoch` into slot [rank] of every rank's flag array (system-scope release)
    for p in tl.static_range(WORLD):
        dst = tl.load(flag_ptrs + p).to(tl.pointer_type(tl.int32))
        tl.atomic_xchg(dst + rank, epoch, sem="release", scope="sys")
    # wait: every slot of my own flag array has reached `epoch` (acquire)
    me = tl.load(flag_ptrs + rank).to(tl.pointer_type(tl.int32))
    for p in tl.static_range(WORLD):
        v = tl.atomic_add(me + p, 0, sem="acquire", scope="sys")
        while v < epoch:
            v = tl.atomic_add(me + p, 0, sem="acquire", scope="sys")


def opt(cclocal=False, fuse=False, copied=False):
    o = flux.AllGatherOption()
    o.mode = flux.AGRingMode.All2All
    o.use_read = True
    o.use_cuda_core_local = cclocal
    o.use_cuda_core_ag = False
    o.fuse_sync = fuse
    o.input_buffer_copied = copied
    return o


class Case:
    def __init__(self, M, K, n, st):
        self.M, m = M, M // W
        self.layer = type("L", (), {"weights": st["w"]})
        x = torch.randn(m, K, device="cuda").to(DT)
        self.x = x
        self.full = torch.empty(M, K, device="cuda", dtype=DT)
        dist.all_gather_into_tensor(self.full, x, group=TP)
        ag, agk = st["ag"], st["agk"]
        buf = ag.local_input_buffer()
        outA = torch.empty(M, n, device="cuda", dtype=DT)
        o_def, o_cc, o_ccf = opt(), opt(cclocal=True), opt(cclocal=True, fuse=True)
        o_cp = opt(copied=True)
        chunk = m * K
        peers = st["ipc"]  # list of W tensors (max_m*K,), IPC-mapped
        mine = peers[RANK]
        mine[RANK * chunk:(RANK + 1) * chunk].copy_(x.view(-1))  # stage own shard
        torch.cuda.synchronize()
        dist.barrier(TP)
        ptrs = st["ptrs"]
        others = [(RANK + 1 + i) % W for i in range(W - 1)]
        streams = st["streams"]

        def ce_1s():
            for p in others:
                mine[p * chunk:(p + 1) * chunk].copy_(peers[p][p * chunk:(p + 1) * chunk], non_blocking=True)

        def ce_7s():
            cur = torch.cuda.current_stream()
            ev = torch.cuda.Event()
            ev.record(cur)
            for s, p in zip(streams, others):
                s.wait_event(ev)
                with torch.cuda.stream(s):
                    mine[p * chunk:(p + 1) * chunk].copy_(peers[p][p * chunk:(p + 1) * chunk], non_blocking=True)
            for s in streams:
                cur.wait_stream(s)

        blk = 2048
        grid = (triton.cdiv(chunk, blk), W - 1)

        def tri():
            _p2p_gather[grid](ptrs, mine, chunk, RANK, WORLD=W, BLOCK=blk)

        def tri_parbar():
            st["epoch"] += 1
            _p2p_barrier_par[(W,)](st["flag_ptrs"], RANK, st["epoch"], WORLD=W)
            _p2p_gather[grid](ptrs, mine, chunk, RANK, WORLD=W, BLOCK=blk)

        def tri_bar():
            # one cross-rank barrier (all shards published) + the P2P gather: a complete
            # AllGather except for writing the own shard (the producer would write it in place)
            st["epoch"] += 1
            _p2p_barrier[(1,)](st["flag_ptrs"], RANK, st["epoch"], WORLD=W)
            _p2p_gather[grid](ptrs, mine, chunk, RANK, WORLD=W, BLOCK=blk)

        def run_ag(o):
            return lambda w: ag.run(x, None, o, torch.cuda.current_stream().cuda_stream)

        def pre_copied(w):
            # input_buffer_copied=True: skip the local copy. The own shard is already in the
            # buffer (every flux_* item writes the same x there; check() restores it), which
            # is what a producer writing its output straight into Flux's buffer would give.
            ag.run(x, None, o_cp, torch.cuda.current_stream().cuda_stream)
        self.buf = buf
        self.mine, self.chunk = mine, chunk
        self.fns = {
            "nccl": lambda w: dist.all_gather_into_tensor(self.full, x, group=TP),
            "flux_default": run_ag(o_def),
            "flux_cclocal": run_ag(o_cc),
            "flux_cclocal_fused": run_ag(o_ccf),
            "flux_copied": pre_copied,
            "A_default": lambda w: agk.forward(x, w, output=outA, transpose_weight=False, all_gather_option=o_def),
            "A_cclocal_fused": lambda w: agk.forward(x, w, output=outA, transpose_weight=False,
                                                     all_gather_option=o_ccf),
            "proto_ce_1s": lambda w: ce_1s(),
            "proto_ce_7s": lambda w: ce_7s(),
            "proto_triton": lambda w: tri(),
            "proto_triton_bar": lambda w: tri_bar(),
            "proto_triton_parbar": lambda w: tri_parbar(),
        }
        self.outA = outA

    def check(self, items):
        res = {}
        ref = torch.mm(self.full.float(), self.layer.weights[0].float().t())
        for it in items:
            err = ""
            try:
                if it.startswith("flux_"):
                    self.buf[:self.M].zero_()
                    if it == "flux_copied":
                        m = self.M // W
                        self.buf[RANK * m:(RANK + 1) * m].copy_(self.x)
                        torch.cuda.synchronize()
                elif it.startswith("proto_"):
                    other = torch.ones_like(self.mine[:self.M * self.full.shape[1]])
                    keep = self.mine[RANK * self.chunk:(RANK + 1) * self.chunk].clone()
                    self.mine[:self.M * self.full.shape[1]].copy_(other)
                    self.mine[RANK * self.chunk:(RANK + 1) * self.chunk].copy_(keep)
                    torch.cuda.synchronize()
                    dist.barrier(TP)
                self.fns[it](self.layer.weights[0])
                torch.cuda.synchronize()
                dist.barrier(TP)
                if it.startswith("flux_"):
                    ok = torch.equal(self.buf[:self.M], self.full)
                elif it.startswith("proto_"):
                    ok = torch.equal(self.mine[:self.M * self.full.shape[1]].view(self.M, -1), self.full)
                elif it.startswith("A_"):
                    ok = torch.allclose(self.outA.float(), ref, atol=0.05, rtol=0.05)
                else:
                    ok = torch.equal(self.full, self.full)
            except Exception as exc:  # noqa: BLE001
                ok, err = False, str(exc).strip().splitlines()[-1][:200]
            f = torch.tensor([1 if ok else 0], device="cuda", dtype=torch.int32)
            dist.all_reduce(f, op=dist.ReduceOp.MIN, group=TP)
            res[it] = {"ok_all_ranks": bool(f.item()), "error": err}
        return res


def main():
    Ms = [int(x) for x in ARGS.Ms.split(",")]
    K, n = ARGS.K, ARGS.n_cols
    max_m = max(Ms)
    st = {
        "ag": flux.AllGatherOp(TP, 1, max_m, K, DT),
        "agk": flux.AGKernel(TP, 1, max_m, n, K, DT, output_dtype=DT),
        "ipc": flux.create_tensor_list([max_m * K], DT, TP),
        "streams": [torch.cuda.Stream() for _ in range(W - 1)],
        "w": [(torch.randn(n, K, device="cuda") * 0.01).to(DT) for _ in range(ARGS.steady_L)],
    }
    st["ptrs"] = torch.tensor([t.data_ptr() for t in st["ipc"]], dtype=torch.int64, device="cuda")
    st["flags"] = flux.create_tensor_list([W], torch.int32, TP)
    st["flags"][RANK].zero_()
    torch.cuda.synchronize()
    dist.barrier(TP)
    st["flag_ptrs"] = torch.tensor([t.data_ptr() for t in st["flags"]], dtype=torch.int64, device="cuda")
    st["epoch"] = 0
    items = ARGS.items.split(",")
    rng = random.Random(ARGS.seed)
    os.makedirs(ARGS.out_dir, exist_ok=True) if RANK == 0 else None
    raw = os.path.join(ARGS.out_dir, f"raw_K{K}.csv")
    fh = open(raw, "a", newline="") if RANK == 0 else None
    wr = csv.writer(fh) if fh else None
    if wr and fh.tell() == 0:
        wr.writerow(["K", "n", "M", "mode", "round", "item", "order_pos", "rank_max_ms", "cpu_launch_rank_max_ms", "kept"])
    meta = {"args": vars(ARGS), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}
    for M in Ms:
        case = Case(M, K, n, st)
        chk = case.check(items)
        good = [it for it in items if chk[it]["ok_all_ranks"]]
        meta["cases"][M] = {"check": chk, "modes": {}}
        if RANK == 0:
            print(f"[K={K} M={M}] check: {'all ok' if len(good) == len(items) else {k: v for k, v in chk.items() if not v['ok_all_ranks']}}",
                  file=sys.stderr, flush=True)
        for mode in ARGS.modes.split(","):
            T, C, Kc, orders, kept, modal = dm.run_mode(case, good, mode, ARGS.rounds, ARGS.warmup, rng)
            if RANK != 0:
                continue
            rmax, cmax = T.max(0).values, C.max(0).values
            ks = [r for r in range(T.shape[1]) if kept[r]]
            for r in range(T.shape[1]):
                for j, it in enumerate(good):
                    wr.writerow([K, n, M, mode, r, it, orders[r].index(it), f"{rmax[r, j]:.5f}",
                                 f"{cmax[r, j]:.5f}", int(kept[r])])
            fh.flush()
            med = {it: statistics.median(rmax[r, j].item() for r in ks) for j, it in enumerate(good)}
            meta["cases"][M]["modes"][mode] = {"kept": len(ks), "median_ms": med}
            print(f"[K={K} M={M} {mode}] kept {len(ks)}  " + "  ".join(f"{k}={v * 1e3:.1f}us" for k, v in med.items()),
                  file=sys.stderr, flush=True)
    if RANK == 0:
        fh.close()
        json.dump(meta, open(os.path.join(ARGS.out_dir, f"meta_K{K}.json"), "w"), indent=1, default=str)
    dist.barrier(TP)
    torch.cuda.synchronize()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8192)
    ap.add_argument("--n_cols", type=int, default=1280, help="GEMM n per rank for the A_* items")
    ap.add_argument("--Ms", default="8,64,256,512,1024,2048,4096")
    ap.add_argument("--modes", default="gpu,steady")
    ap.add_argument("--items", default="nccl,flux_default,flux_cclocal,flux_cclocal_fused,flux_copied,"
                                       "A_default,A_cclocal_fused,proto_ce_1s,proto_ce_7s,proto_triton,proto_triton_bar,proto_triton_parbar")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--steady_L", type=int, default=16)
    ap.add_argument("--pad_cycles", type=int, default=400_000)
    ap.add_argument("--clock_period", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out_dir", required=True)
    ARGS = ap.parse_args()
    TP = initialize_distributed()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    RANK, W = TP.rank(), TP.size()
    dm.TP_GROUP, dm.RANK, dm.W = TP, RANK, W
    dm.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    dm.ARGS = ARGS
    dm.FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
    dm.ALIGN = torch.zeros(1, device="cuda")
    main()
