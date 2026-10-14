################################################################################
# hetero-proxy E2E C, step 1 (2026-10-14; plan agreed with the user): WHY is the NVLink + host-staged PCIe split
# all-reduce slower than NCCL alone (results/e2e/c_bw/)? Each component of the PCIe path, measured on its own:
#   a  D2H, all 8 GPUs at once, each into its own host slot: per-GPU GB/s (rank-max time)        sizes 1 / 4 / 16 MiB
#   b  H2D, all 8 GPUs at once, all from the same host result buffer
#   c  D2H and H2D on every GPU at once (both directions in flight; window method: only H2D copies that end before
#      the D2H stream finishes are counted, CLAUDE.md 5.1.7)
#   d  host reduction of 8 slots: the v1 torch reducer (16 threads) vs a C / OpenMP kernel (hxb_sum_bf16, 16 / 32
#      threads), run on NUMA-0 hyper-thread cpus 64-79 / 64-95                                    sizes 0.1 .. 16 MiB
#   e  synchronisation round trip of the v1 path with a 64-byte payload (D2H -> counter -> reducer -> counter -> H2D)
#   f  NCCL all-reduce time alone vs while every GPU streams D2H + H2D in the background (interference)
# Then the ideal split time T(f) = max(NCCL((1-f) S), D2H(fS) + reduce(fS) + H2D(fS) + sync) from these numbers,
# next to the measured v1 split times, attributes the gap.
# Usage: env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
#        ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 c_allreduce_diag_v1.py --out <dir>
################################################################################
import argparse
import ctypes
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from hxb import cudamem as cm  # noqa: E402
from c_allreduce_bw_v1 import HostAllReduce, split_allreduce  # noqa: E402

W = 8
MiB = 2 ** 20


def reducer_bench(sizes_mib, reps, q):
    """Runs in a separate process on NUMA-0 hyper-thread cpus: host reduction of 8 bf16 slots, torch vs C."""
    from hxb import native
    res = {}
    for impl, cpus, nt in (("torch16", list(range(64, 80)), 16), ("c16", list(range(64, 80)), 16),
                           ("c32", list(range(64, 96)), 32)):
        os.sched_setaffinity(0, cpus)
        torch.set_num_threads(nt)
        for s in sizes_mib:
            n = int(s * MiB) // 2
            slots = [torch.randn(n).to(torch.bfloat16) for _ in range(W)]
            out = torch.empty(n, dtype=torch.bfloat16)
            ptrs = (ctypes.c_void_p * W)(*[t.data_ptr() for t in slots])
            ts = []
            for r in range(reps + 2):
                t0 = time.perf_counter()
                if impl.startswith("torch"):
                    acc = None
                    for t in slots:
                        x = t.float()
                        acc = x if acc is None else acc.add_(x)
                    out.copy_(acc)
                else:
                    native.lib().hxb_sum_bf16(ptrs, W, out.data_ptr(), n, nt)
                if r >= 2:
                    ts.append((time.perf_counter() - t0) * 1e3)
            res[f"{impl}|{s}"] = statistics.median(ts)
    q.put(res)


def timed(fn, reps=20):
    dist.barrier()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) / reps


def rank_max(x):
    allv = [None] * dist.get_world_size()
    dist.all_gather_object(allv, x)
    return allv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    os.sched_setaffinity(0, list(range(4 * rank, 4 * rank + 4)))
    torch.set_num_threads(4)
    out = {}
    har = HostAllReduce(rank, dev, 16 * MiB)
    g = torch.randn(8 * MiB, device=dev).to(torch.bfloat16)  # 16 MiB
    slot = har.base + 4096 + rank * har.slot
    result = har.base + 4096 + W * har.slot
    s1, s2 = torch.cuda.Stream(dev), torch.cuda.Stream(dev)

    # a, b: one direction, all GPUs at once
    for s in (1, 4, 16):
        nb = s * MiB
        ta = timed(lambda: cm.memcpy(torch.cuda.current_stream(), slot, g.data_ptr(), nb), args.reps)
        tb = timed(lambda: cm.memcpy(torch.cuda.current_stream(), g.data_ptr(), result, nb), args.reps)
        out[f"a_d2h_all8|{s}"] = rank_max(ta)
        out[f"b_h2d_all8|{s}"] = rank_max(tb)

    # c: both directions on every GPU (window-restricted H2D)
    for s in (1, 4, 16):
        nb = s * MiB
        reps = 3 * args.reps
        dist.barrier()
        torch.cuda.synchronize()
        ev1, ev2 = [], []
        st = torch.cuda.Event(enable_timing=True)
        st.record()
        s1.wait_event(st)
        s2.wait_event(st)
        for _ in range(reps):
            cm.memcpy(s1, slot, g.data_ptr(), nb)
            e = torch.cuda.Event(enable_timing=True)
            e.record(s1)
            ev1.append(e)
            cm.memcpy(s2, g.data_ptr(), result, nb)
            e = torch.cuda.Event(enable_timing=True)
            e.record(s2)
            ev2.append(e)
        torch.cuda.synchronize()
        t1 = [st.elapsed_time(e) for e in ev1]
        t2 = [st.elapsed_time(e) for e in ev2]
        window = t1[-1]
        d2h = (t1[-1] - 0) / reps
        h2d_in = [t2[i] - (t2[i - 1] if i else 0) for i in range(reps) if t2[i] <= window]
        out[f"c_bidir_d2h|{s}"] = rank_max(d2h)
        out[f"c_bidir_h2d_in_window|{s}"] = rank_max(statistics.median(h2d_in) if h2d_in else float("nan"))

    # e: sync round trip of the v1 PCIe path with a 64-byte payload (f -> tail of 32 elements)
    tiny = torch.zeros(4096, dtype=torch.bfloat16, device=dev)
    te = timed(lambda: split_allreduce(har, tiny, 32 / 4096), args.reps)
    out["e_sync_roundtrip_64B"] = rank_max(te)

    # f: NCCL alone vs with background PCIe traffic on every GPU
    for s in (10, 40, 80):
        t = torch.randn(int(s * MiB) // 2, device=dev).to(torch.bfloat16)
        alone = timed(lambda: dist.all_reduce(t), args.reps)

        def bg():  # ~ enough background copies for the measured window, both directions, side streams
            for _ in range(6):
                cm.memcpy(s1, slot, g.data_ptr(), 4 * MiB)
                cm.memcpy(s2, g.data_ptr(), result, 4 * MiB)
        dist.barrier()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.reps):
            bg()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            dist.all_reduce(t)
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
            torch.cuda.synchronize()
        out[f"f_nccl_alone|{s}"] = rank_max(alone)
        out[f"f_nccl_with_pcie_bg|{s}"] = rank_max(statistics.median(ts))
        del t

    # d: host reduction (separate process, rank 0 only)
    if rank == 0:
        ctx = torch.multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=reducer_bench, args=([0.1, 0.4, 1, 4, 16], 10, q))
        p.start()
        out["d_reduce_ms"] = q.get(timeout=600)
        p.join()
    dist.barrier()
    if rank == 0:
        summ = {}
        for k, v in out.items():
            if isinstance(v, list):
                summ[k] = dict(rank_max_ms=max(v), per_rank_ms=v)
                size = k.split("|")[1] if "|" in k else None
                gbps = (float(size) * MiB / (max(v) / 1e3) / 1e9) if size and k[0] in "abc" else None
                print(f"[diag] {k:<28} rank-max {max(v):8.3f} ms" + (f"  -> {gbps:6.2f} GB/s per GPU" if gbps else ""),
                      flush=True)
            else:
                summ[k] = v
                for kk, vv in v.items():
                    impl, s = kk.split("|")
                    print(f"[diag] d_reduce {impl:<8} {float(s):5.1f} MiB x 8 slots: {vv:8.3f} ms -> "
                          f"{8 * float(s) * MiB / (vv / 1e3) / 1e9:6.2f} GB/s of input", flush=True)
        with open(os.path.join(args.out, "c_diag.json"), "w") as f:
            json.dump(summ, f, indent=1)
    dist.barrier()
    har.close()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
