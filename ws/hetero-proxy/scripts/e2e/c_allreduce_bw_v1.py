################################################################################
# hetero-proxy E2E step 3 (2026-10-14): configuration C's building block - an 8-GPU all-reduce that sends a fraction f
# of the tensor over PCIe through host memory while NCCL (NVLink) reduces the rest, both at the same time
# (FlexLink-style host-staged path). Question: is there any f > 0 for which the split all-reduce is faster than
# NCCL alone? (Prefill TP all-reduce = M x 5120 bf16: 10 MiB at M=1024 ... 160 MiB at M=16384; decode <= 0.6 MiB.)
# PCIe path, per rank, on its own stream: wait for the data -> D2H its tail t[s:] into its slot of a shared host
# buffer -> write deposited[rank] = k (stream memory op) -> wait reduced >= k (cuStreamWaitValue on host memory)
# -> H2D the reduced tail back. A reducer process (spawned by rank 0, NUMA-0 hyper-thread cpus 64-79, 16 threads)
# waits for all 8 deposits, sums the 8 slots in fp32 and stores bf16. Main stream: NCCL all_reduce of t[:s], then
# waits for the PCIe part. Timed per op on every rank with CUDA events (main stream, after both parts); reported as
# the rank-max median over `rounds`; f values interleaved per round. Correctness: max |split - NCCL-only| per case.
# PRE-REGISTERED [推論, STATUS 0]: no f > 0 is more than 3% faster than f = 0 at any size; at <= 10 MiB every f > 0 is
# slower (the host path's fixed latency exceeds NCCL's whole time).
# Usage: env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
#        ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 c_allreduce_bw_v1.py --out <dir>
################################################################################
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from hxb import cudamem as cm  # noqa: E402

W = 8
HDR = 4096  # counters: deposited[0..7] at int64 slots 8..15, reduced at slot 16, sizes ring at 64..64+255


def reducer_main(path, total, slot_bytes, cpus, threads):
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(threads)
    buf = torch.from_file(path, shared=True, size=total, dtype=torch.uint8)
    ctr = buf[:HDR].view(torch.int64).numpy()
    k = 0
    while True:
        k += 1
        while True:
            if ctr[1] != 0:  # stop flag
                return
            if all(ctr[8 + r] >= k for r in range(W)):
                break
            time.sleep(2e-6)
        nbytes = int(ctr[64 + (k % 256)])
        n = nbytes // 2
        acc = None
        for r in range(W):
            x = buf[HDR + r * slot_bytes: HDR + r * slot_bytes + nbytes].view(torch.bfloat16)[:n].float()
            acc = x if acc is None else acc.add_(x)
        res = buf[HDR + W * slot_bytes: HDR + W * slot_bytes + nbytes].view(torch.bfloat16)[:n]
        res.copy_(acc)
        ctr[16] = k


class HostAllReduce:
    def __init__(self, rank, dev, max_bytes):
        self.rank, self.dev = rank, dev
        self.slot = (max_bytes + 4095) // 4096 * 4096
        total = HDR + (W + 1) * self.slot
        name = [f"/dev/shm/hxb_c_{os.getpid()}"] if rank == 0 else [None]
        dist.broadcast_object_list(name, src=0)
        self.path = name[0]
        if rank == 0:
            with open(self.path, "wb") as f:
                f.truncate(total)
        dist.barrier()
        self.buf = torch.from_file(self.path, shared=True, size=total, dtype=torch.uint8)
        if rank == 0:
            self.buf.zero_()
        dist.barrier()
        cm.register(self.buf.data_ptr(), total)
        self.base = self.buf.data_ptr()
        self.ctr = self.buf[:HDR].view(torch.int64).numpy()
        self.k = 0
        self.stream = torch.cuda.Stream(dev)
        self.proc = None
        if rank == 0:
            ctx = torch.multiprocessing.get_context("spawn")
            self.proc = ctx.Process(target=reducer_main, args=(self.path, total, self.slot, list(range(64, 80)), 16),
                                    daemon=True)
            self.proc.start()
        dist.barrier()

    def tail(self, t, s, ready):
        """Enqueue the PCIe part for t[s:] (1-D bf16 GPU tensor) on this rank's stream; returns its done event."""
        self.k += 1
        k, nb = self.k, (t.numel() - s) * 2
        if self.rank == 0:
            self.ctr[64 + (k % 256)] = nb
        st = self.stream
        st.wait_event(ready)
        cm.memcpy(st, self.base + HDR + self.rank * self.slot, t.data_ptr() + 2 * s, nb)
        cm.write64(st, self.base + 8 * (8 + self.rank), k)
        cm.wait_geq(st, self.base + 8 * 16, k)
        cm.memcpy(st, t.data_ptr() + 2 * s, self.base + HDR + W * self.slot, nb)
        done = torch.cuda.Event()
        done.record(st)
        return done

    def close(self):
        if self.rank == 0:
            self.ctr[1] = 1
            self.proc.join(timeout=10)
            os.unlink(self.path)


def split_allreduce(har, t, f):
    """NCCL all-reduce of t[:s] over NVLink while the host path reduces t[s:] over PCIe."""
    n = t.numel()
    s = n if f <= 0 else max(0, min(n, int(round(n * (1 - f) / 64)) * 64))
    if s == n:
        dist.all_reduce(t)
        return
    ready = torch.cuda.Event()
    ready.record()
    done = har.tail(t, s, ready)
    if s > 0:
        dist.all_reduce(t[:s])
    torch.cuda.current_stream().wait_event(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes_mib", default="0.625,2.5,10,40,80,160")
    ap.add_argument("--fracs", default="0,0.01,0.02,0.03,0.05,0.1,1")
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    os.sched_setaffinity(0, list(range(4 * rank, 4 * rank + 4)))
    torch.set_num_threads(4)
    sizes = [float(x) for x in args.sizes_mib.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]
    har = HostAllReduce(rank, dev, int(max(sizes) * 2 ** 20))
    g = torch.Generator(device=dev).manual_seed(rank)
    src = {s: (torch.randn(int(s * 2 ** 20) // 2, device=dev, generator=g) * 0.1).to(torch.bfloat16) for s in sizes}
    ref = {}
    for s in sizes:  # NCCL-only reference result
        t = src[s].clone()
        dist.all_reduce(t)
        ref[s] = t
    err = {}
    for s in sizes:
        for f in fracs:
            t = src[s].clone()
            split_allreduce(har, t, f)
            torch.cuda.synchronize()
            err[f"{s}|{f}"] = (t.float() - ref[s].float()).abs().max().item()
    times = {f"{s}|{f}": [] for s in sizes for f in fracs}
    work = {s: src[s].clone() for s in sizes}
    for r in range(args.rounds + 3):
        for s in sizes:
            for f in (fracs if r % 2 == 0 else fracs[::-1]):
                dist.barrier()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                split_allreduce(har, work[s], f)
                e1.record()
                e1.synchronize()
                if r >= 3:
                    times[f"{s}|{f}"].append(e0.elapsed_time(e1))
    allt = [None] * dist.get_world_size()
    dist.all_gather_object(allt, times)
    if rank == 0:
        res = {}
        for key in times:
            per_round = [max(allt[q][key][i] for q in range(len(allt))) for i in range(args.rounds)]
            res[key] = dict(rank_max_median_ms=statistics.median(per_round), max_abs_err=err[key])
        for s in sizes:
            base = res[f"{s}|0.0"]["rank_max_median_ms"]
            line = " ".join(f"f={f:g}:{res[f'{s}|{f}']['rank_max_median_ms']:.3f}({100 * (res[f'{s}|{f}']['rank_max_median_ms'] / base - 1):+.1f}%)"
                            for f in fracs)
            print(f"[c-bw] {s:7.3f} MiB  NCCL-only {base:.3f} ms | {line}", flush=True)
        print("[c-bw] max |split - NCCL-only| per case: " + ", ".join(f"{k}:{v['max_abs_err']:.3g}" for k, v in res.items()
                                                               if v["max_abs_err"] > 0), flush=True)
        with open(os.path.join(args.out, "c_allreduce_bw.json"), "w") as fo:
            json.dump(dict(results=res, raw_rank_ms=allt, args=vars(args)), fo, indent=1)
    dist.barrier()
    har.close()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
