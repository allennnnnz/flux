################################################################################
# hetero-proxy pflux V2c (2026-11-02): reduce-scatter through host memory with the H2D half pushed by CPU threads
# (GDRCopy into a BAR1-mapped receive buffer) instead of the GPU copy engine. V2 (results/pflux/v2_full) found the
# copy-engine board RS limited to the contended H2D rate (6.45 GB/s per direction, 0.87-0.88x NCCL), which made R2 fail
# (1.29x < 1.3x). V3 found SM copies worse; V3c (results/pflux/v3c_*) found CPU pushes keep ~10 GB/s per GPU in both
# directions on 8 GPUs while the copy engine uploads.
#
# Per rank (reader j): its 7 upload blocks go D2H by copy engine into the readers' host boards (as V2 rs_board) with a
# host counter per (writer, reader); T pusher threads of rank j (pinned to j's NUMA node) spin on the counters of their
# writers, push each block with gdr_copy_to_mapping into j's receive buffer, then push their flag (8 bytes, same
# mapping, after the data); j's GPU stream waits on the flags (cuStreamWaitValue64 GEQ, + FLUSH of remote writes when
# the device supports it) and sums the 7 blocks + its own (fp32 accumulate), as rs_board.
# Same harness as V2 (gate after flush, rank-max of per-rank medians, shuffled rounds, NVLink counters, guard), with
# the V2 reference items in the same rounds: ag_nccl, the best V2 board AG (own/ovl/whole), rs_nccl, rs_board/ovl,
# gemm/* (so R2 is recomputed from one run).
#
# Predictions (registered 2026-11-02 after the V3c smoke run, before any V2c run):
#   C1 rs_push matches rs_nccl within bf16 rounding: max |diff| / max |ref| <= 1.1e-2 (V2 rs_board: 1.03e-2).
#   C2 rs_push at M=4096 (35 MiB each way per GPU): 3.9-4.6 ms (CE upload and CPU push ~10 GB/s each, concurrent,
#      plus ~0.5 ms to get the first block up) -> 1.08-1.28x rs_nccl (V2: 4.993 ms).
#   C3 T=2 is the fastest pusher count at M >= 2048 (V3c smoke on 8 GPUs: T=2 9.7-11.0, T=4 8.8-9.4 GB/s).
#   C4 reference items reproduce V2 within 5% (ag_nccl 4.886, ag_board/own/ovl/0 3.418, rs_nccl 4.993,
#      rs_board/ovl 5.694 ms at M=4096).
# Decision (V3 rule): R2 recomputed with RS_b = min(rs_push, rs_board, rs_nccl) >= 1.3x at M=4096 -> the plan can go
# to week 2 (after the user agrees); otherwise stop at the all-gather-only result.
# Usage (repo root): scripts/pflux/run_v2c_v1.sh (guard + tmux)
################################################################################
import argparse
import ctypes
import json
import mmap
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hxb import cudamem as cm  # noqa: E402

H = 5120
MS = (256, 1024, 2048, 4096, 8192)
GEMMS = {"qkv": (5120, 896), "o_proj": (640, 5120), "fc1": (5120, 6912), "fc2": (3456, 5120)}  # (K, N) per rank
PHYS = {0: list(range(0, 32)), 1: list(range(32, 64))}
MIB = 2 ** 20
MAXCH = 16
GPU_PAGE = 1 << 16
TS = (1, 2)
WAIT_GEQ, WAIT_FLUSH = 0x0, 1 << 30


class MH(ctypes.Structure):
    _fields_ = [("h", ctypes.c_ulong)]


class Info(ctypes.Structure):
    _fields_ = [("va", ctypes.c_uint64), ("mapped_size", ctypes.c_uint64), ("page_size", ctypes.c_uint32),
                ("tm_cycles", ctypes.c_uint64), ("cycles_per_ms", ctypes.c_uint32), ("bits", ctypes.c_uint32),
                ("mapping_type", ctypes.c_int)]


gdr = ctypes.CDLL("libgdrapi.so.2")
gdr.gdr_open.restype = ctypes.c_void_p
gdr.gdr_pin_buffer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint32,
                               ctypes.POINTER(MH)]
gdr.gdr_map.argtypes = [ctypes.c_void_p, MH, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
gdr.gdr_get_info_v2.argtypes = [ctypes.c_void_p, MH, ctypes.POINTER(Info)]
gdr.gdr_copy_to_mapping.argtypes = [MH, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
libcu = ctypes.CDLL("libcuda.so.1")
libcu.cuPointerSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
libcu.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
libcu.cuStreamWaitValue64_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint]


def node_of(rank):
    return rank // 4


def nvlink_counters():
    raw, tot = {}, {}
    for g in range(8):
        out = subprocess.run(["nvidia-smi", "nvlink", "-gt", "d", "-i", str(g)], capture_output=True, text=True,
                             timeout=30).stdout
        raw[g] = out
        tot[g] = sum(int(v) for v in re.findall(r"Data [TR]x: (\d+) KiB", out))
    return raw, tot


class ShmRegion:
    """A /dev/shm file mapped MAP_SHARED in every rank and registered with CUDA (pinned + mapped) in each process."""

    def __init__(self, name, nbytes, create, node=None):
        self.path, self.nbytes = f"/dev/shm/{name}", nbytes
        if create:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            os.ftruncate(fd, nbytes)
        else:
            fd = os.open(self.path, os.O_RDWR)
        self.mm = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        if create:
            keep = os.sched_getaffinity(0)
            if node is not None:
                os.sched_setaffinity(0, PHYS[node])
            ctypes.memset(self.addr, 0, nbytes)
            os.sched_setaffinity(0, keep)

    def register(self):
        cm.register(self.addr, self.nbytes)

    def u64(self, idx):
        return ctypes.c_uint64.from_address(self.addr + 8 * idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--job", default=str(os.getppid()))
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    dev = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(dev)
    cpu = dist.new_group(backend="gloo")
    li, node = rank % 4, node_of(rank)
    main_cpus = [PHYS[node][2 * li], PHYS[node][2 * li + 1]]           # node 0: 0-7, node 1: 32-39
    push_cpus = [PHYS[node][8 + 4 * li + t] for t in range(4)]          # node 0: 8-23, node 1: 40-55
    os.sched_setaffinity(0, main_cpus)
    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"log_{args.tag}_r{rank}.txt"), "w")

    def say(s):
        log.write(s + "\n")
        log.flush()
        if rank == 0:
            print(s, flush=True)

    nv0 = nvlink_counters() if rank == 0 else None

    # ---- host boards (as V2) ----
    maxshard = max(MS) // world * H * 2
    pre = f"pflux_{args.job}"
    ag_board, rs_board = {}, {}
    if rank == 0:
        cnt = ShmRegion(f"{pre}_cnt", 4096 * 8, create=True)
    dist.barrier(cpu)
    if rank != 0:
        cnt = ShmRegion(f"{pre}_cnt", 4096 * 8, create=False)
    ag_board[rank] = ShmRegion(f"{pre}_ag_{rank}", maxshard, create=True, node=node)
    rs_board[rank] = ShmRegion(f"{pre}_rs_{rank}", world * maxshard, create=True, node=node)
    dist.barrier(cpu)
    for j in range(world):
        if j != rank:
            ag_board[j] = ShmRegion(f"{pre}_ag_{j}", maxshard, create=False)
            rs_board[j] = ShmRegion(f"{pre}_rs_{j}", world * maxshard, create=False)
    for r in [cnt, *ag_board.values(), *rs_board.values()]:
        r.register()
    dist.barrier(cpu)
    GO = 0

    def ag_idx(src, ch):
        return 8 + src * MAXCH + ch

    def rs_idx(wr, rd):
        return 8 + world * MAXCH + wr * world + rd

    go_addr = cnt.addr + 8 * GO

    # ---- GDR receive buffer: 7 blocks + one flag line per pusher thread ----
    flags_off = (world - 1) * maxshard
    rb_bytes = flags_off + GPU_PAGE
    rb_raw = torch.zeros(rb_bytes + GPU_PAGE, dtype=torch.uint8, device="cuda")
    rb_base = (rb_raw.data_ptr() + GPU_PAGE - 1) & ~(GPU_PAGE - 1)
    rb_off = rb_base - rb_raw.data_ptr()
    one = ctypes.c_uint32(1)
    assert libcu.cuPointerSetAttribute(ctypes.byref(one), 6, rb_base) == 0  # CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    g_h = gdr.gdr_open()
    mh = MH()
    assert gdr.gdr_pin_buffer(g_h, rb_base, rb_bytes, 0, 0, ctypes.byref(mh)) == 0
    va = ctypes.c_void_p()
    assert gdr.gdr_map(g_h, mh, ctypes.byref(va), rb_bytes) == 0
    inf = Info()
    gdr.gdr_get_info_v2(g_h, mh, ctypes.byref(inf))
    rb_map = va.value + (rb_base - inf.va)
    can_flush = ctypes.c_int(0)
    libcu.cuDeviceGetAttribute(ctypes.byref(can_flush), 98, dev)  # CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES
    wait_flags = WAIT_GEQ | (WAIT_FLUSH if can_flush.value else 0)
    say(f"[gdr r{rank}] mapping_type {inf.mapping_type} can_flush_remote_writes {can_flush.value} "
        f"push cpus {push_cpus} main cpus {main_cpus}")

    def flag_dev(t):
        return rb_base + flags_off + 64 * t

    def wait_flag(stream, t, e):
        r = libcu.cuStreamWaitValue64_v2(ctypes.c_void_p(stream.cuda_stream), flag_dev(t), e, wait_flags)
        assert r == 0, f"cuStreamWaitValue64 {r}"

    # ---- pusher threads ----
    job = {"e": 0, "M": 0, "T": 0}
    job_cv = threading.Condition()
    done_cnt = [0]
    vals = [ctypes.c_uint64(0) for _ in range(4)]

    def pusher(t):
        os.sched_setaffinity(0, [push_cpus[t]])
        seen = 0
        while True:
            with job_cv:
                while job["e"] == seen:
                    job_cv.wait()
                e, M, T = job["e"], job["M"], job["T"]
                seen = e
            if e < 0:
                return
            if t >= T:
                continue
            rows = M // world
            nb = rows * H * 2
            for s in range(1, world):
                if (s - 1) % T != t:
                    continue
                i = (rank - s) % world
                c = cnt.u64(rs_idx(i, rank))
                while c.value < e:
                    time.sleep(0)
                gdr.gdr_copy_to_mapping(mh, rb_map + (s - 1) * maxshard, rs_board[rank].addr + i * maxshard, nb)
            vals[t].value = e
            gdr.gdr_copy_to_mapping(mh, rb_map + flags_off + 64 * t, ctypes.addressof(vals[t]), 8)

    threads = [threading.Thread(target=pusher, args=(t,), daemon=True) for t in range(max(TS))]
    for th in threads:
        th.start()

    def dispatch(e, M, T):
        with job_cv:
            job.update(e=e, M=M, T=T)
            job_cv.notify_all()

    up = torch.cuda.Stream()
    down = torch.cuda.Stream()
    main_s = torch.cuda.current_stream()

    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    X = {M: torch.randn(M // world, H, dtype=torch.bfloat16, device="cuda", generator=gen) for M in MS}
    P = {M: torch.randn(M, H, dtype=torch.bfloat16, device="cuda", generator=gen) for M in MS}
    OUT = {M: torch.empty(M, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    REF = {M: torch.empty(M, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    RECV = {M: torch.empty(world - 1, M // world, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    RSO = {M: torch.empty(M // world, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    RSREF = {M: torch.empty(M // world, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    W = {name: torch.randn(N, K, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.02
         for name, (K, N) in GEMMS.items()}
    XG = {(name, M): torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=gen)
          for name, (K, N) in GEMMS.items() for M in MS}
    rb_bf16 = rb_raw[rb_off:rb_off + flags_off].view(torch.bfloat16)
    RBV = {M: torch.as_strided(rb_bf16, (world - 1, M // world, H), (maxshard // 2, H, 1)) for M in MS}

    epoch = [0]

    def gate(stream):
        cm.wait_geq(stream, go_addr, epoch[0])

    def op_ag_nccl(M):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(main_s)
        a.record(main_s)
        dist.all_gather_into_tensor(OUT[M], X[M])
        b.record(main_s)
        return a, b

    def op_ag_board(M):  # V2 best variant: own placement, download as soon as posted, whole shard
        e = epoch[0]
        x, out = X[M], OUT[M]
        nb = x.numel() * 2
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(up)
        gate(down)
        a.record(up)
        cm.memcpy(up, ag_board[rank].addr, x.data_ptr(), nb)
        cm.write64(up, cnt.addr + 8 * ag_idx(rank, 0), e)
        with torch.cuda.stream(up):
            out[rank * (M // world):(rank + 1) * (M // world)].copy_(x)
        up_end = torch.cuda.Event()
        up_end.record(up)
        for s in range(1, world):
            j = (rank + s) % world
            cm.wait_geq(down, cnt.addr + 8 * ag_idx(j, 0), e)
            cm.memcpy(down, out[j * (M // world):].data_ptr(), ag_board[j].addr, nb)
        down.wait_event(up_end)
        b.record(down)
        return a, b

    def op_rs_nccl(M):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(main_s)
        a.record(main_s)
        dist.reduce_scatter_tensor(RSO[M], P[M])
        b.record(main_s)
        return a, b

    def upload_blocks(M, e):
        p = P[M]
        rows = M // world
        nb = rows * H * 2
        for s in range(1, world):
            j = (rank + s) % world
            cm.memcpy(up, rs_board[j].addr + rank * maxshard, p[j * rows:].data_ptr(), nb)
            cm.write64(up, cnt.addr + 8 * rs_idx(rank, j), e)
        ev = torch.cuda.Event()
        ev.record(up)
        return ev

    def op_rs_board(M):  # V2 rs_board/ovl (copy-engine download)
        e = epoch[0]
        rows = M // world
        nb = rows * H * 2
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(up)
        gate(down)
        a.record(up)
        up_done = upload_blocks(M, e)
        recv = RECV[M]
        for s in range(1, world):
            i = (rank - s) % world
            cm.wait_geq(down, cnt.addr + 8 * rs_idx(i, rank), e)
            cm.memcpy(down, recv[s - 1].data_ptr(), rs_board[rank].addr + i * maxshard, nb)
        with torch.cuda.stream(down):
            acc = torch.sum(recv, dim=0, dtype=torch.float32)
            acc += P[M][rank * rows:(rank + 1) * rows]
            RSO[M].copy_(acc)
        down.wait_event(up_done)
        b.record(down)
        return a, b

    def op_rs_push(M, T):
        e = epoch[0]
        rows = M // world
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(up)
        gate(down)
        a.record(up)
        up_done = upload_blocks(M, e)
        for t in range(T):
            wait_flag(down, t, e)
        with torch.cuda.stream(down):
            acc = torch.sum(RBV[M], dim=0, dtype=torch.float32)
            acc += P[M][rank * rows:(rank + 1) * rows]
            RSO[M].copy_(acc)
        down.wait_event(up_done)
        b.record(down)
        dispatch(e, M, T)  # pushers start spinning on the counters (the GPU is still parked on the gate)
        return a, b

    def op_gemm(name, M):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(main_s)
        a.record(main_s)
        F.linear(XG[(name, M)], W[name])
        b.record(main_s)
        return a, b

    items = []
    for M in MS:
        items.append(("ag_nccl", M, lambda M=M: op_ag_nccl(M)))
        items.append(("ag_board/own/ovl/0", M, lambda M=M: op_ag_board(M)))
        items.append(("rs_nccl", M, lambda M=M: op_rs_nccl(M)))
        items.append(("rs_board/ovl", M, lambda M=M: op_rs_board(M)))
        for T in TS:
            items.append((f"rs_push/T{T}", M, lambda M=M, T=T: op_rs_push(M, T)))
        for name in GEMMS:
            items.append((f"gemm/{name}", M, lambda M=M, name=name: op_gemm(name, M)))

    def run(fn):
        epoch[0] += 1
        torch.cuda.synchronize()
        a, b = fn()
        dist.barrier(cpu)
        if rank == 0:
            cnt.u64(GO).value = epoch[0]
        torch.cuda.synchronize()
        return a.elapsed_time(b)

    def run_open(fn):
        epoch[0] += 1
        torch.cuda.synchronize()
        dist.barrier(cpu)
        if rank == 0:
            cnt.u64(GO).value = epoch[0]
        dist.barrier(cpu)
        fn()
        torch.cuda.synchronize()
        dist.barrier(cpu)

    for _, _, fn in items:
        run_open(fn)
    say(f"[warm r{rank}] ungated pass done")

    check = {}
    for M in MS:
        run(lambda M=M: op_ag_nccl(M))
        REF[M].copy_(OUT[M])
        run(lambda M=M: op_rs_nccl(M))
        RSREF[M].copy_(RSO[M])
        OUT[M].zero_()
        run(lambda M=M: op_ag_board(M))
        check[f"ag_board/own/ovl/0|{M}"] = bool(torch.equal(OUT[M], REF[M]))
        ref = RSREF[M].float()
        for n_, fn in [("rs_board/ovl", lambda M=M: op_rs_board(M))] + \
                      [(f"rs_push/T{T}", lambda M=M, T=T: op_rs_push(M, T)) for T in TS]:
            RSO[M].zero_()
            rb_raw[rb_off:rb_off + flags_off].zero_()  # stale blocks would equal the new ones (same P): clear them
            RECV[M].zero_()
            torch.cuda.synchronize()
            run(fn)
            check[f"{n_}|{M}"] = float((RSO[M].float() - ref).abs().max() / ref.abs().max())
    say(f"[check r{rank}] {json.dumps(check)}")

    rng = random.Random(43)
    times = {(n, M): [] for n, M, _ in items}
    t_start = time.time()
    for r in range(args.rounds):
        order = list(range(len(items)))
        rng.shuffle(order)
        for idx in order:
            n, M, fn = items[idx]
            times[(n, M)].append(run(fn))
    say(f"[timing r{rank}] {args.rounds} rounds x {len(items)} items in {time.time() - t_start:.1f} s")
    res = {}
    for (n, M), ts in times.items():
        med = statistics.median(ts)
        t = torch.tensor([med], device="cuda", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        allmed = [None] * world
        dist.all_gather_object(allmed, med, group=cpu)
        s_ = sorted(ts)
        res[f"{n}|{M}"] = dict(ms_rankmax=t.item(), ms_per_rank=[round(x, 4) for x in allmed],
                               p10=s_[len(s_) // 10], p90=s_[len(s_) * 9 // 10])
    json.dump(dict(rank=rank, raw_ms={f"{n}|{M}": [round(x, 4) for x in ts] for (n, M), ts in times.items()}),
              open(os.path.join(args.out, f"raw_{args.tag}_r{rank}.json"), "w"))
    checks = [None] * world
    dist.all_gather_object(checks, check, group=cpu)
    if rank == 0:
        nv1 = nvlink_counters()
        nvd = {g_: nv1[1][g_] - nv0[1][g_] for g_ in range(8)}
        json.dump(dict(tag=args.tag, args=vars(args), env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                       nccl=torch.cuda.nccl.version(), wait_flags=wait_flags, results=res, checks=checks,
                       nvlink_delta_KiB=nvd), open(os.path.join(args.out, f"v2c_{args.tag}.json"), "w"), indent=1)
        say(f"[nvlink] Data Tx+Rx delta KiB per GPU: {nvd}")
        for k, v in sorted(res.items(), key=lambda kv: (int(kv[0].split('|')[1]), kv[0])):
            say(f"[{args.tag}] {k:24s} {v['ms_rankmax']:8.3f} ms  (p10 {v['p10']:.3f} p90 {v['p90']:.3f})")
    dispatch(-1, 0, 0)
    dist.barrier(cpu)
    for r in [cnt, *ag_board.values(), *rs_board.values()]:
        cm.unregister(r.addr)
    if rank == 0:
        for f in os.listdir("/dev/shm"):
            if f.startswith(pre):
                os.unlink(os.path.join("/dev/shm", f))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
