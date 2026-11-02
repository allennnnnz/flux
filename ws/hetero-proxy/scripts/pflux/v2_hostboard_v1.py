################################################################################
# hetero-proxy pflux V1 + V2 + V2b (2026-11-02): does a host "bulletin board" all-gather beat NCCL when GPUs have no
# P2P and every byte goes through pinned host memory? Plan: reports/20261102_pflux_plan.md sections 4 and 9.
#
# Environment: css-host-158 in PCIe-only mode for NCCL (NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_DISABLE_INTRA=1,
# results/pcieonly/check_v1). Our own path never touches NVLink by construction (only GPU<->host copies); NVLink data
# counters of all 8 GPUs are read before and after the run to prove it (and that NCCL stayed off NVLink).
#
# Items, all timed the same way (gate after flush: every stream waits on a host-memory "go" counter that rank 0 writes
# after a CPU (gloo) barrier, so all 8 GPUs start within microseconds and enqueue cost is excluded; CUDA events right
# after the gate and after the last op; rank-max of per-item medians):
#   ag_nccl            dist.all_gather_into_tensor, output [M, 5120] bf16
#   ag_board/<p>/<m>/<c> ours: each rank copies its shard D2H once into a shared /dev/shm board (cudaHostRegister'ed in
#                      all 8 processes), writes a per-chunk counter (cuStreamWriteValue64); readers wait on the counter
#                      (cuStreamWaitValue64) and copy H2D. Copy engines only, no SM, no host thread.
#                      p = own  : shard j lives on the NUMA node of GPU j (readers on the other socket read over UPI)
#                          repl : shard j is uploaded to both nodes, readers use their local copy (2x upload)
#                      m = sep  : a GPU starts downloading only after its own upload finished (never sends and receives at
#                                 once: same-GPU H2D drops 22.0 -> 6.46 GB/s while D2H runs, params.json pcie_bidirectional)
#                          ovl  : download each chunk as soon as it is posted
#                      c = chunk size in MiB (0 = whole shard)
#   rs_nccl            dist.reduce_scatter_tensor, input [M, 5120] bf16
#   rs_board/<m>       ours, variant (a) of the plan: each rank copies the 7 row blocks of its partial sum D2H into the
#                      board (placed on the READER's node), each owner downloads 7 blocks and sums on the GPU (fp32
#                      accumulate). Inherently bidirectional per GPU (7/8 up, 7/8 down); sep / ovl as above.
#   ag_sm/<p>/<m>, rs_sm/<m>  the same with the copies done by a Triton kernel (32 CTAs, 16-byte loads/stores through
#                      the host mapping) instead of the copy engines, whole shard / block per launch
#   gemm/<layer>       F.linear(x [M, K], W [N, K]) bf16 cuBLAS, Qwen2.5-32B TP8 per-rank shapes (all ranks at once)
# M in {256, 1024, 2048, 4096, 8192}; [M, 5120] bf16 = 2.5 / 10 / 20 / 40 / 80 MiB. Rounds shuffled (same order on all
# ranks), one run per NCCL mode tag (shm = SM copies, shmce = NCCL_SHM_USE_CUDA_MEMCPY=1); our items run in both tags
# (self-consistency check). Correctness: board AG output must equal NCCL AG bit for bit; board RS vs NCCL RS within
# bf16 rounding (max |diff| / max |ref| reported).
# Every op is first run once ungated (go released before enqueue): lazy NCCL communicator init, cuBLAS handles and
# Triton module loads synchronise the context and would deadlock against streams parked on the gate (smoke run
# 2026-11-02 hung in the first NCCL call; hxb hit the same with a Triton module load on 2026-10-09).
#
# Predictions (registered 2026-11-02 before any run; anchors from params.json and results/pcieonly/check_v1):
#   P1 ag_nccl busbw at 20-80 MiB: 7.0-8.0 GB/s (check_v1: 7.6 at 128 MiB, shm_intra). shmce within 15% of shm.
#   P2 best ag_board at 40 MiB: upload 5 MiB with 8 GPUs uploading (~12.6 GB/s each, T1 101-104 aggregate) ~0.4 ms +
#      download 35 MiB at 10.7 GB/s (M0 8-GPU H2D) ~3.4 ms -> ~3.8 ms, busbw ~9.5 GB/s, ratio vs ag_nccl ~1.25x
#      (range 1.1-1.45x).
#   P3 repl >= own (own: 4 of 7 shards cross the socket; NUMA-remote binding cost 18-28% in Phase 0) by 0-20%.
#   P4 sep vs ovl: |difference| < 10% (ovl pipelines but collapses H2D during the upload window).
#   P5 rs_board does NOT beat rs_nccl: sep ~ 35 MiB up at ~12.6 + 35 MiB down at 10.7 ~ 6.2 ms vs NCCL ~4.8 ms
#      (ratio 0.7-0.9x); ovl also <= 1.0x.
#   P6 gemm at M=4096 within +-25% of FLOP estimates at 240 TFLOPS: qkv 0.16, o_proj 0.11, fc1 1.2, fc2 0.6 ms.
#   P7 NVLink Data Tx+Rx delta < 1 MiB per GPU over the whole run (both tags).
#   P8 (added 2026-11-02 after the literature report scratchpad lit/hostboard_prior_art.md, before any timed run:
#      nvbandwidth issue #53 on the same platform type shows copy-engine bidirectional H2D 24.8 -> ~11.3 GB/s but
#      SM copies ~21.5 / 21.5): ag_sm/*/sep within 15% of the CE sep variants (download limited by shared lines either
#      way); ag_sm/*/ovl >= CE ovl; rs_sm/ovl >= 1.3x rs_board/ovl (CE) and 0.9-1.4x rs_nccl.
# Decision rules (registered with the predictions; plan section 4):
#   R1 best ag_board vs best NCCL AG (min over the two tags) at 20 MiB and 40 MiB both >= 1.2x -> selling point holds.
#   R2 per-layer estimate at M=4096 from the measured items: baseline = 2*AG_nccl + 2*RS_nccl + sum(gemm);
#      ours_upper = max(AG_b, qkv) + max(AG_b, fc1) + max(RS_b, o_proj) + max(RS_b, fc2), AG_b = best ag_board,
#      RS_b = min(best rs_board, RS_nccl) (perfect overlap, an upper bound); ratio >= 1.3x -> go to week 2.
#      Also reported: ours without overlap (2*AG_b + 2*RS_b + sum(gemm)).
#   R3 R1 fails -> only overlap is left (bound ~1.25x, inference) -> report to the user before any building.
# Usage (repo root): scripts/pflux/run_v2_v1.sh (guard + tmux)
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
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hxb import cudamem as cm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../common/measure"))
from clock_logger import ClockLogger  # noqa: E402

H = 5120
MS = (256, 1024, 2048, 4096, 8192)
GEMMS = {"qkv": (5120, 896), "o_proj": (640, 5120), "fc1": (5120, 6912), "fc2": (3456, 5120)}  # (K, N) per rank
NODE_CPUS = {0: list(range(0, 32)) + list(range(64, 96)), 1: list(range(32, 64)) + list(range(96, 128))}
MIB = 2 ** 20
MAXCH = 16  # counters per (src, node) / (writer, reader)
AG_VARIANTS = [(p, m, c) for p in ("own", "repl") for m in ("sep", "ovl") for c in (1, 0)]
RS_VARIANTS = ("sep", "ovl")


SM_CTAS, SM_BLOCK = 32, 2048


@triton.jit(do_not_specialize=["n"])
def _copy_k(dst, src, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n, step):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


class Ptr:
    """Raw (UVA) address for a Triton pointer argument; host memory must be registered (mapped)."""
    dtype = torch.int64

    def __init__(self, addr):
        self.addr = addr

    def data_ptr(self):
        return self.addr


def sm_copy(stream, dst, src, nbytes):
    assert nbytes % 8 == 0 and dst % 16 == 0 and src % 16 == 0
    with torch.cuda.stream(stream):
        _copy_k[(SM_CTAS,)](Ptr(dst), Ptr(src), nbytes // 8, BLOCK=SM_BLOCK)


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


def numa_pages(addr, nbytes, samples=64):
    lib = ctypes.CDLL("libnuma.so.1")
    lib.numa_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    pages = (ctypes.c_void_p * samples)(*[(addr + (nbytes - 1) * i // (samples - 1)) & ~4095 for i in range(samples)])
    status = (ctypes.c_int * samples)()
    assert lib.numa_move_pages(0, samples, pages, None, status, 0) == 0
    out = {}
    for v in status:
        out[f"N{v}"] = out.get(f"N{v}", 0) + 1
    return out


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
        if create:  # first touch on `node` -> pages allocated there (tmpfs follows the faulting task's policy)
            keep = os.sched_getaffinity(0)
            if node is not None:
                os.sched_setaffinity(0, NODE_CPUS[node])
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
    os.sched_setaffinity(0, NODE_CPUS[node_of(rank)][2 * dev % 32: 2 * dev % 32 + 2])
    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"log_{args.tag}_r{rank}.txt"), "w")

    def say(s):
        log.write(s + "\n")
        log.flush()
        if rank == 0:
            print(s, flush=True)

    nv0 = nvlink_counters() if rank == 0 else None

    # ---- boards ----
    maxshard = max(MS) // world * H * 2           # 10 MiB
    pre = f"pflux_{args.job}"
    ag_board = {}                                 # (src, node) -> region holding src's shard on node
    rs_board = {}                                 # reader -> region with 8 writer blocks, on reader's node
    if rank == 0:
        cnt = ShmRegion(f"{pre}_cnt", 4096 * 8, create=True)
    dist.barrier(cpu)
    if rank != 0:
        cnt = ShmRegion(f"{pre}_cnt", 4096 * 8, create=False)
    for k in (0, 1):
        ag_board[(rank, k)] = ShmRegion(f"{pre}_ag_{rank}_{k}", maxshard, create=True, node=k)
    rs_board[rank] = ShmRegion(f"{pre}_rs_{rank}", world * maxshard, create=True, node=node_of(rank))
    dist.barrier(cpu)
    for j in range(world):
        for k in (0, 1):
            if j != rank:
                ag_board[(j, k)] = ShmRegion(f"{pre}_ag_{j}_{k}", maxshard, create=False)
        if j != rank:
            rs_board[j] = ShmRegion(f"{pre}_rs_{j}", world * maxshard, create=False)
    for r in [cnt, *ag_board.values(), *rs_board.values()]:
        r.register()
    numa = {f"ag_{rank}_{k}": numa_pages(ag_board[(rank, k)].addr, maxshard) for k in (0, 1)}
    numa[f"rs_{rank}"] = numa_pages(rs_board[rank].addr, world * maxshard)
    say(f"[numa r{rank}] {json.dumps(numa)}")
    dist.barrier(cpu)

    # counter layout (u64 index)
    GO = 0
    def ag_idx(src, node, ch):
        return 8 + (src * 2 + node) * MAXCH + ch
    def rs_idx(wr, rd, ch):
        return 8 + world * 2 * MAXCH + (wr * world + rd) * MAXCH + ch
    assert rs_idx(world - 1, world - 1, MAXCH - 1) < 4096
    go_addr = cnt.addr + 8 * GO

    up = torch.cuda.Stream()
    down = torch.cuda.Stream()
    main_s = torch.cuda.current_stream()

    # ---- tensors ----
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    X = {M: torch.randn(M // world, H, dtype=torch.bfloat16, device="cuda", generator=g) for M in MS}
    P = {M: torch.randn(M, H, dtype=torch.bfloat16, device="cuda", generator=g) for M in MS}
    OUT = {M: torch.empty(M, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    REF = {M: torch.empty(M, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    RECV = {M: torch.empty(world - 1, M // world, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    RSO = {M: torch.empty(M // world, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    RSREF = {M: torch.empty(M // world, H, dtype=torch.bfloat16, device="cuda") for M in MS}
    W = {name: torch.randn(N, K, dtype=torch.bfloat16, device="cuda", generator=g) * 0.02
         for name, (K, N) in GEMMS.items()}
    XG = {(name, M): torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g)
          for name, (K, N) in GEMMS.items() for M in MS}

    epoch = [0]

    def gate(stream):
        cm.wait_geq(stream, go_addr, epoch[0])

    def chunks(nbytes, ch_mib):
        step = nbytes if ch_mib == 0 else min(nbytes, ch_mib * MIB)
        n = (nbytes + step - 1) // step
        assert n <= MAXCH
        return [(i * step, min(step, nbytes - i * step)) for i in range(n)]

    # ---- ops: each enqueues gated work and returns (start_event, end_event) ----
    def op_ag_nccl(M):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(main_s)
        a.record(main_s)
        dist.all_gather_into_tensor(OUT[M], X[M])
        b.record(main_s)
        return a, b

    def op_ag_board(M, place, mode, ch, eng="ce"):
        copy = cm.memcpy if eng == "ce" else sm_copy
        e = epoch[0]
        x, out = X[M], OUT[M]
        nb = x.numel() * 2
        cks = chunks(nb, ch)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(up)
        gate(down)
        a.record(up)
        nodes = (node_of(rank), 1 - node_of(rank)) if place == "repl" else (node_of(rank),)
        for ci, (off, n) in enumerate(cks):
            for k in nodes:
                copy(up, ag_board[(rank, k)].addr + off, x.data_ptr() + off, n)
                cm.write64(up, cnt.addr + 8 * ag_idx(rank, k, ci), e)
        up_done = torch.cuda.Event()
        up_done.record(up)
        with torch.cuda.stream(up):
            out[rank * (M // world):(rank + 1) * (M // world)].copy_(x)
        up_end = torch.cuda.Event()
        up_end.record(up)
        if mode == "sep":
            down.wait_event(up_done)
        for s in range(1, world):
            j = (rank + s) % world
            k = node_of(rank) if place == "repl" else node_of(j)
            dst = out[j * (M // world):].data_ptr()
            for ci, (off, n) in enumerate(cks):
                cm.wait_geq(down, cnt.addr + 8 * ag_idx(j, k, ci), e)
                copy(down, dst + off, ag_board[(j, k)].addr + off, n)
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

    def op_rs_board(M, mode, eng="ce"):
        copy = cm.memcpy if eng == "ce" else sm_copy
        e = epoch[0]
        p = P[M]
        rows = M // world
        nb = rows * H * 2
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gate(up)
        gate(down)
        a.record(up)
        for s in range(1, world):  # send block for reader j into j's board, slot `rank`
            j = (rank + s) % world
            copy(up, rs_board[j].addr + rank * maxshard, p[j * rows:].data_ptr(), nb)
            cm.write64(up, cnt.addr + 8 * rs_idx(rank, j, 0), e)
        up_done = torch.cuda.Event()
        up_done.record(up)
        if mode == "sep":
            down.wait_event(up_done)
        recv = RECV[M]
        for s in range(1, world):
            i = (rank - s) % world  # writers in the order they target us first
            cm.wait_geq(down, cnt.addr + 8 * rs_idx(i, rank, 0), e)
            copy(down, recv[s - 1].data_ptr(), rs_board[rank].addr + i * maxshard, nb)
        with torch.cuda.stream(down):
            acc = torch.sum(recv, dim=0, dtype=torch.float32)
            acc += p[rank * rows:(rank + 1) * rows]
            RSO[M].copy_(acc)
        down.wait_event(up_done)
        b.record(down)
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
        for (p_, m_, c_) in AG_VARIANTS:
            items.append((f"ag_board/{p_}/{m_}/{c_}", M, lambda M=M, p_=p_, m_=m_, c_=c_: op_ag_board(M, p_, m_, c_)))
        for p_ in ("own", "repl"):
            for m_ in ("sep", "ovl"):
                items.append((f"ag_sm/{p_}/{m_}", M, lambda M=M, p_=p_, m_=m_: op_ag_board(M, p_, m_, 0, "sm")))
        items.append(("rs_nccl", M, lambda M=M: op_rs_nccl(M)))
        for m_ in RS_VARIANTS:
            items.append((f"rs_board/{m_}", M, lambda M=M, m_=m_: op_rs_board(M, m_)))
            items.append((f"rs_sm/{m_}", M, lambda M=M, m_=m_: op_rs_board(M, m_, "sm")))
        for name in GEMMS:
            items.append((f"gemm/{name}", M, lambda M=M, name=name: op_gemm(name, M)))

    def run(fn):
        epoch[0] += 1
        torch.cuda.synchronize()
        a, b = fn()
        dist.barrier(cpu)                      # everyone has enqueued behind the gate
        if rank == 0:
            cnt.u64(GO).value = epoch[0]       # release all GPUs at once
        torch.cuda.synchronize()
        return a.elapsed_time(b)

    def run_open(fn):
        """Same op with the gate released before anything is enqueued (lazy inits may synchronise safely)."""
        epoch[0] += 1
        torch.cuda.synchronize()
        dist.barrier(cpu)
        if rank == 0:
            cnt.u64(GO).value = epoch[0]
        dist.barrier(cpu)
        fn()
        torch.cuda.synchronize()
        dist.barrier(cpu)

    tmp = torch.empty(2 ** 20, dtype=torch.uint8, device="cuda")
    sm_copy(main_s, tmp.data_ptr(), tmp.data_ptr() + 2 ** 19, 2 ** 19)  # compile + load the copy kernel now
    torch.cuda.synchronize()
    for _, _, fn in items:
        run_open(fn)
    say(f"[warm r{rank}] ungated pass done")

    # ---- correctness (gated, once per item before timing) ----
    check = {}
    for M in MS:
        run(lambda M=M: op_ag_nccl(M))
        REF[M].copy_(OUT[M])
        run(lambda M=M: op_rs_nccl(M))
        RSREF[M].copy_(RSO[M])
        for n_, M_, fn in items:
            if M_ != M or not n_.startswith(("ag_board", "ag_sm", "rs_board", "rs_sm")):
                continue
            if n_.startswith("ag"):
                OUT[M].zero_()
                run(fn)
                check[f"{n_}|{M}"] = bool(torch.equal(OUT[M], REF[M]))
            else:
                RSO[M].zero_()
                run(fn)
                ref = RSREF[M].float()
                check[f"{n_}|{M}"] = float((RSO[M].float() - ref).abs().max() / ref.abs().max())
    say(f"[check r{rank}] {json.dumps(check)}")

    # ---- warmup + timed rounds ----
    for _ in range(2):
        for _, _, fn in items:
            run(fn)
    rng = random.Random(42)
    times = {(n, M): [] for n, M, _ in items}
    windows = {(n, M): [] for n, M, _ in items}
    t_start = time.time()
    clk = ClockLogger(dev, period_s=0.1) if rank == 0 else None
    if clk:
        clk.__enter__()
    for r in range(args.rounds):
        order = list(range(len(items)))
        rng.shuffle(order)
        for idx in order:
            n, M, fn = items[idx]
            t0 = time.perf_counter()
            ms = run(fn)
            times[(n, M)].append(ms)
            windows[(n, M)].append((t0, time.perf_counter()))
    say(f"[timing r{rank}] {args.rounds} rounds x {len(items)} items in {time.time() - t_start:.1f} s")
    box = [None]
    if rank == 0:
        clk.__exit__()
        clocks = {k: [clk.min_clock_between(*w) for w in v] for k, v in windows.items()}
        allc = [c for v in clocks.values() for c in v if c]
        modal = statistics.mode(allc) if allc else None
        box = [dict(clocks=clocks, modal=modal, n_samples=len(clk.samples),
                    min_clock=min(allc) if allc else None)]
    dist.broadcast_object_list(box, src=0, group=cpu)
    clocks, modal = box[0]["clocks"], box[0]["modal"]
    res = {}
    for (n, M), ts in times.items():
        keep = [t for t, c in zip(ts, clocks[(n, M)]) if c is None or modal is None or c >= 0.95 * modal]
        med = statistics.median(keep)
        t = torch.tensor([med, len(ts) - len(keep)], device="cuda", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        allmed = [None] * world
        dist.all_gather_object(allmed, med, group=cpu)
        res[f"{n}|{M}"] = dict(ms_rankmax=t[0].item(), dropped=int(t[1].item()),
                               ms_per_rank=[round(x, 4) for x in allmed],
                               p10=sorted(keep)[len(keep) // 10], p90=sorted(keep)[len(keep) * 9 // 10])
    raw = {f"{n}|{M}": [round(t, 4) for t in ts] for (n, M), ts in times.items()}
    json.dump(dict(rank=rank, raw_ms=raw),
              open(os.path.join(args.out, f"raw_{args.tag}_r{rank}.json"), "w"))
    checks = [None] * world
    dist.all_gather_object(checks, check, group=cpu)
    if rank == 0:
        nv1 = nvlink_counters()
        nvd = {g_: nv1[1][g_] - nv0[1][g_] for g_ in range(8)}
        json.dump(dict(tag=args.tag, args=vars(args), env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                       nccl=torch.cuda.nccl.version(), clock=dict(modal=modal, min=box[0]["min_clock"],
                       n_samples=box[0]["n_samples"]), clocks_rank0={f"{n}|{M}": c for (n, M), c in clocks.items()},
                       results=res, checks=checks,
                       nvlink_delta_KiB=nvd, nvlink_raw_before=nv0[0], nvlink_raw_after=nv1[0]),
                  open(os.path.join(args.out, f"v2_{args.tag}.json"), "w"), indent=1)
        say(f"[nvlink] Data Tx+Rx delta KiB per GPU: {nvd}")
        for k, v in sorted(res.items(), key=lambda kv: (int(kv[0].split('|')[1]), kv[0])):
            say(f"[{args.tag}] {k:32s} {v['ms_rankmax']:8.3f} ms  (p10 {v['p10']:.3f} p90 {v['p90']:.3f}, dropped {v['dropped']})")
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
