################################################################################
# hetero-proxy X1d (2026-10-28): what exactly does background copy traffic slow, and why is NCCL's LL protocol hit?
# Known so far (results/xpcie/x1_v1, x1b_v1, x1c_v1): an NVLink ping-pong slows x2.1 (H2D) / x4.6 (D2H) only when each
# side uses __threadfence_system(); with __threadfence() or no fence it is unaffected. NCCL Simple fences with
# fence.acq_rel.sys per step (nccl 2.21.5 prims_simple.h:179), but LL has no fence at all (flag inside the 16-byte data
# store, prims_ll.h) and LL128 uses a GPU-scope fence on sm80 - yet x1_proto_LL / LL128 slow as much as Simple.
# Same mechanism reported in NVIDIA/nccl #2218 (2026-06, V100/H200 all-gather, closed without fix).
# One process drives GPU0, GPU1 and GPU2 (peer access enabled). GPU0 and GPU1 share a PCIe switch / x16 uplink, GPU2
# sits behind another root port on the same NUMA node (results/xpcie/topology_20261028.txt). Every victim is timed on
# GPU0 with %globaltimer inside the kernel:
#   fs_sys      GPU0 alone: k x {store a word to GPU1 memory; __threadfence_system()}  -> ns per fence (no partner)
#   fs_gpu      same with __threadfence()                                               (control)
#   pp_sys      GPU0<->GPU1 1-word ping-pong with __threadfence_system() (= x1c pp_fsys, anchor)
#   hostread    GPU0 reads a pinned host word (= x1b hostread, anchor)
#   llpp        GPU0<->GPU1 ping-pong of 16 KiB LL-style messages: 128 threads x 8 lines of {data, flag, data, flag}
#               stored with st.volatile.v4 to the peer, receiver polls the flags; NO fence (NCCL LL's data path)
#   tiny_local  CUDA graph of 1024 one-thread kernels, each stamps %globaltimer then stores a word to GPU0 memory
#   tiny_peer   same, the word goes to GPU1 memory over NVLink (no fence)  -> does a kernel END after remote stores pay
#               a system-scope flush? (each NCCL LL kernel ends after remote stores)
#   tiny_peer_fsys  same plus __threadfence_system() (positive control)
#   tiny_ev     [event-wait node, tiny_local kernel, event-record node] x 1024 on one event created with
#               cudaEventDisableTiming, captured with cudaEventWaitExternal / cudaEventRecordExternal: the exact pattern
#               NCCL adds around every captured operation when NCCL_GRAPH_MIXING_SUPPORT=1 (default;
#               nccl 2.21.5 misc/strongstream.cc:181-229, seen in results/xpcie/x1e_v1/*.dot)
#   tiny_evrec  [tiny_local kernel, event-record node] x 1024 (which of the two nodes costs)
# Background (side stream of the copying GPU, enqueued right before the victim; NUMA-0 pinned host memory):
#   none | d2h_g0 | h2d_g0 (GPU0 itself) | d2h_g0_64k (GPU0, 64 KiB pieces: lower PCIe occupancy) |
#   d2h_g1 (GPU1 only: same switch as GPU0) | d2h_g2 (GPU2 only: other switch) |
#   nvl_g0 (GPU0's copy engine pushes to GPU2 over NVLink: copy traffic that never touches PCIe)
# Per case: whether the background was still running when the victim ended (a stamp kernel queued behind the copies on
# the copying GPU vs the victim's end stamp) and the background GB/s (events on the copy stream). SM clock of GPU0 is
# sampled (common/measure/clock_logger.py); cases below 0.95 x the modal clock are dropped. Cases shuffled per round.
# PRE-REGISTERED [推論]:
#   D1 switch:   fs_sys under d2h_g1 (GPU0 not copying) >= 1.5x none; under d2h_g2 < 1.2x
#   D2 PCIe:     fs_sys under nvl_g0 < 1.3x (the fence waits for PCIe traffic, not for any copy-engine traffic)
#   D3 dose:     fs_sys slows less under d2h_g0_64k than under d2h_g0
#   D4 LL data:  llpp changes < 20% in every mode (fence-free NVLink stores are not hit, as x1c pp_none)
#   D5 kernel end: tiny_peer interval >= 2x none under d2h_g0 while tiny_local stays within 10% (would explain LL)
#   D6 controls: fs_gpu within 10% in every mode; pp_sys and hostread within 20% of x1c / x1b for the matching mode
# ADDED after the 3-round smoke (results/xpcie/x1d_smoke, 16:20, which had no event victims), before the formal run:
#   D7 event nodes: tiny_ev interval >= 2x none under d2h_g0 while tiny_local stays within 10%
#   D8 (separate run, x1_dispatch_v1.py with NCCL_GRAPH_MIXING_SUPPORT=0): no event nodes in the captured graph, and the
#      NCCL all-reduce slowdown under d2h_all falls by >= 50% vs results/xpcie/x1_v1 (+594%)
#   (smoke already showed D2 failing: fs_sys x1.73 under nvl_g0 - kept as registered)
# Usage (repo root, exclusive guard, tmux):
#   env CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml ~/.venvs/hetero-proxy-hf/bin/python \
#     ws/hetero-proxy/scripts/xpcie/x1d_scope_v1.py --out <dir> [--rounds 100]
################################################################################
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "kvtier"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "common", "measure"))
from hxb import cudamem as cm  # noqa: E402
from kvtier_v2 import node_cpus  # noqa: E402
from clock_logger import ClockLogger  # noqa: E402

MiB = 2 ** 20
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void fence(int f) { if (f == 2) __threadfence_system(); else if (f == 1) __threadfence(); }
#define STREAM c10::cuda::getCurrentCUDAStream()
#define P(x) ((long long*)x.data_ptr<int64_t>())

__global__ void fs_k(volatile long long* peer, int k, int f, long long* out) {
  unsigned long long t0 = gtime();
  for (int i = 0; i < k; ++i) { peer[0] = i + 1; fence(f); }
  unsigned long long t1 = gtime(); out[0] = (long long)(t1 - t0); out[1] = (long long)t1;
}
__global__ void pp_k(volatile long long* mine, volatile long long* peer, int k, int side, int f, long long* out) {
  unsigned long long t0 = gtime();
  for (int i = 0; i < k; ++i) {
    if (side == 0) { peer[0] = i + 1; fence(f); while (mine[0] != i + 1) {} }
    else { while (mine[0] != i + 1) {} peer[0] = i + 1; fence(f); }
  }
  unsigned long long t1 = gtime(); out[0] = (long long)(t1 - t0); out[1] = (long long)t1;
}
__global__ void readloop_k(volatile long long* p, int k, long long* out) {
  unsigned long long t0 = gtime(); long long s = 0;
  for (int i = 0; i < k; ++i) s += p[0];
  unsigned long long t1 = gtime(); out[0] = (long long)(t1 - t0); out[1] = (long long)t1; out[2] = s;
}
__device__ __forceinline__ void st_v4(void* p, unsigned a, unsigned b, unsigned c, unsigned d) {
  asm volatile("st.volatile.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void llwait(uint4* mine, unsigned f, int L) {
  for (int l = 0; l < L; ++l) {
    void* p = mine + l * blockDim.x + threadIdx.x; unsigned a, b, c, d;
    do { asm volatile("ld.volatile.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory"); }
    while (b != f || d != f);
  }
  __syncthreads();
}
__global__ void llpp_k(uint4* mine, uint4* peer, int k, int side, int L, long long* out) {
  unsigned long long t0 = gtime();
  for (int i = 0; i < k; ++i) {
    unsigned f = (unsigned)(i + 1);
    if (side == 1) llwait(mine, f, L);
    for (int l = 0; l < L; ++l) st_v4(peer + l * blockDim.x + threadIdx.x, (unsigned)i, f, (unsigned)l, f);
    if (side == 0) llwait(mine, f, L);
  }
  if (threadIdx.x == 0) { unsigned long long t1 = gtime(); out[0] = (long long)(t1 - t0); out[1] = (long long)t1; }
}
__global__ void tiny_k(long long* buf, int i, volatile long long* tgt, int f) {
  buf[i] = (long long)gtime();
  tgt[0] = i + 1; fence(f);
}
__global__ void stamp_k(long long* p) { p[0] = (long long)gtime(); }

void fs(int64_t peer, int64_t k, int64_t f, torch::Tensor out) { fs_k<<<1, 1, 0, STREAM>>>((volatile long long*)peer, (int)k, (int)f, P(out)); }
void pp(int64_t mine, int64_t peer, int64_t k, int64_t side, int64_t f, torch::Tensor out) {
  pp_k<<<1, 1, 0, STREAM>>>((volatile long long*)mine, (volatile long long*)peer, (int)k, (int)side, (int)f, P(out));
}
void readloop(int64_t p, int64_t k, torch::Tensor out) { readloop_k<<<1, 1, 0, STREAM>>>((volatile long long*)p, (int)k, P(out)); }
void llpp(int64_t mine, int64_t peer, int64_t k, int64_t side, int64_t L, int64_t threads, torch::Tensor out) {
  llpp_k<<<1, (int)threads, 0, STREAM>>>((uint4*)mine, (uint4*)peer, (int)k, (int)side, (int)L, P(out));
}
void tiny(torch::Tensor buf, int64_t i, int64_t tgt, int64_t f) { tiny_k<<<1, 1, 0, STREAM>>>(P(buf), (int)i, (volatile long long*)tgt, (int)f); }
void stamp(torch::Tensor p) { stamp_k<<<1, 1, 0, STREAM>>>(P(p)); }
int64_t ev_create() { cudaEvent_t e; TORCH_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess, "ev_create"); return (int64_t)e; }
void ev_record(int64_t e, int64_t external) {
  cudaError_t r = external ? cudaEventRecordWithFlags((cudaEvent_t)e, STREAM, cudaEventRecordExternal) : cudaEventRecord((cudaEvent_t)e, STREAM);
  TORCH_CHECK(r == cudaSuccess, cudaGetErrorString(r));
}
void ev_wait(int64_t e) { cudaError_t r = cudaStreamWaitEvent(STREAM, (cudaEvent_t)e, cudaEventWaitExternal); TORCH_CHECK(r == cudaSuccess, cudaGetErrorString(r)); }
void enable_peer(int64_t a, int64_t b) {
  cudaSetDevice((int)a);
  cudaError_t e = cudaDeviceEnablePeerAccess((int)b, 0);
  if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
  else TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
"""
CPP_SRC = """
void fs(int64_t peer, int64_t k, int64_t f, torch::Tensor out);
void pp(int64_t mine, int64_t peer, int64_t k, int64_t side, int64_t f, torch::Tensor out);
void readloop(int64_t p, int64_t k, torch::Tensor out);
void llpp(int64_t mine, int64_t peer, int64_t k, int64_t side, int64_t L, int64_t threads, torch::Tensor out);
void tiny(torch::Tensor buf, int64_t i, int64_t tgt, int64_t f);
void stamp(torch::Tensor p);
void enable_peer(int64_t a, int64_t b);
int64_t ev_create();
void ev_record(int64_t e, int64_t external);
void ev_wait(int64_t e);
"""
FUNCS = ["fs", "pp", "readloop", "llpp", "tiny", "stamp", "enable_peer", "ev_create", "ev_record", "ev_wait"]
VICTIMS = ["fs_sys", "fs_gpu", "pp_sys", "hostread", "llpp", "tiny_local", "tiny_peer", "tiny_peer_fsys", "tiny_ev",
           "tiny_evrec"]
# mode -> (copying GPU, kind, piece bytes, total bytes)
MODES = {"none": None,
         "d2h_g0": (0, "d2h", 4 * MiB, 512 * MiB), "h2d_g0": (0, "h2d", 4 * MiB, 512 * MiB),
         "d2h_g0_64k": (0, "d2h", 64 * 1024, 256 * MiB),
         "d2h_g1": (1, "d2h", 4 * MiB, 512 * MiB), "d2h_g2": (2, "d2h", 4 * MiB, 512 * MiB),
         "nvl_g0": (0, "nvl", 64 * MiB, 8192 * MiB)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--k", type=int, default=1000, help="iterations of the fence / ping-pong / read loops")
    ap.add_argument("--ll_k", type=int, default=500)
    ap.add_argument("--ll_lines", type=int, default=8)
    ap.add_argument("--ll_threads", type=int, default=128)
    ap.add_argument("--tiny_n", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ext = load_inline("x1d_scope", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=FUNCS,
                      extra_cuda_cflags=["-O3"], verbose=False,
                      extra_ldflags=[f"-L{os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib')}"])  # pixi's libcudart.so
    G = (0, 1, 2)
    dev = {g: torch.device(f"cuda:{g}") for g in G}
    for a in G:
        for b in G:
            if a != b:
                ext.enable_peer(a, b)
    torch.cuda.set_device(0)

    def sync():
        for g in G:
            torch.cuda.synchronize(dev[g])

    word = {g: torch.zeros(8, dtype=torch.int64, device=dev[g]) for g in G}
    out = {g: torch.zeros(4, dtype=torch.int64, device=dev[g]) for g in G}
    ll_elems = args.ll_threads * args.ll_lines * 4  # uint32 words
    llbuf = {g: torch.zeros(ll_elems, dtype=torch.int32, device=dev[g]) for g in (0, 1)}
    bgend = {g: torch.zeros(1, dtype=torch.int64, device=dev[g]) for g in G}
    nb = 512 * MiB
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, node_cpus(0))
    torch.set_num_threads(1)
    host = {g: torch.zeros(nb, dtype=torch.uint8) for g in G}
    hword = torch.zeros(8, dtype=torch.int64)
    os.sched_setaffinity(0, old)
    for g in G:
        cm.register(host[g].data_ptr(), nb)
    cm.register(hword.data_ptr(), 64)
    devbuf = {g: torch.empty(nb, dtype=torch.uint8, device=dev[g]) for g in G}
    nvdst = torch.empty(nb, dtype=torch.uint8, device=dev[2])
    bgs = {g: torch.cuda.Stream(dev[g]) for g in G}
    work = {g: torch.cuda.Stream(dev[g]) for g in G}

    # tiny-kernel graphs on GPU0
    tbuf = torch.zeros(args.tiny_n, dtype=torch.int64, device=dev[0])
    graphs = {}
    with torch.cuda.device(0):
        ev = ext.ev_create()
        ext.ev_record(ev, 0)  # recorded once, so the first captured wait is satisfied
        torch.cuda.synchronize(dev[0])
    for name, tgt, f, evw, evr in (("tiny_local", word[0], 0, False, False), ("tiny_peer", word[1], 0, False, False),
                                   ("tiny_peer_fsys", word[1], 2, False, False), ("tiny_ev", word[0], 0, True, True),
                                   ("tiny_evrec", word[0], 0, False, True)):
        with torch.cuda.device(0):
            s = torch.cuda.Stream(dev[0])
            with torch.cuda.stream(s):
                for i in range(8):
                    ext.tiny(tbuf, i, tgt.data_ptr(), f)
            torch.cuda.synchronize(dev[0])
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(args.tiny_n):
                    if evw:
                        ext.ev_wait(ev)
                    ext.tiny(tbuf, i, tgt.data_ptr(), f)
                    if evr:
                        ext.ev_record(ev, 1)
            torch.cuda.synchronize(dev[0])
            graphs[name] = g

    def background(mode):
        """Enqueue the copies, then a stamp kernel behind them; returns (copying GPU, start event, end event, bytes)."""
        spec = MODES[mode]
        if spec is None:
            return None
        g, kind, piece, total = spec
        s = bgs[g]
        with torch.cuda.device(g):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(s)
            done = 0
            while done < total:
                off = done % nb
                if kind == "d2h":
                    cm.memcpy(s, host[g].data_ptr() + off, devbuf[g].data_ptr() + off, piece)
                elif kind == "h2d":
                    cm.memcpy(s, devbuf[g].data_ptr() + off, host[g].data_ptr() + off, piece)
                else:
                    cm.memcpy(s, nvdst.data_ptr() + off, devbuf[g].data_ptr() + off, piece)
                done += piece
            e1.record(s)
            with torch.cuda.stream(s):
                ext.stamp(bgend[g])
        return g, e0, e1, total

    def victim(name):
        """Launch the victim; returns a function that reads (ns per item, absolute end time ns) after sync."""
        k = args.k
        if name in ("fs_sys", "fs_gpu"):
            with torch.cuda.device(0), torch.cuda.stream(work[0]):
                ext.fs(word[1].data_ptr(), k, 2 if name == "fs_sys" else 1, out[0])
            return lambda: (out[0][0].item() / k, out[0][1].item())
        if name == "pp_sys":
            with torch.cuda.device(1), torch.cuda.stream(work[1]):
                ext.pp(word[1].data_ptr(), word[0].data_ptr(), k, 1, 2, out[1])
            with torch.cuda.device(0), torch.cuda.stream(work[0]):
                ext.pp(word[0].data_ptr(), word[1].data_ptr(), k, 0, 2, out[0])
            return lambda: (out[0][0].item() / k, out[0][1].item())
        if name == "hostread":
            with torch.cuda.device(0), torch.cuda.stream(work[0]):
                ext.readloop(hword.data_ptr(), k, out[0])
            return lambda: (out[0][0].item() / k, out[0][1].item())
        if name == "llpp":
            kk = args.ll_k
            with torch.cuda.device(1), torch.cuda.stream(work[1]):
                ext.llpp(llbuf[1].data_ptr(), llbuf[0].data_ptr(), kk, 1, args.ll_lines, args.ll_threads, out[1])
            with torch.cuda.device(0), torch.cuda.stream(work[0]):
                ext.llpp(llbuf[0].data_ptr(), llbuf[1].data_ptr(), kk, 0, args.ll_lines, args.ll_threads, out[0])
            return lambda: (out[0][0].item() / kk, out[0][1].item())
        with torch.cuda.device(0), torch.cuda.stream(work[0]):
            graphs[name].replay()
        n = args.tiny_n

        def rd():
            st = tbuf.tolist()
            return (st[n - 1] - st[0]) / (n - 1), st[n - 1]
        return rd

    def one(name, mode):
        sync()
        for g in G:
            word[g].zero_()
        for g in (0, 1):
            llbuf[g].zero_()
        tbuf.zero_()
        sync()
        t0 = time.perf_counter()
        bg = background(mode)
        rd = victim(name)
        sync()
        t1 = time.perf_counter()
        ns, vend = rd()
        r = dict(ns=ns, t0=t0, t1=t1)
        if bg:
            g, e0, e1, total = bg
            ms = e0.elapsed_time(e1)
            r.update(bg_gbs=total / ms / 1e6, outlast=bgend[g].item() >= vend)
        return r

    modes = list(MODES)
    cases = [(v, m) for v in VICTIMS for m in modes]
    for _ in range(2):
        for c in cases:
            one(*c)
    raw = {f"{v}|{m}": [] for v, m in cases}
    rng = random.Random(1)
    tstart = time.time()
    with ClockLogger(0) as clk:
        for r in range(args.rounds):
            order = cases[:]
            rng.shuffle(order)
            for c in order:
                raw[f"{c[0]}|{c[1]}"].append(one(*c))
            if (r + 1) % 10 == 0:
                print(f"[progress] round {r + 1}/{args.rounds}, {time.time() - tstart:.0f} s", flush=True)
        for k in raw:
            for x in raw[k]:
                x["clock"] = clk.min_clock_between(x["t0"], x["t1"])
    clocks = [x["clock"] for k in raw for x in raw[k] if x["clock"]]
    modal = statistics.mode(clocks) if clocks else None
    summ = {}
    for k, xs in raw.items():
        kept = [x for x in xs if modal is None or x["clock"] is None or x["clock"] >= 0.95 * modal]
        s = dict(ns=statistics.median(x["ns"] for x in kept), n=len(kept), dropped=len(xs) - len(kept))
        if "bg_gbs" in xs[0]:
            s.update(bg_gbs=statistics.median(x["bg_gbs"] for x in kept), outlast_frac=sum(x["outlast"] for x in kept) / len(kept))
        summ[k] = s
    for v in VICTIMS:
        base = summ[f"{v}|none"]["ns"]
        line = f"[x1d] {v:<14} none {base:9.1f} ns"
        for m in modes[1:]:
            s = summ[f"{v}|{m}"]
            s["x_none"] = s["ns"] / base
            line += f" | {m} {s['ns']:9.1f} (x{s['x_none']:5.2f}, out {s['outlast_frac']:.2f})"
        summ[f"{v}|none"]["x_none"] = 1.0
        print(line, flush=True)
    for m in modes[1:]:
        print(f"[x1d-bg] {m:<11} median {statistics.median(summ[f'{v}|{m}']['bg_gbs'] for v in VICTIMS):7.1f} GB/s", flush=True)
    print(f"[x1d-clock] modal {modal} MHz, dropped {sum(s['dropped'] for s in summ.values())} of "
          f"{sum(len(x) for x in raw.values())} cases", flush=True)
    with open(os.path.join(args.out, "x1d_scope_v1.json"), "w") as f:
        json.dump(dict(args=vars(args), modal_clock=modal, summary=summ, raw=raw), f)


if __name__ == "__main__":
    main()
