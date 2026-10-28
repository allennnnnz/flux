################################################################################
# hetero-proxy X1c (2026-10-28): is it the system-scope fence? X1b (results/xpcie/x1b_v1): an NVLink ping-pong whose
# side writes the peer's word then __threadfence_system() went 2.6 -> 5.2 us (H2D on both GPUs) / 11.2 us (D2H);
# a host-memory read 1.3 -> 3.2 / 4.7 us; a local spin did not change. Variants of the ping-pong after the store to the
# peer: fsys = __threadfence_system() (X1b), fgpu = __threadfence() (device scope), none = no fence (volatile stores
# reach the peer anyway; only ordering differs). Modes as X1b.
# PRE-REGISTERED [推論]: C1 if the system-scope fence is what waits behind PCIe DMA, ping-pong "none" and "fgpu" change
#   by < 20% under d2h_both while "fsys" reproduces X1b (x3-x5); otherwise NVLink small-message latency itself is hit.
# (X1b header follows) which latency does background PCIe DMA inflate? X1 (results/xpcie/x1_v1) found that
# in CUDA graphs tiny-kernel dispatch and decode-sized GEMMs are unaffected, while NCCL's small all-reduces slow
# +66% (H2D on every GPU) to +570% (D2H), also when only the OTHER GPUs copy. Candidates, measured on the GPU itself
# (%globaltimer) by small CUDA C kernels (torch cpp_extension.load_inline):
#   pingpong  GPU0 and GPU1 bounce a counter through each other's memory over NVLink (SM stores to the peer, volatile
#             spin on the local word), K round trips -> ns per round trip       (is NVLink small-message latency hit?)
#   hostread  one GPU thread reads a volatile word of pinned host memory K times -> ns per read (PCIe read latency)
#   localspin same loop on a local word (control)
# One process drives GPU0 and GPU1 (peer access enabled). Background on a side stream of each GPU: 4 MiB cuMemcpyAsync
# pieces from / to NUMA-local pinned host memory. Modes: none | h2d_both | d2h_both | d2h_gpu0 | d2h_gpu1.
# Cases interleaved and shuffled per round; medians over --rounds.
# PRE-REGISTERED [推論]: B1 pingpong is not slowed by > 20% in any mode (NVLink is not the shared resource);
#   B2 hostread is slowed >= 3x by d2h on the reading GPU and less by h2d (a GPU read request waits behind posted
#   writes in the same direction); B3 localspin is unaffected.
# Usage: CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml python x1c_fence_v1.py --out <dir>
################################################################################
import argparse
import json
import os
import random
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "kvtier"))
from hxb import cudamem as cm  # noqa: E402
from kvtier_v2 import node_cpus  # noqa: E402

MiB = 2 ** 20
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// side 0 starts: writes peer[0] = i+1, waits mine[0] == i+1 ; side 1 echoes
__device__ __forceinline__ void fence(int f) { if (f == 2) __threadfence_system(); else if (f == 1) __threadfence(); }
__global__ void pingpong_k(volatile long long* mine, volatile long long* peer, int k, int side, int f, long long* out) {
  unsigned long long t0 = gtime();
  for (int i = 0; i < k; ++i) {
    if (side == 0) { peer[0] = i + 1; fence(f); while (mine[0] != i + 1) {} }
    else { while (mine[0] != i + 1) {} peer[0] = i + 1; fence(f); }
  }
  out[0] = (long long)(gtime() - t0);
}
__global__ void readloop_k(volatile long long* p, int k, long long* out) {
  unsigned long long t0 = gtime(); long long s = 0;
  for (int i = 0; i < k; ++i) s += p[0];
  out[0] = (long long)(gtime() - t0); out[1] = s;
}
void pingpong(int64_t mine, int64_t peer, int64_t k, int64_t side, int64_t f, torch::Tensor out) {
  pingpong_k<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>((volatile long long*)mine, (volatile long long*)peer, (int)k, (int)side, (int)f, (long long*)out.data_ptr<int64_t>());
}
void enable_peer(int64_t a, int64_t b) {
  cudaSetDevice((int)a);
  cudaError_t e = cudaDeviceEnablePeerAccess((int)b, 0);
  if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
  else TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
void readloop(int64_t p, int64_t k, torch::Tensor out) {
  readloop_k<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>((volatile long long*)p, (int)k, (long long*)out.data_ptr<int64_t>());
}
"""
CPP_SRC = "void pingpong(int64_t mine, int64_t peer, int64_t k, int64_t side, int64_t f, torch::Tensor out);\n" \
          "void readloop(int64_t p, int64_t k, torch::Tensor out);\nvoid enable_peer(int64_t a, int64_t b);"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--bg_mb", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ext = load_inline("x1c_lat", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["pingpong", "readloop", "enable_peer"],
                      extra_cuda_cflags=["-O3"], verbose=False,
                      extra_ldflags=[f"-L{os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib')}"])  # pixi's libcudart.so
    d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
    for a, b in ((0, 1), (1, 0)):
        ext.enable_peer(a, b)
    torch.cuda.set_device(0)
    flag = {0: torch.zeros(8, dtype=torch.int64, device=d0), 1: torch.zeros(8, dtype=torch.int64, device=d1)}
    out = {0: torch.zeros(2, dtype=torch.int64, device=d0), 1: torch.zeros(2, dtype=torch.int64, device=d1)}
    nb, ch = args.bg_mb * MiB, 4 * MiB
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, node_cpus(0))
    torch.set_num_threads(1)
    host = {g: torch.zeros(nb, dtype=torch.uint8) for g in (0, 1)}
    hword = torch.zeros(8, dtype=torch.int64)
    os.sched_setaffinity(0, old)
    for g in (0, 1):
        cm.register(host[g].data_ptr(), nb)
    cm.register(hword.data_ptr(), 64)
    devbuf = {0: torch.empty(nb, dtype=torch.uint8, device=d0), 1: torch.empty(nb, dtype=torch.uint8, device=d1)}
    bgs = {0: torch.cuda.Stream(d0), 1: torch.cuda.Stream(d1)}
    work = {0: torch.cuda.Stream(d0), 1: torch.cuda.Stream(d1)}
    K = {"pp_fsys": 2000, "pp_fgpu": 2000, "pp_none": 2000}
    FENCE = {"pp_fsys": 2, "pp_fgpu": 1, "pp_none": 0}

    def background(mode):
        who = {"none": (), "h2d_both": ((0, "h2d"), (1, "h2d")), "d2h_both": ((0, "d2h"), (1, "d2h")),
               "d2h_gpu0": ((0, "d2h"),), "d2h_gpu1": ((1, "d2h"),)}[mode]
        for g, dirn in who:
            s = bgs[g]
            for off in range(0, nb, ch):
                if dirn == "h2d":
                    cm.memcpy(s, devbuf[g].data_ptr() + off, host[g].data_ptr() + off, ch)
                else:
                    cm.memcpy(s, host[g].data_ptr() + off, devbuf[g].data_ptr() + off, ch)

    def one(test, mode):
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        flag[0].zero_()
        flag[1].zero_()
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        background(mode)
        k = K[test]
        with torch.cuda.device(1), torch.cuda.stream(work[1]):
            ext.pingpong(flag[1].data_ptr(), flag[0].data_ptr(), k, 1, FENCE[test], out[1])
        with torch.cuda.device(0), torch.cuda.stream(work[0]):
            ext.pingpong(flag[0].data_ptr(), flag[1].data_ptr(), k, 0, FENCE[test], out[0])
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        return out[0][0].item() / k

    tests = ["pp_fsys", "pp_fgpu", "pp_none"]
    modes = ["none", "h2d_both", "d2h_both", "d2h_gpu0", "d2h_gpu1"]
    cases = [(t, m) for t in tests for m in modes]
    for _ in range(3):
        for c in cases:
            one(*c)
    raw = {f"{t}|{m}": [] for t, m in cases}
    rng = random.Random(1)
    for _ in range(args.rounds):
        order = cases[:]
        rng.shuffle(order)
        for c in order:
            raw[f"{c[0]}|{c[1]}"].append(one(*c))
    summ = {}
    for t in tests:
        base = statistics.median(raw[f"{t}|none"])
        line = f"[x1c] {t:<9} none {base:8.1f} ns"
        for m in modes:
            v = statistics.median(raw[f"{t}|{m}"])
            summ[f"{t}|{m}"] = dict(ns=v, x_none=v / base)
            if m != "none":
                line += f" | {m} {v:8.1f} ns (x{v / base:5.2f})"
        print(line, flush=True)
    with open(os.path.join(args.out, "x1c_fence_v1.json"), "w") as f:
        json.dump(dict(args=vars(args), summary=summ, raw=raw), f)


if __name__ == "__main__":
    main()
