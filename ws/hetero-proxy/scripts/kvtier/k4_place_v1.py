################################################################################
# hetero-proxy KV tier K4.1b (2026-10-17): does WHERE a background PCIe copy runs inside a decode step matter?
# Graph-mode test of the Flux-style idea (K4.3) on the real decode step. (The eager diagnostics k4_diag_v2 / v3 are
# withdrawn: their no-copy baseline moved 11.4 -> 12.9 -> 17.0 ms between runs and "ticks only" came out 24% faster
# than no ticks - eager decode-sized NCCL loops are launch-bound and order-sensitive. k4_interference_v2, graphs +
# shuffled rounds, reproduced its +48% twice.)
# The decode graph (k2_engine_v2, B rows) is captured in three versions:
#   plain   as in K2
#   gu      a 1-element tick += 1 kernel right before every layer's gate|up GEMM (the largest GEMM, ~50 us at M <= 32)
#   ar      the same tick right before every layer's first all-reduce (o_proj output)
# Background (all 8 GPUs, copy stream):
#   none | free (384 MiB in 4 MiB copies, as fast as PCIe goes) | released: 64 copies of --chunk_kb per step, copy j
#   waits (cuStreamWaitValue64) for tick >= base + j + 1, i.e. starts when layer j reaches the release point
#   ticks = the gu graph with no copies (the cost of the tick kernels)
# Directions h2d / d2h. Cases interleaved and shuffled per round (same order on every rank); decode-step CUDA-event
# time per rank -> rank-max -> median; copy volume and duration.
# PRE-REGISTERED [推論]: R1 ticks cost <= 3%; R2 gu-released copies slow the step <= 1/3 as much as ar-released ones
#   in both directions; R3 free h2d reproduces k4_interference_v2 (+40..+55%).
# Usage: torchrun 8 ranks (exclusive guard, tmux): k4_place_v1.py --out <dir> [--rounds 100 --B 16 --chunk_kb 512]
################################################################################
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
sys.path.insert(0, os.path.join(HERE, ".."))
from qwen_tp_v1 import init  # noqa: E402
from qwen_tp_v5 import pin  # noqa: E402
from fused_kernels import add_rmsnorm, silu_mul  # noqa: E402
from k2_engine_v2 import K2Engine  # noqa: E402
from kvtier_v2 import node_cpus  # noqa: E402
from hxb import cudamem as cm  # noqa: E402

MiB = 2 ** 20


class K4Engine(K2Engine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tick = torch.zeros(1, dtype=torch.int64, device=self.dev)
        self._where = None

    def _mlp_out(self, i, x, a, T):
        c, w = self.cfg, self.layers[i]
        o = torch.mm(a, w["wo"].t(), out=self.obuf[:T])
        if self._where == "ar":
            self.tick.add_(1)
        self._ar(o)
        x, h2 = add_rmsnorm(x, w["ln2"], c.eps, r=o)
        if self._where == "gu":
            self.tick.add_(1)
        d = torch.mm(silu_mul(F.linear(h2, w["wgu"])), w["wd"].t(), out=self.dbuf[:T])
        self._ar(d)
        nw = self.layers[i + 1]["ln1"] if i + 1 < self.L else self.norm
        return add_rmsnorm(x, nw, c.eps, r=d)

    def capture_dec(self, B, where):
        self._where = where
        self._capture(("dec", B, where), lambda: self._decode(B), False)
        self._where = None


def main():
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("HXB_STACK_DUMP_S", "0")) or 10 ** 7, repeat=True, exit=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--chunk_kb", type=int, default=512)
    ap.add_argument("--bg_mb", type=int, default=384)
    args = ap.parse_args()
    rank, world, dev = init()
    pin(rank)
    os.makedirs(args.out, exist_ok=True)
    m = K4Engine(rank, world, dev, 40, 4608, 32)
    B = args.B
    m.d_in[0].fill_(m.DUMMY)
    m.d_in[1].zero_()
    for where in (None, "gu", "ar"):
        m.capture_dec(B, where)
    m.d_in_h[0, :B] = torch.arange(B)
    m.d_in_h[1, :B] = 2000 + torch.arange(B)
    m.d_in[:, :B].copy_(m.d_in_h[:, :B])
    torch.cuda.synchronize()
    tickv = [int(m.tick.item())]
    nb, ch, ck = args.bg_mb * MiB, 4 * MiB, args.chunk_kb * 1024
    node = 0 if dev.index < 4 else 1
    old, nt = os.sched_getaffinity(0), torch.get_num_threads()
    os.sched_setaffinity(0, node_cpus(node))
    torch.set_num_threads(1)
    host = torch.zeros(nb, dtype=torch.uint8)
    os.sched_setaffinity(0, old)
    torch.set_num_threads(nt)
    cm.register(host.data_ptr(), nb)
    devbuf = torch.empty(nb, dtype=torch.uint8, device=dev)
    bgs = torch.cuda.Stream(dev)

    def copy(off, n, d):
        if d == "h2d":
            cm.memcpy(bgs, devbuf.data_ptr() + off, host.data_ptr() + off, n)
        else:
            cm.memcpy(bgs, host.data_ptr() + off, devbuf.data_ptr() + off, n)

    @torch.inference_mode()
    def one(mode, d):
        graph = {"none": None, "free": None, "ticks": "gu", "gu": "gu", "ar": "ar"}[mode]
        dist.barrier()
        torch.cuda.synchronize()
        cs = torch.cuda.current_stream()
        b0, b1, e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        vol = 0
        if mode not in ("none", "ticks"):
            bgs.wait_stream(cs)
            b0.record(bgs)
            if mode == "free":
                for off in range(0, nb, ch):
                    copy(off, ch, d)
                vol = nb
            else:
                for j in range(m.L):
                    cm.wait_geq(bgs, m.tick.data_ptr(), tickv[0] + j + 1)
                    copy((j * ck) % nb, ck, d)
                vol = m.L * ck
            b1.record(bgs)
        e0.record(cs)
        m.gs[("dec", B, graph)].replay()
        e1.record(cs)
        torch.cuda.synchronize()
        if graph is not None:
            tickv[0] += m.L
        if vol:
            return e0.elapsed_time(e1), b0.elapsed_time(b1), vol
        return e0.elapsed_time(e1), None, 0

    cases = [("none", "-"), ("ticks", "-")] + [(mo, d) for mo in ("free", "gu", "ar") for d in ("h2d", "d2h")]
    for _ in range(3):
        for c in cases:
            one(*c)
    rng = random.Random(1)
    res = {f"{mo}|{d}": [] for mo, d in cases}
    t0 = time.time()
    for r in range(args.rounds):
        order = cases[:]
        rng.shuffle(order)
        for c in order:
            res[f"{c[0]}|{c[1]}"].append(one(*c))
        if rank == 0 and (r + 1) % 25 == 0:
            print(f"[progress] round {r + 1}/{args.rounds}, {time.time() - t0:.0f} s", flush=True)
    assert int(m.tick.item()) == tickv[0], (int(m.tick.item()), tickv[0])
    allr = [None] * world
    dist.all_gather_object(allr, res)
    if rank == 0:
        n = args.rounds
        med = lambda k: statistics.median([max(allr[q][k][j][0] for q in range(world)) for j in range(n)])  # noqa: E731
        base = med("none|-")
        out = {}
        for k in res:
            ms = med(k)
            o = dict(ms=ms, vs_none=ms / base - 1)
            line = f"[place] B={B} {k:<9} step {ms:8.3f} ms ({100 * (ms / base - 1):+6.1f}%)"
            if allr[0][k][0][2]:
                bg = statistics.median([allr[q][k][j][1] for q in range(world) for j in range(n)])
                vol = allr[0][k][0][2]
                o.update(bg_ms=bg, vol_MB=vol / 1e6)
                line += f"   copies {vol / 1e6:6.1f} MB in {bg:6.2f} ms ({vol / bg / 1e6:5.2f} GB/s per GPU)"
            out[k] = o
            print(line, flush=True)
        with open(os.path.join(args.out, f"k4_place_v1_B{B}.json"), "w") as f:
            json.dump(dict(args=vars(args), summary=out, raw=allr), f)
    dist.barrier()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
