# WITHDRAWN 2026-10-16: bench capture runs out of GPU memory (results/kvtier/bench_v3_oom; all-reduce outputs are not
# reused inside a capture). Replaced by ws/hetero-proxy/scripts/kvtier/kvtier_v4.py. (Its check, results/kvtier/check_v3, passed.)
################################################################################
# hetero-proxy KV tier v3 (2026-10-16): kvtier_v2.py with every timed forward pass captured in a CUDA graph.
# Why: the v2 smoke bench (results/kvtier/bench_smoke, 3 rounds) showed the eager new-turn pass ("ideal", 128 tokens)
# at 50-57 ms for every history length - CPU-launch bound (64 layers x ~20 Python launches), while its GPU work is
# ~5-10 ms (7.8 GB of weights per GPU). The same ~45 ms of launch overhead sits in the eager recompute, so v2's
# speedups and its Cake split (one extra eager pass, ~50 ms) measured the engine's Python, not the memory tier.
# v3: graphs per shape (bench shapes are fixed): new turn ("ext", T tokens at position P, gated), recompute (P + T
# tokens from position 0), Cake's recompute part ([0, X), no head). Restore gating inside the graph uses the hxb
# mechanism: a per-layer int64 flag in GPU memory; the copy stream writes flag[i] = 1 after layer i's K and V
# (cuStreamWriteValue64), the graph waits flag[i] >= 1 right before layer i's attention (cuStreamWaitValue64) and
# resets every flag to 0 at its end. Both are captured as graph nodes (probed 2026-10-16: the graph blocks while the
# flag is 0, runs when another stream writes 1, and the reset node works).
# The copies stay eager on the copy stream, issued right after the graph launch (the copy stream's only dependency
# is an event recorded before the launch, so the parked graph cannot deadlock it).
# Cases as v2 plus ideal_eager (v2's eager new-turn pass, for reference). Check mode as v2 plus graph-vs-eager
# bit identity.
# PRE-REGISTERED for the v3 bench (written after the v2 smoke run, before any v3 run) [推論]:
#   G1 ideal (graph) 8-20 ms at every P (eager 50-57 ms in the smoke run)
#   G2 recompute (graph) within 10% of the smoke run's eager recompute at P >= 8192 (GPU-bound: 445 / 906 / 1940 ms)
#      and 15-45 ms faster at P = 1024 (eager 78 ms)
#   G3 layer = max(load, ideal) + at most 3 ms (P = 8192: 26-31 ms, P = 32000: 98-105 ms)
#   G4 layer vs recompute: >= 2x at P = 1024, >= 8x at 4096, >= 14x at 8192, >= 18x at 32000
#   G5 serial - layer = min(load, ideal) within 3 ms
#   G6 best split vs layer: at most 10% faster, and only where load > ideal (P >= 8192); slower at P <= 4096
#      (the X recompute adds GPU time while the load is already hidden; PCIe traffic also slows the recompute's
#      NCCL all-reduce, results/e2e/c_diag)
#   G7 load 9-12 GB/s per GPU, save 9.5-12 GB/s per GPU (smoke: 9.2-10.7 / 9.9-11.0)
#   G8 check: graph paths bit-identical to the eager path; restore paths bit-identical to resident
# Usage: ws/hetero-proxy/scripts/kvtier/run_bench_v1.sh <out> --mode check|bench ... with KVTIER=kvtier_v3.py
################################################################################
import argparse
import json
import math
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
sys.path.insert(0, os.path.join(HERE, ".."))
from qwen_tp_v1 import init, rotate_half  # noqa: E402
from qwen_tp_v5 import pin  # noqa: E402
from fused_kernels import add_rmsnorm, silu_mul  # noqa: E402
from hxb import cudamem as cm  # noqa: E402
from kvtier_v2 import KVEngine, KVTier, split_point  # noqa: E402


class GKVEngine(KVEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.gate = torch.zeros(self.L, dtype=torch.int64, device=self.dev)
        self.gs = {}

    def gate_addr(self, i):
        return self.gate.data_ptr() + 8 * i

    def _layer_g(self, i, x, h, b, pos0, T, gated):
        c, w = self.cfg, self.layers[i]
        hd, nq, nkv = c.HD, self.nq, self.nkv
        qkv = F.linear(h, w["wqkv"], w["bqkv"])
        q, k, v = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)
        q = q.view(1, T, nq, hd).transpose(1, 2)
        k = k.view(1, T, nkv, hd).transpose(1, 2)
        v = v.view(1, T, nkv, hd).transpose(1, 2)
        cos, sin = self.cos[pos0:pos0 + T], self.sin[pos0:pos0 + T]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        S = pos0 + T
        self.kc[i, b:b + 1, :, pos0:S] = k
        self.vc[i, b:b + 1, :, pos0:S] = v
        if gated:
            cm.wait_geq(torch.cuda.current_stream(), self.gate_addr(i), 1)
        kk, vv = self.kc[i, b:b + 1, :, :S], self.vc[i, b:b + 1, :, :S]
        if pos0 == 0:
            a = F.scaled_dot_product_attention(q, kk, vv, is_causal=T > 1, enable_gqa=True, scale=1 / math.sqrt(hd))
        else:
            a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=causal_lower_right(T, S), enable_gqa=True,
                                               scale=1 / math.sqrt(hd))
        o = F.linear(a.transpose(1, 2).reshape(T, nq * hd), w["wo"])
        dist.all_reduce(o)
        x, h2 = add_rmsnorm(x, w["ln2"], c.eps, r=o)
        d = F.linear(silu_mul(F.linear(h2, w["wgu"])), w["wd"])
        dist.all_reduce(d)
        nw = self.layers[i + 1]["ln1"] if i + 1 < self.L else self.norm
        return add_rmsnorm(x, nw, c.eps, r=d)

    def _fwd(self, ids, pos0, b, gated, head):
        T = ids.numel()
        x = F.embedding(ids, self.embed)
        _, h = add_rmsnorm(x, self.layers[0]["ln1"], self.cfg.eps)
        for i in range(self.L):
            x, h = self._layer_g(i, x, h, b, pos0, T, gated)
        out = F.linear(h[-1:], self.lm_head).float()[0] if head else None
        if gated:
            cs = torch.cuda.current_stream()
            for i in range(self.L):
                cm.write64(cs, self.gate_addr(i), 0)
        return out

    @torch.inference_mode()
    def graph(self, T, pos0, gated, head, b=0):
        """(graph, static ids [T], static logits or None) of one pass; captured once per shape. The capture warm-up
        writes KV [pos0, pos0 + T) of slot b: capture before the cache holds anything that matters."""
        key = (T, pos0, gated, head, b)
        if key not in self.gs:
            ids = torch.zeros(T, dtype=torch.long, device=self.dev)
            cur = torch.cuda.current_stream()
            s = torch.cuda.Stream()
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                for _ in range(3):
                    if gated:
                        self.gate.fill_(1)
                    self._fwd(ids, pos0, b, gated, head)
            cur.wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self._fwd(ids, pos0, b, gated, head)
            torch.cuda.synchronize()
            self.gs[key] = (g, ids, out)
        return self.gs[key]

    @torch.inference_mode()
    def replay(self, T, pos0, gated, head, ids):
        g, buf, out = self.graph(T, pos0, gated, head)
        buf.copy_(ids)
        g.replay()
        return out


class GRestoreJob:
    """H2D of tokens [a, e) of host slot h into GPU slot b; after layer i's K and V the copy stream sets flag i = 1.
    Construct BEFORE launching the graph that waits on the flags (the copy stream must not depend on it)."""

    def __init__(self, tier, m, h, b, a, e):
        self.t, self.m, self.h, self.b, self.a, self.e = tier, m, h, b, a, e
        self.n = (e - a) * tier.HD * 2
        tier.stream.wait_stream(torch.cuda.current_stream())  # the request starts now

    def issue_all(self):
        t, s = self.t, self.t.stream
        for i in range(t.L):
            if self.n:
                for kv in (0, 1):
                    cm.memcpy(s, t._g(i, kv, self.b, self.a), t._h(self.h, i, kv, self.a), self.n)
            cm.write64(s, self.m.gate_addr(i), 1)
        t.done.record(s)
        return t.done


def run_path(m, tier, kind, hist, new, h, x=None):
    """One request on GPU slot 0: history hist [P] (KV in host slot h, or on the GPU for ideal), new turn new [T]."""
    P, T = hist.numel(), new.numel()
    cs = torch.cuda.current_stream()
    if kind == "ideal":
        m.gate.fill_(1)
        return m.replay(T, P, True, True, new)
    if kind == "ideal_eager":
        return m.extend(new, P)
    if kind == "recompute":
        return m.replay(P + T, 0, False, True, torch.cat([hist, new]))
    if kind == "serial":
        job = GRestoreJob(tier, m, h, 0, 0, P)
        cs.wait_event(job.issue_all())
        return m.replay(T, P, True, True, new)
    if kind == "layer":
        job = GRestoreJob(tier, m, h, 0, 0, P)
        out = m.replay(T, P, True, True, new)
        job.issue_all()
        return out
    if kind == "split":
        X = split_point(P, x)
        job = GRestoreJob(tier, m, h, 0, X, P)
        m.replay(X, 0, False, False, hist[:X])
        out = m.replay(T, P, True, True, new)
        job.issue_all()
        return out
    raise ValueError(kind)


def capture_for(m, P, T, xs):
    m.graph(T, P, True, True)
    m.graph(P + T, 0, False, True)
    for x in xs:
        m.graph(split_point(P, x), 0, False, False)


def mode_check(m, tier, rank, args):
    ref = json.load(open(args.reference))
    rows = []
    for it in ref["items"]:
        ids = torch.tensor(it["prompt_ids"], device=m.dev)
        P = it["prefix_len"]
        hist, new = ids[:P], ids[P:]
        T, n_new, hf = new.numel(), len(it["hf_tokens"]), it["hf_tokens"]
        Lb = 128 * math.ceil((ids.numel() + n_new) / 128)
        capture_for(m, P, T, [0.25])
        m.build_decode("A", 1, Lb)
        res, logit0 = {}, {}

        def noise():
            m.kc[:, 0].normal_()
            m.vc[:, 0].normal_()

        def finish(name, logits):
            t0 = logits.argmax(-1)
            toks = [int(t0)] + m.decode_run("A", 1, Lb, t0.view(1), ids.numel(), n_new - 1)[0].tolist()
            logit0[name] = logits.clone()
            res[name] = dict(tokens=toks, vs_hf=next((i for i, (a, b) in enumerate(zip(toks, hf)) if a != b), n_new))

        noise()
        m.extend(hist, 0, head=False)  # the previous turn left the history in the cache (eager prefill)
        torch.cuda.current_stream().wait_event(tier.save(0, 0, 0, P))
        finish("resident_eager", m.extend(new, P))
        noise()
        m.extend(hist, 0, head=False)
        finish("resident", run_path(m, tier, "ideal", hist, new, 0))
        for kind, x in (("serial", None), ("layer", None), ("split", 0.25), ("recompute", None)):
            noise()
            dist.barrier()
            finish(kind if x is None else f"split{x}", run_path(m, tier, kind, hist, new, 0, x))
            assert int(m.gate.sum()) == 0
        base = logit0["resident"]
        for k in res:
            res[k]["logits_bit_identical_to_resident"] = bool(torch.equal(logit0[k], base))
            res[k]["max_abs_logit_diff_vs_resident"] = float((logit0[k] - base).abs().max())
            res[k]["tokens_identical_to_resident"] = res[k]["tokens"] == res["resident"]["tokens"]
        hl = torch.tensor(it["hf_first_logits_top"], device=m.dev)
        for k in res:
            res[k]["hf_top20_max_abs_logit_diff"] = float((logit0[k][hl[:, 0].long()] - hl[:, 1]).abs().max())
            i = res[k]["vs_hf"]
            res[k]["hf_gap_at_divergence"] = it["hf_gaps"][i] if i < n_new else None
        rows.append(dict(name=it["name"], prefix_len=P, new_len=T, n=n_new, results=res))
        if rank == 0:
            print(f"[check] {it['name']} (P {P}, T {T}): " + "  ".join(
                f"{k}: HF {r['vs_hf']}/{n_new} bit={int(r['logits_bit_identical_to_resident'])} "
                f"tok={int(r['tokens_identical_to_resident'])}" for k, r in res.items()), flush=True)
    if rank == 0:
        with open(os.path.join(args.out, "check_kvtier_v3.json"), "w") as f:
            json.dump(dict(numa_pages_rank0=tier.numa, rows=rows), f, indent=1)


def mode_bench(m, tier, rank, args):
    Ps = [int(p) for p in args.P.split(",")]
    T, Pmax = args.T, max(Ps)
    xs = [float(x) for x in args.split.split(",") if x]
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, m.cfg.V, (Pmax + T,), generator=g).to(m.dev)  # same on every rank
    hist_all, new = seq[:Pmax], seq[Pmax:]
    m.extend(hist_all, 0, head=False)
    for h in range(args.slots):
        tier.save(h, 0, 0, Pmax)
    torch.cuda.synchronize()
    t0 = time.time()
    for P in Ps:
        capture_for(m, P, T, xs)
    m.gate.zero_()
    tier.stream.wait_event(GRestoreJob(tier, m, 0, 0, 0, Pmax).issue_all())  # the captures overwrote the history
    torch.cuda.synchronize()
    m.gate.zero_()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"[capture] {len(m.gs)} graphs in {time.time() - t0:.0f} s, GPU mem "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    kinds = [("ideal", None), ("ideal_eager", None), ("recompute", None), ("serial", None), ("layer", None)] + \
            [("split", x) for x in xs]
    cases = [(k, x, P) for P in Ps for k, x in kinds] + [(k, None, P) for P in Ps for k in ("load", "save")]
    name = lambda k, x, P: f"{k if x is None else f'split{x:g}'}|P{P}"  # noqa: E731
    raw = {name(*c): [] for c in cases}
    windows = {name(*c): [] for c in cases}
    rot = [0]

    def one(kind, x, P):
        h = rot[0] % args.slots
        rot[0] += 1
        dist.barrier()
        torch.cuda.synchronize()
        w0 = time.time()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        if kind == "load":
            torch.cuda.current_stream().wait_event(GRestoreJob(tier, m, h, 0, 0, P).issue_all())
        elif kind == "save":
            torch.cuda.current_stream().wait_event(tier.save(h, 0, 0, P))
        else:
            run_path(m, tier, kind, hist_all[:P], new, h, x)
        e1.record()
        e1.synchronize()
        w1 = time.time()
        if kind == "load":
            m.gate.zero_()
            torch.cuda.synchronize()
        assert int(m.gate.sum()) == 0, (kind, x, P)
        return e0.elapsed_time(e1), (w0, w1)

    for _ in range(2):  # warm-up every case
        for c in cases:
            one(*c)
    rng = random.Random(1)
    t_start = time.time()
    for r in range(args.rounds):
        order = cases[:]
        rng.shuffle(order)  # same order on every rank
        for c in order:
            ms, w = one(*c)
            raw[name(*c)].append(ms)
            windows[name(*c)].append(w)
        if rank == 0 and (r + 1) % 10 == 0:
            print(f"[progress] round {r + 1}/{args.rounds}, {time.time() - t_start:.0f} s", flush=True)
    allr = [None] * m.world
    dist.all_gather_object(allr, raw)
    if rank == 0:
        rmax = {k: [max(allr[q][k][j] for q in range(m.world)) for j in range(len(raw[k]))] for k in raw}
        summ = {}
        for k, v in rmax.items():
            sv = sorted(v)
            summ[k] = dict(median_ms=statistics.median(v), p25=sv[len(sv) // 4], p75=sv[3 * len(sv) // 4],
                           rounds=len(v))
        nb = lambda P: P * 32 * 1024  # noqa: E731
        for P in Ps:
            rec = summ[f"recompute|P{P}"]["median_ms"]
            line = [f"P={P:<6}"]
            for k, x in kinds:
                s = summ[name(k, x, P)]["median_ms"]
                line.append(f"{name(k, x, P).split('|')[0]} {s:8.2f}" + (f" (x{rec / s:5.1f})" if k != "recompute" else ""))
            ld, sv = summ[f"load|P{P}"]["median_ms"], summ[f"save|P{P}"]["median_ms"]
            line.append(f"load {ld:7.2f} ms = {nb(P) / ld / 1e6:5.1f} GB/s/GPU  save {sv:7.2f} ms = "
                        f"{nb(P) / sv / 1e6:5.1f} GB/s/GPU")
            print("[bench] " + "  ".join(line), flush=True)
        with open(os.path.join(args.out, "bench_kvtier_v3.json"), "w") as f:
            json.dump(dict(rank_max_ms=rmax, per_rank_ms=allr, windows_rank0=windows, summary=summ, args=vars(args),
                           numa_pages_rank0=tier.numa), f, indent=1)


def main():
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("HXB_STACK_DUMP_S", "0")) or 10 ** 7, exit=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "bench"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--P", default="1024,2048,4096,8192,16384,32000")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--split", default="0.0625,0.125,0.25")
    ap.add_argument("--slots", type=int, default=4, help="host slots rotated in bench (cold DRAM)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=32768)
    args = ap.parse_args()
    rank, world, dev = init()
    pin(rank)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    m = GKVEngine(rank, world, dev, 1, args.max_len)
    if args.mode == "check":
        cap = max(len(it["prompt_ids"]) for it in json.load(open(args.reference))["items"])
        tier = KVTier(m, 1, cap)
    else:
        tier = KVTier(m, args.slots, max(int(p) for p in args.P.split(",")))
    nodes = [None] * world
    dist.all_gather_object(nodes, (tier.node, tier.numa))
    dist.barrier()
    if rank == 0:
        print(f"[load] {time.time() - t0:.1f} s, world {world}, GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB,"
              f" host tier {tier.nbytes / 2**30:.2f} GiB per rank, NUMA pages per rank {nodes}", flush=True)
    ok = False
    try:
        if args.mode == "check":
            mode_check(m, tier, rank, args)
        else:
            mode_bench(m, tier, rank, args)
        ok = True
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if ok:
            dist.barrier()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)  # destroy_process_group() hangs after graph-captured NCCL (qwen_tp_v5.py)


if __name__ == "__main__":
    main()
