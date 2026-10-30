"""V3 v3 (2026-10-30). Adds variant 'hybrid' (gdpu only): the first B bytes go through the window policy (B = sum of
calibrated gap budgets); the excess E = total - B runs on a second side stream as 4 MiB pieces paced at
duty = E / (0.9 x calibrated no-copy step time x rate), starting at step start, so it overlaps collectives at a low rate
instead of spilling past the step end. Motivated by v2: in the balanced case (copy time ~ step time) the window policy
cannot fit the copies into the gaps.
v2 docstring follows.

V3 v2 (2026-10-30). Changes vs v1 (smoke showed the act window variant stalling on leftovers):
  - release points also before the down-projection GEMM (forward) and before every ColLin dgrad (backward), and at
    step start; every point's budget = calibrated time from the point to the next collective start x rate x 0.85;
  - act: the last layer is not offloaded (it is needed first in backward; standard practice);
  - new variants paced50 / paced25: naive timing, but 4 MiB pieces with a GPU sleep in between so the copy runs
    50% / 25% of the time (rate-limit baseline).
v1 docstring follows.

V3 (2026-10-30, topic T validation; plan: reports/20261030_t_validation_plan.md): does host<->GPU offload traffic cost
step time in a real tensor-parallel training step, and can a simple window scheduler recover it?

Model: Qwen2.5-32B decoder layers (hidden 5120, 40 q heads / 8 kv heads, head 128, FFN 27648; no RoPE, no bias),
TP8 + sequence parallel (Megatron style): per rank 5 q heads / 1 kv head / FFN 3456; norm weights replicated
(their gradient all-reduce is omitted). Hand-written autograd for the TP linears so every collective and GEMM is
explicit: ColLin = all-gather -> GEMM (backward: dgrad -> reduce-scatter -> wgrad); RowLin = GEMM -> reduce-scatter
(backward: all-gather -> dgrad -> wgrad). Flash attention (causal).

Scenarios (--scenario):
  gimm : gradient offload, immediate. Each weight gradient is copied to pinned host memory once it is computed
         (ZeRO-Offload without delayed update). 1 micro-batch.
  gdpu : gradient offload with delayed parameter update. The previous step's gradients (same sizes) are copied during
         this step's --micro micro-batches.
  act  : activation offload. The SwiGLU output 'a' (input of the down projection) of every layer is copied to host in
         the forward pass and copied back one layer ahead in the backward pass; backward uses the reloaded copy.
         (Timing study: the GPU copy is not freed.)
Variants (interleaved per round, same order on all ranks):
  none   : no copies.
  naive  : copy as soon as data is ready, whole tensor, on a side stream (what frameworks do).
  serial : no overlap (gimm/gdpu: all copies after the step's compute; act: copies on the compute stream).
  window : copies split into 4 MiB pieces and released only right after a collective ends, with a byte budget
           = (measured gap until the next collective) x copy rate x 0.85; leftovers flushed at the end.
           The act scenario's backward reload stays as in naive (prefetch one layer ahead).
Per step: rank-max of max(compute end, copy end) from CUDA events; per-collective durations recorded in-step.
Lower bound per scenario in the analysis. --check compares TP gradients with a single-GPU fp32 reference on rank 0.
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
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../common/measure"))
from clock_logger import ClockLogger  # noqa: E402

H, NH, NKV, HD, FFN = 5120, 40, 8, 128, 27648
MiB = 1 << 20
PIECE = 4 * MiB
RATE = 12.6e6  # bytes per ms per GPU when all 8 GPUs copy (T1)
SAFETY = 0.85
DT = torch.bfloat16
NUMA_CPUS = {0: list(range(0, 32)) + list(range(64, 96)), 1: list(range(32, 64)) + list(range(96, 128))}


class Sched:
    """Owns the side stream, the copy policy, collective marks and window budgets."""

    def __init__(self, side):
        self.side = side
        self.variant = "none"
        self.calib = False
        self.budgets = {}
        self.reset("none")

    def reset(self, variant):
        self.variant = variant
        self.pending = []      # (src_view, dst_view) pieces
        self.keep = []         # keep sources alive until the step is synchronized
        self.idx = 0           # collective ordinal within the step
        self.marks = []        # (idx, ev_start, ev_end)
        self.h2d_ev = {}
        self.issued_any = False
        self.points = []       # (point ordinal, event, next collective ordinal)
        self.side2_used = False
        self.hybrid_info = None
        self.pidx = 0

    # --- collectives -------------------------------------------------------------------------------------
    def coll(self, fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        self.marks.append((self.idx, s, e))
        self.idx += 1
        self.point()
        return out

    def point(self):
        """A release opportunity; budget = calibrated time to the next collective start."""
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.points.append((self.pidx, ev, self.idx))
        if self.variant in ("window", "hybrid") and self.pending:
            self.side.wait_event(ev)
            self._release(self.budgets.get(self.pidx, 0.0))
        self.pidx += 1

    # --- copies -----------------------------------------------------------------------------------------
    def hybrid_split(self, budget_total, t_step_ms, side2):
        """Move pieces beyond the window budget to a paced second stream (issued now)."""
        tot = sum(sb.numel() for sb, _ in self.pending)
        excess = tot - budget_total
        if excess <= 0:
            return
        keep, extra, acc = [], [], 0
        for pc in self.pending:
            if acc + pc[0].numel() <= budget_total:
                keep.append(pc)
                acc += pc[0].numel()
            else:
                extra.append(pc)
        self.pending = keep
        ex = sum(sb.numel() for sb, _ in extra)
        duty = min(1.0, ex / (0.9 * t_step_ms * RATE))
        gap = int(PIECE / (RATE * 1e3) * (1 - duty) / duty * 1.41e9) if duty < 1 else 0
        ev = torch.cuda.Event()
        ev.record()
        side2.wait_event(ev)
        with torch.cuda.stream(side2):
            for sb, db in extra:
                db.copy_(sb, non_blocking=True)
                if gap:
                    torch.cuda._sleep(gap)
        self.side2_used = True
        self.hybrid_info = (ex, duty)

    def _side_after_now(self):
        ev = torch.cuda.Event()
        ev.record()
        self.side.wait_event(ev)

    def offload(self, src, dst, serial_on_compute=False):
        """src (GPU) -> dst (pinned host), per the current variant."""
        if self.variant == "none":
            return
        self.keep.append(src)
        if self.variant == "naive":
            self._side_after_now()
            with torch.cuda.stream(self.side):
                dst.copy_(src, non_blocking=True)
            self.issued_any = True
        elif self.variant.startswith("paced"):
            duty = int(self.variant[5:]) / 100
            gap = int(PIECE / (RATE * 1e3) * (1 - duty) / duty * 1.41e9)
            self._side_after_now()
            sb = src.view(-1).view(torch.uint8)
            db = dst.view(-1).view(torch.uint8)
            with torch.cuda.stream(self.side):
                for o in range(0, sb.numel(), PIECE):
                    db[o:o + PIECE].copy_(sb[o:o + PIECE], non_blocking=True)
                    torch.cuda._sleep(gap)
            self.issued_any = True
        elif self.variant == "serial" and serial_on_compute:
            dst.copy_(src, non_blocking=True)
        else:  # window, or serial (flushed at the end)
            sb = src.view(-1).view(torch.uint8)
            db = dst.view(-1).view(torch.uint8)
            for o in range(0, sb.numel(), PIECE):
                self.pending.append((sb[o:o + PIECE], db[o:o + PIECE]))

    def _release(self, budget):
        with torch.cuda.stream(self.side):
            while self.pending and budget >= self.pending[0][0].numel():
                s, d = self.pending.pop(0)
                d.copy_(s, non_blocking=True)
                budget -= s.numel()
                self.issued_any = True

    def flush(self):
        if self.pending:
            self._side_after_now()
            with torch.cuda.stream(self.side):
                for s, d in self.pending:
                    d.copy_(s, non_blocking=True)
            self.pending = []
            self.issued_any = True

    def reload(self, li, src_host, dst_gpu):
        """act scenario: H2D prefetch of layer li's activation (side stream, after now)."""
        if self.variant == "none":
            return
        if self.variant == "serial":
            dst_gpu.copy_(src_host, non_blocking=True)
            return
        self._side_after_now()
        with torch.cuda.stream(self.side):
            dst_gpu.copy_(src_host, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        self.h2d_ev[li] = ev
        self.issued_any = True

    def wait_reload(self, li):
        ev = self.h2d_ev.get(li)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)


S = None  # Sched, set in main
CFG = None


def ag(x):
    out = torch.empty((x.shape[0] * CFG.W, *x.shape[1:]), dtype=x.dtype, device=x.device)
    return S.coll(lambda: (dist.all_gather_into_tensor(out, x), out)[1])


def rs(x):
    out = torch.empty((x.shape[0] // CFG.W, *x.shape[1:]), dtype=x.dtype, device=x.device)
    return S.coll(lambda: (dist.reduce_scatter_tensor(out, x.contiguous()), out)[1])


class ColLin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sp, w, li):
        xf = ag(x_sp)
        ctx.save_for_backward(xf, w)
        ctx.li = li
        return xf @ w.t()

    @staticmethod
    def backward(ctx, gy):
        xf, w = ctx.saved_tensors
        S.point()
        dx = rs(gy @ w)
        dw = gy.t() @ xf
        CFG.on_grad(w, dw)
        return dx, dw, None


class RowLin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, w, li, is_down):
        if is_down:
            CFG.on_act(li, a)
            S.point()
        y = rs(a @ w.t())
        ctx.save_for_backward(a, w)
        ctx.li, ctx.is_down = li, is_down
        return y

    @staticmethod
    def backward(ctx, gy_sp):
        a, w = ctx.saved_tensors
        if ctx.is_down:
            a = CFG.on_down_bwd(ctx.li, a)
        gyf = ag(gy_sp)
        da = gyf @ w
        dw = gyf.t() @ a
        CFG.on_grad(w, dw)
        return da, dw, None, None


def rms(x, g):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * g


def attention(qkv, nq, nkv):
    Sx = qkv.shape[0]
    q = qkv[:, :nq * HD].view(Sx, nq, HD).transpose(0, 1).unsqueeze(0)
    k = qkv[:, nq * HD:(nq + nkv) * HD].view(Sx, nkv, HD).transpose(0, 1).unsqueeze(0)
    v = qkv[:, (nq + nkv) * HD:].view(Sx, nkv, HD).transpose(0, 1).unsqueeze(0)
    k = k.repeat_interleave(nq // nkv, dim=1) if nkv > 1 else k.expand(1, nq, Sx, HD)
    v = v.repeat_interleave(nq // nkv, dim=1) if nkv > 1 else v.expand(1, nq, Sx, HD)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.squeeze(0).transpose(0, 1).reshape(Sx, nq * HD)


class Cfg:
    pass


def build(cfg, rank):
    torch.manual_seed(1234 + rank)
    nq, nkv, ffn = NH // cfg.W, NKV // cfg.W, FFN // cfg.W
    layers = []
    for _ in range(cfg.L):
        def p(*shape):
            return (torch.randn(*shape, device="cuda") * 0.02).to(DT).requires_grad_()
        layers.append({
            "g1": torch.ones(H, device="cuda", dtype=DT).requires_grad_(),
            "g2": torch.ones(H, device="cuda", dtype=DT).requires_grad_(),
            "wqkv": p((nq + 2 * nkv) * HD, H), "wo": p(H, nq * HD), "wgu": p(2 * ffn, H), "wdown": p(H, ffn)})
    return layers, nq, nkv


def forward(layers, x, nq, nkv):
    for li, Lw in enumerate(layers):
        h = rms(x, Lw["g1"])
        qkv = ColLin.apply(h, Lw["wqkv"], li)
        o = RowLin.apply(attention(qkv, nq, nkv), Lw["wo"], li, False)
        x = x + o
        h2 = rms(x, Lw["g2"])
        gu = ColLin.apply(h2, Lw["wgu"], li)
        g, u = gu.chunk(2, dim=-1)
        x = x + RowLin.apply(F.silu(g) * u, Lw["wdown"], li, True)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", choices=["gimm", "gdpu", "act"], default="gimm")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--variants", default="none,naive,serial,window,hybrid")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, W = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    os.sched_setaffinity(0, NUMA_CPUS[0 if lr < 4 else 1])
    global S, CFG
    S = Sched(torch.cuda.Stream())
    CFG = Cfg()
    CFG.W, CFG.L = W, args.layers
    if args.check:
        CFG.L = 2
        args.seq = 512
    layers, nq, nkv = build(CFG, rank)
    params = [Lw[k] for Lw in layers for k in ("wqkv", "wo", "wgu", "wdown")]
    host_g = {id(p): torch.empty(p.shape, dtype=DT, pin_memory=True) for p in params}
    prev_g = {id(p): torch.randn(p.shape, device="cuda").to(DT) for p in params}  # gdpu: last step's gradients
    ffn = FFN // W
    act_host = [torch.empty((args.seq, ffn), dtype=DT, pin_memory=True) for _ in range(CFG.L)]
    act_reload = [torch.empty((args.seq, ffn), dtype=DT, device="cuda") for _ in range(CFG.L)]

    scen = args.scenario
    CFG.on_grad = (lambda w, dw: S.offload(dw, host_g[id(w)])) if scen == "gimm" else (lambda w, dw: None)
    CFG.on_act = ((lambda li, a: S.offload(a, act_host[li], serial_on_compute=True) if li < CFG.L - 1 else None)
                  if scen == "act" else (lambda li, a: None))

    def on_down_bwd(li, a):
        if scen != "act" or S.variant == "none":
            return a
        if li > 0:
            S.reload(li - 1, act_host[li - 1], act_reload[li - 1])  # prefetch one layer ahead
        if li == CFG.L - 1:
            return a  # last layer not offloaded
        S.wait_reload(li)
        return act_reload[li]
    CFG.on_down_bwd = on_down_bwd

    x_sp = torch.randn((args.seq // W, H), device="cuda").to(DT)
    gy = (torch.randn((args.seq // W, H), device="cuda") * 1e-2).to(DT)

    if args.check:
        run_check(layers, x_sp, gy, nq, nkv, rank, W, args)
        dist.destroy_process_group()
        return

    side2 = torch.cuda.Stream()
    T_NONE = [0.0]

    def step(variant):
        S.reset(variant)
        for p in params:
            p.grad = None
        t0 = torch.cuda.Event(enable_timing=True)
        t0.record()
        if scen == "gdpu":
            for p in params:
                S.offload(prev_g[id(p)], host_g[id(p)])
            if variant == "hybrid":
                S.hybrid_split(sum(S.budgets.values()), T_NONE[0], side2)
        S.point()
        fwd_end = None
        for m in range(args.micro):
            y = forward(layers, x_sp, nq, nkv)
            if fwd_end is None:
                fwd_end = torch.cuda.Event(enable_timing=True)
                fwd_end.record()
            if scen == "act" and S.variant == "window":
                S.flush()  # D2H pieces not yet released must precede the reloads (same side stream)
            y.backward(gy)
        if S.variant in ("serial", "window", "hybrid"):
            S.flush()
        ce = torch.cuda.Event(enable_timing=True)
        ce.record()
        se = None
        if S.issued_any:
            se = torch.cuda.Event(enable_timing=True)
            S.side.record_event(se)
        se2 = None
        if S.side2_used:
            se2 = torch.cuda.Event(enable_timing=True)
            side2.record_event(se2)
        torch.cuda.synchronize()
        if se2 is not None:
            t_s2 = t0.elapsed_time(se2)
            se = se2 if se is None or t_s2 > t0.elapsed_time(se) else se
        t_c = t0.elapsed_time(ce)
        t_s = t0.elapsed_time(se) if se is not None else 0.0
        colls = [(i, s.elapsed_time(e)) for i, s, e in S.marks]
        starts = {i: s for i, s, e in S.marks}
        gaps = {}
        for pi, ev, nxt in S.points:
            gaps[pi] = ev.elapsed_time(starts[nxt]) if nxt in starts else ev.elapsed_time(ce)
        S.keep = []
        return {"step": max(t_c, t_s), "compute": t_c, "copy_end": t_s, "fwd": t0.elapsed_time(fwd_end),
                "coll_sum": sum(c for _, c in colls), "gaps": gaps}

    variants = args.variants.split(",")
    for v in variants:  # warm-up (hybrid needs the calibration; it warms up after it)
        if v != "hybrid":
            for _ in range(2):
                step(v)
    # calibration of window budgets from 'none' steps (per collective ordinal: gap until the next collective)
    cal_runs = [step("none") for _ in range(5)]
    cal = [c["gaps"] for c in cal_runs]
    S.budgets = {i: statistics.median(c[i] for c in cal) * RATE * SAFETY for i in cal[0]}
    T_NONE[0] = statistics.median(c["compute"] for c in cal_runs)

    if "hybrid" in variants:
        for _ in range(2):
            step("hybrid")
    rec = []
    with ClockLogger(lr, period_s=0.25) as clk:
        for r in range(args.rounds):
            order = variants[:]
            random.Random(3000 + r).shuffle(order)
            for v in order:
                dist.barrier()
                w0 = time.perf_counter()
                res = step(v)
                res.pop("gaps")
                rec.append({"round": r, "variant": v, **res, "w": (w0, time.perf_counter())})
        for x in rec:
            x["clock"] = clk.min_clock_between(*x["w"])
            del x["w"]

    allrec = [None] * W
    dist.all_gather_object(allrec, rec)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        gbytes = sum(p.numel() * 2 for p in params)
        meta = {"scenario": scen, "seq": args.seq, "layers": CFG.L, "micro": args.micro, "rounds": args.rounds,
                "grad_bytes_per_gpu": gbytes, "act_bytes_per_gpu": (CFG.L - 1) * args.seq * ffn * 2,
                "budget_total_bytes": sum(S.budgets.values()),
                "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL", "CUDA"))}}
        with open(os.path.join(args.out, "v3_raw.json"), "w") as f:
            json.dump({"meta": meta, "ranks": allrec}, f)
        summarize(allrec, variants, meta, args.out)
    dist.barrier()
    dist.destroy_process_group()


def summarize(allrec, variants, meta, out):
    by = {}
    for recs in allrec:
        for x in recs:
            by.setdefault((x["round"], x["variant"]), []).append(x)
    clocks = [x["clock"] for recs in allrec for x in recs if x["clock"]]
    modal = statistics.mode(clocks)
    agg = {v: {"step": [], "compute": [], "coll": [], "fwd": []} for v in variants}
    dropped = 0
    for (r, v), xs in by.items():
        if any(x["clock"] and x["clock"] < 0.95 * modal for x in xs):
            dropped += 1
            continue
        agg[v]["step"].append((r, max(x["step"] for x in xs)))
        agg[v]["compute"].append(max(x["compute"] for x in xs))
        agg[v]["coll"].append(max(x["coll_sum"] for x in xs))
        agg[v]["fwd"].append(max(x["fwd"] for x in xs))
    med = {v: statistics.median(t for _, t in agg[v]["step"]) for v in variants}
    lines = [f"V3 {meta['scenario']} seq={meta['seq']} layers={meta['layers']} micro={meta['micro']} "
             f"grad MB/GPU={meta['grad_bytes_per_gpu'] / 1e6:.0f} act MB/GPU(offloaded, last layer kept)={meta['act_bytes_per_gpu'] / 1e6:.0f} "
             f"window budget total MB={meta['budget_total_bytes'] / 1e6:.0f}; modal clock {modal}; dropped windows {dropped}",
             "variant  step_ms(med)  ratio_vs_none[p10-p90]  compute_ms  coll_sum_ms(rank-max)  fwd_ms"]
    base = dict(agg["none"]["step"]) if "none" in agg else {}
    rows = {}
    for v in variants:
        rs_ = sorted(t / base[r] for r, t in agg[v]["step"] if r in base)
        rows[v] = {"step_ms": med[v], "ratio": statistics.median(rs_), "p10": rs_[len(rs_) // 10],
                   "p90": rs_[(9 * len(rs_)) // 10], "compute_ms": statistics.median(agg[v]["compute"]),
                   "coll_ms": statistics.median(agg[v]["coll"]), "fwd_ms": statistics.median(agg[v]["fwd"]), "n": len(rs_)}
        q = rows[v]
        lines.append(f"{v:7s} {q['step_ms']:9.2f}  x{q['ratio']:.3f} [{q['p10']:.3f}-{q['p90']:.3f}]  "
                     f"{q['compute_ms']:9.2f}  {q['coll_ms']:9.2f}  {q['fwd_ms']:8.2f}  n={q['n']}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out, "v3_summary.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(out, "v3_rows.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)


def run_check(layers, x_sp, gy, nq, nkv, rank, W, args):
    """TP gradients vs single-GPU fp32 reference (rank 0), seq 512, 2 layers."""
    S.reset("none")
    y = forward(layers, x_sp, nq, nkv)
    y.backward(gy)
    torch.cuda.synchronize()
    names = ("wqkv", "wo", "wgu", "wdown")
    shards = {n: [] for n in names}
    grads = {n: [] for n in names}
    for li, Lw in enumerate(layers):
        for n in names:
            ws = [torch.empty_like(Lw[n]) for _ in range(W)]
            gs = [torch.empty_like(Lw[n]) for _ in range(W)]
            dist.all_gather(ws, Lw[n].detach())
            dist.all_gather(gs, Lw[n].grad)
            shards[n].append(ws)
            grads[n].append(gs)
    xs = [torch.empty_like(x_sp) for _ in range(W)]
    gys = [torch.empty_like(gy) for _ in range(W)]
    ys = [torch.empty_like(y) for _ in range(W)]
    dist.all_gather(xs, x_sp)
    dist.all_gather(gys, gy)
    dist.all_gather(ys, y.detach().contiguous())
    if rank != 0:
        return
    ffn = FFN // W
    ref_w = [{n: [w.float().clone().requires_grad_() for w in shards[n][li]] for n in names} for li in range(len(layers))]
    x = torch.cat(xs).float()
    for li in range(len(layers)):
        Rw = ref_w[li]
        h = rms(x, layers[li]["g1"].float().detach())
        o = 0
        for r in range(W):
            qkv = h @ Rw["wqkv"][r].t()
            o = o + attention_f32(qkv, nq, nkv) @ Rw["wo"][r].t()
        x = x + o
        h2 = rms(x, layers[li]["g2"].float().detach())
        d = 0
        for r in range(W):
            gu = h2 @ Rw["wgu"][r].t()
            g, u = gu[:, :ffn], gu[:, ffn:]
            d = d + (F.silu(g) * u) @ Rw["wdown"][r].t()
        x = x + d
    (x * torch.cat(gys).float()).sum().backward()
    yerr = (x.detach() - torch.cat(ys).float()).abs().max().item() / x.detach().abs().max().item()
    lines = [f"check seq={args.seq} layers={len(layers)}: output rel max err {yerr:.4f}"]
    worst = 0.0
    for li in range(len(layers)):
        for n in names:
            for r in range(W):
                ref = ref_w[li][n][r].grad
                got = grads[n][li][r].float()
                e = (ref - got).abs().max().item() / ref.abs().max().item()
                worst = max(worst, e)
            lines.append(f"  layer {li} {n}: worst rel max err over ranks so far {worst:.4f}")
    lines.append(f"RESULT {'PASS' if worst < 0.05 and yerr < 0.05 else 'FAIL'} (threshold 0.05; bf16 TP vs fp32 reference)")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "check.txt"), "w") as f:
        f.write(txt + "\n")


def attention_f32(qkv, nq, nkv):
    Sx = qkv.shape[0]
    q = qkv[:, :nq * HD].view(Sx, nq, HD).transpose(0, 1)
    k = qkv[:, nq * HD:(nq + nkv) * HD].view(Sx, nkv, HD).transpose(0, 1).expand(nq, Sx, HD)
    v = qkv[:, (nq + nkv) * HD:].view(Sx, nkv, HD).transpose(0, 1).expand(nq, Sx, HD)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(0, 1).reshape(Sx, nq * HD)


if __name__ == "__main__":
    main()
