################################################################################
# hetero-proxy I2 (2026-10-09): acceptance tests for an hxb backend (spec section 4 / 7), no GPU needed for the
# CPU backend. Every new accelerator backend must pass these before the Bridge is used with it.
#   T1 copy round trip           host -> device -> host is bit-exact (bf16, fp32; partial row ranges)
#   T2 in order                  op k starts after op k-1 ends (trace), even when op k-1 is a slow sleep
#   T3 wait                      an op waiting on a host-set signal does not start before the host sets it
#   T4 release                   once done is observed, the op's writes are visible (read immediately, many times)
#   T5 monotonic                 a lower store never lowers a counter
#   T6 cross-queue dependency    compute on queue B waits for copy_in on queue A (B submitted first)
#   T7 op correctness            gemm / lowrank_gelu against a float64 torch reference (max rel err < 2e-2: bf16)
#   T8 static device             with one queue, interleaved submission (in_i, op_i, out_i) completes
# Usage: python test_hxb_backend_v1.py [--backend cpu|proxy] [--device 2] [--log file]
################################################################################
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hxb.cpu_backend import CpuBackend  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def nxt(sig):
    """Next target value for `sig`, counted by the submitter. Never derive it from sig.value(): an earlier
    asynchronous op may not have completed yet, and two ops would share one target (this test had exactly that
    race until 2026-10-09; it surfaced as 1/200 stale reads)."""
    sig.issued = getattr(sig, "issued", 0) + 1
    return sig.issued


def put(be, q, t, sig):
    """Upload a host tensor through a HostBuf; returns (DevBuf, HostBuf)."""
    h = be.host_alloc(t.shape, t.dtype)
    h.tensor.copy_(t)
    d = be.alloc(t.shape, t.dtype)
    be.copy_in(q, d, None, h, None, done=(sig, nxt(sig)))
    return d, h


def get(be, q, d, sig):
    h = be.host_alloc(d.shape, d.dtype)
    v = nxt(sig)
    be.copy_out(q, h, None, d, None, done=(sig, v))
    sig.wait(v)
    return h.tensor.clone()


def run(be):
    caps = be.caps()
    print(f"backend {caps}", flush=True)
    q1, q2 = be.queue(), be.queue()
    s = be.signal("t")
    g = torch.Generator().manual_seed(0)

    # T1
    ok = True
    for dt in (torch.bfloat16, torch.float32):
        x = torch.randn(1000, 777, generator=g).to(dt)
        d, _ = put(be, q1, x, s)
        y = get(be, q1, d, s)
        ok &= torch.equal(x, y)
        h = be.host_alloc((300, 777), dt)
        v = nxt(s)
        be.copy_out(q1, h, (0, 300), d, (500, 800), done=(s, v))
        s.wait(v)
        ok &= torch.equal(h.tensor, x[500:800])
    check("T1 copy round trip", ok)

    # T2
    be.trace()
    d = be.alloc((64, 64), torch.float32)
    be.launch(q1, "sleep", {"seconds": 0.05}, tag="slow")
    v = nxt(s)
    be.launch(q1, "axpb", {"x": d, "y": d, "rows": (0, 64), "a": 1.0, "b": 1.0}, done=(s, v), tag="next")
    s.wait(v)
    tr = {r["tag"]: r for r in be.trace()}
    check("T2 in order", tr["next"]["t_start"] >= tr["slow"]["t_end"],
          f"(next starts {1e3 * (tr['next']['t_start'] - tr['slow']['t_end']):.3f} ms after slow ends)")

    # T3
    gate = be.signal("gate")
    v = nxt(s)
    be.launch(q1, "axpb", {"x": d, "y": d, "rows": (0, 64), "a": 1.0, "b": 0.0}, wait=[(gate, 1)], done=(s, v),
              tag="gated")
    time.sleep(0.05)
    started_early = s.value() >= v
    t_set = time.perf_counter()
    gate.set(1)
    s.wait(v)
    tr = {r["tag"]: r for r in be.trace()}
    # primary: the op had not completed during the 50 ms before the set; the trace check allows 1 ms because a
    # device-timed trace (CUDA events) is aligned to the host clock only approximately
    check("T3 wait", not started_early and tr["gated"]["t_start"] >= t_set - 1e-3,
          f"(start - set = {1e6 * (tr['gated']['t_start'] - t_set):.1f} us)")

    # T4: overwrite a device buffer in many steps; after each done, the host copy must equal the expectation
    n = 200
    x = torch.zeros(256, 1024, dtype=torch.float32)
    d, _ = put(be, q1, x, s)
    h = be.host_alloc((256, 1024), torch.float32)
    bad = 0
    for i in range(1, n + 1):
        be.launch(q1, "axpb", {"x": d, "y": d, "rows": (0, 256), "a": 1.0, "b": 1.0})
        v = nxt(s)
        be.copy_out(q1, h, None, d, None, done=(s, v))
        s.wait(v)
        bad += int(not torch.all(h.tensor == float(i)).item())
    check("T4 release", bad == 0, f"({bad}/{n} stale reads)")

    # T5
    m = be.signal("mono")
    m.set(10)
    m.set(3)
    check("T5 monotonic", m.value() == 10)

    # T6: B (compute) is submitted before A (copy_in) and must wait for it
    x = torch.randn(512, 256, generator=g)
    hx = be.host_alloc(x.shape, x.dtype)
    hx.tensor.copy_(x)
    dx, dy = be.alloc(x.shape, x.dtype), be.alloc(x.shape, x.dtype)
    arrived, computed = be.signal("arrived"), be.signal("computed")
    be.launch(q2, "axpb", {"x": dx, "y": dy, "rows": (0, 512), "a": 2.0, "b": 0.0}, wait=[(arrived, 1)],
              done=(computed, 1))
    be.launch(q1, "sleep", {"seconds": 0.02})
    be.copy_in(q1, dx, None, hx, None, done=(arrived, 1))
    computed.wait(1)
    y = get(be, q1, dy, s)
    check("T6 cross-queue dependency", torch.equal(y, 2 * x))

    # T7
    M, H, R = 384, 1024, 64
    x = (torch.randn(M, H, generator=g) * 0.5).to(torch.bfloat16)
    w = torch.randn(H, H, generator=g) / H ** 0.5
    u, vv = torch.randn(H, R, generator=g) / H ** 0.5, torch.randn(R, H, generator=g) / R ** 0.5
    dx, _ = put(be, q1, x, s)
    dw, _ = put(be, q1, w, s)
    du, _ = put(be, q1, u, s)
    dv, _ = put(be, q1, vv, s)
    dy = be.alloc((M, H), torch.bfloat16)
    for r0 in range(0, M, 128):
        be.launch(q1, "gemm", {"x": dx, "w": dw, "y": dy, "rows": (r0, r0 + 128)})
    y1 = get(be, q1, dy, s).double()
    ref1 = x.double() @ w.double()
    for r0 in range(0, M, 128):
        be.launch(q1, "lowrank_gelu", {"x": dx, "u": du, "v": dv, "y": dy, "rows": (r0, r0 + 128)})
    y2 = get(be, q1, dy, s).double()
    ref2 = torch.nn.functional.gelu(x.double() @ u.double()) @ vv.double()
    e1 = ((y1 - ref1).abs().max() / ref1.abs().max()).item()
    e2 = ((y2 - ref2).abs().max() / ref2.abs().max()).item()
    check("T7 op correctness", e1 < 2e-2 and e2 < 2e-2, f"(max rel err gemm {e1:.2e}, lowrank_gelu {e2:.2e}; bf16 output)")
    be.sync()


def run_static():
    be = CpuBackend(scheduling="static", dma_threads=4)
    try:
        q = be.queue()
        check("T8a static: queue() always returns the same queue", be.queue() == q)
        g = torch.Generator().manual_seed(1)
        x = torch.randn(1024, 512, generator=g)
        hin, hout = be.host_alloc(x.shape, x.dtype), be.host_alloc(x.shape, x.dtype)
        hin.tensor.copy_(x)
        dx, dy = be.alloc(x.shape, x.dtype), be.alloc(x.shape, x.dtype)
        a, c, o = be.signal("a"), be.signal("c"), be.signal("o")
        for i in range(8):
            r = (i * 128, (i + 1) * 128)
            be.copy_in(q, dx, r, hin, r, done=(a, i + 1))
            be.launch(q, "axpb", {"x": dx, "y": dy, "rows": r, "a": 3.0, "b": 0.0}, wait=[(a, i + 1)], done=(c, i + 1))
            be.copy_out(q, hout, r, dy, r, wait=[(c, i + 1)], done=(o, i + 1))
        o.wait(8, timeout=30)
        check("T8 static device, interleaved submission", torch.equal(hout.tensor, 3 * x))
    finally:
        be.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--device", type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.backend == "cpu":
        os.sched_setaffinity(0, [c for c in range(0, 32)])
        be = CpuBackend()
    else:
        from hxb.proxy_backend import ProxyBackend
        be = ProxyBackend(device=args.device)
    try:
        run(be)
    finally:
        be.close()
    if args.backend == "cpu":
        run_static()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"SUMMARY backend={args.backend}: {len(RESULTS) - n_fail}/{len(RESULTS)} passed", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
