"""One pipeline-stage boundary pass (the H0 role): GPU -> accelerator -> GPU, at chunk level or op level.

    src GPU:  A = X @ W1                    (producer; n row chunks, one CUDA event per chunk)
    device:   B = op(A)  (lowrank_gelu: gelu(A @ U) @ V, per row chunk)
    dst GPU:  Z = B @ W3                    (consumer; chunk-signaled GEMM, tiles wait for their chunk)

n_chunks = 1 is capability level 0 (each step moves / computes the whole tensor), n_chunks > 1 is level 1.
Every pass is pre-enqueued behind a host gate counter, and the gate opens only after backend.flush() (the device
has received every command), so the timed window excludes host enqueue time and command transport (reported
separately as enqueue_s and submit_s; in steady state they overlap the previous pass). Timed window = dst start event (after the gate) -> dst end event (after the consumer);
cross-check: host time from opening the gate to seeing the pass_done counter that the dst stream writes.
A static device (caps.max_queues == 1) gets its copy_in / op / copy_out interleaved per chunk on its only queue.
"""
import time

import torch

from . import cudamem as cm
from .gpu_consumer import chunk_gemm


class Weights:
    def __init__(self, bridge, q, src, dst, H, R, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.H, self.R = H, R
        self.W1 = (torch.randn(H, H, generator=g) / H ** 0.5).to(torch.bfloat16).to(f"cuda:{src}")
        self.W3 = (torch.randn(H, H, generator=g) / H ** 0.5).to(torch.bfloat16).to(f"cuda:{dst}")
        self.U = torch.randn(H, R, generator=g) / H ** 0.5
        self.V = torch.randn(R, H, generator=g) / R ** 0.5
        self.dU, self.dV = bridge.upload(q, self.U), bridge.upload(q, self.V)


class StagePipeline:
    def __init__(self, bridge, queues, weights, src, dst, M, n_chunks, slots=4, op="lowrank_gelu", seed=1,
                 timing=False):
        be = bridge.be
        self.br, self.be, self.w = bridge, be, weights
        self.src, self.dst, self.M, self.n, self.op = src, dst, M, n_chunks, op
        H = weights.H
        assert M % n_chunks == 0
        self.cr = M // n_chunks
        self.q_in, self.q_op, self.q_out = queues
        g = torch.Generator().manual_seed(seed)
        self.X = (torch.randn(M, H, generator=g) * 0.5).to(torch.bfloat16).to(f"cuda:{src}")
        self.A = torch.empty(M, H, dtype=torch.bfloat16, device=f"cuda:{src}")
        self.B = torch.empty(M, H, dtype=torch.bfloat16, device=f"cuda:{dst}")
        self.Z = torch.empty(M, H, dtype=torch.bfloat16, device=f"cuda:{dst}")
        self.flags = torch.zeros(n_chunks, dtype=torch.int32, device=f"cuda:{dst}")
        self.dA, self.dB = be.alloc((M, H), torch.bfloat16), be.alloc((M, H), torch.bfloat16)
        self.ob = bridge.outbound(src, self.cr, H, torch.bfloat16, slots=min(slots, n_chunks) if n_chunks > 1 else 1,
                                  timing=timing)
        self.ib = bridge.inbound(dst, self.cr, H, torch.bfloat16, slots=min(slots, n_chunks) if n_chunks > 1 else 1,
                                 timing=timing)
        self.computed = be.signal("computed")
        self.gate, self.done = be.signal("gate"), be.signal("pass_done")
        self.cs_src = torch.cuda.Stream(src)
        self.cs_dst = self.cs_src if dst == src else torch.cuda.Stream(dst)
        self.p = 0
        self.last = None
        self._warmup()

    def _warmup(self):
        """Load every kernel before any gated pass. Loading a module (lazy loading, first cuBLAS / Triton launch)
        synchronises the context; with a stream parked on the gate's WaitValue that is a deadlock (seen 2026-10-09:
        first torch.mm hung). Also: build every pipeline before enqueueing any pass (cudaHostRegister too)."""
        cr = self.cr
        with torch.cuda.device(self.src), torch.cuda.stream(self.cs_src):
            torch.mm(self.X[:cr], self.w.W1, out=self.A[:cr])
            torch.mm(self.X, self.w.W1, out=self.A)
        with torch.cuda.device(self.dst), torch.cuda.stream(self.cs_dst):
            done = torch.full((self.n,), 1 << 30, dtype=torch.int32, device=self.Z.device)
            chunk_gemm(self.B, self.w.W3, self.Z, flags=done, chunk_rows=cr, target=1)
            chunk_gemm(self.B, self.w.W3, self.Z)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
        torch.cuda.synchronize(self.src)
        torch.cuda.synchronize(self.dst)

    def enqueue(self):
        """Enqueue one pass behind the gate; returns the pass record (call run() to open the gate)."""
        p, n, cr = self.p, self.n, self.cr
        self.p += 1
        t_enq0 = time.perf_counter()
        rec = dict(p=p)
        with torch.cuda.device(self.src):
            cm.wait_geq(self.cs_src, self.gate.addr, p + 1)
        with torch.cuda.device(self.dst):
            if self.cs_dst is not self.cs_src:
                cm.wait_geq(self.cs_dst, self.gate.addr, p + 1)
            with torch.cuda.stream(self.cs_dst):  # same stream as the producer when src == dst: before it
                rec["dst_start"] = torch.cuda.Event(enable_timing=True)
                rec["dst_start"].record()
        with torch.cuda.device(self.src):
            ev = []
            with torch.cuda.stream(self.cs_src):
                rec["src_start"] = torch.cuda.Event(enable_timing=True)
                rec["src_start"].record()
                for i in range(n):
                    torch.mm(self.X[i * cr:(i + 1) * cr], self.w.W1, out=self.A[i * cr:(i + 1) * cr])
                    e = torch.cuda.Event(enable_timing=True)
                    e.record()
                    ev.append(e)
        rec["prod_events"] = ev
        with torch.cuda.device(self.dst):
            with torch.cuda.stream(self.cs_dst):
                chunk_gemm(self.B, self.w.W3, self.Z, flags=self.flags, chunk_rows=cr, target=p + 1)
                rec["dst_end"] = torch.cuda.Event(enable_timing=True)
                rec["dst_end"].record()
            cm.write64(self.cs_dst, self.done.addr, p + 1)
        for i in range(n):
            j = p * n + i
            rows = (i * cr, (i + 1) * cr)
            sig, v = self.ob.send(self.A, rows[0], rows[1], ev[i], self.dA, rows, self.q_in, tag=f"in{j}")
            self.be.launch(self.q_op, self.op, {"x": self.dA, "u": self.w.dU, "v": self.w.dV, "y": self.dB,
                                                "rows": rows}, wait=[(sig, v)], done=(self.computed, j + 1),
                           tag=f"op{j}")
            self.ib.recv(self.dB, rows, self.q_out, [(self.computed, j + 1)], self.B, rows[0], rows[1],
                         flag=(self.flags.data_ptr() + 4 * i, p + 1), tag=f"out{j}")
        rec["enqueue_s"] = time.perf_counter() - t_enq0
        self.last = rec
        return rec

    def run(self, timeout=60.0):
        """Open the gate of the enqueued pass and wait for it; returns the pass record with timings."""
        rec = self.last
        p = rec["p"]
        t_f = time.perf_counter()
        self.be.flush(timeout)  # the device holds every command of this pass before the gate opens
        t0 = time.perf_counter()
        rec["submit_s"] = t0 - t_f
        self.gate.set(p + 1)
        self.done.wait(p + 1, timeout)
        rec["host_s"] = time.perf_counter() - t0
        rec["t_gate"] = t0
        torch.cuda.synchronize(self.dst)
        torch.cuda.synchronize(self.src)
        rec["gpu_s"] = rec["dst_start"].elapsed_time(rec["dst_end"]) / 1e3
        rec["prod_s"] = rec["src_start"].elapsed_time(rec["prod_events"][-1]) / 1e3
        return rec

    def reference(self):
        """fp32 reference of the whole pass, computed on the dst GPU."""
        A = (self.X.float() @ self.w.W1.float().to(self.X.device)).to(torch.bfloat16)
        Bf = torch.nn.functional.gelu(A.float() @ self.w.U.to(A.device)) @ self.w.V.to(A.device)
        B = Bf.to(torch.bfloat16).to(self.Z.device)
        return (B.float() @ self.w.W3.float()).to(torch.bfloat16)
