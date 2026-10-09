"""A100 proxy backend: another A100 plays a PCIe-only accelerator (hetero-proxy STATUS 1, I4).

The device never uses NVLink: peer access is never enabled and nothing ever copies GPU <-> GPU directly; every byte
goes device memory <-> registered host staging (cuMemcpyAsync, the proxy's copy engines). Verify with the NVLink
counters (`nvidia-smi nvlink -gt d`, readable without sudo) before / after a run.

It is the "hardware semaphore" kind of device: a queue is a CUDA stream on the proxy GPU, a wait is
cuStreamWaitValue64 on the host counter, `done` is cuStreamWriteValue64 after the op - no host thread anywhere.
Ops run in bf16 on the proxy GPU (gemm, lowrank_gelu, axpb) or as a calibrated spin (`emulate_ms_per_gflop`:
timing-only compute, models a slower chip; output not written).

A100 property the CPU backend does not have: one A100 receiving (H2D) while it sends (D2H) drops its H2D to ~6 GB/s
(PHASE0_FINDINGS 2.4; I2 v2 section D). caps().full_duplex is False; a Bridge user should then give copy_in and
copy_out one queue (directions serialised) and submit sends ahead of receives (StagePipeline lookahead).
"""
import itertools
import time

import torch

from . import cudamem as cm
from .api import Backend, Caps, DevBuf, HostBuf, Queue, Signal, SignalTable, rows_or_all


class ProxyBackend(Backend):
    def __init__(self, device=2, emulate_ms_per_gflop=None, name=None):
        self.dev = device
        self._name = name or f"a100-proxy-gpu{device}"
        self._table = SignalTable()
        cm.register(self._table.addr, self._table.nbytes)
        self._reg = [self._table.addr]
        self._ids = itertools.count(1)
        self.bufs, self.hosts, self.queues = {}, {}, {}
        self._cast = {}
        self._trace = []
        self.emu = emulate_ms_per_gflop
        with torch.cuda.device(device):
            self._ref = torch.cuda.Event(enable_timing=True)
            self._ref.record()
            self._ref.synchronize()
            self._ref_host = time.perf_counter()
            if self.emu:
                self._cycles_per_ms = self._calibrate_sleep()

    def _calibrate_sleep(self):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1_000_000)
        a.record()
        torch.cuda._sleep(10_000_000)
        b.record()
        b.synchronize()
        return 10_000_000 / a.elapsed_time(b)

    def caps(self):
        return Caps(name=self._name, level=1, scheduling="dynamic", max_queues=8, dma_engines=2,
                    memory="device_local", dtypes=(torch.bfloat16, torch.float16, torch.float32),
                    ops=("gemm", "lowrank_gelu", "axpb", "sleep"), peak_tflops=312.0, link_gbps=22.0)

    @property
    def full_duplex(self):
        return False

    @property
    def table(self):
        return self._table

    def alloc(self, shape, dtype):
        b = DevBuf(next(self._ids), tuple(shape), dtype)
        self.bufs[b.id] = torch.zeros(shape, dtype=dtype, device=f"cuda:{self.dev}")
        return b

    def free(self, buf):
        self.bufs.pop(buf.id, None)

    def host_alloc(self, shape, dtype):
        t = torch.zeros(shape, dtype=dtype).share_memory_()
        h = HostBuf(next(self._ids), t)
        if cm.register(h.addr, h.nbytes):
            self._reg.append(h.addr)
        self.hosts[h.id] = h
        return h

    def queue(self):
        q = Queue(next(self._ids))
        self.queues[q.id] = [torch.cuda.Stream(self.dev), self.signal(f"q{q.id}.completed"), 0]
        return q

    def signal(self, name=""):
        return Signal(self._table, self._table.new_index(), name)

    # -- ops ------------------------------------------------------------------------------------------------
    def _run(self, q, kind, tag, wait, done, body):
        st = self.queues[q.id]
        s = st[0]
        with torch.cuda.device(self.dev):
            for sig, v in wait:
                cm.wait_geq(s, sig.addr, int(v))
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(s)
            body(s)
            e1.record(s)
            if done is not None:
                cm.write64(s, done[0].addr, int(done[1]))
            st[2] += 1
            cm.write64(s, st[1].addr, st[2])
        self._trace.append((q.id, kind, tag, e0, e1))

    def copy_in(self, q, dst, dst_rows, src, src_rows, wait=(), done=None, tag=""):
        d0, d1 = rows_or_all(dst_rows, dst.shape)
        s0, s1 = rows_or_all(src_rows, src.shape)
        D = self.bufs[dst.id]
        rb = D[0].numel() * D.element_size()
        assert d1 - d0 == s1 - s0 and rb == src.row_bytes
        self._cast.pop(dst.id, None)  # a rewritten weight is re-cast at its next use (stream-ordered there)
        self._run(q, "copy_in", tag, wait, done,
                  lambda s: cm.memcpy(s, D.data_ptr() + d0 * rb, src.addr + s0 * rb, (d1 - d0) * rb))

    def copy_out(self, q, dst, dst_rows, src, src_rows, wait=(), done=None, tag=""):
        d0, d1 = rows_or_all(dst_rows, dst.shape)
        s0, s1 = rows_or_all(src_rows, src.shape)
        S = self.bufs[src.id]
        rb = S[0].numel() * S.element_size()
        assert d1 - d0 == s1 - s0 and rb == dst.row_bytes
        self._run(q, "copy_out", tag, wait, done,
                  lambda s: cm.memcpy(s, dst.addr + d0 * rb, S.data_ptr() + s0 * rb, (d1 - d0) * rb))

    def _bf16(self, buf_id):
        t = self._cast.get(buf_id)
        if t is None:
            t = self.bufs[buf_id].to(torch.bfloat16)
            self._cast[buf_id] = t
        return t

    def _op(self, op, a):
        B = self.bufs
        if op == "sleep":
            torch.cuda._sleep(int(a["seconds"] * 1e3 * getattr(self, "_cycles_per_ms", 1.4e6)))
            return
        r0, r1 = a["rows"]
        if self.emu and op in ("gemm", "lowrank_gelu"):
            if op == "gemm":
                k, n = B[a["w"]].shape
                gflop = 2 * (r1 - r0) * k * n / 1e9
            else:
                h, r = B[a["u"]].shape
                gflop = 4 * (r1 - r0) * h * r / 1e9
            torch.cuda._sleep(int(gflop * self.emu * self._cycles_per_ms))
            return
        x, y = B[a["x"]], B[a["y"]]
        if op == "gemm":
            y[r0:r1].copy_(x[r0:r1] @ self._bf16(a["w"]))
        elif op == "lowrank_gelu":
            h = torch.nn.functional.gelu(x[r0:r1] @ self._bf16(a["u"]))
            y[r0:r1].copy_(h @ self._bf16(a["v"]))
        elif op == "axpb":
            y[r0:r1].copy_(x[r0:r1].float() * a["a"] + a["b"])
        else:
            raise ValueError(op)

    def launch(self, q, op, args, wait=(), done=None, tag=""):
        a = {k: (v.id if isinstance(v, DevBuf) else v) for k, v in args.items()}
        if "rows" in a and a["rows"] is None:
            a["rows"] = (0, args["y"].shape[0])

        def body(s):
            with torch.cuda.stream(s):
                self._op(op, a)
        self._run(q, "launch", tag, wait, done, body)

    def prepare(self, op, args):
        """Run `op` once, synchronously, before any stream parks on a counter (loads cuBLAS / torch kernels)."""
        a = {k: (v.id if isinstance(v, DevBuf) else v) for k, v in args.items()}
        with torch.cuda.device(self.dev):
            self._op(op, a)
            torch.cuda.synchronize(self.dev)

    def sync(self, timeout=60.0):
        for s, cnt, n in self.queues.values():
            cnt.wait(n, timeout)

    def trace(self, clear=True):
        out = []
        keep = []
        for qid, kind, tag, e0, e1 in self._trace:
            if not e1.query():
                keep.append((qid, kind, tag, e0, e1))
                continue
            t0 = self._ref_host + self._ref.elapsed_time(e0) / 1e3
            t1 = self._ref_host + self._ref.elapsed_time(e1) / 1e3
            out.append(dict(queue=qid, kind=kind, tag=tag, t_ready=t0, t_start=t0, t_end=t1))
        self._trace = keep if clear else self._trace
        return out

    def close(self):
        torch.cuda.synchronize(self.dev)
        for a in self._reg:
            cm.unregister(a)
        self._reg = []
