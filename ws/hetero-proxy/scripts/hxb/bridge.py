"""Bridge: moves row chunks between NVIDIA GPUs and an hxb backend through host staging (spec section 6).

GPU side, everything is enqueued ahead on CUDA streams (no host thread on the critical path, as in FlexLink):
  Outbound (GPU -> device), chunk j into slot j % S:
     d2h stream: wait producer event -> wait host counter consumed >= j - S + 1 (slot free) -> D2H copy
                 -> write host counter staged = j + 1
     device:     copy_in(slot -> device rows), waits staged >= j + 1, done consumed = j + 1
  Inbound (device -> GPU):
     device:     copy_out(device rows -> slot), waits caller's counters and free >= j - S + 1, done staged = j + 1
     h2d stream: wait staged >= j + 1 -> H2D copy -> write host counter free = j + 1
                 -> write the consumer's GPU flag (stream memory op, never a kernel: a kernel could not run while
                    the consumer's waiting tiles occupy the SMs)
Counters are monotonic over the channel's lifetime (j never resets), so passes need no reset.
Phase 0 rule (PHASE0_FINDINGS 2.4): one GPU sending and receiving at the same time drops its H2D to ~6.5 GB/s,
so Outbound and Inbound may sit on different GPUs (the activation is replicated in a TP domain).
Do not run with CUDA_DEVICE_MAX_CONNECTIONS=1 (launch.sh sets it): a copy stream would queue behind the spinning
consumer kernel. Load every kernel before a stream waits on a counter (CUDA_MODULE_LOADING=EAGER plus warm-up):
lazy module loading synchronises the context and deadlocks against a parked stream (found 2026-10-09).
"""
import os

import torch

from . import cudamem as cm


def _check_env():
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        raise RuntimeError("CUDA_DEVICE_MAX_CONNECTIONS=1 serialises the copy streams behind spinning kernels")
    if os.environ.get("CUDA_MODULE_LOADING", "") != "EAGER":
        print("[hxb] warning: CUDA_MODULE_LOADING is not EAGER; every kernel must be warmed up before a stream is "
              "parked on a WaitValue (module loading synchronises the context)", flush=True)


class Bridge:
    def __init__(self, backend):
        _check_env()
        self.be = backend
        self._reg = []
        self._register(backend.table.addr, backend.table.nbytes)

    def _register(self, addr, nbytes):
        if cm.register(addr, nbytes):
            self._reg.append(addr)

    def host_alloc(self, shape, dtype):
        h = self.be.host_alloc(shape, dtype)
        self._register(h.addr, h.nbytes)
        return h

    def outbound(self, device, chunk_rows, cols, dtype=torch.bfloat16, slots=4, timing=False):
        return Outbound(self, device, chunk_rows, cols, dtype, slots, timing)

    def inbound(self, device, chunk_rows, cols, dtype=torch.bfloat16, slots=4, timing=False):
        return Inbound(self, device, chunk_rows, cols, dtype, slots, timing)

    def upload(self, q, t):
        """Synchronous host -> device copy (weights). t: CPU or GPU tensor."""
        be = self.be
        h = be.host_alloc(tuple(t.shape), t.dtype)
        h.tensor.copy_(t.cpu())
        d = be.alloc(tuple(t.shape), t.dtype)
        s = be.signal("upload")
        be.copy_in(q, d, None, h, None, done=(s, 1))
        s.wait(1)
        return d

    def download(self, q, d):
        be = self.be
        h = be.host_alloc(d.shape, d.dtype)
        s = be.signal("download")
        be.copy_out(q, h, None, d, None, done=(s, 1))
        s.wait(1)
        return h.tensor.clone()

    def close(self):
        for a in self._reg:
            cm.unregister(a)
        self._reg = []


class _Channel:
    def __init__(self, bridge, device, chunk_rows, cols, dtype, slots, timing, prefix):
        self.br, self.be, self.device = bridge, bridge.be, device
        self.slots = [bridge.host_alloc((chunk_rows, cols), dtype) for _ in range(slots)]
        self.row_bytes = self.slots[0].row_bytes
        self.stream = torch.cuda.Stream(device)
        self.j, self.q = 0, None
        self.timing = timing
        self.events = []  # (j, start, end) CUDA events of the GPU-side copy, when timing
        self.prefix = prefix

    def _q(self, q):
        if self.q is None:
            self.q = q
        assert q == self.q, "all device copies of one channel must use one queue (counters complete in order)"

    def _ev(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record(self.stream)
        return e


class Outbound(_Channel):
    """send() = post_device() + post_gpu() for the same chunk index j. The halves can also be called separately:
    a caller may post the device side of many future chunks ahead (no command latency on the critical path) and
    the GPU side when the data is produced. Both halves count j independently, in the same order."""

    def __init__(self, bridge, device, chunk_rows, cols, dtype, slots, timing):
        super().__init__(bridge, device, chunk_rows, cols, dtype, slots, timing, "ob")
        self.staged, self.consumed = self.be.signal("ob.staged"), self.be.signal("ob.consumed")
        self.jd = self.jg = 0

    def post_device(self, dst, dst_rows, n, q, tag=""):
        """Device half of the next chunk: copy_in of n rows from its slot into dst[dst_rows] once staged.
        Returns (signal, value) reached when the rows are in device memory."""
        self._q(q)
        j = self.jd
        self.jd += 1
        slot = self.slots[j % len(self.slots)]
        self.be.copy_in(q, dst, dst_rows, slot, (0, n), wait=[(self.staged, j + 1)], done=(self.consumed, j + 1),
                        tag=tag or f"in{j}")
        return self.consumed, j + 1

    def post_gpu(self, src, r0, r1, ready):
        """GPU half of the next chunk: D2H src[r0:r1] into its slot after `ready` and once the slot is free."""
        S, j = len(self.slots), self.jg
        self.jg += 1
        self.j = self.jg
        slot, n = self.slots[j % S], r1 - r0
        assert src.is_contiguous() and src.device.index == self.device and n <= slot.shape[0]
        with torch.cuda.device(self.device):
            s = self.stream
            if ready is not None:
                s.wait_event(ready)
            if j >= S:
                cm.wait_geq(s, self.consumed.addr, j - S + 1)
            e0 = self._ev() if self.timing else None
            cm.memcpy(s, slot.addr, src.data_ptr() + r0 * self.row_bytes, n * self.row_bytes)
            if self.timing:
                self.events.append((j, e0, self._ev()))
            cm.write64(s, self.staged.addr, j + 1)

    def send(self, src, r0, r1, ready, dst, dst_rows, q, tag=""):
        """Send src[r0:r1] (GPU tensor, contiguous rows) once `ready` (CUDA event) has fired, into dst[dst_rows].
        Returns (signal, value): reached when the rows are in device memory."""
        assert self.jd == self.jg, "send() needs both halves in step; use post_device / post_gpu separately"
        out = self.post_device(dst, dst_rows, r1 - r0, q, tag)
        self.post_gpu(src, r0, r1, ready)
        return out


class Inbound(_Channel):
    """recv() = post_device() + post_gpu(); the halves can be called separately as for Outbound."""

    def __init__(self, bridge, device, chunk_rows, cols, dtype, slots, timing):
        super().__init__(bridge, device, chunk_rows, cols, dtype, slots, timing, "ib")
        self.staged, self.free = self.be.signal("ib.staged"), self.be.signal("ib.free")
        self.jd = self.jg = 0

    def post_device(self, src, src_rows, n, q, wait, tag=""):
        """Device half: copy_out of device src[src_rows] (n rows) into the next slot after `wait` and once the slot
        is free. Returns (signal, value) reached when the rows are in the slot."""
        self._q(q)
        S, j = len(self.slots), self.jd
        self.jd += 1
        slot = self.slots[j % S]
        w = list(wait) + ([(self.free, j - S + 1)] if j >= S else [])
        self.be.copy_out(q, slot, (0, n), src, src_rows, wait=w, done=(self.staged, j + 1), tag=tag or f"out{j}")
        return self.staged, j + 1

    def post_gpu(self, dst, r0, r1, flag=None):
        """GPU half: H2D the next slot into dst[r0:r1] once staged; then free the slot and, if flag = (gpu_address,
        value), store the int32 value there (stream memory op). Returns (signal, value) of the slot release."""
        S, j = len(self.slots), self.jg
        self.jg += 1
        self.j = self.jg
        slot, n = self.slots[j % S], r1 - r0
        assert dst.is_contiguous() and dst.device.index == self.device and n <= slot.shape[0]
        with torch.cuda.device(self.device):
            s = self.stream
            cm.wait_geq(s, self.staged.addr, j + 1)
            e0 = self._ev() if self.timing else None
            cm.memcpy(s, dst.data_ptr() + r0 * self.row_bytes, slot.addr, n * self.row_bytes)
            if self.timing:
                self.events.append((j, e0, self._ev()))
            cm.write64(s, self.free.addr, j + 1)
            if flag is not None:
                cm.write32(s, flag[0], flag[1])
        return self.free, j + 1

    def recv(self, src, src_rows, q, wait, dst, r0, r1, flag=None, tag=""):
        """Bring device src[src_rows] into dst[r0:r1] (GPU tensor) after every (signal, v) in `wait`;
        then, if flag = (gpu_address, value), store the int32 value there (stream memory op)."""
        assert self.jd == self.jg, "recv() needs both halves in step; use post_device / post_gpu separately"
        self.post_device(src, src_rows, r1 - r0, q, wait, tag)
        return self.post_gpu(dst, r0, r1, flag)
