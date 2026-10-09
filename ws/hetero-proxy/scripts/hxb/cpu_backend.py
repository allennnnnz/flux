"""CPU backend: the host CPU's second socket plays a non-NVIDIA accelerator.

The "device" is a separate process pinned to one NUMA node (default node 1; GPU0-3 hang off node 0):
  - device memory   tensors allocated and first-touched inside the device process (its NUMA node); the host
                    process cannot address them, so every transfer is an explicit copy_in / copy_out
  - DMA engine      `dma_threads` persistent C threads (csrc/hxbdma.c) spinning on their own physical cores,
                    one copy at a time split across them (one engine shared by both directions); dma_threads=0:
                    no engine, the queue thread copies by itself
  - compute         fp32 torch / MKL on the remaining physical cores of the node minus 2 control cores (no
                    AVX512_BF16 / AMX on the Xeon 8358, so bf16 is converted to fp32 inside each op); hyper-thread
                    siblings of compute cores stay idle
  - queues          one thread each, executing commands in order; waits are a C spin with the GIL released. A queue
                    starts on the 2 control cores (and their siblings) and moves onto the compute cores for good at
                    its first compute op, so copy queues never take a core from the compute team
  - commands        sent through a multiprocessing queue, like a host runtime writing a device command ring
Knobs for NPU-like simulation (I5): scheduling="static" (one queue, submission order), link_gbps (throttle each
copy to the device's own link speed), emulate_tflops (timing-only compute: sleep flops / rate, output not written),
op_overhead_us (fixed cost per op, e.g. a heavy launch path).
omp_spincount: GOMP_SPINCOUNT of the device process (default 1e6, ~ms: the compute team stays awake between the
ops of a chunked pass; libgomp's default let it fall asleep between chunks, dry run 2026-10-09).

Ops (launch(q, op, args)); row ranges are (r0, r1) of the first dimension:
  gemm          args x, w, y, rows:       y[rows] = x[rows] @ w            (w float32 on the device)
  lowrank_gelu  args x, u, v, y, rows:    y[rows] = gelu(x[rows] @ u) @ v  (u, v float32)
  axpb          args x, y, rows, a, b:    y[rows] = a * x[rows] + b
  sleep         args seconds
"""
import ctypes
import os
import queue as pyqueue
import threading
import time
import traceback

import torch

from . import native
from .api import ERR, Backend, Caps, DevBuf, HostBuf, Queue, Signal, SignalTable, rows_or_all


def node_cpus(node):
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        out = []
        for part in f.read().strip().split(","):
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def siblings(c):
    with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
        out = []
        for part in f.read().strip().split(","):
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def physical_cpus(cpus):
    """Drop hyper-thread siblings: keep the first cpu of each core."""
    seen, keep = set(), []
    for c in cpus:
        with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
            sib = f.read().strip()
        if sib not in seen:
            seen.add(sib)
            keep.append(c)
    return keep


def _timerslack_1us():
    try:
        ctypes.CDLL(None).prctl(29, 1000, 0, 0, 0)  # PR_SET_TIMERSLACK: sleep(10 us) ~ 12 us instead of ~60 us
    except Exception:
        pass


def _flops(op, a, bufs):
    r0, r1 = a["rows"]
    if op == "gemm":
        k, n = bufs[a["w"]].shape
        return 2 * (r1 - r0) * k * n
    if op == "lowrank_gelu":
        h, r = bufs[a["u"]].shape
        return 4 * (r1 - r0) * h * r
    return 0


class _Device:
    """Runs inside the device process."""

    def __init__(self, cfg, table, respq):
        self.cfg, self.table, self.respq = cfg, table, respq
        self.np = table.np
        self.bufs, self.hosts, self.workers, self.trace = {}, {}, {}, []
        self.scratch = {}
        self.link_bps = cfg["link_gbps"] * 1e9 if cfg["link_gbps"] else None
        self.emu = cfg["emulate_tflops"] * 1e12 if cfg["emulate_tflops"] else None
        self.overhead = cfg["op_overhead_us"] * 1e-6
        self.lib = native.lib()
        self.dma_lock = threading.Lock()
        n = cfg["dma_threads"]
        if n:
            cpus = (ctypes.c_int * n)(*cfg["dma_cores"])
            assert self.lib.hxb_dma_start(n, cpus) == 0
        self.err_addr = table.addr + 8 * ERR

    # -- waits / copies -------------------------------------------------------------------------------------
    def wait(self, idx, v):
        if self.np[idx] >= v:
            return
        r = self.lib.hxb_wait_geq(self.table.addr + 8 * idx, v, self.err_addr, 3600.0)
        if r:
            raise RuntimeError("aborted" if r == 1 else f"timeout waiting for slot {idx} >= {v}")

    def dma(self, dst, src, nbytes):
        t0 = time.perf_counter()
        if self.cfg["dma_threads"]:
            with self.dma_lock:
                self.lib.hxb_dma_copy(dst, src, nbytes)
        else:
            ctypes.memmove(dst, src, nbytes)
        if self.link_bps:
            rest = t0 + nbytes / self.link_bps - time.perf_counter()
            if rest > 0:
                time.sleep(rest)

    def _scr(self, q, name, shape, dtype):
        """Scratch tensor of `shape`, a view into a per-(queue, name, dtype) buffer that only ever grows. Allocating
        per shape cost ~20 ms of page faults + kernel zeroing whenever passes with different chunk sizes alternate
        (I3 dry runs, 2026-10-09: level-0 compute 28.7 ms instead of 7.7)."""
        need = 1
        for d in shape:
            need *= d
        key = (q, name, dtype)
        t = self.scratch.get(key)
        if t is None or t.numel() < need:
            t = torch.empty(need, dtype=dtype)
            t.zero_()  # first touch now, on this queue's cores
            self.scratch[key] = t
        return t[:need].view(shape)

    # -- ops ------------------------------------------------------------------------------------------------
    def run(self, q, kind, p):
        if self.overhead:
            time.sleep(self.overhead)
        if kind == "copy_in":
            dst, d0, d1, src, s0, s1 = p
            D, S = self.bufs[dst], self.hosts[src]
            rb = D.element_size() * D[0].numel()
            assert (d1 - d0) == (s1 - s0) and rb == S.element_size() * S[0].numel(), "copy_in shape mismatch"
            self.dma(D.data_ptr() + d0 * rb, S.data_ptr() + s0 * rb, (d1 - d0) * rb)
        elif kind == "copy_out":
            dst, d0, d1, src, s0, s1 = p
            D, S = self.hosts[dst], self.bufs[src]
            rb = S.element_size() * S[0].numel()
            assert (d1 - d0) == (s1 - s0) and rb == D.element_size() * D[0].numel(), "copy_out shape mismatch"
            self.dma(D.data_ptr() + d0 * rb, S.data_ptr() + s0 * rb, (d1 - d0) * rb)
        elif kind == "launch":
            op, a = p
            if op == "sleep":
                time.sleep(a["seconds"])
                return
            if self.emu and op in ("gemm", "lowrank_gelu"):
                time.sleep(_flops(op, a, self.bufs) / self.emu)
                return
            r0, r1 = a["rows"]
            if op == "gemm":
                x, w, y = self.bufs[a["x"]], self.bufs[a["w"]], self.bufs[a["y"]]
                xf = self._scr(q, "xf", (r1 - r0, x.shape[1]), torch.float32)
                yf = self._scr(q, "yf", (r1 - r0, w.shape[1]), torch.float32)
                xf.copy_(x[r0:r1])
                torch.matmul(xf, w, out=yf)
                y[r0:r1].copy_(yf)
            elif op == "lowrank_gelu":
                x, u, v, y = self.bufs[a["x"]], self.bufs[a["u"]], self.bufs[a["v"]], self.bufs[a["y"]]
                xf = self._scr(q, "xf", (r1 - r0, x.shape[1]), torch.float32)
                hf = self._scr(q, "hf", (r1 - r0, u.shape[1]), torch.float32)
                yf = self._scr(q, "yf", (r1 - r0, v.shape[1]), torch.float32)
                xf.copy_(x[r0:r1])
                torch.matmul(xf, u, out=hf)
                torch.ops.aten.gelu_(hf)
                torch.matmul(hf, v, out=yf)
                y[r0:r1].copy_(yf)
            elif op == "axpb":
                x, y = self.bufs[a["x"]], self.bufs[a["y"]]
                y[r0:r1].copy_(x[r0:r1].float() * a["a"] + a["b"])
            else:
                raise ValueError(f"unknown op {op}")
        else:
            raise ValueError(kind)

    def worker(self, qid, inbox, count_idx):
        control, compute = self.cfg["control_cores"], self.cfg["compute_cores"]
        os.sched_setaffinity(0, control)
        _timerslack_1us()
        torch.set_num_threads(len(compute))
        on_compute = False
        try:
            while True:
                item = inbox.get()
                if item is None:
                    return
                seq, kind, p, wait, done, tag = item
                is_compute = kind == "launch" and not (self.emu and p[0] in ("gemm", "lowrank_gelu")) \
                    and p[0] not in ("sleep",)
                t_pick = time.perf_counter()
                for idx, v in wait:
                    self.wait(idx, v)
                if is_compute and not on_compute:
                    # first compute op: this queue becomes a compute queue and stays on the compute cores (its
                    # compute team, created at the first parallel region, inherits the mask); queues that only
                    # copy never leave the control cores. Moving per op cost ~50 us of migration (2026-10-09).
                    os.sched_setaffinity(0, compute)
                    on_compute = True
                t0 = time.perf_counter()
                self.run(qid, kind, p)
                t1 = time.perf_counter()
                if done is not None and done[1] > self.np[done[0]]:
                    self.np[done[0]] = done[1]
                self.np[count_idx] = seq
                self.trace.append((qid, kind, tag, t_pick, t0, t1))
        except Exception:
            self.respq.put(("error", f"queue {qid}: {traceback.format_exc()}"))
            self.table.abort()


def device_main(cfg, table_tensor, cmdq, respq):
    os.sched_setaffinity(0, cfg["control_cores"])
    _timerslack_1us()
    torch.set_num_threads(len(cfg["compute_cores"]))
    table = SignalTable(tensor=table_tensor)
    dev = _Device(cfg, table, respq)
    respq.put(("ready", os.getpid()))
    threads = []
    while True:
        cmd = cmdq.get()
        k = cmd[0]
        try:
            if k == "op":
                dev.workers[cmd[1]].put(cmd[2:])
            elif k == "alloc":
                _, i, shape, dtype = cmd
                dev.bufs[i] = torch.zeros(shape, dtype=dtype)  # first touch inside the device process
            elif k == "free":
                dev.bufs.pop(cmd[1], None)
            elif k == "host":
                dev.hosts[cmd[1]] = cmd[2]
            elif k == "queue":
                _, qid, count_idx = cmd
                inbox = pyqueue.SimpleQueue()
                dev.workers[qid] = inbox
                t = threading.Thread(target=dev.worker, args=(qid, inbox, count_idx), daemon=True)
                t.start()
                threads.append(t)
            elif k == "fence":  # every earlier command has been routed to its queue
                _, idx, v = cmd
                if v > dev.np[idx]:
                    dev.np[idx] = v
            elif k == "trace":
                respq.put(("trace", list(dev.trace)))
                if cmd[1]:
                    dev.trace.clear()
            elif k == "stop":
                for inbox in dev.workers.values():
                    inbox.put(None)
                for t in threads:
                    t.join(timeout=5)
                if cfg["dma_threads"]:
                    dev.lib.hxb_dma_stop()
                respq.put(("stopped",))
                return
        except Exception:
            respq.put(("error", f"dispatcher: {traceback.format_exc()}"))
            table.abort()


class CpuBackend(Backend):
    def __init__(self, numa=1, dma_threads=4, compute_threads=None, scheduling="dynamic", link_gbps=None,
                 emulate_tflops=None, op_overhead_us=0.0, omp_spincount=1_000_000, dma_cores=None, name=None):
        cpus = node_cpus(numa)
        phys = physical_cpus(cpus)
        # Layout (2026-10-09): DMA on its own physical cores; control threads (dispatcher, queue threads: waiting,
        # copy dispatch) on 2 dedicated physical cores plus their hyper-thread siblings; compute on the remaining
        # physical cores with their siblings left idle. The first layout put control threads on the siblings of the
        # compute cores: a spinning waiter halves its sibling's throughput and the statically scheduled compute team
        # waits for its slowest thread -> compute 3-4x slower inside the pipeline than alone (I3 dry run).
        if dma_cores is not None:  # explicit DMA cores (e.g. on the staging side); then all node cores are free
            dma = list(dma_cores)[:dma_threads]
            assert len(dma) == dma_threads
            rest = phys
        else:
            dma = phys[:dma_threads]
            rest = phys[len(dma):]
        ctl_phys = rest[:2]
        control = sorted(set(c for p_ in ctl_phys for c in siblings(p_)) & set(cpus))
        compute = rest[2:]
        if compute_threads:
            compute = compute[:compute_threads]
        self.cfg = dict(dma_threads=dma_threads, dma_cores=dma, compute_cores=compute, control_cores=control,
                        link_gbps=link_gbps, emulate_tflops=emulate_tflops, op_overhead_us=op_overhead_us)
        self.scheduling = scheduling
        self._name = name or f"cpu-numa{numa}"
        self._caps = Caps(
            name=self._name, level=1, scheduling=scheduling, max_queues=1 if scheduling == "static" else 8,
            dma_engines=1 if dma_threads else 0, memory="device_local",
            dtypes=(torch.float32, torch.bfloat16, torch.float16), ops=("gemm", "lowrank_gelu", "axpb", "sleep"),
            peak_tflops=emulate_tflops or len(compute) * 2.6e9 * 32 / 1e12,  # cores x 2.6 GHz x 32 fp32 flop/cycle
            link_gbps=link_gbps or 0.0)
        self._table = SignalTable()
        ctx = torch.multiprocessing.get_context("spawn")
        self.cmdq, self.respq = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=device_main, args=(self.cfg, self._table.tensor, self.cmdq, self.respq),
                                daemon=True)
        saved = os.environ.get("GOMP_SPINCOUNT")
        if omp_spincount is not None:  # inherited by the spawned device process only (read when libgomp loads)
            os.environ["GOMP_SPINCOUNT"] = str(omp_spincount)
        try:
            self.proc.start()
        finally:
            if omp_spincount is not None:
                if saved is None:
                    os.environ.pop("GOMP_SPINCOUNT")
                else:
                    os.environ["GOMP_SPINCOUNT"] = saved
        msg = self.respq.get(timeout=120)
        assert msg[0] == "ready", msg
        self.device_pid = msg[1]
        self._ids = iter(range(1, 1 << 40))
        self._queues, self._hosts, self._closed = {}, {}, False
        self._static_q = None

    # -- interface ------------------------------------------------------------------------------------------
    def caps(self):
        return self._caps

    @property
    def table(self):
        return self._table

    def alloc(self, shape, dtype):
        b = DevBuf(next(self._ids), tuple(shape), dtype)
        self.cmdq.put(("alloc", b.id, b.shape, dtype))
        return b

    def free(self, buf):
        self.cmdq.put(("free", buf.id))

    def host_alloc(self, shape, dtype):
        t = torch.zeros(shape, dtype=dtype).share_memory_()  # first touch here (host process, its NUMA node)
        h = HostBuf(next(self._ids), t)
        self._hosts[h.id] = h
        self.cmdq.put(("host", h.id, t))
        return h

    def queue(self):
        if self.scheduling == "static" and self._static_q is not None:
            return self._static_q
        q = Queue(next(self._ids))
        cnt = self.signal(f"q{q.id}.completed")
        self._queues[q.id] = [cnt, 0]
        self.cmdq.put(("queue", q.id, cnt.index))
        if self.scheduling == "static":
            self._static_q = q
        return q

    def signal(self, name=""):
        return Signal(self._table, self._table.new_index(), name)

    def _submit(self, q, kind, payload, wait, done, tag):
        st = self._queues[q.id]
        st[1] += 1
        w = [(s.index, int(v)) for s, v in wait]
        d = None if done is None else (done[0].index, int(done[1]))
        self.cmdq.put(("op", q.id, st[1], kind, payload, w, d, tag))

    def copy_in(self, q, dst, dst_rows, src, src_rows, wait=(), done=None, tag=""):
        d0, d1 = rows_or_all(dst_rows, dst.shape)
        s0, s1 = rows_or_all(src_rows, src.shape)
        self._submit(q, "copy_in", (dst.id, d0, d1, src.id, s0, s1), wait, done, tag)

    def copy_out(self, q, dst, dst_rows, src, src_rows, wait=(), done=None, tag=""):
        d0, d1 = rows_or_all(dst_rows, dst.shape)
        s0, s1 = rows_or_all(src_rows, src.shape)
        self._submit(q, "copy_out", (dst.id, d0, d1, src.id, s0, s1), wait, done, tag)

    def launch(self, q, op, args, wait=(), done=None, tag=""):
        a = {k: (v.id if isinstance(v, DevBuf) else v) for k, v in args.items()}
        if "rows" in a and a["rows"] is None:
            a["rows"] = (0, args["y"].shape[0])
        self._submit(q, "launch", (op, a), wait, done, tag)

    def flush(self, timeout=60.0):
        """Return once the device has received every command submitted so far (not executed: received)."""
        if not hasattr(self, "_fence"):
            self._fence, self._fence_v = self.signal("fence"), 0
        self._fence_v += 1
        self.cmdq.put(("fence", self._fence.index, self._fence_v))
        self._fence.wait(self._fence_v, timeout)

    def sync(self, timeout=60.0):
        for cnt, n in self._queues.values():
            try:
                cnt.wait(n, timeout)
            except Exception:
                self._raise_device_error()
                raise

    def _raise_device_error(self):
        try:
            while True:
                m = self.respq.get(timeout=1)
                if m[0] == "error":
                    raise RuntimeError(f"device error: {m[1]}")
        except pyqueue.Empty:
            pass

    def trace(self, clear=True):
        self.cmdq.put(("trace", clear))
        while True:
            m = self.respq.get(timeout=60)
            if m[0] == "trace":
                return [dict(queue=q, kind=k, tag=t, t_ready=a, t_start=b, t_end=c) for q, k, t, a, b, c in m[1]]
            if m[0] == "error":
                raise RuntimeError(m[1])

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.cmdq.put(("stop",))
            self.proc.join(timeout=15)
        finally:
            if self.proc.is_alive():
                self.proc.kill()
