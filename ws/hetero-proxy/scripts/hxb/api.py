"""hxb backend interface (spec: ws/hetero-proxy/reports/20261009_i1_backend_interface.md).

Every accelerator plugs in by subclassing Backend. A backend never sees CUDA, NVLink, NCCL or Flux: it moves data
between its own memory and host memory (HostBuf) and runs compute units, ordered by monotonic counters (Signal).
The Bridge (bridge.py) moves data between the NVIDIA GPU and the same host memory with the GPU's copy engines and
reads / writes the same counters with CUDA stream memory operations, so neither side needs the other's runtime.

Semantics every backend must keep (spec section 4):
  1. in order    ops on one Queue start in submission order; op k starts after op k-1 completed AND every
                 (signal, v) in its wait list has value >= v
  2. release     `done=(signal, v)` is stored only after all of the op's memory writes are visible
  3. monotonic   counters only grow; observers test >= ; generations: chunk i of pass g uses g * n + i + 1
  4. ownership   a copy_in source / copy_out destination is not touched until its done value is reached
  5. no deadlock a backend may not hold the resources a pending copy needs while it waits
Signals and HostBufs live in host memory shared with the device runtime; the Bridge registers them with CUDA
(cudaHostRegister, portable + mapped) so GPU streams can wait on / write them.
"""
import abc
import dataclasses
import itertools
import time

import torch

ERR = 0  # signal-table slot 0: non-zero once any party aborted (waiters raise instead of hanging)
FIRST_USER_SLOT = 8


@dataclasses.dataclass(frozen=True)
class Caps:
    name: str
    level: int            # 0 op-level, 1 chunk-level through host staging, 2 direct P2P with the GPU (not in prototype)
    scheduling: str       # "dynamic": several queues run concurrently; "static": one queue, submission order only
    max_queues: int       # 1 for a static device
    dma_engines: int      # 0 = copies use the compute resources (no copy / compute overlap on the device)
    memory: str           # "device_local" (own memory, explicit copies) or "host_shared"
    dtypes: tuple
    ops: tuple            # op names accepted by launch()
    peak_tflops: float    # estimate (fp32 for the CPU backend), for planning only
    link_gbps: float      # estimate of the device's own link to host memory, for planning only
    signal_memory: str = "host"   # where counters live; "host" = GPU can reach them without P2P


@dataclasses.dataclass(frozen=True)
class DevBuf:
    id: int
    shape: tuple
    dtype: torch.dtype

    @property
    def row_bytes(self):
        n = 1
        for d in self.shape[1:]:
            n *= d
        return n * torch.tensor([], dtype=self.dtype).element_size()


@dataclasses.dataclass(frozen=True)
class Queue:
    id: int


class HostBuf:
    """Host memory reachable by both the device runtime and (after Bridge.register) the GPU copy engines."""

    def __init__(self, id, tensor):
        self.id, self.tensor = id, tensor
        self.shape, self.dtype = tuple(tensor.shape), tensor.dtype
        self.addr = tensor.data_ptr()
        self.nbytes = tensor.numel() * tensor.element_size()
        self.row_bytes = self.nbytes // max(self.shape[0], 1)


class SignalTable:
    """int64 counters in shared host memory. Slot 0 is the abort flag; slots 1..7 reserved."""

    def __init__(self, n=4096, tensor=None):
        self.tensor = tensor if tensor is not None else torch.zeros(n, dtype=torch.int64).share_memory_()
        self.np = self.tensor.numpy()
        self.addr = self.tensor.data_ptr()
        self.nbytes = self.tensor.numel() * 8
        self._next = itertools.count(FIRST_USER_SLOT)

    def new_index(self):
        i = next(self._next)
        if i >= len(self.np):
            raise RuntimeError("signal table full")
        return i

    def abort(self):
        """Release every waiter (GPU streams included) so a failed run can exit; results become invalid."""
        self.np[ERR] = 1
        self.np[1:] = 1 << 62


class Signal:
    def __init__(self, table, index, name=""):
        self.table, self.index, self.name = table, index, name
        self.addr = table.addr + 8 * index

    def value(self):
        return int(self.table.np[self.index])

    def set(self, v):
        """Host-side store (tests, host-driven producers). Monotonic: never lowers the value."""
        if v > self.table.np[self.index]:
            self.table.np[self.index] = v

    def wait(self, v, timeout=60.0):
        a, i = self.table.np, self.index
        if a[i] >= v:
            return
        t_end = time.perf_counter() + timeout
        n = 0
        while a[i] < v:
            if a[ERR]:
                raise RuntimeError(f"aborted while waiting for {self.name} >= {v}")
            n += 1
            if n > 2000:
                time.sleep(2e-5)
                if time.perf_counter() > t_end:
                    raise TimeoutError(f"{self.name} = {a[i]} < {v} after {timeout} s")

    def __repr__(self):
        return f"Signal({self.name}#{self.index}={self.value()})"


def rows_or_all(rows, shape):
    return (0, shape[0]) if rows is None else (int(rows[0]), int(rows[1]))


class Backend(abc.ABC):
    """One accelerator. All copy / launch calls are asynchronous; they return immediately."""

    @abc.abstractmethod
    def caps(self) -> Caps: ...

    @abc.abstractmethod
    def alloc(self, shape, dtype) -> DevBuf: ...

    @abc.abstractmethod
    def free(self, buf: DevBuf): ...

    @abc.abstractmethod
    def host_alloc(self, shape, dtype) -> HostBuf: ...

    @abc.abstractmethod
    def queue(self) -> Queue: ...

    @abc.abstractmethod
    def signal(self, name="") -> Signal: ...

    @abc.abstractmethod
    def copy_in(self, q, dst: DevBuf, dst_rows, src: HostBuf, src_rows, wait=(), done=None): ...

    @abc.abstractmethod
    def copy_out(self, q, dst: HostBuf, dst_rows, src: DevBuf, src_rows, wait=(), done=None): ...

    @abc.abstractmethod
    def launch(self, q, op, args, wait=(), done=None):
        """args: dict of DevBufs, row ranges and scalars; the op's own contract is in the backend's docs."""

    @abc.abstractmethod
    def sync(self, timeout=60.0): ...

    def flush(self, timeout=60.0):
        """Return once every submitted command has reached the device (submission, not execution). Backends whose
        submission is synchronous need not override it."""

    def trace(self):
        """List of dicts (queue, kind, tag, t_ready, t_start, t_end) in time.perf_counter() seconds."""
        return []

    def close(self):
        pass

    @property
    def table(self) -> SignalTable:
        raise NotImplementedError
