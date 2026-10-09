"""CUDA driver calls the Bridge needs, through ctypes (no cuda-python on this machine).

Stream memory operations are the mechanism Flux uses for its per-tile flags (CUStreamWriteValue in
src/coll/ths_op/all_gather_op.cc); they execute in the stream's front end, not on SMs, so they never queue behind
a kernel that is spinning on a flag. Probed 2026-10-09 on css-host-158 (driver 615.71.09): the _v2 entry points
exist, 64-bit wait / write work on host memory registered portable + mapped, and a host -> GPU wait -> GPU write
-> host round trip takes ~11 us (median of 200).
"""
import ctypes

import torch

_cu = ctypes.CDLL("libcuda.so.1")
_V, _U64, _U32, _UI, _SZ = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint, ctypes.c_size_t


def _fn(name, argtypes):
    f = getattr(_cu, name)
    f.argtypes, f.restype = argtypes, ctypes.c_int
    return f


_write32 = _fn("cuStreamWriteValue32_v2", [_V, _U64, _U32, _UI])
_write64 = _fn("cuStreamWriteValue64_v2", [_V, _U64, _U64, _UI])
_wait32 = _fn("cuStreamWaitValue32_v2", [_V, _U64, _U32, _UI])
_wait64 = _fn("cuStreamWaitValue64_v2", [_V, _U64, _U64, _UI])
_memcpy = _fn("cuMemcpyAsync", [_U64, _U64, _SZ, _V])
WAIT_GEQ = 0x0          # CU_STREAM_WAIT_VALUE_GEQ: (int64)(*addr - v) >= 0
HOST_REGISTER_FLAGS = 3  # cudaHostRegisterPortable | cudaHostRegisterMapped
_ALREADY_REGISTERED = 712


def _check(r, what):
    if r != 0:
        raise RuntimeError(f"{what}: CUresult {r}")


def _s(stream):
    return _V(stream.cuda_stream)


def write64(stream, addr, v):
    """Store v at addr once all earlier work on `stream` is complete (default flags include a memory barrier)."""
    _check(_write64(_s(stream), addr, v, 0), "cuStreamWriteValue64")


def write32(stream, addr, v):
    _check(_write32(_s(stream), addr, v, 0), "cuStreamWriteValue32")


def wait_geq(stream, addr, v):
    """Later work on `stream` starts only once *addr >= v (addr may be registered host memory)."""
    _check(_wait64(_s(stream), addr, v, WAIT_GEQ), "cuStreamWaitValue64")


def memcpy(stream, dst, src, nbytes):
    """Unified-address async copy (D2H, H2D or D2D) on `stream`; host side must be pinned / registered."""
    _check(_memcpy(dst, src, nbytes, _s(stream)), "cuMemcpyAsync")


def register(addr, nbytes):
    """Pin + map host memory for every device (portable). Idempotent."""
    r = torch.cuda.cudart().cudaHostRegister(addr, nbytes, HOST_REGISTER_FLAGS)
    code = int(getattr(r, "value", r))
    if code not in (0, _ALREADY_REGISTERED):
        raise RuntimeError(f"cudaHostRegister failed: {r}")
    return code == 0


def unregister(addr):
    torch.cuda.cudart().cudaHostUnregister(addr)
