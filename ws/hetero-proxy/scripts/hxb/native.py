"""Loader for csrc/libhxbdma.so (built on first use with gcc). Functions run with the GIL released (ctypes)."""
import ctypes
import os
import subprocess

_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_SRC, _LIB = os.path.join(_HERE, "hxbdma.c"), os.path.join(_HERE, "libhxbdma.so")
_lib = None


def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
            subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-pthread", "-o", _LIB, _SRC], check=True)
        L = ctypes.CDLL(_LIB)
        L.hxb_dma_start.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        L.hxb_dma_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        L.hxb_dma_copy.restype = None
        L.hxb_dma_stop.restype = None
        L.hxb_wait_geq.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_double]
        L.hxb_store_max.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        L.hxb_store_max.restype = None
        _lib = L
    return _lib
