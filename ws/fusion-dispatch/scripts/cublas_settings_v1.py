################################################################################
# Does flux.testing.init_seed() (python/flux/testing/utils.py:49-62) change the
# cuBLAS off-path? It sets deterministic algorithms (warn_only),
# CUBLAS_WORKSPACE_CONFIG=:16:8 and disables bf16 reduced-precision reduction.
# Measures host launch time and GPU time (CUDA events) of torch.mm for the
# shapes the B/C arms run. Each setting runs in its own process because the
# cuBLAS workspace size is fixed at handle creation.
# Usage: python cublas_settings_v1.py <setting>   setting in
#        default | det | ws | nobf16red | all
################################################################################
import os
import statistics
import sys
import time

setting = sys.argv[1]
if setting in ("ws", "all"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch  # noqa: E402

if setting in ("det", "all"):
    torch.use_deterministic_algorithms(True, warn_only=True)
if setting in ("nobf16red", "all"):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

torch.cuda.set_device(0)
dt = torch.bfloat16
FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
shapes = [("G-FC1", 64, 6144, 12288), ("G-FC1", 4096, 6144, 12288), ("P0-4096", 1024, 512, 12288),
          ("L-QKV", 64, 1280, 8192), ("L-GU", 512, 7168, 8192)]
for name, M, n, K in shapes:
    a = torch.randn(M, K, device="cuda", dtype=dt)
    w = torch.randn(n, K, device="cuda", dtype=dt) * 0.01
    o = torch.empty(M, n, device="cuda", dtype=dt)
    fn = lambda: torch.mm(a, w.t(), out=o)  # noqa: E731
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    cpu = []
    for _ in range(30):
        torch.cuda._sleep(100_000_000)
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        cpu.append((time.perf_counter() - t0) / 50 * 1e6)
        torch.cuda.synchronize()
    gpu = []
    for _ in range(200):
        FLUSH.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(400_000)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        gpu.append(s.elapsed_time(e) * 1e3)
    print(f"{setting:<10} {name:<8} M={M:<5} n={n:<5} K={K:<6} cpu {statistics.median(cpu):6.1f} us  "
          f"gpu(cold L2) {statistics.median(gpu):8.1f} us")
