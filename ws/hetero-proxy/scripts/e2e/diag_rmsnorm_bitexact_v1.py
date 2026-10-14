################################################################################
# hetero-proxy E2E diagnostic (2026-10-14): is the CPU backend's C rmsnorm bit-identical to the engine's GPU rmsnorm
# (HF Qwen2RMSNorm order) on real activations-like inputs? Real model weights of the final norm, random bf16 rows
# with the scale of layer-32 hidden states. Reports the fraction of differing bf16 elements and the max difference.
################################################################################
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hxb.bridge import Bridge  # noqa: E402
from hxb.cpu_backend import CpuBackend  # noqa: E402
from qwen_tp_v1 import rmsnorm  # noqa: E402


def main():
    os.sched_setaffinity(0, list(range(0, 32)))
    torch.manual_seed(0)
    rows, H = 4096, 5120
    x = (torch.randn(rows, H) * 3).to(torch.bfloat16)
    x[:, :8] *= 40  # a few large channels, as real hidden states have
    w = (torch.rand(H) * 2).to(torch.bfloat16)
    ref = rmsnorm(x.cuda(), w.cuda(), 1e-6).cpu()
    be = CpuBackend()
    br = Bridge(be)
    q = be.queue()
    dx, dw = br.upload(q, x), br.upload(q, w)
    dy = be.alloc((rows, H), torch.bfloat16)
    s = be.signal()
    be.launch(q, "rmsnorm", {"x": dx, "w": dw, "y": dy, "rows": (0, rows), "eps": 1e-6}, done=(s, 1))
    s.wait(1)
    y = br.download(q, dy)
    diff = (y.float() - ref.float()).abs()
    n_diff = int((y.view(torch.int16) != ref.view(torch.int16)).sum())
    print(f"[bitexact] {n_diff} / {rows * H} bf16 elements differ ({100 * n_diff / (rows * H):.4f}%), "
          f"max |diff| {diff.max().item():.4g}, max rel {(diff / ref.float().abs().clamp_min(1e-3)).max().item():.3g}",
          flush=True)
    br.close()
    be.close()


if __name__ == "__main__":
    main()
