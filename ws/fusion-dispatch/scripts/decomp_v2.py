################################################################################
# fusion-dispatch C1 v2 (2026-11-02): decomp_v1.py with the collectives on vLLM's PyNcclCommunicator and an explicit
# side stream instead of c10d async_op work objects. Why: the C1 v1 oracle run hung in its first decode point
# (CUDA-graph replays, 200 rounds; results/c1_compass/oracle_c1_qwen2.5-32b_tp8_decode: an eager 1-element
# all_reduce timed out at SeqNum 1314 after the captured c10d async collectives), although a 5-round smoke passed.
# vLLM itself captures PyNccl calls on its own streams; here the pattern is the plain CUDA-graph fork/join:
#   ag_gemm: side waits for main; chunk i is all-gathered on side and an event marks it; main waits for chunk i's event,
#            runs chunk i's GEMM (overlapping chunk i+1's all-gather) and copies it into the rank-major output; main
#            joins side at the end.
#   gemm_rs: chunk i's partial GEMM runs on main and an event marks it; side waits for it and reduce-scatters the chunk
#            while main computes chunk i+1; main joins side at the end.
# d = 1 runs the single collective on the main stream (also PyNccl, so sp_dec1 differs from sp_nccl, which uses c10d,
# only in the NCCL front end). Table / fallback behaviour as decomp_v1.py.
################################################################################
from collections import Counter

import torch


class DecompDispatcher:
    def __init__(self, group, pynccl, d=1, table=None, dtype=torch.bfloat16):
        self.W, self.nccl = group.size(), pynccl
        self.d, self.table, self.dtype = d, table, dtype
        self.side = torch.cuda.Stream()
        self.counts, self.misses = Counter(), Counter()

    def _d(self, kind, n, k, M):
        if self.table is None:
            d = self.d
        else:
            v = self.table.get(str(M), {}).get(f"{kind}:{n}:{k}")
            if v is None:
                self.misses[(kind, n, k, M, "no entry")] += 1
                return 1
            if v == "flux":
                raise ValueError("fusion choice in a decomposition-only table")
            d = int(v)
        if (M // self.W) % d:
            self.misses[(kind, n, k, M, d)] += 1
            return 1
        return d

    def ag_gemm(self, x_local, w):
        m, K = x_local.shape
        W, M, n = self.W, x_local.shape[0] * self.W, w.shape[0]
        d = self._d("ag", n, K, M)
        self.counts[("ag", d)] += 1
        main = torch.cuda.current_stream()
        if d == 1:
            full = torch.empty(M, K, dtype=self.dtype, device="cuda")
            self.nccl.all_gather(full, x_local, stream=main)
            return torch.mm(full, w.t())
        c = m // d
        gath = torch.empty(d, W * c, K, dtype=self.dtype, device="cuda")
        out = torch.empty(W, m, n, dtype=self.dtype, device="cuda")
        side = self.side
        side.wait_stream(main)
        evs = []
        for i in range(d):
            self.nccl.all_gather(gath[i], x_local[i * c:(i + 1) * c], stream=side)
            e = torch.cuda.Event()
            e.record(side)
            evs.append(e)
        for i in range(d):
            main.wait_event(evs[i])
            out[:, i * c:(i + 1) * c, :].copy_(torch.mm(gath[i], w.t()).view(W, c, n))
        main.wait_stream(side)
        return out.view(M, n)

    def gemm_rs(self, x, w):
        M, k = x.shape
        W, N = self.W, w.shape[0]
        m = M // W
        d = self._d("rs", N, k, M)
        self.counts[("rs", d)] += 1
        main = torch.cuda.current_stream()
        out = torch.empty(m, N, dtype=self.dtype, device="cuda")
        if d == 1:
            self.nccl.reduce_scatter(out, torch.mm(x, w.t()), stream=main)
            return out
        c = m // d
        xv = x.view(W, m, k)
        side = self.side
        side.wait_stream(main)
        keep = []
        for i in range(d):
            part = torch.matmul(xv[:, i * c:(i + 1) * c, :], w.t())  # (W, c, N), rank-major
            keep.append(part)
            e = torch.cuda.Event()
            e.record(main)
            side.wait_event(e)
            self.nccl.reduce_scatter(out[i * c:(i + 1) * c], part.view(W * c, N), stream=side)
        main.wait_stream(side)
        return out
