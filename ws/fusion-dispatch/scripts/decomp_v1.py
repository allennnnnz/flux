################################################################################
# fusion-dispatch C1 (2026-11-02): Compass's ODP-InterDecom (arXiv 2609.10549, Sec. II-B / IV) as a dispatcher with the
# FluxDispatcher interface (ag_gemm / gemm_rs), so validate_block_v5.py can run it as a policy. Compass publishes no
# code; this follows the paper's description: the communication and the GEMM of an AG+GEMM / GEMM+RS are split into d
# sub-operators that are pipelined with standard collectives (NCCL), d = 1 is the sequential baseline (= sp_nccl).
#   ag_gemm: the local shard's rows are cut into d chunks; chunk i is all-gathered asynchronously (c10d NCCL stream)
#            while chunk i-1's GEMM runs on the compute stream; the (W*c, n) result of chunk i is copied into the
#            rank-major (M, n) output (one GEMM per chunk + a copy, rather than W small GEMMs).
#   gemm_rs: chunk i's partial (rows i of every rank's segment, a strided batched GEMM) is reduce-scattered
#            asynchronously while chunk i+1's GEMM runs; the outputs land in contiguous rows of the local shard.
# A table (Compass's per-op choice, compass_model_v1.py) maps M -> {"ag:<n>:<K>" | "rs:<N>:<k>": d} (output and input
# width: o_proj and down both have N = hidden); without one, d is fixed.
# A d that does not divide the local rows falls back to d = 1 and is counted in .misses.
################################################################################
from collections import Counter

import torch
import torch.distributed as dist


class DecompDispatcher:
    def __init__(self, group, d=1, table=None, dtype=torch.bfloat16):
        self.group, self.W = group, group.size()
        self.d, self.table, self.dtype = d, table, dtype
        self.counts, self.misses = Counter(), Counter()

    def _d(self, kind, n, k, M):
        if self.table is None:
            d = self.d
        else:
            t = self.table.get(str(M), {})
            v = t.get(f"{kind}:{n}:{k}")
            if v is None:
                self.misses[(kind, n, k, M, "no entry")] += 1
                return 1
            if v == "flux":
                raise ValueError("fusion choice in a decomposition-only table")  # NVLink runs not wired yet
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
        if d == 1:
            full = torch.empty(M, K, dtype=self.dtype, device="cuda")
            dist.all_gather_into_tensor(full, x_local, group=self.group)
            return torch.mm(full, w.t())
        c = m // d
        gath = torch.empty(d, W * c, K, dtype=self.dtype, device="cuda")
        works = [dist.all_gather_into_tensor(gath[i], x_local[i * c:(i + 1) * c], group=self.group, async_op=True)
                 for i in range(d)]
        out = torch.empty(W, m, n, dtype=self.dtype, device="cuda")
        for i in range(d):
            works[i].wait()
            out[:, i * c:(i + 1) * c, :].copy_(torch.mm(gath[i], w.t()).view(W, c, n))
        return out.view(M, n)

    def gemm_rs(self, x, w):
        M, k = x.shape
        W, N = self.W, w.shape[0]
        m = M // W
        d = self._d("rs", N, k, M)
        self.counts[("rs", d)] += 1
        out = torch.empty(m, N, dtype=self.dtype, device="cuda")
        if d == 1:
            dist.reduce_scatter_tensor(out, torch.mm(x, w.t()), group=self.group)
            return out
        c = m // d
        xv = x.view(W, m, k)
        works, keep = [], []
        for i in range(d):
            part = torch.matmul(xv[:, i * c:(i + 1) * c, :], w.t())  # (W, c, N), rank-major
            keep.append(part)
            works.append(dist.reduce_scatter_tensor(out[i * c:(i + 1) * c], part.view(W * c, N), group=self.group,
                                                    async_op=True))
        for wk in works:
            wk.wait()
        return out
