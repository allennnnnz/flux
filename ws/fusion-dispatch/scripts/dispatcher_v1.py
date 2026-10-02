################################################################################
# fusion-dispatch F1.1: per-(layer shape, M) path dispatcher for tensor-parallel
# linear layers in the sequence-parallel layout.
#
#   ag_gemm(x_local (M/W, K), w (n, K)) -> (M, n)       column-parallel, AllGather first
#   gemm_rs(x (M, k), w (N, k))         -> (M/W, N)     row-parallel, ReduceScatter after
#
# Paths
#   AG: flux            flux.AGKernel.forward (fused; NOT CUDA-graph capturable, F0.4)
#       nccl            dist.all_gather_into_tensor + torch.mm
#       fluxag          flux.AllGatherOp.run + torch.mm
#       fluxag_fluxgemm flux.AllGatherOp.run + AGKernel.gemm_only (serial)
#   RS: flux            flux.GemmRS.forward (fused; capturable)
#       nccl            torch.mm + dist.reduce_scatter_tensor
#
# Table (JSON, built by build_table_v1.py from measured maps):
#   {"meta": {...}, "ag": {"<n>x<K>": {"M": [...], "rank": [[[path, ms], ...], ...]}},
#                   "rs": {"<N>x<k>": {...}}}
#   rank[i] lists every measured path at M[i], fastest first (ties already resolved).
# Rules
#   * All ranks MUST take the same path (collectives otherwise deadlock): the table is
#     read on rank 0, broadcast, and its sha256 compared across ranks at load.
#   * M not in the table: use the next larger measured M (serving pads to buckets);
#     counted in self.misses. Shape not in the table: M <= 512 -> nccl, else flux.
#   * graph_mode=True: paths that cannot be captured are skipped (next in rank).
#   * Decisions are cached per (kind, dims, M): the hot path is one dict lookup.
#     (2026-09-30, before any F1 result: eager smoke run showed the uncached choose()
#     costing measurable host time in the launch-bound decode regime.)
################################################################################
import bisect
import hashlib
import json
from collections import Counter

import torch
import torch.distributed as dist

import flux

NOT_CAPTURABLE = {("ag", "flux")}  # F0.4, reports/20260930_f04_cuda_graph.md
FALLBACK_SMALL_M = 512


def ag_option():
    o = flux.AllGatherOption()
    o.mode = flux.AGRingMode.All2All
    o.use_read = True
    o.use_cuda_core_local = False
    o.use_cuda_core_ag = False
    o.fuse_sync = False
    o.input_buffer_copied = False
    return o


class DispatchTable:
    def __init__(self, data):
        self.data = data
        self._idx = {kind: {key: ent["M"] for key, ent in data.get(kind, {}).items()} for kind in ("ag", "rs")}

    @classmethod
    def load_broadcast(cls, path, group):
        obj = [None]
        if dist.get_rank(group) == 0:
            with open(path) as f:
                obj[0] = json.load(f)
        dist.broadcast_object_list(obj, src=dist.get_global_rank(group, 0), group=group)
        t = cls(obj[0])
        digests = [None] * dist.get_world_size(group)
        dist.all_gather_object(digests, t.digest(), group=group)
        if len(set(digests)) != 1:
            raise RuntimeError(f"dispatch table differs across ranks: {digests}")
        return t

    def digest(self):
        return hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()

    def ranking(self, kind, key, M):
        """Returns (ranked paths, exact_hit). None if the shape is unknown."""
        ent = self.data.get(kind, {}).get(key)
        if ent is None:
            return None, False
        Ms = self._idx[kind][key]
        i = bisect.bisect_left(Ms, M)
        exact = i < len(Ms) and Ms[i] == M
        i = min(i, len(Ms) - 1)
        return [p for p, _ in ent["rank"][i]], exact


class FluxDispatcher:
    def __init__(self, group, table, max_m, dtype=torch.bfloat16, graph_mode=False, force=None):
        """force: optional {"ag": path, "rs": path} to pin a path (baselines / tests)."""
        self.group, self.W, self.rank = group, group.size(), group.rank()
        self.table, self.max_m, self.dtype = table, max_m, dtype
        self.graph_mode, self.force = graph_mode, force or {}
        self.opt = ag_option()
        self._agk, self._agop, self._rs, self._gbuf = {}, {}, {}, {}
        self.counts, self.misses = Counter(), Counter()
        self._cache = {}

    # ---- lazily built backends (one per shape, shared by all layers of that shape)
    def _ag_kernel(self, n, K):
        if (n, K) not in self._agk:
            self._agk[(n, K)] = flux.AGKernel(self.group, 1, self.max_m, n, K, self.dtype, output_dtype=self.dtype)
        return self._agk[(n, K)]

    def _ag_op(self, K):
        if K not in self._agop:
            self._agop[K] = flux.AllGatherOp(self.group, 1, self.max_m, K, self.dtype)
        return self._agop[K]

    def _gemm_rs(self, N, k):
        if (N, k) not in self._rs:
            self._rs[(N, k)] = flux.GemmRS(self.group, 1, self.max_m, N, self.dtype, self.dtype,
                                           transpose_weight=False)
        return self._rs[(N, k)]

    def _gather_buf(self, K):
        if K not in self._gbuf:
            self._gbuf[K] = torch.empty(self.max_m, K, dtype=self.dtype, device="cuda")
        return self._gbuf[K]

    def prepare(self, ag_shapes=(), rs_shapes=()):
        """Build backends up front (collective constructors must run on all ranks together)."""
        for n, K in ag_shapes:
            self._ag_kernel(n, K)
            self._ag_op(K)
            self._gather_buf(K)
        for N, k in rs_shapes:
            self._gemm_rs(N, k)

    # ---- decision
    def choose(self, kind, key, M):
        if kind in self.force:
            return self.force[kind]
        ranked, exact = self.table.ranking(kind, key, M)
        if ranked is None:
            self.misses[(kind, key, "shape")] += 1
            return "nccl" if M <= FALLBACK_SMALL_M else "flux"
        if not exact:
            self.misses[(kind, key, M)] += 1
        for p in ranked:
            if not (self.graph_mode and (kind, p) in NOT_CAPTURABLE):
                return p
        return "nccl"

    # ---- ops
    def _path(self, kind, a, b, M):
        ck = (kind, a, b, M)
        p = self._cache.get(ck)
        if p is None:
            p = self._cache[ck] = self.choose(kind, f"{a}x{b}", M)
        self.counts[(kind, p)] += 1
        return p

    def ag_gemm(self, x_local, w):
        M = x_local.shape[0] * self.W
        n, K = w.shape
        path = self._path("ag", n, K, M)
        if path == "flux":
            out = torch.empty(M, n, dtype=self.dtype, device="cuda")
            return self._ag_kernel(n, K).forward(x_local, w, output=out, transpose_weight=False,
                                                 all_gather_option=self.opt)
        if path == "nccl":
            full = self._gather_buf(K)[:M]
            dist.all_gather_into_tensor(full, x_local, group=self.group)
            return torch.mm(full, w.t())
        ag = self._ag_op(K)
        ag.run(x_local, None, self.opt, torch.cuda.current_stream().cuda_stream)
        full = ag.local_input_buffer()[:M]
        if path == "fluxag":
            return torch.mm(full, w.t())
        if path == "fluxag_fluxgemm":
            return self._ag_kernel(n, K).gemm_only(full, w, transpose_weight=False)
        raise ValueError(path)

    def gemm_rs(self, x, w):
        M, k = x.shape
        N = w.shape[0]
        path = self._path("rs", N, k, M)
        if path == "flux":
            return self._gemm_rs(N, k).forward(x, w)
        if path == "nccl":
            part = torch.mm(x, w.t())
            out = torch.empty(M // self.W, N, dtype=self.dtype, device="cuda")
            dist.reduce_scatter_tensor(out, part, group=self.group)
            return out
        raise ValueError(path)
