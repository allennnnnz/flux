################################################################################
# hetero-proxy KV tier (2026-10-17): k2_engine_v1.K2Engine with the capture warm-ups off the graph communicator.
# K4.1's first run (results/kvtier/k4/interference_v1_hang_capture) hung inside capture_all: every rank in an eager
# all_reduce of a warm-up pass, GPUs idle - the symptom of kvtier_v4's hang (results/kvtier/bench_v4_hang1). v1 kept
# graph-captured and eager all-reduces apart at run time (own communicator for the graphs) but its capture warm-ups
# (3 eager passes before each of ~110 captures) still ran on that communicator, interleaved with the captures. It
# worked in all 17 K2 runs, so the hazard is probabilistic. v2: warm-up passes all-reduce on the DEFAULT communicator;
# the graph communicator is created by one eager all-reduce at construction (NCCL cannot initialise inside a capture)
# and is then used only by captured all-reduces. Graph contents and numerics are unchanged.
################################################################################
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2_engine_v1 import B_BUCKETS, S_STEP, T_BUCKETS, HostKV, bucket  # noqa: E402,F401
from k2_engine_v1 import K2Engine as _K2EngineV1  # noqa: E402


class K2Engine(_K2EngineV1):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        t = torch.zeros(1, device=self.dev)
        dist.all_reduce(t, group=self.pg_graph)  # create the graph communicator before any capture
        torch.cuda.synchronize()
        self._ar_group = self.pg_graph

    def _ar(self, t):
        dist.all_reduce(t, group=self._ar_group)

    @torch.inference_mode()
    def _capture(self, key, fn, gated):
        cur = torch.cuda.current_stream()
        s = torch.cuda.Stream()
        s.wait_stream(cur)
        self._ar_group = None  # warm-up passes: eager all-reduces on the default communicator
        with torch.cuda.stream(s):
            for _ in range(3):
                if gated:
                    self.gate.fill_(1)
                fn()
        cur.wait_stream(s)
        torch.cuda.synchronize()
        self._ar_group = self.pg_graph  # the captured all-reduces: graph communicator only
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            fn()
        torch.cuda.synchronize()
        self.gs[key] = g
