################################################################################
# hetero-proxy E2E v6 (2026-10-16): qwen_tp_v5.py with side-effect-free decode-graph capture, to re-run the
# correctness checks of v3-v5. Found by the KV-tier check (results/kvtier/check_v1, kvtier_v2.py header):
# Engine._capture runs the decode step 3 times as warm-up before capturing, with the static pos buffer = 0, so it
# overwrote KV position 0 (every layer) of slots [0, B). In mode_check the graphs of each (configuration, B, Lb) are
# built inside the first decode_run, i.e. after that item's prefill: every item of every configuration decoded with a
# corrupted position 0. Timing (bench) is unaffected: graphs are built before any timed prefill.
# v6 = v5 with build_decode saving / restoring that position; everything else (modes, output file names
# check_v5.json / bench_v5.json) is v5's code, run through qwen_tp_v5.main().
# Usage: as qwen_tp_v5.py, with qwen_tp_v6.py and a new --out directory.
################################################################################
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_tp_v5  # noqa: E402


class Engine(qwen_tp_v5.Engine):
    def build_decode(self, cfg, B, Lb):
        if (cfg, B, Lb) in self.graphs:
            return self.graphs[(cfg, B, Lb)]
        torch.cuda.synchronize()
        k0, v0 = self.kc[:, :B, :, 0].clone(), self.vc[:, :B, :, 0].clone()
        r = super().build_decode(cfg, B, Lb)
        torch.cuda.synchronize()
        self.kc[:, :B, :, 0] = k0
        self.vc[:, :B, :, 0] = v0
        return r


qwen_tp_v5.Engine = Engine  # qwen_tp_v5.main() looks the class up at call time

if __name__ == "__main__":
    qwen_tp_v5.main()
