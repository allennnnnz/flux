################################################################################
# hetero-proxy I3 diagnostic (2026-10-09): is the CPU backend's device copy slower when its source staging slot is
# cold (not reused recently)? Hypothesis from the I3 dry runs: interleaving 6 pipelines (each with its own slots)
# made device copies ~25% slower than one pipeline repeated. copy_in of 5 MiB from 1 / 4 / 24 / 96 rotating slots
# (5 MiB .. 480 MiB), each copy waited for; median of the last 40. Usage: python diag_cold_slots_v1.py
################################################################################
import os, sys, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hxb.cpu_backend import CpuBackend
def main():
    os.sched_setaffinity(0, list(range(32)))
    be = CpuBackend(); q = be.queue(); s = be.signal()
    cr, H = 512, 5120
    dev = be.alloc((4096, H), torch.bfloat16); v = 0
    for nslots in (1, 4, 24, 96):
        slots = [be.host_alloc((cr, H), torch.bfloat16) for _ in range(nslots)]
        for sl in slots: sl.tensor.fill_(1.0)
        be.trace()
        for i in range(max(48, 2 * nslots)):
            v += 1; be.copy_in(q, dev, ((i % 8) * cr, (i % 8 + 1) * cr), slots[i % nslots], None, done=(s, v), tag="in"); s.wait(v)
        tr = be.trace()
        d = [r["t_end"] - r["t_start"] for r in tr][max(0, len(tr) - 40):]
        print(f"slots {nslots:3d} ({nslots*cr*H*2/2**20:6.0f} MiB rotating)  copy_in 5 MiB median {statistics.median(d)*1e3:.3f} ms -> {cr*H*2/statistics.median(d)/1e9:.1f} GB/s", flush=True)
    be.close()
if __name__ == "__main__":
    main()
