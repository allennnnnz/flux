# GemmRS_multinode missing-barrier test (2026-10-06 03:28 UTC): NOT RUN

`scripts/xnode_flux_rs_smoke_v2.py` (hypothesis: python/flux/gemm_rs_sm80.py calls the intra-node GemmRS twice per
forward with no sync, and sm80 GemmRS's forward_barrier is a no-op, so early ranks scatter call-2 data into peers still
copying call 1 -> node 0, which keeps call 1, is wrong; node 1, which keeps call 2, is right).
css-host-159 was occupied by another process of the same account (`sglang::server`, ~74 GB on each of the 8 GPUs);
exclusive_guard_v2 aborted on 159 at preflight (see node1.out), node 0 waited until the command was stopped.
The hypothesis is still untested.
