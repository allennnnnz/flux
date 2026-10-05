# E4a2 attempt 1 (2026-10-05 16:26-17:10): stopped by hand, kept for the record

- `run_xnode.sh` passed node 1's arguments through ssh unquoted; `--cases "5120:...;4096:..."` was split at ';'
  on css-host-159 (`node1.out`: "4096:...: command not found"), node 1 exited with code 2, and node 0's two
  ranks waited for it (38 min, 113% CPU each) until stopped. Had it run on, the TP4 fit would have failed and
  the pipeline would have skipped the probes and the oracle for ALL configs.
- Nothing had been pre-registered yet (only the setup commit 51f3f05), so the whole run was restarted after
  the fix (printf %q quoting, verified through ssh). The GEMM probes and the TP8 calibration here were valid
  but are re-measured in attempt 2 so that every number comes from one run.
