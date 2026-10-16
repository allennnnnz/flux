#!/bin/bash
# hetero-proxy KV tier (2026-10-16): check v5 -> 3-round traced bench (the configuration that hung in v4) -> 200-round
# bench, each step only if the previous one exited 0. Run from the repo root inside tmux.
R=ws/hetero-proxy/results/kvtier
G="python3 common/measure/exclusive_guard_v2.py"
S=ws/hetero-proxy/scripts/kvtier/run_bench_v1.sh
$G --log $R/check_v5/guard.log -- $S $R/check_v5 --mode check --reference $R/hf_ref_v3/hf_reference_kv_v3.json 2>&1 | tee $R/check_v5/stdout.txt
grep -q 'END rc=0' $R/check_v5/guard.log || exit 1
HXB_TRACE=1 HXB_STACK_DUMP_S=120 $G --log $R/debug_v5/guard.log -- $S $R/debug_v5 --mode bench --rounds 3 2>&1 | tee $R/debug_v5/stdout.txt
grep -q 'END rc=0' $R/debug_v5/guard.log || exit 1
$G --log $R/bench_v5/guard.log -- $S $R/bench_v5 --mode bench --rounds 200 2>&1 | tee $R/bench_v5/stdout.txt
