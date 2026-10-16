#!/bin/bash
# hetero-proxy KV tier K2 (2026-10-16): the pre-registered runs (STATUS section 0): load L (96 seats, 24 slots) then
# load H (192 seats, 32 slots, batch <= 48); per load ideal, tier, recompute, recompute, tier, ideal (two runs each,
# interleaved); warm-up 180 s, measured 600 s. Every run inside exclusive_guard_v2. Run from the repo root in tmux.
G="python3 common/measure/exclusive_guard_v2.py"
S=ws/hetero-proxy/scripts/kvtier/run_k2_v1.sh
run() {  # dir policy tag extra-args...
    d=$1; p=$2; t=$3; shift 3
    $G --log $d/guard_${p}_${t}.log -- $S $d --mode run --policy $p --tag $t --warmup 180 --duration 600 "$@" \
        2>&1 | tee $d/stdout_${p}_${t}.txt
}
L=ws/hetero-proxy/results/kvtier/k2/loadL
H=ws/hetero-proxy/results/kvtier/k2/loadH
mkdir -p $L $H
run $L ideal r1 --seats 96 --slots 24
run $L tier r1 --seats 96 --slots 24
run $L recompute r1 --seats 96 --slots 24
run $L recompute r2 --seats 96 --slots 24
run $L tier r2 --seats 96 --slots 24
run $L ideal r2 --seats 96 --slots 24
python3 ws/hetero-proxy/scripts/kvtier/k2_analyze_v1.py $L > /dev/null
run $H ideal r1 --seats 192 --slots 32 --max_b 48
run $H tier r1 --seats 192 --slots 32 --max_b 48
run $H recompute r1 --seats 192 --slots 32 --max_b 48
run $H recompute r2 --seats 192 --slots 32 --max_b 48
run $H tier r2 --seats 192 --slots 32 --max_b 48
run $H ideal r2 --seats 192 --slots 32 --max_b 48
python3 ws/hetero-proxy/scripts/kvtier/k2_analyze_v1.py $H > /dev/null
