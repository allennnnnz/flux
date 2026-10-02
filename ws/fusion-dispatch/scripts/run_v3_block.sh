#!/bin/bash
# F1.3: block-level validation (validate_block_v3.py), 4 Llama-3-70B blocks.
# Tables from E1 (AG) + E5 (RS) maps; eager uses the steady-mode table, graph the gpu-mode table.
# Two runs with different seeds: run "cal" calibrates the layout choice (tp_ar vs sp_dispatch),
# run "eval" evaluates it on fresh rounds (validate_block_v3.py --seed).
# Run from repo root.
set -u
R=ws/fusion-dispatch/results
OUT=$R/${OUT_TAG:-v3_block_vllm_ar}
mkdir -p "$OUT"
for m in steady gpu; do
  python3 ws/fusion-dispatch/scripts/build_table_v1.py --points $R/e1_merged_points.csv $R/e5_rs_map_v1/summary_points.csv \
    --mode $m --out $OUT/table_$m.json > $OUT/table_$m.txt
done
S=ws/fusion-dispatch/scripts/validate_block_v3.py
run() {  # phase mode Ms table seed tag
  mkdir -p $OUT/$6
  timeout 3600 bash ws/fusion-dispatch/scripts/launch_vllm_env.sh $S --phase $1 --mode $2 --Ms $3 --table $4 \
    --L ${L:-4} --rounds ${ROUNDS:-200} --warmup 10 --seed $5 --out_dir $OUT/$6 > $OUT/$6/log_$1_$2.txt 2>&1
  echo "$6 $1 $2 exit $?"; grep -E "^\[" $OUT/$6/log_$1_$2.txt
}
for pass in "cal 20260930" "eval 20261001"; do
  set -- $pass
  run decode graph 8,16,32,64,128,256,384,512 $OUT/table_gpu.json $2 $1
  run decode eager 8,16,32,64,128,256,384,512 $OUT/table_steady.json $2 $1
  run prefill eager 1024,2048,4096,8192 $OUT/table_steady.json $2 $1
done
