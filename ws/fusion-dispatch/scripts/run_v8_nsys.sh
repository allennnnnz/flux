#!/bin/bash
# V8: nsys timelines (nsys_probe_v2.py, interleaved) for the three AG layers not profiled in E2,
# three M each (below / near / above the E1 crossover), plus G-FC1 M=4096 again (v1 probe was blocked).
# Parses with nsys_parse_v1.py, keeps the .nsys-rep, deletes the derived sqlite. Run from repo root.
set -u
OUT=ws/fusion-dispatch/results/v8_nsys
mkdir -p "$OUT"
for spec in "G-QKV 64 12288" "G-QKV 2048 12288" "G-QKV 4096 12288" "L-QKV 64 8192" "L-QKV 2048 8192" "L-QKV 4096 8192" \
            "L-GU 512 8192" "L-GU 3072 8192" "L-GU 4096 8192" "G-FC1 4096 12288"; do
  set -- $spec
  tag=$1_M$2
  timeout 900 pixi run --manifest-path pixi.toml nsys profile -t cuda,nvtx --cuda-memory-usage=false --force-overwrite true \
    -o $OUT/$tag ./launch.sh ws/fusion-dispatch/scripts/nsys_probe_v2.py --layer $1 --M $2 --reps 20 > $OUT/log_$tag.txt 2>&1
  echo "$tag exit $?"
  nsys export --type sqlite --force-overwrite true -o $OUT/$tag.sqlite $OUT/$tag.nsys-rep > /dev/null 2>&1
  { echo "== $tag"; python3 ws/fusion-dispatch/scripts/nsys_parse_v1.py $OUT/$tag.sqlite --shard_bytes $(( $2 / 8 * $3 * 2 )) --device 0 | tail -6; } >> $OUT/parse_all.txt
  rm -f $OUT/$tag.sqlite
done
echo "# sqlite exports deleted (derived). Regenerate: nsys export --type sqlite -o X.sqlite X.nsys-rep" >> $OUT/parse_all.txt
