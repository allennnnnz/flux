#!/bin/bash
# G3: op maps for situations the predictor has never seen. Same protocol as E1 / E5
# (dispatch_map_v2.py, dispatch_map_rs_v1.py: interleaved, >= 200 rounds, L2 flush, gpu + steady,
# rank-max, clock filter). One process per (TP, layer), each under exclusive_guard; a finished
# (TP, layer) leaves done_<layer>, so the script can be re-run after an interruption (a partial raw
# file is moved aside, never appended to). CONFIGS / MS must match predict_g3_v1.py, whose
# predictions were committed before this ran.
#   tmux new -d -s g3map 'bash ws/fusion-dispatch/scripts/run_g3_map_v1.sh ws/fusion-dispatch/results/g3_map'
cd /home/rogerlee/flux
R=${1:-ws/fusion-dispatch/results/g3_map}
S=ws/fusion-dispatch/scripts; G=common/measure/exclusive_guard.py
MS=16,64,136,256,512,1024,2048,3072,4096,8192
mkdir -p $R
exec >> $R/run_log.txt 2>&1
echo "[G3] start $(date '+%F %T')"
run() {  # tp side layer
  local tp=$1 side=$2 layer=$3 out=$R/tp$1
  mkdir -p $out
  [ -f $out/done_$layer ] && { echo "[G3] skip tp$tp $layer (done)"; return; }
  [ -f $out/raw_$layer.csv ] && mv $out/raw_$layer.csv $out/raw_$layer.csv.partial.$(date +%s)
  local launch script
  if [ "$tp" = 8 ]; then launch=./launch.sh; else launch="$S/launch_tp.sh $tp"; fi
  if [ "$side" = ag ]; then script=dispatch_map_v2.py; else script=dispatch_map_rs_v1.py; fi
  python3 $G --log $out/guard_$layer.log -- timeout 3600 pixi run --manifest-path pixi.toml $launch $S/$script \
    --layer $layer --Ms $MS --modes gpu,steady --rounds 200 --warmup 30 --clock_period 0.02 --out_dir $out \
    > $out/log_$layer.txt 2>&1
  local rc=$?
  echo "[G3] tp$tp $layer exit $rc $(date '+%T') $(tail -1 $out/guard_$layer.log)"
  [ $rc = 0 ] && touch $out/done_$layer
}
run 8 ag Q-GU; run 8 rs Q-down
run 8 ag L8-QKV; run 8 ag L8-GU; run 8 rs L8-O; run 8 rs L8-down
run 4 ag L-QKV; run 4 ag L-GU; run 4 rs L-O; run 4 rs L-down
run 4 ag Q-GU; run 4 rs Q-down
run 2 ag L8-QKV; run 2 ag L8-GU; run 2 rs L8-O; run 2 rs L8-down
echo "[G3] done $(date '+%F %T')"
touch $R/DONE
