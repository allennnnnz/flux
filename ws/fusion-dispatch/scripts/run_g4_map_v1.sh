#!/bin/bash
# G4 step 6: oracle op maps of the fresh test set (same protocol as E1 / G3: 200 rounds, gpu + steady),
# measured only AFTER predictions_g4.csv and decisions_g4.csv were committed. One process per
# (TP, layer), each under exclusive_guard; resumable (done_<layer>; a partial raw file is moved aside).
#   tmux new -d -s g4map 'bash ws/fusion-dispatch/scripts/run_g4_map_v1.sh'
cd /home/rogerlee/flux
R=${1:-ws/fusion-dispatch/results/g4_map}
S=ws/fusion-dispatch/scripts; G=common/measure/exclusive_guard.py
MS=24,96,200,384,768,1536,2560,6144
mkdir -p $R
exec >> $R/run_log.txt 2>&1
echo "[G4-map] start $(date '+%F %T')"
run() {  # tp side layer
  local tp=$1 side=$2 layer=$3 out=$R/tp$1
  mkdir -p $out
  [ -f $out/done_$layer ] && { echo "[G4-map] skip tp$tp $layer (done)"; return; }
  [ -f $out/raw_$layer.csv ] && mv $out/raw_$layer.csv $out/raw_$layer.csv.partial.$(date +%s)
  local launch script
  if [ "$tp" = 8 ]; then launch=./launch.sh; else launch="$S/launch_tp.sh $tp"; fi
  if [ "$side" = ag ]; then script=dispatch_map_v2.py; else script=dispatch_map_rs_v1.py; fi
  python3 $G --log $out/guard_$layer.log -- timeout 3600 pixi run --manifest-path pixi.toml $launch $S/$script \
    --layer $layer --Ms $MS --modes gpu,steady --rounds 200 --warmup 30 --clock_period 0.02 --out_dir $out \
    > $out/log_$layer.txt 2>&1
  local rc=$?
  echo "[G4-map] tp$tp $layer exit $rc $(date '+%T') $(tail -1 $out/guard_$layer.log)"
  [ $rc = 0 ] && touch $out/done_$layer
}
for tp in 8 4; do run $tp ag Q32-QKV; run $tp ag Q32-GU; run $tp rs Q32-O; run $tp rs Q32-down; done
run 4 ag L8-QKV; run 4 ag L8-GU; run 4 rs L8-O; run 4 rs L8-down
echo "[G4-map] done $(date '+%F %T')"
touch $R/DONE
