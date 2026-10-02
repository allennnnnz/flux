#!/bin/bash
# G2 calibration: one process per shape group (calibrate_hw_v1.py header explains why), each under
# common/measure/exclusive_guard.py. Run from anywhere, ideally in tmux:
#   tmux new -d -s g2cal 'bash ws/fusion-dispatch/scripts/run_calibration_v1.sh ws/fusion-dispatch/results/g2_calibration'
# Extra arguments after the out dir are passed to every calibrate_hw_v1.py call (e.g. --rounds 5 for a smoke test).
# TP=<n> (environment) runs on the first n GPUs via launch_tp.sh; default 8 uses launch.sh (as in G2).
cd /home/rogerlee/flux
R=${1:?out_dir}; shift
S=ws/fusion-dispatch/scripts; G=common/measure/exclusive_guard.py
mkdir -p $R
exec >> $R/run_log.txt 2>&1
TP=${TP:-8}
if [ "$TP" = 8 ]; then LAUNCH="./launch.sh"; else LAUNCH="$S/launch_tp.sh $TP"; fi
echo "[G2] start $(date '+%F %T')  TP=$TP  extra args: $*"
run() {  # tag, then calibrate_hw_v1.py arguments
  local tag=$1; shift
  python3 $G --log $R/guard_$tag.log -- timeout 1200 pixi run --manifest-path pixi.toml $LAUNCH \
    $S/calibrate_hw_v1.py --tag $tag --modes gpu,steady --out_dir $R "$@" > $R/log_$tag.txt 2>&1
  echo "[G2] $tag exit $? $(date '+%T') $(tail -1 $R/guard_$tag.log)"
}
run comm --groups comm "$@"
for s in 2560x5120 5120x10240 1536x16384; do run ag_$s --groups ag --ag_shapes $s "$@"; done
for s in 6144x2560 10240x1280; do run rs_$s --groups rs --rs_shapes $s "$@"; done
echo "[G2] done $(date '+%F %T')"
touch $R/DONE
