#!/bin/bash
################################################################################
# fusion-dispatch C1 (2026-11-02): Compass (arXiv 2609.10549) vs our decider on the E4a3 cross-node points.
# Compass publishes no code: its ODP-InterDecom is decomp_v1.py, its selector (Eq. 2-5, Algorithm 1) compass_model_v1.py;
# its fusion kernel (TA-IntraFusion) does not exist here and fusion cannot run across nodes anyway, so on this
# environment Compass = "SP layout, choose the decomposition degree d per op". Both selectors get the same inputs
# (E4a3's single-GPU cuBLAS times and calibrated NCCL curves); Compass's eta(d) regression uses its own calibration at
# M = 768,1024,1536,2048,3072,4096 (disjoint from the test points 1280,2560,3584 and the decode points).
# Our decider's decisions are E4a3's, already pushed before any C1 data (4092e58).
# Pre-registered (2026-11-02, before any C1 calibration or oracle data). Disclosed smoke runs (results/c1_compass/smoke,
# 3-5 rounds): prefill M=1280 TP8 every d > 1 slower than d = 1; decode M=96 TP8 (graph) d=4 6.2 ms vs SP d=1 2.3 vs
# vLLM default 2.0; a self-test of the selector on the 5-round smoke calibration chose d = 4 at most decode points.
#   C1-1 at prefill the best fixed d > 1 beats SP d = 1 at >= 5 of the 9 points (by 2-15%).
#   C1-2 prefill: Compass's regret against the oracle over all measured policies <= 5%; our decider (no decomposition
#        option) has >= 2% there (decomposition adds something it cannot choose).
#   C1-3 decode: Compass (literal, its model's d) regret >= 50% (outside its design domain: per-collective cost);
#        Compass (prefill only: d = 1 at decode = SP) regret >= 5%; our decider <= 1% (it can choose TP + all-reduce).
#   C1-4 decider+compass (our layout choice; inside SP, Compass prefill-only) has the lowest pooled regret over 21 points.
# Unattended in tmux on css-host-158 (repo root):  tmux new -d -s c1 'bash ws/fusion-dispatch/scripts/run_c1_v1.sh'
################################################################################
set -u
REPO=/home/rogerlee/flux
cd $REPO
S=ws/fusion-dispatch/scripts
R=ws/fusion-dispatch/results/c1_compass
T=ws/fusion-dispatch/results/g4_block_tables
LOG=$R/run_log.txt
mkdir -p $R
export XNODE_ENV="NVSHMEM_HCA_LIST=mlx5_0 XNODE_HCAS_158=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7 XNODE_HCAS_159=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5"
export XNODE_TIMEOUT=3600
export E4A_OUT_DIR=$REPO/ws/fusion-dispatch/results/e4a3_layout E4A_NEW_DECODE=96,160,288,416 E4A_NEW_PREFILL=1280,2560,3584
export C1_OUT_DIR=$REPO/$R
CO="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
CAL_MS=768,1024,1536,2048,3072,4096
CONFIGS=("qwen2.5-32b 8 4 0,1,6,7 tp8x2n3" "qwen2.5-32b 4 2 0,6 tp4x2n3" "llama3-8b 4 2 0,6 tp4x2n3")
TAG=c1

say() { echo "[C1] $(date '+%F %T') $*" | tee -a $LOG; }

gitsync() {  # gitsync "<message>" <paths...>; returns 0 only if the local commit succeeded
  local msg=$1; shift
  git add -A "$@" >> $LOG 2>&1
  if git diff --cached --quiet; then say "git: nothing to commit for: $msg"; return 0; fi
  if ! git commit -q -m "$msg" -m "$CO" >> $LOG 2>&1; then say "git: COMMIT FAILED: $msg"; return 1; fi
  say "git: committed $(git log --oneline -1)"
  for i in 1 2 3; do
    if git pull -q --rebase --autostash git@github.com:allennnnnz/flux.git fusion-dispatch >> $LOG 2>&1; then
      if git push -q git@github.com:allennnnnz/flux.git HEAD:fusion-dispatch >> $LOG 2>&1; then
        say "git: pushed $(git rev-parse --short HEAD)"; return 0
      fi
    else
      git rebase --abort >> $LOG 2>&1
    fi
    sleep 20
  done
  say "git: PUSH FAILED (local commit kept: $(git rev-parse --short HEAD))"
  return 0
}

say "start (pid $$)"
gitsync "fusion-dispatch C1: setup (Compass decomposition dispatcher, selector re-implementation, block v5; pre-registered C1-1..4)" \
  $S/decomp_v1.py $S/validate_block_v5.py $S/calibrate_compass_v1.py $S/compass_model_v1.py $S/run_c1_v1.sh $R/smoke \
  ws/fusion-dispatch/STATUS.md ws/fusion-dispatch/JOURNAL.md $LOG

say "== 1. Compass calibration (op level, M=$CAL_MS, d=1..16) and fit"
ok=1
for c in "${CONFIGS[@]}"; do
  set -- $c; model=$1 tp=$2 ppn=$3 gpus=$4
  XNODE_GPUS=$gpus bash $S/run_xnode.sh $R/cal_${model}_tp$tp $ppn vllm $S/calibrate_compass_v1.py --model $model --Ms $CAL_MS \
    --rounds 50 --out_dir $R/cal_${model}_tp$tp 2>&1 | tail -1 | tee -a $LOG
  python3 $S/compass_model_v1.py fit $R/cal_${model}_tp$tp $model $tp $TAG 2>&1 | tee -a $LOG || ok=0
done

say "== 2. Compass choices for the E4a3 points (pre-registration)"
for c in "${CONFIGS[@]}"; do
  set -- $c
  python3 $S/compass_model_v1.py predict $TAG $1 $2 $5 2>&1 | tee -a $LOG || ok=0
done
if [ $ok = 1 ] && gitsync "fusion-dispatch C1: Compass calibration, fitted eta(d), pre-registered Compass choices (before the oracle run)" $R $LOG; then
  say "== 3. oracle: vLLM default, SP d=1, SP d=2/4/8/16, Compass (200 rounds, interleaved; decode in CUDA graphs)"
  for c in "${CONFIGS[@]}"; do
    set -- $c; model=$1 tp=$2 ppn=$3 gpus=$4
    for ph in decode prefill; do
      mode=eager; Ms=$E4A_NEW_PREFILL
      [ $ph = decode ] && { mode=graph; Ms=$E4A_NEW_DECODE; }
      out=$R/oracle_${TAG}_${model}_tp${tp}_$ph
      XNODE_GPUS=$gpus bash $S/run_xnode.sh $out $ppn vllm $S/validate_block_v5.py --model $model --phase $ph --mode $mode \
        --Ms $Ms --table_g4 $T/table_g4_${model}_tp${tp}_$ph.json --table_g3 $T/table_g3_${model}_tp${tp}_$ph.json \
        --policies tp_ar_vllm,sp_nccl,sp_dec2,sp_dec4,sp_dec8,sp_dec16,sp_compass --decomp_ds 2,4,8,16 \
        --compass_table $R/compass_table_${TAG}_${model}_tp${tp}.json --rounds 200 --warmup 5 --out_dir $out 2>&1 | tail -1 | tee -a $LOG
    done
  done
  say "== 4. evaluation"
  python3 $S/compass_model_v1.py report $TAG qwen2.5-32b:8:tp8x2n3,qwen2.5-32b:4:tp4x2n3,llama3-8b:4:tp4x2n3 2>&1 | tee -a $LOG
  gitsync "fusion-dispatch C1: prefill oracle and evaluation (Compass vs decider)" $R $LOG
else
  say "pre-registration failed -> oracle NOT run"
fi
say "done"
