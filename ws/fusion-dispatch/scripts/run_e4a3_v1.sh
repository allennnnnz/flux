#!/bin/bash
################################################################################
# fusion-dispatch E4a3 (2026-11-02): the frozen E4a2 method (e4a_layout_v2.py: non-monotone curves, exact-size
# calibration, G4 probe rule, EPS 3%) on ANOTHER set of fresh points, in a changed cross-node environment:
#   - css-host-159 lost its enp150s0 NIC (rails 5/6); rails 7/8 are mlx5_4/5 there, mlx5_6/7 on css-host-158;
#     css-host-158 mlx5_3 (rail 4) is back -> per-host HCA lists (launch_xnode.sh XNODE_HCAS_158/159) and GPUs whose
#     PXB NIC exists on both nodes: TP8 = GPUs 0,1,6,7 per node (E4a2: 0,1,4,5), TP4 = GPUs 0,6 (E4a2: 0,4).
#     Anchor before this run: NCCL 4+4 busbw 13.8-14.9 GB/s at 256 MiB-1 GiB (results/e4a3_layout/anchor_ppn4; E4: 14.7-14.8).
#   - points never used before: decode (graph) M = 96, 160, 288, 416; prefill (eager) M = 1280, 2560, 3584
#     (E4a: 32,128,256,384,512 / 1024,2048,4096; E4a2: 64,192,320,448 / 768,1536,3072).
#   - the model is re-calibrated from scratch in the new environment (new hw profiles tp8x2n3 / tp4x2n3).
# Pre-registered (2026-11-02, before any E4a3 calibration or oracle data):
#   E1 decider (final, after probes) regret <= 2% over the 21 points and right on >= 17/21 (E4a2: 0.11%, 20/21).
#   E2 at least one frozen simple rule (R1, R1p, R2, R3) has regret >= 5% or a worst point >= 20% slower
#      (E4a2: R1/R2 4.30% worst 103%, R1p 5.81%, R3 3.84%).
#   E3 the decider has the lowest total regret of all policies.
#   Success = E1 and E2 (plan criterion: decider <= 2% while a simple rule fails in some environment).
# Unattended, inside tmux on css-host-158, repo root:  tmux new -d -s e4a3 'bash ws/fusion-dispatch/scripts/run_e4a3_v1.sh'
#   1 single-GPU cuBLAS at the new M; 2 cross-node calibration + fit; 3 model decisions + probe list -> COMMIT + PUSH;
#   4 probes (30 rounds) + final decisions -> COMMIT + PUSH; 5 oracle (200 rounds, both nodes guarded) + eval -> COMMIT + PUSH
################################################################################
set -u
REPO=/home/rogerlee/flux
cd $REPO
S=ws/fusion-dispatch/scripts
R=ws/fusion-dispatch/results/e4a3_layout
T=ws/fusion-dispatch/results/g4_block_tables
G=common/measure/exclusive_guard_v2.py
LOG=$R/run_log.txt
mkdir -p $R/gemm
export XNODE_ENV="NVSHMEM_HCA_LIST=mlx5_0 XNODE_HCAS_158=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7 XNODE_HCAS_159=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5"
export E4A_OUT_DIR=$REPO/$R E4A_NEW_DECODE=96,160,288,416 E4A_NEW_PREFILL=1280,2560,3584
export XNODE_TIMEOUT=3600
CO="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
NEW_GEMM_MS=96,160,288,416,1280,2560,3584
ALL_MS=32,64,96,128,160,192,256,288,320,384,416,448,512,768,1024,1280,1536,2048,2560,3072,3584,4096
CONFIGS=("tp8x2n3 qwen2.5-32b 8 4 0,1,6,7" "tp4x2n3 qwen2.5-32b 4 2 0,6" "tp4x2n3 llama3-8b 4 2 0,6")

say() { echo "[E4a3] $(date '+%F %T') $*" | tee -a $LOG; }

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

block() {  # block <out> <model> <tp> <ppn> <phase> <Ms> <rounds> <gpus>
  local out=$1 model=$2 tp=$3 ppn=$4 phase=$5 Ms=$6 rounds=$7 gpus=$8 mode=eager
  [ "$phase" = decode ] && mode=graph
  XNODE_GPUS=$gpus bash $S/run_xnode.sh $out $ppn vllm $S/validate_block_v4.py --model $model --phase $phase --mode $mode --Ms $Ms \
    --table_g4 $T/table_g4_${model}_tp${tp}_${phase}.json --table_g3 $T/table_g3_${model}_tp${tp}_${phase}.json \
    --policies tp_ar_vllm,sp_nccl --rounds $rounds --warmup 5 --out_dir $out 2>&1 | tail -1 | tee -a $LOG
}

say "start (pid $$)"
gitsync "fusion-dispatch E4a3: setup (frozen E4a2 method, fresh M points, changed cross-node env: 6 rails, GPUs 0,1,6,7 / 0,6; pre-registered E1-E3)" \
  $S/launch_xnode.sh $S/e4a_layout_v2.py $S/run_e4a3_v1.sh $R/anchor_prediction.txt $R/anchor_ppn4 \
  ws/fusion-dispatch/STATUS.md ws/fusion-dispatch/JOURNAL.md PROJECT.md $LOG

say "== 1. single-GPU cuBLAS at the new M ($NEW_GEMM_MS), css-host-158"
for cfg in "8 qwen2.5-32b" "4 qwen2.5-32b" "4 llama3-8b"; do
  set -- $cfg; tp=$1; model=$2
  if [ "$tp" = 8 ]; then L=./launch.sh; else L="$S/launch_tp.sh $tp"; fi
  python3 $G --log $R/gemm/guard_${model}_tp$tp.log -- timeout 1800 pixi run --manifest-path pixi.toml $L \
    $S/probe_gemm_v1.py --model $model --Ms $NEW_GEMM_MS --out_dir $R/gemm > $R/gemm/log_${model}_tp$tp.txt 2>&1
  say "gemm tp$tp $model exit $? $(grep -o 'END rc=[0-9-]* [A-Z]*' $R/gemm/guard_${model}_tp$tp.log | tail -1)"
done

say "== 2. cross-node calibration at exact sizes"
XNODE_GPUS=0,1,6,7 bash $S/run_xnode.sh $R/cal_tp8x2n3 4 vllm $S/calibrate_xnode_v2.py --cases "5120:$ALL_MS" --out_dir $R/cal_tp8x2n3 2>&1 | tail -1 | tee -a $LOG
XNODE_GPUS=0,6 bash $S/run_xnode.sh $R/cal_tp4x2n3 2 vllm $S/calibrate_xnode_v2.py --cases "5120:$ALL_MS;4096:$ALL_MS" --out_dir $R/cal_tp4x2n3 2>&1 | tail -1 | tee -a $LOG
python3 $S/e4a_layout_v2.py fit $R/cal_tp8x2n3 tp8x2n3 2>&1 | tee -a $LOG
python3 $S/e4a_layout_v2.py fit $R/cal_tp4x2n3 tp4x2n3 2>&1 | tee -a $LOG

say "== 3. model decisions (new points) and sanity predictions (old points)"
ok=1
for c in "${CONFIGS[@]}"; do
  set -- $c
  python3 $S/e4a_layout_v2.py predict $1 $2 $3 new 2>&1 | tee -a $LOG || ok=0
done
if [ $ok = 1 ] && gitsync "fusion-dispatch E4a3: pre-registered model decisions and probe lists (before probes and oracle)" \
     $R/gemm $R/cal_tp8x2n3 $R/cal_tp4x2n3 $R/predictions_* \
     common/cost_model/hw_profiles/xnode158-159_tp8x2n3_gpu_block.json common/cost_model/hw_profiles/xnode158-159_tp8x2n3_steady_block.json \
     common/cost_model/hw_profiles/xnode158-159_tp4x2n3_gpu_block.json common/cost_model/hw_profiles/xnode158-159_tp4x2n3_steady_block.json $LOG; then
  say "== 4. layout probes (30 rounds) and final decisions"
  for c in "${CONFIGS[@]}"; do
    set -- $c; tag=$1 model=$2 tp=$3 ppn=$4 gpus=$5
    while read -r phase Ms; do
      [ -z "${Ms:-}" ] && continue
      say "probe $model tp$tp $phase ($Ms)"
      block $R/probe_${tag}_${model}_tp${tp}_$phase $model $tp $ppn $phase $Ms 30 $gpus
    done < $R/predictions_${tag}_${model}_tp${tp}_new_probes.txt
    python3 $S/e4a_layout_v2.py finalize $tag $model $tp 2>&1 | tee -a $LOG
  done
  if gitsync "fusion-dispatch E4a3: probes done, final pre-registered decisions (before the oracle run)" $R $LOG; then
    say "== 5. oracle on the fresh points (200 rounds)"
    for c in "${CONFIGS[@]}"; do
      set -- $c; tag=$1 model=$2 tp=$3 ppn=$4 gpus=$5
      block $R/oracle_${tag}_${model}_tp${tp}_decode $model $tp $ppn decode $E4A_NEW_DECODE 200 $gpus
      block $R/oracle_${tag}_${model}_tp${tp}_prefill $model $tp $ppn prefill $E4A_NEW_PREFILL 200 $gpus
      python3 $S/e4a_layout_v2.py eval $tag $model $tp new 2>&1 | tee -a $LOG
    done
  else
    say "pre-registration 2 commit failed -> oracle NOT run"
  fi
else
  say "pre-registration 1 failed -> probes and oracle NOT run"
fi
{
  echo
  echo "## $(date '+%F') · E4a3 自動流程完成（run_e4a3_v1.sh 自動追加；結論待人工撰寫）"
  echo
  echo '```'
  for f in $R/eval_*_new.txt; do [ -f $f ] && { cat $f; echo; }; done
  echo '```'
} >> ws/fusion-dispatch/JOURNAL.md
gitsync "fusion-dispatch E4a3: oracle + evaluation (fresh points, changed cross-node env) (auto JOURNAL entry)" ws/fusion-dispatch/JOURNAL.md $R $LOG
say "done"
