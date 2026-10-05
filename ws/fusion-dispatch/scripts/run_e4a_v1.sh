#!/bin/bash
################################################################################
# fusion-dispatch E4a (2026-10-05): unattended, pre-registered layout experiment on the slow interconnect
# (TP across css-host-158 + css-host-159). Runs inside tmux from css-host-158, repo root:
#   tmux new -d -s e4a 'bash ws/fusion-dispatch/scripts/run_e4a_v1.sh'
# For each config: calibrate (cross-node) -> fit -> decider + frozen-rule decisions -> git commit + push
# (pre-registration; if the commit fails, the oracle is NOT run) -> oracle (validate_block_v4.py,
# tp_ar_vllm vs sp_nccl, 200 rounds, both nodes guarded) -> eval -> commit + push.
#   A  qwen2.5-32b  TP=8 = 4+4 GPUs   (same shapes as G4 Qwen TP8 on one node)
#   B  qwen2.5-32b  TP=4 = 2+2 GPUs   (same shapes as G4 Qwen TP4)
#   C  llama3-8b    TP=4 = 2+2 GPUs   (reuses B's calibration: same world size and interconnect)
# NVSHMEM_HCA_LIST=mlx5_0: flux.testing.initialize_distributed() starts NVSHMEM, which needs all-pairs
# reachability (only within one rail here); NCCL itself keeps all 7 working rails (launch_xnode.sh).
################################################################################
set -u
REPO=/home/rogerlee/flux
cd $REPO
S=ws/fusion-dispatch/scripts
R=ws/fusion-dispatch/results/e4a_layout
T=ws/fusion-dispatch/results/g4_block_tables
LOG=$R/run_log.txt
mkdir -p $R
export XNODE_ENV="NVSHMEM_HCA_LIST=mlx5_0"
export XNODE_TIMEOUT=3600
CO="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

say() { echo "[E4a] $(date '+%F %T') $*" | tee -a $LOG; }

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

oracle() {  # oracle <tag> <model> <tp> <ppn>
  local tag=$1 model=$2 tp=$3 ppn=$4
  for spec in "decode graph 32,128,256,384,512" "prefill eager 1024,2048,4096"; do
    set -- $spec
    local phase=$1 mode=$2 Ms=$3 out=$R/oracle_${tag}_${model}_tp${tp}_$1
    say "oracle $model tp$tp $phase ($Ms)"
    bash $S/run_xnode.sh $out $ppn vllm $S/validate_block_v4.py --model $model --phase $phase --mode $mode \
      --Ms $Ms --table_g4 $T/table_g4_${model}_tp${tp}_${phase}.json --table_g3 $T/table_g3_${model}_tp${tp}_${phase}.json \
      --policies tp_ar_vllm,sp_nccl --rounds 200 --out_dir $out 2>&1 | tail -1 | tee -a $LOG
  done
}

config() {  # config <tag> <model> <tp> <ppn> <calibrate:yes|no>
  local tag=$1 model=$2 tp=$3 ppn=$4 cal=$5
  if [ "$cal" = yes ]; then
    say "== calibrate $tag (TP=$tp, $ppn GPUs per node)"
    bash $S/run_xnode.sh $R/cal_$tag $ppn vllm $S/calibrate_xnode_v1.py --out_dir $R/cal_$tag 2>&1 | tail -1 | tee -a $LOG
    if ! python3 $S/e4a_layout_v1.py fit $R/cal_$tag $tag >> $LOG 2>&1; then say "FIT FAILED for $tag; skipping"; return 1; fi
  fi
  say "== predict $tag $model tp$tp"
  if ! python3 $S/e4a_layout_v1.py predict $tag $model $tp 2>&1 | tee -a $LOG; then say "PREDICT FAILED; skipping"; return 1; fi
  if ! gitsync "fusion-dispatch E4a: pre-registered layout decisions ($tag $model tp$tp, before the oracle run)" \
       $R/cal_$tag $R/predictions_${tag}_${model}_tp${tp}.csv $R/predictions_${tag}_${model}_tp${tp}_meta.json \
       common/cost_model/hw_profiles/xnode158-159_${tag}_gpu_block.json common/cost_model/hw_profiles/xnode158-159_${tag}_steady_block.json \
       $LOG; then
    say "pre-registration commit failed -> oracle NOT run for $tag $model"; return 1
  fi
  oracle $tag $model $tp $ppn
  python3 $S/e4a_layout_v1.py eval $tag $model $tp 2>&1 | tee -a $LOG
  gitsync "fusion-dispatch E4a: oracle + evaluation ($tag $model tp$tp)" $R $LOG
}

say "start (pid $$)"
gitsync "fusion-dispatch E4a: setup (cross-node calibration, layout predictor, unattended pipeline, smoke tests)" \
  $S/validate_block_v4.py $S/calibrate_xnode_v1.py $S/e4a_layout_v1.py $S/run_e4a_v1.sh \
  ws/fusion-dispatch/results/e4_smoke_block ws/fusion-dispatch/STATUS.md ws/fusion-dispatch/JOURNAL.md PROJECT.md $LOG
config tp8x2n qwen2.5-32b 8 4 yes
config tp4x2n qwen2.5-32b 4 2 yes
config tp4x2n llama3-8b 4 2 no
{
  echo
  echo "## $(date '+%F') · E4a 自動流程完成（run_e4a_v1.sh 自動追加；結論待人工撰寫）"
  echo
  for f in $R/eval_*_summary.csv; do echo "- \`$f\`"; done
  echo
  echo '```'
  for f in $R/eval_*.txt; do cat $f; echo; done
  echo '```'
} >> ws/fusion-dispatch/JOURNAL.md
gitsync "fusion-dispatch E4a: unattended run finished (auto JOURNAL entry; conclusions to be written)" ws/fusion-dispatch/JOURNAL.md $R $LOG
say "done"
