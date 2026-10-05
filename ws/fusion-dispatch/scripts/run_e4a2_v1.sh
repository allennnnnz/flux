#!/bin/bash
################################################################################
# fusion-dispatch E4a2 (2026-10-05): the E4a layout experiment re-run with the corrected decider
# (e4a_layout_v2.py: non-monotone curves, exact-size calibration, G4 probe rule) on FRESH points.
# Unattended, inside tmux on css-host-158, repo root:  tmux new -d -s e4a2 'bash ws/fusion-dispatch/scripts/run_e4a2_v1.sh'
#   1  single-GPU cuBLAS at the new M (probe_gemm_v1.py, css-host-158 only)
#   2  cross-node calibration at the exact (H, M) the blocks use (calibrate_xnode_v2.py), fit (no monotone, no merge)
#   3  model decisions + probe list for the new points (and the old 24 as sanity)  -> COMMIT + PUSH (pre-registration 1)
#   4  layout probes (30 rounds, both layouts) where |delta| < 3% of t_sp, finalize   -> COMMIT + PUSH (pre-registration 2)
#   5  oracle on the new points (200 rounds, both nodes guarded), eval (new + old sanity) -> COMMIT + PUSH
# A pre-registration commit that fails stops the later measurements of that config.
################################################################################
set -u
REPO=/home/rogerlee/flux
cd $REPO
S=ws/fusion-dispatch/scripts
R=ws/fusion-dispatch/results/e4a2_layout
T=ws/fusion-dispatch/results/g4_block_tables
G=common/measure/exclusive_guard_v2.py
LOG=$R/run_log.txt
mkdir -p $R/gemm
export XNODE_ENV="NVSHMEM_HCA_LIST=mlx5_0"
export XNODE_TIMEOUT=3600
CO="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
NEW_GEMM_MS=64,192,320,448,768,1536,3072
ALL_MS=32,64,128,192,256,320,384,448,512,768,1024,1536,2048,3072,4096
CONFIGS=("tp8x2n2 qwen2.5-32b 8 4 tp8x2n" "tp4x2n2 qwen2.5-32b 4 2 tp4x2n" "tp4x2n2 llama3-8b 4 2 tp4x2n")

say() { echo "[E4a2] $(date '+%F %T') $*" | tee -a $LOG; }

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

block() {  # block <out> <model> <tp> <ppn> <phase> <Ms> <rounds>
  local out=$1 model=$2 tp=$3 ppn=$4 phase=$5 Ms=$6 rounds=$7 mode=eager
  [ "$phase" = decode ] && mode=graph
  bash $S/run_xnode.sh $out $ppn vllm $S/validate_block_v4.py --model $model --phase $phase --mode $mode --Ms $Ms \
    --table_g4 $T/table_g4_${model}_tp${tp}_${phase}.json --table_g3 $T/table_g3_${model}_tp${tp}_${phase}.json \
    --policies tp_ar_vllm,sp_nccl --rounds $rounds --warmup 5 --out_dir $out 2>&1 | tail -1 | tee -a $LOG
}

say "start (pid $$)"
gitsync "fusion-dispatch E4a2: setup (non-monotone curves, exact-size calibration, probe rule, fresh M points)" \
  common/cost_model/predictor/curves.py $S/calibrate_xnode_v2.py $S/e4a_layout_v2.py $S/run_e4a2_v1.sh \
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
bash $S/run_xnode.sh $R/cal_tp8x2n2 4 vllm $S/calibrate_xnode_v2.py --cases "5120:$ALL_MS" --out_dir $R/cal_tp8x2n2 2>&1 | tail -1 | tee -a $LOG
bash $S/run_xnode.sh $R/cal_tp4x2n2 2 vllm $S/calibrate_xnode_v2.py --cases "5120:$ALL_MS;4096:$ALL_MS" --out_dir $R/cal_tp4x2n2 2>&1 | tail -1 | tee -a $LOG
python3 $S/e4a_layout_v2.py fit $R/cal_tp8x2n2 tp8x2n2 2>&1 | tee -a $LOG
python3 $S/e4a_layout_v2.py fit $R/cal_tp4x2n2 tp4x2n2 2>&1 | tee -a $LOG

say "== 3. model decisions (new points) and sanity predictions (old points)"
ok=1
for c in "${CONFIGS[@]}"; do
  set -- $c
  python3 $S/e4a_layout_v2.py predict $1 $2 $3 new 2>&1 | tee -a $LOG || ok=0
  python3 $S/e4a_layout_v2.py predict $1 $2 $3 old 2>&1 | tee -a $LOG || ok=0
done
if [ $ok = 1 ] && gitsync "fusion-dispatch E4a2: pre-registered model decisions and probe lists (before probes and oracle)" \
     $R/gemm $R/cal_tp8x2n2 $R/cal_tp4x2n2 $R/predictions_* \
     common/cost_model/hw_profiles/xnode158-159_tp8x2n2_gpu_block.json common/cost_model/hw_profiles/xnode158-159_tp8x2n2_steady_block.json \
     common/cost_model/hw_profiles/xnode158-159_tp4x2n2_gpu_block.json common/cost_model/hw_profiles/xnode158-159_tp4x2n2_steady_block.json $LOG; then
  say "== 4. layout probes (30 rounds) and final decisions"
  for c in "${CONFIGS[@]}"; do
    set -- $c; tag=$1 model=$2 tp=$3 ppn=$4
    while read -r phase Ms; do
      [ -z "${Ms:-}" ] && continue
      say "probe $model tp$tp $phase ($Ms)"
      block $R/probe_${tag}_${model}_tp${tp}_$phase $model $tp $ppn $phase $Ms 30
    done < $R/predictions_${tag}_${model}_tp${tp}_new_probes.txt
    python3 $S/e4a_layout_v2.py finalize $tag $model $tp 2>&1 | tee -a $LOG
  done
  if gitsync "fusion-dispatch E4a2: probes done, final pre-registered decisions (before the oracle run)" $R $LOG; then
    say "== 5. oracle on the fresh points (200 rounds)"
    for c in "${CONFIGS[@]}"; do
      set -- $c; tag=$1 model=$2 tp=$3 ppn=$4
      block $R/oracle_${tag}_${model}_tp${tp}_decode $model $tp $ppn decode 64,192,320,448 200
      block $R/oracle_${tag}_${model}_tp${tp}_prefill $model $tp $ppn prefill 768,1536,3072 200
      python3 $S/e4a_layout_v2.py eval $tag $model $tp new 2>&1 | tee -a $LOG
    done
  else
    say "pre-registration 2 commit failed -> oracle NOT run"
  fi
else
  say "pre-registration 1 failed -> probes and oracle NOT run"
fi
for c in "${CONFIGS[@]}"; do
  set -- $c
  python3 $S/e4a_layout_v2.py eval $1 $2 $3 old $5 2>&1 | tee -a $LOG
done
{
  echo
  echo "## $(date '+%F') · E4a2 自動流程完成（run_e4a2_v1.sh 自動追加；結論待人工撰寫）"
  echo
  echo '```'
  for f in $R/eval_*_new.txt $R/eval_*_old.txt; do [ -f $f ] && { cat $f; echo; }; done
  echo '```'
} >> ws/fusion-dispatch/JOURNAL.md
gitsync "fusion-dispatch E4a2: oracle + evaluation (fresh points) and sanity check (auto JOURNAL entry)" ws/fusion-dispatch/JOURNAL.md $R $LOG
say "done"
