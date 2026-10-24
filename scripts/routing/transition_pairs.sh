#!/bin/bash
# Fig. 6 and the state-transitions table on all eight models: label the sentences that follow a Read or Verify
# sentence (or the SOURCE_CLASSES you name) and redo each native-dynamics panel with them. Per model it submits a plan
# (CPU), SHARDS judge jobs (2 GPUs, 4 hours each) and a summary (CPU); one last job joins the eight panels into
# report/final-routing-20261015-r1/driver/TRANSITION_PAIRS.<variant>.json. Qwen3.6 first rebuilds its sentence table,
# which covers only 509 of its 1,486 question families, and its original-sample panel.
#   scripts/routing/transition_pairs.sh [SOURCE_CLASSES [SHARDS]]       default: Read,Verify and 2 shards per model
#   scripts/routing/transition_pairs.sh Read,Verify,Plan,Explore,Monitor 3
# AFTER="JOB ..." names jobs the judges wait for. A wider rule reuses the labels of a narrower one (one label store per
# model), so its judges must not run while the narrower ones still do. DRY_RUN=1 prints the sbatch lines only.
# A relaunch continues the chain: a stage whose receipt exists is not submitted again (a model whose panel exists is
# done, one whose plan exists goes straight to its judges, which resume), and the stages are idempotent in any case.
set -euo pipefail
classes=${1:-Read,Verify}; shards=${2:-2}
models=(gpt nemotron qwen35 qwen36 gemma glm qwen330b lfm25)
cpu=(-p boost_usr_prod -c 8 --mem=40G -t 01:00:00)
study=${STUDY_DATA_ROOT:-report/final-routing-20261015-r1}
variant=$(tr 'A-Z,' 'a-z-' <<< "$classes")  # the receipt folder of this rule (routing.transition_pairs.variant)
join() { local IFS=:; echo "$*"; }
have() { test -e "$study/$1"; }
submit() {  # sbatch -J NAME ...: prints the job ID (the name in a dry run) and shows "NAME ID"
  if [ -n "${DRY_RUN:-}" ]; then echo "sbatch $*" >&2; echo "$2"; return; fi
  local job; job=$(sbatch --parsable --kill-on-invalid-dep=yes "$@"); echo "$2 $job" >&2; echo "$job"
}
afterok() {  # --dependency=afterok:ID:... for the jobs that were submitted (nothing when none was)
  local ids=(); for id in "$@"; do test -z "$id" || ids+=("$id"); done
  test ${#ids[@]} -eq 0 || echo "--dependency=afterok:$(join "${ids[@]}")"
}
table=""
have transition-pairs/qwen36/SENTENCES.json ||
  table=$(submit -J pairs-table "${cpu[@]}" scripts/slurm/cpu.sh routing.sentence_table rebuild --model qwen36)
have transition-pairs/qwen36/NATIVE_DYNAMICS_SUMMARY.original-sample.json ||
  submit -J pairs-original-qwen36 "${cpu[@]}" $(afterok $table) \
    scripts/slurm/cpu.sh routing.transition_pairs summary --model qwen36 --original-sample-only > /dev/null
summaries=()
for m in "${models[@]}"; do
  have transition-pairs/$m/$variant/NATIVE_DYNAMICS_SUMMARY.json && continue
  plan=""
  have transition-pairs/$m/$variant/PAIR_PLAN.json || plan=$(submit -J pairs-plan-$m "${cpu[@]}" \
    $(test $m != qwen36 || afterok $table) \
    scripts/slurm/cpu.sh routing.transition_pairs plan --model $m --source-classes $classes)
  dependencies=(); test -z "$plan" || dependencies+=(afterok:$plan)
  test -z "${AFTER:-}" || dependencies+=(afterany:$(join $AFTER))
  dependency=""; test ${#dependencies[@]} -eq 0 || dependency=--dependency=$(IFS=,; echo "${dependencies[*]}")
  judges=()
  for ((i = 0; i < shards; i++)); do
    judges+=($(submit -J pairs-judge-$m-$i $dependency scripts/routing/pairs_judge.sbatch $m $i/$shards $classes))
  done
  summaries+=($(submit -J pairs-summary-$m "${cpu[@]}" $(afterok "${judges[@]}") \
    scripts/slurm/cpu.sh routing.transition_pairs summary --model $m --source-classes $classes))
done
have driver/TRANSITION_PAIRS.$variant.json ||
  submit -J pairs-collect -p boost_usr_prod -c 2 --mem=8G -t 00:20:00 $(afterok "${summaries[@]}") \
    scripts/slurm/cpu.sh routing.transition_pairs collect --source-classes $classes > /dev/null
